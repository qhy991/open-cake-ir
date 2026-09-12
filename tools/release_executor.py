#!/usr/bin/env python3
"""Create one immutable content-bound Lab/Evaluation/Evidence Executor Revision."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

_SOURCE_ROOTS = (
    "src/open_cake_ir/lab",
    "src/open_cake_ir/evaluation",
    "src/open_cake_ir/evidence",
    "src/open_cake_ir/tasks",
)
_SOURCE_FILES = (
    "contracts/scaffolds/open-cake-clean-start-v1.json",
    "contracts/scaffolds/direct-cuda-clean-start-v1.cu",
    "contracts/scaffolds/matched-search-v1.md",
    "docs/en/PAIRED_TRITON.md",
    "docs/en/PAIRED_CUTE.md",
    "contracts/providers/native-cute-candidate-v1.schema.json",
    "examples/gpu/flash_kmeans_quickstart.py",
    "src/open_cake_ir/__init__.py",
    "src/open_cake_ir/cli.py",
    "src/open_cake_ir/serialization.py",
    "src/open_cake_ir/tasks/qsa/assets/qsa_direct_reference_v1.cu",
    "src/open_cake_ir/tasks/qsa/assets/qsa_direct_reference_v1.json",
    "tools/capture_executor_host.py",
    "tools/observe_target_peak.py",
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _source_paths(root: Path) -> list[Path]:
    paths = [root / value for value in _SOURCE_FILES]
    for value in _SOURCE_ROOTS:
        paths.extend((root / value).rglob("*.py"))
        paths.extend((root / value).rglob("*.swift"))
    resolved = sorted({path.resolve(strict=True) for path in paths})
    if any(root not in path.parents or path.is_symlink() for path in resolved):
        raise ValueError("Executor source custody differs")
    return resolved


def _released_executor_paths(root: Path) -> tuple[Path, ...]:
    current = (root / "runtime/executors").glob("*.json")
    archived = (root / "evidence/executors").glob("*/runtime/executor.json")
    return tuple(sorted({*current, *archived}))


def _indexed_document(root: Path, record: dict) -> dict:
    """Verify the inventory-to-descriptor relation at a publication boundary."""
    from open_cake_ir.lab.executor import _relative_file
    if not isinstance(record, dict) or set(record) != {
        'executor_id', 'path', 'canonical_sha256', 'descriptor_raw_sha256', 'source_count',
    }:
        raise ValueError('current Executor index fields differ')
    _, path = _relative_file(root, record['path'], 'current Executor descriptor')
    raw = path.read_bytes()
    document = json.loads(raw)
    if (not isinstance(document, dict) or document.get('state') != 'released'
        or document.get('executor_id') != record['executor_id']
        or sha256(raw).hexdigest() != record['descriptor_raw_sha256']
        or sha256(_canonical_json_bytes(document)).hexdigest() != record['canonical_sha256']
        or not isinstance(document.get('sources'), list)
        or type(record['source_count']) is not int
        or len(document['sources']) != record['source_count']):
        raise ValueError('current Executor indexed descriptor identity differs')
    return document


def _source_map(document: dict) -> dict:
    from open_cake_ir.lab.executor import _file_record
    result = {}
    for value in document['sources']:
        value = _file_record(value, 'Executor source')
        if value['path'] in result:
            raise ValueError('duplicate Executor source path')
        result[value['path']] = (value['sha256'], value['size_bytes'])
    if not result:
        raise ValueError('empty Executor source closure')
    return result


def refresh_inventory(root: Path, target: str, record: dict) -> tuple[str, ...]:
    """Project current targets onto one verified runtime source closure.

    The caller has just produced or source-verified ``record``. Other hosts may
    retain their pointer only when their complete bound source map agrees. Their
    immutable descriptors remain available as history when a host rebind is due.
    No host is admitted here and no release identity is created by this projection.
    """
    root = root.resolve(strict=True)
    path = root / 'inventory/EXECUTOR_REVISIONS.json'
    before = path.read_bytes()
    inventory = json.loads(before)
    if (inventory.get('schema_version') != 2
        or not isinstance(inventory.get('current_by_target'), dict)
        or not isinstance(inventory.get('superseded'), list)):
        raise ValueError('Executor inventory schema differs')
    current = inventory['current_by_target']
    # Historical rows may use archive_root rather than an active descriptor path.
    # They remain opaque history; only newly retired active records need a path.
    history = list(inventory['superseded'])
    expected = _source_map(_indexed_document(root, record))
    def retain(item):
        previous = [entry for entry in history if entry.get('path') == item['path']]
        if any(entry != item for entry in previous):
            raise ValueError('historical Executor index identity differs')
        if not previous:
            history.append(item)
    previous = current.get(target)
    if previous is not None and previous != record:
        _indexed_document(root, previous)
        retain(previous)
    current[target] = record
    stale = []
    for name, item in list(current.items()):
        if _source_map(_indexed_document(root, item)) != expected:
            retain(item)
            del current[name]
            stale.append(name)
    current_paths = {item['path'] for item in current.values()}
    inventory['superseded'] = sorted(
        (item for item in history if item.get('path') not in current_paths),
        key=lambda item: item['executor_id'])
    payload = (json.dumps(inventory, indent=2, sort_keys=True) + '\n').encode()
    if payload != before:
        # All identities are checked before any write. Replacement never changes
        # descriptor bytes and avoids exposing a half-written inventory.
        import os, tempfile
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        try:
            temporary.chmod(path.stat().st_mode & 0o777)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return tuple(sorted(stale))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.project_root.resolve(strict=True)
    if arguments.proposal.is_symlink():
        raise ValueError("Executor proposal custody differs")
    proposal = arguments.proposal.resolve(strict=True)
    output_argument = arguments.output.absolute()
    output = output_argument.parent.resolve(strict=True) / output_argument.name
    if output.exists() or output.is_symlink():
        raise FileExistsError("refusing to overwrite Executor Revision")
    document = json.loads(proposal.read_text(encoding="utf-8"))
    if (
        not isinstance(document, dict)
        or set(document)
        != {
            "schema_version",
            "executor_id",
            "state",
            "sources",
            "host_environment",
        }
        or document.get("schema_version") != 1
        or document.get("state") != "draft"
        or not isinstance(document.get("executor_id"), str)
        or not document["executor_id"]
    ):
        raise ValueError("Executor proposal fields, schema, or state differ")
    for released_path in _released_executor_paths(root):
        released = json.loads(released_path.read_text(encoding="utf-8"))
        if (
            isinstance(released, dict)
            and released.get("state") == "released"
            and released.get("executor_id") == document["executor_id"]
        ):
            raise FileExistsError(
                f"Executor identity already released at {released_path.relative_to(root)}"
            )
    records = []
    for source in _source_paths(root):
        payload = source.read_bytes()
        records.append(
            {
                "path": source.relative_to(root).as_posix(),
                "sha256": sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    document["sources"] = records
    document["state"] = "released"
    authority = {"sources": records, "host_environment": document["host_environment"]}
    document["executor_id"] += "+" + sha256(_canonical_json_bytes(authority)).hexdigest()
    for released_path in _released_executor_paths(root):
        released = json.loads(released_path.read_text(encoding="utf-8"))
        if released.get("state") == "released" and released.get("executor_id") == document["executor_id"]:
            raise FileExistsError(
                f"Executor identity already released at {released_path.relative_to(root)}"
            )
    with output.open("xb") as stream:
        stream.write(_canonical_json_bytes(document))
        stream.write(b"\n")
    output.chmod(0o644)
    print(
        json.dumps(
            {
                "executor_id": document["executor_id"],
                "canonical_sha256": sha256(_canonical_json_bytes(document)).hexdigest(),
                "source_count": len(records),
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
