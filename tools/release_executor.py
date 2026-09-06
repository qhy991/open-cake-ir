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
)
_SOURCE_FILES = (
    "docs/en/PAIRED_TRITON.md",
    "examples/gpu/flash_kmeans_quickstart.py",
    "src/open_cake_ir/__init__.py",
    "src/open_cake_ir/cli.py",
    "src/open_cake_ir/evaluation/assets/qsa_direct_reference_v1.cu",
    "src/open_cake_ir/evaluation/assets/qsa_direct_reference_v1.json",
    "tools/evaluate_flash_candidate.py",
    "tools/evaluate_qsa_candidate.py",
    "tools/observe_target_peak.py",
    "tools/project_qsa_feedback.py",
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
        paths.extend((root / value).glob("*.py"))
    resolved = sorted({path.resolve(strict=True) for path in paths})
    if any(root not in path.parents or path.is_symlink() for path in resolved):
        raise ValueError("Executor source custody differs")
    return resolved


def _released_executor_paths(root: Path) -> tuple[Path, ...]:
    current = (root / "runtime/executors").glob("*.json")
    archived = (root / "evidence/executors").glob("*/runtime/executor.json")
    return tuple(sorted({*current, *archived}))


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
