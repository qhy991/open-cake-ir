#!/usr/bin/env python3
"""Create one immutable content-bound Lab/Evaluation/Evidence Executor Revision."""

from __future__ import annotations

import argparse
import json
import re
import sys
from hashlib import sha256
from pathlib import Path

_SOURCE_ROOTS = (
    "src/open_cake_ir/lab",
    "src/open_cake_ir/evaluation",
    "src/open_cake_ir/evidence",
)
_SOURCE_FILES = (
    "examples/gpu/amd_triton_quickstart.py",
    "examples/gpu/flash_kmeans_quickstart.py",
    "examples/gpu/rmsnorm_amd_quickstart.py",
    "examples/gpu/rmsnorm_amd_search.py",
    "examples/gpu/swiglu_amd_quickstart.py",
    "src/open_cake_ir/__init__.py",
    "src/open_cake_ir/cli.py",
    "tools/evaluate_flash_candidate.py",
)


def _validate_output_boundary(
    root: Path,
    output: Path,
    *,
    schema_version: int,
    executor_id: str,
) -> None:
    if schema_version != 2:
        return
    runtime = (root / "runtime/executors").resolve(strict=True)
    candidate_name = re.compile(
        rf"\.{re.escape(executor_id)}\.candidate\.[1-9][0-9]*\.json"
    )
    if output.parent != runtime or candidate_name.fullmatch(output.name) is None:
        raise ValueError(
            "schema v2 writer may only assemble a hidden candidate for live admission"
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
    parser.add_argument(
        "--replace-unwitnessed",
        type=Path,
        help="existing same-id working descriptor proven reclaimable by witness policy",
    )
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
        or document.get("schema_version") not in {1, 2}
        or document.get("state") != "draft"
        or not isinstance(document.get("executor_id"), str)
        or not document["executor_id"]
    ):
        raise ValueError("Executor proposal fields, schema, or state differ")
    _validate_output_boundary(
        root,
        output,
        schema_version=int(document["schema_version"]),
        executor_id=str(document["executor_id"]),
    )
    sys.path.insert(0, str(root / "src"))
    from open_cake_ir.lab.executor import ExecutorRevision

    ExecutorRevision._validate_host_document(
        int(document["schema_version"]), document["host_environment"]
    )
    sys.path.insert(0, str(root))
    from tools.executor_revision_witnesses import (
        parse_executor_revision_id,
        plan_executor_revision_cycle,
    )

    identity = parse_executor_revision_id(document["executor_id"])
    if document["schema_version"] == 2 and (
        identity is None or identity.family != "gfx1151"
    ):
        raise ValueError("Executor schema v2 gfx1151 identity differs")

    replacement: Path | None = None
    if arguments.replace_unwitnessed is not None:
        unresolved = arguments.replace_unwitnessed
        if unresolved.is_symlink():
            raise ValueError("replacement Executor custody differs")
        replacement = unresolved.resolve(strict=True)
        runtime_root = (root / "runtime/executors").resolve(strict=True)
        if replacement.parent != runtime_root:
            raise ValueError("replacement Executor path differs")
        plan = plan_executor_revision_cycle(root, document["executor_id"])
        relative = replacement.relative_to(root).as_posix()
        if relative not in plan.reclaimable_descriptors:
            raise ValueError("replacement Executor is not an unwitnessed working revision")
        replacement_document = json.loads(replacement.read_text(encoding="utf-8"))
        if (
            not isinstance(replacement_document, dict)
            or replacement_document.get("executor_id") != document["executor_id"]
            or replacement.name != f"{document['executor_id']}.json"
        ):
            raise ValueError("replacement Executor identity differs")
    for released_path in _released_executor_paths(root):
        if replacement is not None and released_path.resolve() == replacement:
            continue
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
