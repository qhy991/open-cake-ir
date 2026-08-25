#!/usr/bin/env python3
"""Archive every source byte bound by one released Compiler Revision."""

from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Mapping, cast

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.evidence import EvidenceStore  # noqa: E402


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return cast(Mapping[str, object], value)


def _source_path(root: Path, value: object) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        raise ValueError("Compiler source path is invalid")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "\\" in value:
        raise ValueError("Compiler source path is unsafe")
    path = (root / value).resolve(strict=True)
    if root not in path.parents:
        raise ValueError("Compiler source path escapes repository")
    return value, path


def _new_external_path(value: Path, evidence_root: Path) -> Path:
    path = value.parent.resolve(strict=True) / value.name
    if path.exists() or path.is_symlink():
        raise ValueError("Compiler release index output must be new")
    if path == evidence_root or evidence_root in path.parents:
        raise ValueError("Compiler release index must be outside the Evidence root")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--revision", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--index-output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    root = args.project_root.resolve(strict=True)
    evidence_path = args.evidence_root.parent.resolve(strict=True) / args.evidence_root.name
    index_output = _new_external_path(args.index_output, evidence_path)
    revision_bytes = args.revision.resolve(strict=True).read_bytes()
    revision = _object(json.loads(revision_bytes), "compiler_revision")
    if revision.get("state") != "released" or not isinstance(revision.get("sources"), list):
        raise ValueError("Compiler Revision is not released or source-complete")
    authority = sha256(_canonical_json_bytes(revision)).hexdigest()
    evidence = (
        EvidenceStore.writer(evidence_path)
        if evidence_path.exists()
        else EvidenceStore.create(evidence_path)
    )
    objects = [evidence.put(revision_bytes, media_type="application/json")]
    observed_paths: set[str] = set()
    for index, raw_source in enumerate(cast(list[object], revision["sources"])):
        source = _object(raw_source, f"compiler_revision.sources[{index}]")
        relative, path = _source_path(root, source.get("path"))
        if relative in observed_paths:
            raise ValueError(f"Compiler source {relative!r} is duplicated")
        observed_paths.add(relative)
        payload = path.read_bytes()
        if source.get("sha256") != sha256(payload).hexdigest() or source.get(
            "size_bytes"
        ) != len(payload):
            raise ValueError(f"Compiler source {relative!r} differs")
        media_type = "application/json" if relative.endswith(".json") else "text/plain"
        objects.append(evidence.put(payload, media_type=media_type))
    run = evidence.start_run(
        args.run_id,
        authority_sha256=authority,
        authority=revision,
    )
    run.append(
        "compiler_release_sources",
        {
            "revision_id": revision.get("revision_id"),
            "source_count": len(observed_paths),
            "objects": [
                item.reference(
                    "compiler_revision" if index == 0 else f"source_{index - 1:03d}"
                )
                for index, item in enumerate(objects)
            ],
        },
    )
    run.seal(
        protocol_adherence="adhered",
        endpoint_observation="qualified",
        endpoint={"release_source_complete": True, "source_count": len(observed_paths)},
    )
    audit = evidence.audit_run(args.run_id)
    if (
        not audit.archive_integrity
        or not audit.filesystem_custody_verified
        or audit.terminal_seal_sha256 is None
    ):
        raise ValueError("Compiler release archive failed immediate audit")
    index = {
        "schema_version": 2,
        "kind": "compiler_release_evidence_index",
        "run_id": audit.run_id,
        "evidence_root": str(evidence.root),
        "compiler_revision_id": revision.get("revision_id"),
        "authority_sha256": authority,
        "source_count": len(observed_paths),
        "object_count": len(objects),
        "immediate_audit_integrity": audit.archive_integrity,
        "terminal_seal_sha256": audit.terminal_seal_sha256,
    }
    with index_output.open("xb") as stream:
        stream.write(_canonical_json_bytes(index) + b"\n")
    index_output.chmod(0o644)
    print(json.dumps({**index, "index_output": str(index_output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
