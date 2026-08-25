#!/usr/bin/env python3
"""Archive a no-launch Triton lowering-to-CUBIN qualification in Evidence v2."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler  # noqa: E402
from open_cake_ir.evidence import EvidenceStore  # noqa: E402
from open_cake_ir.lab import BuildRequest, TritonToolchainBuilder  # noqa: E402


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _media_type(role: str) -> str:
    if role == "cubin":
        return "application/x-elf"
    if role == "launch_manifest":
        return "application/json"
    return "text/plain"


def _new_external_path(value: Path, evidence_root: Path) -> Path:
    path = value.parent.resolve(strict=True) / value.name
    if path.exists() or path.is_symlink():
        raise ValueError("Triton qualification anchor output must be new")
    if path == evidence_root or evidence_root in path.parents:
        raise ValueError("Triton qualification anchor must be outside the Evidence root")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--revision", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--anchor-output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    evidence_path = args.evidence_root.parent.resolve(strict=True) / args.evidence_root.name
    anchor_output = _new_external_path(args.anchor_output, evidence_path)
    root = args.project_root.resolve(strict=True)
    compiler = Compiler.load(root, args.revision)
    assessment = compiler.assess_file(args.schedule)
    lowering = compiler.lower(assessment)
    builder_source = root / "src/open_cake_ir/lab/environments.py"
    authority = {
        "schema_version": 1,
        "kind": "triton_compile_only_qualification",
        "compiler_revision_id": assessment.compiler_revision_id,
        "compiler_revision_sha256": assessment.compiler_revision_sha256,
        "schedule_id": assessment.schedule_id,
        "schedule_sha256": assessment.schedule_sha256,
        "lowering_source_sha256": lowering.source_sha256,
        "builder_source_sha256": sha256(builder_source.read_bytes()).hexdigest(),
        "target": lowering.target,
        "python": platform.python_version(),
        "torch": importlib.metadata.version("torch"),
        "triton": importlib.metadata.version("triton"),
        "gpu_execution_authorized": False,
    }
    authority_sha256 = sha256(_canonical_json_bytes(authority)).hexdigest()
    evidence = (
        EvidenceStore.writer(evidence_path)
        if evidence_path.exists()
        else EvidenceStore.create(evidence_path)
    )
    ledger = evidence.start_run(
        args.run_id,
        authority_sha256=authority_sha256,
        authority=authority,
    )
    try:
        request = BuildRequest(
            candidate_sha256=lowering.schedule_sha256,
            source=lowering.source.encode(),
            source_role="lowered_source",
            source_sha256=lowering.source_sha256,
            target=lowering.target,
            entry_point=lowering.route.entry_point,
            toolchain_requirements=lowering.toolchain_requirements,
        )
        candidate = TritonToolchainBuilder().build(request)
        references = [
            evidence.put(payload, media_type=_media_type(role)).reference(role)
            for role, payload in sorted(candidate.artifact_payloads.items())
        ]
        ledger.append(
            "toolchain_compiled",
            {
                "candidate_record_sha256": candidate.canonical_sha256,
                "objects": references,
            },
        )
        ledger.seal(
            protocol_adherence="adhered",
            endpoint_observation="observed",
            endpoint={
                "compile_artifact_complete": True,
                "artifact_roles": sorted(candidate.artifact_roles),
                "cubin_size_bytes": len(candidate.artifact_payloads["cubin"]),
                "kernel_calls": 0,
            },
        )
    except Exception as error:
        ledger.append(
            "run_fault",
            {"fault": "harness_fault", "exception_type": type(error).__name__},
        )
        ledger.seal(
            protocol_adherence="harness_fault",
            endpoint_observation="missing",
        )
        raise
    audit = evidence.audit_run(args.run_id)
    if (
        not audit.archive_integrity
        or not audit.filesystem_custody_verified
        or audit.protocol_adherence != "adhered"
        or audit.terminal_seal_sha256 is None
    ):
        raise ValueError("Triton qualification failed immediate Evidence audit")
    anchor = {
        "schema_version": 1,
        "kind": "triton_toolchain_qualification_evidence_anchor",
        "run_id": audit.run_id,
        "evidence_root": str(evidence.root),
        "authority_sha256": authority_sha256,
        "qualification_artifact_sha256": candidate.canonical_sha256,
        "immediate_audit_integrity": audit.archive_integrity,
        "terminal_seal_sha256": audit.terminal_seal_sha256,
    }
    with anchor_output.open("xb") as stream:
        stream.write(_canonical_json_bytes(anchor) + b"\n")
    anchor_output.chmod(0o644)
    sys.stdout.write(
        json.dumps(
            {
                "run_id": audit.run_id,
                "authority_sha256": authority_sha256,
                "qualification_artifact_sha256": candidate.canonical_sha256,
                "anchor_output": str(anchor_output),
                "terminal_seal_sha256": audit.terminal_seal_sha256,
                "integrity": audit.archive_integrity,
                "endpoint": audit.endpoint,
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
