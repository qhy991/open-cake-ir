#!/usr/bin/env python3
"""Verify the publishable AKA review projection and write a durable receipt."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def jsonl(path: Path) -> list[dict[str, Any]]:
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if any(not isinstance(value, dict) for value in values):
        raise ValueError(f"non-object JSONL row: {path}")
    return values


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    root = args.export_dir.resolve(strict=True)
    if args.receipt.exists() or args.receipt.is_symlink():
        raise SystemExit(f"refusing existing receipt: {args.receipt}")

    manifest = load(root / "manifest.json")
    if manifest.get("status") != "verified_projection":
        raise ValueError("manifest status differs")
    checked_artifacts = 0
    for artifact in manifest.get("artifacts", []):
        path = root / artifact["path"]
        if (
            not path.is_file()
            or path.stat().st_size != artifact["bytes"]
            or digest(path) != artifact["sha256"]
        ):
            raise ValueError(f"manifest artifact identity differs: {path}")
        checked_artifacts += 1

    phase = jsonl(root / "phase-a-review.jsonl")
    if len(phase) != 677 or len({row.get("case_id") for row in phase}) != 677:
        raise ValueError("Phase A row identity differs")
    phase_status = Counter(row.get("final_status") for row in phase)
    phase_owner = Counter(row.get("owner") for row in phase if row.get("final_status") == "accepted")
    if phase_status != Counter({"accepted": 439, "rejected": 237, "infrastructure_failure": 1}):
        raise ValueError(f"Phase A status counts differ: {phase_status}")
    if phase_owner != Counter({
        "ir_gap": 326,
        "schedule": 57,
        "program_composition": 27,
        "insufficient_evidence": 21,
        "workload_evidence": 8,
    }):
        raise ValueError(f"Phase A owner counts differ: {phase_owner}")
    for row in phase:
        if row.get("schema") != "open-cake.aka-qualified-ir-review-export-row.v1":
            raise ValueError("Phase A schema differs")
        if row["final_status"] == "accepted":
            if row.get("owner") is None or row.get("verifier", {}).get("deterministic") is not True:
                raise ValueError(f"accepted Phase A row lacks authority: {row['case_id']}")
        elif row.get("current_ir_expressibility") is not None or row.get("owner") is not None:
            raise ValueError(f"non-accepted Phase A row claims assessment: {row['case_id']}")

    lab = jsonl(root / "lab-terminal-results.jsonl")
    if len(lab) != 57 or len({row.get("case_id") for row in lab}) != 57:
        raise ValueError("Lab row identity differs")
    lab_status = Counter(row.get("final_status") for row in lab)
    if lab_status != Counter({"dynamic_valid": 56, "authoring_rejected": 1}):
        raise ValueError(f"Lab status counts differ: {lab_status}")
    for row in lab:
        if (
            row.get("schema") != "open-cake.aka-ir-lab-terminal-row.v1"
            or row.get("owner") != "schedule"
            or row.get("current_ir_expressibility") != "expressible"
            or row.get("lowering_status") != "passed"
            or row.get("training_eligible") is not False
            or row.get("performance_measured") is not False
        ):
            raise ValueError(f"Lab boundary differs: {row.get('case_id')}")
        gpu = row.get("gpu", {})
        if row["final_status"] == "dynamic_valid":
            if any(gpu.get(key) != value for key, value in {
                "eligible": True,
                "outcome": "completed",
                "validity": "valid",
                "correctness": "passed",
                "memcheck": "passed",
                "racecheck": "passed",
            }.items()) or not gpu.get("independent_evidence_ref"):
                raise ValueError(f"dynamic Lab row differs: {row['case_id']}")
        elif any(gpu.get(key) != value for key, value in {
            "eligible": False,
            "outcome": "not_run",
            "correctness": "not_run",
            "memcheck": "not_run",
            "racecheck": "not_run",
        }.items()):
            raise ValueError(f"rejected Lab row differs: {row['case_id']}")

    clusters = load(root / "ir-gap-clusters.json")
    cluster_receipt = load(root / "ir-gap-cluster-verification.json")
    if (
        clusters.get("status") != "proposal"
        or clusters.get("ir_change_status") != "proposal_only"
        or cluster_receipt.get("status") != "accepted"
        or cluster_receipt.get("partition_exact") is not True
        or cluster_receipt.get("source_gap_cases") != 326
        or cluster_receipt.get("source_exact_candidate_names") != 296
    ):
        raise ValueError("IR-gap cluster boundary differs")

    for name in (
        "phase-a-review.jsonl",
        "lab-terminal-results.jsonl",
        "ir-gap-clusters.json",
        "l000214-authoring-rejection.json",
    ):
        text = (root / name).read_text(encoding="utf-8")
        if "auth.json" in text or "/home/qhy-sol/.codex" in text:
            raise ValueError(f"credential path leaked into export: {name}")

    sensitive_patterns = {
        "auth_filename": re.compile(r"auth[.]json", re.IGNORECASE),
        "private_key": re.compile(r"BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY"),
        "openai_secret": re.compile(
            r"(?<![A-Za-z0-9])sk-(?:proj-)?[A-Za-z0-9_-]{32,}"
        ),
        "jwt_three_segment": re.compile(
            r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{10,}[.]"
            r"[A-Za-z0-9_-]{10,}[.][A-Za-z0-9_-]{10,}(?![A-Za-z0-9_-])"
        ),
    }
    scanned_files = [path for path in root.rglob("*") if path.is_file()]
    sensitive_findings: list[tuple[str, int, str]] = []
    for path in sorted(scanned_files):
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for kind, pattern in sensitive_patterns.items():
                if pattern.search(line):
                    sensitive_findings.append(
                        (path.relative_to(root).as_posix(), line_number, kind)
                    )
    if sensitive_findings:
        raise ValueError(
            "sensitive pattern detected without exposing the matching content: "
            f"{sensitive_findings}"
        )

    receipt = {
        "schema": "open-cake.aka-qualified-ir-review-export-verification.v1",
        "status": "accepted",
        "deterministic": True,
        "manifest_artifacts_checked": checked_artifacts,
        "phase_a_rows": 677,
        "phase_a_unique_case_ids": 677,
        "phase_a_status_counts": dict(sorted(phase_status.items())),
        "phase_a_owner_counts": dict(sorted(phase_owner.items())),
        "lab_rows": 57,
        "lab_unique_case_ids": 57,
        "lab_status_counts": dict(sorted(lab_status.items())),
        "ir_gap_cases": 326,
        "ir_gap_exact_candidate_names": 296,
        "ir_gap_semantic_clusters": cluster_receipt["semantic_clusters"],
        "cluster_partition_exact": True,
        "credential_path_scan": "passed",
        "sensitive_pattern_scan": "passed",
        "sensitive_scan_files": len(scanned_files),
        "sensitive_findings": 0,
        "performance_measured": False,
        "compiler_change_approved": False,
        "release_performed": False,
    }
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
