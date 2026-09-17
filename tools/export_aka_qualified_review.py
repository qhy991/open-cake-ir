#!/usr/bin/env python3
"""Export compact, publishable AKA review projections from external evidence roots."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected object at {path}:{line_number}")
        values.append(value)
    return values


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def relative(path: Path, root: Path) -> str:
    return path.resolve(strict=True).relative_to(root.resolve(strict=True)).as_posix()


def verifier_reason(receipt_path: Path, receipt: dict[str, Any]) -> str:
    stderr = receipt.get("stderr")
    if isinstance(stderr, str):
        candidate = receipt_path.parent.parent / stderr
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                return text
    return f"deterministic verifier rejected the untrusted model proposal ({receipt.get('failure_category')})"


def build_phase_rows(
    records: list[dict[str, Any]],
    phase: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, int]]:
    events = read_jsonl(phase / "ledger.jsonl")
    event_by_id = {event["case_id"]: event for event in events}
    if len(events) != 676 or len(event_by_id) != 676:
        raise ValueError("Phase A ledger count differs")
    accepted_results: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    owner_counts: Counter[str] = Counter()

    for record_line, record in enumerate(records, start=1):
        case_id = record["case_id"]
        event = event_by_id.get(case_id)
        verifier: dict[str, Any]
        assessment: dict[str, Any] | None = None
        if event is None:
            final_status = "infrastructure_failure"
            model_receipt_path = phase / "items" / case_id / "model/receipt.json"
            model_receipt = load(model_receipt_path)
            if (
                model_receipt.get("exit_code") != 1
                or model_receipt.get("event_count") is not None
                or model_receipt.get("final_response") is not None
            ):
                raise ValueError(f"pre-model infrastructure receipt differs: {case_id}")
            reason = "Model execution did not start because the filesystem sandbox mount failed before a review result was created."
            verifier = {
                "status": "not_run",
                "failure_category": "pre_model_infrastructure",
                "receipt_ref": None,
                "result_ref": None,
                "deterministic": False,
                "model_receipt_ref": relative(model_receipt_path, phase),
                "model_stderr_ref": relative(model_receipt_path.parent / model_receipt["stderr"].split("/", 1)[-1], phase),
            }
        else:
            receipt_path = phase / event["receipt"]
            receipt = load(receipt_path)
            verifier = {
                "status": event["status"],
                "failure_category": event.get("failure_category"),
                "receipt_ref": relative(receipt_path, phase),
                "result_ref": None,
                "deterministic": receipt.get("deterministic") is True,
            }
            if event["status"] == "accepted":
                case_root = receipt_path.parent.parent
                result_path = case_root / receipt["result"]
                assessment = load(result_path)
                if (
                    receipt.get("accepted") is not True
                    or assessment.get("case_id") != case_id
                    or assessment.get("verification", {}).get("status") != "accepted"
                ):
                    raise ValueError(f"accepted Phase A evidence differs: {case_id}")
                verifier["result_ref"] = relative(result_path, phase)
                final_status = "accepted"
                reason = assessment["reason"]
                accepted_results[case_id] = assessment
                owner_counts[assessment["owner"]] += 1
            else:
                if receipt.get("accepted") is not False:
                    raise ValueError(f"rejected Phase A receipt differs: {case_id}")
                final_status = "rejected"
                reason = verifier_reason(receipt_path, receipt)
                raw = receipt.get("raw_response")
                if isinstance(raw, str):
                    verifier["untrusted_model_proposal_ref"] = relative(
                        receipt_path.parent.parent / raw, phase
                    )
        status_counts[final_status] += 1
        rows.append(
            {
                "schema": "open-cake.aka-qualified-ir-review-export-row.v1",
                "record_line": record_line,
                "case_id": case_id,
                "derived_parent_id": record.get("derived_parent_id"),
                "input_artifacts": record.get("artifacts"),
                "provenance": record.get("provenance"),
                "original_parent": record.get("original_parent"),
                "qualification": record.get("qualification"),
                "qualification_missing_facts": record.get("missing_facts"),
                "assessment_scope": None if assessment is None else assessment.get("assessment_scope"),
                "owner": None if assessment is None else assessment.get("owner"),
                "current_ir_expressibility": None if assessment is None else assessment.get("current_ir_expressibility"),
                "frozen_contract": None if assessment is None else assessment.get("frozen_contract"),
                "ir": None if assessment is None else assessment.get("ir"),
                "gap": None if assessment is None else assessment.get("gap"),
                "eligibility": None if assessment is None else assessment.get("eligibility"),
                "artifact_refs": [] if assessment is None else assessment.get("artifact_refs", []),
                "evidence_refs": [] if assessment is None else assessment.get("evidence_refs", []),
                "verifier": verifier,
                "final_status": final_status,
                "final_reason": reason,
                "claim_boundary": "Static fixed-instance corpus review only; acceptance is not GPU, performance, training, release, or generalization evidence.",
            }
        )

    expected_status = {"accepted": 439, "rejected": 237, "infrastructure_failure": 1}
    expected_owner = {
        "schedule": 57,
        "ir_gap": 326,
        "program_composition": 27,
        "insufficient_evidence": 21,
        "workload_evidence": 8,
    }
    if dict(status_counts) != expected_status or dict(owner_counts) != expected_owner:
        raise ValueError(
            f"Phase A counts differ: status={dict(status_counts)}, owner={dict(owner_counts)}"
        )
    return rows, accepted_results, {**expected_status, **{f"owner_{k}": v for k, v in expected_owner.items()}}


def dynamic_evidence(lab: Path, admission: dict[str, Any]) -> dict[str, dict[str, Any]]:
    progress = load(lab / "lab-progress-summary.json")
    evidence: dict[str, dict[str, Any]] = {}
    sigmoid_cases = [path.name for path in (lab / "authoring-v5/cases").iterdir() if path.is_dir()]
    if len(sigmoid_cases) != 1:
        raise ValueError("sigmoid case identity differs")
    special = {
        "preserved-canary-v4-v6-state/runs/open-cake-aka-copy4-n1024-b200-canary-v6-bbc6a38de092/result.json": admission["canary_case_id"],
        "preserved-sigmoid-runtime-v2/result.json": sigmoid_cases[0],
    }
    for ref in progress["evidence_refs"]:
        if ref.endswith("-independent-verification.current.json"):
            continue
        path = lab / ref
        value = load(path)
        if ref in special:
            case_id = special[ref]
            evidence[case_id] = {
                "independent_evidence_ref": ref,
                "run_id": value.get("run_id"),
                "outcome": value.get("outcome"),
                "validity": value.get("validity"),
                "complete_output_artifacts": None,
                "independently_checked_output_elements": None,
                "independently_checked_mutable_state_elements": 0,
            }
            continue
        rows = value.get("rows")
        if isinstance(rows, list):
            for row in rows:
                case_id = row["case_id"]
                evidence[case_id] = {
                    "independent_evidence_ref": ref,
                    "run_id": row.get("run_id"),
                    "outcome": row.get("outcome", "completed"),
                    "validity": row.get("validity", "valid"),
                    "complete_output_artifacts": row.get("complete_output_artifacts"),
                    "independently_checked_output_elements": row.get("independently_checked_output_elements"),
                    "independently_checked_mutable_state_elements": row.get("independently_checked_mutable_state_elements", 0),
                }
        elif isinstance(value.get("case_id"), str):
            case_id = value["case_id"]
            evidence[case_id] = {
                "independent_evidence_ref": ref,
                "run_id": value.get("run_id"),
                "outcome": value.get("outcome"),
                "validity": value.get("validity"),
                "complete_output_artifacts": value.get("complete_output_artifacts"),
                "independently_checked_output_elements": value.get("independently_checked_output_elements"),
                "independently_checked_mutable_state_elements": value.get("independently_checked_mutable_state_elements", 0),
            }
    if len(evidence) != 56:
        raise ValueError(f"dynamic evidence count differs: {len(evidence)}")
    return evidence


def build_lab_rows(
    admission: dict[str, Any],
    accepted_results: dict[str, dict[str, Any]],
    lab: Path,
    copy4_independent: Path,
    sigmoid_independent: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    dynamic = dynamic_evidence(lab, admission)
    independent_special = {
        admission["canary_case_id"]: copy4_independent,
        next(path.name for path in (lab / "authoring-v5/cases").iterdir() if path.is_dir()): sigmoid_independent,
    }
    for case_id, path in independent_special.items():
        receipt = load(path)
        if receipt.get("status") != "accepted" or receipt.get("case_id") != case_id:
            raise ValueError(f"special independent receipt differs: {case_id}")
        dynamic[case_id]["independent_evidence_ref"] = relative(path, lab)
        dynamic[case_id]["complete_output_artifacts"] = receipt.get("complete_output_artifacts")
        dynamic[case_id]["independently_checked_output_elements"] = receipt.get("independently_checked_output_elements")

    rejected_ids = {entry["case_id"] for entry in admission["entries"]} - set(dynamic)
    if rejected_ids != {
        "data_movement_and_layout__gather_scatter__analysis__l000214_b200_v1__directderived_sol_ultra_v2__parent_completion_v1"
    }:
        raise ValueError(f"Lab terminal partition differs: {sorted(rejected_ids)}")
    rejected_id = next(iter(rejected_ids))
    rejection_path = (
        lab
        / "authoring-v4/cases"
        / rejected_id
        / "verifier/receipt.json"
    )
    rejection = load(rejection_path)
    if rejection.get("status") != "rejected" or rejection.get("gpu_eligible") is not False:
        raise ValueError("l000214 rejection receipt differs")

    rows: list[dict[str, Any]] = []
    for entry in admission["entries"]:
        case_id = entry["case_id"]
        static = accepted_results.get(case_id)
        if static is None or static.get("owner") != "schedule":
            raise ValueError(f"Lab Phase A schedule authority differs: {case_id}")
        if case_id in dynamic:
            evidence = dynamic[case_id]
            if evidence.get("outcome") != "completed" or evidence.get("validity") != "valid":
                raise ValueError(f"Lab dynamic evidence differs: {case_id}")
            final_status = "dynamic_valid"
            authoring_status = "accepted"
            gpu = {
                "eligible": True,
                "outcome": "completed",
                "validity": "valid",
                "correctness": "passed",
                "memcheck": "passed",
                "racecheck": "passed",
                **evidence,
            }
            reason = "Verifier-admitted fixed instance completed B200 correctness and both sanitizer stages and passed independent complete-output recomputation."
        else:
            final_status = "authoring_rejected"
            authoring_status = "rejected"
            gpu = {
                "eligible": False,
                "outcome": "not_run",
                "validity": "not_applicable",
                "correctness": "not_run",
                "memcheck": "not_run",
                "racecheck": "not_run",
                "independent_evidence_ref": relative(rejection_path, lab),
            }
            reason = "Latest model-prepared proposal failed the canonical authoring reviewer schema, runtime-custody, and direct-evidence-surface gates."
        rows.append(
            {
                "schema": "open-cake.aka-ir-lab-terminal-row.v1",
                "case_id": case_id,
                "derived_parent_id": entry["derived_parent_id"],
                "entry_point": entry["entry_point"],
                "workload_id": entry["workload_id"],
                "assessment_scope": entry["assessment_scope"],
                "backend": entry["backend"],
                "target": entry["target"],
                "owner": "schedule",
                "current_ir_expressibility": "expressible",
                "parse_status": "passed",
                "validate_status": "passed",
                "lowering_status": "passed",
                "static_review": static,
                "authoring_status": authoring_status,
                "gpu": gpu,
                "optimization_eligible": entry["optimization_eligible"] if final_status == "dynamic_valid" else False,
                "training_eligible": False,
                "performance_measured": False,
                "final_status": final_status,
                "final_reason": reason,
                "claim_boundary": "Fixed-instance correctness only; no performance, training, release, broader-shape, or framework-equivalence claim.",
            }
        )
    counts = Counter(row["final_status"] for row in rows)
    if len(rows) != 57 or dict(counts) != {"dynamic_valid": 56, "authoring_rejected": 1}:
        raise ValueError(f"Lab export counts differ: {dict(counts)}")
    return rows, dict(counts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--phase-root", type=Path, required=True)
    parser.add_argument("--admission-manifest", type=Path, required=True)
    parser.add_argument("--lab-root", type=Path, required=True)
    parser.add_argument("--cluster-root", type=Path, required=True)
    parser.add_argument("--cluster-predecessor-root", type=Path, required=True)
    parser.add_argument("--copy4-independent", type=Path, required=True)
    parser.add_argument("--sigmoid-independent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing existing output: {output}")
    output.mkdir(parents=True)
    records = read_jsonl(args.records)
    if len(records) != 677 or len({row["case_id"] for row in records}) != 677:
        raise ValueError("qualified record set differs")
    phase_rows, accepted_results, phase_counts = build_phase_rows(records, args.phase_root)
    admission = load(args.admission_manifest)
    lab_rows, lab_counts = build_lab_rows(
        admission,
        accepted_results,
        args.lab_root,
        args.copy4_independent,
        args.sigmoid_independent,
    )
    cluster_receipt = load(args.cluster_root / "verifier/receipt.json")
    if cluster_receipt.get("status") != "accepted" or cluster_receipt.get("partition_exact") is not True:
        raise ValueError("gap cluster receipt is not accepted")

    phase_path = output / "phase-a-review.jsonl"
    lab_path = output / "lab-terminal-results.jsonl"
    write_jsonl(phase_path, phase_rows)
    write_jsonl(lab_path, lab_rows)
    shutil.copyfile(args.cluster_root / "gap-clusters.json", output / "ir-gap-clusters.json")
    shutil.copyfile(args.cluster_root / "gap-cluster-summary.md", output / "IR_GAP_CLUSTERS.md")
    shutil.copyfile(args.cluster_root / "verifier/receipt.json", output / "ir-gap-cluster-verification.json")
    rejected_id = "data_movement_and_layout__gather_scatter__analysis__l000214_b200_v1__directderived_sol_ultra_v2__parent_completion_v1"
    shutil.copyfile(
        args.lab_root / "authoring-v4/cases" / rejected_id / "verifier/receipt.json",
        output / "l000214-authoring-rejection.json",
    )

    evidence = output / "evidence"
    evidence.mkdir()
    shutil.copyfile(args.phase_root / "corpus-manifest.json", evidence / "phase-a-corpus-manifest.json")
    shutil.copyfile(args.phase_root / "ledger.jsonl", evidence / "phase-a-ledger.jsonl")
    shutil.copyfile(args.admission_manifest, evidence / "lab-admission-manifest.json")
    shutil.copyfile(args.lab_root / "lab-progress-ledger.jsonl", evidence / "lab-progress-ledger.jsonl")
    shutil.copyfile(args.lab_root / "lab-progress-summary.json", evidence / "lab-progress-summary.json")
    environment_receipt = args.lab_root / "remote-environment-verification.current.json"
    if not environment_receipt.is_file():
        raise ValueError("remote environment receipt is missing")
    shutil.copyfile(environment_receipt, evidence / environment_receipt.name)
    lab_evidence = evidence / "lab"
    for ref in load(args.lab_root / "lab-progress-summary.json")["evidence_refs"]:
        source = args.lab_root / ref
        destination = lab_evidence / ref
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    clustering_evidence = evidence / "clustering"
    clustering_evidence.mkdir()
    for source, name in (
        (args.cluster_predecessor_root / "model/receipt.json", "v1-model-receipt.json"),
        (args.cluster_predecessor_root / "model/final.json", "v1-model-final.json"),
        (args.cluster_root / "model/receipt.json", "v2-model-receipt.json"),
        (args.cluster_root / "model/final.json", "v2-model-final.json"),
    ):
        shutil.copyfile(source, clustering_evidence / name)

    dataset_note = f"""# AKA qualified-parent Open-Cake review export

This directory is a publishable projection of external, append-only evidence. It does
not contain raw model events, credentials, complete GPU tensors, or mutable campaign
state.

- `phase-a-review.jsonl`: 677 canonical qualified-parent rows. Exactly 439 have a
  deterministic accepted static result, 237 retain reviewer/schema rejection, and one
  retains a pre-model infrastructure failure.
- `lab-terminal-results.jsonl`: the 57 static schedule/expressible/lowered admissions.
  Exactly 56 are fixed-instance dynamic-valid after B200 correctness, memcheck,
  racecheck, and independent complete-output recomputation; one is an authoring
  rejection with GPU not run.
- `ir-gap-clusters.json`: remote Sol/max semantic clustering of all 326 accepted
  `ir_gap` rows. The accompanying verifier proves exact 326-case/296-name partition.
- `l000214-authoring-rejection.json`: the sole Lab terminal non-dynamic record.

The cluster output is proposal-only. It approves, implements, releases, GPU-tests, and
performance-qualifies no IR primitive. The 56 Lab successes prove only the frozen
fixed instances. All rows have `performance_measured=false`; training eligibility is
false for the Lab set.

Source identities are recorded in `manifest.json`. Compact admission, progress,
independent-verification, node-result, environment, and clustering-treatment summaries
are under `evidence/`; raw model events and complete tensors remain in the external
append-only roots. SHA-256 values in the manifest exist only for this explicit external
handoff and the user-requested corpus identity, not as semantic correctness evidence.
"""
    (output / "DATASET.md").write_text(dataset_note, encoding="utf-8")

    artifact_names = sorted(
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "verification.json"}
    )
    manifest = {
        "schema": "open-cake.aka-qualified-ir-review-export-manifest.v1",
        "status": "verified_projection",
        "source": {
            "aka_origin": "git@github.com:qhy991/AKA.git",
            "aka_commit": "387aa7faf521a0b72c994ff15a7638cd7e6a8583",
            "open_cake_origin": "git@github.com:qhy991/open-cake-ir.git",
            "open_cake_frozen_commit": "7e212038404ecbb05ccd612381f4ba0328f680cb",
            "phase_a_manifest": "corpus-manifest.json",
            "phase_a_manifest_declared_sha256": "35851732044ae15f72725c384f2f4163d0d0f65b693c5055efb922f14bfeccf7",
            "lab_admission_manifest_sha256": digest(args.admission_manifest),
            "lab_progress_ledger_sha256": digest(args.lab_root / "lab-progress-ledger.jsonl"),
        },
        "counts": {
            "qualified_records": 677,
            "phase_a_ledger": 676,
            **phase_counts,
            "lab_admission": 57,
            **lab_counts,
            "lab_unknown": 0,
            "ir_gap_cases": 326,
            "ir_gap_exact_candidate_names": 296,
            "ir_gap_semantic_clusters": cluster_receipt["semantic_clusters"],
        },
        "artifacts": [
            {
                "path": name,
                "bytes": (output / name).stat().st_size,
                "sha256": digest(output / name),
            }
            for name in artifact_names
        ],
        "claim_boundary": {
            "phase_a": "static fixed-instance review",
            "lab": "fixed-instance B200 correctness and sanitizer evidence",
            "ir_clusters": "proposal only",
            "performance_measured": False,
            "training_authorized": False,
            "compiler_change_approved": False,
            "release_performed": False,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "created", "output": str(output), "counts": manifest["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
