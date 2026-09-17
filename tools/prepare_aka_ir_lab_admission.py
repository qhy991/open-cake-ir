#!/usr/bin/env python3
"""Materialize static AKA Schedule survivors for external Lab admission."""

from __future__ import annotations

import argparse
import ast
import json
import shutil
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA = "open-cake.aka-ir-lab-admission.v1"
ENTRY_SCHEMA = "open-cake.aka-ir-lab-candidate.v1"
RESULT_SCHEMA = "open-cake.aka-qualified-ir-result.v1"
PORTABLE_SCHEMA = "aka.portable-kernel-parent.v1"


class AdmissionError(RuntimeError):
    pass


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AdmissionError(f"cannot read {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise AdmissionError(f"{label} must be one JSON object: {path}")
    return value


def _inside(root: Path, path: Path, label: str) -> Path:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise AdmissionError(f"{label} escapes its owner root: {path}") from error
    if not resolved.is_file() or resolved.is_symlink():
        raise AdmissionError(f"{label} must be a regular non-symlink file: {path}")
    return resolved


def _copy(source: Path, destination: Path) -> None:
    if destination.exists():
        raise AdmissionError(f"refusing to overwrite materialized input: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _accepted_schedule(result: dict[str, Any]) -> bool:
    verification = result.get("verification")
    eligibility = result.get("eligibility")
    if not isinstance(verification, dict) or not isinstance(eligibility, dict):
        return False
    gpu = eligibility.get("gpu")
    training = eligibility.get("training")
    return (
        result.get("schema") == RESULT_SCHEMA
        and result.get("status") == "reviewed"
        and result.get("owner") == "schedule"
        and result.get("current_ir_expressibility") == "expressible"
        and verification.get("status") == "accepted"
        and all(
            isinstance(verification.get(name), dict)
            and verification[name].get("status") == "passed"
            for name in ("parse", "validate", "lower")
        )
        and isinstance(gpu, dict)
        and gpu.get("eligible") is True
        and isinstance(training, dict)
        and training.get("eligible") is False
    )


def _artifact_paths(
    *, portable: dict[str, Any], dataset_root: Path
) -> list[tuple[str, Path]]:
    artifacts = portable.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise AdmissionError("portable record has no artifact roles")
    observed: set[str] = set()
    result: list[tuple[str, Path]] = []
    for role in sorted(artifacts):
        values = artifacts[role]
        if not isinstance(values, list):
            raise AdmissionError(f"portable artifact role {role!r} must be a list")
        for value in values:
            if not isinstance(value, str) or not value or value in observed:
                raise AdmissionError(f"invalid or duplicate portable artifact: {value!r}")
            observed.add(value)
            path = _inside(dataset_root, dataset_root / value, "portable artifact")
            result.append((value, path))
    return result


def _entry(
    *,
    run_root: Path,
    dataset_root: Path,
    event: dict[str, Any],
    output_root: Path,
    canary_case_id: str,
) -> dict[str, Any] | None:
    case_id = event.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise AdmissionError("ledger event has no case_id")
    item = run_root / "items" / case_id
    if not item.is_dir() or item.is_symlink():
        raise AdmissionError(f"item root is unavailable: {case_id}")
    receipt_ref = event.get("receipt")
    if not isinstance(receipt_ref, str):
        raise AdmissionError(f"ledger receipt is missing: {case_id}")
    receipt_path = _inside(run_root, run_root / receipt_ref, "verifier receipt")
    receipt = _load_json(receipt_path, "verifier receipt")
    if receipt.get("case_id") != case_id:
        raise AdmissionError(f"receipt case differs: {case_id}")
    result_ref = receipt.get("result")
    if not isinstance(result_ref, str):
        raise AdmissionError(f"receipt result is missing: {case_id}")
    result_path = _inside(item, item / result_ref, "review result")
    result = _load_json(result_path, "review result")
    if not _accepted_schedule(result):
        return None

    portable_path = _inside(item, item / "portable_record.json", "portable record")
    portable = _load_json(portable_path, "portable record")
    if (
        portable.get("schema") != PORTABLE_SCHEMA
        or portable.get("case_id") != case_id
        or portable.get("outcome") != "qualified"
        or portable.get("missing_facts") != []
        or portable.get("training_eligibility") is not False
        or portable.get("qualification", {}).get("validity") != "valid"
    ):
        raise AdmissionError(f"portable parent is not fully admitted: {case_id}")

    assessment_path = _inside(item, item / "verifier/assessment.json", "assessment")
    lowering_path = _inside(item, item / "verifier/lowering.json", "lowering")
    schedule_path = _inside(item, item / "verifier/schedule.json", "schedule")
    source_path = _inside(item, item / "verifier/lowered-source.py", "lowered source")
    assessment = _load_json(assessment_path, "assessment")
    lowering = _load_json(lowering_path, "lowering")
    if (
        assessment.get("accepted") is not True
        or assessment.get("lowering_eligible") is not True
        or assessment.get("target") != "sm_100a"
        or lowering.get("target") != "sm_100a"
        or lowering.get("source") != "verifier/lowered-source.py"
    ):
        raise AdmissionError(f"static target/lowering gate differs: {case_id}")
    route = lowering.get("route")
    if not isinstance(route, dict) or route.get("backend") != "triton":
        raise AdmissionError(f"Lab v1 admits only the observed Triton route: {case_id}")
    entry_point = route.get("entry_point")
    if not isinstance(entry_point, str) or not entry_point:
        raise AdmissionError(f"lowered entry point is missing: {case_id}")
    source = source_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(source_path))
    except SyntaxError as error:
        raise AdmissionError(f"lowered source is not valid Python: {case_id}") from error
    if not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == entry_point
        for node in tree.body
    ):
        raise AdmissionError(f"lowered wrapper entry point differs: {case_id}")

    workload = result.get("frozen_contract", {}).get("workload")
    if not isinstance(workload, dict) or not isinstance(workload.get("id"), str):
        raise AdmissionError(f"fixed workload is missing: {case_id}")
    optimization = result.get("eligibility", {}).get("optimization")
    if not isinstance(optimization, dict) or not isinstance(
        optimization.get("eligible"), bool
    ):
        raise AdmissionError(f"optimization eligibility is missing: {case_id}")

    case_output = output_root / "cases" / case_id
    inputs = case_output / "inputs"
    candidate = case_output / "candidate"
    candidate.mkdir(parents=True, exist_ok=False)
    fixed_inputs = {
        "problem.json": item / "problem.json",
        "portable_record.json": portable_path,
        "review_result.json": result_path,
        "schedule.json": schedule_path,
        "assessment.json": assessment_path,
        "lowering.json": lowering_path,
        "kernel.py": source_path,
    }
    for name, source_input in fixed_inputs.items():
        _copy(_inside(item, source_input, f"case input {name}"), inputs / name)
    _copy(source_path, candidate / "kernel.py")

    materialized_artifacts: list[dict[str, str]] = []
    for relative, artifact in _artifact_paths(
        portable=portable, dataset_root=dataset_root
    ):
        destination = inputs / "parent" / relative
        _copy(artifact, destination)
        materialized_artifacts.append(
            {"source_ref": relative, "materialized": str(destination.relative_to(case_output))}
        )

    metadata = {
        "schema": ENTRY_SCHEMA,
        "case_id": case_id,
        "derived_parent_id": result.get("derived_parent_id"),
        "assessment_scope": result.get("assessment_scope"),
        "target": "sm_100a",
        "backend": "triton",
        "entry_point": entry_point,
        "workload_id": workload["id"],
        "optimization_eligible": optimization["eligible"],
        "training_eligible": False,
        "canary": case_id == canary_case_id,
        "artifacts": materialized_artifacts,
    }
    (case_output / "candidate/provenance.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def prepare(
    *,
    run_root: Path,
    dataset_root: Path,
    output_root: Path,
    expected_count: int,
    canary_case_id: str,
) -> dict[str, Any]:
    run = run_root.expanduser().resolve(strict=True)
    dataset = dataset_root.expanduser().resolve(strict=True)
    output = output_root.expanduser().resolve()
    if output.exists():
        raise AdmissionError(f"output root already exists: {output}")
    output.mkdir(parents=True)
    ledger = _inside(run, run / "ledger.jsonl", "Phase A ledger")
    corpus = _load_json(_inside(run, run / "corpus-manifest.json", "corpus manifest"), "corpus manifest")

    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    for number, line in enumerate(ledger.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise AdmissionError(f"invalid ledger JSON at line {number}") from error
        if not isinstance(event, dict):
            raise AdmissionError(f"ledger line {number} is not an object")
        case_id = event.get("case_id")
        if case_id in seen:
            raise AdmissionError(f"duplicate ledger case: {case_id}")
        seen.add(case_id)
        if event.get("status") == "accepted":
            events.append(event)

    entries = [
        entry
        for event in sorted(events, key=lambda value: str(value.get("case_id")))
        if (
            entry := _entry(
                run_root=run,
                dataset_root=dataset,
                event=event,
                output_root=output,
                canary_case_id=canary_case_id,
            )
        )
        is not None
    ]
    if len(entries) != expected_count:
        raise AdmissionError(
            f"static survivor count differs: expected {expected_count}, observed {len(entries)}"
        )
    canaries = [entry for entry in entries if entry["canary"]]
    if len(canaries) != 1:
        raise AdmissionError(f"expected exactly one canary, observed {len(canaries)}")
    optimization_count = sum(bool(entry["optimization_eligible"]) for entry in entries)
    body = {
        "schema": MANIFEST_SCHEMA,
        "status": "static_candidates_materialized",
        "claim_boundary": (
            "Admission candidates only: no generated-Cake GPU correctness, sanitizer, "
            "performance, training, release, or generalization claim."
        ),
        "source": {
            "phase_a_ledger": str(ledger),
            "aka_revision": corpus.get("manifest", {}).get("source", {}).get("git_commit"),
            "compiler": corpus.get("manifest", {}).get("compiler", {}),
        },
        "policy": {
            "single_canary_before_scale": True,
            "max_in_flight_after_canary": 5,
            "fixed_target": "sm_100a/B200",
            "automatic_retry": False,
            "automatic_reroute": False,
        },
        "counts": {
            "phase_a_ledger": len(seen),
            "static_candidates": len(entries),
            "optimization_priority": optimization_count,
            "correctness_only_priority": len(entries) - optimization_count,
        },
        "canary_case_id": canary_case_id,
        "entries": entries,
    }
    (output / "admission-manifest.json").write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-a-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--canary-case-id", required=True)
    arguments = parser.parse_args()
    result = prepare(
        run_root=arguments.phase_a_root,
        dataset_root=arguments.dataset_root,
        output_root=arguments.output_root,
        expected_count=arguments.expected_count,
        canary_case_id=arguments.canary_case_id,
    )
    print(json.dumps({"status": result["status"], "counts": result["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
