"""Paired Evaluation receipt projection and pure exact-case cost fitting.

The common Evaluation receipt owns measured latency. Fitting alone does not
establish source or broker custody and qualifies no Study selection policy.
"""
from __future__ import annotations

import json
import itertools
import math
import statistics
from typing import Mapping

from open_cake_ir.compiler import EmpiricalCostModel
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.evaluation import EvaluationReceipt, LaunchableCandidate
from open_cake_ir.evaluation.paired import (
    PAIRED_KIND,
    candidate_identity,
    paired_protocol,
    validate_paired_broker,
    validate_paired_receipt,
)
from .selection import _empirical_filter, _paired_empirical_context


def observed_paired_cost(
    receipt: EvaluationReceipt, *, candidate: LaunchableCandidate,
    baseline: LaunchableCandidate, evaluation_protocol: Mapping[str, object],
    worker_result: Mapping[str, object],
) -> dict[str, object]:
    """Replay one successful fixed-baseline CUPTI observation in microseconds."""
    protocol = paired_protocol(evaluation_protocol)
    if (protocol is None or evaluation_protocol["paired_timing"]["kind"] != PAIRED_KIND
            or not receipt.artifact_payloads):
        raise ValueError("paired cost observation requires retained CUPTI artifacts")
    raw = json.loads(receipt.artifact_payloads["timing_samples"])
    correctness = json.loads(receipt.artifact_payloads["correctness_output"])
    launch = json.loads(receipt.artifact_payloads["launch_receipt"])
    validate_paired_receipt(
        receipt, raw, correctness, launch, evaluation=evaluation_protocol,
        baseline=candidate_identity(baseline), candidate=candidate,
    )
    if (worker_result.get("admitted") is not True or worker_result.get("mode") != "exclusive"
            or worker_result.get("error") is not None
            or worker_result.get("failure_class") is not None):
        raise ValueError("paired cost observation lacks exclusive worker admission")
    result_receipt = worker_result.get("receipt")
    if (not isinstance(result_receipt, Mapping)
            or any(result_receipt.get(key) != value for key, value in {
                "correctness_passed": receipt.correctness_passed,
                "correctness": dict(receipt.correctness),
                "kernel_calls": receipt.kernel_calls,
                "fallback_calls": receipt.fallback_calls,
                "timing": dict(receipt.timing) if receipt.timing is not None else None,
            }.items())):
        raise ValueError("paired cost worker and retained receipt differ")
    counters = worker_result.get("counters")
    if not isinstance(counters, Mapping):
        raise ValueError("paired cost worker counters are missing")
    validate_paired_broker(receipt, worker_result.get("job_id"), counters)
    if (receipt.correctness_passed is not True or receipt.timing is None
            or receipt.timing.get("measurement_quality_passed") is not True):
        raise ValueError("paired cost requires correct, stable device measurement")
    medians = receipt.timing["pooled_medians_ms"]
    durations = {role: float(medians[role]) * 1000 for role in protocol.arms}
    if any(not math.isfinite(value) or value <= 0 for value in durations.values()):
        raise ValueError("paired cost durations must be finite and positive")
    return {
        "candidate_record_sha256": candidate.canonical_sha256,
        "baseline_record_sha256": baseline.canonical_sha256,
        "workload_sha256": receipt.workload_sha256,
        "case_id": receipt.case_id,
        "evaluation_protocol_sha256": receipt.evaluation_protocol_sha256,
        "job_id": worker_result["job_id"],
        "gpu_uuid": raw["gpu_uuid"],
        "candidate_us": durations["candidate"],
        "baseline_us": durations["baseline"],
        "sample_count": receipt.timing["pooled_sample_counts"]["candidate"],
    }


def derive_paired_cost_model(
    *, model_id: str, compiler_revision_id: str, target: str, executor,
    workload_sha256: str, case_id: str, evaluation_protocol: Mapping[str, object],
    baseline: LaunchableCandidate, rows: list[Mapping[str, object]],
    varying_dimensions: list[dict[str, object]], acceptance: Mapping[str, object],
) -> tuple[bool, dict[str, object], dict[str, object]]:
    """Fit exact-pool points from separately verified paired observations.

    The caller must first establish each row's Schedule/compiled artifact and GPU
    Infra custody. This pure derivation cannot turn unverified supplier rows into
    device evidence; its model remains external and advisory.
    """
    required_limits = {"maximum_baseline_drift_ratio", "maximum_mape",
                       "maximum_relative_error", "maximum_top2_regret_ratio",
                       "envelope_allowance"}
    if (set(acceptance) != required_limits or baseline.target != target
            or not isinstance(model_id, str) or not model_id
            or not isinstance(compiler_revision_id, str) or not compiler_revision_id
            or not isinstance(varying_dimensions, list) or not varying_dimensions):
        raise ValueError("paired cost calibration authority differs")
    for key in required_limits:
        value = acceptance[key]
        if (type(value) not in (int, float) or not math.isfinite(value)
                or value < (0 if key == "envelope_allowance" else 1 if key.endswith("ratio") else 0)
                or key in {"maximum_mape", "maximum_relative_error"} and value == 0):
            raise ValueError(f"paired cost acceptance {key} differs")
    if acceptance["envelope_allowance"] >= 1:
        raise ValueError("paired cost envelope allowance differs")
    context = _paired_empirical_context(
        executor, workload_sha256=workload_sha256, case_id=case_id,
        evaluation_protocol=evaluation_protocol,
        baseline_identity=candidate_identity(baseline),
    )
    scope = json.loads(context["input_scope"])
    groups: dict[str, dict[str, Mapping[str, object]]] = {}
    templates: dict[str, dict] = {}
    identities: dict[str, str] = {}
    baseline_us: list[float] = []
    gpu_uuids: set[str] = set()
    jobs: set[str] = set()
    for row in rows:
        if (not isinstance(row, Mapping) or set(row) != {"candidate_id", "schedule", "split", "observation"}
                or not isinstance(row["candidate_id"], str) or not row["candidate_id"]
                or row["split"] not in {"fit", "calibration", "audit"}
                or not isinstance(row["observation"], Mapping)):
            raise ValueError("paired cost row fields differ")
        name, split, observation = row["candidate_id"], row["split"], row["observation"]
        if (observation.get("workload_sha256") != workload_sha256
                or observation.get("case_id") != case_id
                or observation.get("evaluation_protocol_sha256") != scope["evaluation_protocol_sha256"]
                or observation.get("baseline_record_sha256") != baseline.canonical_sha256):
            raise ValueError("paired cost row assay or baseline differs")
        for field in ("candidate_us", "baseline_us"):
            value = observation.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("paired cost row duration differs")
        if (not isinstance(observation.get("candidate_record_sha256"), str)
                or not isinstance(observation.get("gpu_uuid"), str) or not observation["gpu_uuid"]
                or not isinstance(observation.get("job_id"), str) or not observation["job_id"]
                or type(observation.get("sample_count")) is not int
                or observation["sample_count"] != len(evaluation_protocol["paired_timing"]["pair_order"]) * evaluation_protocol["paired_timing"]["samples_per_cohort"]):
            raise ValueError("paired cost row device evidence differs")
        template = Schedule.from_dict(row["schedule"]).canonical_document(row["schedule"])
        if template["target"] != target:
            raise ValueError("paired cost Schedule target differs")
        if name in templates and templates[name] != template:
            raise ValueError("paired cost Schedule changed between splits")
        if name in identities and identities[name] != observation["candidate_record_sha256"]:
            raise ValueError("paired cost compiled candidate changed between splits")
        templates[name] = template
        identities[name] = observation["candidate_record_sha256"]
        group = groups.setdefault(name, {})
        if split in group:
            raise ValueError("paired cost split is duplicated")
        group[split] = observation
        baseline_us.append(float(observation["baseline_us"]))
        gpu_uuids.add(observation["gpu_uuid"])
        jobs.add(observation["job_id"])
    if len(groups) != 3 or any(set(group) != {"fit", "calibration", "audit"} for group in groups.values()):
        raise ValueError("paired cost requires three complete fit/calibration/audit candidates")
    if len(gpu_uuids) != 1:
        raise ValueError("paired cost observations span multiple GPUs")
    baseline_drift = max(baseline_us) / min(baseline_us)
    if baseline_drift > acceptance["maximum_baseline_drift_ratio"]:
        raise ValueError("paired cost fixed baseline drift differs")
    document: dict[str, object] = {
        "schema_version": 3, "model_id": model_id,
        "compiler_revision_id": compiler_revision_id, "target": target,
        "context": context,
        "reported_evidence": {"scope": "paired exact-case candidate pool; external advisory only",
                              "job_ids": sorted(jobs), "gpu_uuid": next(iter(gpu_uuids))},
        "curves": [],
    }
    for name in sorted(groups):
        template = templates[name]
        buffers = {buffer["name"]: buffer for buffer in template["buffers"]}
        try:
            extents = [buffers[binding["buffer"]]["shape"][binding["dimension"]]
                       for binding in varying_dimensions]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("paired cost varying dimensions differ") from error
        if not extents or len(set(extents)) != 1 or type(extents[0]) is not int or extents[0] <= 0:
            raise ValueError("paired cost varying extent differs")
        fit_us = float(groups[name]["fit"]["candidate_us"])
        calibration_us = float(groups[name]["calibration"]["candidate_us"])
        envelope = abs(calibration_us / fit_us - 1) + float(acceptance["envelope_allowance"])
        if envelope >= 1:
            raise ValueError("paired cost calibration envelope is unbounded")
        document["curves"].append({
            "template": template, "varying_dimensions": varying_dimensions,
            "extent_multiple": 1,
            "points": [{"extent": extents[0], "kernel_us": fit_us}],
            "relative_error_envelope": envelope,
        })
    model = EmpiricalCostModel(document)
    audit_rows = []
    for name in sorted(groups):
        prediction = model.estimate(
            templates[name], compiler_revision_id=compiler_revision_id, target=target,
            compiled_compiler_version=context["runtime"]["compiler_version"],
        )
        if prediction["covered"] is not True:
            raise ValueError(f"paired cost model does not cover its candidate: {prediction['reason']}")
        observed = float(groups[name]["audit"]["candidate_us"])
        audit_rows.append({"candidate_id": name, "observed_us": observed,
                           "predicted_us": prediction["predicted_kernel_us"],
                           "relative_error": abs(prediction["predicted_kernel_us"] / observed - 1),
                           "inside_descriptive_range": (prediction["empirical_range_us"][0] <= observed <= prediction["empirical_range_us"][1]),
                           "empirical_cost": prediction})
    best = min(row["observed_us"] for row in audit_rows)
    regrets = []
    for provider_order in itertools.permutations(audit_rows):
        ordered, decision = _empirical_filter([
            {"candidate_sha256": row["candidate_id"], "disposition": "launchable",
             "empirical_cost": row["empirical_cost"]}
            for row in provider_order
        ])
        if not decision["order_applied"]:
            raise ValueError("paired cost audit has incomplete candidate coverage")
        survivors = {row["candidate_sha256"] for row in ordered[:2]}
        regret = min(row["observed_us"] for row in audit_rows
                     if row["candidate_id"] in survivors) / best
        regrets.append({"provider_order": [row["candidate_id"] for row in provider_order],
                        "selected_ids": [row["candidate_sha256"] for row in ordered[:2]],
                        "top2_regret_ratio": regret})
    metrics = {"audit_case_count": len(audit_rows),
               "mean_relative_error": statistics.mean(row["relative_error"] for row in audit_rows),
               "max_relative_error": max(row["relative_error"] for row in audit_rows),
               "max_top2_regret_ratio": max(row["top2_regret_ratio"] for row in regrets),
               "baseline_drift_ratio": baseline_drift,
               "descriptive_range_covered_count": sum(row["inside_descriptive_range"] for row in audit_rows)}
    passed = (metrics["mean_relative_error"] <= acceptance["maximum_mape"]
              and metrics["max_relative_error"] <= acceptance["maximum_relative_error"]
              and metrics["max_top2_regret_ratio"] <= acceptance["maximum_top2_regret_ratio"])
    audit = {"passed": passed, "metrics": metrics,
             "rows": [{key: value for key, value in row.items() if key != "empirical_cost"}
                      for row in audit_rows], "regrets": regrets}
    document["reported_evidence"]["validation"] = metrics
    return passed, document, audit
