"""Candidate-independent task throughput projected from audited confirmations.

The tensor ABI is a fixed byte convention, not a proof of minimum DRAM traffic.
Hardware rates belong to Target; qualification and promotion remain owned by Lab.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

from open_cake_ir.compiler.target import Target
from open_cake_ir.evidence import EvidenceStore
from open_cake_ir.lab.bindings import source_reference_path
from .workloads import load_workload

POLICY = "task_efficiency_v1"
_WIDTHS = {"fp32": 4, "bf16": 2, "int32": 4}


def task_work(workload, case_id: str) -> dict:
    """Count each declared input once and each output once, independent of Schedule.

    In-place aliases and selective accesses require their own task convention;
    these counts are explicitly the ABI reference, never actual issued loads.
    """
    arguments = workload.tensor_abi(case_id)
    inputs = sum(math.prod(arg.shape) * _WIDTHS[arg.dtype]
                 for arg in arguments if arg.mode == "input")
    outputs = sum(math.prod(arg.shape) * _WIDTHS[arg.dtype]
                  for arg in arguments if arg.mode == "output")
    return {"convention": "tensor_abi_read_inputs_once_write_outputs_once",
            "scope": "logical_task_io_not_physical_traffic_or_minimum_dram_bytes",
            "read_bytes": inputs, "write_bytes": outputs,
            "logical_io_bytes": inputs + outputs,
            "shape": workload.case(case_id)["shape"],
            "arithmetic_operations": None,
            "arithmetic_status": "task_arithmetic_convention_and_matching_rate_not_bound"}


def score_measurement(work: Mapping, target: Target, latency_ms: float, *, cache_protocol: str) -> dict:
    """A fixed-task reference score; it cannot infer hardware activity from time."""
    if (type(latency_ms) not in (int, float) or not math.isfinite(latency_ms)
            or latency_ms <= 0):
        raise ValueError("task efficiency requires a positive finite qualified latency")
    count = work.get("logical_io_bytes")
    if type(count) is not int or count <= 0:
        raise ValueError("task efficiency requires a positive fixed task byte count")
    bandwidth = target.peak.memory_bandwidth if target.peak is not None else None
    throughput = count / (latency_ms / 1000.0)
    fraction = throughput / bandwidth.value if bandwidth is not None else None
    return {
        "latency_ms": float(latency_ms),
        "effective_bandwidth_gb_per_second": throughput / 1e9,
        "primary_score": {
            "metric": "logical_task_bandwidth_reference_pct",
            "value": None if fraction is None else 100 * fraction,
            "unit": "percent",
            "status": "unavailable" if fraction is None else "reference_ratio",
            "direction": "higher_is_better_within_exact_task_and_assay",
        },
        "bandwidth_reference": None if bandwidth is None else {
            "bytes_per_second": bandwidth.value, "source": bandwidth.source.value,
            "observed_at": bandwidth.observed_at, "target": target.target_id,
        },
        "cache_protocol": cache_protocol,
        "physical_dram_utilization_pct": None,
        "traffic_efficiency": None,
        "arithmetic_efficiency_pct": None,
        "roofline_efficiency_pct": None,
        "coverage": "logical_task_io_reference_only",
        "missing": [*([] if bandwidth is not None else ["target_memory_bandwidth_reference"]),
                    "measured_memory_level_traffic", "task_arithmetic_convention_and_matching_rate",
                    "complete_applicable_roofline"],
        "notes": ["The reference score is not measured DRAM utilization or a time breakdown.",
                  "Warm-cache logical rates may exceed external-memory bandwidth; ratios are not clamped.",
                  "Arithmetic, minimum-byte and traffic-counter coverage are not inferred from tensor names."],
    }


def campaign_performance(project_root, campaign, report) -> dict:
    """Project only confirmations already admitted by the existing full Run audit.

    Compiler source and receipt semantics are verified by TaskLab.audit before this
    projection. Do not rehash its closure or manufacture acceptance from a report.
    """
    root = Path(project_root).resolve()
    lock = campaign.lock.document
    if campaign.lock.claim_scope != "artifact_optimization_only":
        raise ValueError("task efficiency reporting requires artifact optimization scope")
    if lock["analysis_plan"].get("performance_reporting") != POLICY:
        raise ValueError("task efficiency reporting is not declared by this Campaign")
    case_id = lock["evaluation_protocol"]["case_id"]
    result = {"policy": POLICY, "workload_id": campaign.lock.workload_id, "case_id": case_id,
              "target": lock["execution"]["target"], "rows": [], "missing": [],
              "ranking_scope": "within_exact_workload_case_target_and_assay_only",
              "threshold_status": "not_calibrated_no_hard_efficiency_threshold",
              "selection": "descending_reference_score_equivalent_to_ascending_confirmed_latency",
              "speedup_role": "auxiliary_fixed_baseline_comparison"}
    eligible = [audit for audit in report.run_audits
                if audit.archive_integrity and audit.filesystem_custody_verified
                and audit.protocol_adherence == "adhered"
                and report.descriptive["semantic_replay_by_run"].get(audit.run_id) is True]
    if not eligible:
        result["missing"] = ["no_adhered_custody_verified_semantically_replayed_run"]
        return result
    _, workload_path = source_reference_path(root, lock["workload"]["path"], "performance workload")
    workload = load_workload(workload_path)
    if (workload.canonical_sha256 != lock["workload"]["canonical_sha256"]
            or workload.target != result["target"]):
        raise ValueError("performance Workload differs from the audited Campaign")
    try:
        work = task_work(workload, case_id)
    except ValueError as error:
        result["missing"] = [f"task_io_convention_unavailable: {error}"]
        return result
    result["work"] = work
    # Read the target reference from the already admitted Compiler lock, not a
    # current/default device table that could silently change the denominator.
    _, revision_path = source_reference_path(root, lock["compiler_revision"]["path"], "performance Compiler")
    revision = json.loads(revision_path.read_bytes())
    _, target_path = source_reference_path(root, revision["target_definitions"][workload.target]["path"], "performance target")
    target = Target.load(target_path)
    if target.target_id != workload.target:
        raise ValueError("performance target differs from the Workload")
    result["target_citations"] = json.loads(target_path.read_bytes()).get("citations", [])
    evidence = EvidenceStore.open(campaign.evidence_root)
    paired = lock["evaluation_protocol"].get("paired_timing", {})
    cache = "warm_no_explicit_flush" if "metal" in paired.get("kind", "") else "cold_l2_cache"
    baseline = lock["execution"].get("fixed_baseline", {}).get("candidate", {})
    for audit in eligible:
        for event in evidence.replay_events(audit.run_id):
            payload = event.get("payload", {})
            if event.get("kind") != "candidate_evaluated" or payload.get("purpose") != "confirmatory":
                continue
            refs = [ref for ref in payload.get("objects", []) if ref.get("role") == "evaluation_receipt"]
            if len(refs) != 1:
                raise ValueError("performance confirmation receipt reference differs")
            receipt = json.loads(evidence.read_object(refs[0]))
            timing = receipt.get("timing")
            if (receipt.get("purpose") != "confirmatory" or receipt.get("case_id") != case_id
                    or receipt.get("candidate_sha256") != payload["candidate_sha256"]):
                raise ValueError("performance confirmation identity differs")
            if (receipt.get("correctness_passed") is not True or not isinstance(timing, Mapping)
                    or timing.get("measurement_quality_passed") is not True):
                continue
            medians = timing.get("pooled_medians_ms", {"candidate": timing.get("pooled_median_ms")})
            for role in ("candidate", "baseline"):
                if role not in medians:
                    continue
                row = score_measurement(work, target, medians[role], cache_protocol=cache)
                row.update(run_id=audit.run_id, turn=payload["turn"], role=role,
                           candidate_id=payload["candidate_sha256"] if role == "candidate" else baseline.get("candidate_sha256"),
                           confirmation_event=event.get("sequence"),
                           speedup=timing.get("speedup") if role == "candidate" else 1.0)
                result["rows"].append(row)
    result["rows"].sort(key=lambda row: (row["latency_ms"], row["run_id"], row["turn"], row["role"]))
    if not result["rows"]:
        result["missing"].append("no_correct_stable_confirmatory_measurement")
    elif target.peak is None or target.peak.memory_bandwidth is None:
        result["missing"].append("target_memory_bandwidth_reference")
    return result
