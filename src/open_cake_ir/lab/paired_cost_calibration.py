"""Validated paired Evaluation observation for a future empirical cost fitter.

This projection does not fit a model or qualify a Study policy. The common
Evaluation receipt, rather than a supplier's latency field, owns the measurement.
"""
from __future__ import annotations

import json
import math
from typing import Mapping

from open_cake_ir.evaluation import EvaluationReceipt, LaunchableCandidate
from open_cake_ir.evaluation.paired import (
    PAIRED_KIND,
    candidate_identity,
    paired_protocol,
    validate_paired_broker,
    validate_paired_receipt,
)


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
