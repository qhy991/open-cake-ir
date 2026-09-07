"""Common external correctness audit after arm-specific build."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, cast

from open_cake_ir.evaluation.workload import WorkloadContract


@dataclass(frozen=True)
class CorrectnessObservation:
    """Confirmed fixed-cell correctness and route observation."""

    case_id: str
    correctness_passed: bool
    exact_match: bool
    mismatch_count: int
    near_tie_mismatch_count: int
    tie_aware_distance_match: bool
    max_chosen_distance_excess: float
    output_sha256: str
    output_size_bytes: int
    kernel_calls: int
    fallback_calls: int


def _object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return cast(Mapping[str, object], value)


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{context} must be an integer >= {minimum}")
    return value


def _number(value: object, context: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{context} must be numeric")
    return float(value)


def _digest(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA256 digest")
    return value


def audit_flash_kmeans_assignment(
    result: object,
    workload: WorkloadContract,
    *,
    case_id: str,
) -> CorrectnessObservation:
    """Audit one legacy or new arm result against the common Flash-KMeans Workload Contract."""

    document = _object(result, "result")
    workload_document = workload.document
    if workload_document.get("operator") != "flash_kmeans_assign":
        raise ValueError("workload is not Flash-KMeans assignment")
    case = workload.case(case_id)
    shape = _object(case.get("shape"), "workload.case.shape")
    materialized = _object(case.get("materialized"), "workload.case.materialized")
    input_observation = _object(document.get("input"), "result.input")
    if input_observation.get("case_id") != case_id:
        raise ValueError("result input case differs")
    for tensor in ("tokens", "centroids"):
        expected = _object(materialized.get(tensor), f"workload.materialized.{tensor}")
        if (
            input_observation.get(f"{tensor}_sha256") != expected.get("sha256")
            or input_observation.get(f"{tensor}_bytes") != expected.get("size_bytes")
        ):
            raise ValueError(f"result {tensor} materialization differs")
    oracle = _object(document.get("oracle"), "result.oracle")
    expected_oracle = _object(
        materialized.get("oracle_assignments"), "workload.materialized.oracle_assignments"
    )
    if oracle.get("output_sha256") != expected_oracle.get("sha256") or oracle.get(
        "output_bytes"
    ) != expected_oracle.get("size_bytes"):
        raise ValueError("result oracle materialization differs")

    candidate = _object(document.get("candidate"), "result.candidate")
    metrics = _object(candidate.get("metrics"), "result.candidate.metrics")
    validation = _object(workload_document.get("validation"), "workload.validation")
    expected_assignments = _integer(shape.get("B"), "shape.B", minimum=1) * _integer(
        shape.get("N"), "shape.N", minimum=1
    )
    if (
        document.get("correctness_passed") is not True
        or document.get("performance_measured") is not False
        or candidate.get("output_contract_passed") is not True
        or metrics.get("schema_version") != 1
        or metrics.get("total_assignments") != expected_assignments
        or metrics.get("tie_atol") != validation.get("tie_diagnostic_atol")
        or metrics.get("tie_rtol") != validation.get("tie_diagnostic_rtol")
        or metrics.get("tie_aware_distance_match") is not True
        or _number(metrics.get("max_chosen_distance_excess"), "metrics.max_excess") != 0.0
    ):
        raise ValueError("result correctness contract differs")
    route = _object(document.get("route"), "result.route")
    progress = _object(document.get("progress"), "result.progress")
    compile_observation = _object(document.get("compile"), "result.compile")
    same_cubin = route.get("same_cubin_before_after") is True or route.get("same_cubin") is True
    if (
        compile_observation.get("compile_passed") is not True
        or not same_cubin
        or route.get("post_launch_synchronized") is not True
        or route.get("performance_measured") is not False
    ):
        raise ValueError("result compile or route contract differs")
    return CorrectnessObservation(
        case_id=case_id,
        correctness_passed=True,
        exact_match=metrics.get("exact_match") is True,
        mismatch_count=_integer(metrics.get("mismatch_count"), "metrics.mismatch_count"),
        near_tie_mismatch_count=_integer(
            metrics.get("near_tie_mismatch_count"), "metrics.near_tie_mismatch_count"
        ),
        tie_aware_distance_match=True,
        max_chosen_distance_excess=0.0,
        output_sha256=_digest(candidate.get("output_sha256"), "candidate.output_sha256"),
        output_size_bytes=_integer(candidate.get("output_bytes"), "candidate.output_bytes", minimum=1),
        kernel_calls=_integer(route.get("kernel_calls"), "route.kernel_calls", minimum=1),
        fallback_calls=_integer(progress.get("fallback_calls"), "progress.fallback_calls"),
    )
