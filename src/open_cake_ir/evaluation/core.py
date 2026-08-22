"""Common Evaluation from sealed launchable artifacts to append-only receipts."""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass, field
from hashlib import sha256
from types import MappingProxyType
from typing import Mapping, Protocol, cast

from .flash_kmeans import (
    classify_flash_kmeans_output,
    flash_kmeans_oracle,
    generate_flash_kmeans_case,
)
from .workload import WorkloadContract

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_ROLES = {
    "authored_source",
    "lowered_source",
    "compiler_expanded_source",
    "ttir",
    "ttgir",
    "llir",
    "ptx",
    "cubin",
    "sass",
    "launch_manifest",
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _plain_json(value: object) -> object:
    """Detach supported JSON values from caller-owned mutable containers."""

    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("canonical record object keys must be strings")
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("canonical record contains a non-JSON value")


def _freeze_json(value: object) -> object:
    """Recursively freeze a detached JSON value without changing its content."""

    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


@dataclass(frozen=True)
class LaunchableCandidate:
    """Sealed arm output at the seam before common correctness/timing."""

    candidate_sha256: str
    target: str
    entry_point: str
    artifact_roles: Mapping[str, str]
    launch_spec_sha256: str
    artifact_payloads: Mapping[str, bytes] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            _DIGEST.fullmatch(self.candidate_sha256) is None
            or _DIGEST.fullmatch(self.launch_spec_sha256) is None
            or self.target != "sm_100a"
            or not self.entry_point
            or not self.artifact_roles
            or "cubin" not in self.artifact_roles
            or any(
                role not in _ARTIFACT_ROLES or _DIGEST.fullmatch(digest) is None
                for role, digest in self.artifact_roles.items()
            )
        ):
            raise ValueError("LaunchableCandidate identity or artifacts differ")
        if self.artifact_payloads and (
            set(self.artifact_payloads) != set(self.artifact_roles)
            or any(
                not isinstance(payload, bytes)
                or not payload
                or sha256(payload).hexdigest() != self.artifact_roles[role]
                for role, payload in self.artifact_payloads.items()
            )
        ):
            raise ValueError("LaunchableCandidate artifact bytes differ from their seals")
        if (
            "lowered_source" in self.artifact_roles
            and "compiler_expanded_source" in self.artifact_roles
            and self.artifact_roles["lowered_source"]
            == self.artifact_roles["compiler_expanded_source"]
        ):
            raise ValueError("lowering input and compiler-expanded source are distinct artifacts")
        object.__setattr__(
            self,
            "artifact_roles",
            MappingProxyType(dict(self.artifact_roles)),
        )
        object.__setattr__(
            self,
            "artifact_payloads",
            MappingProxyType(dict(self.artifact_payloads)),
        )

    @property
    def canonical_sha256(self) -> str:
        return sha256(
            _canonical_json_bytes(
                {
                    "candidate_sha256": self.candidate_sha256,
                    "target": self.target,
                    "entry_point": self.entry_point,
                    "artifact_roles": dict(sorted(self.artifact_roles.items())),
                    "launch_spec_sha256": self.launch_spec_sha256,
                }
            )
        ).hexdigest()


@dataclass(frozen=True)
class EvaluationProtocol:
    """Frozen assay purpose and Workload case."""

    protocol_id: str
    purpose: str
    workload_sha256: str
    case_id: str
    timing: str

    def __post_init__(self) -> None:
        if (
            not self.protocol_id
            or self.purpose not in {"search", "confirmatory"}
            or _DIGEST.fullmatch(self.workload_sha256) is None
            or not self.case_id
            or self.timing not in {"none", "paired_cupti"}
        ):
            raise ValueError("EvaluationProtocol differs")

    @property
    def canonical_sha256(self) -> str:
        return sha256(
            _canonical_json_bytes(
                {
                    "protocol_id": self.protocol_id,
                    "purpose": self.purpose,
                    "workload_sha256": self.workload_sha256,
                    "case_id": self.case_id,
                    "timing": self.timing,
                }
            )
        ).hexdigest()


@dataclass(frozen=True)
class LaunchObservation:
    """Observed common launch result; arm build details remain behind the seam."""

    output: object
    kernel_calls: int
    fallback_calls: int
    launch_receipt_sha256: str

    def __post_init__(self) -> None:
        if (
            self.kernel_calls != 1
            or self.fallback_calls != 0
            or _DIGEST.fullmatch(self.launch_receipt_sha256) is None
        ):
            raise ValueError("LaunchObservation violates common route policy")


class CandidateLauncher(Protocol):
    """Adapter that launches one already-sealed candidate."""

    def launch(
        self,
        candidate: LaunchableCandidate,
        tokens: object,
        centroids: object,
        centroid_sq: object,
    ) -> LaunchObservation:
        """Launch exactly once and return output plus route evidence."""


@dataclass(frozen=True)
class EvaluationReceipt:
    """Append-only common Evaluation observation."""

    candidate_sha256: str
    workload_sha256: str
    evaluation_protocol_sha256: str
    purpose: str
    case_id: str
    correctness_passed: bool
    correctness: Mapping[str, object]
    kernel_calls: int
    fallback_calls: int
    launch_receipt_sha256: str
    timing: Mapping[str, object] | None
    artifact_payloads: Mapping[str, bytes] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            _DIGEST.fullmatch(self.candidate_sha256) is None
            or _DIGEST.fullmatch(self.workload_sha256) is None
            or _DIGEST.fullmatch(self.evaluation_protocol_sha256) is None
            or _DIGEST.fullmatch(self.launch_receipt_sha256) is None
            or self.purpose not in {"search", "confirmatory"}
            or not self.case_id
            or self.kernel_calls != 1
            or self.fallback_calls != 0
        ):
            raise ValueError("EvaluationReceipt identity or route differs")
        if self.artifact_payloads:
            if (
                set(self.artifact_payloads)
                != {"correctness_output", "launch_receipt", "timing_samples"}
                or any(not isinstance(payload, bytes) or not payload for payload in self.artifact_payloads.values())
                or sha256(self.artifact_payloads["launch_receipt"]).hexdigest()
                != self.launch_receipt_sha256
            ):
                raise ValueError("EvaluationReceipt artifact custody differs")
            try:
                correctness_raw = json.loads(self.artifact_payloads["correctness_output"])
                timing_raw = json.loads(self.artifact_payloads["timing_samples"])
                launch_raw = json.loads(self.artifact_payloads["launch_receipt"])
            except (UnicodeError, json.JSONDecodeError) as error:
                raise ValueError("EvaluationReceipt raw artifacts are not JSON") from error
            if not isinstance(correctness_raw, Mapping) or not isinstance(
                launch_raw, Mapping
            ):
                raise ValueError("EvaluationReceipt raw correctness or launch receipt differs")
            raw_metrics = correctness_raw.get("metrics", correctness_raw)
            if not isinstance(raw_metrics, Mapping) or dict(raw_metrics) != dict(
                self.correctness
            ):
                raise ValueError("EvaluationReceipt correctness summary differs from raw output")
            derived_correctness = correctness_raw.get("passed")
            if derived_correctness is None and {
                "tie_aware_distance_match",
                "max_chosen_distance_excess",
            } <= set(raw_metrics):
                derived_correctness = bool(raw_metrics["tie_aware_distance_match"]) and (
                    raw_metrics["max_chosen_distance_excess"] == 0.0
                )
            if derived_correctness is not None and (
                derived_correctness is True
            ) != self.correctness_passed:
                raise ValueError("EvaluationReceipt correctness disposition differs from raw output")
            if launch_raw.get("candidate_sha256") not in {
                None,
                self.candidate_sha256,
            }:
                raise ValueError("EvaluationReceipt launch candidate differs")
            if self.timing is not None:
                cohorts_value = (
                    timing_raw.get("cohorts_ms")
                    if isinstance(timing_raw, Mapping)
                    else [timing_raw]
                )
                if not isinstance(cohorts_value, list) or not cohorts_value:
                    raise ValueError("EvaluationReceipt timing samples differ")
                cohorts: list[list[float]] = []
                for cohort in cohorts_value:
                    if not isinstance(cohort, list) or not cohort:
                        raise ValueError("EvaluationReceipt timing cohort differs")
                    values = [float(value) for value in cohort]
                    if any(not math.isfinite(value) or value <= 0 for value in values):
                        raise ValueError("EvaluationReceipt timing sample differs")
                    cohorts.append(values)
                pooled = statistics.median(value for cohort in cohorts for value in cohort)
                stable = all(
                    (
                        math.sqrt(
                            sum((value - statistics.fmean(cohort)) ** 2 for value in cohort)
                            / len(cohort)
                        )
                        / statistics.fmean(cohort)
                    )
                    <= 0.05
                    for cohort in cohorts
                )
                if (
                    not math.isclose(
                        float(self.timing.get("pooled_median_ms")),
                        pooled,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    or (self.timing.get("measurement_quality_passed") is True)
                    != stable
                ):
                    raise ValueError("EvaluationReceipt timing summary differs from raw samples")
        detached_correctness = json.loads(
            _canonical_json_bytes(_plain_json(self.correctness))
        )
        object.__setattr__(
            self,
            "correctness",
            cast(Mapping[str, object], _freeze_json(detached_correctness)),
        )
        if self.timing is not None:
            detached_timing = json.loads(
                _canonical_json_bytes(_plain_json(self.timing))
            )
            object.__setattr__(
                self,
                "timing",
                cast(Mapping[str, object], _freeze_json(detached_timing)),
            )
        object.__setattr__(
            self,
            "artifact_payloads",
            MappingProxyType(dict(self.artifact_payloads)),
        )

    @property
    def candidate_disposition(self) -> str:
        return "qualified" if self.correctness_passed else "correctness_rejected"

    @property
    def measurement_quality(self) -> str:
        if self.timing is None:
            return "not_measured"
        return "stable" if self.timing.get("measurement_quality_passed") is True else "unstable"

    @property
    def canonical_sha256(self) -> str:
        return sha256(
            _canonical_json_bytes(
                {
                    "candidate_sha256": self.candidate_sha256,
                    "workload_sha256": self.workload_sha256,
                    "evaluation_protocol_sha256": self.evaluation_protocol_sha256,
                    "purpose": self.purpose,
                    "case_id": self.case_id,
                    "correctness_passed": self.correctness_passed,
                    "correctness": _plain_json(self.correctness),
                    "kernel_calls": self.kernel_calls,
                    "fallback_calls": self.fallback_calls,
                    "launch_receipt_sha256": self.launch_receipt_sha256,
                    "timing": _plain_json(self.timing) if self.timing is not None else None,
                    "artifact_payload_sha256": {
                        role: sha256(payload).hexdigest()
                        for role, payload in sorted(self.artifact_payloads.items())
                    },
                }
            )
        ).hexdigest()


def evaluate_flash_kmeans(
    candidate: LaunchableCandidate,
    workload: WorkloadContract,
    protocol: EvaluationProtocol,
    launcher: CandidateLauncher,
    *,
    device: str,
) -> EvaluationReceipt:
    """Run materialization, external oracle, one launch, then correctness."""

    if protocol.workload_sha256 != workload.canonical_sha256:
        raise ValueError("EvaluationProtocol Workload bytes differ")
    if protocol.timing != "none":
        raise ValueError("timing protocol requires a separately retained timing assay")
    tokens, centroids = generate_flash_kmeans_case(
        workload, protocol.case_id, device=device
    )
    oracle = flash_kmeans_oracle(
        workload, tokens, centroids, case_id=protocol.case_id
    )
    torch = __import__("torch")
    centroids_fp32 = centroids.to(torch.float32)
    centroid_sq = (centroids_fp32 * centroids_fp32).sum(
        dim=-1, dtype=torch.float32
    ).contiguous()
    launch = launcher.launch(candidate, tokens, centroids, centroid_sq)
    passed, metrics = classify_flash_kmeans_output(
        workload,
        tokens,
        centroids,
        launch.output,
        oracle,
        case_id=protocol.case_id,
    )
    return EvaluationReceipt(
        candidate_sha256=candidate.candidate_sha256,
        workload_sha256=workload.canonical_sha256,
        evaluation_protocol_sha256=protocol.canonical_sha256,
        purpose=protocol.purpose,
        case_id=protocol.case_id,
        correctness_passed=passed,
        correctness=MappingProxyType(dict(metrics)),
        kernel_calls=launch.kernel_calls,
        fallback_calls=launch.fallback_calls,
        launch_receipt_sha256=launch.launch_receipt_sha256,
        timing=None,
    )
