"""Manifest-driven exact-shape portfolio and fail-closed dispatch."""

from __future__ import annotations
from .seed import ExactShape

import json
import math
import re
from dataclasses import dataclass
from hashlib import sha256
from statistics import median
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence

from open_cake_ir.evaluation.core import LaunchableCandidate
from open_cake_ir.evaluation.workload import WorkloadContract

_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")




@dataclass(frozen=True)
class PortfolioEntry:
    """One Workload case mapped to one sealed launchable candidate."""

    case_id: str
    semantic_key: ExactShape
    candidate: LaunchableCandidate


@dataclass(frozen=True)
class PortfolioArtifact:
    """The sole case/key-to-candidate mapping used by the dispatcher."""

    workload_sha256: str
    seed_sha256: str
    entries: tuple[PortfolioEntry, ...]
    unsupported_policy: str = "reject"
    fallback_policy: str = "forbidden"

    @classmethod
    def build(
        cls,
        workload: WorkloadContract,
        seed_sha256: str,
        candidates: Mapping[str, LaunchableCandidate],
    ) -> "PortfolioArtifact":
        """Build from Workload cases, never from a second handwritten shape table."""

        if _DIGEST.fullmatch(seed_sha256) is None or not candidates:
            raise ValueError("portfolio seed or candidates differ")
        entries: list[PortfolioEntry] = []
        seen_keys: set[ExactShape] = set()
        for case_id in sorted(candidates):
            candidate = candidates[case_id]
            case = workload.case(case_id)
            key = ExactShape.from_case(case)
            if key in seen_keys:
                raise ValueError("portfolio semantic keys must be unique")
            seen_keys.add(key)
            entries.append(PortfolioEntry(case_id, key, candidate))
        return cls(workload.canonical_sha256, seed_sha256, tuple(entries))

    @property
    def document(self) -> Mapping[str, object]:
        return {
            "workload_sha256": self.workload_sha256,
            "seed_sha256": self.seed_sha256,
            "entries": [
                {
                    "case_id": entry.case_id,
                    "semantic_key": entry.semantic_key.as_dict(),
                    "candidate": {
                        "candidate_sha256": entry.candidate.candidate_sha256,
                        "candidate_record_sha256": entry.candidate.canonical_sha256,
                        "target": entry.candidate.target,
                        "entry_point": entry.candidate.entry_point,
                        "artifact_roles": dict(
                            sorted(entry.candidate.artifact_roles.items())
                        ),
                        "launch_spec_sha256": entry.candidate.launch_spec_sha256,
                    },
                }
                for entry in self.entries
            ],
            "unsupported_policy": self.unsupported_policy,
            "fallback_policy": self.fallback_policy,
        }

    @classmethod
    def from_document(cls, value: object) -> "PortfolioArtifact":
        if not isinstance(value, Mapping) or set(value) != {
            "workload_sha256",
            "seed_sha256",
            "entries",
            "unsupported_policy",
            "fallback_policy",
        }:
            raise ValueError("PortfolioArtifact fields differ")
        raw_entries = value["entries"]
        if not isinstance(raw_entries, list) or not raw_entries:
            raise ValueError("PortfolioArtifact entries differ")
        entries: list[PortfolioEntry] = []
        for raw in raw_entries:
            if not isinstance(raw, Mapping) or set(raw) != {
                "case_id",
                "semantic_key",
                "candidate",
            }:
                raise ValueError("PortfolioArtifact entry fields differ")
            key = raw["semantic_key"]
            candidate_value = raw["candidate"]
            if not isinstance(key, Mapping) or not isinstance(candidate_value, Mapping):
                raise ValueError("PortfolioArtifact entry payload differs")
            candidate = LaunchableCandidate(
                candidate_sha256=str(candidate_value["candidate_sha256"]),
                target=str(candidate_value["target"]),
                entry_point=str(candidate_value["entry_point"]),
                artifact_roles=dict(candidate_value["artifact_roles"]),
                launch_spec_sha256=str(candidate_value["launch_spec_sha256"]),
            )
            if candidate.canonical_sha256 != candidate_value.get("candidate_record_sha256"):
                raise ValueError("PortfolioArtifact candidate record differs")
            entries.append(
                PortfolioEntry(
                    str(raw["case_id"]),
                    ExactShape(
                        int(key["B"]), int(key["N"]), int(key["K"]), int(key["D"])
                    ),
                    candidate,
                )
            )
        artifact = cls(
            str(value["workload_sha256"]),
            str(value["seed_sha256"]),
            tuple(entries),
            str(value["unsupported_policy"]),
            str(value["fallback_policy"]),
        )
        if len({entry.case_id for entry in entries}) != len(entries):
            raise ValueError("PortfolioArtifact case identities differ")
        return artifact

    @property
    def canonical_sha256(self) -> str:
        return sha256(_canonical_json_bytes(self.document)).hexdigest()


class TensorLike(Protocol):
    """Minimum host-visible tensor contract required before dispatch."""

    shape: Sequence[int]
    dtype: object
    device: object
    is_cuda: bool

    def is_contiguous(self) -> bool: ...

    def data_ptr(self) -> int: ...


class PortfolioCandidateLauncher(Protocol):
    """Launch one selected candidate after host admission."""

    def launch(self, candidate: LaunchableCandidate, arguments: Sequence[TensorLike]) -> None:
        """Launch exactly once or raise without invoking a fallback."""


@dataclass(frozen=True)
class DispatchReceipt:
    """One route observation with explicit no-fallback accounting."""

    selected_case_id: str | None
    selected_candidate_sha256: str | None
    kernel_call_delta: int
    rejection_delta: int
    fallback_call_delta: int


class ExactShapeDispatcher:
    """Validate tensors and select only through a PortfolioArtifact manifest."""

    def __init__(self, artifact: PortfolioArtifact, launcher: PortfolioCandidateLauncher) -> None:
        self._artifact = artifact
        self._launcher = launcher
        self._by_key = {entry.semantic_key: entry for entry in artifact.entries}
        if len(self._by_key) != len(artifact.entries):
            raise ValueError("portfolio dispatcher mapping differs")
        self.selections = {entry.case_id: 0 for entry in artifact.entries}
        self.kernel_calls = 0
        self.rejections = 0
        self.fallback_calls = 0

    def _reject(self, message: str) -> None:
        self.rejections += 1
        raise ValueError(message)

    def dispatch(self, arguments: Sequence[TensorLike]) -> DispatchReceipt:
        """Reject unsupported or invalid calls before the candidate launcher."""

        if len(arguments) != 4:
            self._reject("dispatcher requires exactly four tensors")
        tokens, centroids, centroid_sq, assignments = arguments
        shapes = tuple(tuple(int(axis) for axis in value.shape) for value in arguments)
        if len(shapes[0]) != 3 or len(shapes[1]) != 3:
            self._reject("dispatcher tensor ranks differ")
        key = ExactShape(shapes[0][0], shapes[0][1], shapes[1][1], shapes[0][2])
        if (
            shapes[1] != (key.batch, key.centroids, key.features)
            or shapes[2] != (key.batch, key.centroids)
            or shapes[3] != (key.batch, key.tokens)
        ):
            self._reject("dispatcher tensor shapes are inconsistent")
        expected_dtypes = ("torch.bfloat16", "torch.bfloat16", "torch.float32", "torch.int32")
        if tuple(str(value.dtype) for value in arguments) != expected_dtypes:
            self._reject("dispatcher tensor dtypes differ")
        if any(not bool(value.is_cuda) or not value.is_contiguous() for value in arguments):
            self._reject("dispatcher tensors must be contiguous CUDA tensors")
        devices = {str(value.device) for value in arguments}
        pointers = [value.data_ptr() for value in arguments]
        if len(devices) != 1 or any(
            not isinstance(pointer, int) or isinstance(pointer, bool) or pointer <= 0
            for pointer in pointers
        ) or len(set(pointers)) != len(pointers):
            self._reject("dispatcher tensor device or pointer custody differs")
        entry = self._by_key.get(key)
        if entry is None:
            self._reject("dispatcher shape is unsupported")
        before = self.kernel_calls
        self._launcher.launch(entry.candidate, arguments)
        self.kernel_calls += 1
        self.selections[entry.case_id] += 1
        return DispatchReceipt(
            entry.case_id,
            entry.candidate.candidate_sha256,
            self.kernel_calls - before,
            0,
            0,
        )


@dataclass(frozen=True)
class PortfolioCaseObservation:
    """Raw declared observations for one exact case across both routes."""

    direct_preflight_correct: bool
    dispatcher_preflight_correct: bool
    postflight_correct: bool
    kernel_cohorts_ms: tuple[tuple[float, ...], ...]
    dispatcher_cohorts_ms: tuple[tuple[float, ...], ...]
    correctness_receipts: Mapping[str, Mapping[str, object]]
    candidate_record_sha256: str
    cubin_sha256: str
    launch_spec_sha256: str
    module_admission: Mapping[str, object]

    @property
    def document(self) -> Mapping[str, object]:
        return {
            "direct_preflight_correct": self.direct_preflight_correct,
            "dispatcher_preflight_correct": self.dispatcher_preflight_correct,
            "postflight_correct": self.postflight_correct,
            "kernel_cohorts_ms": [list(cohort) for cohort in self.kernel_cohorts_ms],
            "dispatcher_cohorts_ms": [
                list(cohort) for cohort in self.dispatcher_cohorts_ms
            ],
            "correctness_receipts": {
                route: dict(receipt)
                for route, receipt in sorted(self.correctness_receipts.items())
            },
            "candidate_record_sha256": self.candidate_record_sha256,
            "cubin_sha256": self.cubin_sha256,
            "launch_spec_sha256": self.launch_spec_sha256,
            "module_admission": dict(self.module_admission),
        }

    @classmethod
    def from_document(cls, value: object) -> "PortfolioCaseObservation":
        if not isinstance(value, Mapping) or set(value) != {
            "direct_preflight_correct",
            "dispatcher_preflight_correct",
            "postflight_correct",
            "kernel_cohorts_ms",
            "dispatcher_cohorts_ms",
            "correctness_receipts",
            "candidate_record_sha256",
            "cubin_sha256",
            "launch_spec_sha256",
            "module_admission",
        }:
            raise ValueError("portfolio case observation fields differ")

        def cohorts(field: str) -> tuple[tuple[float, ...], ...]:
            raw = value[field]
            if not isinstance(raw, list) or any(not isinstance(item, list) for item in raw):
                raise ValueError("portfolio raw cohorts differ")
            return tuple(tuple(float(sample) for sample in item) for item in raw)

        receipts = value["correctness_receipts"]
        module_admission = value["module_admission"]
        if not isinstance(receipts, Mapping) or any(
            not isinstance(receipt, Mapping) for receipt in receipts.values()
        ) or not isinstance(module_admission, Mapping):
            raise ValueError("portfolio correctness receipts differ")
        return cls(
            direct_preflight_correct=value["direct_preflight_correct"] is True,
            dispatcher_preflight_correct=value["dispatcher_preflight_correct"] is True,
            postflight_correct=value["postflight_correct"] is True,
            kernel_cohorts_ms=cohorts("kernel_cohorts_ms"),
            dispatcher_cohorts_ms=cohorts("dispatcher_cohorts_ms"),
            correctness_receipts={
                str(route): dict(receipt) for route, receipt in receipts.items()
            },
            candidate_record_sha256=str(value["candidate_record_sha256"]),
            cubin_sha256=str(value["cubin_sha256"]),
            launch_spec_sha256=str(value["launch_spec_sha256"]),
            module_admission=dict(module_admission),
        )


@dataclass(frozen=True)
class PortfolioEvaluationReceipt:
    """Orthogonal correctness, measurement quality, routing, and scope facts."""

    portfolio_sha256: str
    evaluation_protocol_sha256: str
    observations: Mapping[str, PortfolioCaseObservation]
    correctness_by_case: Mapping[str, bool]
    kernel_measurement_quality_by_case: Mapping[str, str]
    dispatcher_measurement_quality_by_case: Mapping[str, str]
    pooled_medians_ms: Mapping[str, Mapping[str, float]]
    unsupported_rejected_before_launch: bool
    route_counts: Mapping[str, object]
    unsupported_kernel_call_delta: int

    @property
    def document(self) -> Mapping[str, object]:
        return {
            "portfolio_sha256": self.portfolio_sha256,
            "evaluation_protocol_sha256": self.evaluation_protocol_sha256,
            "observations": {
                case_id: observation.document
                for case_id, observation in sorted(self.observations.items())
            },
            "correctness_by_case": dict(self.correctness_by_case),
            "kernel_measurement_quality_by_case": dict(
                self.kernel_measurement_quality_by_case
            ),
            "dispatcher_measurement_quality_by_case": dict(
                self.dispatcher_measurement_quality_by_case
            ),
            "pooled_medians_ms": {
                case_id: dict(values) for case_id, values in self.pooled_medians_ms.items()
            },
            "unsupported_rejected_before_launch": self.unsupported_rejected_before_launch,
            "route_counts": dict(self.route_counts),
            "unsupported_kernel_call_delta": self.unsupported_kernel_call_delta,
        }

    @property
    def canonical_sha256(self) -> str:
        return sha256(_canonical_json_bytes(self.document)).hexdigest()

    @property
    def correctness_supported(self) -> bool:
        return all(self.correctness_by_case.values())

    @property
    def stable_kernel_performance_supported(self) -> bool:
        return self.correctness_supported and all(
            value == "stable" for value in self.kernel_measurement_quality_by_case.values()
        )

    @property
    def stable_dispatcher_performance_supported(self) -> bool:
        return self.correctness_supported and all(
            value == "stable" for value in self.dispatcher_measurement_quality_by_case.values()
        )


def _cohort_cv(samples: Sequence[float]) -> float:
    if not samples or any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0
        for value in samples
    ):
        raise ValueError("portfolio timing samples differ")
    mean = sum(float(value) for value in samples) / len(samples)
    variance = sum((float(value) - mean) ** 2 for value in samples) / len(samples)
    return math.sqrt(variance) / mean


def evaluate_portfolio_observations(
    artifact: PortfolioArtifact,
    observations: Mapping[str, PortfolioCaseObservation],
    *,
    evaluation_protocol_sha256: str,
    route_counts: Mapping[str, object],
    unsupported_kernel_call_delta: int,
    cohorts_per_boundary: int = 5,
    samples_per_cohort: int = 25,
    maximum_cv: float = 0.05,
) -> PortfolioEvaluationReceipt:
    """Apply the frozen r45 assay without combining correctness and timing status."""

    case_ids = tuple(entry.case_id for entry in artifact.entries)
    if set(observations) != set(case_ids):
        raise ValueError("portfolio observation case coverage differs")
    expected_route_counts = {
        "automatic_retries": 0,
        "compiler_invocations": 3,
        "module_loads": 3,
        "module_unloads": 3,
        "direct_preflight_calls": 3,
        "dispatcher_preflight_calls": 3,
        "cupti_candidate_calls": 540,
        "host_dispatch_calls": 450,
        "l2_flush_calls": 450,
        "dispatcher_postflight_calls": 3,
        "unsupported_probes": 1,
        "candidate_kernel_calls": 999,
        "dispatcher_kernel_calls": 456,
        "fallback_calls": 0,
        "selections": {case_id: 152 for case_id in case_ids},
    }
    if (
        _DIGEST.fullmatch(evaluation_protocol_sha256) is None
        or
        cohorts_per_boundary != 5
        or samples_per_cohort != 25
        or maximum_cv != 0.05
        or unsupported_kernel_call_delta != 0
        or route_counts != expected_route_counts
    ):
        raise ValueError("portfolio frozen assay or route accounting differs")
    correctness: dict[str, bool] = {}
    kernel_quality: dict[str, str] = {}
    dispatcher_quality: dict[str, str] = {}
    medians: dict[str, Mapping[str, float]] = {}
    for case_id in case_ids:
        item = observations[case_id]
        entry = next(entry for entry in artifact.entries if entry.case_id == case_id)
        expected_admission = {
            "candidate_record_sha256": entry.candidate.canonical_sha256,
            "cubin_sha256": entry.candidate.artifact_roles.get("cubin"),
            "launch_spec_sha256": entry.candidate.launch_spec_sha256,
            "module_loaded": True,
        }
        if (
            item.candidate_record_sha256 != entry.candidate.canonical_sha256
            or item.cubin_sha256 != entry.candidate.artifact_roles.get("cubin")
            or item.launch_spec_sha256 != entry.candidate.launch_spec_sha256
            or any(item.module_admission.get(key) != value for key, value in expected_admission.items())
            or not isinstance(item.module_admission.get("gpu_uuid"), str)
            or not isinstance(item.module_admission.get("broker_job_id"), str)
        ):
            raise ValueError("portfolio candidate/module lineage differs")
        expected_receipts = {
            "direct_preflight": item.direct_preflight_correct,
            "dispatcher_preflight": item.dispatcher_preflight_correct,
            "postflight": item.postflight_correct,
        }
        if set(item.correctness_receipts) != set(expected_receipts) or any(
            receipt.get("passed") is not expected_receipts[route]
            for route, receipt in item.correctness_receipts.items()
        ):
            raise ValueError("portfolio correctness receipt binding differs")
        for cohorts in (item.kernel_cohorts_ms, item.dispatcher_cohorts_ms):
            if len(cohorts) != cohorts_per_boundary or any(
                len(cohort) != samples_per_cohort for cohort in cohorts
            ):
                raise ValueError("portfolio cohort grid differs")
        correctness[case_id] = (
            item.direct_preflight_correct
            and item.dispatcher_preflight_correct
            and item.postflight_correct
        )
        kernel_stable = all(_cohort_cv(cohort) <= maximum_cv for cohort in item.kernel_cohorts_ms)
        dispatcher_stable = all(
            _cohort_cv(cohort) <= maximum_cv for cohort in item.dispatcher_cohorts_ms
        )
        kernel_quality[case_id] = "stable" if kernel_stable else "unstable"
        dispatcher_quality[case_id] = "stable" if dispatcher_stable else "unstable"
        medians[case_id] = MappingProxyType(
            {
                "kernel": median(
                    value for cohort in item.kernel_cohorts_ms for value in cohort
                ),
                "dispatcher": median(
                    value for cohort in item.dispatcher_cohorts_ms for value in cohort
                ),
            }
        )
    return PortfolioEvaluationReceipt(
        artifact.canonical_sha256,
        evaluation_protocol_sha256,
        MappingProxyType(dict(observations)),
        MappingProxyType(correctness),
        MappingProxyType(kernel_quality),
        MappingProxyType(dispatcher_quality),
        MappingProxyType(medians),
        True,
        MappingProxyType(dict(route_counts)),
        unsupported_kernel_call_delta,
    )


def replay_portfolio_receipt(
    value: object,
    artifact: PortfolioArtifact,
    *,
    expected_protocol_sha256: str,
) -> PortfolioEvaluationReceipt:
    """Recompute every portfolio projection from retained raw cohorts and receipts."""

    if not isinstance(value, Mapping) or set(value) != {
        "portfolio_sha256",
        "evaluation_protocol_sha256",
        "observations",
        "correctness_by_case",
        "kernel_measurement_quality_by_case",
        "dispatcher_measurement_quality_by_case",
        "pooled_medians_ms",
        "unsupported_rejected_before_launch",
        "route_counts",
        "unsupported_kernel_call_delta",
    }:
        raise ValueError("portfolio receipt fields differ")
    raw_observations = value["observations"]
    if not isinstance(raw_observations, Mapping):
        raise ValueError("portfolio receipt observations differ")
    observations = {
        str(case_id): PortfolioCaseObservation.from_document(observation)
        for case_id, observation in raw_observations.items()
    }
    route_counts = value["route_counts"]
    if not isinstance(route_counts, Mapping):
        raise ValueError("portfolio receipt route counts differ")
    if value["evaluation_protocol_sha256"] != expected_protocol_sha256:
        raise ValueError("portfolio receipt Evaluation Protocol differs")
    observed = evaluate_portfolio_observations(
        artifact,
        observations,
        evaluation_protocol_sha256=expected_protocol_sha256,
        route_counts=route_counts,
        unsupported_kernel_call_delta=int(value["unsupported_kernel_call_delta"]),
    )
    if _canonical_json_bytes(observed.document) != _canonical_json_bytes(value):
        raise ValueError("portfolio receipt projections differ from raw observations")
    return observed
