"""One logical Evaluation attempt with bounded infrastructure admission recovery."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Mapping

from .core import EvaluationReceipt, LaunchableCandidate

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_JOB_ID = re.compile(r"^gpuq-[0-9a-f]{12}$")


@dataclass(frozen=True)
class BrokerAttempt:
    """One broker job and its exact pre/post-admission work accounting."""

    job_id: str
    mode: str
    candidate_sha256: str
    manifest_sha256: str
    policy_sha256: str
    evaluator_arguments_sha256: str
    admitted: bool
    error: str | None
    compiler_invocations: int
    module_loads: int
    preflight_calls: int
    kernel_calls: int
    timing_samples: int
    fallback_calls: int
    receipt: EvaluationReceipt | None
    artifact_payloads: Mapping[str, bytes] = field(default_factory=dict)

    def __post_init__(self) -> None:
        digests = (
            self.candidate_sha256,
            self.manifest_sha256,
            self.policy_sha256,
            self.evaluator_arguments_sha256,
        )
        counters = (
            self.compiler_invocations,
            self.module_loads,
            self.preflight_calls,
            self.kernel_calls,
            self.timing_samples,
            self.fallback_calls,
        )
        if (
            _JOB_ID.fullmatch(self.job_id) is None
            or self.mode != "exclusive"
            or any(_DIGEST.fullmatch(value) is None for value in digests)
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counters)
        ):
            raise ValueError("broker attempt fields differ")
        if self.receipt is not None and (
            not self.admitted
            or self.error is not None
            or self.receipt.candidate_sha256 != self.candidate_sha256
        ):
            raise ValueError("broker attempt receipt contradicts admission or candidate")
        if self.artifact_payloads and (
            not {"broker_record", "stdout", "stderr"} <= set(self.artifact_payloads)
            or any(not isinstance(payload, bytes) for payload in self.artifact_payloads.values())
        ):
            raise ValueError("broker attempt artifact custody differs")

    @property
    def authority(self) -> tuple[str, str, str, str]:
        return (
            self.candidate_sha256,
            self.manifest_sha256,
            self.policy_sha256,
            self.evaluator_arguments_sha256,
        )


def is_resubmittable_admission_failure(attempt: BrokerAttempt) -> bool:
    """Recognize only the exact zero-work exclusive-card race seen in r41."""

    return (
        not attempt.admitted
        and attempt.error == "gpu_admission_differs"
        and attempt.receipt is None
        and (
            attempt.compiler_invocations,
            attempt.module_loads,
            attempt.preflight_calls,
            attempt.kernel_calls,
            attempt.timing_samples,
            attempt.fallback_calls,
        )
        == (0, 0, 0, 0, 0, 0)
    )


@dataclass(frozen=True)
class LogicalEvaluationAttempt:
    """One candidate evaluation, retaining every infrastructure job."""

    candidate_sha256: str
    attempts: tuple[BrokerAttempt, ...]
    final_receipt: EvaluationReceipt | None


def evaluate_with_admission_recovery(
    candidate: LaunchableCandidate,
    submit: Callable[[int], BrokerAttempt],
) -> LogicalEvaluationAttempt:
    """Submit once, with at most one same-candidate zero-work admission resubmission."""

    first = submit(1)
    if first.candidate_sha256 != candidate.candidate_sha256:
        raise ValueError("broker attempt candidate identity differs")
    attempts = [first]
    if is_resubmittable_admission_failure(first):
        second = submit(2)
        if second.authority != first.authority:
            raise ValueError("admission resubmission authority differs")
        attempts.append(second)
    final = attempts[-1]
    if is_resubmittable_admission_failure(final):
        return LogicalEvaluationAttempt(candidate.candidate_sha256, tuple(attempts), None)
    final_receipt = (
        final.receipt if final.admitted and final.error is None else None
    )
    return LogicalEvaluationAttempt(candidate.candidate_sha256, tuple(attempts), final_receipt)
