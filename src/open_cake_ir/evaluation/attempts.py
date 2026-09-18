"""One logical Evaluation attempt with bounded infrastructure admission recovery."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Mapping

from .core import EvaluationReceipt, LaunchableCandidate
from .platforms import PLATFORMS

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
# A job id names the allocator that issued it, not the API the candidate uses. `gpuq` is
# the cluster allocator; every local device family is reached through the local broker
# under its own prefix, because a DCU job recorded as a Metal one would be the same
# mislabelling as a DCU latency recorded as CUPTI. Each platform row declares which
# prefixes its allocators issue, so the grammar here is a view over those rows.
_LOCAL_PREFIXES = tuple(
    row.local_job_prefix for row in PLATFORMS.values() if row.local_job_prefix is not None)
_JOB_MODES = {
    **{row.exclusive_job_prefix: "exclusive" for row in PLATFORMS.values()
       if row.exclusive_job_prefix is not None},
    **{prefix: "local_serialized" for prefix in _LOCAL_PREFIXES},
}
_JOB_ID = re.compile(rf"^({'|'.join(_JOB_MODES)})-[0-9a-f]{{12}}$")


def job_mode(job_id: object) -> str | None:
    """The mode the allocator that issued this job id runs under, or None for no allocator."""
    match = _JOB_ID.fullmatch(job_id) if isinstance(job_id, str) else None
    return _JOB_MODES[match[1]] if match else None


def valid_job_mode(job_id: str, mode: str) -> bool:
    match = _JOB_ID.fullmatch(job_id) if isinstance(job_id, str) else None
    return bool(match and mode == _JOB_MODES[match[1]]
                and (match[1] not in _LOCAL_PREFIXES
                     or job_id != f"{match[1]}-000000000000"))


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
            not valid_job_mode(self.job_id, self.mode)
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
        attempt.mode == "exclusive"
        and not attempt.admitted
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
