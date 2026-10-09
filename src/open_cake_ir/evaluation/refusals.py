"""Evidence-backed Evaluation refusals, without a launch or correctness receipt."""
from __future__ import annotations

from dataclasses import dataclass
import json
import re

from .attempts import valid_job_mode
from .failures import failure_artifacts
from .platforms import platform_for


_COUNTERS = frozenset({'compiler_invocations', 'module_loads', 'preflight_calls',
                       'kernel_calls', 'timing_samples', 'fallback_calls'})


def is_local_allocation_refusal(result, stdout: bytes, stderr: bytes, *, target) -> bool:
    """A local broker can issue an attempt id without admitting a device job."""
    try:
        paths = failure_artifacts(result)
        prefix = platform_for(target).local_job_prefix
    except (ValueError, TypeError):
        return False
    if (paths or not prefix or result.get('schema_version') != 1
            or result.get('admitted') is not False or result.get('receipt') is not None
            or result.get('failure_class') != 'admission'
            or result.get('error') != f'{prefix}_broker_busy'
            or result.get('mode') != 'local_serialized'
            or not valid_job_mode(result.get('job_id'), result.get('mode'))
            or not result['job_id'].startswith(prefix + '-') or stdout or stderr):
        return False
    counters = result.get('counters')
    return (isinstance(counters, dict) and set(counters) == _COUNTERS
            and all(type(value) is int and value == 0 for value in counters.values()))


@dataclass(frozen=True)
class EvaluationRefusal:
    """One unmeasured outcome. It cannot qualify a candidate or carry latency."""

    candidate_sha256: str
    case_id: str
    purpose: str
    job_id: str
    disposition: str
    diagnostic: str

    def __post_init__(self):
        if (re.fullmatch(r'[0-9a-f]{64}', self.candidate_sha256) is None
                or not self.case_id or self.purpose not in {'search', 'attribution'}
                or not valid_job_mode(self.job_id, 'local_serialized')
                or self.disposition not in {'allocation_refused', 'candidate_resource_rejected'}
                or not self.diagnostic
                or (self.disposition == 'candidate_resource_rejected' and self.purpose != 'search')):
            raise ValueError('Evaluation refusal fields differ')

    @property
    def document(self):
        return {'schema_version': 1, 'candidate_sha256': self.candidate_sha256,
                'case_id': self.case_id, 'purpose': self.purpose, 'job_id': self.job_id,
                'disposition': self.disposition, 'diagnostic': self.diagnostic}

    @property
    def correctness_passed(self):
        return None

    @property
    def candidate_disposition(self):
        return self.disposition

    @property
    def measurement_quality(self):
        return 'not_measured'

    @property
    def timing(self):
        return None

    @property
    def attribution_feedback(self):
        return {'kind': 'evaluation_refusal_v1', **self.document}


def refusal_from_attempts(attempts, *, candidate, case_id, purpose):
    """Derive a refusal from one complete retained attempt; never retry it here."""
    if len(attempts) != 1 or purpose not in {'search', 'attribution'}:
        return None
    item = attempts[0]
    if item.receipt is not None:
        return None
    try:
        result = json.loads(item.artifact_payloads['evaluator_result'])
        if (any(result.get(name) != getattr(item, name) for name in ('job_id', 'mode', 'admitted', 'error'))
                or any(result.get('counters', {}).get(name) != getattr(item, name) for name in _COUNTERS)):
            return None
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    return refusal_from_artifacts(item.artifact_payloads, candidate=candidate,
        case_id=case_id, purpose=purpose, job_id=item.job_id)


def refusal_from_artifacts(raw, *, candidate, case_id, purpose, job_id):
    """Shared semantic classification after live or replay custody validation."""
    if purpose not in {'search', 'attribution'}:
        return None
    try:
        request = json.loads(raw['evaluator_request'])
        result = json.loads(raw['evaluator_result'])
        if (request.get('candidate_sha256') != candidate.candidate_sha256
                or request.get('target') != candidate.target
                or request.get('purpose') != purpose or request.get('case_id') != case_id
                or result.get('job_id') != job_id):
            return None
        if is_local_allocation_refusal(result, raw['stdout'], raw['stderr'], target=candidate.target):
            return EvaluationRefusal(candidate.candidate_sha256, case_id, purpose, job_id,
                'allocation_refused', 'The local device lock was not acquired. No kernel was launched; '
                'this attempt has no correctness or performance result and was not retried.')
        from .metax_failures import is_local_launch_resource_failure
        diagnostic = json.loads(raw['failure_launch_resource'])
        if (purpose == 'search' and platform_for(candidate.target).local_job_prefix == 'maca'
                and diagnostic.get('candidate_sha256') == candidate.candidate_sha256
                and diagnostic.get('target') == candidate.target
                and diagnostic.get('input_case_id') == case_id
                and is_local_launch_resource_failure(result, diagnostic,
                    raw['stdout'].decode(), raw['stderr'].decode())):
            resources = diagnostic['resources']
            return EvaluationRefusal(candidate.candidate_sha256, case_id, purpose, job_id,
                'candidate_resource_rejected', 'The candidate failed MACA preflight dispatch with '
                f'mcErrorMemoryValueTooLarge; local_bytes={resources["local_bytes"]}. '
                'No target call completed and teardown completed. Choose another candidate mapping.')
    except (ValueError, KeyError, TypeError, UnicodeError):
        return None
    return None
