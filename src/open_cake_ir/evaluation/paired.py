"""Fixed-baseline paired assay authority and raw evidence validation.

The Study owns the assay values; neither a target calibration nor the directional
classification decides whether a correct, stable Candidate is a valid outcome.
"""
from __future__ import annotations

import json
import math
import re
from hashlib import sha256
from typing import Mapping

from .timing import PairedTimingProtocol, derive_paired_timing

PAIRED_KIND = 'fixed_baseline_paired_cupti_v1'


def _plain(value):
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def _canonical(value):
    return json.dumps(_plain(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def paired_protocol(evaluation: Mapping[str, object]) -> PairedTimingProtocol | None:
    """An explicit policy enables the successor; absent policy retains old replay."""
    if not isinstance(evaluation, Mapping):
        raise ValueError('paired evaluation policy must be an object')
    value = evaluation.get('paired_timing')
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {
        'kind', 'arms', 'pair_order', 'samples_per_cohort', 'route_calls_per_cohort',
        'maximum_cv', 'materiality_ratio', 'required_pair_wins',
    } or value.get('kind') != PAIRED_KIND or value.get('arms') != ['candidate', 'baseline']:
        raise ValueError('fixed-baseline paired policy fields or roles differ')
    if (evaluation.get('search_evaluation') != 'correctness_then_paired_cupti'
        or evaluation.get('confirmatory_evaluation') != 'fresh_fixed_candidate_correctness_then_paired_cupti'):
        raise ValueError('paired policy requires search and fresh confirmation')
    order = value['pair_order']
    if (not isinstance(order, list) or any(not isinstance(p, list) or len(p) != 2 or any(type(role) is not str for role in p) for p in order)
        or any(type(value[k]) is not int for k in ('samples_per_cohort', 'route_calls_per_cohort', 'required_pair_wins'))
        or any(type(value[k]) not in (int, float) for k in ('maximum_cv', 'materiality_ratio'))):
        raise ValueError('fixed-baseline paired policy value types differ')
    protocol = PairedTimingProtocol(tuple(value['arms']), tuple(tuple(p) for p in order),
        value['samples_per_cohort'], value['route_calls_per_cohort'], value['maximum_cv'],
        value['materiality_ratio'], value['required_pair_wins'])
    # This is the retained helper's existing invocation contract, not another engine.
    if protocol.route_calls_per_cohort != 6 + 11 + protocol.samples_per_cohort:
        raise ValueError('paired policy differs from retained CUPTI callback contract')
    return protocol


def candidate_identity(candidate):
    return {'candidate_sha256': candidate.candidate_sha256, 'target': candidate.target,
        'entry_point': candidate.entry_point, 'artifact_roles': dict(candidate.artifact_roles),
        'launch_spec_sha256': candidate.launch_spec_sha256,
        'candidate_record_sha256': candidate.canonical_sha256}


def candidate_from_identity(value, payloads=None):
    from .core import LaunchableCandidate
    if not isinstance(value, Mapping) or set(value) != {
        'candidate_sha256', 'target', 'entry_point', 'artifact_roles',
        'launch_spec_sha256', 'candidate_record_sha256',
    }:
        raise ValueError('paired participant identity fields differ')
    candidate = LaunchableCandidate(**{k: v for k, v in value.items() if k != 'candidate_record_sha256'},
                                   artifact_payloads=payloads or {})
    if candidate.canonical_sha256 != value['candidate_record_sha256']:
        raise ValueError('paired participant record identity differs')
    return candidate


def validate_pair_candidates(candidate, baseline, workload, case_id):
    from .core import TensorLaunchManifest
    manifests = {}
    for role, item in [('candidate', candidate), ('baseline', baseline)]:
        if not item.artifact_payloads or 'launch_manifest' not in item.artifact_payloads:
            raise ValueError('paired participant has no sealed launch manifest')
        document = json.loads(item.artifact_payloads['launch_manifest'])
        if not isinstance(document, Mapping) or document.get('abi') != 'workload_tensors_v1':
            raise ValueError('paired policy requires the explicit Workload tensor ABI')
        manifest = TensorLaunchManifest.from_dict(document)
        manifest.check_workload(workload, case_id)
        if (item.target != manifest.target or item.entry_point != manifest.kernel_name
            or item.launch_spec_sha256 != manifest.canonical_sha256):
            raise ValueError('paired participant and launch manifest differ')
        manifests[role] = manifest
    return manifests


def paired_summary(raw):
    """Recompute the directional observation through the sole pure derivation owner."""
    protocol = paired_protocol(raw['evaluation_protocol'])
    if protocol is None:
        raise ValueError('paired raw evidence has no explicit policy')
    measurements = raw['measurements']
    if not isinstance(measurements, list):
        raise ValueError('paired measurements must be a list')
    for row in measurements:
        if not isinstance(row, Mapping) or type(row.get('pair_index')) is not int:
            raise ValueError('paired index must be an integer')
        arms = row.get('arms')
        if not isinstance(arms, Mapping) or any(not isinstance(record, Mapping)
            or type(record.get('position')) is not int for record in arms.values()):
            raise ValueError('paired position must be an integer')
    observation = derive_paired_timing(measurements, protocol)
    return {'kind': PAIRED_KIND,
        'measurement_quality_passed': observation.measurement_quality_passed,
        'pooled_median_ms': observation.pooled_medians_ms['candidate'],
        'pooled_medians_ms': dict(observation.pooled_medians_ms),
        'pooled_sample_counts': dict(observation.pooled_sample_counts),
        'pair_wins': dict(observation.pair_wins), 'tied_pairs': observation.tied_pairs,
        'speedup': observation.speedup, 'classification': observation.classification}


def validate_paired_receipt(receipt, raw, correctness, launch, *, evaluation=None, baseline=None, candidate=None):
    """Bind both identities, actual order, every oracle check and one job to the lock."""
    if not isinstance(raw, Mapping) or raw.get('kind') != PAIRED_KIND:
        raise ValueError('paired policy rejects single-candidate timing evidence')
    protocol = paired_protocol(raw.get('evaluation_protocol', {}))
    if protocol is None or sha256(_canonical(raw['evaluation_protocol'])).hexdigest() != receipt.evaluation_protocol_sha256:
        raise ValueError('paired receipt evaluation policy differs')
    if evaluation is not None and _canonical(raw['evaluation_protocol']) != _canonical(evaluation):
        raise ValueError('paired receipt differs from Campaign evaluation policy')
    if (raw.get('workload_sha256') != receipt.workload_sha256 or raw.get('case_id') != receipt.case_id
        or raw.get('purpose') != receipt.purpose):
        raise ValueError('paired receipt Workload, case or purpose differs')
    participants = raw.get('participants')
    if not isinstance(participants, Mapping) or set(participants) != set(protocol.arms):
        raise ValueError('paired receipt partner is missing')
    for role in protocol.arms:
        candidate_from_identity(participants[role])
    if (participants['candidate']['candidate_sha256'] != receipt.candidate_sha256
        or participants['candidate']['target'] != participants['baseline']['target']
        or launch.get('participants') != participants
        or not isinstance(raw.get('gpu_uuid'), str) or not raw['gpu_uuid']
        or not isinstance(raw.get('job_id'), str) or re.fullmatch(r'gpuq-[0-9a-f]{12}', raw['job_id']) is None
        or raw['job_id'] == 'gpuq-000000000000'
        or (launch.get('job_id'), launch.get('gpu_uuid')) != (raw['job_id'], raw['gpu_uuid'])):
        raise ValueError('paired receipt participant or allocation identity differs')
    if baseline is not None and participants['baseline'] != baseline:
        raise ValueError('paired receipt fixed baseline differs from Campaign Lock')
    if candidate is not None and participants['candidate'] != candidate_identity(candidate):
        raise ValueError('paired receipt candidate record differs from sealed Candidate')
    checks = correctness.get('participants')
    if not isinstance(checks, Mapping) or set(checks) != set(protocol.arms):
        raise ValueError('paired correctness partner is missing')
    measured = receipt.timing is not None
    passed = True
    combined = {'output_mismatches': 0, 'max_abs_error': 0.0, 'inputs_unchanged': True}
    def oracle_check(check, metrics):
        if (not isinstance(metrics, Mapping)
            or type(metrics.get('output_mismatches')) is not int or metrics['output_mismatches'] < 0
            or type(metrics.get('inputs_unchanged')) is not bool
            or type(metrics.get('max_abs_error')) not in (int, float)
            or not math.isfinite(metrics['max_abs_error']) or metrics['max_abs_error'] < 0
            or check['passed'] != (metrics['output_mismatches'] == 0 and metrics['inputs_unchanged'])):
            raise ValueError('paired oracle metrics and disposition differ')
        combined['output_mismatches'] += metrics['output_mismatches']
        combined['max_abs_error'] = max(combined['max_abs_error'], metrics['max_abs_error'])
        combined['inputs_unchanged'] = combined['inputs_unchanged'] and metrics['inputs_unchanged']
    for role in protocol.arms:
        value = checks[role]
        if not isinstance(value, Mapping) or set(value) != {'preflight', 'postflight', 'timed_output_checks'}:
            raise ValueError('paired correctness coverage differs')
        for phase in ('preflight', 'postflight') if measured else ('preflight',):
            check = value[phase]
            if not isinstance(check, Mapping) or type(check.get('passed')) is not bool:
                raise ValueError('paired oracle check is missing')
            oracle_check(check, check.get('metrics'))
            passed = passed and check['passed']
        timed = value['timed_output_checks']
        if not isinstance(timed, list) or len(timed) != (len(protocol.pair_order) if measured else 0):
            raise ValueError('paired fresh-output cohort coverage differs')
        for check in timed:
            if (not isinstance(check, Mapping) or type(check.get('checked_launches')) is not int
                or check['checked_launches'] != protocol.route_calls_per_cohort
                or type(check.get('passed')) is not bool):
                raise ValueError('paired fresh-output callback coverage differs')
            oracle_check(check, check)
            passed = passed and check['passed']
    if combined != dict(receipt.correctness):
        raise ValueError('paired correctness summary differs from all oracle checks')
    if passed != receipt.correctness_passed:
        raise ValueError('paired correctness disposition differs from both oracle outputs')
    if measured:
        summary = paired_summary(raw)
        if _canonical(summary) != _canonical(receipt.timing):
            raise ValueError('paired timing summary differs from raw samples')
        for i, measurement in enumerate(raw['measurements']):
            for role in protocol.arms:
                record = measurement['arms'][role]
                if record.get('candidate_record_sha256') != participants[role]['candidate_record_sha256']:
                    raise ValueError('paired cohort participant identity differs')
                if record.get('output_check') != checks[role]['timed_output_checks'][i]:
                    raise ValueError('paired cohort oracle check differs')
    elif (raw.get('measurements') != [] or receipt.correctness_passed
          or raw.get('not_measured') != 'correctness_rejected'):
        raise ValueError('paired missing timing is not a correctness rejection')


def validate_receipt_policy(receipt, evaluation, baseline, candidate=None):
    if receipt.purpose == 'attribution' or paired_protocol(evaluation) is None:
        return
    if baseline is None or not receipt.artifact_payloads:
        raise ValueError('paired policy requires the fixed baseline and raw evidence')
    validate_paired_receipt(receipt,
        json.loads(receipt.artifact_payloads['timing_samples']),
        json.loads(receipt.artifact_payloads['correctness_output']),
        json.loads(receipt.artifact_payloads['launch_receipt']),
        evaluation=evaluation, baseline=baseline, candidate=candidate)


def validate_paired_broker(receipt, job_id, counters):
    """Tie a paired receipt to its actual broker allocation and complete work count."""
    if receipt.purpose == 'attribution' or not receipt.artifact_payloads:
        return
    raw = json.loads(receipt.artifact_payloads['timing_samples'])
    if not isinstance(raw, Mapping) or raw.get('kind') != PAIRED_KIND:
        return
    protocol = paired_protocol(raw['evaluation_protocol'])
    cohorts = len(protocol.pair_order) * 2 if receipt.timing is not None else 0
    correctness_calls = 4 if cohorts else 2
    expected = {'compiler_invocations': 0, 'module_loads': 2, 'preflight_calls': 2,
        'kernel_calls': correctness_calls + cohorts * protocol.route_calls_per_cohort,
        'timing_samples': cohorts * protocol.samples_per_cohort, 'fallback_calls': 0}
    launch = json.loads(receipt.artifact_payloads['launch_receipt'])
    if (raw['job_id'] != job_id or job_id == 'gpuq-000000000000'
        or dict(counters) != expected or launch.get('correctness_launches') != correctness_calls):
        raise ValueError('paired receipt broker job or work counters differ')
