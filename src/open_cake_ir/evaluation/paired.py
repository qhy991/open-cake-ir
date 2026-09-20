"""Fixed-baseline paired assay authority and raw evidence validation.

The Study owns the assay values; neither a target calibration nor the directional
classification decides whether a correct, stable Candidate is a valid outcome.
"""
from __future__ import annotations

from open_cake_ir.serialization import canonical_json_bytes

import json
import math
import re
from hashlib import sha256
from typing import Mapping

from open_cake_ir.compiler.target import CodeObject

from .timing import PairedTimingProtocol, derive_paired_timing
from .artifacts import executable_role
from .platforms import PLATFORMS, platform_for, platform_for_paired_kind

# The declared assay names, kept as constants for the readers that spell them. Which
# measurement source each names, how many times its cohort calls the route and which
# object it times are the execution platform row's facts; everything below is a view
# over those rows, so a kind no row declares is refused by name rather than measured as
# whichever source an `else` reached. CUPTI used to be whatever was not Metal.
#
# The AMDGCN kind was withheld until something produced a measurement, which is the rule
# the rows keep: a policy kind is a name for a measurement, and minting one first would
# have labelled evidence with a profiler that never produced it. `hip_dispatch` is
# roctracer's per-dispatch device time, read through the profiler the admitted torch
# already carries, measured on a BW1101 against rocprofv2's own reading of the same
# kernel and shape (3.071 us against 3.36 us) and shown to attribute the harness's own
# `hipModuleLaunchKernel` dispatches. See `evaluation/hip_benchmark.py` for what the
# interval includes and what resets the device.
PAIRED_KIND = 'fixed_baseline_paired_cupti_v1'
PAIRED_METAL_KIND = 'fixed_baseline_paired_metal_v1'
# The successor declares how many dispatches each timed command buffer encodes, so a
# kernel shorter than the fixed command overhead is not measured through it. v1 keeps
# its exact single-dispatch meaning; frozen Studies replay unchanged.
PAIRED_METAL_BATCHED_KIND = 'fixed_baseline_paired_metal_v2'
# One dispatch of one AMDGCN kernel, timed by roctracer. Named `v1` for the same reason
# the others are: what the interval includes and what resets the device are part of the
# policy, and a successor states its own.
PAIRED_HIP_KIND = 'fixed_baseline_paired_hip_dispatch_v1'
METAL_KINDS = PLATFORMS[CodeObject.METAL_BINARY_ARCHIVE].paired_kinds
PAIRED_KINDS = frozenset().union(*(row.paired_kinds for row in PLATFORMS.values()))
# How many times a cohort calls the route, per measurement source that declares it on its
# row. CUPTI's six extra calls are its calibration callbacks; the HIP assay has none.
# Metal's is not here: its cohort is the native observer's snapshot, sized against that
# observer's payload bound, and it lives with the launcher check it must not drift from
# (`tasks/normalization/study._ROUTE_CALLS_PER_COHORT`, F-2026-09-10-002). Named so a
# reader of this table does not conclude Metal has no count.
ROUTE_CALLS_PER_COHORT = {
    row.measurement_source: row.route_calls_per_cohort
    for row in PLATFORMS.values() if row.route_calls_per_cohort is not None
}
_BASE_FIELDS = {
    'kind', 'arms', 'pair_order', 'samples_per_cohort', 'route_calls_per_cohort',
    'maximum_cv', 'materiality_ratio', 'required_pair_wins',
}


def _plain(value):
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def _canonical(value):
    return canonical_json_bytes(_plain(value))


def paired_protocol(evaluation: Mapping[str, object]) -> PairedTimingProtocol | None:
    """An explicit policy enables the successor; absent policy retains old replay."""
    if not isinstance(evaluation, Mapping):
        raise ValueError('paired evaluation policy must be an object')
    value = evaluation.get('paired_timing')
    if value is None:
        return None
    kind = value.get('kind') if isinstance(value, Mapping) else None
    successor = {'dispatches_per_sample', 'maximum_relative_iqr'}
    expected_fields = _BASE_FIELDS | (successor if kind == PAIRED_METAL_BATCHED_KIND else set())
    if (not isinstance(value, Mapping) or set(value) != expected_fields
            or kind not in PAIRED_KINDS or value.get('arms') != ['candidate', 'baseline']):
        raise ValueError('fixed-baseline paired policy fields or roles differ')
    backend = platform_for_paired_kind(kind).measurement_source
    if (evaluation.get('search_evaluation') != f'correctness_then_paired_{backend}'
        or evaluation.get('confirmatory_evaluation') != f'fresh_fixed_candidate_correctness_then_paired_{backend}'):
        raise ValueError('paired policy requires search and fresh confirmation')
    order = value['pair_order']
    if (not isinstance(order, list) or any(not isinstance(p, list) or len(p) != 2 or any(type(role) is not str for role in p) for p in order)
        or any(type(value[k]) is not int for k in ('samples_per_cohort', 'route_calls_per_cohort', 'required_pair_wins'))
        or any(type(value[k]) not in (int, float) for k in ('maximum_cv', 'materiality_ratio'))):
        raise ValueError('fixed-baseline paired policy value types differ')
    dispatches = value.get('dispatches_per_sample', 1)
    spread = value.get('maximum_relative_iqr')
    if type(dispatches) is not int or (spread is not None and type(spread) not in (int, float)):
        raise ValueError('fixed-baseline paired policy value types differ')
    protocol = PairedTimingProtocol(tuple(value['arms']), tuple(tuple(p) for p in order),
        value['samples_per_cohort'], value['route_calls_per_cohort'], value['maximum_cv'],
        value['materiality_ratio'], value['required_pair_wins'], dispatches, spread)
    # This is the retained helper's existing invocation contract, not another engine.
    if value['kind'] == PAIRED_KIND and protocol.route_calls_per_cohort != 6 + 11 + protocol.samples_per_cohort:
        raise ValueError('paired policy differs from retained CUPTI callback contract')
    # The HIP benchmark calls the route exactly once per warmup and once per sample; it
    # has no calibration callbacks of its own, which is where CUPTI's extra six go.
    if value['kind'] == PAIRED_HIP_KIND and protocol.route_calls_per_cohort != 11 + protocol.samples_per_cohort:
        raise ValueError('paired policy differs from the HIP dispatch invocation contract')
    if kind in METAL_KINDS:
        if protocol.route_calls_per_cohort <= protocol.samples_per_cohort:
            raise ValueError('Metal assay requires declared warmup calls before timestamp samples')
    if kind in METAL_KINDS or 'validation_case_ids' in evaluation:
        validation_case_ids(evaluation)
    return protocol


def validation_case_ids(evaluation: Mapping) -> tuple[str, ...]:
    """Validate an explicit frozen projection of Workload validation cases."""
    cases = evaluation.get('validation_case_ids')
    if (not isinstance(cases, list) or not cases or any(not isinstance(case, str) or not case for case in cases)
            or len(set(cases)) != len(cases) or evaluation.get('case_id') not in cases):
        raise ValueError('evaluation validation_case_ids differ')
    return tuple(cases)


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


def _manifest_spellings():
    """The Workload tensor manifest class per `abi`, from the two spellings that exist."""
    from .core import TensorLaunchManifest
    from .metal_manifest import MetalTensorLaunchManifest
    return {cls.abi: cls for cls in (TensorLaunchManifest, MetalTensorLaunchManifest)}


def validate_pair_candidates(candidate, baseline, workload, case_id):
    spellings = _manifest_spellings()
    manifests = {}
    for role, item in [('candidate', candidate), ('baseline', baseline)]:
        if not item.artifact_payloads or 'launch_manifest' not in item.artifact_payloads:
            raise ValueError('paired participant has no sealed launch manifest')
        document = json.loads(item.artifact_payloads['launch_manifest'])
        # The row for the participant's target says which spelling it seals; a manifest
        # in another spelling is not this participant's, whatever else it parses as.
        if (not isinstance(document, Mapping)
                or document.get('abi') != platform_for(item.target).launch_abi):
            raise ValueError('paired policy requires the explicit Workload tensor ABI')
        manifest = spellings[document['abi']].from_dict(document)
        manifest.check_workload(workload, case_id)
        if hasattr(manifest, 'check_complete_domain'):
            manifest.check_complete_domain()
        if getattr(manifest, 'aligned_variant', None):
            from .kernel_bundle import alignment_component
            alignment_component(item, manifest)
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
    kind = raw['evaluation_protocol']['paired_timing']['kind']
    if kind in METAL_KINDS:
        from .metal_observations import METAL_TIMER, METAL_CACHE, validate_command_samples
        if raw.get('timer') != METAL_TIMER or raw.get('cache_policy') != METAL_CACHE:
            raise ValueError('Metal timer/cache observation differs')
        for row in measurements:
            for record in row['arms'].values():
                validate_command_samples(record, route_calls=protocol.route_calls_per_cohort,
                                         sample_count=protocol.samples_per_cohort,
                                         dispatches_per_sample=protocol.dispatches_per_sample)
    observation = derive_paired_timing(measurements, protocol)
    return {'kind': kind,
        'measurement_quality_passed': observation.measurement_quality_passed,
        'pooled_median_ms': observation.pooled_medians_ms['candidate'],
        'pooled_medians_ms': dict(observation.pooled_medians_ms),
        'pooled_sample_counts': dict(observation.pooled_sample_counts),
        'pair_wins': dict(observation.pair_wins), 'tied_pairs': observation.tied_pairs,
        'speedup': observation.speedup, 'classification': observation.classification}


def admit_device_identity(raw, launch, participants) -> None:
    """Check the device and allocation identity the declared assay implies.

    One branch per declared kind, and a kind with no branch refused by name. This was
    `if Metal ... elif <everything else>`, so an AMDGCN pair went down CUDA's branch and
    was refused for not being a cubin -- true, and not the point. It is a function so it
    can be exercised directly: reaching it through `validate_paired_receipt` means
    constructing every earlier check's state, which is why no test had reached it.
    """

    broker_pair = (str(raw.get('job_id', '')).startswith('gpuq-')
                   and raw.get('broker_allocation') is not None)
    if broker_pair:
        from .gpuq import validate_allocation
        validate_allocation(raw.get('broker_allocation'),
                            target=participants['candidate']['target'], job_id=raw['job_id'])
        if (launch.get('job_id') != raw['job_id']
                or launch.get('broker_allocation') != raw['broker_allocation']):
            raise ValueError('paired broker allocation evidence differs')
    if raw['kind'] in METAL_KINDS:
        from .metal_observations import validate_host
        host = validate_host(raw.get('host'))
        prefix = PLATFORMS[CodeObject.METAL_BINARY_ARCHIVE].local_job_prefix
        mode = 'exclusive' if broker_pair else 'local_serialized'
        if ((not broker_pair and (re.fullmatch(rf'{prefix}-[0-9a-f]{{12}}', raw['job_id']) is None or raw['job_id'] == f'{prefix}-000000000000'))
                or raw.get('allocation_mode') != mode or launch.get('allocation_mode') != mode
                or raw.get('external_gpu_activity') != 'not_excluded' or launch.get('external_gpu_activity') != 'not_excluded'
                or launch.get('host') != host or raw.get('device_registry_id') != host['device_registry_id']
                or host['target'] != participants['candidate']['target']):
            raise ValueError('paired Metal device/host identity differs')
    elif raw['kind'] == PAIRED_KIND:
        # CUPTI timing evidence is taken under the exclusive cluster lease (ADR 0011,
        # 0056), so the job is the one the row's exclusive allocator issued; a
        # local-broker `cuda` job admits a correctness check and never this receipt.
        row = platform_for_paired_kind(PAIRED_KIND)
        if (executable_role(participants['candidate']['target']) != row.code_object.value
                or not isinstance(raw.get('gpu_uuid'), str) or not raw['gpu_uuid']
                or re.fullmatch(rf'{row.exclusive_job_prefix}-[0-9a-f]{{12}}', raw['job_id']) is None
                or raw['job_id'] == f'{row.exclusive_job_prefix}-000000000000'
                or launch.get('gpu_uuid') != raw['gpu_uuid']):
            raise ValueError('paired CUDA device/host identity differs')
    elif raw['kind'] == PAIRED_HIP_KIND:
        # An AMDGCN pair: the hsaco role, a local-broker job, and whatever the runtime
        # says about a device id. A DTK device reports no UUID and `observe_local_hip`
        # records that in words rather than inventing one, so the check is that both
        # records agree on what was said -- not that something UUID-shaped was said.
        row = platform_for_paired_kind(PAIRED_HIP_KIND)
        if (executable_role(participants['candidate']['target']) != row.code_object.value
                or not isinstance(raw.get('gpu_uuid'), str) or not raw['gpu_uuid']
                or (not broker_pair and (re.fullmatch(rf'{row.local_job_prefix}-[0-9a-f]{{12}}', raw['job_id']) is None
                or raw['job_id'] == f'{row.local_job_prefix}-000000000000'))
                or launch.get('gpu_uuid') != raw['gpu_uuid']):
            raise ValueError('paired AMDGCN device/host identity differs')
    else:
        # Every declared kind is checked by name above. Reaching here means a policy kind
        # was admitted upstream that nothing here knows how to check, which is not the
        # same as the evidence being wrong -- and was previously CUDA's branch, so a third
        # vendor's pair was refused for not being a cubin.
        raise ValueError(
            f"paired assay {raw['kind']!r} has no declared device/host identity check")


def validate_paired_receipt(receipt, raw, correctness, launch, *, evaluation=None, baseline=None, candidate=None):
    """Bind both identities, actual order, every oracle check and one job to the lock."""
    if not isinstance(raw, Mapping) or raw.get('kind') not in PAIRED_KINDS:
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
    participant_work(raw)
    # Five distinct facts, each said by name. As one condition this reported that
    # something about the participants or the allocation differed and left the reader to
    # find which, from a worker whose artifacts are gone by the time anyone reads it.
    if participants['candidate']['candidate_sha256'] != receipt.candidate_sha256:
        raise ValueError(
            f"paired receipt candidate {participants['candidate']['candidate_sha256'][:12]} "
            f"is not the evaluated candidate {receipt.candidate_sha256[:12]}")
    if participants['candidate']['target'] != participants['baseline']['target']:
        raise ValueError(
            f"paired arms target different devices: candidate "
            f"{participants['candidate']['target']!r}, baseline "
            f"{participants['baseline']['target']!r}")
    if launch.get('participants') != participants:
        differing = sorted(
            role for role in set(participants) | set(launch.get('participants') or {})
            if (launch.get('participants') or {}).get(role) != participants.get(role))
        raise ValueError(
            f"paired launch receipt and timing record disagree on {', '.join(differing)}")
    if not isinstance(raw.get('job_id'), str) or not raw['job_id']:
        raise ValueError('paired timing record names no broker job')
    if launch.get('job_id') != raw['job_id']:
        raise ValueError(
            f"paired launch receipt job {launch.get('job_id')!r} is not the timing "
            f"record's job {raw['job_id']!r}")
    if raw['kind'] != raw['evaluation_protocol']['paired_timing']['kind']:
        raise ValueError('paired raw kind differs from the declared assay')
    admit_device_identity(raw, launch, participants)
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
            if raw['kind'] in METAL_KINDS:
                validate_metal_correctness_checks(check, validation_case_ids(raw['evaluation_protocol']))
            elif 'validation_case_ids' in raw['evaluation_protocol']:
                _validate_case_checks(check, validation_case_ids(raw['evaluation_protocol']), metal=False)
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
            if raw['kind'] in METAL_KINDS:
                validate_metal_correctness_checks(check, (receipt.case_id,) * protocol.route_calls_per_cohort, timed=True)
            oracle_check(check, check)
            passed = passed and check['passed']
    if raw['kind'] in METAL_KINDS:
        from .metal_observations import validate_launch_sequence
        commands = [row['command_buffer'] for role in protocol.arms for row in checks[role]['preflight']['launches']]
        if measured:
            commands += [command for measurement in raw['measurements'] for role in measurement['order']
                         for command in measurement['arms'][role]['command_buffers']]
            commands += [row['command_buffer'] for role in protocol.arms for row in checks[role]['postflight']['launches']]
        if [command['launch_index'] for command in commands] != list(range(len(commands))):
            raise ValueError('Metal receipt physical launch order differs from its assay')
        validate_launch_sequence(commands)
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
                if raw['kind'] in METAL_KINDS and record.get('command_buffers') != [
                        check['command_buffer'] for check in record['output_check']['launches']]:
                    raise ValueError('Metal timing and correctness launch observations differ')
    elif (raw.get('measurements') != [] or receipt.correctness_passed
          or raw.get('not_measured') != 'correctness_rejected'):
        raise ValueError('paired missing timing is not a correctness rejection')


def _validate_case_checks(check, case_ids, *, timed=False, metal):
    """One owner for complete per-case oracle metrics and their aggregation."""
    label = 'Metal' if metal else 'CUDA'
    launches = check.get('launches')
    if (not isinstance(launches, list) or any(not isinstance(row, Mapping) for row in launches)
            or [row.get('input_case_id') for row in launches] != list(case_ids)):
        raise ValueError(f'{label} correctness input-case coverage differs')
    combined = {'output_mismatches': 0, 'max_abs_error': 0.0, 'inputs_unchanged': True}
    seen, passed = set(), True
    for row in launches:
        if metal:
            from .metal_observations import command_buffer_ms
            command = row.get('command_buffer')
            command_buffer_ms(command)
            if command['launch_index'] in seen:
                raise ValueError('Metal correctness launch is duplicated')
            seen.add(command['launch_index'])
        elif set(row) != {'input_case_id', 'passed', 'metrics'}:
            raise ValueError('CUDA per-case correctness fields differ')
        metrics = row.get('metrics')
        if (not isinstance(metrics, Mapping) or type(metrics.get('output_mismatches')) is not int
                or metrics['output_mismatches'] < 0 or type(metrics.get('inputs_unchanged')) is not bool
                or type(metrics.get('max_abs_error')) not in (float, int)
                or not math.isfinite(metrics['max_abs_error']) or metrics['max_abs_error'] < 0
                or type(row.get('passed')) is not bool
                or row['passed'] != (metrics['output_mismatches'] == 0 and metrics['inputs_unchanged'])):
            raise ValueError(f'{label} per-launch oracle metrics differ')
        combined['output_mismatches'] += metrics['output_mismatches']
        combined['max_abs_error'] = max(combined['max_abs_error'], metrics['max_abs_error'])
        combined['inputs_unchanged'] &= metrics['inputs_unchanged']
        passed &= row['passed']
    projected = {key: check.get(key) for key in combined} if timed else check.get('metrics')
    if projected != combined or check.get('passed') is not passed:
        raise ValueError(f'{label} aggregate correctness differs from per-launch checks')


def validate_metal_correctness_checks(check, case_ids, *, timed=False):
    _validate_case_checks(check, case_ids, timed=timed, metal=True)


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


def participant_work(raw):
    """Derive physical work from manifest bytes bound by participant identities."""
    participants = raw['participants']
    declarations = raw.get('launch_manifests')
    if declarations is None:
        if any('kernel_bundle' in item['artifact_roles'] for item in participants.values()):
            raise ValueError('composed participant requires its sealed launch manifest for work accounting')
        return {role: {'modules': 1, 'kernels': 1} for role in participants}
    if not isinstance(declarations, Mapping) or set(declarations) != set(participants):
        raise ValueError('paired launch manifest coverage differs')
    result = {}
    for role, document in declarations.items():
        manifest = _manifest_spellings()[document['abi']].from_dict(document)
        if manifest.canonical_sha256 != participants[role]['launch_spec_sha256']:
            raise ValueError('paired launch manifest differs from its participant seal')
        if hasattr(manifest, 'check_complete_domain'):
            manifest.check_complete_domain()
        result[role] = {'modules': getattr(manifest, 'module_count', 1),
                        'kernels': getattr(manifest, 'kernels_per_call', 1)}
    return result


def validate_paired_broker(receipt, job_id, counters):
    """Tie a paired receipt to its actual broker allocation and complete work count."""
    if receipt.purpose == 'attribution' or not receipt.artifact_payloads:
        return
    raw = json.loads(receipt.artifact_payloads['timing_samples'])
    if not isinstance(raw, Mapping) or raw.get('kind') not in PAIRED_KINDS:
        return
    protocol = paired_protocol(raw['evaluation_protocol'])
    cohorts = len(protocol.pair_order) * 2 if receipt.timing is not None else 0
    cases = (len(validation_case_ids(raw['evaluation_protocol']))
             if raw['kind'] in METAL_KINDS or 'validation_case_ids' in raw['evaluation_protocol'] else 1)
    correctness_calls = (4 if cohorts else 2) * cases
    work = participant_work(raw)
    kernel_sum = sum(item['kernels'] for item in work.values())
    kernel_calls = kernel_sum * ((2 if cohorts else 1) * cases
                                + (len(protocol.pair_order) * protocol.route_calls_per_cohort if cohorts else 0))
    expected = {'compiler_invocations': 0, 'module_loads': 2 if raw['kind'] in METAL_KINDS else sum(item['modules'] for item in work.values()) * cases,
        'preflight_calls': 2 * cases,
        'kernel_calls': kernel_calls,
        'timing_samples': cohorts * protocol.samples_per_cohort, 'fallback_calls': 0}
    launch = json.loads(receipt.artifact_payloads['launch_receipt'])
    # No allocator's zero placeholder is a job: the worker writes one before any broker
    # admits it, under whichever prefix its environment carried.
    if (raw['job_id'] != job_id or not isinstance(job_id, str) or job_id.endswith('-000000000000')
        or dict(counters) != expected or launch.get('correctness_launches') != correctness_calls):
        raise ValueError('paired receipt broker job or work counters differ')
