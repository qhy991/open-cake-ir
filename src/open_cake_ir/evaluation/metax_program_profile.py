"""MCPTI attribution for one sealed, ordered Program invocation.

Every stage must appear once, in order on one stream. Durations and inter-stage
gaps are attribution observations, never paired latency or performance evidence.
"""
from dataclasses import asdict
from typing import Mapping

from open_cake_ir.serialization import canonical_json_bytes
from .attribution import TensorProfileFormat, load_instrumented_profile
from .metax_activity import activity_collector, collect_activity
from .metax_benchmark import kernel_records, validate_loaded_resources
from .metax_observations import NOT_COLLECTED, validate_profile_correctness
from .triton_metax import validate_maca_admission

KIND = 'maca_program_activity_v1'


def capture_program_activity(launch, *, candidate, admission, activity_library):
    import torch
    from .program import program_components
    from .paired import candidate_identity
    manifest, children, manifests = program_components(candidate)
    collector = activity_collector(activity_library)
    primary = None
    activity = None
    try:
        activity = collect_activity(collector, launch, synchronize=torch.cuda.synchronize)
    except BaseException as error:
        primary = error
        activity = getattr(error, 'activity_snapshot', None)
    raw = {'activity': activity, 'manifest': manifest.as_dict(),
           'stage_manifests': {name: item.as_dict() for name, item in manifests.items()},
           'stage_candidates': {name: candidate_identity(item) for name, item in children.items()},
           'device_admission': asdict(admission), 'not_collected': list(NOT_COLLECTED)}
    if primary is not None:
        error = RuntimeError(str(primary))
        error.artifact_payloads = {'program_activity': canonical_json_bytes(raw)}
        raise error from primary
    # The caller saves this raw observation before numerical/profile validation.
    return raw


def program_launch_manifests(raw):
    """Read the existing sealed parent/child launch facts for profile or timing."""
    from .core import TensorLaunchManifest
    from .program import ProgramLaunchManifest, stage_abi
    from .paired import candidate_from_identity
    if not isinstance(raw, Mapping):
        raise ValueError('MACA Program launch facts differ')
    manifest = ProgramLaunchManifest.from_dict(raw.get('manifest'))
    names = [stage.name for stage in manifest.program.stages]
    if (not isinstance(raw.get('stage_manifests'), Mapping)
            or not isinstance(raw.get('stage_candidates'), Mapping)
            or set(raw['stage_manifests']) != set(names) or set(raw['stage_candidates']) != set(names)):
        raise ValueError('MACA Program profile must bind every sealed stage')
    manifests = {}
    for stage in manifest.program.stages:
        child = candidate_from_identity(raw['stage_candidates'][stage.name])
        spec = TensorLaunchManifest.from_dict(raw['stage_manifests'][stage.name])
        spec.check_complete_domain()
        if (spec.target != manifest.target or spec.workload_sha256 != manifest.workload_sha256
                or spec.case_id != manifest.case_id or spec.tensor_abi != stage_abi(stage)
                or child.target != spec.target or child.entry_point != spec.kernel_name
                or child.launch_spec_sha256 != spec.canonical_sha256
                or child.artifact_roles.get('lowered_source') != manifest.lowered_sources[stage.name]
                or spec.hidden_null_pointer_parameters != 0 or spec.aligned_variant):
            raise ValueError('MACA Program stage differs from its sealed native launch')
        manifests[stage.name] = spec
    return manifest, manifests


def validate_program_dispatches(records, launches, manifests):
    """One complete ordered invocation, already decoded by kernel_records."""
    if len(records) != len(manifests):
        raise ValueError('MACA Program profile dispatch count differs')
    previous = None
    stream = None
    for spec, kernel in zip(manifests.values(), records, strict=True):
        if (kernel['name'] != spec.kernel_name or tuple(kernel['grid']) != spec.grid
                or tuple(kernel['block']) != spec.block
                or kernel['dynamic_shared_bytes'] != spec.dynamic_shared_memory_bytes
                or launches[kernel['correlation']]['cbid'] != 60):
            raise ValueError('MACA Program stage differs from its sealed native launch')
        current = (kernel['device'], kernel['context'], kernel['stream'])
        if stream is not None and current != stream:
            raise ValueError('MACA Program stages changed stream or context')
        if previous is not None and kernel['start_ns'] < previous:
            raise ValueError('MACA ordered Program stages overlap')
        previous, stream = kernel['end_ns'], current


def program_profile_summary(raw):
    if not isinstance(raw, Mapping) or list(raw.get('not_collected') or ()) != list(NOT_COLLECTED):
        raise ValueError('MACA Program profile coverage differs')
    manifest, manifests = program_launch_manifests(raw)
    admission = raw.get('device_admission')
    validate_maca_admission(admission, target_id=manifest.target,
                           job_id=admission.get('broker_job_id') if isinstance(admission, Mapping) else None)
    records = kernel_records(raw.get('activity'))
    launches = {row['correlation']: row for row in raw['activity']['records'] if row['kind'] == 5}
    validate_program_dispatches(records, launches, manifests)
    stages = []
    previous = None
    for name, kernel in zip(manifests, records, strict=True):
        stages.append({'stage': name, 'kernel_name': manifests[name].kernel_name,
            'device_time_us': (kernel['end_ns'] - kernel['start_ns']) / 1000,
            'preceding_gap_us': 0 if previous is None else (kernel['start_ns'] - previous) / 1000,
            **{key: kernel[key] for key in ('correlation', 'grid', 'block', 'registers_per_thread',
                                           'static_shared_bytes', 'dynamic_shared_bytes')},
            'mcpti_reported_local_bytes_per_thread': kernel['local_bytes_per_thread']})
        previous = kernel['end_ns']
    return {'coverage': 'one_ordered_program_and_all_stage_resources', 'stages': stages,
            'program_span_us': (records[-1]['end_ns'] - records[0]['start_ns']) / 1000,
            'summed_stage_time_us': sum(row['device_time_us'] for row in stages),
            'non_target_dispatches': 0, 'pci_bus_id': admission['pci_bus_id'],
            'not_collected': list(NOT_COLLECTED), 'not_qualified': ['local_memory_reservation'],
            'timing_use': 'attribution_only'}


def load_program_profile(payload, *, expected_candidate_sha256, expected_case_id,
                         expected_protocol_sha256=None):
    import json
    from .program import ProgramLaunchManifest
    document = json.loads(payload)
    if not isinstance(document, Mapping) or not isinstance(document.get('raw'), Mapping):
        raise ValueError('MACA Program profile must contain its sealed Program')
    manifest = ProgramLaunchManifest.from_dict(document['raw'].get('manifest'))
    profile = load_instrumented_profile(payload, kind=KIND, job_prefix='maca', label='MACA Program',
        summary=program_profile_summary, raw_name='MCPTI Program activity',
        expected_candidate_sha256=expected_candidate_sha256, expected_case_id=expected_case_id,
        expected_protocol_sha256=expected_protocol_sha256,
        name_valid=lambda name: name == manifest.kernel_name)
    raw = profile['raw']
    if (profile['kernel_name'] != manifest.kernel_name or expected_case_id != manifest.case_id
            or raw['device_admission']['broker_job_id'] != profile['job_id']
            or raw['device_admission'].get('gpu_uuid') != profile.get('gpu_uuid')
            or any(item['candidate_sha256'] != expected_candidate_sha256
                   for item in raw['stage_candidates'].values())):
        raise ValueError('MACA Program profile candidate or case identity differs')
    return profile


def validate_program_profile_launch(profile, launch, correctness):
    from .program import ProgramLaunchManifest
    raw = profile['raw']
    manifest = ProgramLaunchManifest.from_dict(raw['manifest'])
    if (launch.get('candidate_sha256') != profile['candidate_sha256']
            or launch.get('manifest_sha256') != manifest.canonical_sha256
            or launch.get('stage_candidates') != raw['stage_candidates']
            or launch.get('device_admission') != raw['device_admission']
            or launch.get('job_id') != profile['job_id'] or launch.get('gpu_uuid') != profile.get('gpu_uuid')
            or launch.get('kernel_calls') != manifest.kernels_per_call
            or launch.get('module_unloaded') is not True or launch.get('fallback_calls') != 0):
        raise ValueError('MACA Program profile differs from its loaded launch')
    validate_profile_correctness(launch, correctness)
    resources = launch.get('resources')
    names = {stage.name for stage in manifest.program.stages}
    if (not isinstance(resources, Mapping) or resources.get('kind') != 'ordered_program'
            or not isinstance(resources.get('stages'), Mapping) or set(resources['stages']) != names):
        raise ValueError('MACA Program loaded stage resource coverage differs')
    for stage in profile['summary']['stages']:
        validate_loaded_resources(resources['stages'][stage['stage']], stage)


def program_attribution_feedback(profile, launch):
    return {'kind': 'maca_program_attribution', 'program': profile['kernel_name'], **profile['summary'],
            'stages': [{**stage, 'function_local_bytes_per_thread':
                       launch['resources']['stages'][stage['stage']]['local_bytes']}
                       for stage in profile['summary']['stages']]}


MACA_PROGRAM_PROFILE = TensorProfileFormat(KIND, program_profile_summary, load_program_profile,
                                         program_attribution_feedback, validate_program_profile_launch)
