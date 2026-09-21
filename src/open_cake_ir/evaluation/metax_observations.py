"""One separately instrumented MACA launch with raw MCPTI records and resources.

This is attribution, never a cohort latency. Report observed registers/shared memory
and the independently queried function-local requirement. MCPTI's local reservation
counter has no qualified interpretation; preserve its raw value without presenting
it as the function's usage. Occupancy, bandwidth and ISA counters are uncollected.
"""
from dataclasses import asdict
import math
from typing import Mapping

from open_cake_ir.compiler.target import declared_target
from .attribution import TensorProfileFormat, load_instrumented_profile
from .metax_activity import activity_collector
from .metax_benchmark import dispatch_samples, kernel_records, validate_loaded_resources
from .triton_metax import validate_maca_admission

MCPTI_PROFILE_KIND = 'maca_dispatch_activity_v1'
NOT_COLLECTED = ('achieved_occupancy', 'bandwidth', 'instruction_counters')


def collect_maca_activity(launch, kernel_name, *, manifest, admission, activity_library):
    import torch
    from .loaders import LifecycleError
    if kernel_name != manifest.kernel_name:
        raise ValueError('MACA profile kernel differs from its manifest')
    collector = activity_collector(activity_library)
    torch.cuda.synchronize()
    collector.begin()
    try:
        launch()
        torch.cuda.synchronize()
    except BaseException as primary:
        errors = [primary]
        try:
            torch.cuda.synchronize()
        except BaseException as error:
            errors.append(error)
        try:
            collector.finish()
        except BaseException as error:
            errors.append(error)
        if len(errors) > 1:
            raise LifecycleError(*errors) from primary
        raise
    raw = {'activity': collector.finish(), 'manifest': manifest.as_dict(),
           'device_admission': asdict(admission), 'not_collected': list(NOT_COLLECTED)}
    maca_profile_summary(raw)
    return raw


def maca_profile_summary(raw: Mapping) -> dict:
    from .core import TensorLaunchManifest
    if not isinstance(raw, Mapping) or list(raw.get('not_collected') or ()) != list(NOT_COLLECTED):
        raise ValueError('MACA profile coverage differs')
    manifest = TensorLaunchManifest.from_dict(raw.get('manifest'))
    manifest.check_complete_domain()
    if manifest.hidden_null_pointer_parameters != 0:
        raise ValueError('MACA profile declares unsupported hidden launch parameters')
    target = declared_target(manifest.target)
    admission = raw.get('device_admission')
    validate_maca_admission(admission, target_id=target.target_id,
                           job_id=admission.get('broker_job_id') if isinstance(admission, Mapping) else None)
    samples = dispatch_samples(raw.get('activity'), kernel_name=manifest.kernel_name,
        grid=manifest.grid, block=manifest.block, repeats=1, reset_record=None)
    kernel = kernel_records(raw['activity'])[0]
    launches = [row for row in raw['activity']['records']
                if row['kind'] == 5 and row['correlation'] == kernel['correlation']]
    if len(launches) != 1 or launches[0]['cbid'] != 60:
        raise ValueError('MACA profile is not the sealed module launch')
    if kernel['dynamic_shared_bytes'] != manifest.dynamic_shared_memory_bytes:
        raise ValueError('MACA profile shared memory differs from its native dispatch')
    return {'coverage': 'one_device_dispatch_and_reported_resources', 'device_time_us': samples[0] * 1000,
        'correlation': kernel['correlation'], 'grid': kernel['grid'], 'block': kernel['block'],
        'registers_per_thread': kernel['registers_per_thread'],
        'static_shared_bytes': kernel['static_shared_bytes'],
        'dynamic_shared_bytes': kernel['dynamic_shared_bytes'],
        'mcpti_reported_local_bytes_per_thread': kernel['local_bytes_per_thread'],
        'non_target_dispatches': 0, 'pci_bus_id': admission['pci_bus_id'],
        'not_collected': list(NOT_COLLECTED), 'not_qualified': ['local_memory_reservation'],
        'timing_use': 'attribution_only'}


def load_maca_profile(payload: bytes, *, expected_candidate_sha256, expected_case_id,
                      expected_protocol_sha256=None):
    profile = load_instrumented_profile(payload, kind=MCPTI_PROFILE_KIND, job_prefix='maca',
        label='MACA', summary=maca_profile_summary, raw_name='MCPTI activity',
        expected_candidate_sha256=expected_candidate_sha256, expected_case_id=expected_case_id,
        expected_protocol_sha256=expected_protocol_sha256)
    raw = profile['raw']
    if (raw['manifest']['kernel_name'] != profile['kernel_name']
            or raw['manifest']['case_id'] != expected_case_id
            or raw['device_admission']['broker_job_id'] != profile['job_id']
            or raw['device_admission'].get('gpu_uuid') != profile.get('gpu_uuid')):
        raise ValueError('MACA profile and its observed launch identity differ')
    return profile


def maca_attribution_feedback(profile, launch):
    return {'kind': 'maca_dispatch_attribution', 'kernel_name': profile['kernel_name'], **profile['summary'],
            'function_local_bytes_per_thread': launch['resources']['local_bytes']}


def _validate_launch(profile, launch, correctness):
    from .core import TensorLaunchManifest
    if launch.get('job_id') != profile['job_id'] or launch.get('gpu_uuid') != profile.get('gpu_uuid'):
        raise ValueError('MACA attribution launch differs from instrumented profile')
    manifest = TensorLaunchManifest.from_dict(profile['raw']['manifest'])
    if (launch.get('candidate_sha256') != profile['candidate_sha256']
            or launch.get('manifest_sha256') != manifest.canonical_sha256
            or launch.get('device_admission') != profile['raw']['device_admission']):
        raise ValueError('MACA profile manifest or device differs from the loaded launch')
    validate_profile_correctness(launch, correctness)
    # Compare only device-verified function/activity quantities.
    validate_loaded_resources(launch.get('resources'), profile['summary'])


def validate_profile_correctness(launch, correctness):
    """Both single-kernel and ordered-Program profiles require two full checks."""
    instrumented = correctness.get('instrumented')
    if (not isinstance(instrumented, Mapping) or instrumented.get('passed') is not True
            or not isinstance(instrumented.get('metrics'), Mapping)
            or instrumented['metrics'].get('output_mismatches') != 0
            or instrumented['metrics'].get('inputs_unchanged') is not True
            or correctness.get('correctness_launches') != 2 or launch.get('correctness_launches') != 2):
        raise ValueError('MACA profile lacks the instrumented output oracle check')
    preflight = correctness.get('preflight')
    checks = (preflight, instrumented['metrics'])
    if any(not isinstance(check, Mapping)
           or type(check.get('output_mismatches')) is not int or check['output_mismatches'] != 0
           or check.get('inputs_unchanged') is not True
           or type(check.get('max_abs_error')) not in (int, float)
           or not math.isfinite(check['max_abs_error']) or check['max_abs_error'] < 0 for check in checks):
        raise ValueError('MACA profile preflight or instrumented oracle metrics differ')
    combined = {'output_mismatches': 0, 'inputs_unchanged': True,
                'max_abs_error': max(check['max_abs_error'] for check in checks)}
    if correctness.get('passed') is not True or correctness.get('metrics') != combined:
        raise ValueError('MACA profile correctness aggregate differs from both oracle checks')


MACA_PROFILE = TensorProfileFormat(MCPTI_PROFILE_KIND, maca_profile_summary, load_maca_profile,
                                   maca_attribution_feedback, _validate_launch)
