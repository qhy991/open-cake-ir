"""One separately instrumented MACA launch with raw MCPTI records and resources.

This is attribution, never a cohort latency. Report allocated registers/shared/local
storage, observed launch geometry and dispatch time; occupancy, bandwidth and ISA
counters are explicitly outside the current collection coverage.
"""
from dataclasses import asdict
from pathlib import Path
import re
from typing import Mapping

from open_cake_ir.compiler.target import CodeObject, declared_target
from .attribution import TensorProfileFormat, load_instrumented_profile
from .metax_activity import activity_collector
from .metax_benchmark import dispatch_samples, kernel_records

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
    target = declared_target(manifest.target)
    admission = raw.get('device_admission')
    if (target.code_object is not CodeObject.MCFATBIN or not isinstance(admission, Mapping)
            or admission.get('target') != target.target_id or admission.get('device_arch') != target.target_id
            or admission.get('device_name') not in target.device_names or admission.get('warp_size') != target.warp_size
            or not isinstance(admission.get('pci_bus_id'), str)
            or re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}', admission['pci_bus_id']) is None
            or not isinstance(admission.get('runtime_library'), str)
            or not Path(admission['runtime_library']).is_absolute()):
        raise ValueError('MACA profile device identity differs')
    samples = dispatch_samples(raw.get('activity'), kernel_name=manifest.kernel_name,
        grid=manifest.grid, block=manifest.block, repeats=1, reset_record=None)
    kernel = kernel_records(raw['activity'])[0]
    return {'coverage': 'one_device_dispatch_and_allocated_resources', 'device_time_us': samples[0] * 1000,
        'correlation': kernel['correlation'], 'grid': kernel['grid'], 'block': kernel['block'],
        'registers_per_thread': kernel['registers_per_thread'],
        'static_shared_bytes': kernel['static_shared_bytes'],
        'dynamic_shared_bytes': kernel['dynamic_shared_bytes'],
        'local_bytes_per_thread': kernel['local_bytes_per_thread'],
        'non_target_dispatches': 0, 'pci_bus_id': admission['pci_bus_id'],
        'not_collected': list(NOT_COLLECTED), 'timing_use': 'attribution_only'}


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


def maca_attribution_feedback(profile):
    return {'kind': 'maca_dispatch_attribution', 'kernel_name': profile['kernel_name'], **profile['summary']}


def _validate_launch(profile, launch, correctness):
    if launch.get('job_id') != profile['job_id'] or launch.get('gpu_uuid') != profile.get('gpu_uuid'):
        raise ValueError('MACA attribution launch differs from instrumented profile')
    # The producer's preflight and separately captured native resource queries must
    # agree with the instrumented native record rather than borrowing a compiler estimate.
    resources = launch.get('resources')
    summary = profile['summary']
    if not isinstance(resources, Mapping) or any(resources.get(a) != summary[b] for a, b in (
            ('registers_per_thread','registers_per_thread'), ('local_bytes','local_bytes_per_thread'),
            ('dynamic_shared_bytes','dynamic_shared_bytes'))):
        raise ValueError('MACA instrumented resources differ from the loaded kernel')


MACA_PROFILE = TensorProfileFormat(MCPTI_PROFILE_KIND, maca_profile_summary, load_maca_profile,
                                   maca_attribution_feedback, _validate_launch)
