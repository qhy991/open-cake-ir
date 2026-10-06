"""Unprofiled default-stream event intervals for sealed MACA launches.

The event interval includes exposed host submission gaps. It is not MCPTI's
kernel-only duration. The reset is host-ordered on the same stream before the
start event; no profiler or device-activity coverage is claimed by this timer.
"""
from __future__ import annotations

import math
from typing import Mapping

from open_cake_ir.compiler.target import CodeObject, declared_target

TIMER = 'maca_pytorch_event_elapsed_ms'
RESET = 'fp32_fill_ones_4x_declared_l2_default_stream_before_start_event'
INTERVAL = 'default_stream_events_around_sealed_launch_including_submission_gaps'
COVERAGE = 'trusted_loader_and_host_ordering_without_activity_trace'


class MacaEventBenchmark:
    def __init__(self, manifest, *, l2_cache_bytes: int):
        from .program import ProgramLaunchManifest
        if isinstance(manifest, ProgramLaunchManifest) or manifest.aligned_variant:
            raise ValueError('MACA event timing admits one non-variant sealed dispatch')
        target = declared_target(manifest.target)
        if (target.code_object is not CodeObject.MCFATBIN
                or type(l2_cache_bytes) is not int or l2_cache_bytes <= 0
                or target.l2_cache_bytes != l2_cache_bytes):
            raise ValueError('MACA event target or L2 declaration differs')
        self.manifest, self.l2_cache_bytes = manifest, l2_cache_bytes
        self._reset = None
        self.last_activity = None
        self.non_target_dispatches = None  # No profiler observes foreign dispatches.

    def __call__(self, function, *, dry_run_iters, repeat_iters, cold_l2_cache, use_cuda_graph):
        import torch
        if (use_cuda_graph or cold_l2_cache is not True or dry_run_iters != 11
                or type(repeat_iters) is not int or repeat_iters <= 0):
            raise ValueError('MACA event warmup, reset or graph contract differs')
        if not getattr(torch.version, 'maca', None):
            raise ValueError('MACA event timer requires the admitted MACA PyTorch')
        stream = torch.cuda.default_stream(0)
        if stream.cuda_stream != 0:
            raise ValueError('MACA native launch and event stream differ')
        if torch.cuda.get_device_properties(0).L2_cache_size != self.l2_cache_bytes:
            raise ValueError('MACA event observed L2 differs from Target')
        if self._reset is None:
            self._reset = torch.empty(self.l2_cache_bytes, dtype=torch.float32, device='cuda:0')
        observations = []
        self.last_activity = {
            'kind': 'maca_event_samples_v1', 'timer': TIMER, 'cache_policy': RESET,
            'interval': INTERVAL, 'coverage': COVERAGE, 'profiler_enabled': False,
            'target': self.manifest.target, 'device': 0, 'stream': 0,
            'l2_cache_bytes': self.l2_cache_bytes, 'reset_bytes': 4 * self.l2_cache_bytes,
            'warmup_calls': dry_run_iters, 'event_pair_primed': False,
            'launch': {'kernel_name': self.manifest.kernel_name,
                       'grid': list(self.manifest.grid), 'block': list(self.manifest.block),
                       'dynamic_shared_memory_bytes': self.manifest.dynamic_shared_memory_bytes},
            'samples': observations,
        }
        with torch.cuda.stream(stream):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            # Instantiate lazy event handles outside the measured interval.
            begin.record(stream); end.record(stream); end.synchronize()
            self.last_activity['event_pair_primed'] = True
            for _ in range(dry_run_iters):
                function()
            stream.synchronize()
            for index in range(repeat_iters):
                self._reset.fill_(1.0)
                begin.record(stream)
                function()
                end.record(stream)
                end.synchronize()
                elapsed = float(begin.elapsed_time(end))
                observations.append({'index': index, 'elapsed_ms': elapsed,
                    'reset_enqueued_before_start': True, 'end_synchronized': True})
                if not math.isfinite(elapsed) or elapsed <= 0:
                    raise ValueError('MACA event interval must be finite and positive')
        return [row['elapsed_ms'] for row in observations]


def validate_cohort(record, manifest, *, sample_count):
    from .program import ProgramLaunchManifest
    target = declared_target(manifest.target)
    raw = record.get('native_activity')
    expected_launch = {'kernel_name': manifest.kernel_name,
                       'grid': list(manifest.grid), 'block': list(manifest.block),
                       'dynamic_shared_memory_bytes': manifest.dynamic_shared_memory_bytes}
    expected = {'kind': 'maca_event_samples_v1', 'timer': TIMER, 'cache_policy': RESET,
                'interval': INTERVAL, 'coverage': COVERAGE, 'profiler_enabled': False,
                'target': manifest.target, 'device': 0, 'stream': 0,
                'l2_cache_bytes': target.l2_cache_bytes,
                'reset_bytes': 4 * target.l2_cache_bytes if target.l2_cache_bytes else None,
                'warmup_calls': 11, 'event_pair_primed': True, 'launch': expected_launch}
    if (target.code_object is not CodeObject.MCFATBIN or isinstance(manifest, ProgramLaunchManifest)
            or manifest.aligned_variant or not isinstance(raw, Mapping)
            or set(raw) != set(expected) | {'samples'}
            or any(raw.get(key) != value or type(raw.get(key)) is not type(value)
                   for key, value in expected.items() if key != 'launch')
            or raw.get('launch') != expected_launch
            or record.get('non_target_dispatches') is not None):
        raise ValueError('MACA event interval, reset, coverage or launch declaration differs')
    rows = raw['samples']
    if not isinstance(rows, list) or len(rows) != sample_count:
        raise ValueError('MACA event sample count differs')
    values = []
    for index, row in enumerate(rows):
        if (not isinstance(row, Mapping)
                or set(row) != {'index', 'elapsed_ms', 'reset_enqueued_before_start', 'end_synchronized'}
                or type(row['index']) is not int or row['index'] != index
                or row['reset_enqueued_before_start'] is not True or row['end_synchronized'] is not True
                or type(row['elapsed_ms']) not in (int, float)
                or not math.isfinite(row['elapsed_ms']) or row['elapsed_ms'] <= 0):
            raise ValueError('MACA event sample, reset order or synchronization differs')
        values.append(float(row['elapsed_ms']))
    if record.get('samples_ms') != values:
        raise ValueError('MACA timing samples differ from retained event observations')


def validate_paired_events(raw, protocol):
    from .core import TensorLaunchManifest
    documents, participants = raw.get('launch_manifests'), raw.get('participants')
    if (not isinstance(documents, Mapping) or set(documents) != set(protocol.arms)
            or not isinstance(participants, Mapping) or set(participants) != set(protocol.arms)):
        raise ValueError('MACA event pair requires both sealed manifests')
    for role in protocol.arms:
        manifest = TensorLaunchManifest.from_dict(documents[role])
        if (manifest.canonical_sha256 != participants[role].get('launch_spec_sha256')
                or manifest.target != participants[role].get('target')
                or manifest.workload_sha256 != raw.get('workload_sha256')
                or manifest.case_id != raw.get('case_id')):
            raise ValueError('MACA event manifest differs from its sealed participant')
        for measurement in raw['measurements']:
            validate_cohort(measurement['arms'][role], manifest,
                            sample_count=protocol.samples_per_cohort)


def validate_paired_device(raw, launch, participants):
    from .triton_metax import validate_maca_admission
    admission = raw.get('device_admission')
    validate_maca_admission(admission, target_id=participants['candidate']['target'], job_id=raw.get('job_id'))
    if (participants['baseline']['target'] != participants['candidate']['target']
            or launch.get('job_id') != raw.get('job_id')
            or launch.get('device_admission') != admission
            or raw.get('gpu_uuid') is not None or launch.get('gpu_uuid') is not None
            or raw.get('allocation_mode') != 'local_serialized'
            or launch.get('allocation_mode') != 'local_serialized'
            or raw.get('external_gpu_activity') != 'not_excluded'
            or launch.get('external_gpu_activity') != 'not_excluded'):
        raise ValueError('MACA event paired device and allocation identity differ')
    resources = launch.get('resources')
    if not isinstance(resources, Mapping) or set(resources) != set(participants):
        raise ValueError('MACA event loaded resource coverage differs')
    for role, values in resources.items():
        if any(type(values.get(name)) is not int or values[name] < 0
               for name in ('registers_per_thread', 'local_bytes', 'dynamic_shared_bytes')):
            raise ValueError('MACA event loaded resources differ')
        manifest = raw['launch_manifests'][role]
        if values['dynamic_shared_bytes'] != manifest['dynamic_shared_memory_bytes']:
            raise ValueError('MACA event loaded shared memory differs')
