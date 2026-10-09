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
NATIVE_TIMER = 'maca_native_event_elapsed_ms'
NATIVE_RESET = 'runtime_memset_d32_ones_4x_declared_l2_before_start_event'
NATIVE_INTERVAL = 'native_default_stream_events_around_prevalidated_sealed_launch'
TORCH_RESET = 'torch_fp32_fill_ones_twice_4x_declared_l2_before_start_event'
GATED_TIMER = 'maca_native_gated_event_elapsed_ms'
GATED_RESET = 'torch_fp32_fill_ones_twice_4x_declared_l2_owned_stream_before_start_event'
GATED_INTERVAL = 'owned_stream_device_events_after_all_target_commands_are_enqueued'
PROGRAM_INTERVAL = 'default_stream_events_around_complete_program_including_checks_and_submission_gaps'


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

    def _event_record(self, warmups, observations):
        return {
            'kind': 'maca_event_samples_v1', 'timer': TIMER, 'cache_policy': RESET,
            'interval': INTERVAL, 'coverage': COVERAGE, 'profiler_enabled': False,
            'target': self.manifest.target, 'device': 0, 'stream': 0,
            'l2_cache_bytes': self.l2_cache_bytes, 'reset_bytes': 4 * self.l2_cache_bytes,
            'warmup_calls': warmups, 'event_pair_primed': False,
            'launch': {'kernel_name': self.manifest.kernel_name,
                       'grid': list(self.manifest.grid), 'block': list(self.manifest.block),
                       'dynamic_shared_memory_bytes': self.manifest.dynamic_shared_memory_bytes},
            'samples': observations,
        }

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
        self.last_activity = self._event_record(dry_run_iters, observations)
        with torch.cuda.stream(stream):
            # Warm the reset operation before the first formal interval as well.
            self._reset.fill_(1.0)
            stream.synchronize()
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


class MacaProgramEventBenchmark(MacaEventBenchmark):
    """One interval covers all ordered stages; allocation precedes every interval.

    The implementation can be exercised by device qualification tools. Production
    Run admission remains with ``admit_program_execution`` and its evidence set.
    """

    def __init__(self, manifest, *, l2_cache_bytes):
        from .program import ProgramLaunchManifest
        target = declared_target(manifest.target)
        if (not isinstance(manifest, ProgramLaunchManifest) or manifest.aligned_stages
                or target.code_object is not CodeObject.MCFATBIN
                or type(l2_cache_bytes) is not int or l2_cache_bytes <= 0
                or target.l2_cache_bytes != l2_cache_bytes):
            raise ValueError('MACA Program event manifest or L2 declaration differs')
        self.manifest, self.l2_cache_bytes = manifest, l2_cache_bytes
        self._reset = self.last_activity = self.non_target_dispatches = None

    def _event_record(self, warmups, observations):
        return {
            'kind': 'maca_program_event_samples_v1', 'timer': TIMER, 'cache_policy': RESET,
            'interval': PROGRAM_INTERVAL, 'coverage': COVERAGE, 'profiler_enabled': False,
            'target': self.manifest.target, 'device': 0, 'stream': 0,
            'l2_cache_bytes': self.l2_cache_bytes, 'reset_bytes': 4 * self.l2_cache_bytes,
            'warmup_calls': warmups, 'event_pair_primed': False,
            'stage_names': [stage.name for stage in self.manifest.program.stages],
            'completed_program_calls': 0, 'completed_stage_calls': 0, 'samples': observations,
        }

    def __call__(self, *args, **kwargs):
        raise ValueError('MACA Program events require the complete loaded Program adapter')

    def capture_loaded_cohort(self, loaded, arguments, *, dry_run_iters, repeat_iters):
        from .program import LoadedProgram
        if (not isinstance(loaded.loaded, LoadedProgram)
                or loaded.manifest.as_dict() != self.manifest.as_dict()
                or loaded.loaded.manifest.as_dict() != self.manifest.as_dict()
                or dry_run_iters != 11 or repeat_iters != 5
                or len(arguments) != dry_run_iters + repeat_iters
                or len({id(values) for values in arguments}) != len(arguments)):
            raise ValueError('MACA Program event arguments or cohort differs')
        used = 0
        def launch():
            nonlocal used
            if used >= len(arguments):
                raise ValueError('MACA Program event invocation budget exceeded')
            before = loaded.loaded.launch_calls
            loaded.launch(arguments[used])
            if loaded.loaded.launch_calls - before != self.manifest.kernels_per_call:
                raise ValueError('MACA Program event stage count differs')
            used += 1
            self.last_activity['completed_program_calls'] = used
            self.last_activity['completed_stage_calls'] += self.manifest.kernels_per_call
        values = super().__call__(launch, dry_run_iters=dry_run_iters,
                                  repeat_iters=repeat_iters, cold_l2_cache=True, use_cuda_graph=False)
        if used != len(arguments):
            raise ValueError('MACA Program event invocation count differs')
        return values


class MacaNativeEventBenchmark(MacaEventBenchmark):
    """Submit one cohort in native code with tensor checks outside all samples."""
    native_torch_reset = False
    native_gated = False

    def capture_loaded_cohort(self, loaded, arguments, *, dry_run_iters, repeat_iters):
        import torch
        from .metax_native_events import capture
        if (loaded.manifest is not self.manifest or dry_run_iters != 11 or repeat_iters != 5
                or not getattr(torch.version, 'maca', None)
                or torch.cuda.default_stream(0).cuda_stream != 0
                or torch.cuda.get_device_properties(0).L2_cache_size != self.l2_cache_bytes):
            raise ValueError('Native MACA event manifest, stream or reset target differs')
        if self._reset is None:
            self._reset = torch.empty(self.l2_cache_bytes, dtype=torch.float32, device='cuda:0')
        try:
            values = capture(loaded, arguments, self._reset, warmups=dry_run_iters,
                             samples=repeat_iters, torch_reset=self.native_torch_reset,
                             gated=self.native_gated)
        except Exception as error:
            self.last_activity = {'kind': 'failed_maca_native_event_capture_v1',
                'timer': GATED_TIMER if self.native_gated else NATIVE_TIMER,
                'interval': GATED_INTERVAL if self.native_gated else NATIVE_INTERVAL,
                'cache_policy': GATED_RESET if self.native_gated else TORCH_RESET if self.native_torch_reset else NATIVE_RESET,
                'target': self.manifest.target,
                'kernel_name': self.manifest.kernel_name,
                'observations': getattr(error, 'native_observations', None),
                'error': str(error)}
            raise
        self.last_activity = {
            'kind': 'maca_gated_event_samples_v1' if self.native_gated else 'maca_native_torch_reset_samples_v1' if self.native_torch_reset else 'maca_native_event_samples_v1',
            'timer': GATED_TIMER if self.native_gated else NATIVE_TIMER,
            'cache_policy': GATED_RESET if self.native_gated else TORCH_RESET if self.native_torch_reset else NATIVE_RESET,
            'interval': GATED_INTERVAL if self.native_gated else NATIVE_INTERVAL,
            'coverage': COVERAGE, 'profiler_enabled': False,
            'target': self.manifest.target, 'device': 0,
            'stream': 'owned_nonblocking' if self.native_gated else 0,
            'l2_cache_bytes': self.l2_cache_bytes, 'reset_bytes': 4 * self.l2_cache_bytes,
            'warmup_calls': dry_run_iters, 'event_pair_primed': True,
            'launch': {'kernel_name': self.manifest.kernel_name,
                'grid': list(self.manifest.grid), 'block': list(self.manifest.block),
                'dynamic_shared_memory_bytes': self.manifest.dynamic_shared_memory_bytes},
            'samples': [{'index': i, 'elapsed_ms': value,
                'reset_enqueued_before_start': True, 'end_synchronized': True}
                for i, value in enumerate(values)],
        }
        if self.native_gated:
            self.last_activity['host_gate'] = 'released_after_begin_target_end_are_queued'
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError('Native MACA event interval must be finite and positive')
        return values


class MacaTorchResetEventBenchmark(MacaNativeEventBenchmark):
    """Native event submission after two captured-Torch reset fills outside timing."""
    native_torch_reset = True


class MacaGatedEventBenchmark(MacaTorchResetEventBenchmark):
    """A second-stream host gate defers GPU start until all commands are queued."""
    native_gated = True


def validate_cohort(record, manifest, *, sample_count, native=False, torch_reset=False, gated=False):
    from .program import ProgramLaunchManifest
    target = declared_target(manifest.target)
    raw = record.get('native_activity')
    if isinstance(manifest, ProgramLaunchManifest):
        if native or torch_reset or gated:
            raise ValueError('MACA Program events do not use a single-dispatch native timer')
        return validate_program_cohort(record, manifest, sample_count=sample_count)
    expected_launch = {'kernel_name': manifest.kernel_name,
                       'grid': list(manifest.grid), 'block': list(manifest.block),
                       'dynamic_shared_memory_bytes': manifest.dynamic_shared_memory_bytes}
    expected = {'kind': ('maca_gated_event_samples_v1' if gated else 'maca_native_torch_reset_samples_v1' if torch_reset else
                        'maca_native_event_samples_v1' if native else 'maca_event_samples_v1'),
                'timer': GATED_TIMER if gated else NATIVE_TIMER if (native or torch_reset) else TIMER,
                'cache_policy': GATED_RESET if gated else TORCH_RESET if torch_reset else NATIVE_RESET if native else RESET,
                'interval': GATED_INTERVAL if gated else NATIVE_INTERVAL if (native or torch_reset) else INTERVAL,
                'coverage': COVERAGE, 'profiler_enabled': False,
                'target': manifest.target, 'device': 0,
                'stream': 'owned_nonblocking' if gated else 0,
                'l2_cache_bytes': target.l2_cache_bytes,
                'reset_bytes': 4 * target.l2_cache_bytes if target.l2_cache_bytes else None,
                'warmup_calls': 11, 'event_pair_primed': True, 'launch': expected_launch}
    if gated:
        expected['host_gate'] = 'released_after_begin_target_end_are_queued'
    if (target.code_object is not CodeObject.MCFATBIN or isinstance(manifest, ProgramLaunchManifest)
            or manifest.aligned_variant or not isinstance(raw, Mapping)
            or set(raw) != set(expected) | {'samples'}
            or any(raw.get(key) != value or type(raw.get(key)) is not type(value)
                   for key, value in expected.items() if key != 'launch')
            or raw.get('launch') != expected_launch
            or record.get('non_target_dispatches') is not None):
        raise ValueError('MACA event interval, reset, coverage or launch declaration differs')
    _validate_samples(record, sample_count)


def validate_program_cohort(record, manifest, *, sample_count):
    target = declared_target(manifest.target)
    raw = record.get('native_activity')
    calls = 11 + sample_count
    expected = {
        'kind': 'maca_program_event_samples_v1', 'timer': TIMER, 'cache_policy': RESET,
        'interval': PROGRAM_INTERVAL, 'coverage': COVERAGE, 'profiler_enabled': False,
        'target': manifest.target, 'device': 0, 'stream': 0,
        'l2_cache_bytes': target.l2_cache_bytes, 'reset_bytes': 4 * target.l2_cache_bytes,
        'warmup_calls': 11, 'event_pair_primed': True,
        'stage_names': [stage.name for stage in manifest.program.stages],
        'completed_program_calls': calls, 'completed_stage_calls': calls * manifest.kernels_per_call,
    }
    if (target.code_object is not CodeObject.MCFATBIN or manifest.aligned_stages
            or not isinstance(raw, Mapping) or set(raw) != set(expected) | {'samples'}
            or any(raw.get(key) != value or type(raw.get(key)) is not type(value)
                   for key, value in expected.items())
            or record.get('non_target_dispatches') is not None):
        raise ValueError('MACA Program interval, stage count, reset or coverage differs')
    _validate_samples(record, sample_count)


def _validate_samples(record, sample_count):
    rows = record['native_activity']['samples']
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
    from .program import ProgramLaunchManifest
    from .paired import PAIRED_MACA_NATIVE_EVENT_KIND, PAIRED_MACA_TORCH_EVENT_KIND, PAIRED_MACA_GATED_EVENT_KIND
    documents, participants = raw.get('launch_manifests'), raw.get('participants')
    if (not isinstance(documents, Mapping) or set(documents) != set(protocol.arms)
            or not isinstance(participants, Mapping) or set(participants) != set(protocol.arms)):
        raise ValueError('MACA event pair requires both sealed manifests')
    for role in protocol.arms:
        manifest_type = (ProgramLaunchManifest if documents[role].get('abi') == ProgramLaunchManifest.abi
                         else TensorLaunchManifest)
        manifest = manifest_type.from_dict(documents[role])
        if (manifest.canonical_sha256 != participants[role].get('launch_spec_sha256')
                or manifest.target != participants[role].get('target')
                or manifest.workload_sha256 != raw.get('workload_sha256')
                or manifest.case_id != raw.get('case_id')):
            raise ValueError('MACA event manifest differs from its sealed participant')
        for measurement in raw['measurements']:
            validate_cohort(measurement['arms'][role], manifest,
                            sample_count=protocol.samples_per_cohort,
                            native=raw['kind'] == PAIRED_MACA_NATIVE_EVENT_KIND,
                            torch_reset=raw['kind'] == PAIRED_MACA_TORCH_EVENT_KIND,
                            gated=raw['kind'] == PAIRED_MACA_GATED_EVENT_KIND)


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
        manifest = raw['launch_manifests'][role]
        if manifest.get('abi') == 'ordered_program_v1':
            names = {stage['name'] for stage in manifest['program']['stages']}
            if (values.get('kind') != 'ordered_program' or not isinstance(values.get('stages'), Mapping)
                    or set(values['stages']) != names):
                raise ValueError('MACA Program loaded stage resource coverage differs')
            for stage in values['stages'].values():
                _validate_resource_values(stage)
            continue
        _validate_resource_values(values)
        if values['dynamic_shared_bytes'] != manifest['dynamic_shared_memory_bytes']:
            raise ValueError('MACA event loaded shared memory differs')


def _validate_resource_values(values):
    if not isinstance(values, Mapping) or any(type(values.get(name)) is not int or values[name] < 0
            for name in ('registers_per_thread', 'local_bytes', 'dynamic_shared_bytes')):
        raise ValueError('MACA event loaded resources differ')
