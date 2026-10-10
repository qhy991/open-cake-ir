"""Complete-launch queued event capture; production registration is separate."""
from __future__ import annotations
import ctypes as C
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from collections.abc import Mapping

from .loaders import UndrainedDeviceWork
from open_cake_ir.serialization import canonical_json_bytes

_HELPER = None
_BUILD = None
_RETAIN_UNTIL_EXIT = []
_RESET = C.CFUNCTYPE(C.c_int, C.c_void_p)
_LAUNCH = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_uint)
SYMBOLS = ('mcEventCreateWithFlags', 'mcEventRecord', 'mcEventSynchronize',
           'mcEventElapsedTime', 'mcEventDestroy', 'mcStreamCreateWithFlags',
           'mcStreamDestroy', 'mcStreamWaitEvent', 'mcLaunchHostFunc', 'mcStreamSynchronize')
TIMER = 'maca_fully_queued_default_stream_event_elapsed_ms'
INTERVAL = 'default_stream_events_after_complete_launch_is_queued'
RESET = 'one_fp32_fill_4x_declared_l2_before_warmups_and_each_sample'


def _event_descriptor(manifest, target):
    from .program import ProgramLaunchManifest
    return {'kind':'maca_queued_event_samples_v1', 'timer':TIMER, 'interval':INTERVAL,
        'cache_policy':RESET, 'profiler_enabled':False, 'target':manifest.target,
        'device':0, 'stream':0, 'l2_cache_bytes':target.l2_cache_bytes,
        'reset_bytes':4*target.l2_cache_bytes, 'warmup_calls':11,
        'stage_names':([stage.name for stage in manifest.program.stages]
                       if isinstance(manifest,ProgramLaunchManifest) else [manifest.kernel_name])}


class CaptureFailure(RuntimeError):
    def __init__(self, observation, *, unsafe_to_release=False):
        super().__init__('complete-launch gate failed; retain component observations')
        self.observation = observation
        self.unsafe_to_release = unsafe_to_release


def prepare_helper():
    """Build host-only code before allocation. Never load the device runtime here."""
    global _HELPER, _BUILD
    if _HELPER is not None:
        return _HELPER
    if os.environ.get('METAL_BROKER_LOCK_FD') or os.environ.get('GPUQ_JOB_ID'):
        raise ValueError('host gate compilation must precede device allocation')
    compiler = shutil.which('c++')
    if compiler is None:
        raise ValueError('complete-launch gate requires a host C++ compiler')
    directory = tempfile.TemporaryDirectory(prefix='cake-default-stream-gate-')
    output = Path(directory.name) / 'gate.so'
    environment = {**os.environ, **{key: '' for key in (
        'CUDA_VISIBLE_DEVICES','MACA_VISIBLE_DEVICES','MCR_VISIBLE_DEVICES',
        'HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES')}}
    result = subprocess.run([compiler,'-std=c++17','-O2','-shared','-fPIC','-pthread',
        str(Path(__file__).with_name('metax_queued_gate.cc')),
        '-o',str(output)], capture_output=True, timeout=60, env=environment)
    if result.returncode:
        directory.cleanup()
        raise ValueError('host gate compilation failed: '+result.stderr.decode(errors='replace')[-2000:])
    library = C.CDLL(str(output))
    function = library.cake_default_stream_cohort
    function.argtypes = [C.POINTER(C.c_void_p), _RESET, _LAUNCH, C.c_void_p,
        C.c_uint,C.c_uint,C.POINTER(C.c_float),C.POINTER(C.c_uint),C.POINTER(C.c_uint),
        C.POINTER(C.c_int),C.POINTER(C.c_uint)]
    function.restype = C.c_int
    _BUILD, _HELPER = directory, library
    return library


class CompleteLaunchCapture:
    """Caller owns exact device admission, physical lock and original oracle.

    An unsafe_to_release failure requires process termination with owners held.
    The component must never be used inside a generic finally-unload path.
    """
    def __init__(self, manifest, admission):
        from open_cake_ir.compiler.target import declared_target
        from open_cake_ir.evaluation.program import ProgramLaunchManifest
        if _HELPER is None:
            raise ValueError('prepare the host gate before device allocation')
        target = declared_target(manifest.target)
        if (target.target_id != 'xcore1002' or admission.target != target.target_id
                or admission.device_arch != target.target_id
                or admission.warp_size != target.warp_size
                or admission.device_name not in target.device_names
                or type(target.l2_cache_bytes) is not int or target.l2_cache_bytes <= 0):
            raise ValueError('complete-launch component requires exact admitted C550')
        if ((isinstance(manifest, ProgramLaunchManifest) and manifest.aligned_stages)
                or (not isinstance(manifest, ProgramLaunchManifest) and manifest.aligned_variant)):
            raise ValueError('component does not admit aligned variants')
        runtime = Path(admission.runtime_library)
        if not runtime.is_absolute():
            raise ValueError('use the admitted absolute MACA runtime path')
        self.manifest, self.target = manifest, target
        self._runtime = C.CDLL(str(runtime))
        self._api = (C.c_void_p * len(SYMBOLS))(*(
            C.cast(getattr(self._runtime, name), C.c_void_p).value for name in SYMBOLS))
        self.last_activity = None
        self._reset = None
        self._failed = False

    def capture_loaded_cohort(self, loaded, arguments, *, dry_run_iters, repeat_iters):
        import torch
        if type(dry_run_iters) is not int or dry_run_iters != 11 or type(repeat_iters) is not int or repeat_iters != 5:
            raise ValueError('complete-launch capture requires eleven warmups and five samples per cohort')
        count = dry_run_iters + repeat_iters
        if (self._failed or _RETAIN_UNTIL_EXIT or loaded.loaded.closed
                or loaded.manifest.as_dict() != self.manifest.as_dict()
                or loaded.loaded.manifest.as_dict() != self.manifest.as_dict()
                or len(arguments) != count
                or len({id(value) for value in arguments}) != count):
            raise ValueError('complete-launch manifest, ownership or cohort differs')
        def check_stream():
            if (not getattr(torch.version, 'maca', None)
                    or torch.cuda.current_stream().cuda_stream != 0
                    or torch.cuda.default_stream(0).cuda_stream != 0
                    or torch.cuda.get_device_properties(0).L2_cache_size != self.target.l2_cache_bytes):
                raise ValueError('component requires the admitted default stream and L2')
        check_stream()
        if self._reset is None:
            self._reset = torch.empty(self.target.l2_cache_bytes, dtype=torch.float32, device='cuda:0')
        errors, stage_counts = [], []
        invoked = 0
        def reset(_):
            try:
                check_stream()
                self._reset.fill_(1.0)
                return 0
            except BaseException as error:
                errors.append(error)
                return -42001
        def launch(_, index):
            nonlocal invoked
            before = loaded.loaded.launch_calls
            try:
                check_stream()
                if index != invoked or index >= len(arguments):
                    raise ValueError('component invocation order differs')
                invoked += 1
                loaded.launch(arguments[index])
                if loaded.loaded.launch_calls - before != self.manifest.kernels_per_call:
                    raise ValueError('complete Program stage count differs')
                return 0
            except BaseException as error:
                errors.append(error)
                return -42002
            finally:
                stage_counts.append(loaded.loaded.launch_calls - before)
        reset_callback, launch_callback = _RESET(reset), _LAUNCH(launch)
        values = (C.c_float * repeat_iters)()
        completed, phase, cleanup, drained = C.c_uint(), C.c_uint(), C.c_int(), C.c_uint()
        owners = (self, loaded, arguments, reset_callback, launch_callback, _HELPER)
        try:
            status = _HELPER.cake_default_stream_cohort(self._api, reset_callback, launch_callback,
                None, dry_run_iters, repeat_iters, values, C.byref(completed), C.byref(phase), C.byref(cleanup), C.byref(drained))
        except BaseException as error:
            self._failed = True
            if not drained.value or isinstance(error, UndrainedDeviceWork):
                _RETAIN_UNTIL_EXIT.append(owners)
                failure = CaptureFailure({'phase':'native_call_interrupted', 'drained':bool(drained.value),
                    'capture_completed':False, 'requires_process_exit':True}, unsafe_to_release=True)
                self.last_activity = failure.observation
                raise failure from error
            raise
        terminal = next((error for error in errors if isinstance(error, UndrainedDeviceWork)), None)
        unsafe = not drained.value or terminal is not None
        if unsafe:
            self._failed = True
            _RETAIN_UNTIL_EXIT.append(owners)
        def message(error):
            try:
                return str(error)
            except BaseException:
                return '<exception message unavailable>'
        observation = {'kind':'unadmitted_complete_launch_gate_component',
            'performance_qualified':False, 'stream':0, 'profiler_enabled':False,
            'interval':INTERVAL, 'reset':RESET,
            'status':status, 'phase':phase.value, 'cleanup_status':cleanup.value,
            'drained':bool(drained.value), 'completed_callbacks':completed.value,
            'stage_calls_per_invocation':stage_counts, 'observed_stage_calls':sum(stage_counts),
            'warmup_calls':dry_run_iters, 'sample_count':repeat_iters,
            'raw_event_slots_ms':[float(value) if math.isfinite(value) else str(float(value)) for value in values],
            'callback_errors':[{'type':type(e).__name__,'message':message(e)} for e in errors]}
        complete = bool(drained.value and not status and not cleanup.value and not errors
            and completed.value == count and invoked == count
            and stage_counts == [self.manifest.kernels_per_call] * count
            and all(math.isfinite(value) and value > 0 for value in values))
        observation.update(capture_completed=complete, requires_process_exit=unsafe)
        self.last_activity = observation
        if unsafe:
            raise CaptureFailure(observation, unsafe_to_release=True) from terminal
        if not complete:
            self._failed = True
            failure = CaptureFailure(observation)
            if errors:
                raise failure from errors[0]
            raise failure
        return list(values)


class MacaQueuedEventBenchmark(CompleteLaunchCapture):
    """Five observations per AB/BA cohort; the paired mean uses ten per arm.

    This adapter is not a production policy registration or performance gate.
    The caller prepares the helper before allocation and owns the original oracle.
    """
    def capture_loaded_cohort(self, loaded, arguments, *, dry_run_iters, repeat_iters):
        if type(repeat_iters) is not int or repeat_iters != 5:
            raise ValueError('queued measurement requires five samples per AB/BA cohort')
        try:
            samples = super().capture_loaded_cohort(loaded, arguments,
                dry_run_iters=dry_run_iters, repeat_iters=repeat_iters)
        except CaptureFailure as failure:
            if not failure.unsafe_to_release:
                raise
            terminal = (failure.__cause__ if isinstance(failure.__cause__, UndrainedDeviceWork)
                        else UndrainedDeviceWork('queued capture could not establish safe device ownership'))
            terminal.retain(self, loaded, arguments, failure)
            try:
                terminal.artifact_payloads['native_observation'] = canonical_json_bytes(failure.observation)
            except BaseException:
                terminal.artifact_payloads['native_observation'] = b'{"diagnostic_serialization_failed":true,"requires_process_exit":true}'
            raise terminal from None
        capture = self.last_activity
        self.last_activity = {**_event_descriptor(self.manifest,self.target), 'capture':capture,
            'samples':[{'index':i,'elapsed_ms':value,'reset_enqueued_before_start':True,
                        'end_synchronized':True} for i,value in enumerate(samples)]}
        return samples


def validate_cohort(record, manifest, *, sample_count):
    """Keep this interval distinct from the older submission-gap observations."""
    from open_cake_ir.compiler.target import declared_target
    from .program import ProgramLaunchManifest
    from .metax_event_benchmark import _validate_samples
    target=declared_target(manifest.target)
    if (manifest.target!='xcore1002' or type(target.l2_cache_bytes) is not int
            or target.l2_cache_bytes<=0):
        raise ValueError('queued events require exact C550 and its declared L2')
    expected=_event_descriptor(manifest,target)
    raw=record.get('native_activity')
    if (sample_count!=5 or manifest.target!='xcore1002'
            or (manifest.aligned_stages if isinstance(manifest,ProgramLaunchManifest) else manifest.aligned_variant)
            or not isinstance(raw,Mapping) or set(raw)!=set(expected)|{'capture','samples'}
            or any(raw.get(key)!=value or type(raw.get(key)) is not type(value) for key,value in expected.items())
            or record.get('non_target_dispatches') is not None):
        raise ValueError('queued event interval, target, reset or complete-launch binding differs')
    _validate_samples(record,sample_count)
    capture=raw['capture']
    fixed={'kind':'unadmitted_complete_launch_gate_component','performance_qualified':False,
        'stream':0,'profiler_enabled':False,'interval':INTERVAL,'reset':RESET,
        'status':0,'phase':20,'cleanup_status':0,'drained':True,
        'completed_callbacks':16,'warmup_calls':11,'sample_count':5,
        'observed_stage_calls':16*manifest.kernels_per_call,
        'capture_completed':True,'requires_process_exit':False}
    if (not isinstance(capture,Mapping)
            or any(capture.get(key)!=value or type(capture.get(key)) is not type(value) for key,value in fixed.items())
            or capture.get('callback_errors')!=[]
            or capture.get('stage_calls_per_invocation')!=[manifest.kernels_per_call]*16
            or any(type(value) is not int for value in capture.get('stage_calls_per_invocation',[]))
            or capture.get('raw_event_slots_ms')!=record['samples_ms']):
        raise ValueError('queued event native completion, samples or stage accounting differs')
