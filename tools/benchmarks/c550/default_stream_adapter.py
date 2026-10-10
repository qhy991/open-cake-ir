"""Component-only complete-launch capture. No production timer registration."""
from __future__ import annotations
import ctypes as C
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

_HELPER = None
_BUILD = None
_RETAIN_UNTIL_EXIT = []
_RESET = C.CFUNCTYPE(C.c_int, C.c_void_p)
_LAUNCH = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_uint)
SYMBOLS = ('mcEventCreateWithFlags', 'mcEventRecord', 'mcEventSynchronize',
           'mcEventElapsedTime', 'mcEventDestroy', 'mcStreamCreateWithFlags',
           'mcStreamDestroy', 'mcStreamWaitEvent', 'mcLaunchHostFunc', 'mcStreamSynchronize')


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
    compiler = shutil.which('c++')
    if compiler is None:
        raise ValueError('complete-launch gate requires a host C++ compiler')
    directory = tempfile.TemporaryDirectory(prefix='cake-default-stream-gate-')
    output = Path(directory.name) / 'gate.so'
    environment = {**os.environ, **{key: '' for key in (
        'CUDA_VISIBLE_DEVICES','MACA_VISIBLE_DEVICES','MCR_VISIBLE_DEVICES',
        'HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES')}}
    result = subprocess.run([compiler,'-std=c++17','-O2','-shared','-fPIC','-pthread',
        str(Path(__file__).with_name('default_stream_gate.cc')),
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
                or admission.device_name not in target.device_names):
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
        if (self._failed or _RETAIN_UNTIL_EXIT or loaded.loaded.closed
                or loaded.manifest.as_dict() != self.manifest.as_dict()
                or loaded.loaded.manifest.as_dict() != self.manifest.as_dict()
                or dry_run_iters != 11 or repeat_iters != 5 or len(arguments) != 16
                or len({id(value) for value in arguments}) != 16):
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
        values = (C.c_float * 5)()
        completed, phase, cleanup, drained = C.c_uint(), C.c_uint(), C.c_int(), C.c_uint()
        status = _HELPER.cake_default_stream_cohort(self._api, reset_callback, launch_callback,
            None, 11, 5, values, C.byref(completed), C.byref(phase), C.byref(cleanup), C.byref(drained))
        observation = {'kind':'unadmitted_complete_launch_gate_component',
            'performance_qualified':False, 'stream':0, 'profiler_enabled':False,
            'interval':'default_stream_events_after_complete_launch_is_queued',
            'reset':'one_fp32_fill_4x_declared_l2_before_warmups_and_each_sample',
            'status':status, 'phase':phase.value, 'cleanup_status':cleanup.value,
            'drained':bool(drained.value), 'completed_callbacks':completed.value,
            'stage_calls_per_invocation':stage_counts, 'observed_stage_calls':sum(stage_counts),
            'raw_event_slots_ms':[float(value) for value in values],
            'callback_errors':[{'type':type(e).__name__,'message':str(e)} for e in errors]}
        self.last_activity = observation
        if not drained.value:
            self._failed = True
            _RETAIN_UNTIL_EXIT.append((self, loaded, arguments, reset_callback, launch_callback, _HELPER))
            raise CaptureFailure(observation, unsafe_to_release=True)
        if (status or cleanup.value or errors or completed.value != 16 or invoked != 16
                or stage_counts != [self.manifest.kernels_per_call] * 16
                or any(not math.isfinite(value) or value <= 0 for value in values)):
            self._failed = True
            failure = CaptureFailure(observation)
            if errors:
                raise failure from errors[0]
            raise failure
        return list(values)
