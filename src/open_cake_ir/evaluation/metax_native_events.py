"""CPU-built native event submission for the exact loaded MACA dispatch."""
from __future__ import annotations

import ctypes as C
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading

_HELPER = None
_DIRECTORY = None
_LOCK = threading.Lock()
SYMBOLS = ('mcEventCreateWithFlags', 'mcEventRecord', 'mcEventSynchronize',
           'mcEventElapsedTime', 'mcEventDestroy', 'mcMemsetD32Async',
           'mcModuleLaunchKernel', 'mcStreamSynchronize')
GATED_SYMBOLS = SYMBOLS + ('mcStreamCreateWithFlags','mcStreamDestroy',
                          'mcStreamWaitEvent','mcLaunchHostFunc')



def prepare_helper():
    """Compile trusted host-only code before a device lease; no runtime is loaded."""
    global _HELPER, _DIRECTORY
    with _LOCK:
        if _HELPER is not None:
            return _HELPER
        compiler = shutil.which('c++')
        if compiler is None:
            raise ValueError('MACA native event submission requires a host C++ compiler')
        directory = tempfile.TemporaryDirectory(prefix='cake-maca-event-host-')
        source = Path(__file__).with_suffix('.cc')
        output = Path(directory.name) / 'cohort.so'
        result = subprocess.run([compiler, '-std=c++17', '-O2', '-shared', '-fPIC',
                                 str(source), '-o', str(output)], capture_output=True,
                                timeout=60, env=_cpu_environment())
        if result.returncode:
            directory.cleanup()
            raise ValueError('MACA native event host compilation failed: '
                             + result.stderr.decode(errors='replace')[-2000:])
        library = C.CDLL(str(output))
        function = library.cake_maca_event_cohort
        function.restype = C.c_int
        function.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.POINTER(C.c_uint), C.c_uint,
                            C.POINTER(C.POINTER(C.c_void_p)), C.c_uint, C.c_uint, C.c_void_p,
                            C.c_size_t, C.POINTER(C.c_float), C.POINTER(C.c_uint), C.POINTER(C.c_uint)]
        library.cake_maca_gated_event_cohort.restype = function.restype
        library.cake_maca_gated_event_cohort.argtypes = function.argtypes
        _DIRECTORY, _HELPER = directory, library
        return library


def _cpu_environment():
    import os
    return {**os.environ, **{key: '' for key in
        ('CUDA_VISIBLE_DEVICES', 'MACA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')}}


def capture(loaded, argument_sets, reset, *, warmups, samples, torch_reset=False, gated=False):
    """Validate and pack every distinct tensor set before the first timed event."""
    from .metax_driver import LoadedMetaxCandidate
    if (type(loaded.loaded) is not LoadedMetaxCandidate or loaded.loaded.closed
            or len(argument_sets) != warmups + samples or warmups != 11 or samples != 5):
        raise ValueError('Native MACA event cohort requires one admitted non-variant dispatch')
    if _HELPER is None:
        raise ValueError('Native event helper was not built in the CPU phase')
    kernel = loaded.loaded
    packed = [kernel.prepare_arguments(args, tensor_contract=loaded.manifest)
              for args in argument_sets]
    pointers = (C.POINTER(C.c_void_p) * len(packed))(*(row[1] for row in packed))
    if gated and not torch_reset:
        raise ValueError('Gated MACA events require the explicit Torch reset')
    symbols = GATED_SYMBOLS if gated else SYMBOLS
    addresses = [C.cast(getattr(kernel._api, name), C.c_void_p).value for name in symbols]
    reset_errors = []
    reset_callback = None
    if torch_reset:
        Reset = C.CFUNCTYPE(C.c_int, C.c_size_t, C.c_uint, C.c_size_t, C.c_void_p)
        def fill(address, value, words, stream):
            try:
                if address != reset.data_ptr() or words != reset.numel() or value != 0x3f800000 or (stream and not gated):
                    raise ValueError('Torch reset buffer or default stream differs')
                import torch
                selected = (torch.cuda.ExternalStream(stream,device=0)
                            if gated else torch.cuda.default_stream(0))
                with torch.cuda.stream(selected):
                    reset.fill_(1.0)
                    reset.fill_(1.0)
                return 0
            except BaseException as error:
                reset_errors.append(str(error)); return -20001
        reset_callback = Reset(fill)
        addresses[5] = C.cast(reset_callback, C.c_void_p).value
    api = (C.c_void_p * len(addresses))(*addresses)
    dimensions = (C.c_uint * 6)(*loaded.manifest.grid, *loaded.manifest.block)
    elapsed, calls, phase = (C.c_float * samples)(), C.c_uint(), C.c_uint()
    entry = _HELPER.cake_maca_gated_event_cohort if gated else _HELPER.cake_maca_event_cohort
    status = entry(api, kernel._function, dimensions,
        loaded.manifest.dynamic_shared_memory_bytes, pointers, warmups, samples,
        reset.data_ptr(), reset.numel(), elapsed, C.byref(calls), C.byref(phase))
    kernel.launch_calls += calls.value
    if status:
        error = RuntimeError(f'MACA native event cohort phase {phase.value} failed with '
                             f'status {status}; completed target calls {calls.value}')
        error.native_observations = {'status': status, 'phase': phase.value,
            'completed_target_calls': calls.value,
            'elapsed_slots_ms': [float(value) for value in elapsed],
            'coverage': 'partial_native_capture_not_a_valid_timing_receipt', 'reset_errors':reset_errors}
        raise error
    if calls.value != len(argument_sets):
        raise ValueError('MACA native event target-call count differs')
    return [float(value) for value in elapsed]
