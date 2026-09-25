"""One-GPU development oracle for Cake's packed model-width SwiGLU tile.

`prepare` and `verify` use CPU only. `run` requires one broker-issued
exclusive B300-M4 GPU and a prebuilt exact native host wrapper. This proves
neither a complete expert FFN nor EP4 scheduling or performance.
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
import math
import os
from pathlib import Path
import struct
import time


CASES = ('zero_gate', 'mixed_sign')
SOURCE_COMMIT = 'cb72263a6fcf9845325c742eaefb581c28ad5e84'
LABEL = 'cake-weave-model-swiglu-cb72263a'
CUDA_RUNTIME = '/usr/local/cuda-13.1/lib64/libcudart.so'
INPUT_BYTES = 128 * 1536 * 4
OUTPUT_BYTES = 128 * 768 * 2


def document(path: Path) -> dict:
    return json.loads(path.read_text())


def checked(code: int, operation: str) -> None:
    if code:
        raise RuntimeError(f'{operation}: CUDA status {code}')


def prepare(root: Path) -> str:
    assessment = document(root / 'assessment.json')
    requirements = document(root / 'requirements.json')
    schedule = document(root / 'schedule.json')
    commit = assessment['compiler_commit']
    if (commit != SOURCE_COMMIT
            or requirements['target'] != 'sm_103a'
            or requirements['source_language'] != 'cuda_cpp'
            or requirements['argument_order'] != ['up_gate', 'activated']
            or requirements['grid'] != [128, 1, 1]
            or requirements['block'] != [32, 1, 1]
            or [(row['name'], row['dtype'], row['shape'], row['mode'])
                for row in requirements['arguments']] != [
                    ('up_gate', 'fp32', [128, 1536], 'input'),
                    ('activated', 'bf16', [128, 768], 'output')]
            or schedule['target'] != 'sm_103a'
            or schedule['schedule_id'] != 'weave-model-swiglu-bf16-b300'
            or any(row['blocks_lowering'] or row['blocks_acceptance']
                   for row in assessment['findings'])):
        raise ValueError('model SwiGLU source, target or typed ABI differs')
    for name in CASES:
        case = document(root / name / 'case.json')
        if (case['name'] != name or case['input_shape'] != [128, 1536]
                or case['output_shape'] != [128, 768]
                or case['input_dtype'] != 'fp32'
                or case['output_dtype'] != 'bf16'
                or (root / name / 'up_gate.fp32').stat().st_size != INPUT_BYTES
                or (root / name / 'expected.bf16').stat().st_size != OUTPUT_BYTES):
            raise ValueError(f'{name} input/oracle extent differs')
    return commit


def admitted() -> str:
    path = Path(os.environ.get('BROKER_RECEIPT', '/nonexistent'))
    for _ in range(30):
        if path.is_file():
            break
        time.sleep(.1)
    else:
        raise RuntimeError('broker admission receipt missing')
    row = document(path)
    if (row.get('label') != LABEL or row.get('mode') != 'exclusive'
            or row.get('gpu_count') != 1 or len(row.get('gpu_ids', [])) != 1
            or not isinstance(row.get('job_id'), str)):
        raise RuntimeError('one exact exclusive B300 GPU admission differs')
    return row['job_id']


class Runtime:
    def __init__(self, root: Path):
        self.root = root
        self.commit = prepare(root)
        self.job = admitted()
        build = document(root / 'build_report.json')
        if (build['compiler_commit'] != self.commit
                or build['target'] != 'sm_103a'
                or build['exit_codes'] != {'host_wrapper': 0, 'cubin': 0}
                or (root / 'kernel.so').read_bytes()[:4] != b'\x7fELF'
                or (root / 'kernel.cubin').read_bytes()[:4] != b'\x7fELF'):
            raise RuntimeError('compiled model SwiGLU products differ')
        self.cuda = C.CDLL(CUDA_RUNTIME, mode=C.RTLD_GLOBAL)
        cuda = self.cuda
        cuda.cudaGetDeviceCount.argtypes = [C.POINTER(C.c_int)]
        cuda.cudaMalloc.argtypes = [C.POINTER(C.c_void_p), C.c_size_t]
        cuda.cudaFree.argtypes = [C.c_void_p]
        cuda.cudaMemcpy.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int]
        cuda.cudaMemset.argtypes = [C.c_void_p, C.c_int, C.c_size_t]
        cuda.cudaDeviceSynchronize.argtypes = []
        count = C.c_int()
        checked(cuda.cudaGetDeviceCount(C.byref(count)), 'device count')
        if count.value != 1:
            raise RuntimeError('broker must expose one logical GPU')
        self.library = C.CDLL(str((root / 'kernel.so').resolve()))
        abi = document(root / 'requirements.json')['host_abi']
        self.create = getattr(self.library, abi['create'])
        self.create.argtypes = [C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
        self.create.restype = C.c_int
        self.launch = getattr(self.library, abi['launch'])
        self.launch.argtypes = [C.c_void_p, C.c_void_p]
        self.launch.restype = C.c_int
        self.destroy = getattr(self.library, abi['destroy'])
        self.destroy.argtypes = [C.c_void_p]
        self.destroy.restype = C.c_int
        self.buffers = []

    def allocate(self, nbytes: int) -> C.c_void_p:
        pointer = C.c_void_p()
        checked(self.cuda.cudaMalloc(C.byref(pointer), nbytes), 'cudaMalloc')
        self.buffers.append(pointer)
        return pointer

    def write(self, pointer: C.c_void_p, data: bytes) -> None:
        host = C.create_string_buffer(data)
        checked(self.cuda.cudaMemcpy(pointer, C.cast(host, C.c_void_p),
                                     len(data), 1), 'input H2D')

    def read(self, pointer: C.c_void_p, nbytes: int) -> bytes:
        host = C.create_string_buffer(nbytes)
        checked(self.cuda.cudaMemcpy(C.cast(host, C.c_void_p), pointer,
                                     nbytes, 2), 'output D2H')
        return host.raw

    def run(self) -> None:
        if (self.root / 'device.json').exists():
            raise ValueError('device evidence must be create-only')
        output = self.root / 'device_outputs'
        output.mkdir(exist_ok=False)
        input_ptr = self.allocate(INPUT_BYTES)
        output_ptr = self.allocate(OUTPUT_BYTES)
        calls = 0
        for name in CASES:
            case = self.root / name
            retained = output / name
            retained.mkdir()
            self.write(input_ptr, (case / 'up_gate.fp32').read_bytes())
            checked(self.cuda.cudaMemset(output_ptr, 0, OUTPUT_BYTES),
                    'output reset')
            checked(self.cuda.cudaDeviceSynchronize(), 'input/reset completion')
            handle = C.c_void_p()
            arguments = (C.c_void_p * 2)(input_ptr.value, output_ptr.value)
            checked(self.create(arguments, C.byref(handle)), 'native create')
            try:
                checked(self.launch(handle, C.c_void_p()), 'native launch')
                checked(self.cuda.cudaDeviceSynchronize(), 'native completion')
                calls += 1
                (retained / 'actual.bf16').write_bytes(
                    self.read(output_ptr, OUTPUT_BYTES))
                (retained / 'observed_input.fp32').write_bytes(
                    self.read(input_ptr, INPUT_BYTES))
            finally:
                checked(self.destroy(handle), 'native destroy')
        (self.root / 'device.json').write_text(json.dumps({
            'broker_job': self.job, 'compiler_commit': self.commit,
            'target': 'sm_103a', 'cases': list(CASES), 'launch_calls': calls,
            'scope': 'one-GPU model SwiGLU/cast development correctness',
        }, indent=2) + '\n')

    def close(self) -> None:
        for pointer in reversed(self.buffers):
            checked(self.cuda.cudaFree(pointer), 'cudaFree')


def device_run(root: Path) -> None:
    runtime = Runtime(root)
    try:
        runtime.run()
    except Exception as error:
        (root / 'launch_failure.json').write_text(json.dumps({
            'broker_job': runtime.job, 'error': f'{type(error).__name__}: {error}'
        }, indent=2) + '\n')
        raise
    finally:
        runtime.close()


def bf16_value(bits: int) -> float:
    return struct.unpack('<f', struct.pack('<I', bits << 16))[0]


def verify(root: Path) -> None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('CPU oracle must run after broker lease release')
    if (root / 'report.json').exists():
        raise ValueError('oracle report must be create-only')
    commit = prepare(root)
    device = document(root / 'device.json')
    receipt = document(root / 'gpuq-admission.json')
    if (device['compiler_commit'] != commit
            or device['broker_job'] != receipt['job_id']
            or device['cases'] != list(CASES)
            or device['launch_calls'] != len(CASES)):
        raise ValueError('device/Compiler/broker case binding differs')
    outcomes = []
    for name in CASES:
        case = root / name
        retained = root / 'device_outputs' / name
        actual = (retained / 'actual.bf16').read_bytes()
        expected = (case / 'expected.bf16').read_bytes()
        if len(actual) != OUTPUT_BYTES or len(expected) != OUTPUT_BYTES:
            raise ValueError(f'{name} output extent differs')
        exact = actual == expected
        mismatches, nonfinite, signed_zero_differences = 0, 0, 0
        max_abs_error = 0.0
        for (actual_bits,), (expected_bits,) in zip(
                struct.iter_unpack('<H', actual),
                struct.iter_unpack('<H', expected), strict=True):
            observed, wanted = bf16_value(actual_bits), bf16_value(expected_bits)
            if not math.isfinite(observed):
                nonfinite += 1
                continue
            error = abs(observed - wanted)
            max_abs_error = max(max_abs_error, error)
            mismatches += observed != wanted
            signed_zero_differences += (actual_bits != expected_bits
                                        and observed == wanted == 0.0)
        unchanged = ((retained / 'observed_input.fp32').read_bytes()
                     == (case / 'up_gate.fp32').read_bytes())
        outcomes.append({'case': name,
                         'passed': not mismatches and not nonfinite and unchanged,
                         'bitwise_equal': exact, 'signed_zero_differences': signed_zero_differences,
                         'inputs_unchanged': unchanged,
                         'mismatches': mismatches, 'nonfinite': nonfinite,
                         'max_abs_error': max_abs_error})
    report = {'compiler_commit': commit, 'broker_job': device['broker_job'],
              'target': 'sm_103a', 'cases': outcomes,
              'passed': all(row['passed'] for row in outcomes),
              'scope': 'one SwiGLU/BF16 cast tile; no complete FFN, EP4, latency or serving claim'}
    (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'run', 'verify'))
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve(strict=True)
    {'prepare': prepare, 'run': device_run, 'verify': verify}[
        args.phase](root)


if __name__ == '__main__':
    main()
