"""Standalone one-GPU correctness probe for Cake B300 expert GEMM tiles.

`prepare` and `verify` are CPU-only. `run` requires one exclusive B300-M4
broker GPU and the nvcc-built host wrapper in the external evidence directory.
This validates one projection tile, not a ranked MoE layer or performance.
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


CASES = ('dyadic_random', 'zero_rows')
CUDA_RUNTIME = '/usr/local/cuda-13.1/lib64/libcudart.so'
TILES = {
    'weave-model-upgate-tile-b300': {
        'tag': 'upgate', 'grid': [1, 24, 1],
        'a': [128, 2048], 'b': [1536, 2048], 'c': [128, 1536],
    },
    'weave-model-down-tile-b300': {
        'tag': 'down', 'grid': [1, 32, 1],
        'a': [128, 768], 'b': [2048, 768], 'c': [128, 2048],
    },
}


def document(path: Path) -> dict:
    return json.loads(path.read_text())


def checked(code: int, operation: str) -> None:
    if code:
        raise RuntimeError(f'{operation}: CUDA status {code}')


def prepare(root: Path) -> dict:
    assessment = document(root / 'assessment.json')
    requirements = document(root / 'requirements.json')
    schedule = document(root / 'schedule.json')
    commit = assessment['compiler_commit']
    geometry = TILES.get(schedule.get('schedule_id'))
    if geometry is None:
        raise ValueError('unknown model projection tile')
    label = f'cake-weave-model-{geometry["tag"]}-tile-{commit[:8]}'
    if (len(commit) != 40 or any(char not in '0123456789abcdef' for char in commit)
            or requirements['target'] != 'sm_103a'
            or requirements['source_language'] != 'cuda_cpp'
            or requirements['argument_order'] != ['a', 'b', 'c']
            or requirements['grid'] != geometry['grid']
            or requirements['block'] != [192, 1, 1]
            or requirements['dynamic_shared_bytes'] != 49200
            or [(row['name'], row['dtype'], row['shape'], row['mode'])
                for row in requirements['arguments']] != [
                    ('a', 'bf16', geometry['a'], 'input'),
                    ('b', 'bf16', geometry['b'], 'input'),
                    ('c', 'fp32', geometry['c'], 'output')]
            or schedule['target'] != 'sm_103a'
            or any(row['blocks_lowering'] or row['blocks_acceptance']
                   for row in assessment['findings'])):
        raise ValueError('model projection tile source, target or typed ABI differs')
    extents = {'a.bf16': math.prod(geometry['a']) * 2,
               'b.bf16': math.prod(geometry['b']) * 2,
               'expected.fp32': math.prod(geometry['c']) * 4}
    for name in CASES:
        case = document(root / name / 'case.json')
        if (case['name'] != name or case['a_shape'] != geometry['a']
                or case['b_shape'] != geometry['b']
                or case['output_shape'] != geometry['c']
                or case['input_dtype'] != 'bf16'
                or case['output_dtype'] != 'fp32'
                or case['expected_exact_fp32'] is not True):
            raise ValueError(f'{name} input/oracle declaration differs')
        for filename, expected in extents.items():
            if (root / name / filename).stat().st_size != expected:
                raise ValueError(f'{name}/{filename} byte extent differs')
    return {'compiler_commit': commit, 'requirements': requirements,
            'geometry': geometry, 'extents': extents, 'label': label}


def admitted(expected_label: str) -> str:
    path = Path(os.environ.get('BROKER_RECEIPT', '/nonexistent'))
    for _ in range(30):
        if path.is_file():
            break
        time.sleep(.1)
    else:
        raise RuntimeError('broker admission receipt missing')
    row = document(path)
    if (row.get('label') != expected_label or row.get('mode') != 'exclusive'
            or row.get('gpu_count') != 1 or len(row.get('gpu_ids', [])) != 1
            or not isinstance(row.get('job_id'), str)):
        raise RuntimeError('one exact exclusive B300 GPU admission differs')
    return row['job_id']


class Runtime:
    def __init__(self, root: Path):
        self.root = root
        binding = prepare(root)
        self.job = admitted(binding['label'])
        self.binding = binding
        self.commit = binding['compiler_commit']
        self.req = binding['requirements']
        build = document(root / 'build_report.json')
        if (build['compiler_commit'] != self.commit
                or build['target'] != 'sm_103a'
                or build['exit_codes'] != {'host_wrapper': 0, 'cubin': 0}
                or (root / 'kernel.so').read_bytes()[:4] != b'\x7fELF'
                or (root / 'kernel.cubin').read_bytes()[:4] != b'\x7fELF'):
            raise RuntimeError('compiled host wrapper or cubin differs')
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
            raise RuntimeError('broker must expose exactly one logical GPU')
        self.library = C.CDLL(str((root / 'kernel.so').resolve()))
        abi = self.req['host_abi']
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
        extents = self.binding['extents']
        a = self.allocate(extents['a.bf16'])
        b = self.allocate(extents['b.bf16'])
        c = self.allocate(extents['expected.fp32'])
        calls = 0
        for name in CASES:
            case = self.root / name
            retained = output / name
            retained.mkdir()
            self.write(a, (case / 'a.bf16').read_bytes())
            self.write(b, (case / 'b.bf16').read_bytes())
            checked(self.cuda.cudaMemset(c, 0, extents['expected.fp32']),
                    'output reset')
            checked(self.cuda.cudaDeviceSynchronize(), 'input/reset completion')
            handle = C.c_void_p()
            arguments = (C.c_void_p * 3)(a.value, b.value, c.value)
            checked(self.create(arguments, C.byref(handle)), 'native create')
            try:
                checked(self.launch(handle, C.c_void_p()), 'native launch')
                checked(self.cuda.cudaDeviceSynchronize(), 'native completion')
                calls += 1
                (retained / 'actual.fp32').write_bytes(
                    self.read(c, extents['expected.fp32']))
                (retained / 'observed_a.bf16').write_bytes(
                    self.read(a, extents['a.bf16']))
                (retained / 'observed_b.bf16').write_bytes(
                    self.read(b, extents['b.bf16']))
            finally:
                checked(self.destroy(handle), 'native destroy')
        (self.root / 'device.json').write_text(json.dumps({
            'broker_job': self.job, 'compiler_commit': self.commit,
            'target': 'sm_103a', 'tile': self.binding['geometry']['tag'],
            'cases': list(CASES), 'launch_calls': calls,
            'scope': 'one-GPU B300 native tensor-tile development correctness',
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


def verify(root: Path) -> None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('CPU oracle must run after broker lease release')
    if (root / 'report.json').exists():
        raise ValueError('oracle report must be create-only')
    binding = prepare(root)
    device = document(root / 'device.json')
    receipt = document(root / 'gpuq-admission.json')
    if (device['compiler_commit'] != binding['compiler_commit']
            or device['broker_job'] != receipt['job_id']
            or device['tile'] != binding['geometry']['tag']
            or device['cases'] != list(CASES)
            or device['launch_calls'] != len(CASES)):
        raise ValueError('device/Compiler/broker case binding differs')
    outcomes = []
    for name in CASES:
        case = root / name
        retained = root / 'device_outputs' / name
        actual_bytes = (retained / 'actual.fp32').read_bytes()
        expected_bytes = (case / 'expected.fp32').read_bytes()
        if (len(actual_bytes) != len(expected_bytes)
                or len(actual_bytes) != binding['extents']['expected.fp32']):
            raise ValueError(f'{name} output extent differs')
        exact = actual_bytes == expected_bytes
        mismatches = 0
        nonfinite = 0
        max_abs_error = 0.0
        for (actual,), (expected,) in zip(
                struct.iter_unpack('<f', actual_bytes),
                struct.iter_unpack('<f', expected_bytes), strict=True):
            if not math.isfinite(actual):
                nonfinite += 1
                continue
            error = abs(actual - expected)
            max_abs_error = max(max_abs_error, error)
            mismatches += error > 1e-6
        unchanged = ((retained / 'observed_a.bf16').read_bytes()
                     == (case / 'a.bf16').read_bytes()
                     and (retained / 'observed_b.bf16').read_bytes()
                     == (case / 'b.bf16').read_bytes())
        outcomes.append({'case': name,
                         'passed': not mismatches and not nonfinite and unchanged,
                         'bitwise_equal': exact, 'inputs_unchanged': unchanged,
                         'mismatches': mismatches, 'nonfinite': nonfinite,
                         'max_abs_error': max_abs_error})
    report = {'compiler_commit': binding['compiler_commit'],
              'broker_job': device['broker_job'], 'target': 'sm_103a',
              'tile': binding['geometry']['tag'],
              'cases': outcomes, 'passed': all(row['passed'] for row in outcomes),
              'scope': 'one expert projection tile; no full FFN, EP4, latency or serving claim'}
    (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'run', 'verify'))
    parser.add_argument('--root', type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.root.expanduser().resolve(strict=True)
    {'prepare': prepare, 'run': device_run, 'verify': verify}[
        arguments.phase](root)


if __name__ == '__main__':
    main()
