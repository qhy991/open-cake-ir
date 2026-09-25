"""One-GPU ordered Cake expert FFN development oracle on B300-M4.

`prepare` and `verify` are CPU-only. `run` requires one broker-issued
exclusive GPU and three prebuilt Cake native host wrappers. It exercises
up/gate, explicit SwiGLU→BF16, and down in order; no dynamic expert routing,
four-rank mailbox, kernel fusion or performance claim follows.
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import time


SOURCE_COMMIT = '984318739f135354f3ce03741d25904940d93f86'
LABEL = 'cake-weave-model-local-ffn-98431873'
CUDA_RUNTIME = '/usr/local/cuda-13.1/lib64/libcudart.so'
NVCC = '/usr/local/cuda-13.1/bin/nvcc'
NVCC_FLAGS = ['-std=c++17', '--gpu-architecture=compute_103a',
              '--gpu-code=sm_103a', '-O3', '--fmad=false', '-lineinfo',
              '-Xptxas=-v']
CASES = ('sparse_one_hot', 'dense_dyadic')
TENSORS = {
    'x': ([128, 2048], 'bf16'),
    'w_up_gate': ([1536, 2048], 'bf16'),
    'w_down': ([2048, 768], 'bf16'),
    'up_gate': ([128, 1536], 'fp32'),
    'activated': ([128, 768], 'bf16'),
    'out': ([128, 2048], 'fp32'),
}
STAGES = (
    ('up_gate', {'a': 'x', 'b': 'w_up_gate', 'c': 'up_gate'},
     [1, 24, 1], [192, 1, 1], 49200),
    ('activation', {'up_gate': 'up_gate', 'activated': 'activated'},
     [128, 1, 1], [32, 1, 1], 0),
    ('down', {'a': 'activated', 'b': 'w_down', 'c': 'out'},
     [1, 32, 1], [192, 1, 1], 49200),
)
FILES = {'x': 'x.bf16', 'w_up_gate': 'w_up_gate.bf16',
         'w_down': 'w_down.bf16'}
EXPECTED = {'up_gate': 'expected_up_gate.fp32',
            'activated': 'expected_activated.bf16',
            'out': 'expected_out.fp32'}


def document(path: Path) -> dict:
    return json.loads(path.read_text())


def nbytes(name: str) -> int:
    shape, dtype = TENSORS[name]
    return math.prod(shape) * (2 if dtype == 'bf16' else 4)


def checked(code: int, operation: str) -> None:
    if code:
        raise RuntimeError(f'{operation}: CUDA status {code}')


def prepare(root: Path) -> dict:
    program = document(root / 'program.json')
    if ((root / 'compiler_revision_id.txt').read_text().strip()
            != 'open-cake-ir@' + SOURCE_COMMIT
            or program['program_id'] != 'weave-model-local-expert-ffn-native-b300'
            or program['target'] != 'sm_103a'
            or program['inputs'] != list(FILES)
            or program['outputs'] != ['out']
            or program['tensors'] != {
                name: {'shape': shape, 'dtype': dtype}
                for name, (shape, dtype) in TENSORS.items()}
            or [stage['name'] for stage in program['stages']]
            != [stage[0] for stage in STAGES]):
        raise ValueError('complete local FFN Program or Compiler identity differs')
    requirements = {}
    for name, bindings, grid, block, shared in STAGES:
        stage = next(stage for stage in program['stages'] if stage['name'] == name)
        row = document(root / name / 'requirements.json')
        if (stage['bindings'] != bindings
                or row['target'] != 'sm_103a'
                or row['source_language'] != 'cuda_cpp'
                or row['grid'] != grid or row['block'] != block
                or row['dynamic_shared_bytes'] != shared
                or row['nvcc_flags'] != NVCC_FLAGS
                or row['link_libraries'] != ['cuda', 'cudart']
                or row['argument_order'] != list(bindings)
                or [(arg['name'], arg['dtype'], arg['shape'])
                    for arg in row['arguments']] != [
                    (local, TENSORS[tensor][1], TENSORS[tensor][0])
                    for local, tensor in bindings.items()]
                or not (root / name / 'kernel.cu').is_file()):
            raise ValueError(f'{name} stage source or tensor ABI differs')
        requirements[name] = row
    for name in CASES:
        case = document(root / name / 'case.json')
        if (case['name'] != name or case['shape'] != {'M': 128, 'H': 2048, 'I': 768}
                or case['input_dtype'] != 'bf16'
                or case['up_gate_dtype'] != 'fp32'
                or case['activation_dtype'] != 'bf16'
                or case['output_dtype'] != 'fp32'):
            raise ValueError(f'{name} CPU input/oracle contract differs')
        for tensor, filename in FILES.items():
            if (root / name / filename).stat().st_size != nbytes(tensor):
                raise ValueError(f'{name}/{filename} byte extent differs')
        for tensor, filename in EXPECTED.items():
            if (root / name / filename).stat().st_size != nbytes(tensor):
                raise ValueError(f'{name}/{filename} oracle extent differs')
    return requirements


def build(root: Path) -> None:
    """Compile the three sealed stage sources without acquiring a GPU."""
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('nvcc build must run outside a broker GPU lease')
    requirements = prepare(root)
    compiler = Path(NVCC)
    if not compiler.is_file() or not os.access(compiler, os.X_OK):
        raise FileNotFoundError(f'exact CUDA 13.1 nvcc missing: {compiler}')
    for name, _, _, _, _ in STAGES:
        stage = root / name
        if any((stage / filename).exists() for filename in (
                'kernel.so', 'kernel.cubin', 'compile_plan.json',
                'build_report.json', 'build_failure.json')):
            raise ValueError(f'{name} build evidence must be create-only')
    for name, _, _, _, _ in STAGES:
        stage = root / name
        common = [str(compiler), *requirements[name]['nvcc_flags']]
        commands = {
            'host_wrapper': [*common, '-Xcompiler=-fPIC', '-shared',
                             'kernel.cu', '-o', 'kernel.so', '-lcuda', '-lcudart'],
            'cubin': [*common, '--cubin', 'kernel.cu', '-o', 'kernel.cubin'],
        }
        (stage / 'compile_plan.json').write_text(json.dumps(
            {'compiler_commit': SOURCE_COMMIT, 'target': 'sm_103a',
             'commands': commands}, indent=2) + '\n')
        exits = {}
        for phase, command in commands.items():
            try:
                result = subprocess.run(command, cwd=stage, capture_output=True,
                                        text=True, timeout=180, check=False)
            except (OSError, subprocess.TimeoutExpired) as error:
                (stage / 'build_failure.json').write_text(json.dumps({
                    'phase': phase, 'error': f'{type(error).__name__}: {error}',
                    'exit_codes': exits}, indent=2) + '\n')
                raise RuntimeError(f'{name} {phase} nvcc invocation failed') from error
            (stage / f'{phase}.compile.log').write_text(
                result.stdout + result.stderr or '(no compiler output)\n')
            exits[phase] = result.returncode
            if result.returncode:
                (stage / 'build_failure.json').write_text(json.dumps({
                    'phase': phase, 'exit_codes': exits}, indent=2) + '\n')
                raise RuntimeError(f'{name} {phase} nvcc rejected source')
        if any(not (stage / filename).is_file()
               or (stage / filename).read_bytes()[:4] != b'\x7fELF'
               for filename in ('kernel.so', 'kernel.cubin')):
            (stage / 'build_failure.json').write_text(json.dumps({
                'error': 'missing or non-ELF compiler product',
                'exit_codes': exits}, indent=2) + '\n')
            raise RuntimeError(f'{name} compiler product differs')
        (stage / 'build_report.json').write_text(json.dumps({
            'compiler_commit': SOURCE_COMMIT, 'target': 'sm_103a',
            'exit_codes': exits,
            'scope': 'CPU nvcc/PTXAS build only; no GPU correctness or performance',
        }, indent=2) + '\n')


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


class NativeStage:
    def __init__(self, root: Path, name: str, requirements: dict,
                 bindings: dict[str, str], pointers: dict[str, C.c_void_p]):
        build = document(root / name / 'build_report.json')
        if (build['compiler_commit'] != SOURCE_COMMIT
                or build['target'] != 'sm_103a'
                or build['exit_codes'] != {'host_wrapper': 0, 'cubin': 0}
                or (root / name / 'kernel.so').read_bytes()[:4] != b'\x7fELF'
                or (root / name / 'kernel.cubin').read_bytes()[:4] != b'\x7fELF'):
            raise RuntimeError(f'{name} compiled host wrapper or cubin differs')
        self.name = name
        self.library = C.CDLL(str((root / name / 'kernel.so').resolve()))
        abi = requirements['host_abi']
        self.create = getattr(self.library, abi['create'])
        self.create.argtypes = [C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
        self.create.restype = C.c_int
        self.launch = getattr(self.library, abi['launch'])
        self.launch.argtypes = [C.c_void_p, C.c_void_p]
        self.launch.restype = C.c_int
        self.destroy = getattr(self.library, abi['destroy'])
        self.destroy.argtypes = [C.c_void_p]
        self.destroy.restype = C.c_int
        self.handle = C.c_void_p()
        self.arguments = (C.c_void_p * len(bindings))(
            *(pointers[tensor].value for tensor in bindings.values()))
        checked(self.create(self.arguments, C.byref(self.handle)),
                f'{name} create')

    def run(self) -> None:
        checked(self.launch(self.handle, C.c_void_p()), f'{self.name} launch')

    def close(self) -> None:
        if self.handle.value:
            checked(self.destroy(self.handle), f'{self.name} destroy')
            self.handle = C.c_void_p()


class Runtime:
    def __init__(self, root: Path):
        self.root = root
        self.requirements = prepare(root)
        self.job = admitted()
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
        self.pointers = {}

    def allocate(self, name: str) -> None:
        pointer = C.c_void_p()
        checked(self.cuda.cudaMalloc(C.byref(pointer), nbytes(name)),
                f'{name} cudaMalloc')
        self.pointers[name] = pointer

    def write(self, name: str, data: bytes) -> None:
        if len(data) != nbytes(name):
            raise ValueError(f'{name} input extent differs')
        host = C.create_string_buffer(data)
        checked(self.cuda.cudaMemcpy(self.pointers[name],
                                     C.cast(host, C.c_void_p),
                                     len(data), 1), f'{name} H2D')

    def read(self, name: str) -> bytes:
        host = C.create_string_buffer(nbytes(name))
        checked(self.cuda.cudaMemcpy(C.cast(host, C.c_void_p),
                                     self.pointers[name], nbytes(name), 2),
                f'{name} D2H')
        return host.raw

    def run(self) -> None:
        if (self.root / 'device.json').exists():
            raise ValueError('device evidence must be create-only')
        output = self.root / 'device_outputs'
        output.mkdir(exist_ok=False)
        for name in TENSORS:
            self.allocate(name)
        calls = 0
        for case_name in CASES:
            case = self.root / case_name
            retained = output / case_name
            retained.mkdir()
            for tensor, filename in FILES.items():
                self.write(tensor, (case / filename).read_bytes())
            for tensor in ('up_gate', 'activated', 'out'):
                checked(self.cuda.cudaMemset(self.pointers[tensor], 0,
                                             nbytes(tensor)),
                        f'{tensor} reset')
            checked(self.cuda.cudaDeviceSynchronize(), 'inputs/reset completion')
            stages = []
            try:
                for name, bindings, _, _, _ in STAGES:
                    stages.append(NativeStage(self.root, name,
                                              self.requirements[name], bindings,
                                              self.pointers))
                for stage in stages:
                    stage.run()
                    calls += 1
                checked(self.cuda.cudaDeviceSynchronize(), 'ordered FFN completion')
                for tensor in ('up_gate', 'activated', 'out'):
                    suffix = 'bf16' if TENSORS[tensor][1] == 'bf16' else 'fp32'
                    (retained / f'actual_{tensor}.{suffix}').write_bytes(
                        self.read(tensor))
                for tensor, filename in FILES.items():
                    (retained / f'observed_{filename}').write_bytes(self.read(tensor))
            finally:
                for stage in reversed(stages):
                    stage.close()
        (self.root / 'device.json').write_text(json.dumps({
            'broker_job': self.job, 'compiler_commit': SOURCE_COMMIT,
            'target': 'sm_103a', 'cases': list(CASES),
            'stage_order': [stage[0] for stage in STAGES],
            'launch_calls': calls,
            'scope': 'one-GPU ordered local expert FFN development correctness',
        }, indent=2) + '\n')

    def close(self) -> None:
        for name, pointer in reversed(tuple(self.pointers.items())):
            checked(self.cuda.cudaFree(pointer), f'{name} cudaFree')


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


def compare_fp32(actual: bytes, expected: bytes, *, tolerance: float) -> dict:
    exact = actual == expected
    mismatches, nonfinite, max_abs = 0, 0, 0.0
    for (observed,), (wanted,) in zip(
            struct.iter_unpack('<f', actual),
            struct.iter_unpack('<f', expected), strict=True):
        if not math.isfinite(observed):
            nonfinite += 1
            continue
        error = abs(observed - wanted)
        max_abs = max(max_abs, error)
        mismatches += error > tolerance
    return {'passed': not mismatches and not nonfinite,
            'bitwise_equal': exact, 'mismatches': mismatches,
            'nonfinite': nonfinite, 'max_abs_error': max_abs}


def compare_bf16(actual: bytes, expected: bytes) -> dict:
    exact = actual == expected
    mismatches, nonfinite, max_abs, signed_zero = 0, 0, 0.0, 0
    for (a,), (b,) in zip(struct.iter_unpack('<H', actual),
                         struct.iter_unpack('<H', expected), strict=True):
        observed, wanted = bf16_value(a), bf16_value(b)
        if not math.isfinite(observed):
            nonfinite += 1
            continue
        error = abs(observed - wanted)
        max_abs = max(max_abs, error)
        mismatches += observed != wanted
        signed_zero += a != b and observed == wanted == 0.0
    return {'passed': not mismatches and not nonfinite,
            'bitwise_equal': exact, 'signed_zero_differences': signed_zero,
            'mismatches': mismatches, 'nonfinite': nonfinite,
            'max_abs_error': max_abs}


def verify(root: Path) -> None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('CPU oracle must run after broker lease release')
    if (root / 'report.json').exists():
        raise ValueError('oracle report must be create-only')
    prepare(root)
    device = document(root / 'device.json')
    receipt = document(root / 'gpuq-admission.json')
    if (device['compiler_commit'] != SOURCE_COMMIT
            or device['broker_job'] != receipt['job_id']
            or device['cases'] != list(CASES)
            or device['stage_order'] != [stage[0] for stage in STAGES]
            or device['launch_calls'] != len(CASES) * len(STAGES)):
        raise ValueError('device/Compiler/broker FFN binding differs')
    results = []
    for name in CASES:
        case = root / name
        retained = root / 'device_outputs' / name
        stages = {}
        for tensor, filename in EXPECTED.items():
            suffix = 'bf16' if TENSORS[tensor][1] == 'bf16' else 'fp32'
            actual = (retained / f'actual_{tensor}.{suffix}').read_bytes()
            expected = (case / filename).read_bytes()
            if len(actual) != nbytes(tensor) or len(expected) != nbytes(tensor):
                raise ValueError(f'{name}/{tensor} output extent differs')
            stages[tensor] = (compare_bf16(actual, expected) if suffix == 'bf16'
                              else compare_fp32(actual, expected,
                                                tolerance=0.0 if tensor == 'up_gate'
                                                else 1e-6))
        unchanged = all((retained / f'observed_{filename}').read_bytes()
                        == (case / filename).read_bytes()
                        for filename in FILES.values())
        results.append({'case': name, 'passed': unchanged
                        and all(row['passed'] for row in stages.values()),
                        'inputs_unchanged': unchanged, 'stages': stages})
    report = {'compiler_commit': SOURCE_COMMIT,
              'broker_job': device['broker_job'], 'target': 'sm_103a',
              'cases': results, 'passed': all(row['passed'] for row in results),
              'scope': 'one local expert FFN tile; no dynamic expert routing, EP4, fusion, latency or serving claim'}
    (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'build', 'run', 'verify'))
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve(strict=True)
    {'prepare': prepare, 'build': build, 'run': device_run,
     'verify': verify}[args.phase](root)


if __name__ == '__main__':
    main()
