"""Standalone B300 expert-bin pack probe with brokered device oracle.

The kernel consumes all four source ranks on one GPU. `prepare`, `build` and
`verify` are CPU-only; `run` requires one exclusive broker GPU. This checks a
dynamic tile input layout, not ranked communication or model FFN arithmetic.
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np


R, T, K, E, H = 4, 512, 8, 128, 2048
LOCAL_E = E // R
MAX_ROWS = R * T
ROUTES = R * T * K
LABEL = 'cake-weave-model-expert-bin-pack-b300'
NVCC = '/usr/local/cuda-13.1/bin/nvcc'
CUDA_RUNTIME = '/usr/local/cuda-13.1/lib64/libcudart.so'
FLAGS = ['-std=c++17', '--gpu-architecture=compute_103a',
         '--gpu-code=sm_103a', '-O3', '-lineinfo', '-Xptxas=-v']


def document(path: Path) -> dict:
    return json.loads(path.read_text())


def require_case(root: Path) -> None:
    case = document(root / 'case.json')
    if (case.get('geometry') != {'R': R, 'T': T, 'K': K, 'E': E, 'H': H}
            or case.get('input_contract') != 'weave-ep4-qwen3-30b-fanin-b300-v2'
            or (root / 'hidden.bf16').stat().st_size != R*T*H*2
            or (root / 'expert_ids.i32').stat().st_size != ROUTES*4
            or not (root / 'model_expert_bin_pack.cu').is_file()):
        raise ValueError('exact model bin-pack input or source differs')


def prepare(root: Path, input_root: Path, source: Path) -> None:
    if root.exists():
        raise ValueError('bin-pack evidence root must be create-only')
    hidden_rows, id_rows = [], []
    for rank in range(R):
        with np.load(input_root / f'rank{rank}-input.npz') as snapshot:
            hidden = snapshot['hidden'].copy()
            ids = snapshot['ids'].copy()
        if (hidden.shape != (T, H) or hidden.dtype != np.float32
                or not np.isfinite(hidden).all()
                or ids.shape != (T, K) or ids.dtype != np.int32
                or np.any(ids < 0) or np.any(ids >= E)
                or any(len(set(map(int, row))) != K for row in ids)):
            raise ValueError(f'rank {rank} input shape, dtype or route domain differs')
        bits = hidden.view(np.uint32)
        if np.any(bits & 0xffff):
            raise ValueError(f'rank {rank} hidden is not exactly BF16-representable')
        hidden_rows.append((bits >> 16).astype('<u2'))
        id_rows.append(ids.astype('<i4', copy=False))
    root.mkdir(parents=True)
    (root / 'model_expert_bin_pack.cu').write_bytes(source.read_bytes())
    (root / 'hidden.bf16').write_bytes(np.concatenate(hidden_rows).tobytes())
    (root / 'expert_ids.i32').write_bytes(np.concatenate(id_rows).tobytes())
    (root / 'case.json').write_text(json.dumps({
        'geometry': {'R': R, 'T': T, 'K': K, 'E': E, 'H': H},
        'input_contract': 'weave-ep4-qwen3-30b-fanin-b300-v2',
        'input_snapshot': str(input_root.resolve()),
        'scope': 'one-GPU BF16 expert bin-pack correctness probe',
    }, indent=2) + '\n')
    require_case(root)


def build(root: Path) -> None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('nvcc build must run outside a broker lease')
    require_case(root)
    compiler = Path(NVCC)
    if not compiler.is_file() or not os.access(compiler, os.X_OK):
        raise FileNotFoundError(f'exact CUDA 13.1 nvcc missing: {compiler}')
    if any((root / name).exists() for name in
           ('kernel.so', 'kernel.cubin', 'build_plan.json',
            'build_report.json', 'build_failure.json')):
        raise ValueError('bin-pack build products must be create-only')
    commands = {
        'host_wrapper': [NVCC, *FLAGS, '-Xcompiler=-fPIC', '-shared',
                         'model_expert_bin_pack.cu', '-o', 'kernel.so', '-lcudart'],
        'cubin': [NVCC, *FLAGS, '--cubin', 'model_expert_bin_pack.cu',
                  '-o', 'kernel.cubin'],
    }
    (root / 'build_plan.json').write_text(json.dumps(commands, indent=2) + '\n')
    exits = {}
    for phase, command in commands.items():
        try:
            result = subprocess.run(command, cwd=root, capture_output=True,
                                    text=True, timeout=180, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            (root / 'build_failure.json').write_text(json.dumps({
                'phase': phase, 'error': f'{type(error).__name__}: {error}',
                'exit_codes': exits}, indent=2) + '\n')
            raise RuntimeError(f'{phase} nvcc invocation failed') from error
        (root / f'{phase}.compile.log').write_text(
            result.stdout + result.stderr or '(no compiler output)\n')
        exits[phase] = result.returncode
        if result.returncode:
            (root / 'build_failure.json').write_text(json.dumps({
                'phase': phase, 'exit_codes': exits}, indent=2) + '\n')
            raise RuntimeError(f'{phase} nvcc rejected source')
    if any(not (root / name).is_file()
           or (root / name).read_bytes()[:4] != b'\x7fELF'
           for name in ('kernel.so', 'kernel.cubin')):
        (root / 'build_failure.json').write_text(json.dumps({
            'error': 'missing or non-ELF compiler product',
            'exit_codes': exits}, indent=2) + '\n')
        raise RuntimeError('bin-pack compiler product differs')
    (root / 'build_report.json').write_text(json.dumps({
        'target': 'sm_103a', 'exit_codes': exits,
        'scope': 'CPU nvcc/PTXAS only; no device correctness or timing',
    }, indent=2) + '\n')


def checked(code: int, operation: str) -> None:
    if code:
        raise RuntimeError(f'{operation}: CUDA status {code}')


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
        require_case(root)
        build_report = document(root / 'build_report.json')
        if (build_report.get('target') != 'sm_103a'
                or build_report.get('exit_codes')
                != {'host_wrapper': 0, 'cubin': 0}):
            raise ValueError('bin-pack compiled target differs')
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
        self.library = C.CDLL(str((root / 'kernel.so').resolve()))
        self.launch = self.library.cake_weave_bin_pack_launch
        self.launch.argtypes = [C.c_void_p] * 6 + [C.c_int, C.c_void_p]
        self.launch.restype = C.c_int
        self.pointers = {}

    def allocate(self, name: str, size: int) -> None:
        pointer = C.c_void_p()
        checked(self.cuda.cudaMalloc(C.byref(pointer), size), f'{name} cudaMalloc')
        self.pointers[name] = pointer

    def write(self, name: str, data: bytes) -> None:
        source = C.create_string_buffer(data)
        checked(self.cuda.cudaMemcpy(self.pointers[name],
                                     C.cast(source, C.c_void_p), len(data), 1),
                f'{name} H2D')

    def read(self, name: str, size: int, offset: int = 0) -> bytes:
        destination = C.create_string_buffer(size)
        source = C.c_void_p(self.pointers[name].value + offset)
        checked(self.cuda.cudaMemcpy(C.cast(destination, C.c_void_p), source,
                                     size, 2), f'{name} D2H')
        return destination.raw

    def run(self) -> None:
        if (self.root / 'device.json').exists() or (self.root / 'device_outputs').exists():
            raise ValueError('bin-pack device evidence must be create-only')
        self.allocate('hidden', R*T*H*2)
        self.allocate('ids', ROUTES*4)
        self.allocate('counts', LOCAL_E*4)
        self.allocate('keys', LOCAL_E*MAX_ROWS*4)
        self.allocate('packed', LOCAL_E*MAX_ROWS*H*2)
        self.allocate('error', 4)
        self.write('hidden', (self.root / 'hidden.bf16').read_bytes())
        self.write('ids', (self.root / 'expert_ids.i32').read_bytes())
        output = self.root / 'device_outputs'
        output.mkdir()
        for owner in range(R):
            row = output / f'owner{owner}'
            row.mkdir()
            for name, value, size in [('counts', 0, LOCAL_E*4),
                                      ('keys', 255, LOCAL_E*MAX_ROWS*4),
                                      ('error', 0, 4)]:
                checked(self.cuda.cudaMemset(self.pointers[name], value, size),
                        f'{name} reset')
            checked(self.launch(*(self.pointers[name] for name in
                                  ('hidden', 'ids', 'counts', 'keys', 'packed',
                                   'error')), owner, C.c_void_p()),
                    f'owner {owner} launch')
            checked(self.cuda.cudaDeviceSynchronize(),
                    f'owner {owner} completion')
            error = int.from_bytes(self.read('error', 4), 'little', signed=True)
            if error:
                raise RuntimeError(f'owner {owner} device domain/capacity error {error}')
            counts_raw = self.read('counts', LOCAL_E*4)
            counts = np.frombuffer(counts_raw, dtype='<u4')
            if np.any(counts > MAX_ROWS):
                raise RuntimeError(f'owner {owner} expert count exceeds capacity')
            keys_raw = self.read('keys', LOCAL_E*MAX_ROWS*4)
            (row / 'counts.u32').write_bytes(counts_raw)
            (row / 'keys.i32').write_bytes(keys_raw)
            packed_rows = bytearray()
            for local_expert, count in enumerate(counts):
                offset = local_expert * MAX_ROWS * H * 2
                packed_rows.extend(self.read('packed', int(count)*H*2, offset))
            (row / 'rows.bf16').write_bytes(packed_rows)
            (row / 'device.json').write_text(json.dumps({
                'owner_rank': owner, 'used_rows': int(sum(map(int, counts))),
                'error_flag': error}, indent=2) + '\n')
        (output / 'observed_hidden.bf16').write_bytes(
            self.read('hidden', R*T*H*2))
        (output / 'observed_expert_ids.i32').write_bytes(
            self.read('ids', ROUTES*4))
        (self.root / 'device.json').write_text(json.dumps({
            'broker_job': self.job, 'target': 'sm_103a', 'owners': R,
            'route_ctas_per_owner': ROUTES,
            'scope': 'one-GPU expert-owned BF16 row pack; no EP4 transport',
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


def verify(root: Path) -> None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('CPU oracle must run after broker lease release')
    if (root / 'report.json').exists():
        raise ValueError('bin-pack oracle report must be create-only')
    require_case(root)
    device = document(root / 'device.json')
    receipt = document(root / 'gpuq-admission.json')
    if (device['broker_job'] != receipt['job_id']
            or device['target'] != 'sm_103a' or device['owners'] != R
            or device['route_ctas_per_owner'] != ROUTES):
        raise ValueError('device/broker/bin-pack binding differs')
    ids = np.frombuffer((root / 'expert_ids.i32').read_bytes(),
                        dtype='<i4').reshape(R*T, K)
    hidden = np.frombuffer((root / 'hidden.bf16').read_bytes(),
                           dtype='<u2').reshape(R*T, H)
    output = root / 'device_outputs'
    if ((output / 'observed_hidden.bf16').read_bytes()
            != (root / 'hidden.bf16').read_bytes()
            or (output / 'observed_expert_ids.i32').read_bytes()
            != (root / 'expert_ids.i32').read_bytes()):
        raise ValueError('device modified a declared input')
    all_keys = []
    owner_reports = []
    for owner in range(R):
        row = output / f'owner{owner}'
        status = document(row / 'device.json')
        counts = np.frombuffer((row / 'counts.u32').read_bytes(),
                               dtype='<u4')
        keys = np.frombuffer((row / 'keys.i32').read_bytes(),
                             dtype='<i4').reshape(LOCAL_E, MAX_ROWS)
        rows = np.frombuffer((row / 'rows.bf16').read_bytes(),
                             dtype='<u2')
        expected = np.bincount(
            ids.ravel()[(ids.ravel() // LOCAL_E) == owner]
            - owner*LOCAL_E, minlength=LOCAL_E)
        if (status['owner_rank'] != owner or status['error_flag'] != 0
                or len(counts) != LOCAL_E or not np.array_equal(counts, expected)
                or status['used_rows'] != int(counts.sum())
                or rows.size != int(counts.sum())*H):
            raise ValueError(f'owner {owner} count, error or extent differs')
        cursor = 0
        for local_expert, count in enumerate(counts):
            count = int(count)
            if not np.all(keys[local_expert, count:] == -1):
                raise ValueError(f'owner {owner} expert {local_expert} wrote unused keys')
            for index in range(count):
                key = int(keys[local_expert, index])
                if not 0 <= key < ROUTES:
                    raise ValueError(f'owner {owner} invalid return key {key}')
                if int(ids[key // K, key % K]) != owner*LOCAL_E+local_expert:
                    raise ValueError(f'owner {owner} expert {local_expert} key {key} is misrouted')
                if not np.array_equal(rows[(cursor+index)*H:(cursor+index+1)*H],
                                      hidden[key // K]):
                    raise ValueError(f'owner {owner} expert {local_expert} key {key} BF16 row differs')
                all_keys.append(key)
            cursor += count
        owner_reports.append({'owner_rank': owner,
                              'routes': int(counts.sum()),
                              'experts_nonempty': int(np.count_nonzero(counts)),
                              'min_expert_rows': int(counts.min()),
                              'max_expert_rows': int(counts.max())})
    if len(all_keys) != ROUTES or set(all_keys) != set(range(ROUTES)):
        raise ValueError('route keys are duplicated or missing across owners')
    report = {'passed': True, 'target': 'sm_103a',
              'broker_job': device['broker_job'], 'routes': ROUTES,
              'input_bytes_unchanged': True, 'owner_reports': owner_reports,
              'scope': 'single-GPU expert bin pack; no EP4 mailbox, tile publication, FFN or latency claim'}
    (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'build', 'run', 'verify'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--input-root', type=Path)
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().absolute()
    if args.phase == 'prepare':
        if args.input_root is None or args.source is None:
            parser.error('prepare requires --input-root and --source')
        prepare(root, args.input_root, args.source)
    else:
        {'build': build, 'run': device_run, 'verify': verify}[args.phase](
            root.resolve(strict=True))


if __name__ == '__main__':
    main()
