"""Brokered planner-only B300 replay for adversarial ranked-tile routes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from tile_packing_analysis import analyze


LABEL = 'cake-weave-ep4-capacity-plan-b300'
NVCC = '/usr/local/cuda-13.1/bin/nvcc'
FLAGS = ['-std=c++17', '--gpu-architecture=compute_103a',
         '--gpu-code=sm_103a', '-O3', '--fmad=false', '-lineinfo', '-Xptxas=-v']
R, T, K, E, H, ROWS = 4, 512, 8, 128, 2048, 128


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def contract(root: Path) -> dict:
    manifest = read(root / 'manifest.json')
    if (manifest.get('target') != 'sm_103a'
            or manifest.get('generation') != 'cake_ep4_capacity_stages'
            or len(manifest.get('source_commit', '')) != 40
            or not isinstance(manifest.get('case_id'), str)
            or not manifest['case_id']
            or manifest.get('world_size') != R
            or not (root / 'model_tile_ready_ffn_capped.cu').is_file()
            or not (root / 'combine/kernel.cu').is_file()):
        raise ValueError('planner source, target or generation differs')
    return manifest


def prepare(root: Path, bin_root: Path, bridge_root: Path) -> None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('planner preparation must be outside the GPU lease')
    manifest = contract(root)
    if (root / 'case.json').exists() or (root / 'device_outputs').exists():
        raise ValueError('planner inputs are create-only')
    ids = np.fromfile(bin_root / 'expert_ids.i32', dtype='<i4')
    if (ids.size != R*T*K or (bin_root / 'hidden.bf16').stat().st_size
            != R*T*H*2 or (bridge_root / 'route_weights.fp32').stat().st_size
            != R*T*K*4 or (bridge_root / 'weights_upgate.bf16').stat().st_size
            != E*1536*H*2 or (bridge_root / 'weights_down.bf16').stat().st_size
            != E*H*768*2):
        raise ValueError('planner model input extents differ')
    ids = ids.reshape(R, T, K)
    plan = analyze(list(ids), experts=E, tile_rows=ROWS,
                   source_chunk_tokens=128, early_flush_min_rows=64)
    if (plan['summary']['routes'] != R*T*K
            or plan['summary']['safe_tile_queue_capacity_per_owner'] != 255
            or max(plan['summary']['owner_tile_tasks']) > 255):
        raise ValueError('planner CPU witness exceeds schema-2 capacity')
    (root / 'device_outputs').mkdir()
    for rank in range(R):
        (root / 'device_outputs' / f'rank{rank}').mkdir()
    (root / 'case.json').write_text(json.dumps({
        'source_commit': manifest['source_commit'],
        'case_id': manifest['case_id'],
        'bin_root': str(bin_root.resolve()),
        'bridge_root': str(bridge_root.resolve()),
        'geometry': {'R': R, 'T': T, 'K': K, 'E': E, 'H': H, 'tile_rows': ROWS},
        'cpu_owner_routes': plan['summary']['owner_routes'],
        'cpu_owner_tiles': plan['summary']['owner_tile_tasks'],
        'cpu_full_tiles': plan['summary']['full_tiles_published_during_dispatch'],
        'cpu_early_tiles': plan['summary']['early_partial_tiles'],
        'cpu_terminal_tiles': plan['summary']['terminal_partial_tiles'],
        'safe_tile_capacity': 255,
        'scope': 'planner-only; no FFN output or latency claim',
    }, indent=2) + '\n')


def build(root: Path) -> None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('nvcc build must be outside the GPU lease')
    manifest = contract(root)
    case = read(root / 'case.json')
    if (case['source_commit'] != manifest['source_commit']
            or any((root / name).exists() for name in
                   ('live_probe', 'build_plan.json', 'build_report.json',
                    'build_failure.json'))):
        raise ValueError('planner build source or create-only state differs')
    command = [NVCC, *FLAGS, 'model_tile_ready_ffn_capped.cu',
               '-o', 'live_probe', '-lcuda', '-lcudart']
    (root / 'build_plan.json').write_text(json.dumps(command, indent=2) + '\n')
    result = subprocess.run(command, cwd=root, capture_output=True,
                            text=True, timeout=240, check=False)
    (root / 'compile.log').write_text(result.stdout + result.stderr or
                                      '(no compiler output)\n')
    if result.returncode:
        (root / 'build_failure.json').write_text(json.dumps({
            'exit_code': result.returncode}, indent=2) + '\n')
        raise RuntimeError('planner CUDA build failed')
    if (root / 'live_probe').read_bytes()[:4] != b'\x7fELF':
        raise RuntimeError('planner executable is not ELF')
    (root / 'build_report.json').write_text(json.dumps({
        'source_commit': manifest['source_commit'], 'target': 'sm_103a',
        'exit_code': 0, 'scope': 'CPU nvcc/PTXAS only'}, indent=2) + '\n')


def admitted() -> str:
    path = Path(os.environ.get('BROKER_RECEIPT', '/nonexistent'))
    for _ in range(30):
        if path.is_file():
            break
        time.sleep(.1)
    else:
        raise RuntimeError('broker admission receipt missing')
    receipt = read(path)
    if (receipt.get('label') != LABEL or receipt.get('mode') != 'exclusive'
            or receipt.get('gpu_count') != R
            or len(receipt.get('gpu_ids', [])) != R):
        raise RuntimeError('planner admission differs from four-GPU lease')
    return receipt['job_id']


def run(root: Path) -> None:
    manifest = contract(root)
    case = read(root / 'case.json')
    if (read(root / 'build_report.json')['source_commit']
            != manifest['source_commit']
            or (root / 'device.json').exists()
            or (root / 'launch_failure.json').exists()):
        raise ValueError('planner run source or create-only state differs')
    job = admitted()
    try:
        result = subprocess.run([
            str(root / 'live_probe'), str(root), case['bin_root'],
            case['bridge_root'], '1', '0', 'plan-only',
        ], cwd=root, capture_output=True, text=True,
            timeout=300, check=False)
        (root / 'device.log').write_text(result.stdout + result.stderr or
                                         '(no device output)\n')
        if result.returncode or not (root / 'plan_device_report.json').is_file():
            raise RuntimeError(f'planner device exited {result.returncode}')
        (root / 'device.json').write_text(json.dumps({
            'broker_job': job, 'source_commit': manifest['source_commit'],
            'target': 'sm_103a', 'world_size': R,
            'scope': 'GPU route dispatch and variable tile planner only',
        }, indent=2) + '\n')
    except Exception as error:
        (root / 'launch_failure.json').write_text(json.dumps({
            'broker_job': job, 'error': f'{type(error).__name__}: {error}'},
            indent=2) + '\n')
        raise


def verify(root: Path) -> None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('planner oracle must run after broker release')
    if (root / 'report.json').exists():
        raise ValueError('planner report must be create-only')
    manifest = contract(root)
    case = read(root / 'case.json')
    device = read(root / 'device.json')
    receipt = read(root / 'gpuq-admission.json')
    observed = read(root / 'plan_device_report.json')
    if (device['source_commit'] != manifest['source_commit']
            or device['broker_job'] != receipt['job_id']
            or observed['target'] != 'sm_103a'
            or observed['safe_tile_capacity'] != 255
            or observed['routes'] != R*T*K
            or observed['owner_routes'] != case['cpu_owner_routes']
            or observed['logical_tiles_by_owner'] != case['cpu_owner_tiles']):
        raise ValueError('planner source, broker or capacity differs')
    ids = np.fromfile(Path(case['bin_root']) / 'expert_ids.i32',
                      dtype='<i4').reshape(R, T, K)
    plan = analyze(list(ids), experts=E, tile_rows=ROWS,
                   source_chunk_tokens=128, early_flush_min_rows=64)
    mismatches = 0
    expected_waves = []
    for owner in range(R):
        tasks = [task for task in plan['tasks']
                 if task['owner_rank'] == owner]
        def wave(task):
            publication = task['published_after']
            if publication == 'all_dispatch_done':
                return 3
            return publication[1]
        tasks.sort(key=lambda task: (wave(task), task['expert_id'],
                                      task['tile_index']))
        waves = [sum(wave(task) == w for task in tasks) for w in range(4)]
        expected_waves.append(waves)
        output = root / 'device_outputs' / f'rank{owner}'
        actual_waves = np.fromfile(output / 'wave_counts.i32', dtype='<i4').tolist()
        actual_experts = np.fromfile(output / 'tile_expert.i32', dtype='<i4')
        actual_keys = np.fromfile(output / 'tile_route_keys.i32',
                                  dtype='<i4')
        if (actual_waves != waves or actual_experts.size != len(tasks)
                or actual_keys.size != len(tasks)*ROWS):
            raise ValueError(f'owner {owner} tile/wave extents differ')
        expected_experts = np.array([task['expert_id'] % (E//R)
                                     for task in tasks], dtype='<i4')
        expected_keys = np.full((len(tasks), ROWS), -1, dtype='<i4')
        for index, task in enumerate(tasks):
            for row, (source, token, route) in enumerate(task['rows']):
                expected_keys[index, row] = (source*T + token)*K + route
        mismatches += int(np.count_nonzero(actual_experts != expected_experts))
        mismatches += int(np.count_nonzero(
            actual_keys.reshape(-1, ROWS) != expected_keys))
    if observed['tile_waves_by_owner'] != expected_waves:
        raise ValueError('planner wave publication counts differ')
    report = {'passed': mismatches == 0,
              'source_commit': manifest['source_commit'],
              'broker_job': device['broker_job'],
              'case_id': case['case_id'],
              'routes': R*T*K,
              'safe_tile_capacity': 255,
              'owner_tiles': case['cpu_owner_tiles'],
              'tile_waves_by_owner': expected_waves,
              'full_tiles': case['cpu_full_tiles'],
              'early_partial_tiles': case['cpu_early_tiles'],
              'terminal_partial_tiles': case['cpu_terminal_tiles'],
              'manifest_mismatches': mismatches,
              'scope': 'planner-only; no FFN output or performance claim'}
    (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'build', 'run', 'verify'))
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--bin-root', type=Path)
    parser.add_argument('--bridge-root', type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve(strict=True)
    if args.phase == 'prepare':
        if args.bin_root is None or args.bridge_root is None:
            parser.error('prepare requires --bin-root and --bridge-root')
        prepare(root, args.bin_root.expanduser().resolve(strict=True),
                args.bridge_root.expanduser().resolve(strict=True))
    else:
        {'build': build, 'run': run, 'verify': verify}[args.phase](root)


if __name__ == '__main__':
    main()
