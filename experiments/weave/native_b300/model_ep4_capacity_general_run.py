"""Four-GPU B300 full-chain replay on changed expert routes and a CPU oracle."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

import model_ep4_capacity_plan_run as planner
from tile_packing_analysis import analyze


LABEL = 'cake-weave-ep4-capacity-general-b300'
R, T, K, E, H, ROWS = (planner.R, planner.T, planner.K,
                       planner.E, planner.H, planner.ROWS)
ROUTES = R*T*K
ADMITTED_CONTROLS = {(1, 0), (74, 5888)}


def prepare(root: Path, bin_root: Path, bridge: Path,
            oracle_root: Path) -> None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('full-chain preparation must be outside a GPU lease')
    manifest = planner.contract(root)
    controls = (manifest.get('communication_ctas'),
                manifest.get('steal_budget'))
    if controls not in ADMITTED_CONTROLS:
        raise ValueError('full-chain spatial or steal control is not admitted')
    if (root / 'case.json').exists() or (root / 'device_outputs').exists():
        raise ValueError('full-chain inputs are create-only')
    ids = np.fromfile(bin_root / 'expert_ids.i32', dtype='<i4')
    if (ids.size != ROUTES or (bin_root / 'hidden.bf16').stat().st_size
            != R*T*H*2 or (bridge / 'route_weights.fp32').stat().st_size
            != ROUTES*4 or (bridge / 'weights_upgate.bf16').stat().st_size
            != E*1536*H*2 or (bridge / 'weights_down.bf16').stat().st_size
            != E*H*768*2):
        raise ValueError('full-chain model input extents differ')
    ids = ids.reshape(R, T, K)
    plan = analyze(list(ids), experts=E, tile_rows=ROWS,
                   source_chunk_tokens=128, early_flush_min_rows=64)
    if (plan['summary']['routes'] != ROUTES
            or plan['summary']['safe_tile_queue_capacity_per_owner'] != 255
            or max(plan['summary']['owner_tile_tasks']) > 255):
        raise ValueError('full-chain route plan exceeds schema-2 capacity')
    expected = np.load(oracle_root / 'expected_output.npy', mmap_mode='r')
    oracle = planner.read(oracle_root / 'oracle_report.json')
    if (expected.shape != (R, T, H) or expected.dtype != np.float32
            or not np.all(np.isfinite(expected))
            or not oracle['all_finite']
            or oracle['geometry'] != {'R': R, 'T': T, 'K': K,
                                      'E': E, 'H': H}
            or Path(oracle['ids_path']).resolve(strict=True)
            != (bin_root / 'expert_ids.i32').resolve(strict=True)):
        raise ValueError('independent changed-route CPU oracle differs')
    (root / 'device_outputs').mkdir()
    for rank in range(R):
        (root / 'device_outputs' / f'rank{rank}').mkdir()
    (root / 'case.json').write_text(json.dumps({
        'source_commit': manifest['source_commit'],
        'case_id': manifest['case_id'],
        'bin_root': str(bin_root.resolve()),
        'bridge_root': str(bridge.resolve()),
        'oracle_root': str(oracle_root.resolve()),
        'geometry': {'R': R, 'T': T, 'K': K, 'E': E, 'H': H, 'tile_rows': ROWS},
        'owner_routes': plan['summary']['owner_routes'],
        'owner_tiles': plan['summary']['owner_tile_tasks'],
        'full_tiles': plan['summary']['full_tiles_published_during_dispatch'],
        'early_tiles': plan['summary']['early_partial_tiles'],
        'terminal_tiles': plan['summary']['terminal_partial_tiles'],
        'safe_tile_capacity': 255,
        'communication_ctas': controls[0],
        'steal_budget': controls[1],
        'scope': 'one four-GPU full chain with changed expert IDs and external FP64 CPU oracle',
    }, indent=2) + '\n')


def admitted() -> str:
    path = Path(os.environ.get('BROKER_RECEIPT', '/nonexistent'))
    for _ in range(30):
        if path.is_file():
            break
        time.sleep(.1)
    else:
        raise RuntimeError('broker admission receipt missing')
    receipt = planner.read(path)
    if (receipt.get('label') != LABEL or receipt.get('mode') != 'exclusive'
            or receipt.get('gpu_count') != R
            or len(receipt.get('gpu_ids', [])) != R):
        raise RuntimeError('full-chain four-GPU admission differs')
    return receipt['job_id']


def run(root: Path) -> None:
    manifest = planner.contract(root)
    case = planner.read(root / 'case.json')
    if (planner.read(root / 'build_report.json')['source_commit']
            != manifest['source_commit']
            or (root / 'device.json').exists()
            or (root / 'launch_failure.json').exists()):
        raise ValueError('full-chain source or create-only state differs')
    job = admitted()
    try:
        result = subprocess.run([
            str(root / 'live_probe'), str(root), case['bin_root'],
            case['bridge_root'], str(case['communication_ctas']),
            str(case['steal_budget']), 'general',
        ], cwd=root, capture_output=True, text=True,
            timeout=300, check=False)
        (root / 'device.log').write_text(result.stdout + result.stderr or
                                         '(no device output)\n')
        if result.returncode or not (root / 'device_report.json').is_file():
            raise RuntimeError(f'full-chain device exited {result.returncode}')
        (root / 'device.json').write_text(json.dumps({
            'broker_job': job, 'source_commit': manifest['source_commit'],
            'target': 'sm_103a', 'world_size': R,
            'scope': 'one four-GPU full chain on changed expert IDs; no qualified timing',
        }, indent=2) + '\n')
    except Exception as error:
        (root / 'launch_failure.json').write_text(json.dumps({
            'broker_job': job, 'error': f'{type(error).__name__}: {error}'},
            indent=2) + '\n')
        raise


def _wave(task: dict) -> int:
    publication = task['published_after']
    return 3 if publication == 'all_dispatch_done' else publication[1]


def verify(root: Path) -> None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('full-chain oracle requires released GPU lease')
    if (root / 'report.json').exists():
        raise ValueError('full-chain report must be create-only')
    manifest = planner.contract(root)
    case = planner.read(root / 'case.json')
    device = planner.read(root / 'device.json')
    receipt = planner.read(root / 'gpuq-admission.json')
    observed = planner.read(root / 'device_report.json')
    if (device['source_commit'] != manifest['source_commit']
            or device['broker_job'] != receipt['job_id']
            or observed['target'] != 'sm_103a'
            or observed['safe_tile_capacity'] != 255
            or observed['valid_routes_by_owner'] != case['owner_routes']
            or observed['bin_rows_by_owner'] != case['owner_routes']
            or observed['logical_tiles_by_owner'] != case['owner_tiles']
            or observed['communication_ctas'] != case['communication_ctas']
            or observed['steal_budget'] != case['steal_budget']
            or observed['worker_grid_ctas'] != [96]*R):
        raise ValueError('full-chain broker, source or device capacity differs')
    stolen = observed['stolen_by_owner']
    if (len(stolen) != R
            or any(type(value) is not int or value < 0
                   or value > case['steal_budget'] for value in stolen)
            or (case['steal_budget'] == 0 and stolen != [0]*R)
            or (case['steal_budget'] > 0 and
                any(value == 0 for value, tiles in zip(
                    stolen, case['owner_tiles'], strict=True) if tiles > 0))):
        raise ValueError('bounded-steal device or coverage evidence differs')
    bin_root = Path(case['bin_root'])
    output = root / 'device_outputs'
    if ((output / 'observed_hidden.bf16').read_bytes()
            != (bin_root / 'hidden.bf16').read_bytes()
            or (output / 'observed_expert_ids.i32').read_bytes()
            != (bin_root / 'expert_ids.i32').read_bytes()):
        raise ValueError('device changed declared hidden or route IDs')
    ids = np.fromfile(bin_root / 'expert_ids.i32', dtype='<i4').reshape(R, T, K)
    plan = analyze(list(ids), experts=E, tile_rows=ROWS,
                   source_chunk_tokens=128, early_flush_min_rows=64)
    hidden = np.memmap(bin_root / 'hidden.bf16', mode='r', dtype='<u2',
                       shape=(R*T, H))
    returned = np.memmap(output / 'contributions.fp32', mode='r',
                         dtype='<f4', shape=(ROUTES, H))
    flags = np.fromfile(output / 'return_ready.i32', dtype='<i4')
    if (flags.size != ROUTES or not np.all(flags == 1)
            or not np.all(np.isfinite(returned))):
        raise ValueError('returned route coverage or FP32 values differ')
    expected_waves = []
    manifest_mismatches = gather_mismatches = return_mismatches = 0
    covered = np.zeros(ROUTES, dtype=np.uint8)
    for owner in range(R):
        tasks = [task for task in plan['tasks'] if task['owner_rank'] == owner]
        tasks.sort(key=lambda task: (_wave(task), task['expert_id'],
                                      task['tile_index']))
        waves = [sum(_wave(task) == wave for task in tasks) for wave in range(4)]
        expected_waves.append(waves)
        rank = output / f'rank{owner}'
        if (np.fromfile(rank / 'wave_counts.i32', dtype='<i4').tolist() != waves
                or (rank / 'tile_input.bf16').stat().st_size
                != len(tasks)*ROWS*H*2
                or (rank / 'down.fp32').stat().st_size
                != len(tasks)*ROWS*H*4):
            raise ValueError(f'owner {owner} wave or tile extent differs')
        experts = np.fromfile(rank / 'tile_expert.i32', dtype='<i4')
        keys = np.fromfile(rank / 'tile_route_keys.i32', dtype='<i4')
        if experts.size != len(tasks) or keys.size != len(tasks)*ROWS:
            raise ValueError(f'owner {owner} manifest extent differs')
        keys = keys.reshape(-1, ROWS)
        expected_experts = np.array([task['expert_id'] % (E//R)
                                     for task in tasks], dtype='<i4')
        expected_keys = np.full((len(tasks), ROWS), -1, dtype='<i4')
        for tile, task in enumerate(tasks):
            for row, (source, token, route) in enumerate(task['rows']):
                expected_keys[tile, row] = (source*T+token)*K+route
        manifest_mismatches += int(np.count_nonzero(experts != expected_experts))
        manifest_mismatches += int(np.count_nonzero(keys != expected_keys))
        if not tasks:
            continue
        tile_input = np.memmap(rank / 'tile_input.bf16', mode='r',
                               dtype='<u2', shape=(len(tasks), ROWS, H))
        down = np.memmap(rank / 'down.fp32', mode='r',
                         dtype='<f4', shape=(len(tasks), ROWS, H))
        for tile, task in enumerate(tasks):
            valid = task['valid_rows']
            if np.count_nonzero(tile_input[tile, valid:]):
                gather_mismatches += 1
            for row in range(valid):
                key = int(expected_keys[tile, row])
                if key < 0 or covered[key]:
                    raise ValueError('route key missing or duplicated')
                covered[key] = 1
                gather_mismatches += int(np.count_nonzero(
                    tile_input[tile, row] != hidden[key//K]))
                return_mismatches += int(np.count_nonzero(
                    down[tile, row].view('<u4') != returned[key].view('<u4')))
    if not np.all(covered == 1) or observed['tile_waves_by_owner'] != expected_waves:
        raise ValueError('GPU tile plan lost a route or changed wave publication')
    raw_output = np.fromfile(output / 'output.bf16', dtype='<u2')
    if raw_output.size != R*T*H:
        raise ValueError('final BF16 output extent differs')
    actual = (raw_output.astype('<u4') << 16).view('<f4').reshape(R, T, H)
    if not np.all(np.isfinite(actual)):
        raise ValueError('final BF16 output contains nonfinite values')
    expected = np.load(Path(case['oracle_root']) / 'expected_output.npy',
                       mmap_mode='r')
    difference = np.abs(actual.astype(np.float64)
                        - expected.astype(np.float64))
    failures = difference > .01 + .01*np.abs(expected.astype(np.float64))
    report = {
        'passed': manifest_mismatches == 0 and gather_mismatches == 0
                  and return_mismatches == 0 and not np.any(failures),
        'source_commit': manifest['source_commit'],
        'broker_job': device['broker_job'], 'case_id': case['case_id'],
        'routes': ROUTES, 'owner_tiles': case['owner_tiles'],
        'communication_ctas': case['communication_ctas'],
        'steal_budget': case['steal_budget'],
        'actual_stolen_by_owner': stolen,
        'tile_waves_by_owner': expected_waves,
        'manifest_mismatches': manifest_mismatches,
        'gathered_input_bit_mismatches': gather_mismatches,
        'return_contribution_bit_mismatches': return_mismatches,
        'final_failing_elements': int(np.count_nonzero(failures)),
        'max_abs_error': float(np.max(difference)),
        'atol': .01, 'rtol': .01,
        'scope': 'one four-GPU full chain with changed routes; independent CPU FP64 output oracle; no timing claim',
    }
    (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'build', 'run', 'verify'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--bin-root', type=Path)
    parser.add_argument('--bridge-root', type=Path)
    parser.add_argument('--oracle-root', type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve(strict=True)
    if args.phase == 'prepare':
        if any(value is None for value in
               (args.bin_root, args.bridge_root, args.oracle_root)):
            parser.error('prepare needs --bin-root, --bridge-root, --oracle-root')
        prepare(root, args.bin_root.expanduser().resolve(strict=True),
                args.bridge_root.expanduser().resolve(strict=True),
                args.oracle_root.expanduser().resolve(strict=True))
    elif args.phase == 'build':
        planner.build(root)
    elif args.phase == 'run':
        run(root)
    else:
        verify(root)


if __name__ == '__main__':
    main()
