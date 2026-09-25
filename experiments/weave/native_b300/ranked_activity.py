"""Validate one Nsight CUDA activity trace for a Cake ranked EP4 development run.

This is a post-lease diagnostic projection, not the admitted CUPTI/L2 timer.
It requires the broker receipt, one complete four-rank oracle and one kernel
record per logical rank before describing relative kernel activity.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3


def read_kernel_rows(database: Path) -> list[dict]:
    with closing(sqlite3.connect(
            f'file:{database.resolve()}?mode=ro', uri=True)) as connection:
        try:
            rows = connection.execute('''
                SELECT k.deviceId, k.start, k.end, k.registersPerThread,
                       k.gridX, k.blockX, s.value
                FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
                JOIN StringIds AS s ON s.id = k.demangledName
            ''').fetchall()
        except sqlite3.DatabaseError as error:
            raise ValueError('Nsight kernel activity schema differs') from error
    return [dict(zip(('device_id', 'start_ns', 'end_ns', 'registers',
                      'grid_x', 'block_x', 'name'), row, strict=True))
            for row in rows]


def analyze(case: dict, requirements: dict, receipt: dict,
            device: dict, report: dict, rows: list[dict]) -> dict:
    shape = case['shape']
    ranks = shape['R']
    commit = case['source_commit']
    job = receipt.get('job_id')
    physical = receipt.get('gpu_ids')
    if (ranks != 4 or case['target'] != 'sm_103a'
            or requirements.get('compiler_commit') != commit
            or requirements.get('target') != case['target']
            or requirements.get('world_size') != ranks
            or receipt.get('mode') != 'exclusive'
            or receipt.get('gpu_count') != ranks
            or not isinstance(physical, list) or len(physical) != ranks
            or any(type(item) is not int or item < 0 for item in physical)
            or len(set(physical)) != ranks
            or device.get('broker_job') != job or report.get('broker_job') != job
            or device.get('source_commit') != commit
            or report.get('source_commit') != commit
            or device.get('compiler_revision_id') != case['compiler_revision_id']
            or report.get('compiler_revision_id') != case['compiler_revision_id']
            or device.get('launch_calls') != 1
            or report.get('case_id') != case['case_id']
            or report.get('passed') is not True
            or [item.get('rank') for item in report.get('ranks', [])]
            != list(range(ranks))
            or any(item.get('passed') is not True for item in report['ranks'])):
        raise ValueError('ranked source, broker, launch or oracle evidence differs')
    grids = requirements.get('grid_per_rank')
    block = requirements.get('block')
    if (not isinstance(grids, list) or len(grids) != ranks
            or any(not isinstance(grid, list) or len(grid) != 3
                   or any(type(extent) is not int or extent <= 0 for extent in grid)
                   or grid != grids[0] for grid in grids)
            or not isinstance(block, list) or block != [32, 1, 1]):
        raise ValueError('ranked CUDA grid or block declaration differs')
    if len(rows) != ranks or sorted(row['device_id'] for row in rows) != list(range(ranks)):
        raise ValueError('Nsight needs exactly one kernel on every logical rank')
    for row in rows:
        if (not isinstance(row['name'], str)
                or 'cake_ranked_ep4_kernel' not in row['name']
                or any(type(row[field]) is not int for field in (
                    'device_id', 'start_ns', 'end_ns', 'registers', 'grid_x', 'block_x'))
                or row['start_ns'] <= 0 or row['end_ns'] <= row['start_ns']
                or not 0 < row['registers'] <= 255
                or row['grid_x'] != grids[row['device_id']][0]
                or row['block_x'] != block[0]):
            raise ValueError('Nsight ranked kernel identity, timestamp or geometry differs')
    first = min(row['start_ns'] for row in rows)
    last = max(row['end_ns'] for row in rows)
    intersection = max(0, min(row['end_ns'] for row in rows)
                       - max(row['start_ns'] for row in rows))
    return {
        'schema_version': 1,
        'scope': 'development_cuda_kernel_activity_only; not an admitted MoE-layer timer',
        'source_commit': commit,
        'case_id': case['case_id'],
        'broker_job': job,
        'kernel_activity_union_ns': last - first,
        'all_rank_active_intersection_ns': intersection,
        'ranks': [
            {'rank': row['device_id'], 'physical_gpu': physical[row['device_id']],
             'start_offset_ns': row['start_ns'] - first,
             'end_offset_ns': row['end_ns'] - first,
             'kernel_activity_duration_ns': row['end_ns'] - row['start_ns'],
             'registers_per_thread': row['registers']}
            for row in sorted(rows, key=lambda row: row['device_id'])
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('evidence_dir', type=Path)
    parser.add_argument('--sqlite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Nsight analysis output must be create-only')

    def document(name: str) -> dict:
        return json.loads((args.evidence_dir / name).read_text())

    result = analyze(
        document('case.json'), document('requirements.json'),
        document('gpuq-admission.json'), document('device.json'),
        document('report.json'), read_kernel_rows(args.sqlite))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
