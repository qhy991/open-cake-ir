#!/usr/bin/env python3
"""Observe original-factory input views under the existing MACA device lease.

This is bounded input/ABI preparation, not candidate search or performance
measurement. The original factory can perform GPU work; no candidate is run.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from open_cake_ir.tasks.c550_bench.binding import BENCH_COMMIT, BenchProblem, TARGET
from open_cake_ir.evaluation.local_broker import admit_local_job
from open_cake_ir.lab.executor import ExecutorRevision
from open_cake_ir.source_identity import checkout_commit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench-root', type=Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--physical-device', type=int, required=True)
    parser.add_argument('--runtime-device', type=int, required=True)
    parser.add_argument('--expected-pci', required=True)
    parser.add_argument('--queue-seconds', type=float, default=300)
    args = parser.parse_args(argv)
    commit = checkout_commit(ROOT)
    problem = BenchProblem.open(args.bench_root, args.task)
    host = ExecutorRevision.for_target(ROOT, TARGET).admit_host()
    if args.output.resolve().is_relative_to(ROOT):
        parser.error('input observations must remain outside the source checkout')
    report = {'source_commit': commit, 'bench_commit': BENCH_COMMIT, 'task': args.task,
              'target': TARGET, 'scope': 'original_input_factory_metadata_only',
              'candidate_kernel_calls': 0, 'factory_device_dispatches': 'not_collected',
              'performance': 'not_measured', 'cases': [], 'status': 'preparing'}
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    def save():
        temporary = args.output.with_suffix(args.output.suffix + '.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(args.output)
    started = time.monotonic()
    try:
        job = admit_local_job('maca', device=args.physical_device, runtime_device=args.runtime_device,
            expected_pci=args.expected_pci, lock_scope='device', queue_seconds=args.queue_seconds)
        report.update(job_id=job, queue_seconds=time.monotonic() - started)
        for row in problem.workloads:
            case_started = time.monotonic()
            views, metadata, admission = problem.discover_input_views_on_target(
                row.uuid, runtime_library=host['runtime_library'])
            report['cases'].append({'workload_uuid': row.uuid, 'input_views': views,
                'metadata': metadata, 'device_admission': asdict(admission),
                'zero_copy_verified': True, 'preparation_wall_seconds': time.monotonic() - case_started})
            save()
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        save()
        raise
    finally:
        report['wall_seconds'] = time.monotonic() - started
        save()
    print(json.dumps({'task': args.task, 'original_cases': len(report['cases']), 'status': report['status']}))


if __name__ == '__main__':
    main()
