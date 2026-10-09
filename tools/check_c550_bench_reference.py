#!/usr/bin/env python3
"""Run the pinned Bench reference self-check under the existing MACA lease.

The original CLI owns input generation, reference execution and comparison.
This entry binds the host and device, then delegates without a second evaluator.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from open_cake_ir.evaluation.local_broker import admit_local_job
from open_cake_ir.evaluation.triton_metax import observe_local_metax
from open_cake_ir.lab.custody import admit_new_campaign_path
from open_cake_ir.lab.executor import ExecutorRevision
from open_cake_ir.source_identity import checkout_commit
from open_cake_ir.tasks.c550_bench.binding import BenchProblem, BENCH_COMMIT, TARGET


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench-root', type=Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--output', type=Path, required=True, help='new external output directory')
    parser.add_argument('--physical-device', type=int, required=True)
    parser.add_argument('--runtime-device', type=int, required=True)
    parser.add_argument('--expected-pci', required=True)
    parser.add_argument('--queue-seconds', type=float, default=120)
    args = parser.parse_args(argv)
    source = checkout_commit(ROOT)
    problem = BenchProblem.open(args.bench_root, args.task)
    host = ExecutorRevision.for_target(ROOT, TARGET).admit_host()
    output = admit_new_campaign_path(ROOT, args.output, role='Bench reference qualification')
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / 'reference-selfcheck.json'
    command = [str(problem.root / 'c550bench.py'), 'check', '--task', args.task,
               '--device', 'cuda:0', '--reference-selfcheck', '--workloads', 'all',
               '--rounds', '10', '--seed', str(problem.api.document('suite.json')['seed']),
               '--output', str(report_path)]
    authority = {'source_commit': source, 'bench_commit': BENCH_COMMIT,
                 'task': args.task, 'scope': 'original_reference_selfcheck_only',
                 'candidate_qualified': False, 'performance': 'not_measured',
                 'original_command': command}
    with (output / 'authority.json').open('x') as stream:
        json.dump(authority, stream, indent=2)
    admit_local_job('maca', device=args.physical_device, runtime_device=args.runtime_device,
                    expected_pci=args.expected_pci, lock_scope='device', queue_seconds=args.queue_seconds)
    admission = observe_local_metax(TARGET, runtime_library=host['runtime_library'])
    with (output / 'device-admission.json').open('x') as stream:
        json.dump(asdict(admission), stream, indent=2)
    original_argv = sys.argv
    try:
        sys.argv = command
        # SystemExit is deliberately preserved: the Bench owns success/failure.
        runpy.run_path(command[0], run_name='__main__')
    finally:
        sys.argv = original_argv


if __name__ == '__main__':
    main()
