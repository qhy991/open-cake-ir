#!/usr/bin/env python3
"""Bind existing baseline-only artifacts and run the original Bench correctness loop."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'tools')]

from benchmarks.c550 import native_bridge
from open_cake_ir.tasks.c550_bench.binding import BenchProblem, BENCH_COMMIT, TARGET
from open_cake_ir.source_identity import checkout_commit
from open_cake_ir.lab.bindings import external_file


def write(path,value):
    from open_cake_ir.serialization import canonical_json_bytes
    with path.open('xb') as stream:
        stream.write(canonical_json_bytes(value))


def evaluate_original(problem,cases,output,*,physical_device,runtime_device,expected_pci):
    from open_cake_ir.lab.executor import ExecutorRevision
    from open_cake_ir.evaluation.local_broker import admit_local_job
    from open_cake_ir.evaluation.triton_metax import observe_local_metax
    suite=problem.api.document('suite.json')
    if suite['correctness_rounds'] != 10:
        raise ValueError('native bridge requires the fixed original ten-round suite')
    if native_bridge._ACTIVE is not None:
        raise ValueError('native Bench bridge cannot nest another active callback')
    host=ExecutorRevision.for_target(ROOT,TARGET).admit_host()
    job=admit_local_job('maca',device=physical_device,runtime_device=runtime_device,
        expected_pci=expected_pci,lock_scope='device',queue_seconds=300)
    admission=observe_local_metax(TARGET,runtime_library=host['runtime_library'])
    trace=output/'native-calls.jsonl'
    trace.touch(exist_ok=False)
    rows=[]
    def record(row):
        with trace.open('a') as stream:
            stream.write(json.dumps(row,sort_keys=True)+'\n')
        rows.append(row)
    adapter=output/'candidate.py'
    adapter.write_text('from benchmarks.c550.native_bridge import run\n')
    native_bridge._ACTIVE=native_bridge.NativeBench(cases,10,admission,record)
    try:
        report=problem.api.check_problem(problem.api.task_record(problem.task_id),device='cuda:0',
            candidate_path=adapter,symbol='run',reference_selfcheck=False,workload_scope='all',
            rounds=10,output=output/'original-bench.json',seed=suite['seed'],threads=4)
    finally:
        native_bridge._ACTIVE=None
    expected=[(case.uuid,round_index) for case in cases for round_index in range(10)]
    native_ok=([(row['uuid'],row['round']) for row in rows]==expected
        and all(row['kernel_calls']==row['expected_stage_calls'] and row['module_closed'] is True
                and row['input_unchanged'] is True and row['error'] is None for row in rows))
    original_ok=([(row['workload_uuid'],row['round']) for row in report['cases']]==expected
                 and all(row['passed'] for row in report['cases']))
    return {'job_id':job,'passed':report['status']=='passed' and report['full_device_correctness'] is True
                and native_ok and original_ok,'original_checks':len(report['cases']),
            'full_device_correctness':bool(report['full_device_correctness'] and native_ok and original_ok),
            'native_checks_passed':native_ok,'native_kernel_calls':sum(row['kernel_calls'] for row in rows),
            'factory_reference_device_dispatches':'not_collected','performance':'not_measured'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('inspect','evaluate'))
    parser.add_argument('--bench-root',type=Path,required=True)
    parser.add_argument('--task',required=True)
    parser.add_argument('--baselines',type=Path,required=True,help='external ordered UUID/prepared-baseline locator index')
    parser.add_argument('--input-views',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--physical-device',type=int)
    parser.add_argument('--runtime-device',type=int)
    parser.add_argument('--expected-pci')
    args=parser.parse_args(argv)
    import open_cake_ir.compiler
    if not Path(open_cake_ir.compiler.__file__).resolve().is_relative_to((ROOT/'src').resolve()):
        parser.error('Compiler import came from another checkout')
    commit=checkout_commit(ROOT)
    if commit is None or os.environ.get('METAL_BROKER_LOCK_FD') or os.environ.get('GPUQ_JOB_ID'):
        parser.error('qualification requires clean committed source and starts outside a lease')
    if args.command=='evaluate' and any(value is None for value in (args.physical_device,args.runtime_device,args.expected_pci)):
        parser.error('device qualification requires physical/runtime/PCI binding')
    from open_cake_ir.lab.custody import admit_new_campaign_path
    output=admit_new_campaign_path(ROOT,args.output,role='original Bench native qualification')
    output.mkdir(parents=True,exist_ok=False)
    result={'source_commit':commit,'bench_commit':BENCH_COMMIT,'task':args.task,'target':TARGET,
            'passed':False,'full_device_correctness':False,'performance':'not_measured',
            'scope':'original Bench all/rounds10 qualification; no author Run'}
    try:
        problem=BenchProblem.open(args.bench_root,args.task)
        views=(json.loads(external_file(ROOT,str(args.input_views),'input-view qualification').read_text())
               if args.input_views is not None else None)
        cases=native_bridge.bind_all(problem,args.baselines,input_views=views)
        result.update(bound_cases=len(cases),stage_counts=[case.manifest.kernels_per_call for case in cases])
        if args.command=='inspect':
            result.update(passed=True,scope='CPU binding only; no device execution')
        else:
            result.update(evaluate_original(problem,cases,output,physical_device=args.physical_device,
                runtime_device=args.runtime_device,expected_pci=args.expected_pci))
    except Exception as error:
        result.update(error=str(error),failure_class=type(error).__name__)
        (output/'failure.txt').write_text(traceback.format_exc())
    write(output/'result.json',result)
    print(json.dumps(result,indent=2))
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
