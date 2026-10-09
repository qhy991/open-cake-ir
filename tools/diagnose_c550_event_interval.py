#!/usr/bin/env python3
"""One instrumented RMS cohort; no paired comparison or performance verdict."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from benchmarks.c550 import native_bridge
from benchmarks.c550.host_timing_trace import HostTimeline, instrument_host_phases

TASK = 'L1/069_rms_norm'
CASE_INDEX = 1


def capture_once(loaded, workload, expected, timer, torch_module, result):
    from open_cake_ir.evaluation.core import compare_tile_outputs
    from open_cake_ir.tasks.evaluate import _fresh_tile_cohort
    timeline = HostTimeline()
    result['host_timeline'] = timeline.rows
    result['checks'] = {}

    def check():
        loaded.launch()
        observed, after = loaded.snapshot()
        passed, values = compare_tile_outputs(workload, loaded.validation_inputs,
                                              expected, observed, after)
        if not passed:
            result['checks']['failed_observation'] = values
            raise ValueError('original Bench correctness failed; stop the diagnostic')
        return values

    try:
        result['checks']['preflight'] = check()
        with instrument_host_phases(loaded.loaded, torch_module, timeline):
            samples, checks = _fresh_tile_cohort(loaded, timer, workload,
                loaded.validation_inputs, expected, samples_per_cohort=5,
                route_calls_per_cohort=16)
        result['device_events_ms'] = samples
        result['checks']['cohort'] = checks
        if not checks['passed']:
            raise ValueError('original Bench cohort correctness failed; stop the diagnostic')
        result['checks']['postflight'] = check()
        if loaded.loaded.launch_calls != 18:
            raise ValueError('diagnostic native dispatch count differs from eighteen')
        result['completed'] = True
    finally:
        result['native_calls'] = loaded.loaded.launch_calls
        result['event_observation'] = timer.last_activity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench-root', type=Path, required=True)
    parser.add_argument('--baselines', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--physical-device', type=int, required=True)
    parser.add_argument('--runtime-device', type=int, required=True)
    parser.add_argument('--expected-pci', required=True)
    args = parser.parse_args(argv)
    from open_cake_ir.source_identity import checkout_commit
    from open_cake_ir.lab.custody import admit_new_campaign_path
    from open_cake_ir.lab.executor import ExecutorRevision
    from open_cake_ir.tasks.c550_bench.binding import BenchProblem, TARGET, require_oracle_numerics
    from open_cake_ir.tasks.c550_bench.workload import prepare_evaluation_case, _recorded_statistics
    from open_cake_ir.evaluation.local_broker import admit_local_job
    from open_cake_ir.evaluation.triton_metax import observe_local_metax
    from open_cake_ir.evaluation.core import LoadedTorchTensorCandidate, _plain_json
    from open_cake_ir.evaluation.metax_event_benchmark import MacaEventBenchmark
    from open_cake_ir.evaluation.loaders import LifecycleError
    from open_cake_ir.compiler.target import declared_target
    commit = checkout_commit(ROOT)
    if commit is None or os.environ.get('METAL_BROKER_LOCK_FD') or os.environ.get('GPUQ_JOB_ID'):
        parser.error('diagnostic requires a clean source and no inherited device lease')
    output = admit_new_campaign_path(ROOT, args.output, role='single C550 host timing diagnosis')
    output.mkdir(parents=True, exist_ok=False)
    result = {'source_commit': commit, 'task': TASK, 'case_index': CASE_INDEX,
        'completed': False, 'scope': 'one instrumented single-arm cohort; no qualification verdict',
        'instrumentation_changes_interval': True, 'performance_qualified': False,
        'provider_calls': 0, 'optimization_runs_started': 0, 'promotion': 'No promotion',
        'schedule': {'preflight': 1, 'warmups': 11, 'samples': 5, 'postflight': 1},
        'external_gpu_activity': 'not_excluded'}
    loaded = None
    failure = None
    policy = None
    try:
        problem = BenchProblem.open(args.bench_root, TASK)
        case = native_bridge.bind_all(problem, args.baselines)[CASE_INDEX]
        if case.candidate.is_program or case.manifest.aligned_variant:
            raise ValueError('diagnostic requires the original single RMS dispatch')
        result['workload_uuid'] = case.uuid
        host = ExecutorRevision.for_target(ROOT, TARGET).admit_host()
        result['job_id'] = admit_local_job('maca', device=args.physical_device,
            runtime_device=args.runtime_device, expected_pci=args.expected_pci,
            lock_scope='device', queue_seconds=120)
        admission = observe_local_metax(TARGET, runtime_library=host['runtime_library'])
        policy = case.workload.document['semantics']['oracle_numerics']
        require_oracle_numerics(policy, phase='before host timing diagnosis')
        inputs, expected = prepare_evaluation_case(case.workload, 'primary', admission=admission)
        loaded = LoadedTorchTensorCandidate(case.candidate, case.manifest, inputs, admission,
                                            preserve_output_tensors=True)
        timer = MacaEventBenchmark(case.manifest, l2_cache_bytes=declared_target(TARGET).l2_cache_bytes)
        import torch
        capture_once(loaded, case.workload, expected, timer, torch, result)
    except BaseException as error:
        failure = error
    finally:
        if loaded is not None:
            try:
                loaded.close()
            except BaseException as error:
                failure = LifecycleError(failure, error) if failure is not None else error
            result['module_closed'] = loaded.loaded.closed
        if policy is not None:
            try:
                require_oracle_numerics(policy, phase='after host timing diagnosis')
            except BaseException as error:
                failure = LifecycleError(failure, error) if failure is not None else error
        if failure is not None:
            result.update(completed=False, error=str(failure), error_type=type(failure).__name__)
            (output / 'failure.txt').write_text(''.join(traceback.format_exception(failure)))
        with (output / 'result.json').open('x') as stream:
            json.dump(_recorded_statistics(_plain_json(result)), stream, indent=2, allow_nan=False)
    print(json.dumps({'completed': result['completed'], 'performance_qualified': False,
                      'provider_calls': 0}))
    return 0 if result['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
