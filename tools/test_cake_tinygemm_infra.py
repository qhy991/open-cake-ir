#!/usr/bin/env python3
"""Bounded TinyGEMM development qualification behind the existing GPU Infra broker.

Checks two complete generated candidates on the three public fixtures. This is not
a provider Campaign, a timing result, or a claim that Triton preserved the CAKE schedule.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import statistics
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.evaluation.admission import observe_exclusive_cuda
from open_cake_ir.evaluation.core import compare_tile_outputs
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.source_identity import checkout_commit
from open_cake_ir.tasks.tinygemm import reproduction as task
from tools.compare_flashinfer_reference import admit_judge_source


def specification(path):
    value = json.loads(path.read_text())
    if (not isinstance(value, dict) or set(value) not in ({'task', 'target'}, {'task', 'target', 'benchmark'})
            or value.get('task') != task.TASK or value.get('target') != 'sm_103a'
            or type(value.get('benchmark', False)) is not bool):
        raise ValueError('qualification input must name exact TinyGEMM task on sm_103a')
    return value


def qualify(output, compiler, torch, *, include_official=False):
    rows = []
    timing_inputs = []
    catalog = json.loads((ROOT / 'experiments/flashinfer_rewrites/catalog.json').read_text())
    spec = next(r['assessment'] for r in catalog['tasks'] if r['id'] == '029_cake_tinygemm2')
    for shape in spec['benchmark_rows']:
        document = task.workload_document(rows=shape['batch'], columns=shape['output_features'],
                                          depth=shape['input_features'], backend='triton-b300')
        workload = WorkloadContract(document)
        folder = output / shape['id']
        folder.mkdir(parents=True)
        (folder / 'workload.json').write_text(json.dumps(document, indent=2) + '\n')
        entries = {}
        for stages in (4, 8):
            source = task.partitioned_source(workload, stages=stages)
            assessment = compiler.assess(frontend.parse(source).document)
            (folder / f'assessment-s{stages}.json').write_text(json.dumps(
                [f.to_dict() for f in assessment.findings], indent=2) + '\n')
            lowering = compiler.lower(assessment)
            (folder / f'candidate-s{stages}.py').write_text(source)
            path = folder / f'lowered-s{stages}.py'
            path.write_text(lowering.source)
            name = f'cake_tiny_{shape["id"]}_s{stages}'
            module_spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[name] = module
            module_spec.loader.exec_module(module)
            entries[stages] = getattr(module, lowering.route.entry_point)
        for case_id in workload.case_ids:
            cpu = task.materialize_case(workload, case_id)
            before = {name: list(values) for name, values in cpu.items()}
            expected = task.reference_outputs(workload, case_id, cpu)
            arguments = workload.tensor_abi(case_id)
            inputs = {a.name: torch.tensor(cpu[a.name], dtype=torch.bfloat16, device='cuda').reshape(a.shape)
                      for a in arguments if a.mode == 'input'}
            out_arg = next(a for a in arguments if a.mode == 'output')
            if include_official:
                # Every declared fixture selects stage4 under the pinned dispatcher.
                grid = math.ceil(shape['output_features']/16) * math.ceil(shape['batch']/8)
                sms = torch.cuda.get_device_properties(0).multi_processor_count
                if not (shape['input_features'] <= 1024 or grid > 2*sms):
                    raise ValueError('fixture no longer belongs to the declared official stage4 route')
                official = task.official_cake_module(workload.target)
                official_out = torch.full(out_arg.shape, float('nan'), dtype=torch.bfloat16, device='cuda')
                official.stage4_op(inputs['x'], inputs['weight'], inputs['bias'], official_out)
                torch.cuda.synchronize()
                actual = {'out': official_out.cpu().reshape(-1).tolist()}
                after = {name: value.cpu().reshape(-1).tolist() for name, value in inputs.items()}
                official_passed, official_metrics = compare_tile_outputs(workload, before, expected, actual, after)
                (folder / f'{case_id}-official.json').write_text(json.dumps(
                    {'passed': official_passed, 'metrics': official_metrics}, indent=2)+'\n')
                if not official_passed:
                    raise ValueError('official CAKE export failed the same complete bitwise gate')
                if case_id == 'primary':
                    timing_inputs.append((shape, workload, inputs, out_arg.shape, entries, expected))
            for stages, entry in entries.items():
                result = torch.full(out_arg.shape, float('nan'), dtype=torch.bfloat16, device='cuda')
                returned = entry(**inputs, out=result)
                torch.cuda.synchronize()
                if returned.data_ptr() != result.data_ptr():
                    raise ValueError('generated candidate replaced the caller output')
                observed = {'out': result.cpu().reshape(-1).tolist()}
                after = {name: value.cpu().reshape(-1).tolist() for name, value in inputs.items()}
                passed, metrics = compare_tile_outputs(workload, before, expected, observed, after)
                row = {'shape': shape['id'], 'case_id': case_id, 'stages': stages,
                       'applied_loop_stages': stages if shape['input_features'] > 1024 else None,
                       'passed': passed, 'metrics': metrics, 'elements': result.numel()}
                rows.append(row)
                # Whole outputs are retained for independent replay, including failures.
                (folder / f'{case_id}-s{stages}.json').write_text(json.dumps(
                    {'expected': expected,
                     'observed': {'out': [v if math.isfinite(v) else str(v) for v in observed['out']]},
                     'nonfinite_encoding': 'nan/inf/-inf strings', 'metrics': metrics}, allow_nan=False) + '\n')
                (output / 'progress.json').write_text(json.dumps(rows, indent=2) + '\n')
                print(shape['id'], case_id, stages, metrics, flush=True)
    return rows, timing_inputs


def measure_matched(timing_inputs, output, torch):
    """Five alternating-order paired rounds; single-kernel CUPTI intervals, no fallback."""
    import flashinfer.testing as helper
    from open_cake_ir.evaluation.benchmark import StrictCuptiBenchmark
    bench = StrictCuptiBenchmark(helper)
    summaries = []
    for shape, workload, inputs, out_shape, entries, expected in timing_inputs:
        outputs = {name: torch.empty(out_shape, dtype=torch.bfloat16, device='cuda')
                   for name in ('trt_reference', 'official_cake', 'open_cake_s4', 'open_cake_s8')}
        reference = task._peer_module(workload.target)
        official = task.official_cake_module(workload.target)
        functions = {
            'trt_reference': lambda: reference.tinygemm2_op(inputs['x'], inputs['weight'], inputs['bias'], outputs['trt_reference'], False),
            'official_cake': lambda: official.stage4_op(inputs['x'], inputs['weight'], inputs['bias'], outputs['official_cake']),
            'open_cake_s4': lambda: entries[4](**inputs, out=outputs['open_cake_s4']),
            'open_cake_s8': lambda: entries[8](**inputs, out=outputs['open_cake_s8']),
        }
        rounds = []
        before = {name: value.cpu().reshape(-1).tolist() for name, value in inputs.items()}
        for round_id in range(5):
            order = list(functions) if round_id % 2 == 0 else list(reversed(functions))
            record = {'round': round_id+1, 'order': order, 'arms': {}}
            for arm in order:
                samples = [float(x) for x in bench(functions[arm], dry_run_iters=5, repeat_iters=25,
                                                    cold_l2_cache=True, use_cuda_graph=False)]
                if len(samples) != 25 or any(not math.isfinite(x) or x <= 0 for x in samples):
                    raise ValueError('CUPTI did not return 25 positive finite samples')
                observed = {'out': outputs[arm].cpu().reshape(-1).tolist()}
                after = {name: value.cpu().reshape(-1).tolist() for name, value in inputs.items()}
                if not compare_tile_outputs(workload, before, expected, observed, after)[0]:
                    raise ValueError('post-timing output/input verification failed')
                record['arms'][arm] = {'median_ms': statistics.median(samples), 'samples_ms': samples,
                                       'cv': statistics.pstdev(samples)/statistics.mean(samples)}
            rounds.append(record)
        # A separate profiler observation proves the executed CUDA activity boundary.
        symbols = {}
        for arm, function in functions.items():
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profile:
                function()
                torch.cuda.synchronize()
            events = [e.name for e in profile.events() if str(e.device_type).endswith('CUDA')]
            symbols[arm] = events
            if len(events) != 1:
                raise ValueError(f'{arm} does not have a witnessed single-kernel boundary: {events}')
        median = {arm: statistics.median(r['arms'][arm]['median_ms'] for r in rounds) for arm in functions}
        summaries.append({'shape': shape, 'median_ms': median, 'rounds': rounds, 'cuda_symbols': symbols,
                          'open_cake_s4_vs_official': statistics.median(r['arms']['official_cake']['median_ms']/r['arms']['open_cake_s4']['median_ms'] for r in rounds),
                          'open_cake_s8_vs_official': statistics.median(r['arms']['official_cake']['median_ms']/r['arms']['open_cake_s8']['median_ms'] for r in rounds)})
        (output/'timing-progress.json').write_text(json.dumps(summaries,indent=2)+'\n')
    return {'scope':'single_kernel_CUPTI_ms', 'cold_l2':True, 'cuda_graph':False, 'pdl':False,
            'rounds':5, 'samples_per_arm_per_round':25, 'paired_order':'forward_reverse_alternating',
            'latency_qualification':'measured descriptive comparison; no predeclared parity threshold',
            'cases':summaries}


def main():
    stage = Path(os.environ['KERNELINFRA_STAGE_DIR'])
    destination = Path(os.environ['KERNELINFRA_RESULT'])
    result = {'schema': 'kernelinfra.stage-result.v1', 'status': 'failed', 'validity': 'unknown',
              'summary': 'not evaluated', 'artifacts': {}}
    try:
        commit = checkout_commit(ROOT)
        admit_judge_source(commit, json.loads(Path(os.environ['KERNELINFRA_TASK']).read_text()),
                           os.environ['KERNELINFRA_STAGE_ID'])
        config = specification(Path(os.environ['KERNELINFRA_CANDIDATE_DIR']) / 'qualification.json')
        compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')
        if not compiler.check_corpus().passed:
            raise ValueError('Corpus Gate failed before GPU execution')
        admission = observe_exclusive_cuda('sm_103a')
        import torch
        import triton
        import flashinfer
        if torch.cuda.device_count() != 1:
            raise ValueError('one broker-owned GPU is required')
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        output = stage / 'checks'
        output.mkdir(exist_ok=False)
        rows, timing_inputs = qualify(output, compiler, torch, include_official=config.get('benchmark', False))
        passed = len(rows) == 30 and all(row['passed'] for row in rows)
        timing = measure_matched(timing_inputs, output, torch) if passed and config.get('benchmark', False) else 'not_measured'
        report = {'source_commit': commit, 'target': 'sm_103a', 'device': torch.cuda.get_device_name(),
                  'allocation': admission.mode, 'torch': torch.__version__, 'triton': triton.__version__,
                  'flashinfer_jit_sdk': flashinfer.__version__, 'peer_source_commit': task.PEER_COMMIT,
                  'scope': 'development_correctness_only; no_Campaign_or_promotion',
                  'passed': passed, 'timing': timing, 'mechanism_equivalence': 'not_established',
                  'rows': rows}
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        result.update(status='passed' if passed else 'failed', validity='valid' if passed else 'invalid',
                      summary=f'{sum(r["passed"] for r in rows)}/{len(rows)} strict peer checks passed; no performance claim',
                      artifacts={'report': 'checks/report.json'})
    except Exception as error:
        result['summary'] = f'{type(error).__name__}: {error}'
        (stage / 'error.txt').write_text(traceback.format_exc())
        result['artifacts']['error'] = 'error.txt'
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
