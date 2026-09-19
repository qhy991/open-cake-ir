#!/usr/bin/env python3
"""Development-only B300 checks of generated FlashInfer starters, outside Campaigns.

Uses the task's CPU oracle and every required input distribution. Results are not
sealed common-Evaluation receipts and do not authorize promotion or scientific claims.
Run only inside the cluster's exclusive GPU allocation after software gates pass.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.frontend import parse
from open_cake_ir.source_identity import checkout_commit
from open_cake_ir.tasks.solx_fib.catalog import TASK_IDS, task_owner
from open_cake_ir.tasks.workloads import create_task, materialize_case, reference_outputs
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.evaluation.benchmark import StrictCuptiBenchmark
from open_cake_ir.evaluation.admission import observe_exclusive_cuda


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', choices=TASK_IDS, action='append')
    parser.add_argument('--timing', action='store_true', help='CUPTI cold-L2 timing after all cases pass')
    args = parser.parse_args()
    commit = checkout_commit(ROOT)
    output = args.output.resolve()
    if any((parent / '.git').exists() for parent in (output, *output.parents)):
        raise ValueError('test output must stay outside source checkouts')
    output.mkdir(parents=True, exist_ok=False)
    import torch
    import triton
    admission = observe_exclusive_cuda('sm_103a')
    if torch.cuda.device_count() != 1:
        raise ValueError('run under an exclusive one-GPU allocation')
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    compiler = Compiler.load(ROOT, ROOT/'compiler/revision.json')
    # Match the Target document rather than accepting another CUDA device.
    target_document = json.loads((ROOT/'compiler/targets/sm_103a.json').read_text())
    expected = tuple(target_document['compute_capability'])
    if capability != expected or name not in target_document['device_names']:
        raise ValueError(f'exact B300 required: expected {expected}, got {name} {capability}')
    summary = {'source_commit': commit, 'target': 'sm_103a', 'allocation': admission.mode,
               'device': name, 'capability': capability, 'torch': torch.__version__,
               'triton': triton.__version__, 'scope': 'development_test_not_Campaign_or_promotion', 'tasks': []}
    dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}
    for task_id in args.task or TASK_IDS:
        task, owner = task_owner(task_id)
        row = {'task_id': task_id, 'cases': [], 'timing': 'not_requested'}
        if owner is None:
            row.update(status='not_integrated')
            summary['tasks'].append(row)
            continue
        try:
            spec = owner.SPECS[task]
            rows = owner.default_rows(task) if hasattr(owner, 'default_rows') else min(spec['batches'])
            columns = spec['hidden'] if 'hidden' in spec else spec['N']
            document, source = create_task(task, backend='triton-b300', rows=rows, columns=columns)
            workload = WorkloadContract(document)
            assessment = compiler.assess(parse(source).document)
            lowering = compiler.lower(assessment)
            directory = output/task_id
            directory.mkdir()
            (directory/'workload.json').write_text(json.dumps(document, indent=2)+'\n')
            (directory/'starter.py').write_text(source)
            path = directory/'lowered.py'
            path.write_text(lowering.source)
            module_spec = importlib.util.spec_from_file_location('fib_'+task_id, path)
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[module_spec.name] = module
            module_spec.loader.exec_module(module)
            entry = getattr(module, lowering.route.entry_point)
            validation = document['validation']
            timing_inputs = None
            for case_id in workload.case_ids:
                values = materialize_case(workload, case_id)
                expected_values = reference_outputs(workload, case_id, values)['out']
                abi = workload.tensor_abi(case_id)
                inputs = {}
                for argument in abi:
                    if argument.mode == 'input':
                        inputs[argument.name] = torch.tensor(values[argument.name], dtype=dtype[argument.dtype], device='cuda').reshape(argument.shape)
                out_arg = next(a for a in abi if a.mode == 'output')
                expected_tensor = torch.tensor(expected_values, dtype=dtype[out_arg.dtype]).reshape(out_arg.shape)
                snapshots = {key: value.clone() for key,value in inputs.items()}
                out = torch.full(out_arg.shape, float('nan'), dtype=dtype[out_arg.dtype], device='cuda')
                result = entry(**inputs, out=out)
                torch.cuda.synchronize()
                if result.data_ptr() != out.data_ptr():
                    raise ValueError('entry replaced the supplied output')
                torch.testing.assert_close(result.cpu(), expected_tensor, atol=validation['atol'],
                                           rtol=validation['rtol'], equal_nan=False)
                if not all(torch.equal(inputs[key], old) for key,old in snapshots.items()):
                    raise ValueError('candidate mutated an input')
                row['cases'].append({'case_id': case_id, 'status': 'passed', 'elements': out.numel()})
                if case_id == validation['primary_case']:
                    timing_inputs = inputs, out
                del values, expected_values, snapshots, inputs, out, result
                print(task_id, case_id, 'passed', flush=True)
            if args.timing:
                import flashinfer.testing as helper
                inputs, out = timing_inputs
                samples = list(StrictCuptiBenchmark(helper)(lambda: entry(**inputs, out=out),
                    dry_run_iters=5, repeat_iters=25, cold_l2_cache=True, use_cuda_graph=False))
                if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
                    raise ValueError('CUPTI returned no positive samples')
                row['timing'] = {'source': 'cupti', 'cold_l2_cache': True, 'samples_ms': samples,
                                 'median_ms': statistics.median(samples)}
            row.update(status='passed', workload_id=workload.workload_id)
        except Exception as error:
            row.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
            print(task_id, row['error'], flush=True)
        summary['tasks'].append(row)
        (output/'progress.json').write_text(json.dumps(summary, indent=2)+'\n')
    (output/'report.json').write_text(json.dumps(summary, indent=2)+'\n')
    return 0 if all(row['status']=='passed' for row in summary['tasks']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
