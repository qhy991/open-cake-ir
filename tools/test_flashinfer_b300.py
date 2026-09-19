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
import os
import socket
import struct
from hashlib import sha256
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
from open_cake_ir.evaluation.admission import observe_exclusive_cuda, _observe_cuda_device
from open_cake_ir.compiler.target import declared_target


def _broker_request(path: Path, request: dict) -> dict:
    """Read the existing broker status/receipt protocol; never create a lease here."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(10)
        client.connect(str(path))
        peer = struct.unpack("3i", client.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        if peer[1] not in {0, os.getuid()}:
            raise ValueError("broker peer UID differs")
        with client.makefile("rwb") as stream:
            stream.write((json.dumps(request) + "\n").encode())
            stream.flush()
            return json.loads(stream.readline())


def _admit_receipt(socket_path: Path, receipt_path: Path):
    """Bind this development process to the broker's live exclusive receipt.

    Broker v0.6 supplies a receipt but does not export GPUQ_JOB_ID. Query the issuer
    and bind the existing launch digest once at this handoff, without inventing an ID
    or modifying the production broker or the worker's environment admission rule.
    """
    receipt = json.loads(receipt_path.read_text())
    job_id = receipt.get('job_id')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if (receipt.get('schema') != 'gpuq.admission-receipt.v1'
            or not isinstance(job_id, str) or not job_id.startswith('gpuq-')
            or receipt.get('mode') != 'exclusive' or receipt.get('gpu_count') != 1
            or not isinstance(receipt.get('gpu_ids'), list) or len(receipt['gpu_ids']) != 1
            or visible != str(receipt['gpu_ids'][0]) or receipt.get('cwd') != str(ROOT)):
        raise ValueError('development broker receipt allocation differs')
    # This is the existing broker launch-identity boundary, not a new digest inventory.
    argv_bytes = json.dumps(sys.orig_argv, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    if (receipt.get('argv_count') != len(sys.orig_argv)
            or receipt.get('argv_sha256') != sha256(argv_bytes).hexdigest()
            or receipt.get('resolved_executable') != str(Path(sys.executable).resolve())):
        raise ValueError('development broker receipt command differs')
    issued = _broker_request(socket_path, {'op': 'receipt', 'job_id': job_id})
    if issued.get('ok') is not True or issued.get('receipt') != receipt:
        raise ValueError('development receipt differs from its live issuer')
    snapshot = _broker_request(socket_path, {'op': 'status'}).get('snapshot', {})
    jobs = [job for job in snapshot.get('running', []) if job.get('job_id') == job_id]
    if (snapshot.get('probe_error') or snapshot.get('instance_id') != receipt.get('broker_instance_id')
            or len(jobs) != 1 or jobs[0].get('state') != 'running'
            or jobs[0].get('mode') != 'exclusive' or jobs[0].get('gpu_ids') != receipt['gpu_ids']):
        raise ValueError('development broker lease is not live and exclusive')
    return _observe_cuda_device(declared_target('sm_103a'), visible, job_id, 'exclusive')


def _test_launch_plan(owner, task_id, task, compiler, output, torch, timing_requested):
    """Compile every stage through Cake, then check the complete ordered candidate."""
    row = {'task_id': task_id, 'route': 'cake_launch_plan', 'cases': [], 'variants': [],
           'timing': 'whole_plan_interval_unqualified' if timing_requested else 'not_requested'}
    types = {'fp16':torch.float16, 'bf16':torch.bfloat16, 'fp32':torch.float32,
             'int32':torch.int32, 'fp8_e4m3':torch.float8_e4m3fn}
    for variant in owner.VARIANTS:
        document = owner.workload_document(task, variant=variant)
        workload = WorkloadContract(document)
        author = owner.author_plan(workload)
        plan = author.finish()
        compiled = plan.compile(compiler)
        directory = output/task_id/variant
        directory.mkdir(parents=True)
        (directory/'workload.json').write_text(json.dumps(document,indent=2)+'\n')
        (directory/'launch-plan.json').write_bytes(plan.document_bytes)
        entries = {}
        for stage, lowering in zip(plan.stages, compiled.lowerings, strict=True):
            (directory/(stage.name+'.cake.py')).write_text(author.sources[stage.name])
            path = directory/(stage.name+'.triton.py')
            path.write_text(lowering.source)
            module_spec = importlib.util.spec_from_file_location('fib_'+task_id+'_'+variant+'_'+stage.name,path)
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[module_spec.name] = module
            module_spec.loader.exec_module(module)
            entries[stage.name] = getattr(module,lowering.route.entry_point)
        for case_id in workload.case_ids:
            cpu_inputs = owner.materialize_tensors(workload,case_id)
            expected = owner.reference_tensors(workload,case_id,cpu_inputs)
            inputs = {name:value.to(device='cuda:0') for name,value in cpu_inputs.items()}
            snapshots = {name:value.clone() for name,value in inputs.items()}
            def allocate(name,spec):
                poison = -(2**31) if spec.dtype.value=='int32' else float('nan')
                return torch.full(spec.shape,poison,dtype=types[spec.dtype.value],device='cuda:0')
            def check_tensor(value,spec):
                if (tuple(value.shape)!=spec.shape or value.dtype!=types[spec.dtype.value]
                        or not value.is_cuda or value.device.index!=0 or not value.is_contiguous()):
                    raise ValueError('launch plan platform tensor binding differs')
            def storage_span(value):
                return str(value.device), value.data_ptr(), value.data_ptr()+value.numel()*value.element_size()
            def context():
                return torch.cuda.current_device(),torch.cuda.current_stream().cuda_stream
            prepared = compiled.prepare(inputs,allocate=allocate,check_tensor=check_tensor,
                storage_span=storage_span,execution_context=context,load_kernel=lambda name,lowering:entries[name])
            result = prepared.run()
            torch.cuda.synchronize()
            if prepared.launch_calls!=len(plan.stages):
                raise ValueError('the invocation did not execute every launch-plan stage')
            validation = document['validation']
            if set(result)!=set(expected):raise ValueError('launch plan public outputs differ')
            for name, reference in expected.items():
                rule = validation['outputs'][name] if validation['comparison']=='per_output' else validation
                torch.testing.assert_close(result[name].cpu(), reference, atol=rule['atol'], rtol=rule['rtol'],
                                           equal_nan=rule['comparison']=='elementwise_atol_rtol_ieee')
            if not all(torch.equal(inputs[name].view(torch.uint8),old.view(torch.uint8))
                       for name,old in snapshots.items()):
                raise ValueError('ordered candidate mutated a public input')
            row['cases'].append({'variant':variant,'case_id':case_id,'status':'passed',
                'elements':sum(value.numel() for value in result.values()),'stage_launches':prepared.launch_calls,
                'expected_nonfinite':{name:{'nan':int(torch.isnan(value).sum()),'negative_infinity':int(torch.isneginf(value).sum())}
                                      for name,value in expected.items()}})
            print(task_id,variant,case_id,'passed',flush=True)
            del prepared,inputs,snapshots,cpu_inputs,expected,result
        row['variants'].append({'variant':variant,'workload_id':workload.workload_id,'stages':len(plan.stages),
                                'allocated_bytes':plan.allocated_bytes})
    row['status']='passed'
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', choices=TASK_IDS, action='append')
    parser.add_argument('--timing', action='store_true', help='CUPTI cold-L2 timing after all cases pass')
    parser.add_argument('--broker-socket', type=Path)
    parser.add_argument('--admission-receipt', type=Path)
    args = parser.parse_args()
    if bool(args.broker_socket) != bool(args.admission_receipt):
        parser.error('broker socket and admission receipt must be supplied together')
    commit = checkout_commit(ROOT)
    output = args.output.resolve()
    if any((parent / '.git').exists() for parent in (output, *output.parents)):
        raise ValueError('test output must stay outside source checkouts')
    output.mkdir(parents=True, exist_ok=False)
    import torch
    import triton
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    admission = (_admit_receipt(args.broker_socket, args.admission_receipt)
                 if args.admission_receipt else observe_exclusive_cuda('sm_103a'))
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
            if hasattr(owner, 'author_plan'):
                row = _test_launch_plan(owner, task_id, task, compiler, output, torch, args.timing)
                summary['tasks'].append(row)
                (output/'progress.json').write_text(json.dumps(summary,indent=2)+'\n')
                continue
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
