"""Bounded engineering replay using existing Compiler, Lab builder and evaluator.

No provider Study or promotion is fabricated. Each original Workload and paired
protocol remains authoritative; this script only selects explicit pass parameters
and retains the existing builder/evaluator artifacts outside source.
"""
import argparse
from dataclasses import fields
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, document):
    from open_cake_ir.evaluation.core import _plain_json
    serialized = json.dumps(_plain_json(document), indent=2, allow_nan=False) + '\n'
    with path.open('x') as stream:
        stream.write(serialized)


def retain(root, payloads):
    root.mkdir()
    for role, payload in payloads.items():
        if not role.isidentifier():
            raise ValueError('artifact role is not a file name')
        with (root / role).open('xb') as stream:
            stream.write(payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--predecessor', type=Path, required=True)
    parser.add_argument('--task', action='append', help='retained predecessor task; repeat for a bounded set')
    args = parser.parse_args()
    tasks = tuple(args.task or ('rmsnorm_input_gradient', 'swiglu'))
    if len(set(tasks)) != len(tasks) or any(not task.isidentifier() for task in tasks):
        parser.error('tasks must be unique identifiers')
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists() or any((p / '.git').exists() for p in output.parents):
        raise ValueError('new evidence output must be outside source checkouts')
    output.mkdir(parents=True)
    sys.path[:0] = [str(source), str(source / 'src')]
    from open_cake_ir.compiler import frontend
    from open_cake_ir.evaluation.core import _plain_json
    from open_cake_ir.evaluation.paired import validate_pair_candidates
    from open_cake_ir.lab.bindings import load_baseline_bundle
    from open_cake_ir.lab.environments import CandidateSubmission
    from open_cake_ir.lab.runtime import BoundedBrokerEvaluator, CommandBrokerSubmitter
    from open_cake_ir.tasks.environments import TaskOpenCakeEnvironment
    from open_cake_ir.tasks.workloads import load_workload
    from open_cake_ir.serialization import canonical_json_bytes
    from tools.launch_task import _admit_stack, _triton_builder, _runtime_config

    plans = []
    results = []
    write(output / 'plan.json', {
        'source_commit': subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
        'scope': 'bounded deterministic engineering replay, no provider comparison or Run promotion',
        'tasks': list(tasks), 'widths': [1, 8, 16],
        'selection': 'minimum stable correct search latency, fresh confirmation then NCU',
        'baseline': 'same predecessor sealed artifact per task; report current floor separately',
        'predecessor': str(args.predecessor), 'provider_calls': 0,
    })
    # Build every candidate before any GPU allocation.
    for task in tasks:
        task_root = output / task
        task_root.mkdir()
        old = args.predecessor / task
        workload = load_workload(old / 'workload.json')
        study = json.loads((old / 'study.json').read_text())
        protocol = study['evaluation_protocol']
        document = frontend.read_schedule(old / 'starter.py').document
        if document['roles'][0]['execution_groups'] != [0]:
            raise ValueError('planned width-1 floor differs from retained starter')
        compiler, executor, _, compiler_ref = _admit_stack(source, task_root, workload.target, 'triton')
        executor.admit_host()
        executor.admit_profiler()
        baseline_path = args.predecessor / 'baseline-preflight' / task / 'baseline/candidate.json'
        baseline = load_baseline_bundle(source, str(baseline_path))
        runtime = _runtime_config(task_root, executor, None, 'triton', allocation='gpu_run',
                                 gpu_run=Path('/usr/local/bin/gpu-run'), broker_socket=Path('/tmp/h210fix-gpu.sock'))
        write(task_root / 'binding.json', {'compiler': compiler_ref, 'executor': dict(executor.reference),
              'workload': str(old / 'workload.json'), 'protocol': protocol,
              'fixed_baseline': str(baseline_path), 'runtime': runtime})
        builder = _triton_builder(executor, workload)
        environment = TaskOpenCakeEnvironment(compiler, builder,
            authority_document=study['arms']['open_cake'], workload=workload, case_id='primary', executor=executor)
        broker = runtime['broker']
        submitter = CommandBrokerSubmitter(command=tuple(broker['command']), workload_path=old / 'workload.json',
            workload_sha256=workload.canonical_sha256, protocol_sha256=sha256(canonical_json_bytes(protocol)).hexdigest(),
            cwd=source, executor=executor, compiler_reference=compiler_ref,
            service_user=broker['service_user'], service_group=broker['service_group'],
            timeout_seconds=broker['timeout_seconds'], evaluation_protocol=protocol,
            baseline=baseline, workload_loader=load_workload)
        evaluator = BoundedBrokerEvaluator(protocol, submitter)
        candidates = []
        for width in (1, 8, 16):
            case_root = task_root / f'w{width}'
            case_root.mkdir()
            if width == 1:
                candidate_document = document
            else:
                # Keep the bound ABI entry point; only Schedule identity and width
                # differ. The pass API permits an explicit unchanged entry name.
                specialization = compiler.specialize_triton_warps(document, num_warps=width,
                    schedule_id=document['schedule_id'] + f'-w{width}',
                    entry_point=document['lowering']['entry_point'])
                write(case_root / 'pass.json', {'applied': specialization.applied,
                      'reason': specialization.reason, 'message': specialization.message})
                if not specialization.applied:
                    raise ValueError(f'{task} width {width}: {specialization.message}')
                candidate_document = specialization.schedule
            write(case_root / 'schedule.json', candidate_document)
            submission = CandidateSubmission.seal(environment.media_type, canonical_json_bytes(candidate_document))
            built = environment.build(submission)
            if built.artifact_payloads:
                retain(case_root / 'build-artifacts', built.artifact_payloads)
            write(case_root / 'build.json', {'disposition': built.disposition, 'feedback': dict(built.feedback)})
            if built.launchable is None:
                raise ValueError(f'{task} width {width} refused: {built.feedback}')
            retain(case_root / 'compiled', built.launchable.artifact_payloads)
            validate_pair_candidates(built.launchable, baseline, workload, 'primary')
            candidates.append((width, case_root, built.launchable))
        plans.append((task, task_root, evaluator, candidates))
    write(output / 'builds-complete.json', {'tasks': len(tasks), 'candidates': len(tasks) * 3})

    def evaluate(evaluator, candidate, directory, purpose):
        directory.mkdir()
        started = time.time()
        logical = evaluator.evaluate(candidate, case_id='primary', purpose=purpose)
        attempts = []
        for index, attempt in enumerate(logical.attempts, 1):
            retain(directory / f'attempt-{index}', attempt.artifact_payloads)
            attempts.append({field.name: getattr(attempt, field.name) for field in fields(attempt)
                             if field.name not in {'artifact_payloads', 'receipt'}})
        write(directory / 'attempts.json', attempts)
        receipt = logical.final_receipt
        if receipt is None:
            raise ValueError('No admitted final receipt; original broker observations retained')
        retain(directory / 'receipt-artifacts', receipt.artifact_payloads)
        record = {field.name: getattr(receipt, field.name) for field in fields(receipt)
                  if field.name != 'artifact_payloads'}
        write(directory / 'receipt.json', record)
        print(json.dumps(_plain_json({'evaluation': str(directory.relative_to(output)), 'correct': receipt.correctness_passed,
                          'timing': dict(receipt.timing) if receipt.timing is not None else None,
                          'elapsed_seconds': time.time() - started}), allow_nan=False), flush=True)
        return receipt

    for task, task_root, evaluator, candidates in plans:
        observed = []
        for width, case_root, candidate in candidates:
            receipt = evaluate(evaluator, candidate, case_root / 'search', 'search')
            timing = receipt.timing or {}
            observed.append({'width': width, 'correct': receipt.correctness_passed,
                             'timing': dict(timing)})
        survivors = [row for row in observed if row['correct'] and row['timing'].get('measurement_quality_passed')]
        if not survivors:
            results.append({'task': task, 'search': observed, 'outcome': 'no_correct_stable_candidate'})
            continue
        best = min(survivors, key=lambda r: r['timing']['pooled_median_ms'])
        width, case_root, candidate = next(c for c in candidates if c[0] == best['width'])
        confirm = evaluate(evaluator, candidate, case_root / 'confirmatory', 'confirmatory')
        profile = evaluate(evaluator, candidate, case_root / 'attribution', 'attribution')
        accepted = bool(confirm.correctness_passed and (confirm.timing or {}).get('measurement_quality_passed')
                        and profile.correctness_passed and 'profile' in profile.artifact_payloads)
        results.append({'task': task, 'search': observed, 'selected_width': width,
                        'confirmation': dict(confirm.timing or {}), 'engineering_evaluation_passed': accepted,
                        'profile_path': str(case_root / 'attribution/receipt-artifacts/profile')})
        write(task_root / 'result.json', results[-1])
    write(output / 'result.json', {'scope': 'common evaluator engineering receipts, not provider Study', 'results': results})
    return 0 if len(results) == len(tasks) and all(r.get('engineering_evaluation_passed') for r in results) else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr, flush=True)
        raise
