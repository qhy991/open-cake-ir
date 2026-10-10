#!/usr/bin/env python3
"""Bounded C550 width qualification through existing build and correctness owners.

These are explicit Schedule candidates, not a bypass of the qualified pass API.
No timing, provider search, or transformation promotion is performed here.
"""
import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.serialization import canonical_json_bytes
from open_cake_ir.source_identity import checkout_commit
from open_cake_ir.tasks.workloads import create_task, load_workload

CASES = (('fib_rmsnorm_h4096', 64, 4096, (1, 2, 4)),
         ('fib_fused_add_rmsnorm_h4096', 64, 4096, (1, 2, 4)),
         ('fib_rmsnorm_h128', 32, 128, (1, 4)))


def write(path, value):
    with path.open('xb') as stream:
        stream.write(canonical_json_bytes(value))


class WidthCandidateRefused(ValueError):
    """Retain the existing Compiler's named rejection of an explicit proposal."""
    def __init__(self, assessment):
        self.codes = tuple(f.code for f in assessment.findings
                           if f.blocks_lowering or f.blocks_acceptance)
        super().__init__('Proposed width refused: ' + ', '.join(self.codes))


def width_source(compiler, source, width):
    if type(width) is not int or width not in (1, 2, 4, 8, 16):
        raise ValueError('This qualification fixes execution groups to 1, 2, 4, 8 or 16')
    original = frontend.parse(source).document
    before = compiler.assess(original)
    if not before.lowering_eligible or original['target'] != 'xcore1002':
        raise ValueError('Require an eligible exact C550 starter')
    if len(original['roles']) != 1 or original['roles'][0]['execution_groups'] != [0]:
        raise ValueError('Require the declared one-group starter')
    tree = ast.parse(source)
    roles = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == 'role']
    if len(roles) != 1:
        raise ValueError('Require exactly one author role')
    groups = next(k for k in roles[0].keywords if k.arg == 'execution_groups')
    groups.value = ast.List(elts=[ast.Constant(i) for i in range(width)], ctx=ast.Load())
    candidate = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    document = frontend.parse(candidate).document
    expected = deepcopy(original)
    expected['roles'][0]['execution_groups'] = list(range(width))
    if document != expected:
        raise ValueError('The candidate changed more than execution groups')
    assessed = compiler.assess(document)
    if not assessed.lowering_eligible:
        raise WidthCandidateRefused(assessed)
    def kernel_body(assessment):
        lowered = compiler.lower(assessment)
        node = next(n for n in ast.parse(lowered.source).body if isinstance(n, ast.FunctionDef)
                    and n.name == lowered.toolchain_requirements['kernel_entry_point'])
        return ast.dump(ast.Module(body=node.body, type_ignores=[])), lowered
    body_before, lower_before = kernel_body(before)
    body_after, lower_after = kernel_body(assessed)
    if body_before != body_after:
        raise ValueError('The emitted arithmetic body changed')
    for name in ('signature', 'compile_constants', 'grid'):
        if lower_before.toolchain_requirements[name] != lower_after.toolchain_requirements[name]:
            raise ValueError('The candidate changed ABI, constants or grid')
    options = dict(lower_before.toolchain_requirements['compile_options'])
    options['num_warps'] = width
    if dict(lower_after.toolchain_requirements['compile_options']) != options:
        raise ValueError('The candidate changed another compile option')
    return candidate


def prepare(output, suite='initial'):
    compiler = Compiler.load(ROOT)
    rows = []
    refusals = []
    if suite in {'extension', 'output-loop'}:
        from tools.benchmarks.c550.width_fixtures import extension_cases, output_loop_cases
        cases = extension_cases() if suite == 'extension' else output_loop_cases()
    else:
        cases = []
        for task, batch, hidden, widths in CASES:
            workload, source = create_task(task, backend='triton-metax', rows=batch, columns=hidden)
            cases.append(dict(name=task, task=task, rows=batch, columns=hidden,
                              widths=widths, workload=workload, source=source))
    for row in cases:
        for width in row['widths']:
            name = row['name'] + '-w' + str(width)
            directory = output / name
            directory.mkdir()
            try:
                candidate = width_source(compiler, row['source'], width)
            except WidthCandidateRefused as error:
                if suite not in {'extension', 'output-loop'} or width != 16 or error.codes != ('MACA_WARP_COUNT_UNQUALIFIED',):
                    raise
                record = dict(name=name, task=row['task'], width=width, codes=error.codes,
                              scope='expected exact-route refusal; zero native or device calls')
                write(directory / 'refused.json', record)
                refusals.append(record)
                continue
            write(directory / 'workload.json', row['workload'])
            (directory / 'starter.py').write_text(candidate)
            write(directory / 'schedule.json', frontend.parse(candidate).document)
            rows.append(dict(name=name, task=row['task'], rows=row['rows'],
                             columns=row['columns'], width=width))
    write(output / 'plan.json', dict(source_commit=checkout_commit(ROOT), candidates=rows, suite=suite, refusals=refusals,
        scope='explicit width candidates; promotion is recorded separately',
        reference='unchanged task-owned Workload oracle', timing='none', provider_calls=0))
    return dict(prepared=len(rows), refused=len(refusals), gpu_calls=0)


def build(prepared, output):
    from tools.launch_task import _admit_stack, _prepare_baseline
    from open_cake_ir.tasks.normalization.study import task_run_inputs
    plan = json.loads((prepared / 'plan.json').read_text())
    if plan['source_commit'] != checkout_commit(ROOT):
        raise ValueError('Preparation and native build source commits differ')
    rows = []
    for row in plan['candidates']:
        source_dir = prepared / row['name']
        directory = output / row['name']
        directory.mkdir()
        workload_path = source_dir / 'workload.json'
        starter = source_dir / 'starter.py'
        workload = load_workload(workload_path)
        compiler, executor, host, reference = _admit_stack(ROOT, directory, workload.target, 'triton')
        inputs = task_run_inputs(ROOT, workload, workload_path, starter,
            harness='claude-code', model='glm-5.3', effort='high')
        candidate = _prepare_baseline(ROOT, directory, compiler, executor, host, workload,
            inputs['authoring'], starter.read_text(), reference, 'triton')
        rows.append({**row, 'candidate_bundle': str(candidate), 'workload': str(workload_path)})
        write(directory / 'built.json', rows[-1])
    write(output / 'builds.json', dict(source_commit=checkout_commit(ROOT), candidates=rows))
    return dict(built=len(rows), gpu_calls=0)


def check(built, output, physical, runtime, pci):
    from open_cake_ir.evaluation.local_broker import admit_local_job
    from open_cake_ir.evaluation.triton_metax import observe_local_metax
    from open_cake_ir.evaluation.core import EvaluationProtocol, LoadedTorchTensorCandidate
    from open_cake_ir.tasks.tiles.evaluation import PreparedTensorCase, evaluate_tile_validation_case
    from open_cake_ir.tasks.launch import parse_launch_manifest
    from open_cake_ir.lab.bindings import load_baseline_bundle
    from open_cake_ir.lab.executor import ExecutorRevision
    plan = json.loads((built / 'builds.json').read_text())
    if plan['source_commit'] != checkout_commit(ROOT):
        raise ValueError('Native build and device check source commits differ')
    host = ExecutorRevision.for_target(ROOT, 'xcore1002').admit_host()
    job = admit_local_job('maca', device=physical, runtime_device=runtime,
                         expected_pci=pci, lock_scope='device', queue_seconds=300)
    admission = observe_local_metax('xcore1002', runtime_library=host['runtime_library'])
    results = []
    for row in plan['candidates']:
        directory = output / row['name']
        directory.mkdir()
        workload = load_workload(Path(row['workload']))
        candidate = load_baseline_bundle(ROOT, row['candidate_bundle'])
        manifest = parse_launch_manifest(json.loads(candidate.artifact_payloads['launch_manifest']))
        checks = []
        for case in workload.case_ids:
            case_dir = directory / case
            case_dir.mkdir()
            prepared = PreparedTensorCase(workload, case)
            loaded = LoadedTorchTensorCandidate(candidate, manifest, prepared.inputs, admission)
            try:
                protocol = EvaluationProtocol('c550-width-correctness', 'confirmatory',
                    workload.canonical_sha256, case, 'none')
                receipt = evaluate_tile_validation_case(candidate, workload, protocol, loaded, prepared=prepared)
                resources = dict(loaded.loaded.resources)
                for role, payload in receipt.artifact_payloads.items():
                    if not role.isidentifier():
                        raise ValueError('Invalid artifact role')
                    with (case_dir / role).open('xb') as stream:
                        stream.write(payload)
                check = dict(case=case, passed=receipt.correctness_passed,
                    kernel_calls=receipt.kernel_calls, metrics=dict(receipt.correctness), resources=resources)
            finally:
                loaded.close()
            check['module_closed'] = loaded.loaded.closed
            write(case_dir / 'result.json', check)
            checks.append(check)
            if not checks[-1]['passed']:
                raise ValueError('Original correctness refused; stop further candidates')
        results.append(dict(name=row['name'], width=row['width'], checks=checks))
        write(directory / 'result.json', results[-1])
    return dict(job_id=job, passed=True, candidates=len(results),
        original_checks=sum(len(r['checks']) for r in results), results=results,
        timing='none', provider_calls=0, promotion='not performed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'build', 'check'))
    parser.add_argument('--input', type=Path)
    parser.add_argument('--suite', choices=('initial', 'extension', 'output-loop'), default='initial')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--physical-device', type=int)
    parser.add_argument('--runtime-device', type=int)
    parser.add_argument('--expected-pci')
    args = parser.parse_args()
    if checkout_commit(ROOT) is None:
        parser.error('Require a clean committed source')
    if args.phase != 'prepare' and args.input is None:
        parser.error('Build and check require their prior phase input')
    if args.phase == 'check' and any(v is None for v in (args.physical_device, args.runtime_device, args.expected_pci)):
        parser.error('Device check requires physical/runtime/PCI binding')
    from open_cake_ir.lab.custody import admit_new_campaign_path
    output = admit_new_campaign_path(ROOT, args.output, role='bounded C550 width qualification')
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = (prepare(output, args.suite) if args.phase == 'prepare' else
                  build(args.input, output) if args.phase == 'build' else
                  check(args.input, output, args.physical_device, args.runtime_device, args.expected_pci))
    except Exception as error:
        write(output / 'failure.json', dict(error=str(error), failure_class=type(error).__name__))
        raise
    write(output / 'result.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'results'}))


if __name__ == '__main__':
    main()
