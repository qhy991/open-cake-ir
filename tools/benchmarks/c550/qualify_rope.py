"""Build and check the original RoPE before production math admission.

The original Bench owns input generation, reference computation and comparison.
The adapter only validates arguments, allocates outputs and calls sealed kernels.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from open_cake_ir.tasks.c550_bench.binding import BenchProblem, BENCH_COMMIT
from open_cake_ir.tasks.c550_bench.workload import BenchWorkload
from open_cake_ir.serialization import canonical_json_bytes
from open_cake_ir.source_identity import checkout_commit
from benchmarks.c550.rope import TASK, source_for_workload


def write(path, document):
    from open_cake_ir.cli import _json_projection
    with path.open('xb') as stream:
        stream.write(canonical_json_bytes(_json_projection(document)))


def emission_for(workload):
    from open_cake_ir.compiler import Compiler, Target, frontend
    from open_cake_ir.compiler.ir import Schedule
    from open_cake_ir.compiler.backends.triton import emit, preflight
    from open_cake_ir.compiler.verifier import verify
    raw = frontend.parse(source_for_workload(workload, 'primary')).document
    target = Target.load(ROOT / 'compiler/targets/xcore1002.json')
    if {'maca.sin.f32', 'maca.cos.f32'} & target.instruction_contracts:
        raise ValueError('this pre-admission qualification requires the original closed math Target')
    probe = replace(target, instruction_contracts=target.instruction_contracts | {'maca.sin.f32', 'maca.cos.f32'})
    schedule = Schedule.from_dict(raw)
    findings = (*verify(schedule, probe), *preflight(schedule, probe))
    if any(item.blocks_lowering for item in findings):
        raise ValueError('the original RoPE candidate fails its synthetic math capability probe')
    real = Compiler.load(ROOT).assess(raw)
    if real.lowering_eligible:
        raise ValueError('production unexpectedly admitted the pre-admission RoPE')
    return raw, emit(schedule, probe)


def build(args, result):
    from hashlib import sha256
    from open_cake_ir.compiler import Compiler
    from open_cake_ir.lab import CandidateSubmission, OpenCakeEnvironment
    from open_cake_ir.lab.build import BuildRequest, TritonToolchainBuilder
    from open_cake_ir.lab.executor import ExecutorRevision
    from open_cake_ir.lab.triton_build import IsolatedTritonCompiler
    from open_cake_ir.evaluation.paired import candidate_identity
    from launch_task import _triton_toolchain_config
    problem = BenchProblem.open(args.bench_root, TASK)
    gate = Compiler.load(ROOT).check_corpus()
    write(args.output / 'compiler-gate.json', gate)
    if not gate.passed:
        raise ValueError('Corpus Gate failed')
    executor = ExecutorRevision.for_target(ROOT, 'xcore1002')
    executor.admit_host()
    isolated = IsolatedTritonCompiler(**_triton_toolchain_config(executor))
    isolated.check_executor(executor, author_workspace=args.output)
    result.update(bench_commit=BENCH_COMMIT, task=TASK, cases=[])
    for row in problem.workloads:
        directory = args.output / row.uuid
        directory.mkdir()
        document = problem.workload_document(row.uuid)
        workload = BenchWorkload(document)
        raw, emission = emission_for(workload)
        requirements = {'compiler': 'triton', 'source_language': 'python', 'target': 'xcore1002', **emission.toolchain}
        source = emission.source.encode()
        submission = CandidateSubmission.seal(OpenCakeEnvironment.media_type, canonical_json_bytes(raw))
        request = BuildRequest(submission.sha256, source, 'lowered_source', sha256(source).hexdigest(),
                               'xcore1002', requirements['kernel_entry_point'], requirements)
        abi = tuple((a.name, a.shape, a.dtype, a.mode) for a in workload.tensor_abi('primary'))
        candidate = TritonToolchainBuilder(workload=workload, case_id='primary', isolated_compiler=isolated).build_stage(request, abi)
        write(directory / 'workload.json', document)
        write(directory / 'candidate.json', candidate_identity(candidate))
        for role, payload in candidate.artifact_payloads.items():
            with (directory / (role + '.bin')).open('xb') as stream:
                stream.write(payload)
        result['cases'].append({'uuid': row.uuid, 'shape': list(abi[0][1])})
    result['passed'] = True


def read_candidate(built_root, uuid):
    from open_cake_ir.evaluation.paired import candidate_from_identity
    from open_cake_ir.evaluation.core import TensorLaunchManifest
    from open_cake_ir.evaluation.loaders import check_candidate_authority
    from open_cake_ir.tasks.evaluate import _input_path
    built_root = built_root.resolve(strict=True)
    if not isinstance(uuid, str) or Path(uuid).name != uuid or uuid in ('', '.', '..'):
        raise ValueError('RoPE case directory differs')
    directory = built_root / uuid
    if directory.is_symlink() or not directory.is_dir() or built_root not in directory.resolve().parents:
        raise ValueError('RoPE case directory escapes its built root')
    directory = directory.resolve()
    identity = json.loads(_input_path(directory, 'candidate.json', 'RoPE candidate').read_text())
    candidate = candidate_from_identity(identity, {role: _input_path(directory, role + '.bin', role).read_bytes()
                                                  for role in identity['artifact_roles']})
    manifest = TensorLaunchManifest.from_dict(json.loads(candidate.artifact_payloads['launch_manifest']))
    check_candidate_authority(candidate, candidate.artifact_payloads['mcfatbin'], 'mcfatbin', manifest)
    return candidate, manifest


def bind_original_build(built_root, bench_root, commit):
    from open_cake_ir.compiler import Target
    from open_cake_ir.lab import CandidateSubmission, OpenCakeEnvironment
    from open_cake_ir.evaluation.program import check_triton_launch_record
    from open_cake_ir.tasks.evaluate import _input_path
    built_root = built_root.resolve(strict=True)
    index = json.loads(_input_path(built_root, 'result.json', 'RoPE build').read_text())
    problem = BenchProblem.open(bench_root, TASK)
    target = Target.load(ROOT / 'compiler/targets/xcore1002.json')
    if (index.get('source_commit') != commit or not index.get('passed') or index.get('bench_commit') != BENCH_COMMIT
            or index.get('task') != TASK or [c['uuid'] for c in index['cases']] != [c.uuid for c in problem.workloads]):
        raise ValueError('RoPE build must bind this source and all ordered original workloads')
    shapes = set()
    for case in index['cases']:
        workload = BenchWorkload(problem.workload_document(case['uuid']))
        raw, emission = emission_for(workload)
        candidate, manifest = read_candidate(built_root, case['uuid'])
        manifest.check_workload(workload, 'primary')
        submission = CandidateSubmission.seal(OpenCakeEnvironment.media_type, canonical_json_bytes(raw))
        shape = tuple(workload.tensor_abi('primary')[0].shape)
        if (case['shape'] != list(shape) or shape in shapes or candidate.candidate_sha256 != submission.sha256
                or candidate.artifact_payloads['lowered_source'] != emission.source.encode()
                or manifest.kernel_name != emission.toolchain['kernel_entry_point']
                or list(manifest.grid) != emission.toolchain['grid']
                or tuple(manifest.block) != (emission.toolchain['compile_options']['num_warps'] * target.warp_size, 1, 1)):
            raise ValueError('RoPE artifact or unique original shape binding differs')
        check_triton_launch_record(candidate, manifest, candidate.artifact_roles['lowered_source'])
        shapes.add(shape)
    return problem


class NativeRope:
    """One sealed native call per original Bench invocation, with explicit teardown."""
    def __init__(self, built_root, trace_path):
        from open_cake_ir.lab.executor import ExecutorRevision
        from open_cake_ir.evaluation.triton_metax import observe_local_metax
        from open_cake_ir.tasks.evaluate import _input_path
        self.root, self.trace = Path(built_root).resolve(strict=True), Path(trace_path)
        index = json.loads(_input_path(self.root, 'result.json', 'RoPE build').read_text())
        if (not index.get('passed') or index.get('source_commit') != checkout_commit(ROOT)
                or index.get('bench_commit') != BENCH_COMMIT or index.get('task') != TASK):
            raise ValueError('RoPE runtime requires the current complete build')
        self.cases = {tuple(item['shape']): item['uuid'] for item in index['cases']}
        host = ExecutorRevision.for_target(ROOT, 'xcore1002').admit_host()
        self.admission = observe_local_metax('xcore1002', runtime_library=host['runtime_library'])

    def run(self, position_ids, inv_freq, attention_scaling):
        import torch
        from open_cake_ir.evaluation.metax_driver import LoadedMetaxCandidate
        shape = tuple(position_ids.shape)
        if shape not in self.cases or type(attention_scaling) not in (float, int) or attention_scaling != 1.0:
            raise ValueError('RoPE input shape or original scalar differs')
        candidate, manifest = read_candidate(self.root, self.cases[shape])
        before = [value.view(torch.uint8).cpu().clone() for value in (position_ids, inv_freq)]
        out = torch.full((*shape, 128, 2), float('nan'), dtype=torch.bfloat16, device='cuda:0')
        loaded = LoadedMetaxCandidate.load(candidate, manifest, self.admission)
        primary = None
        unchanged = False
        try:
            loaded.launch((position_ids, inv_freq, out), tensor_contract=manifest,
                          stream=int(torch.cuda.current_stream().cuda_stream))
            torch.cuda.synchronize()
            unchanged = all(torch.equal(value.view(torch.uint8).cpu(), prior)
                            for value, prior in zip((position_ids, inv_freq), before, strict=True))
            if not unchanged:
                raise ValueError('RoPE candidate changed an original input')
        except BaseException as error:
            primary = error
        finally:
            try:
                loaded.close(synchronize=torch.cuda.synchronize, primary=primary)
            except BaseException as error:
                primary = error
            with self.trace.open('a') as stream:
                stream.write(json.dumps({'uuid': self.cases[shape], 'kernel_calls': loaded.launch_calls,
                    'input_unchanged': unchanged, 'module_closed': loaded.closed, 'error': str(primary) if primary else None}) + '\n')
        if primary is not None:
            raise primary
        return out


def evaluate(args, result):
    from open_cake_ir.evaluation.local_broker import admit_local_job
    problem = bind_original_build(args.built, args.bench_root, result['source_commit'])
    trace = args.output / 'native-calls.jsonl'
    trace.touch(exist_ok=False)
    adapter = args.output / 'candidate.py'
    adapter.write_text('from benchmarks.c550.qualify_rope import NativeRope\n'
                       f'_native = NativeRope({str(args.built.resolve())!r}, {str(trace)!r})\n'
                       'run = _native.run\n')
    result['job_id'] = admit_local_job('maca', device=args.physical_device, runtime_device=args.runtime_device,
        expected_pci=args.expected_pci, lock_scope='device', queue_seconds=300)
    suite = problem.api.document('suite.json')
    report = problem.api.check_problem(problem.api.task_record(TASK), device='cuda:0', candidate_path=adapter,
        symbol='run', reference_selfcheck=False, workload_scope='all', rounds=suite['correctness_rounds'],
        output=args.output / 'original-bench.json', seed=suite['seed'], threads=4)
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    expected = [item.uuid for item in problem.workloads for _ in range(suite['correctness_rounds'])]
    native_ok = ([row['uuid'] for row in rows] == expected and all(row['kernel_calls'] == 1
        and row['module_closed'] and row['input_unchanged'] and row['error'] is None for row in rows))
    result.update(passed=report['status'] == 'passed' and report['full_device_correctness'] and native_ok,
        original_checks=len(report['cases']), original_passed=sum(row['passed'] for row in report['cases']),
        full_device_correctness=report['full_device_correctness'], native_checks_passed=native_ok,
        native_kernel_calls=sum(row['kernel_calls'] for row in rows), target_admission_changed=False,
        performance='not_measured')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    builder = sub.add_parser('build')
    runner = sub.add_parser('evaluate')
    runner.add_argument('--built', type=Path, required=True)
    runner.add_argument('--physical-device', type=int, required=True)
    runner.add_argument('--runtime-device', type=int, required=True)
    runner.add_argument('--expected-pci', required=True)
    for command in (builder, runner):
        command.add_argument('--bench-root', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    commit = checkout_commit(ROOT)
    if commit is None or os.environ.get('METAL_BROKER_LOCK_FD') or os.environ.get('GPUQ_JOB_ID'):
        parser.error('qualification needs a clean commit and starts outside a GPU allocation')
    args.output = args.output.resolve()
    if args.output == ROOT or ROOT in args.output.parents:
        parser.error('evidence must stay outside source')
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'source_commit': commit, 'passed': False,
              'scope': 'original RoPE correctness under a declared math probe; no optimization or performance'}
    try:
        (build if args.command == 'build' else evaluate)(args, result)
    except Exception as error:
        result.update(error=str(error), failure_class=type(error).__name__)
        (args.output / 'failure.txt').write_text(traceback.format_exc())
    write(args.output / 'result.json', result)
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
