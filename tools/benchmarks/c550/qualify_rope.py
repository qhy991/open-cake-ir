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
from open_cake_ir.tasks.c550_bench.binding import BenchProblem, BENCH_COMMIT, validate_oracle_numerics
from open_cake_ir.tasks.c550_bench.workload import BenchWorkload
from open_cake_ir.serialization import canonical_json_bytes
from open_cake_ir.source_identity import checkout_commit
from benchmarks.c550.rope import TASK, source_for_workload
from benchmarks.c550.native_bridge import BoundCase
from qualify_c550_bench_native import evaluate_original


def write(path, document):
    from open_cake_ir.cli import _json_projection
    with path.open('xb') as stream:
        stream.write(canonical_json_bytes(_json_projection(document)))


def read_oracle_numerics(path):
    from open_cake_ir.lab.bindings import external_file
    path = external_file(ROOT, str(path), 'RoPE oracle numerics observation')
    return validate_oracle_numerics(json.loads(path.read_text()))


def emission_for(workload):
    from open_cake_ir.compiler import Compiler, Target, frontend
    from open_cake_ir.compiler.ir import Schedule
    from open_cake_ir.compiler.backends.triton import emit, preflight
    from open_cake_ir.compiler.verifier import verify
    raw = frontend.parse(source_for_workload(workload, 'primary')).document
    target = Target.load(ROOT / 'compiler/targets/xcore1002.json')
    if {'maca.sin.f32', 'maca.cos.f32', 'triton.dot.fp32_tf32'} & target.instruction_contracts:
        raise ValueError('this pre-admission qualification requires the original closed math Target')
    probe = replace(target, instruction_contracts=target.instruction_contracts | {'maca.sin.f32', 'maca.cos.f32'})
    schedule = Schedule.from_dict(raw)
    findings = (*verify(schedule, probe), *preflight(schedule, probe))
    if any(item.blocks_lowering for item in findings):
        raise ValueError('the original RoPE candidate fails its synthetic math capability probe')
    real = Compiler.load(ROOT).assess(raw)
    if {item.code for item in real.findings if item.blocks_lowering} != {'TARGET_INSTRUCTION_UNSUPPORTED'}:
        raise ValueError('production RoPE refusal differs from the closed math Target')
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
    policy = read_oracle_numerics(args.oracle_numerics)
    problem = BenchProblem.open(args.bench_root, TASK)
    gate = Compiler.load(ROOT).check_corpus()
    write(args.output / 'compiler-gate.json', gate)
    if not gate.passed:
        raise ValueError('Corpus Gate failed')
    executor = ExecutorRevision.for_target(ROOT, 'xcore1002')
    executor.admit_host()
    isolated = IsolatedTritonCompiler(**_triton_toolchain_config(executor))
    isolated.check_executor(executor, author_workspace=args.output)
    result.update(bench_commit=BENCH_COMMIT, task=TASK, target='xcore1002', oracle_numerics=policy, cases=[])
    for row in problem.workloads:
        directory = args.output / row.uuid
        directory.mkdir()
        document = problem.workload_document(row.uuid, oracle_numerics=policy)
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


def bind_original_build(built_root, bench_root, commit, *, oracle_numerics):
    from open_cake_ir.compiler import Target
    from open_cake_ir.lab import CandidateSubmission, OpenCakeEnvironment
    from open_cake_ir.evaluation.program import check_triton_launch_record
    from open_cake_ir.compiler.metax_toolchain import native_pointer_parameters
    from open_cake_ir.tasks.evaluate import _input_path
    built_root = built_root.resolve(strict=True)
    policy = validate_oracle_numerics(oracle_numerics)
    index = json.loads(_input_path(built_root, 'result.json', 'RoPE build').read_text())
    problem = BenchProblem.open(bench_root, TASK)
    target = Target.load(ROOT / 'compiler/targets/xcore1002.json')
    expected_uuids = [row.uuid for row in problem.workloads]
    if (not commit or len(expected_uuids) != 16 or len(set(expected_uuids)) != 16
            or index.get('source_commit') != commit or index.get('passed') is not True
            or index.get('bench_commit') != BENCH_COMMIT or index.get('target') != 'xcore1002'
            or index.get('oracle_numerics') != policy or index.get('task') != TASK
            or [c['uuid'] for c in index['cases']] != expected_uuids):
        raise ValueError('RoPE build must bind this source and all ordered original workloads')
    shapes, cases = set(), []
    for case in index['cases']:
        workload = BenchWorkload(problem.workload_document(case['uuid'], oracle_numerics=policy))
        raw, emission = emission_for(workload)
        candidate, manifest = read_candidate(built_root, case['uuid'])
        original_path = _input_path(built_root, case['uuid'] + '/workload.json', 'RoPE original Workload')
        if json.loads(original_path.read_text()) != workload.document:
            raise ValueError('retained RoPE original Workload or numerical policy differs')
        manifest.check_workload(workload, 'primary')
        submission = CandidateSubmission.seal(OpenCakeEnvironment.media_type, canonical_json_bytes(raw))
        shape = tuple(workload.tensor_abi('primary')[0].shape)
        if (case['shape'] != list(shape) or shape in shapes or candidate.candidate_sha256 != submission.sha256
                or candidate.artifact_payloads['lowered_source'] != emission.source.encode()
                or manifest.kernel_name != emission.toolchain['kernel_entry_point']
                or list(manifest.grid) != emission.toolchain['grid']
                or tuple(manifest.block) != (emission.toolchain['compile_options']['num_warps'] * target.warp_size, 1, 1)
                or manifest.aligned_variant or manifest.pointer_alignments):
            raise ValueError('RoPE artifact or unique original shape binding differs')
        check_triton_launch_record(candidate, manifest, candidate.artifact_roles['lowered_source'])
        hidden = native_pointer_parameters(candidate.artifact_payloads['mcfatbin'], target.architecture,
                                           manifest.kernel_name) - len(manifest.tensor_abi)
        if hidden not in (0, 2) or hidden != manifest.hidden_null_pointer_parameters:
            raise ValueError('RoPE native pointer ABI differs')
        shapes.add(shape)
        cases.append(BoundCase(case['uuid'], workload, candidate, manifest))
    return problem, tuple(cases)


def evaluate(args, result):
    policy = read_oracle_numerics(args.oracle_numerics)
    problem, cases = bind_original_build(args.built, args.bench_root, result['source_commit'],
                                         oracle_numerics=policy)
    result.update(evaluate_original(problem, cases, args.output,
        physical_device=args.physical_device, runtime_device=args.runtime_device,
        expected_pci=args.expected_pci))
    result.update(oracle_numerics=policy, target_admission_changed=False)


def main(argv=None):
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
        command.add_argument('--oracle-numerics', type=Path, required=True)
    args = parser.parse_args(argv)
    commit = checkout_commit(ROOT)
    if commit is None or os.environ.get('METAL_BROKER_LOCK_FD') or os.environ.get('GPUQ_JOB_ID'):
        parser.error('qualification needs a clean commit and starts outside a GPU allocation')
    from open_cake_ir.lab.custody import admit_new_campaign_path
    args.output = admit_new_campaign_path(ROOT, args.output, role='bounded original RoPE qualification')
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'source_commit': commit, 'passed': False,
              'target': 'xcore1002', 'target_admission_changed': False,
              'scope': 'bounded original RoPE under a synthetic math probe; no optimization or performance'}
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
