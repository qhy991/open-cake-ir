#!/usr/bin/env python3
"""Build a sealed Program and qualify tensor-oracle cases with native HIP or MACA execution.

Build runs in the captured CPU-only compilation environment. Each evaluate command
prepares exactly one original CPU case before acquiring the existing local broker;
the process retains the allocation to exit. No timer, profiler or Run promotion is
implied by these correctness receipts.
"""
import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]

from open_cake_ir.cli import _json_projection
from open_cake_ir.compiler import Compiler, Program
from open_cake_ir.compiler.target import CodeObject
from open_cake_ir.evaluation.core import EvaluationProtocol
from open_cake_ir.evaluation.local_broker import admit_local_job
from open_cake_ir.evaluation.paired import candidate_from_identity, candidate_identity
from open_cake_ir.evaluation.platforms import platform_for
from open_cake_ir.lab import CandidateSubmission, OpenCakeEnvironment
from open_cake_ir.lab.bindings import CURRENT_RELEASE_BINDING, resolve_executor
from open_cake_ir.serialization import canonical_json_bytes
from open_cake_ir.source_identity import checkout_commit
from open_cake_ir.tasks.evaluate import _input_path
from open_cake_ir.tasks.launch import parse_launch_manifest
from open_cake_ir.tasks.program_evaluation import PreparedProgramCase, evaluate_program_case
from open_cake_ir.tasks.workloads import load_workload


def write(path, value):
    with path.open('xb') as stream:
        stream.write(canonical_json_bytes(_json_projection(value)))


def build(args, result):
    from launch_task import _triton_toolchain_config
    from open_cake_ir.lab.build import TritonToolchainBuilder, build_program_candidate
    from open_cake_ir.lab.triton_build import IsolatedTritonCompiler
    workload = load_workload(args.workload)
    program = Program.from_dict(json.loads(args.program.read_text()))
    compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')
    gate = compiler.check_corpus()
    write(args.output / 'compiler-gate.json', gate)
    if not gate.passed:
        raise ValueError('Compiler Corpus Gate failed')
    executor = resolve_executor(ROOT, CURRENT_RELEASE_BINDING, 'tensor Program build', template=True, target=workload.target)
    executor.admit_host()
    isolated = IsolatedTritonCompiler(**_triton_toolchain_config(executor))
    isolated.check_executor(executor, author_workspace=args.output)
    builder = TritonToolchainBuilder(workload=workload, case_id='primary', isolated_compiler=isolated)
    submission = CandidateSubmission.seal(OpenCakeEnvironment.media_type, program.document_bytes)
    try:
        candidate = build_program_candidate(compiler.lower_program(program), builder,
            candidate_sha256=submission.sha256, workload=workload, case_id='primary')
    except Exception as error:
        write(args.output / 'build-feedback.json', {'stage': 'build', 'error': str(error)})
        raise
    write(args.output / 'build-feedback.json', {'stage': 'built',
        'program_stages': [stage.name for stage in program.stages],
        'cost_model_coverage': 'whole_program_unmodeled'})
    write(args.output / 'workload.json', workload.document)
    write(args.output / 'program.json', program.document)
    for role, payload in candidate.artifact_payloads.items():
        with (args.output / (role + '.bin')).open('xb') as stream:
            stream.write(payload)
    write(args.output / 'candidate.json', candidate_identity(candidate))
    result.update(phase='build_complete', passed=True, target=workload.target,
                  workload_id=workload.workload_id, kernels_per_call=candidate.kernels_per_call)


def evaluate(args, result):
    workspace = args.built.resolve(strict=True)
    identity = json.loads((workspace / 'candidate.json').read_text())
    payloads = {role: _input_path(workspace, role + '.bin', role).read_bytes()
                for role in identity['artifact_roles']}
    candidate = candidate_from_identity(identity, payloads)
    workload = load_workload(workspace / 'workload.json')
    manifest = parse_launch_manifest(json.loads(payloads['launch_manifest']))
    manifest.check_complete_domain()
    manifest.check_validation_case(workload, args.case)
    platform = platform_for(workload.target)
    if platform.code_object not in {CodeObject.HSACO, CodeObject.MCFATBIN}:
        raise ValueError('this local tensor qualification command implements HIP and MACA allocation adapters')
    protocol = EvaluationProtocol('tensor-program-correctness', 'confirmatory', workload.canonical_sha256,
                                  args.case, 'none')
    result.update(phase='cpu_preparation', target=workload.target, workload_id=workload.workload_id, case_id=args.case)
    prepared = PreparedProgramCase(workload, args.case)
    executor = resolve_executor(ROOT, CURRENT_RELEASE_BINDING, 'tensor Program evaluation', template=True, target=workload.target)
    result.update(phase='device', job_id=admit_local_job(platform.local_job_prefix), allocation_mode='local_serialized',
                  external_gpu_activity='not_excluded')
    lock = os.fstat(int(os.environ['METAL_BROKER_LOCK_FD']))
    result['lock'] = {'device': lock.st_dev, 'inode': lock.st_ino, 'uid': lock.st_uid, 'nlink': lock.st_nlink}
    host = executor.admit_host()
    if platform.code_object is CodeObject.HSACO:
        from open_cake_ir.evaluation.triton_hip import observe_local_hip
        admission = observe_local_hip(workload.target)
    else:
        from open_cake_ir.evaluation.triton_metax import observe_local_metax
        admission = observe_local_metax(workload.target, runtime_library=host['runtime_library'])
    receipt = evaluate_program_case(candidate, workload, protocol, admission, prepared=prepared)
    for role, payload in receipt.artifact_payloads.items():
        with (args.output / (role + '.json')).open('xb') as stream:
            stream.write(payload)
    write(args.output / 'receipt.json', {field.name: getattr(receipt, field.name)
          for field in fields(receipt) if field.name != 'artifact_payloads'})
    result.update(phase='device_complete', passed=receipt.correctness_passed,
                  kernel_calls=receipt.kernel_calls, correctness=receipt.correctness, timing_samples=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    builder = sub.add_parser('build')
    builder.add_argument('--workload', type=Path, required=True)
    builder.add_argument('--program', type=Path, required=True)
    runner = sub.add_parser('evaluate')
    runner.add_argument('--built', type=Path, required=True)
    runner.add_argument('--case', required=True)
    for command in (builder, runner):
        command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output == ROOT or ROOT in args.output.parents:
        parser.error('generated qualification artifacts must stay outside the checkout')
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'source_commit': checkout_commit(ROOT), 'phase': 'admission', 'passed': False,
              'scope': 'sealed Program construction or original-case correctness only'}
    try:
        if result['source_commit'] is None:
            raise ValueError('qualification requires a clean committed source')
        if os.environ.get('METAL_BROKER_LOCK_FD') or os.environ.get('GPUQ_JOB_ID'):
            raise ValueError('qualification must start outside an existing GPU allocation')
        (build if args.command == 'build' else evaluate)(args, result)
    except Exception as error:
        result.update(error=str(error), failure_class=type(error).__name__)
        (args.output / 'failure.txt').write_text(traceback.format_exc())
        for role, payload in getattr(error, 'artifact_payloads', {}).items():
            if not isinstance(role, str) or not role.isidentifier() or not isinstance(payload, bytes):
                raise ValueError('fault artifact role or bytes differ') from error
            with (args.output / ('fault-' + role + '.bin')).open('xb') as stream:
                stream.write(payload)
    write(args.output / 'result.json', result)
    print(json.dumps(_json_projection(result), indent=2), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
