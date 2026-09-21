#!/usr/bin/env python3
"""Build a sealed Program and qualify tensor-oracle cases with native HIP or MACA execution.

Build runs in the captured CPU-only compilation environment. Each device command
prepares exactly one original CPU case before acquiring the existing local broker;
the process retains the allocation to exit. Profile captures one separate complete
Program invocation with its own full output check; it is not performance timing.
"""
import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import sys
import traceback
from hashlib import sha256

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]

from open_cake_ir.cli import _json_projection
from open_cake_ir.compiler import Compiler, Program
from open_cake_ir.compiler.target import CodeObject
from open_cake_ir.evaluation.core import EvaluationProtocol, EvaluationReceipt
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


def profile_program(candidate, workload, protocol, admission, prepared, host):
    """Compose the existing correctness assay and native Program attribution source."""
    from open_cake_ir.evaluation.metax_program_profile import capture_program_activity, MACA_PROGRAM_PROFILE
    from open_cake_ir.evaluation.program import program_components
    from open_cake_ir.lab.faults import RunProtocolFault
    manifest, children, _ = program_components(candidate)
    if protocol.case_id != manifest.case_id:
        raise ValueError('Program attribution uses the sealed primary case')
    retained = {}
    try:
        preflight = evaluate_program_case(candidate, workload, protocol, admission, prepared=prepared)
        retained.update({'preflight_' + role: payload for role, payload in preflight.artifact_payloads.items()})
        if not preflight.correctness_passed:
            raise ValueError('Program profile requires a passing original oracle preflight')
        instrumented = evaluate_program_case(candidate, workload, protocol, admission, prepared=prepared,
            observe=lambda launch: capture_program_activity(launch, candidate=candidate, admission=admission,
                                                             activity_library=host['activity_library']))
        retained.update({'instrumented_' + role: payload for role, payload in instrumented.artifact_payloads.items()})
        if not instrumented.correctness_passed:
            raise ValueError('instrumented Program output failed the original oracle')
        first = json.loads(preflight.artifact_payloads['correctness_output'])
        second = json.loads(instrumented.artifact_payloads['correctness_output'])
        raw = second.pop('native_activity')
        metrics = {**dict(instrumented.correctness),
                   'max_abs_error': max(preflight.correctness['max_abs_error'], instrumented.correctness['max_abs_error'])}
        correctness = {'passed': True, 'metrics': metrics, 'correctness_launches': 2,
            'preflight': dict(preflight.correctness), 'instrumented': {'passed': True, 'metrics': dict(instrumented.correctness)},
            'observations': {'preflight': first, 'instrumented': second}}
        launch = {**json.loads(instrumented.artifact_payloads['launch_receipt']),
            'correctness_launches': 2, 'job_id': admission.broker_job_id, 'gpu_uuid': admission.gpu_uuid,
            'stage_candidates': {name: candidate_identity(child) for name, child in children.items()}}
        policy = {'case_id': protocol.case_id, 'attribution_evaluation': 'correctness_then_profile'}
        profile = {'kind': MACA_PROGRAM_PROFILE.kind, 'candidate_sha256': candidate.candidate_sha256,
            'case_id': protocol.case_id, 'kernel_name': manifest.kernel_name, 'job_id': admission.broker_job_id,
            'gpu_uuid': admission.gpu_uuid, 'allocation_mode': 'local_serialized', 'external_gpu_activity': 'not_excluded',
            'separate_instrumented_launch': True, 'evaluation_protocol': policy,
            'raw': raw, 'summary': MACA_PROGRAM_PROFILE.summary(raw)}
        payloads = {role: canonical_json_bytes(document) for role, document in (
            ('correctness_output', correctness), ('launch_receipt', launch), ('profile', profile))}
        retained.update(payloads)
        return EvaluationReceipt(candidate.candidate_sha256, workload.canonical_sha256,
            sha256(canonical_json_bytes(policy)).hexdigest(), 'attribution', protocol.case_id, True, metrics,
            manifest.kernels_per_call, 0, sha256(payloads['launch_receipt']).hexdigest(), None, artifact_payloads=payloads)
    except Exception as error:
        retained.update(getattr(error, 'artifact_payloads', {}))
        raise RunProtocolFault('harness_fault', str(error), artifact_payloads=retained) from error


def _admit_captured_host(executor):
    if executor.document['host_environment'].get('kind') == 'hip':
        return executor.admit_hip_host()
    return executor.admit_host()


def measure_program(args, candidate, workload, protocol, admission, prepared, expected_values, host, result):
    """Bounded timer discovery, with the ordinary fresh-output check on every call.

    No paired policy or timing EvaluationReceipt is minted by this command.
    """
    from open_cake_ir.compiler.target import declared_target
    from open_cake_ir.evaluation.core import LoadedTorchTensorCandidate
    from open_cake_ir.evaluation.metax_program_benchmark import McptiProgramBenchmark
    from open_cake_ir.evaluation.metax_benchmark import kernel_records, validate_loaded_resources
    from open_cake_ir.evaluation.timing import summarize_cohort
    from open_cake_ir.evaluation.loaders import LifecycleError
    from open_cake_ir.tasks.evaluate import _fresh_tile_cohort
    from open_cake_ir.lab.faults import RunProtocolFault
    manifest = parse_launch_manifest(json.loads(candidate.artifact_payloads['launch_manifest']))
    preflight = evaluate_program_case(candidate, workload, protocol, admission, prepared=prepared)
    for role, payload in preflight.artifact_payloads.items():
        (args.output / ('preflight-' + role + '.json')).write_bytes(payload)
    if not preflight.correctness_passed:
        raise ValueError('Program timer discovery requires a passing original oracle')
    diagnostic = {'scope': 'native Program timer discovery; not paired performance or Run qualification',
                  'candidate': candidate_identity(candidate), 'case_id': protocol.case_id, 'cohorts': [],
                  'maximum_cv': 0.05, 'passed': False}
    loaded = None
    benchmark = None
    primary = None
    try:
        loaded = LoadedTorchTensorCandidate(candidate, manifest, prepared.inputs, admission)
        benchmark = McptiProgramBenchmark(candidate, activity_library=host['activity_library'],
            l2_cache_bytes=declared_target(candidate.target).l2_cache_bytes)
        for index in range(5):
            samples, checks = _fresh_tile_cohort(loaded, benchmark, workload, prepared.inputs, expected_values,
                samples_per_cohort=25, route_calls_per_cohort=36)
            row = {'cohort': index, 'samples_ms': samples, 'summary': summarize_cohort(samples),
                   'output_check': checks, 'native_activity': benchmark.last_activity}
            write(args.output / ('cohort-' + str(index) + '.json'), row)
            diagnostic['cohorts'].append({key: value for key, value in row.items() if key != 'native_activity'})
            records = kernel_records(benchmark.last_activity['activity'])
            width = manifest.kernels_per_call + 1
            for start in range(0, len(records), width):
                for stage, kernel in zip(manifest.program.stages, records[start + 1:start + width], strict=True):
                    validate_loaded_resources(loaded.loaded.resources['stages'][stage.name], kernel)
            if not checks['passed']:
                raise ValueError('Program timer fresh output failed the original oracle')
        if loaded.loaded.launch_calls != 5 * 36 * manifest.kernels_per_call:
            raise ValueError('Program timer native call count differs from the fresh invocation budget')
        diagnostic['resources'] = loaded.loaded.resources
    except BaseException as error:
        primary = error
        diagnostic.update(error=str(error), failure_class=type(error).__name__,
                          rejected_activity=None if benchmark is None else benchmark.last_activity,
                          partial_activity=getattr(error, 'activity_snapshot', None))
    finally:
        if loaded is not None:
            diagnostic['kernel_calls'] = loaded.loaded.launch_calls
            try:
                loaded.close()
                if not loaded.loaded.closed:
                    raise ValueError('Program timer modules remain open')
            except BaseException as cleanup:
                primary = LifecycleError(primary, cleanup) if primary is not None else cleanup
            diagnostic['module_unloaded'] = loaded.loaded.closed
    if primary is None:
        try:
            postflight = evaluate_program_case(candidate, workload, protocol, admission, prepared=prepared)
            for role, payload in postflight.artifact_payloads.items():
                (args.output / ('postflight-' + role + '.json')).write_bytes(payload)
            diagnostic.update(correctness_passed=postflight.correctness_passed,
                measurement_quality_passed=all(row['summary']['cv'] <= 0.05 for row in diagnostic['cohorts']))
            diagnostic['passed'] = diagnostic['correctness_passed'] and diagnostic['measurement_quality_passed']
        except Exception as error:
            primary = error
            diagnostic.update(error=str(error), failure_class=type(error).__name__, failure_phase='postflight')
    write(args.output / 'measurement-diagnostic.json', diagnostic)
    if primary is not None:
        raise RunProtocolFault('harness_fault', str(primary),
            artifact_payloads={**getattr(primary, 'artifact_payloads', {}),
                'measurement_diagnostic': canonical_json_bytes(_json_projection(diagnostic))}) from primary
    result.update(phase='device_complete', scope=diagnostic['scope'], passed=diagnostic['passed'],
                  correctness_passed=diagnostic['correctness_passed'],
                  measurement_quality_passed=diagnostic['measurement_quality_passed'],
                  kernel_calls=diagnostic['kernel_calls'] + 2 * manifest.kernels_per_call,
                  timing_samples=125)


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
    _admit_captured_host(executor)
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
    if args.command in {'profile', 'measure'} and platform.code_object is not CodeObject.MCFATBIN:
        raise ValueError('this Program profile source implements only MACA attribution')
    if args.command == 'measure' and (not candidate.is_program or args.case != manifest.case_id):
        raise ValueError('Program timer discovery requires a sealed primary Program')
    protocol = EvaluationProtocol('tensor-program-correctness', 'confirmatory', workload.canonical_sha256,
                                  args.case, 'none')
    result.update(phase='cpu_preparation', target=workload.target, workload_id=workload.workload_id, case_id=args.case)
    prepared = PreparedProgramCase(workload, args.case)
    expected_values = ({name: value.reshape(-1).tolist() for name, value in prepared.expected.items()}
                       if args.command == 'measure' else None)
    executor = resolve_executor(ROOT, CURRENT_RELEASE_BINDING, 'tensor Program evaluation', template=True, target=workload.target)
    result.update(phase='device', job_id=admit_local_job(platform.local_job_prefix), allocation_mode='local_serialized',
                  external_gpu_activity='not_excluded')
    lock = os.fstat(int(os.environ['METAL_BROKER_LOCK_FD']))
    result['lock'] = {'device': lock.st_dev, 'inode': lock.st_ino, 'uid': lock.st_uid, 'nlink': lock.st_nlink}
    host = _admit_captured_host(executor)
    if platform.code_object is CodeObject.HSACO:
        from open_cake_ir.evaluation.triton_hip import observe_local_hip
        admission = observe_local_hip(workload.target)
    elif platform.code_object is CodeObject.MCFATBIN:
        from open_cake_ir.evaluation.triton_metax import observe_local_metax
        admission = observe_local_metax(workload.target, runtime_library=host['runtime_library'])
    if args.command == 'measure':
        return measure_program(args, candidate, workload, protocol, admission, prepared, expected_values, host, result)
    receipt = (profile_program(candidate, workload, protocol, admission, prepared, host)
               if args.command == 'profile' else
               evaluate_program_case(candidate, workload, protocol, admission, prepared=prepared))
    for role, payload in receipt.artifact_payloads.items():
        with (args.output / (role + '.json')).open('xb') as stream:
            stream.write(payload)
    write(args.output / 'receipt.json', {field.name: getattr(receipt, field.name)
          for field in fields(receipt) if field.name != 'artifact_payloads'})
    result.update(phase='device_complete', passed=receipt.correctness_passed,
                  kernel_calls=receipt.kernel_calls, correctness=receipt.correctness, timing_samples=0)
    if args.command == 'profile':
        result.update(scope='separate Program attribution with preflight and instrumented correctness; not performance timing',
                      native_kernel_calls=2 * receipt.kernel_calls, attribution=receipt.attribution_feedback)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    builder = sub.add_parser('build')
    builder.add_argument('--workload', type=Path, required=True)
    builder.add_argument('--program', type=Path, required=True)
    runner = sub.add_parser('evaluate')
    profiler = sub.add_parser('profile')
    measurement = sub.add_parser('measure', help='bounded native timer discovery, not a paired performance qualification')
    for command in (runner, profiler, measurement):
        command.add_argument('--built', type=Path, required=True)
        command.add_argument('--case', required=True)
    for command in (builder, runner, profiler, measurement):
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
