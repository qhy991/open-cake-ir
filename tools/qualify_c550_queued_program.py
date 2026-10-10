#!/usr/bin/env python3
"""Fixed attention component controls; no Bench search or production admission."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from open_cake_ir.evaluation.metax_queued_events import CompleteLaunchCapture, CaptureFailure, prepare_helper
from open_cake_ir.evaluation.core import LoadedTorchTensorCandidate, compare_tile_outputs, _plain_json
from open_cake_ir.tasks.c550_bench.workload import _recorded_statistics

WORKLOAD = 'solx-fib-gqa-paged-decode-h32-kv4-d128-ps1-bf16-triton-metax-boundary-v2'
STAGES = ('attention_metadata','attention_scores','attention_normalize','attention_values')
CASES = ('primary','zeros','permuted','scaled','segmented')


def capture_checked(loaded, timer, workload, expected):
    """Preserve Program intermediates if native drain has not been established."""
    arguments = loaded.fresh_argument_sets(16)
    unsafe = False
    try:
        samples = timer.capture_loaded_cohort(loaded, arguments, dry_run_iters=11, repeat_iters=5)
        checks = []
        for values in arguments:
            observed, after = loaded.snapshot(values)
            passed, metrics = compare_tile_outputs(workload, loaded.validation_inputs, expected, observed, after)
            checks.append({'passed':passed, 'metrics':metrics})
        return {'samples_ms':samples, 'checks':checks, 'activity':timer.last_activity,
                'passed':all(row['passed'] for row in checks)}
    except CaptureFailure as error:
        unsafe = error.unsafe_to_release
        raise
    finally:
        # The ordinary cohort helper always releases on failure; that is unsafe here.
        if not unsafe:
            loaded.release_argument_sets(arguments)


def finish(output, result, *, unsafe=False):
    """A fatal drain failure must not unwind through tensor/module teardown."""
    try:
        with (output/'result.json').open('x') as stream:
            json.dump(_recorded_statistics(_plain_json(result)),stream,indent=2,allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if unsafe:
            os._exit(74)
    return 0 if result.get('component_controls_passed') else 1


def execute(args, result, output):
    from open_cake_ir.lab.bindings import external_file
    from open_cake_ir.source_identity import checkout_commit
    from open_cake_ir.evaluation.paired import candidate_from_identity
    from open_cake_ir.tasks.launch import parse_launch_manifest
    from open_cake_ir.tasks.workloads import load_workload
    from open_cake_ir.tasks.evaluate import _input_path
    from open_cake_ir.tasks.program_evaluation import PreparedProgramCase, evaluate_program_case
    from open_cake_ir.evaluation.core import EvaluationProtocol
    from open_cake_ir.lab.executor import ExecutorRevision
    from open_cake_ir.evaluation.local_broker import admit_local_job
    from open_cake_ir.evaluation.triton_metax import observe_local_metax
    from qualify_tensor_program import profile_program, write
    from open_cake_ir.compiler import Program
    from open_cake_ir.lab import CandidateSubmission, OpenCakeEnvironment
    built = external_file(ROOT,str(args.built/'result.json'),'component build result').parent
    build = json.loads((built/'result.json').read_text())
    gate = json.loads((built/'compiler-gate.json').read_text())
    commit = checkout_commit(ROOT)
    if (not build.get('passed') or build.get('source_commit') != commit or not gate.get('passed')
            or gate.get('compiler_revision_id') != 'open-cake-ir@'+commit):
        raise ValueError('component requires same-source accepted native build')
    identity = json.loads((built/'candidate.json').read_text())
    payloads = {role:_input_path(built,role+'.bin',role).read_bytes() for role in identity['artifact_roles']}
    candidate = candidate_from_identity(identity,payloads)
    workload = load_workload(built/'workload.json')
    manifest = parse_launch_manifest(json.loads(payloads['launch_manifest']))
    program = Program.from_dict(json.loads((built/'program.json').read_text()))
    submission = CandidateSubmission.seal(OpenCakeEnvironment.media_type,program.document_bytes)
    if (workload.workload_id != WORKLOAD or workload.target != 'xcore1002'
            or not candidate.is_program or manifest.aligned_stages
            or manifest.program.document != program.document
            or candidate.candidate_sha256 != submission.sha256
            or tuple(stage.name for stage in program.stages) != STAGES
            or tuple(workload.case_ids) != CASES):
        raise ValueError('component fixture differs from the retained attention contract')
    manifest.check_complete_domain()
    for case in CASES:manifest.check_validation_case(workload,case)
    prepared = {case:PreparedProgramCase(workload,case) for case in CASES}
    prepare_helper()
    host = ExecutorRevision.for_target(ROOT,'xcore1002').admit_host()
    result['job_id'] = admit_local_job('maca',device=args.physical_device,runtime_device=args.runtime_device,
        expected_pci=args.expected_pci,lock_scope='device',queue_seconds=120)
    admission = observe_local_metax('xcore1002',runtime_library=host['runtime_library'])
    result['preflight'] = []
    for case in CASES:
        protocol = EvaluationProtocol('queued-program-component','confirmatory',workload.canonical_sha256,case,'none')
        receipt = evaluate_program_case(candidate,workload,protocol,admission,prepared=prepared[case])
        for role,payload in receipt.artifact_payloads.items():
            (output/f'preflight-{case}-{role}.json').write_bytes(payload)
        result['preflight'].append({'case':case,'passed':receipt.correctness_passed,'stage_calls':receipt.kernel_calls})
        if not receipt.correctness_passed:raise ValueError('original component preflight failed')
    loaded = LoadedTorchTensorCandidate(candidate,manifest,prepared['primary'].inputs,admission)
    timer = None
    unsafe = False
    result['cohorts'] = []
    try:
        timer = CompleteLaunchCapture(manifest,admission)
        expected = {name:value.reshape(-1).tolist() for name,value in prepared['primary'].expected.items()}
        result['cohorts'].append(capture_checked(loaded,timer,workload,expected))
        if not result['cohorts'][-1]['passed']:raise ValueError('normal component outputs failed')
        ordinary = loaded.launch
        count = 0
        delays = []
        def delayed(arguments):
            nonlocal count
            if count >= 11:
                before = time.perf_counter_ns()
                time.sleep(.005)
                delays.append((time.perf_counter_ns()-before)/1e6)
            count += 1
            ordinary(arguments)
        loaded.launch = delayed
        result['injected_host_delays_ms'] = delays
        try:result['cohorts'].append(capture_checked(loaded,timer,workload,expected))
        finally:loaded.launch = ordinary
        if not result['cohorts'][-1]['passed']:raise ValueError('delayed component outputs failed')
        if loaded.loaded.launch_calls != 128 or len(delays) != 5:
            raise ValueError('component stage or injection count differs')
        result['delay_control_passed'] = all(value < min(delays) for value in result['cohorts'][-1]['samples_ms'])
        result['scope_note'] = 'Durations compare a large injected delay; they do not decompose GPU time or prove a speedup.'
        if not result['delay_control_passed']:raise ValueError('delay exclusion control inconclusive; stop without repeating')
    except CaptureFailure as error:
        unsafe = error.unsafe_to_release
        result['unsafe_to_release'] = unsafe
        raise
    finally:
        result['active_capture'] = timer.last_activity if timer is not None else None
        result['capture_stage_calls'] = loaded.loaded.launch_calls
        if not unsafe:
            loaded.close()
            result['capture_modules_closed'] = loaded.loaded.closed
    protocol = EvaluationProtocol('queued-program-component','confirmatory',workload.canonical_sha256,'primary','none')
    post = evaluate_program_case(candidate,workload,protocol,admission,prepared=prepared['primary'])
    for role,payload in post.artifact_payloads.items():(output/f'postflight-{role}.json').write_bytes(payload)
    if not post.correctness_passed:raise ValueError('original component postflight failed')
    result['postflight'] = {'passed':True,'stage_calls':post.kernel_calls}
    profile = profile_program(candidate,workload,protocol,admission,prepared['primary'],host)
    for role,payload in profile.artifact_payloads.items():(output/f'profile-{role}.json').write_bytes(payload)
    result['separate_profile'] = {'passed':profile.correctness_passed,'summary':profile.attribution_feedback}
    result['native_stage_calls'] = 20+128+4+8
    result['component_controls_passed'] = profile.correctness_passed


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--built',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--physical-device',type=int,required=True)
    parser.add_argument('--runtime-device',type=int,required=True)
    parser.add_argument('--expected-pci',required=True)
    args=parser.parse_args(argv)
    from open_cake_ir.source_identity import checkout_commit
    from open_cake_ir.lab.custody import admit_new_campaign_path
    commit=checkout_commit(ROOT)
    if commit is None or os.environ.get('METAL_BROKER_LOCK_FD') or os.environ.get('GPUQ_JOB_ID'):
        parser.error('start with clean source outside a device lease')
    output=admit_new_campaign_path(ROOT,args.output,role='queued Program component')
    output.mkdir(parents=True,exist_ok=False)
    result={'source_commit':commit,'component_controls_passed':False,'performance_qualified':False,
        'scope':'one attention component block; no Bench, A/A, provider or optimization Run',
        'external_gpu_activity':'not_excluded','plan':{'cases':list(CASES),'stage_names':list(STAGES),
        'normal_cohorts':1,'delayed_cohorts':1,'warmups_per_cohort':11,'samples_per_cohort':5,
        'delay_seconds':.005,'repeats_on_failure':0,'separate_profile_calls':2},'promotion':'No promotion'}
    (output/'plan.json').write_text(json.dumps(result,indent=2))
    try:execute(args,result,output)
    except BaseException as error:
        result.update(error=str(error),error_type=type(error).__name__)
        if isinstance(error,CaptureFailure):
            result['unsafe_to_release']=error.unsafe_to_release
            result['failure_observation']=error.observation
        if not result.get('unsafe_to_release'):
            (output/'failure.txt').write_text(traceback.format_exc())
    return finish(output,result,unsafe=result.get('unsafe_to_release',False))


if __name__=='__main__':raise SystemExit(main())
