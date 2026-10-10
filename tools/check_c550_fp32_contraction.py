#!/usr/bin/env python3
"""Check one sealed C550 kernel with its original standard-library task oracle."""
import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from open_cake_ir.evaluation.core import EvaluationProtocol, LoadedTorchTensorCandidate, TensorLaunchManifest
from open_cake_ir.evaluation.local_broker import admit_local_job
from open_cake_ir.evaluation.paired import candidate_from_identity
from open_cake_ir.evaluation.triton_metax import observe_local_metax
from open_cake_ir.lab.executor import ExecutorRevision
from open_cake_ir.source_identity import checkout_commit
from open_cake_ir.tasks.evaluate import _input_path
from open_cake_ir.tasks.launch import parse_launch_manifest
from open_cake_ir.tasks.tiles.evaluation import PreparedTensorCase, evaluate_tile_validation_case
from open_cake_ir.tasks.workloads import load_workload
from qualify_tensor_program import write


def check(built, case, output, physical, runtime, pci, result):
    if json.loads((built / 'result.json').read_text())['source_commit'] != checkout_commit(ROOT):
        raise ValueError('Build and check sources differ')
    identity = json.loads((built / 'candidate.json').read_text())
    candidate = candidate_from_identity(identity, {role: _input_path(built,role+'.bin',role).read_bytes()
        for role in identity['artifact_roles']})
    workload = load_workload(built / 'workload.json')
    manifest = parse_launch_manifest(json.loads(candidate.artifact_payloads['launch_manifest']))
    if workload.target != 'xcore1002' or not isinstance(manifest, TensorLaunchManifest):
        raise ValueError('Require the original single-kernel C550 ABI')
    manifest.check_complete_domain()
    manifest.check_validation_case(workload,case)
    result['phase'] = 'cpu_preparation'
    prepared = PreparedTensorCase(workload,case)
    host = ExecutorRevision.for_target(ROOT,'xcore1002').admit_host()
    result.update(phase='device',job_id=admit_local_job('maca',device=physical,
        runtime_device=runtime,expected_pci=pci,lock_scope='device',queue_seconds=300))
    admission = observe_local_metax('xcore1002',runtime_library=host['runtime_library'])
    loaded = LoadedTorchTensorCandidate(candidate,manifest,prepared.inputs,admission)
    try:
        protocol = EvaluationProtocol('c550-fp32-contraction-correctness','confirmatory',
                                      workload.canonical_sha256,case,'none')
        receipt = evaluate_tile_validation_case(candidate,workload,protocol,loaded,prepared=prepared)
        for role,payload in receipt.artifact_payloads.items():
            if not role.isidentifier():
                raise ValueError('Invalid receipt artifact role')
            with (output/role).open('xb') as stream:stream.write(payload)
        write(output/'receipt.json',{field.name:getattr(receipt,field.name)
            for field in fields(receipt) if field.name!='artifact_payloads'})
        result.update(passed=receipt.correctness_passed,kernel_calls=receipt.kernel_calls,
                      correctness=dict(receipt.correctness),resources=dict(loaded.loaded.resources),timing_samples=0)
    finally:
        loaded.close()
        result['module_closed'] = loaded.loaded.closed
    result['phase']='device_complete'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--built',type=Path,required=True)
    parser.add_argument('--case',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--physical-device',type=int,required=True)
    parser.add_argument('--runtime-device',type=int,required=True)
    parser.add_argument('--expected-pci',required=True)
    args=parser.parse_args()
    commit=checkout_commit(ROOT)
    if commit is None or any(os.environ.get(k) for k in ('METAL_BROKER_LOCK_FD','GPUQ_JOB_ID')):
        parser.error('Require clean source outside an allocation')
    output=args.output.resolve()
    if output==ROOT or ROOT in output.parents:parser.error('Evidence must stay outside source')
    output.mkdir(parents=True,exist_ok=False)
    result=dict(source_commit=commit,case_id=args.case,phase='admission',passed=False,timing_samples=0)
    try:
        check(args.built.resolve(strict=True),args.case,output,args.physical_device,args.runtime_device,args.expected_pci,result)
    except Exception as error:
        result.update(error=str(error),failure_class=type(error).__name__)
    write(output/'result.json',result)
    print(json.dumps(result),flush=True)
    return 0 if result['passed'] else 1

if __name__=='__main__':
    raise SystemExit(main())
