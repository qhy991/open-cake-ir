#!/usr/bin/env python3
"""Seal a complete authored Schedule or Program against an unchanged comparison control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'src')]

from open_cake_ir.compiler import Compiler, Program, frontend
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.evaluation.paired import candidate_identity, validate_pair_candidates
from open_cake_ir.lab.bindings import load_baseline_bundle
from open_cake_ir.lab.build import TritonToolchainBuilder
from open_cake_ir.lab.environments import CandidateSubmission, OpenCakeEnvironment
from open_cake_ir.lab.executor import ExecutorRevision
from open_cake_ir.lab.triton_build import IsolatedTritonCompiler
from open_cake_ir.serialization import canonical_json_bytes
from open_cake_ir.source_identity import checkout_commit
from tools.compare_rewrite_artifacts import regular, reference_spec


def prepare(source, input_root, output, runtime_path, control_role, mechanism):
    if output.exists():raise FileExistsError(output)
    if output != output.resolve():raise ValueError('comparison output must be canonical')
    if any((parent/'.git').exists() for parent in (output,*output.parents)):
        raise ValueError('comparison output must stay outside source')
    workload = WorkloadContract(json.loads(regular(input_root,'workload.json').read_text()))
    reference_spec(input_root)
    original = source.read_bytes()
    schedule = frontend.parse(original.decode(),filename=str(source)).document if source.suffix=='.py' else json.loads(original)
    is_program = 'program_id' in schedule
    if is_program:
        schedule = Program.from_dict(schedule).document
        route = schedule['stages'][0]['schedule']['lowering']
    else:
        route = schedule['lowering']
    if workload.target != 'sm_103a' or route['backend'] != 'triton':
        raise ValueError('this preparation binds the exact B300 Triton route')
    compiler = Compiler.load(ROOT,ROOT/'compiler/revision.json')
    runtime = json.loads(runtime_path.read_text())['toolchain']
    if runtime.get('pointer_alignment') is not None:
        raise ValueError('work-assignment comparison must not silently combine an alignment treatment')
    isolated = IsolatedTritonCompiler(**runtime)
    isolated.check_executor(ExecutorRevision.for_target(ROOT,workload.target),author_workspace=output)
    builder = TritonToolchainBuilder(workload=workload,case_id='primary',isolated_compiler=isolated)
    environment = OpenCakeEnvironment(compiler,builder,workload=workload,case_id='primary',
        authority_document={'input_format':'schedule_or_python_v1','lowering_route':route})
    submission = CandidateSubmission.seal(environment.media_type,canonical_json_bytes(schedule))
    result = environment.build(submission)
    if result.launchable is None:raise ValueError(f'authored implementation refused: {dict(result.feedback)}')
    candidate = result.launchable
    control_path = regular(input_root,control_role+'/candidate.json')
    control = load_baseline_bundle(ROOT,control_path)
    validate_pair_candidates(candidate,control,workload,'primary')
    output.mkdir(parents=True)
    shutil.copyfile(input_root/'workload.json',output/'workload.json')
    shutil.copytree(input_root/'reference',output/'reference')
    shutil.copytree(control_path.parent,output/'starter')
    (output/('authored-program.json' if is_program else 'authored-schedule.json')).write_bytes(submission.payload)
    (output/('authored-input'+source.suffix)).write_bytes(original)
    artifacts = output/'optimized';artifacts.mkdir()
    paths = {}
    for role,payload in candidate.artifact_payloads.items():
        paths[role] = role+'.bin';(artifacts/paths[role]).write_bytes(payload)
    (artifacts/'candidate.json').write_text(json.dumps({'candidate':candidate_identity(candidate),'artifact_paths':paths})+'\n')
    metadata = {'kind':'authored_program_comparison' if is_program else 'authored_schedule_comparison','source_commit':checkout_commit(ROOT),
        'source_input':str(input_root),'authored_input':str(source),'control_role':control_role,
        'mechanism':mechanism,'preparation_scope':'CPU compilation only; no GPU qualification is inferred',
        'original_task_starter':str(input_root/'starter/candidate.json'),
        'findings':dict(result.feedback)}
    (output/'comparison.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schedule',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--control-role',choices=['optimized','starter'],required=True)
    parser.add_argument('--mechanism',required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.schedule.resolve(strict=True),args.input.resolve(strict=True),
        args.output.absolute(),args.runtime.resolve(strict=True),args.control_role,args.mechanism)))


if __name__ == '__main__':main()
