"""Freeze ordinary task inputs directly as an independent Run."""
from __future__ import annotations

import json
from pathlib import Path

from open_cake_ir.evaluation.paired import paired_protocol
from open_cake_ir.compiler import Compiler
from open_cake_ir.lab import RunSpecification
from open_cake_ir.lab.bindings import (
    external_file, bind_cli_provider, bind_runtime_execution, bind_fixed_baseline,
    source_reference_path,
)
from open_cake_ir.lab.admission import validate_paired_baseline, validate_backend_assay
from open_cake_ir.lab.python_reference import read_skeleton_reference
from open_cake_ir.lab.toolchains import toolchain_for
from open_cake_ir.serialization import canonical_json_bytes
from .runtime import TaskLab
from .authoring import prepare_schedule
from .launch import parse_launch_manifest
from .workloads import load_workload


def prepare_task_run(project_root,inputs,*,compiler_reference,executor,qualification_path,
                     qualification_anchor_path,runtime_config_path,baseline_path,baseline_selection):
    """Return a fully admitted Run; no Study, evidence root or provider is created."""
    root = Path(project_root).resolve(strict=True)
    document = json.loads(canonical_json_bytes(inputs))
    if paired_protocol(document['evaluation_protocol']) is None:
        raise ValueError('task optimization requires an available fixed-baseline timed assay')
    authoring = document['authoring']
    row = toolchain_for(authoring['lowering_route']['backend'])
    runtime = external_file(root,str(runtime_config_path),'runtime configuration')
    receipt = external_file(root,str(qualification_path),'qualification')
    anchor = external_file(root,str(qualification_anchor_path),'qualification anchor')
    authoring['provider'],config = bind_cli_provider(root,authoring['provider'],row,
        runtime_path=runtime,receipt_path=receipt,anchor_path=anchor)
    authoring['toolchain_sha256'],execution = bind_runtime_execution(root,row,executor,config,runtime)
    document['compiler_revision'] = dict(compiler_reference)
    authoring['compiler_revision'] = dict(compiler_reference)
    document['execution'].update(execution)
    document['execution']['fixed_baseline'] = bind_fixed_baseline(root,baseline_path,baseline_selection)
    specification = TaskLab(root).preflight_run(RunSpecification.from_dict(document))
    _,workload_path = source_reference_path(root,document['workload']['path'],'Run Workload')
    workload = load_workload(workload_path)
    validate_backend_assay(route=authoring['lowering_route'],evaluation=document['evaluation_protocol'],workload=workload,
        attribution_evaluation=document['evaluation_protocol'].get('attribution_evaluation'))
    _,skeleton = read_skeleton_reference(root,authoring['schedule_skeleton'])
    baseline = prepare_schedule(skeleton,workload,document['evaluation_protocol']['case_id'],authoring)
    compiler = Compiler.load(root,root/compiler_reference['path'])
    assessment = compiler.assess(baseline)
    if not assessment.lowering_eligible:
        raise ValueError('task optimization baseline is not lowerable')
    validate_paired_baseline(project_root=root,workload=workload,evaluation=document['evaluation_protocol'],
        execution=document['execution'],route=authoring['lowering_route'],baseline_lowering=compiler.lower(assessment),
        manifest_parser=parse_launch_manifest)
    return specification
