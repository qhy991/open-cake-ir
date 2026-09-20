"""Freeze ordinary task inputs directly as an independent Run."""
from __future__ import annotations

import json
from pathlib import Path

from open_cake_ir.evaluation.paired import paired_protocol
from open_cake_ir.lab import RunSpecification
from open_cake_ir.lab.bindings import (
    external_file, bind_cli_provider, bind_runtime_execution, bind_fixed_baseline,
)
from open_cake_ir.lab.toolchains import toolchain_for
from open_cake_ir.serialization import canonical_json_bytes
from .runtime import TaskLab


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
    return TaskLab(root).preflight_run(RunSpecification.from_dict(document))
