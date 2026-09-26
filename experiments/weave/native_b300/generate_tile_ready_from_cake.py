"""Compose the proven B300 tile-ready control plane with Cake FFN stages.

This is an offline generation step. The input Program and Target come from one
clean Compiler commit; all three stage bodies are emitted from their
Schedules. The worker template owns dispatch, completion and stealing only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from open_cake_ir.compiler import Compiler, Program
from open_cake_ir.compiler.backends.native_cuda_tile_stage import (
    emit_model_activation_stage, emit_tensor_tile_stage,
)


ROOT = Path(__file__).resolve().parents[3]
PROGRAM = ROOT / 'examples/programs/weave-model-local-expert-ffn-native-b300.json'
TEMPLATES = {
    'cake_full_ffn_stages': Path(__file__).with_name(
        'model_tile_ready_from_cake_template.cu'),
    'cake_two_expert_stages': Path(__file__).with_name(
        'model_tile_ready_two_expert_template.cu'),
    'cake_ep4_owner_stages': Path(__file__).with_name(
        'model_ep4_worker_owner_template.cu'),
    'cake_ep4_live_chain_stages': Path(__file__).with_name(
        'model_ep4_live_chain_template.cu'),
    'cake_ep4_gpu_plan_stages': Path(__file__).with_name(
        'model_ep4_gpu_plan_template.cu'),
}
OUTPUT = 'model_tile_ready_ffn_capped.cu'
MARKS = ('@CAKE_HELPERS@', '@UPGATE_STAGE@',
         '@ACTIVATION_STAGE@', '@DOWN_STAGE@')


def generate(evidence_root: Path) -> None:
    compiler = Compiler.load(ROOT)
    if compiler.commit is None:
        raise ValueError('tile-ready generation needs a clean fixed Compiler commit')
    manifest = json.loads((evidence_root / 'manifest.json').read_text())
    generation=manifest.get('generation')
    if (manifest.get('target') != 'sm_103a'
            or manifest.get('source_commit') != compiler.commit
            or not isinstance(generation, str) or generation not in TEMPLATES
            or any((evidence_root / name).exists()
                   for name in (OUTPUT, 'stage_lowering.json'))):
        raise ValueError('tile-ready generation root or source identity differs')
    program_document = json.loads(PROGRAM.read_text())
    program = Program.from_dict(program_document)
    target = compiler._revision.targets['sm_103a']
    if [stage.name for stage in program.stages] != ['up_gate', 'activation', 'down']:
        raise ValueError('complete model FFN Program stage order differs')
    results = []
    for index, name in ((0, 'cake_upgate_stage_work'),
                        (1, 'cake_activation_stage_work'),
                        (2, 'cake_down_stage_work')):
        stage = program.stages[index]
        assessment = compiler.assess(program_document['stages'][index]['schedule'])
        if not assessment.lowering_eligible:
            raise ValueError(f'{stage.name} Schedule refused before worker composition')
        lower = (emit_model_activation_stage if index == 1
                 else emit_tensor_tile_stage)
        results.append(lower(stage.schedule, target, function_name=name))
    if results[0].instruction_helpers != results[2].instruction_helpers:
        raise ValueError('tensor stages disagree on native CUDA PTX helper contract')
    template = TEMPLATES[generation].read_text()
    if any(template.count(mark) != 1 for mark in MARKS):
        raise ValueError('worker template stage seams differ')
    source = (template.replace('@CAKE_HELPERS@', results[0].instruction_helpers)
                      .replace('@UPGATE_STAGE@', results[0].source)
                      .replace('@ACTIVATION_STAGE@', results[1].source)
                      .replace('@DOWN_STAGE@', results[2].source))
    if re.search(r'@[A-Z_]+@', source):
        raise ValueError('worker source retains an unbound lowering seam')
    (evidence_root / OUTPUT).write_text(source)
    (evidence_root / 'stage_lowering.json').write_text(json.dumps({
        'source_commit': compiler.commit,
        'program': str(PROGRAM.relative_to(ROOT)),
        'target': target.target_id,
        'generation': generation,
        'stages': [
            {'name': program.stages[index].name,
             'function': emitted.function_name,
             'dynamic_shared_bytes': emitted.dynamic_shared_bytes,
             'tmem_columns': emitted.tmem_columns,
             'mapped_operations': list(emitted.mapped_operations)}
            for index, emitted in enumerate(results)
        ],
        'scope': 'all three stage math bodies from Cake; tile-ready worker control retained',
    }, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence-root', required=True, type=Path)
    generate(parser.parse_args().evidence_root.expanduser().resolve(strict=True))
