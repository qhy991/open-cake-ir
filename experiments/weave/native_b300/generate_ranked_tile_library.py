"""Create-only source for the exact B300 ranked-tile tensor-pointer ABI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from open_cake_ir.compiler import Compiler, Program, RankedTileEffects, Schedule
from open_cake_ir.compiler.backends.native_cuda_ranked_tile import (
    emit_source_event_library,
)
from open_cake_ir.compiler.backends.native_cuda_tile_stage import (
    compose_model_ranked_tile_stages,
)


ROOT = Path(__file__).resolve().parents[3]
PROGRAM = ROOT / 'examples/programs/weave-model-local-expert-ffn-native-b300.json'
EFFECTS = ROOT / 'examples/programs/weave-model-ranked-tile-effects-b300.json'
COMBINE = ROOT / 'examples/schedules/native/weave-model-weighted-combine-rank512-b300.json'


def generate(evidence_root: Path) -> None:
    compiler = Compiler.load(ROOT)
    if compiler.commit is None:
        raise ValueError('ranked-tile library generation needs a clean commit')
    manifest = json.loads((evidence_root / 'manifest.json').read_text())
    if (manifest.get('source_commit') != compiler.commit
            or manifest.get('target') != 'sm_103a'
            or manifest.get('generation') != 'cake_ranked_tile_pointer_abi'
            or any((evidence_root / name).exists()
                   for name in ('ranked_tile.cu', 'lowering_report.json'))):
        raise ValueError('ranked-tile library source or create-only root differs')
    local = Program.from_dict(json.loads(PROGRAM.read_text()))
    effects = RankedTileEffects.from_dict(json.loads(EFFECTS.read_text()))
    combine_doc = json.loads(COMBINE.read_text())
    combine = Schedule.from_dict(combine_doc)
    analysis = compiler.assess_ranked_tiles(effects, local, combine)
    target = compiler._revision.targets[local.target]
    composition = compose_model_ranked_tile_stages(effects, local, combine, target)
    assessment = compiler.assess(combine_doc)
    if not assessment.lowering_eligible:
        raise ValueError('ranked-tile combine Schedule is not lowerable')
    lowered_combine = compiler.lower(assessment)
    source = emit_source_event_library(
        composition, combine_source=lowered_combine.source,
        entry=effects.lowering.entry_point)
    (evidence_root / 'ranked_tile.cu').write_text(source)
    (evidence_root / 'lowering_report.json').write_text(json.dumps({
        'source_commit': compiler.commit,
        'target': local.target,
        'effects': str(EFFECTS.relative_to(ROOT)),
        'program': str(PROGRAM.relative_to(ROOT)),
        'combine': str(COMBINE.relative_to(ROOT)),
        'entry_point': effects.lowering.entry_point,
        'logical_tile_capacity': analysis.logical_tile_slots_per_rank,
        'stage_task_capacity': analysis.stage_task_slots_per_rank,
        'stage_work_units': [list(item) for item in composition.stage_work_units],
        'emitted_shared_bytes': composition.emitted_shared_bytes,
        'host_abi': [effects.lowering.entry_point + '_' + name for name in
                     ('ranks', 'source_events', 'output_bytes',
                      'create', 'launch', 'destroy')],
        'scope': 'development tensor-pointer ABI source; no public Compiler.lower_ranked_tiles result or device qualification',
    }, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence-root', type=Path, required=True)
    generate(parser.parse_args().evidence_root.expanduser().resolve(strict=True))
