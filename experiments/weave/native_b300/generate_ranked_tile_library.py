"""Create-only source for the exact B300 ranked-tile tensor-pointer ABI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from open_cake_ir.compiler import Compiler, Program, RankedTileEffects, Schedule


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
    combine = Schedule.from_dict(json.loads(COMBINE.read_text()))
    lowered = compiler.lower_ranked_tiles(effects, local, combine)
    lowered.validate_binding()
    req = lowered.toolchain_requirements
    (evidence_root / 'ranked_tile.cu').write_text(lowered.source)
    (evidence_root / 'lowering_report.json').write_text(json.dumps({
        'source_commit': compiler.commit,
        'target': local.target,
        'effects': str(EFFECTS.relative_to(ROOT)),
        'program': str(PROGRAM.relative_to(ROOT)),
        'combine': str(COMBINE.relative_to(ROOT)),
        'entry_point': effects.lowering.entry_point,
        'logical_tile_capacity': lowered.analysis.logical_tile_slots_per_rank,
        'stage_task_capacity': lowered.analysis.stage_task_slots_per_rank,
        'stage_work_units': [list(item) for item in
                             lowered.analysis.stage_work_units],
        'emitted_shared_bytes': req['dynamic_shared_bytes'],
        'host_abi': list(req['host_abi'].values()),
        'controls_abi': 'rank_local_int32_arrays',
        'source_map': {name:list(lines) for name,lines in
                       lowered.source_map.items()},
        'scope': 'public Compiler.lower_ranked_tiles source for exact B300; device and Evaluation qualification remain separate',
    }, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence-root', type=Path, required=True)
    generate(parser.parse_args().evidence_root.expanduser().resolve(strict=True))
