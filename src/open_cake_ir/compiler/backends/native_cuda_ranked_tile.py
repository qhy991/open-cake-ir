"""Backend-owned B300 ranked-tile control plane source composition.

The device protocol lives beside the native CUDA emitters. This renderer
joins it to checked Cake stage bodies; an executable ranked-tile lowering
still needs a backend-owned host ABI and Evaluation binding.
"""
from __future__ import annotations

from pathlib import Path
import re

from .common import EmitError
from .native_cuda_tile_stage import RankedTileStageComposition


_TEMPLATE = Path(__file__).resolve().parent / 'templates/ranked_tile_b300_device.cu'
_PLACEHOLDER = re.compile(r'@[A-Z_]+@')


def emit_source_event_device(composition: RankedTileStageComposition,
                             *, combine_source: str) -> str:
    """Render the evidenced four-rank device protocol without a host runner.

    `combine_source` is an explicit backend dependency. The experiment uses
    its separately retained Cake combine translation unit; a complete public
    lowering must inline the combine emitted from the same clean Compiler.
    """
    if (not isinstance(composition, RankedTileStageComposition)
            or len(composition.stages) != 3
            or composition.safe_logical_tile_slots_per_rank != 255
            or composition.safe_stage_task_slots_per_rank != 46920
            or composition.emitted_shared_bytes != 49200
            or not isinstance(combine_source, str)
            or not combine_source.strip()):
        raise EmitError('ranked tile device needs exact checked B300 composition')
    up_gate, activation, down = composition.stages
    if (up_gate.function_name != 'cake_upgate_stage_work'
            or activation.function_name != 'cake_activation_stage_work'
            or down.function_name != 'cake_down_stage_work'
            or up_gate.instruction_helpers != down.instruction_helpers):
        raise EmitError('ranked tile device stage functions or PTX helpers differ')
    replacements = {
        '@COMBINE_SOURCE@': combine_source,
        '@CAKE_HELPERS@': up_gate.instruction_helpers,
        '@UPGATE_STAGE@': up_gate.source,
        '@ACTIVATION_STAGE@': activation.source,
        '@DOWN_STAGE@': down.source,
    }
    source = _TEMPLATE.read_text()
    if any(source.count(mark) != 1 for mark in replacements):
        raise EmitError('ranked tile device template seams differ')
    for mark, value in replacements.items():
        source = source.replace(mark, value)
    if _PLACEHOLDER.search(source) or 'int main(' in source:
        raise EmitError('ranked tile device retains a placeholder or host main')
    return source
