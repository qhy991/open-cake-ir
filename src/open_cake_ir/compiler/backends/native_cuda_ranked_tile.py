"""Backend-owned B300 ranked-tile control plane source composition.

The device protocol lives beside the native CUDA emitters. This renderer
joins it to checked Cake stage bodies; an executable ranked-tile lowering
still needs a backend-owned host ABI and Evaluation binding.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping

from .common import EmitError
from .native_cuda_tile_stage import RankedTileStageComposition


_TEMPLATE = Path(__file__).resolve().parent / 'templates/ranked_tile_b300_device.cu'
_HOST_TEMPLATE = Path(__file__).resolve().parent / 'templates/ranked_tile_b300_host.cu'
_PLACEHOLDER = re.compile(r'@[A-Z_]+@')
_IDENTIFIER = re.compile(r'[A-Za-z_][A-Za-z_0-9]*\Z')
_EFFECTS = frozenset({
    'payload.reserve', 'payload.publish', 'payload.acquire',
    'bin.reserve', 'bin.publish', 'bin.acquire',
    'source.complete', 'tile.construct', 'tile.publish', 'tile.acquire',
    'task.reserve', 'task.claim', 'task.publish', 'task.acquire',
    'task.predecessor.publish', 'task.predecessor.acquire',
    'return.index', 'return.publish', 'return.acquire',
    'steal.permit', 'steal.account',
    'input_domain.admit', 'state.reset', 'peer_pair.admit',
    'controls.admit', 'launch.rank', 'status.acquire',
})


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
            or composition.target_id != 'sm_103a'
            or composition.compute_capability != (10, 3)
            or composition.warp_size != 32
            or not composition.device_names
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
        '@CUDA_ARCH@': str(composition.compute_capability[0]*100+
                              composition.compute_capability[1]*10),
        '@TARGET_ID@': composition.target_id,
        '@THREADS@': str(composition.required_execution_groups*
                            composition.warp_size),
        '@SHARED_BYTES@': str(composition.emitted_shared_bytes),
        '@TILE_CAPACITY@': str(composition.safe_logical_tile_slots_per_rank),
        '@UPGATE_UNITS@': str(composition.stage_work_units[0][1]),
        '@ACTIVATION_UNITS@': str(composition.stage_work_units[1][1]),
        '@DOWN_UNITS@': str(composition.stage_work_units[2][1]),
    }
    source = _TEMPLATE.read_text()
    if any(source.count(mark) != 1 for mark in replacements):
        raise EmitError('ranked tile device template seams differ')
    for mark, value in replacements.items():
        source = source.replace(mark, value)
    if _PLACEHOLDER.search(source) or 'int main(' in source:
        raise EmitError('ranked tile device retains a placeholder or host main')
    return source


def emit_source_event_library(composition: RankedTileStageComposition,
                              *, combine_source: str, entry: str) -> str:
    """Render the pointer-based development ABI beside the device protocol.

    This emits no Workload inputs or experiment paths. The public Compiler
    remains a refusal until Evaluation can bind and audit this ABI, including
    failure cleanup and repeated-run state ownership.
    """
    if _IDENTIFIER.fullmatch(entry) is None:
        raise EmitError('ranked tile host ABI needs an ASCII C identifier')
    if 'cake_weave_rank512_combine_kernel' not in combine_source:
        raise EmitError('ranked tile host ABI needs the checked Cake combine')
    device = emit_source_event_device(composition, combine_source=combine_source)
    host = _HOST_TEMPLATE.read_text()
    if host.count('@ENTRY@') < 4:
        raise EmitError('ranked tile host ABI entry seams differ')
    host_facts = {
        '@ENTRY@': entry,
        '@CC_MAJOR@': str(composition.compute_capability[0]),
        '@CC_MINOR@': str(composition.compute_capability[1]),
        '@SMS@': str(composition.target_multiprocessors),
        '@DEVICE_NAME_CHECK@': ' && '.join(
            f'std::strcmp(prop.name,{json.dumps(name)}) != 0'
            for name in composition.device_names),
    }
    if any(host.count(mark)<1 for mark in host_facts):
        raise EmitError('ranked tile host ABI target seams differ')
    for mark,value in host_facts.items():
        host = host.replace(mark,value)
    source = device + host
    if (_PLACEHOLDER.search(source) or 'int main(' in source
            or 'fopen(' in source or 'fread(' in source or 'fwrite(' in source):
        raise EmitError('ranked tile library retained a template or file runner')
    return source


def source_event_map(source: str, composition: RankedTileStageComposition,
                     *, combine_source: str) -> Mapping[str, tuple[int, int]]:
    """Locate every admitted math operation and ranked protocol edge."""
    source_map: dict[str, tuple[int, int]] = {}
    effects: set[str] = set()
    for line_number,line in enumerate(source.splitlines(),start=1):
        if '// CAKE_EFFECT: ' not in line:
            continue
        name=line.split('// CAKE_EFFECT: ',1)[1].strip()
        if name in effects or name not in _EFFECTS:
            raise EmitError(f'ranked tile effect source repeats or invents {name!r}')
        effects.add(name)
        source_map['effect.'+name]=(line_number,line_number)
    if effects!=_EFFECTS:
        raise EmitError(f'ranked tile effect source omits {sorted(_EFFECTS-effects)!r}')

    def map_math(prefix: str, fragment: str, expected: tuple[str, ...]) -> None:
        if source.count(fragment)!=1:
            raise EmitError(f'ranked tile {prefix} math source is not unique')
        start=source.index(fragment)
        first_line=source[:start].count('\n')+1
        mapped=[]
        for offset,line in enumerate(fragment.splitlines()):
            if '// CAKE_OP: ' not in line:
                continue
            op=line.split('// CAKE_OP: ',1)[1].strip()
            key=prefix+'.'+op
            if key in source_map:
                raise EmitError(f'ranked tile math source repeats {key!r}')
            source_map[key]=(first_line+offset,first_line+offset)
            mapped.append(op)
        if tuple(mapped)!=expected:
            raise EmitError(f'ranked tile {prefix} operation source map differs')

    for name,stage in zip(('up_gate','activation','down'),
                          composition.stages,strict=True):
        map_math(name,stage.source,stage.mapped_operations)
    combine_ops=tuple(line.split('// CAKE_OP: ',1)[1].strip()
                      for line in combine_source.splitlines()
                      if '// CAKE_OP: ' in line)
    map_math('combine',combine_source,combine_ops)
    return MappingProxyType(source_map)
