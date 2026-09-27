"""Backend-owned B300 ranked-tile control plane source composition.

The device protocol and pointer ABI live beside the native CUDA emitters.
This renderer joins them to checked Cake stage bodies, with runtime chunk
choices bounded by the ranked effect's four-wave capacity.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping

from . import native_cuda
from .common import EmitError
from .native_cuda_tile_stage import (RankedTileStageComposition,
                                     compose_model_ranked_tile_stages)
from ..ir import Program, RankedTileAnalysis, RankedTileEffects, Schedule


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
_REQUIRED_INSTRUCTIONS = frozenset({
    'tcgen05.mma.cta_group::1.kind::f16',
    'ptx.atom.relaxed.gpu.global.add.s32',
    'ptx.atom.acq_rel.gpu.global.add.s32',
    'ptx.atom.relaxed.sys.global.add.s32',
})
_REQUIRED_HANDOFFS = frozenset({
    'ptx.st.release.sys.global.s32',
    'ptx.ld.acquire.sys.global.s32',
    'ptx.st.release.gpu.global.s32',
    'ptx.ld.acquire.gpu.global.s32',
})


@dataclass(frozen=True)
class NativeRankedTileLowering:
    """Exact B300 source binding complete Cake math to ranked-tile effects."""

    local_program: Program
    combine_schedule: Schedule
    effects: RankedTileEffects
    analysis: RankedTileAnalysis
    compiler_revision_id: str
    source: str
    source_map: Mapping[str, tuple[int, int]]
    toolchain_requirements: Mapping[str, object]

    def validate_binding(self) -> None:
        if (Program.from_dict(self.local_program.document)!=self.local_program
                or RankedTileEffects.from_dict(self.effects.document)!=self.effects
                or self.effects.analyze(self.local_program,self.combine_schedule)
                !=self.analysis):
            raise ValueError('ranked-tile lowering typed binding differs')
        req=self.toolchain_requirements
        if (req.get('target')!=self.local_program.target
                or req.get('entry_point')!=self.effects.lowering.entry_point
                or req.get('world_size')!=self.analysis.world_size
                or req.get('tokens_per_rank')!=self.analysis.items_per_rank
                or req.get('routes_per_token')!=self.analysis.routes_per_item
                or req.get('logical_tile_capacity')
                !=self.analysis.logical_tile_slots_per_rank
                or req.get('stage_task_capacity')
                !=self.analysis.stage_task_slots_per_rank
                or req.get('return_slots')!=self.analysis.return_slots_per_rank
                or req.get('source_events')!=20
                or req.get('supported_chunks')!=[1,2,4]
                or req.get('source_chunk_tokens_by_chunks')!={
                    '1':self.analysis.items_per_rank,
                    '2':self.analysis.items_per_rank//2,
                    '4':self.analysis.items_per_rank//4}
                or req.get('input_domain_runtime_check') is not True
                or req.get('peer_pair_runtime_check') is not True
                or req.get('failed_partial_launch_requires_process_exit')
                is not True):
            raise ValueError('ranked-tile lowering target, capacity or runtime gate differs')
        expected={f'{stage.name}.{op.op_id}'
                  for stage in self.local_program.stages
                  for op in stage.schedule.operations}
        expected.update(f'combine.{op.op_id}'
                        for op in self.combine_schedule.operations)
        expected.update('effect.'+name for name in _EFFECTS)
        if set(self.source_map)!=expected:
            raise ValueError('ranked-tile lowering math or effect map differs')
        lines=self.source.splitlines()
        for name,span in self.source_map.items():
            if (len(span)!=2 or span[0]!=span[1]
                    or not 1<=span[0]<=len(lines)):
                raise ValueError('ranked-tile source-map span differs')
            marker=('// CAKE_EFFECT: '+name.removeprefix('effect.')
                    if name.startswith('effect.')
                    else '// CAKE_OP: '+name.split('.',1)[1])
            if marker not in lines[span[0]-1]:
                raise ValueError(f'ranked-tile source-map marker {name!r} differs')


def emit_source_event_device(composition: RankedTileStageComposition,
                             *, combine_source: str) -> str:
    """Render the evidenced four-rank device protocol without a host runner.

    `combine_source` is the Cake combine emitted by the same clean Compiler.
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

    This emits no Workload inputs or experiment paths. Evaluation binds the
    pointer ABI and owns isolated-process failure cleanup and repeated state.
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


def lower_ranked_tiles(compiler, effects: RankedTileEffects,
                       local_program: Program, combine_schedule: Schedule,
                       analysis: RankedTileAnalysis) -> NativeRankedTileLowering:
    """Emit the exact B300 source and typed ABI without Workload data."""
    if compiler.commit is None:
        raise EmitError('native ranked tiles need a clean Compiler commit')
    target=compiler._revision.targets.get(local_program.target)
    if (target is None or target.target_id!=combine_schedule.target
            or not _REQUIRED_INSTRUCTIONS<=target.instruction_contracts
            or not _REQUIRED_HANDOFFS<=target.synchronization_contracts):
        raise EmitError('native ranked tiles need the exact Target PTX contracts')
    if effects.analyze(local_program,combine_schedule)!=analysis:
        raise EmitError('native ranked tile analysis differs from complete math')
    composition=compose_model_ranked_tile_stages(
        effects,local_program,combine_schedule,target)
    combine=native_cuda.emit(combine_schedule,target).source
    if not combine.startswith('// Generated by Open-Cake native CUDA;'):
        raise EmitError('native ranked tile combine source header differs')
    # Standalone Schedule lowering adds its own schedule-byte header. The
    # ranked result binds the typed combine and one Compiler Revision instead.
    combine=combine.split('\n',1)[1]
    entry=effects.lowering.entry_point
    source=emit_source_event_library(composition,combine_source=combine,
                                     entry=entry)
    mapped=source_event_map(source,composition,combine_source=combine)
    abi={name:entry+'_'+name for name in
         ('abi_version','ranks','source_events','bin_bytes','output_bytes',
          'create','launch','destroy','stolen','payloads','tile_counts')}
    requirements={
        'source_language':'cuda_cpp','target':local_program.target,
        'entry_point':entry,'compiler_commit':compiler.commit,
        'world_size':analysis.world_size,
        'tokens_per_rank':analysis.items_per_rank,
        'routes_per_token':analysis.routes_per_item,
        'experts':analysis.experts,
        'experts_per_rank':analysis.experts_per_rank,
        'feature_width':analysis.feature_width,
        'logical_tile_capacity':analysis.logical_tile_slots_per_rank,
        'stage_task_capacity':analysis.stage_task_slots_per_rank,
        'return_slots':analysis.return_slots_per_rank,
        'source_events':20,'supported_chunks':[1,2,4],
        'source_chunk_tokens_by_chunks':{
            str(chunks):analysis.items_per_rank//chunks for chunks in (1,2,4)},
        'rank_inputs':[
            {'name':'hidden','shape':[512,2048],'dtype':'bf16'},
            {'name':'expert_ids','shape':[512,8],'dtype':'int32'},
            {'name':'route_weights','shape':[512,8],'dtype':'fp32'},
            {'name':'w_up_gate','shape':[32,1536,2048],'dtype':'bf16'},
            {'name':'w_down','shape':[32,2048,768],'dtype':'bf16'},
        ],
        'rank_output':{'shape':[512,2048],'dtype':'bf16'},
        'host_abi':abi,
        'state_reset':effects.document['reset'],
        'peer_pair_runtime_check':True,
        'input_domain_runtime_check':True,
        'rank_local_controls':True,
        'worker_grid_ctas':96,'threads_per_cta':
            composition.required_execution_groups*composition.warp_size,
        'dynamic_shared_bytes':composition.emitted_shared_bytes,
        'synchronous_launch':True,
        'failed_partial_launch_requires_process_exit':True,
        'nvcc_flags':['-std=c++17',
                      '--gpu-architecture='+local_program.target.replace(
                          'sm_','compute_',1),
                      '--gpu-code='+local_program.target,'-O3','--fmad=false',
                      '-lineinfo','-Xptxas=-v'],
        'link_libraries':['cuda','cudart'],
    }
    result=NativeRankedTileLowering(
        local_program,combine_schedule,effects,analysis,
        compiler._revision.revision_id,source,mapped,
        MappingProxyType(requirements))
    result.validate_binding()
    return result
