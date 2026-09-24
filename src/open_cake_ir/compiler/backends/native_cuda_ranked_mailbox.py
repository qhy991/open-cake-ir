"""B300 ranked-mailbox source from explicit effects and complete Cake math."""
from __future__ import annotations

import json
from pathlib import Path
import re
from types import MappingProxyType

from .native_cuda_ep_math import lower_ep_math
from ..ir import LoweringBackend, Program, RankedMailboxAnalysis, RankedMailboxEffects, Schedule
from ..program import LoweredRankedMailbox
from ..target import CodeObject


_IDENTIFIER = re.compile(r'[A-Za-z_][A-Za-z_0-9]*\Z')
_PLACEHOLDER = re.compile(r'@[A-Z_]+@')
_ATOMICS = frozenset({
    'ptx.atom.relaxed.sys.global.add.s32',
    'ptx.atom.acq_rel.sys.global.add.s32',
    'ptx.atom.release.sys.global.add.s32',
})
_HANDOFFS = frozenset({'ptx.st.release.sys.global.s32',
                       'ptx.ld.acquire.sys.global.s32'})
_TEMPLATE_ROOT = Path(__file__).resolve().parent / 'templates'


def _refuse(message):
    raise ValueError(f'native ranked mailbox lowering refused: {message}')


def _admit(compiler, effects: RankedMailboxEffects, local: Program,
           combine: Schedule, analysis: RankedMailboxAnalysis):
    target = compiler._revision.targets.get(local.target)
    if (effects.lowering.backend is not LoweringBackend.NATIVE_CUDA
            or _IDENTIFIER.fullmatch(effects.lowering.entry_point) is None):
        _refuse('explicit native CUDA route and ASCII C entry are required')
    if (target is None or target.target_id != combine.target
            or target.code_object is not CodeObject.CUBIN
            or target.compute_capability != (10, 3) or target.warp_size != 32
            or target.cooperative_grid is not True or target.occupancy is None
            or target.occupancy.multiprocessor_count != 148
            or not target.device_names
            or not _ATOMICS <= target.instruction_contracts
            or not _HANDOFFS <= target.synchronization_contracts):
        _refuse('exact B300 Target lacks cooperative, occupancy or PTX system contracts')
    if (analysis.world_size != 4 or analysis.items_per_rank not in (7, 8)
            or analysis.routes_per_item != 2 or analysis.feature_width != 16
            or analysis.remote_payload_capacity_per_rank
            != (analysis.world_size - 1) * analysis.items_per_rank
            or analysis.compute_task_capacity_per_rank
            != analysis.world_size * analysis.items_per_rank * analysis.routes_per_item
            or analysis.return_slots_per_rank
            != analysis.items_per_rank * analysis.routes_per_item):
        _refuse('development ranked queue geometry or capacities differ')
    if (any(stage.schedule.lowering.backend is not LoweringBackend.NATIVE_CUDA
            for stage in local.stages)
            or combine.lowering.backend is not LoweringBackend.NATIVE_CUDA):
        _refuse('every mathematical leaf needs its own native CUDA route')
    math = lower_ep_math(local, combine, target,
                         entry=effects.lowering.entry_point,
                         rewrite='ranked_mailbox')
    if (math.tokens != analysis.items_per_rank or math.hidden != analysis.feature_width
            or math.local_experts * analysis.world_size != 8
            or math.intermediate != 32):
        _refuse('inline math and ranked queue geometry differ')
    return target, math


def _emit_source(entry, analysis, target, math):
    template = (_TEMPLATE_ROOT / 'ranked_mailbox_b300.cu').read_text()
    chunk = (_TEMPLATE_ROOT / 'chunk_math.hpp').read_text().replace('#pragma once', '')
    replacements = {
        'CHUNK_MATH': chunk,
        'INLINE_MATH': math.source,
        'ENTRY': entry,
        'RANKS': analysis.world_size,
        'TOKENS': analysis.items_per_rank,
        'ROUTES': analysis.routes_per_item,
        'EXPERTS': math.local_experts * analysis.world_size,
        'HIDDEN': math.hidden,
        'INTERMEDIATE': math.intermediate,
        'SMS': target.occupancy.multiprocessor_count,
        'MAJOR': target.compute_capability[0],
        'MINOR': target.compute_capability[1],
        'DEVICE_NAME_CHECK': ' && '.join(
            f'std::strcmp(prop.name, {json.dumps(name)}) != 0'
            for name in target.device_names),
    }
    for name, value in replacements.items():
        token = '@' + name + '@'
        if token not in template:
            _refuse(f'protocol template omits {name}')
        template = template.replace(token, str(value))
    if _PLACEHOLDER.search(template):
        _refuse('protocol template has an unbound source placeholder')
    source_map = {}
    for number, line in enumerate(template.splitlines(), start=1):
        for prefix, namespace in (('// CAKE_OP: ', ''),
                                  ('// CAKE_EFFECT: ', 'effect.')):
            if prefix in line:
                name = namespace + line.split(prefix, 1)[1]
                if name in source_map:
                    _refuse(f'protocol source repeats mapped operation {name!r}')
                source_map[name] = (number, number)
    if not set(math.source_map) <= set(source_map):
        _refuse('protocol source omitted an admitted mathematical operation')
    return template, MappingProxyType(source_map)


def lower_ranked_mailbox(compiler, effects: RankedMailboxEffects,
                         local: Program, combine: Schedule,
                         analysis: RankedMailboxAnalysis) -> LoweredRankedMailbox:
    """Generate one four-rank source; Evaluation still owns allocation and runs."""
    if compiler.commit is None:
        _refuse('a clean Compiler commit is required')
    target, math = _admit(compiler, effects, local, combine, analysis)
    entry = effects.lowering.entry_point
    source, source_map = _emit_source(entry, analysis, target, math)
    sms = target.occupancy.multiprocessor_count
    names = {name: entry + '_' + name for name in (
        'mailbox_bytes', 'tokens', 'output_offset', 'output_bytes',
        'status_offset', 'payload_tail_offset', 'task_tail_offset',
        'stolen_offset', 'compute_completed_offset', 'dispatch_done_offset',
        'prepare_rank', 'launch')}
    requirements = {
        'source_language': 'cuda_cpp', 'target': local.target,
        'entry_point': entry, 'kernel_entry_point': entry + '_kernel',
        'compiler_commit': compiler.commit,
        'world_size': analysis.world_size,
        'tokens_per_rank': analysis.items_per_rank,
        'routes_per_token': analysis.routes_per_item,
        'feature_width': analysis.feature_width,
        'payload_capacity': analysis.remote_payload_capacity_per_rank,
        'task_capacity': analysis.compute_task_capacity_per_rank,
        'return_slots': analysis.return_slots_per_rank,
        'grid_per_rank': [[sms, 1, 1] for _ in range(analysis.world_size)],
        'block': [32, 1, 1], 'cooperative_grid': True,
        'activated_shared_bytes': math.activated_shared_bytes,
        'mailbox_bytes_query': names['mailbox_bytes'],
        'state_reset': effects.document['reset'],
        'host_abi': names,
        'peer_pair_runtime_check': True,
        'input_domain_runtime_check': True,
        'nvcc_flags': ['-std=c++17', '--gpu-architecture=compute_103a',
                       '--gpu-code=sm_103a', '-O3', '--fmad=false',
                       '-lineinfo', '-Xptxas=-v'],
        'link_libraries': ['cuda', 'cudart'],
    }
    result = LoweredRankedMailbox(local, combine, effects, analysis,
                                  compiler._revision.revision_id, source,
                                  source_map, MappingProxyType(requirements))
    result.validate_binding()
    return result
