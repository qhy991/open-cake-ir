"""Explicit Program rewrites; guards remain with the owning Schedule passes.

This module proves the complete composition boundary and rebinds local results to
its unchanged public tensors. There is no pass-to-target capability catalogue.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Mapping

from .errors import CompilerError
from .ir import Program, BufferMode, MemorySpace, ScheduleParseError

if TYPE_CHECKING:
    from .core import Compiler


@dataclass(frozen=True)
class ProgramRewriteResult:
    program: Program | None
    reason: str
    message: str
    region: tuple[str, ...]

    @property
    def applied(self) -> bool:
        return self.program is not None


def _refuse(reason, message, region=()):
    return ProgramRewriteResult(None, reason, message, tuple(region))


def _fuse(compiler, program, *, producer, epilogue, schedule_id, entry_point,
          transformation="fuse_pointwise_epilogue"):
    region = (producer, epilogue)
    names = [stage.name for stage in program.stages]
    if producer not in names or epilogue not in names:
        return _refuse('stage_selection', 'Select two existing Program stages.', region)
    index = names.index(producer)
    if names.index(epilogue) != index + 1:
        return _refuse('stage_order', 'This fusion admits adjacent producer/epilogue stages only.', region)
    p, e = program.stages[index:index + 2]
    ps, es = p.schedule, e.schedule
    if not ps.outputs or len(es.outputs) != 1 or (transformation == 'fuse_pointwise_epilogue' and len(ps.outputs) != 1):
        return _refuse('composition_boundary', 'Select the supported producer outputs and one consumer output.', region)
    middles = {name: p.bindings[name].tensor for name in ps.outputs}
    if any(middle in program.outputs for middle in middles.values()):
        return _refuse('public_intermediate', 'Fusion cannot remove a public Program output.', region)
    if any(program.consumers(middle) != (epilogue,) for middle in middles.values()):
        return _refuse('intermediate_consumers', 'The selected epilogue must be the only consumer.', region)
    # A singleton view changes the rank/access interpretation at the seam. The
    # Schedule pass proves whole-row ownership only for an identical binding.
    if any(binding.singleton_view for stage in (p, e) for binding in stage.bindings.values()):
        return _refuse('binding_view', 'This fusion does not cross singleton-axis views.', region)
    ep_inputs = [b.name for b in es.buffers
                 if b.space is MemorySpace.GLOBAL and b.mode is BufferMode.INPUT]
    if len(ep_inputs) != len(middles) or {e.bindings[name].tensor for name in ep_inputs} != set(middles.values()):
        return _refuse('composition_boundary', 'The epilogue must consume exactly the selected private intermediates.', region)
    if not isinstance(schedule_id, str) or schedule_id in names:
        return _refuse('result_identity', 'The fused stage needs a fresh Program-local name.', region)
    producer_document = program.document['stages'][index]['schedule']
    epilogue_document = program.document['stages'][index + 1]['schedule']
    if transformation == 'fuse_tiled_epilogue':
        from .tiled_epilogue import fuse_tiled_epilogue
        seams = {name: next(local for local in ep_inputs if e.bindings[local].tensor == tensor)
                 for name, tensor in middles.items()}
        result = fuse_tiled_epilogue(compiler, producer_document, epilogue_document,
                                    seams=seams, schedule_id=schedule_id, entry_point=entry_point)
    else:
        result = compiler.fuse_pointwise_epilogue(
            producer_document, epilogue_document, private_intermediate=ps.outputs[0],
            schedule_id=schedule_id, entry_point=entry_point)
    if not result.applied:
        return _refuse(result.reason, result.message, region)
    schedule = result.schedule
    document = program.document
    bindings = {b.name: p.bindings[b.name].tensor for b in ps.buffers
                if b.space is MemorySpace.GLOBAL and b.mode is BufferMode.INPUT}
    # The Schedule pass alpha-renames the epilogue output; its public identity
    # belongs to Program and survives exactly, including its position in outputs.
    bindings[schedule['outputs'][0]] = e.bindings[es.outputs[0]].tensor
    document['stages'][index:index + 2] = [{
        'name': schedule_id, 'schedule': schedule, 'bindings': bindings,
    }]
    for middle in middles.values():
        del document['tensors'][middle]
    return ProgramRewriteResult(Program.from_dict(document), 'applied', result.message, region)


def _specialize(compiler, program, *, stage, transform, **parameters):
    names = [item.name for item in program.stages]
    if stage not in names:
        return _refuse('stage_selection', 'Select an existing Program stage.', (stage,))
    document = program.document
    index = names.index(stage)
    result = transform(compiler, document['stages'][index]['schedule'], **parameters)
    if not result.applied:
        return _refuse(result.reason, result.message, (stage,))
    document['stages'][index]['schedule'] = result.schedule
    return ProgramRewriteResult(Program.from_dict(document), 'applied', result.message, (stage,))


@dataclass(frozen=True)
class Transformation:
    name: str
    parameters: tuple[str, ...]
    description: str


# The callable surface, shared by authoring grants and tools. Hardware legality is
# deliberately absent: the owning pass and Compiler assess each concrete request.
TRANSFORMATIONS = (
    Transformation('tile_pointwise_outputs', ('stage', 'output_tile', 'schedule_id', 'entry_point'),
                   'Partition whole-row pure pointwise outputs into explicit column tiles; preserve arithmetic and public tensors. No reductions, state or synchronization.'),
    Transformation('specialize_squared_difference',
                   ('stage', 'output_tile', 'k_tile', 'loop_unroll_factor', 'schedule_id', 'entry_point'),
                   'Jointly choose output/K tiles and explicit single-stage K unroll for the original pure FP32 squared-distance graph. A full-extent tile keeps that dimension unchanged; smaller tiles are powers of two. The unroll factor divides the fixed K trip count. Structural feedback is not a latency prediction.'),
    Transformation('tile_squared_difference_outputs', ('stage', 'output_tile', 'schedule_id', 'entry_point'),
                   'Partition independent FP32 squared-distance output columns while retaining the full K sum. output_tile is a power of two below N; same pure loop-free row/centroid domain as K tiling.'),
    Transformation('tile_squared_difference', ('stage', 'k_tile', 'schedule_id', 'entry_point'),
                   'Tile a pure FP32 row squared-difference sum over K; retain subtraction before square. k_tile is a power of two below K. Requires loop-free ordinary row/centroid loads and one output store.'),
    Transformation('fuse_tiled_epilogue', ('producer', 'epilogue', 'schedule_id', 'entry_point'),
                   'Fuse matching masked rank-two private BF16/FP16 tiles into a loop-free pointwise consumer; retain producer loops and rounding casts.'),
    Transformation('fuse_pointwise_epilogue', ('producer', 'epilogue', 'schedule_id', 'entry_point'),
                   'Fuse a private rounded row intermediate or pure FP32 copy into its only pointwise consumer.'),
    Transformation('specialize_triton_store_loop',
                   ('stage', 'loop_name', 'num_stages', 'schedule_id', 'entry_point'),
                   'Choose existing num_stages for one fixed independent gfx938 store-loop region. Preserve arithmetic/accesses/ABI; no automatic depth choice or performance guarantee.'),
    Transformation('specialize_triton_warps', ('stage', 'num_warps', 'schedule_id', 'entry_point'),
                   'Choose an explicit CTA width within the pass\'s qualified domain.'),
    Transformation('specialize_output_columns', ('stage', 'schedule_id', 'entry_point'),
                   'Choose explicit per-output-column programs within the pass domain.'),
)


def rewrite_program(compiler: Compiler, program: Program, transformation: str,
                    parameters: Mapping[str, object]) -> ProgramRewriteResult:
    declaration = next((item for item in TRANSFORMATIONS if item.name == transformation), None)
    if declaration is None:
        return _refuse('unknown_transform', 'The Compiler does not declare this transformation.')
    if not isinstance(parameters, Mapping) or set(parameters) != set(declaration.parameters):
        return _refuse('transform_parameters', f'Required parameters: {declaration.parameters}.')
    if any(not isinstance(parameters[key], str) or not parameters[key]
           for key in declaration.parameters if key not in {'num_warps', 'k_tile', 'output_tile', 'loop_unroll_factor', 'num_stages'}):
        return _refuse('transform_parameters', 'Stage names and result identity must be nonempty strings.')
    try:
        program = Program.from_dict(program.document)
        # A complete candidate has no unassessed side region. Keep findings localized
        # even when the requested rewrite happens to select different stages.
        for stage in program.stages:
            if transformation == 'specialize_output_columns' and stage.name == parameters['stage']:
                # The selected pass owns its input domain, including the specific
                # storage refusal it can repair. Other regions still need to lower.
                continue
            assessment = compiler.assess(json.loads(stage.schedule_bytes))
            if not assessment.lowering_eligible:
                blocking = [f for f in assessment.findings if f.blocks_lowering or f.blocks_acceptance]
                if (transformation in {'specialize_squared_difference', 'tile_pointwise_outputs'}
                    and stage.name == parameters['stage'] and assessment.accepted
                    and blocking and all(f.code == 'TRITON_ARANGE_RANGE_UNSUPPORTED' for f in blocking)):
                    # These passes can replace non-power-of-two vector extents
                    # with masked tiles. Its guard and final assessment still own
                    # the selected candidate; unrelated input refusals stay intact.
                    continue
                return _refuse('input_refused', ', '.join(f.code for f in assessment.findings), (stage.name,))
        if transformation in {'fuse_pointwise_epilogue', 'fuse_tiled_epilogue'}:
            return _fuse(compiler, program, transformation=transformation, **parameters)
        from .pointwise_tiling import tile_pointwise_outputs
        from .passes import specialize_triton_warps, specialize_output_columns, specialize_triton_store_loop
        from .reduction_tiling import (tile_squared_difference, tile_squared_difference_outputs,
                                       specialize_squared_difference)
        transform = {
            'tile_pointwise_outputs': tile_pointwise_outputs,
            'specialize_squared_difference': specialize_squared_difference,
            'tile_squared_difference_outputs': tile_squared_difference_outputs,
            'tile_squared_difference': tile_squared_difference,
            'specialize_triton_warps': specialize_triton_warps,
            'specialize_triton_store_loop': specialize_triton_store_loop,
            'specialize_output_columns': specialize_output_columns,
        }[transformation]
        return _specialize(compiler, program, transform=transform, **parameters)
    except (CompilerError, ScheduleParseError, ValueError, TypeError) as error:
        return _refuse('result_refused', str(error))
