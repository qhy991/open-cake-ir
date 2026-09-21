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


def _fuse(compiler, program, *, producer, epilogue, schedule_id, entry_point):
    region = (producer, epilogue)
    names = [stage.name for stage in program.stages]
    if producer not in names or epilogue not in names:
        return _refuse('stage_selection', 'Select two existing Program stages.', region)
    index = names.index(producer)
    if names.index(epilogue) != index + 1:
        return _refuse('stage_order', 'This fusion admits adjacent producer/epilogue stages only.', region)
    p, e = program.stages[index:index + 2]
    ps, es = p.schedule, e.schedule
    if len(ps.outputs) != 1 or len(es.outputs) != 1:
        return _refuse('composition_boundary', 'Both stages must have exactly one output.', region)
    middle = p.bindings[ps.outputs[0]].tensor
    if middle in program.outputs:
        return _refuse('public_intermediate', 'Fusion cannot remove a public Program output.', region)
    if program.consumers(middle) != (epilogue,):
        return _refuse('intermediate_consumers', 'The selected epilogue must be the only consumer.', region)
    # A singleton view changes the rank/access interpretation at the seam. The
    # Schedule pass proves whole-row ownership only for an identical binding.
    if any(binding.singleton_view for stage in (p, e) for binding in stage.bindings.values()):
        return _refuse('binding_view', 'This fusion does not cross singleton-axis views.', region)
    ep_inputs = [b.name for b in es.buffers
                 if b.space is MemorySpace.GLOBAL and b.mode is BufferMode.INPUT]
    if len(ep_inputs) != 1 or e.bindings[ep_inputs[0]].tensor != middle:
        return _refuse('composition_boundary', 'The epilogue must consume only the selected intermediate.', region)
    if not isinstance(schedule_id, str) or schedule_id in names:
        return _refuse('result_identity', 'The fused stage needs a fresh Program-local name.', region)
    result = compiler.fuse_pointwise_epilogue(
        program.document['stages'][index]['schedule'],
        program.document['stages'][index + 1]['schedule'],
        private_intermediate=ps.outputs[0], schedule_id=schedule_id, entry_point=entry_point,
    )
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
    Transformation('fuse_pointwise_epilogue', ('producer', 'epilogue', 'schedule_id', 'entry_point'),
                   'Fuse a private rounded row intermediate into its only pointwise consumer.'),
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
           for key in declaration.parameters if key != 'num_warps'):
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
                return _refuse('input_refused', ', '.join(f.code for f in assessment.findings), (stage.name,))
        if transformation == 'fuse_pointwise_epilogue':
            return _fuse(compiler, program, **parameters)
        from .passes import specialize_triton_warps, specialize_output_columns
        transform = (specialize_triton_warps if transformation == 'specialize_triton_warps'
                     else specialize_output_columns)
        return _specialize(compiler, program, transform=transform, **parameters)
    except (CompilerError, ScheduleParseError, ValueError, TypeError) as error:
        return _refuse('result_refused', str(error))
