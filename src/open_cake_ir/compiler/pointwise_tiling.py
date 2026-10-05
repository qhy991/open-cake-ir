"""Explicit column tiles for independent pointwise outputs; no implicit choice.

The graph, arithmetic order, dtype, public tensors and storage effects remain.
Compiler guards admit the rewrite; Lab must select and measure each candidate.
"""
from copy import deepcopy
from collections.abc import Mapping

from .errors import CompilerError
from .ir import AccessIndexKind, BufferMode, MemorySpace, OperationKind, ScheduleParseError
from .passes import SpecializationResult


def _refuse(reason, message):
    return SpecializationResult(None, reason, message)


def tile_pointwise_outputs(compiler, schedule, *, output_tile, schedule_id, entry_point):
    """Partition one whole-row, pure rank-2 pointwise graph without a new IR form."""
    if not isinstance(schedule, Mapping):
        return _refuse('input_refused', 'Require a Schedule document object.')
    if type(output_tile) is not int or output_tile <= 0 or output_tile & (output_tile - 1):
        return _refuse('tile_extent', 'output_tile must be a positive power of two.')
    try:
        assessment = compiler.assess(schedule)
    except (CompilerError, ScheduleParseError, ValueError, TypeError) as error:
        return _refuse('input_refused', str(error))
    blockers = [f for f in assessment.findings if f.blocks_acceptance or f.blocks_lowering]
    whole_column_range_only = (assessment.accepted and bool(blockers)
                              and all(f.code == 'TRITON_ARANGE_RANGE_UNSUPPORTED' for f in blockers))
    if (not assessment.lowering_eligible and not whole_column_range_only) or assessment.typed_schedule is None:
        return _refuse('input_refused', ', '.join(f.code for f in assessment.findings
                                                if f.blocks_acceptance or f.blocks_lowering))
    s = assessment.typed_schedule
    if (not isinstance(schedule_id, str) or not schedule_id or schedule_id == s.schedule_id
            or not isinstance(entry_point, str) or not entry_point.isidentifier()):
        return _refuse('result_identity', 'Require a fresh Schedule id and valid entry point.')
    if (s.tile_loops or s.allocations or s.pipelines or s.barriers or s.residency
            or len(s.roles) != 1 or s.roles[0].registers_per_thread is not None
            or any(op.waits or op.signals or op.pipeline for op in s.operations)):
        return _refuse('execution_commitments', 'Require one pure role without loops, state, explicit storage or synchronization.')
    if (s.program_map is None or s.program_map.persistent or len(s.program_map.axes) != 1
            or s.program_map.axes[0].axis != 0 or s.program_map.axes[0].dimension != 0
            or s.program_map.axes[0].tile != 1):
        return _refuse('program_shape', 'Require one nonpersistent scalar row program axis.')
    allowed = {OperationKind.LOAD, OperationKind.ELEMENTWISE, OperationKind.CAST,
               OperationKind.COMPARE, OperationKind.SELECT, OperationKind.STORE}
    if any(op.kind not in allowed for op in s.operations):
        return _refuse('coupled_output_axis', 'Require pointwise arithmetic; reductions, contractions, scans, atomics and indexed gathers may couple output columns.')
    outputs = [s.buffer(name) for name in s.outputs]
    if (not outputs or any(b is None or b.space is not MemorySpace.GLOBAL
                           or b.mode is not BufferMode.OUTPUT or len(b.shape) != 2 for b in outputs)):
        return _refuse('output_domain', 'Require ordinary rank-2 global outputs.')
    rows, columns = outputs[0].shape
    if any(b.shape != (rows, columns) for b in outputs):
        return _refuse('output_domain', 'All outputs must share the row and independent column extents.')
    if output_tile >= columns:
        return _refuse('tile_extent', 'The output tile must be smaller than the whole column extent.')
    row = s.program_map.axes[0]
    anchor = s.buffer(row.buffer)
    if anchor is None or anchor.shape[0] != rows:
        return _refuse('program_shape', 'The row program must span the output rows.')
    if any(b.space not in {MemorySpace.GLOBAL, MemorySpace.REGISTER}
           or b.mode is BufferMode.STATE or b.allocation is not None or b.byte_offset
           or b.stages != 1 or b.swizzle or b.scale_of or b.valid_extent for b in s.buffers):
        return _refuse('storage_domain', 'Require ordinary nonaliasing global/register buffers without state, views or validity relations.')
    registers = [b for b in s.buffers if b.space is MemorySpace.REGISTER]
    if any(b.shape != (columns,) for b in registers):
        return _refuse('value_shape', 'All internal pointwise values must be column vectors of the same extent.')
    if any(op.kind is OperationKind.ELEMENTWISE and op.parameters.broadcast_axis is not None
           for op in s.operations):
        return _refuse('broadcast_domain', 'Require aligned vectors and scalar arithmetic, not cross-axis broadcasting.')
    if any(b.space is MemorySpace.GLOBAL and b.mode not in {BufferMode.INPUT, BufferMode.OUTPUT}
           for b in s.buffers):
        return _refuse('storage_domain', 'Global buffers must be immutable inputs or declared outputs.')
    stores = [op for op in s.operations if op.kind is OperationKind.STORE]
    if (len(stores) != len(outputs) or {op.writes[0] for op in stores} != set(s.outputs)
            or any(len(op.reads) != 1 or len(op.writes) != 1 for op in stores)):
        return _refuse('output_ownership', 'Require exactly one ordinary writer for each public output.')
    # Access geometry proves independence, rather than inferring it from an
    # operator name, Target or a successful single-shape timing.
    column_dimensions = {}
    for access in s.access_maps:
        b = s.buffer(access.buffer)
        if b is None or b.space is not MemorySpace.GLOBAL:
            return _refuse('access_domain', 'Only ordinary global load/store accesses participate.')
        op = s.operation(access.operation)
        if op is None or op.kind not in {OperationKind.LOAD, OperationKind.STORE}:
            return _refuse('access_domain', 'Require direct global loads and stores.')
        if b.shape == (rows, columns) and len(access.indices) == 2:
            first, second = access.indices
            if (first.source is not AccessIndexKind.PROGRAM or first.name != row.name
                    or first.offset or first.extent is not None):
                return _refuse('access_domain', 'Rank-2 accesses must use the same scalar row program.')
            dim = 1
        elif b.mode is BufferMode.INPUT and b.shape == (columns,) and len(access.indices) == 1:
            second = access.indices[0]
            dim = 0
        else:
            return _refuse('access_domain', 'Require whole-row rank-2 accesses or whole-column-vector inputs.')
        if (second.source is not AccessIndexKind.DIMENSION or second.dimension != dim
                or second.offset or second.extent is not None
                or access.boundary.value != 'mask_tiled_axes'):
            return _refuse('access_domain', 'Require whole independent columns and declared tail masking.')
        column_dimensions[(access.operation, access.buffer)] = dim
    for op in s.operations:
        if op.kind is OperationKind.LOAD:
            if len(op.reads) != 1 or len(op.writes) != 1:
                return _refuse('operation_domain', 'Require one directly loaded column vector per load.')
            if (op.op_id, op.reads[0]) not in column_dimensions:
                return _refuse('access_domain', 'Every global load must have the proven column access.')
        elif op.kind is OperationKind.STORE:
            if (op.op_id, op.writes[0]) not in column_dimensions:
                return _refuse('access_domain', 'Every store must have the proven column access.')
        elif any(s.buffer(name).space is not MemorySpace.REGISTER for name in op.reads + op.writes):
            return _refuse('value_storage', 'Pointwise arithmetic must consume and produce register values.')
    d = deepcopy(dict(schedule))
    names = {b.name for b in s.buffers} | {r.name for r in s.roles} | {op.op_id for op in s.operations} | {row.name}
    col = 'output_column'
    while col in names: col += '_tile'
    d['schedule_id'] = schedule_id
    d['lowering']['entry_point'] = entry_point
    d['program_map']['axes'].append(dict(name=col, axis=1, buffer=outputs[0].name,
                                          dimension=1, tile=output_tile))
    for buffer in d['buffers']:
        if buffer['space'] == 'register': buffer['shape'] = [output_tile]
    for access in d['access_maps']:
        dim = column_dimensions[(access['operation'], access['buffer'])]
        access['indices'][dim] = dict(source='program_tile', name=col)
    try:
        result = compiler.assess(d)
        if not result.lowering_eligible:
            return _refuse('result_refused', ', '.join(f.code for f in result.findings
                                                      if f.blocks_lowering or f.blocks_acceptance))
        compiler.lower(result)
    except (CompilerError, ScheduleParseError, ValueError, TypeError) as error:
        return _refuse('result_refused', str(error))
    return SpecializationResult(result, 'applied', 'Partitioned independent pointwise output columns; arithmetic and public ABI preserved. No performance choice or qualification is implied.')
