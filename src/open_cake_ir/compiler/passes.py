"""Explicit, bounded Schedule transformations; never part of implicit lowering.

The caller supplies the composition boundary, not a model graph: producer's sole
output is private to this composition and consumed only by the supplied epilogue.
The result has producer inputs and the epilogue output. Old Workload seals do not
apply to that new ABI, so its metadata is deliberately unbound.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Mapping

from .errors import CompilerError
from .ir import (AccessIndexKind, BufferMode, DType, ElementwiseOp, LoweringBackend,
                 MemorySpace, OperationKind, ReduceOp, Schedule, ScheduleParseError)

if TYPE_CHECKING:
    from .core import Assessment, Compiler


@dataclass(frozen=True)
class FusionResult:
    assessment: Assessment | None
    reason: str
    message: str

    @property
    def applied(self) -> bool:
        return self.assessment is not None

    @property
    def schedule(self) -> dict | None:
        """Return a fresh projection of the accepted candidate, not a second authority."""
        return None if self.assessment is None else json.loads(self.assessment.schedule_bytes)


def _refuse(reason: str, message: str) -> FusionResult:
    return FusionResult(None, reason, message)


@dataclass(frozen=True)
class SpecializationResult:
    assessment: Assessment | None
    reason: str
    message: str

    @property
    def applied(self) -> bool:
        return self.assessment is not None

    @property
    def schedule(self) -> dict | None:
        """Return a fresh projection of the accepted candidate, not a second authority."""
        return None if self.assessment is None else json.loads(self.assessment.schedule_bytes)


def _whole_dimension(index, dimension: int) -> bool:
    return (index.source is AccessIndexKind.DIMENSION and index.dimension == dimension
            and index.offset == 0 and index.extent is None)


def _row_axis(schedule: Schedule, rows: int):
    mapping = schedule.program_map
    if mapping is None or mapping.persistent or len(mapping.axes) != 1:
        return None
    axis = mapping.axes[0]
    buffer = schedule.buffer(axis.buffer)
    if (axis.axis != 0 or axis.dimension != 0 or axis.tile != 1 or buffer is None
            or buffer.shape[0] != rows):
        return None
    return axis


def _whole_row(schedule: Schedule, op: str, buffer: str, axis: str) -> bool:
    access = schedule.access_map(op, buffer)
    if access is None or len(access.indices) != 2:
        return False
    row, col = access.indices
    return (row.source is AccessIndexKind.PROGRAM and row.name == axis
            and col.source is AccessIndexKind.DIMENSION and col.dimension == 1
            and col.offset == 0 and col.extent is None)


def fuse_pointwise_epilogue(compiler: Compiler, producer: Mapping, epilogue: Mapping, *,
                           private_intermediate: str, schedule_id: str,
                           entry_point: str) -> FusionResult:
    """Fuse one row-owned, low-precision materialization into a unary epilogue.

    This transforms the explicit composition epilogue(producer(inputs)); it cannot
    prove absence of consumers in an unseen graph. The caller must select a private
    intermediate and later bind the composed Workload. Neither source is mutated.
    Only direct-global Triton, one matching role/grid, no loops/state/synchronization,
    an explicit BF16/FP16 producer cast and a whole-row unary epilogue are admitted.
    FP32 materialization is refused: an identity cast is not a rounding barrier.
    """
    originals = []
    typed = []
    for label, document in (("producer", producer), ("epilogue", epilogue)):
        try:
            copied = deepcopy(dict(document))
            assessed = compiler.assess(copied)
            if not assessed.lowering_eligible:
                return _refuse('input_refused', f'{label}: ' + ', '.join(
                    f.code for f in assessed.findings if f.blocks_lowering or f.blocks_acceptance))
            originals.append(copied)
            typed.append(Schedule.from_dict(copied))
        except (CompilerError, ScheduleParseError, TypeError, ValueError) as error:
            return _refuse('input_refused', f'{label}: {error}')
    p, e = typed
    if (p.target != e.target or p.lowering.backend is not LoweringBackend.TRITON
            or e.lowering.backend is not LoweringBackend.TRITON):
        return _refuse('target_route', 'Both stages must use the same exact Target and Triton route.')
    if (not isinstance(schedule_id, str) or not schedule_id or schedule_id in {p.schedule_id, e.schedule_id}
            or not isinstance(entry_point, str) or not entry_point):
        return _refuse('result_identity', 'The composition needs a new nonempty Schedule id.')
    for label, s in zip(('producer', 'epilogue'), typed):
        if (s.allocations or s.pipelines or s.barriers or s.tile_loops or len(s.roles) != 1
                or any(op.waits or op.signals or op.pipeline for op in s.operations)
                or any(b.space not in {MemorySpace.GLOBAL, MemorySpace.REGISTER}
                       or b.mode is BufferMode.STATE or b.allocation is not None or b.byte_offset
                       or b.stages != 1 or b.swizzle or b.scale_of or b.valid_extent for b in s.buffers)):
            return _refuse('unsupported_effects', f'{label}: require one role and ordinary nonaliasing buffers without state, loops or synchronization.')
    if (p.roles[0].warps != e.roles[0].warps
            or p.roles[0].registers_per_thread != e.roles[0].registers_per_thread
            or p.residency != e.residency):
        return _refuse('execution_controls', 'Warp ownership and residency commitments must match.')
    ep_inputs = [b for b in e.buffers if b.space is MemorySpace.GLOBAL and b.mode is BufferMode.INPUT]
    if (p.outputs != (private_intermediate,) or len(e.outputs) != 1 or len(ep_inputs) != 1):
        return _refuse('composition_boundary', 'Require the named sole producer output and a unary single-output epilogue.')
    middle = p.buffer(private_intermediate)
    ep_input, ep_output = ep_inputs[0], e.buffer(e.outputs[0])
    if (middle is None or ep_output is None or len(middle.shape) != 2
            or ep_input.shape != middle.shape or ep_input.dtype != middle.dtype
            or ep_output.shape != middle.shape):
        return _refuse('intermediate_abi', 'Both stages must agree on the full rank-2 intermediate and output shape.')
    if middle.dtype not in {DType.BF16, DType.FP16}:
        return _refuse('rounding_boundary', 'Only an explicit BF16/FP16 rounding seam is preserved; FP32 store/load fusion is not admitted.')
    pa, ea = _row_axis(p, middle.shape[0]), _row_axis(e, middle.shape[0])
    if pa is None or ea is None:
        return _refuse('row_ownership', 'Each stage must own one complete row per program.')
    p_stores = [op for op in p.operations if op.kind is OperationKind.STORE]
    e_loads = [op for op in e.operations if op.kind is OperationKind.LOAD]
    e_stores = [op for op in e.operations if op.kind is OperationKind.STORE]
    if (len(p_stores) != 1 or p.operations[-1] != p_stores[0]
            or len(e_loads) != 1 or e.operations[0] != e_loads[0]
            or len(e_stores) != 1 or e.operations[-1] != e_stores[0]
            or any(op.kind not in {OperationKind.LOAD, OperationKind.CAST, OperationKind.ELEMENTWISE,
                                  OperationKind.REDUCE, OperationKind.MMA, OperationKind.STORE} for op in p.operations)
            or any(op.kind not in {OperationKind.LOAD, OperationKind.CAST,
                                  OperationKind.ELEMENTWISE, OperationKind.STORE} for op in e.operations)):
        return _refuse('operation_domain', 'Require a pure producer with one final store and a pointwise load/body/store epilogue.')
    store, load, final = p_stores[0], e_loads[0], e_stores[0]
    if (store.writes != (middle.name,) or load.reads != (ep_input.name,)
            or final.writes != (ep_output.name,) or len(store.reads) != 1 or len(load.writes) != 1
            or not _whole_row(p, store.op_id, middle.name, pa.name)
            or not _whole_row(e, load.op_id, ep_input.name, ea.name)
            or not _whole_row(e, final.op_id, ep_output.name, ea.name)):
        return _refuse('access_domain', 'Intermediate store, reload and final store must address the same complete row without permutation or slicing.')
    if any(middle.name in op.reads for op in p.operations):
        return _refuse('intermediate_consumers', 'The producer must not reread its exported intermediate.')
    bridge, loaded = p.buffer(store.reads[0]), e.buffer(load.writes[0])
    definitions = [op for op in p.operations if bridge is not None and bridge.name in op.writes]
    if (bridge is None or loaded is None or len(definitions) != 1
            or definitions[0].kind is not OperationKind.CAST
            or definitions[0].parameters.to != middle.dtype
            or bridge.dtype != middle.dtype or bridge.shape != (middle.shape[1],)
            or loaded.dtype != bridge.dtype or loaded.shape != bridge.shape):
        return _refuse('rounding_boundary', 'The forwarded value must be the explicit low-precision cast result, not its pre-round input.')
    # Reject hidden global arguments: every surviving global is a public input or output.
    if any(b.space is MemorySpace.GLOBAL and b.mode not in {BufferMode.INPUT, BufferMode.OUTPUT}
           for s in typed for b in s.buffers):
        return _refuse('composition_boundary', 'Global scratch ABI is outside this composition contract.')
    out = originals[0]
    out['schedule_id'] = schedule_id
    out['lowering'] = {'backend': 'triton', 'entry_point': entry_point}
    out['metadata'] = {}  # The task owner must bind a new composed Workload, not reuse a stage seal.
    out['buffers'] = [b for b in out['buffers'] if b['name'] != middle.name]
    out['operations'] = [op for op in out['operations'] if op['id'] != store.op_id]
    out['access_maps'] = [a for a in out.get('access_maps', []) if a['operation'] != store.op_id]
    used = {b.name for b in p.buffers} | {op.op_id for op in p.operations} | {pa.name, p.roles[0].name, entry_point}
    def fresh(name):
        proposed = 'ep_' + name
        while proposed in used:
            proposed = 'ep_' + proposed
        used.add(proposed)
        return proposed
    buffers = {b.name: fresh(b.name) for b in e.buffers if b.name not in {ep_input.name, loaded.name}}
    buffers[loaded.name] = bridge.name
    operations = {op.op_id: fresh(op.op_id) for op in e.operations if op != load}
    operations[load.op_id] = definitions[0].op_id
    for b in originals[1]['buffers']:
        if b['name'] not in {ep_input.name, loaded.name}:
            b['name'] = buffers[b['name']]
            out['buffers'].append(b)
    for op in originals[1]['operations']:
        if op['id'] == load.op_id:
            continue
        op['id'] = operations[op['id']]
        op['role'] = p.roles[0].name
        op['reads'] = [buffers[name] for name in op['reads']]
        op['writes'] = [buffers[name] for name in op['writes']]
        op['depends_on'] = [operations[name] for name in op.get('depends_on', [])]
        out['operations'].append(op)
    for access in originals[1].get('access_maps', []):
        if access['operation'] == load.op_id:
            continue
        access['operation'] = operations[access['operation']]
        access['buffer'] = buffers[access['buffer']]
        for coordinate in access['indices']:
            if coordinate.get('source') == 'program':
                coordinate['name'] = pa.name
        out['access_maps'].append(access)
    out['outputs'] = [buffers[ep_output.name]]
    try:
        assessed = compiler.assess(out)
    except (CompilerError, ScheduleParseError, TypeError, ValueError) as error:
        return _refuse('result_refused', str(error))
    if not assessed.lowering_eligible:
        return _refuse('result_refused', ', '.join(f.code for f in assessed.findings
            if f.blocks_lowering or f.blocks_acceptance))
    return FusionResult(assessed, 'applied', 'Removed one intermediate global store and reload; preserved the explicit low-precision cast. No performance qualification is implied.')


def specialize_output_columns(compiler: Compiler, schedule: Mapping, *,
                              schedule_id: str, entry_point: str) -> SpecializationResult:
    """Rewrite one row-programmed rank-2 contraction into per-column programs.

    The matched invariant is an independent output-column axis with a pointwise
    epilogue (F-2026-09-11-004): the source assigns one program to each output row,
    loads the whole second operand, reduces K for every column and stores one row.
    The candidate adds one scalar program axis over the column extent, loads one
    B[:, col] vector, reduces to one scalar and applies the same epilogue arithmetic
    to it. That trades more programs and strided column access for a factor-N
    reduction in per-program private contraction state. The global ABI is unchanged,
    so the original Workload binding stays valid. Only the Metal route is admitted:
    capability evidence exists on Apple family7 and family9 targets alone, and this
    pass never claims a speedup -- Lab selection chooses between the two schedules.
    """
    def refused(reason: str, message: str) -> SpecializationResult:
        return SpecializationResult(None, reason, message)

    try:
        copied = deepcopy(dict(schedule))
        assessed = compiler.assess(copied)
        if not assessed.lowering_eligible:
            return refused('input_refused', ', '.join(
                f.code for f in assessed.findings if f.blocks_lowering or f.blocks_acceptance))
    except (CompilerError, ScheduleParseError, TypeError, ValueError) as error:
        return refused('input_refused', f'{error}')
    s = Schedule.from_dict(copied)
    if (not isinstance(schedule_id, str) or not schedule_id or schedule_id == s.schedule_id
            or not isinstance(entry_point, str) or not entry_point):
        return refused('result_identity', 'The candidate needs a new nonempty Schedule id and entry point.')
    if s.lowering.backend is not LoweringBackend.METAL:
        return refused('target_route', 'Only the Metal route is admitted; column-specialization evidence exists on Apple targets alone.')
    if (s.allocations or s.pipelines or s.barriers or s.tile_loops or len(s.roles) != 1
            or any(op.waits or op.signals or op.pipeline for op in s.operations)
            or any(b.space not in {MemorySpace.GLOBAL, MemorySpace.REGISTER}
                   or b.mode is BufferMode.STATE or b.allocation is not None or b.byte_offset
                   or b.stages != 1 or b.swizzle or b.scale_of or b.valid_extent for b in s.buffers)):
        return refused('unsupported_effects', 'Require one role and ordinary nonaliasing buffers without state, loops or synchronization.')
    mapping = s.program_map
    row = None
    if mapping is not None and not mapping.persistent and len(mapping.axes) == 1:
        axis = mapping.axes[0]
        if axis.axis == 0 and axis.dimension == 0 and axis.tile == 1:
            row = axis
    if row is None:
        return refused('program_shape', 'Require one nonpersistent scalar program axis over the output-row extent.')
    if any(op.kind not in {OperationKind.LOAD, OperationKind.ELEMENTWISE, OperationKind.CAST,
                           OperationKind.REDUCE, OperationKind.STORE} for op in s.operations):
        return refused('operation_domain', 'Only loads, elementwise arithmetic, casts, one fold and one store participate in this mechanism.')
    stores = [op for op in s.operations if op.kind is OperationKind.STORE]
    folds = [op for op in s.operations if op.kind is OperationKind.REDUCE]
    if not folds:
        return refused('contraction_shape', 'The mechanism matches a rank-2 contraction; its fold is missing.')
    if len(folds) > 1:
        return refused('coupled_output_axis', 'A second fold reads across the output columns; the epilogue must stay pointwise per column.')
    if len(stores) != 1 or s.operations[-1] is not stores[0]:
        return refused('operation_domain', 'Require exactly one store, last in the program.')
    fold = folds[0]
    store = stores[0]
    if (fold.parameters.op is not ReduceOp.SUM or fold.parameters.axis != 0
            or len(fold.reads) != 1 or len(fold.writes) != 1):
        return refused('contraction_shape', 'Require a sum over axis 0 of one products buffer.')
    products = s.buffer(fold.reads[0])
    producers = [op for op in s.operations if fold.reads[0] in op.writes]
    if (products is None or len(products.shape) != 2 or len(producers) != 1
            or producers[0].kind is not OperationKind.ELEMENTWISE):
        return refused('contraction_shape', 'The fold input must be one rank-2 elementwise product.')
    product = producers[0]
    if (product.parameters.op is not ElementwiseOp.MUL or product.parameters.broadcast_axis != 0
            or len(product.reads) != 2 or len(product.writes) != 1):
        return refused('contraction_shape', 'Require one broadcast multiply of the row vector with the whole second operand.')
    operands = [s.buffer(name) for name in product.reads]
    if any(buffer is None for buffer in operands):
        return refused('contraction_shape', 'The multiply operands differ.')
    second = next((b for b in operands if len(b.shape) == 2), None)
    first = next((b for b in operands if len(b.shape) == 1), None)
    if (second is None or first is None or tuple(second.shape) != tuple(products.shape)):
        return refused('contraction_shape', 'Require one rank-2 and one rank-1 multiply operand.')
    def sole_producer(name: str):
        writers = [op for op in s.operations if name in op.writes]
        return writers[0] if len(writers) == 1 else None
    second_load = sole_producer(second.name)
    first_load = sole_producer(first.name)
    if (second_load is None or first_load is None or second_load.kind is not OperationKind.LOAD
            or first_load.kind is not OperationKind.LOAD or len(second_load.reads) != 1
            or len(first_load.reads) != 1):
        return refused('contraction_shape', 'Both multiply operands must be direct loads.')
    b_global = s.buffer(second_load.reads[0])
    a_global = s.buffer(first_load.reads[0])
    out_global = s.buffer(store.writes[0]) if len(store.writes) == 1 else None
    anchor = s.buffer(row.buffer)
    if (b_global is None or a_global is None or out_global is None or anchor is None
            or len(a_global.shape) != 2 or len(b_global.shape) != 2 or len(out_global.shape) != 2
            or a_global.shape[1] != b_global.shape[0]
            or out_global.shape != (a_global.shape[0], b_global.shape[1])
            or anchor.shape[row.dimension] != a_global.shape[0]):
        return refused('contraction_shape', 'Require a[M,K] with b[K,N] storing to out[M,N] under the row axis.')
    K, N = b_global.shape
    a_access = s.access_map(first_load.op_id, a_global.name)
    b_access = s.access_map(second_load.op_id, b_global.name)
    out_access = s.access_map(store.op_id, out_global.name)
    if (a_access is None or len(a_access.indices) != 2
            or a_access.indices[0].source is not AccessIndexKind.PROGRAM or a_access.indices[0].name != row.name
            or not _whole_dimension(a_access.indices[1], 1)):
        return refused('access_domain', 'The first operand must be loaded one whole row per program.')
    if (b_access is None or len(b_access.indices) != 2
            or not _whole_dimension(b_access.indices[0], 0) or not _whole_dimension(b_access.indices[1], 1)):
        return refused('access_domain', 'The second operand must be loaded whole; sliced or offset access is not this mechanism.')
    if (out_access is None or len(out_access.indices) != 2
            or out_access.indices[0].source is not AccessIndexKind.PROGRAM or out_access.indices[0].name != row.name
            or not _whole_dimension(out_access.indices[1], 1)):
        return refused('access_domain', 'The store must write one whole output row per program.')
    # Everything the fold's result reaches must be a per-column rank-1 value; any other
    # shape or a further fold there is what couples the output axis.
    column: dict[str, object] = {}
    pending = [fold.writes[0]]
    while pending:
        name = pending.pop()
        if name in column:
            continue
        buffer = s.buffer(name)
        if buffer is None or buffer.space is not MemorySpace.REGISTER or tuple(buffer.shape) != (N,):
            return refused('contraction_shape', f'Epilogue values must be per-column rank-1 buffers of extent {N}; {name!r} differs.')
        column[name] = buffer
        for op in s.operations:
            if name not in op.reads or op is store:
                continue
            if op.kind not in {OperationKind.ELEMENTWISE, OperationKind.CAST}:
                return refused('contraction_shape', 'Only pointwise arithmetic may consume the contraction result.')
            for operand in op.reads + op.writes:
                linked = s.buffer(operand)
                if linked is not None and linked.space is MemorySpace.REGISTER and operand not in column:
                    pending.append(operand)
    if len(store.reads) != 1 or store.reads[0] not in column:
        return refused('contraction_shape', 'The contraction result must reach the store through per-column values.')
    vector_loads = []
    for name in column:
        writer = sole_producer(name)
        if writer is fold:
            continue
        if writer is None:
            return refused('contraction_shape', f'Per-column value {name!r} has no single producer.')
        if writer.kind in {OperationKind.ELEMENTWISE, OperationKind.CAST}:
            continue
        if writer.kind is not OperationKind.LOAD or len(writer.reads) != 1:
            return refused('contraction_shape', 'Per-column values must come from pointwise arithmetic or one load.')
        source = s.buffer(writer.reads[0])
        access = s.access_map(writer.op_id, writer.reads[0])
        if (source is None or source.space is not MemorySpace.GLOBAL or tuple(source.shape) != (N,)
                or access is None or len(access.indices) != 1 or not _whole_dimension(access.indices[0], 0)):
            return refused('access_domain', 'Per-column vector loads must read one whole rank-1 global of the column extent.')
        vector_loads.append(writer)
    out = copied
    out['schedule_id'] = schedule_id
    out['lowering'] = {'backend': 'metal', 'entry_point': entry_point}
    used = {b.name for b in s.buffers} | {op.op_id for op in s.operations} | {row.name, s.roles[0].name, entry_point}
    column_axis = 'col'
    while column_axis in used:
        column_axis = 'col_' + column_axis
    out['program_map']['axes'].append(
        {'name': column_axis, 'axis': 1, 'buffer': b_global.name, 'dimension': 1, 'tile': 1})
    for buffer in out['buffers']:
        if buffer['name'] in {second.name, products.name}:
            buffer['shape'] = buffer['shape'][:1]
        elif buffer['name'] in column:
            buffer['shape'] = [1]
    for access in out['access_maps']:
        if access['operation'] == second_load.op_id and access['buffer'] == b_global.name:
            access['indices'][1] = {'source': 'program', 'name': column_axis}
        elif access['operation'] == store.op_id and access['buffer'] == out_global.name:
            access['indices'][1] = {'source': 'program', 'name': column_axis}
        elif any(access['operation'] == writer.op_id and access['buffer'] == writer.reads[0]
                 for writer in vector_loads):
            access['indices'][0] = {'source': 'program', 'name': column_axis}
    for op in out['operations']:
        if op['id'] == product.op_id:
            del op['parameters']['broadcast_axis']
    try:
        result = compiler.assess(out)
    except (CompilerError, ScheduleParseError, TypeError, ValueError) as error:
        return refused('result_refused', str(error))
    if not result.lowering_eligible:
        return refused('result_refused', ', '.join(f.code for f in result.findings
            if f.blocks_lowering or f.blocks_acceptance))
    return SpecializationResult(result, 'applied',
        'Added one output-column program axis and reduced per-program private contraction state by the column extent; the pointwise epilogue arithmetic is unchanged. No performance qualification is implied.')
