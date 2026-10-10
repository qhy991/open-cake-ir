"""Explicit FP32 multiply/sum to tiled IEEE MMA candidate construction.

This pass preserves public tensors and the pointwise epilogue. It changes the
contraction's accumulation order, so the original external oracle must qualify
the result. Tile selection and performance acceptance belong to the caller.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

from .errors import CompilerError
from .ir import (AccessIndexKind, BufferMode, DType, ElementwiseOp, LoweringBackend,
                 MemorySpace, OperationKind, ReduceOp, ScheduleParseError)
from .passes import SpecializationResult


def _refuse(reason, message):
    return SpecializationResult(None, reason, message)


@dataclass(frozen=True)
class _Contraction:
    document: dict
    left: str
    right: str
    fold: str
    product: str
    k_axis: int
    epilogue_shapes: dict[str, str]


def _match(compiler, schedule):
    if not isinstance(schedule, Mapping):
        return _refuse('input_refused', 'Require a Schedule document object.')
    try:
        assessment = compiler.assess(schedule)
    except (CompilerError, ScheduleParseError, ValueError, TypeError) as error:
        return _refuse('input_refused', str(error))
    blockers = [f for f in assessment.findings if f.blocks_lowering or f.blocks_acceptance]
    # Fixed masked tiles can replace whole non-power-of-two arange extents.
    range_only = (assessment.accepted and blockers
                  and all(f.code == 'TRITON_ARANGE_RANGE_UNSUPPORTED' for f in blockers))
    if (not assessment.lowering_eligible and not range_only) or assessment.typed_schedule is None:
        return _refuse('input_refused', ', '.join(f.code for f in blockers))
    s = assessment.typed_schedule
    if s.lowering.backend is not LoweringBackend.TRITON or s.target != 'gfx938':
        return _refuse('target_route', 'This rewrite has retained source evidence only for gfx938 Triton.')
    if (s.tile_loops or s.allocations or s.pipelines or s.barriers or s.residency
            or len(s.roles) != 1 or s.roles[0].registers_per_thread is not None
            or s.roles[0].execution_groups != tuple(range(len(s.roles[0].execution_groups)))
            or any(op.waits or op.signals or op.pipeline for op in s.operations)):
        return _refuse('execution_commitments', 'Require one ordinary role without loops, storage, synchronization or residency commitments.')
    if (s.program_map is None or s.program_map.persistent or len(s.program_map.axes) != 1
            or s.program_map.axes[0].axis != 0 or s.program_map.axes[0].dimension != 0
            or s.program_map.axes[0].tile != 1):
        return _refuse('program_shape', 'Require one nonpersistent scalar row program axis.')
    if any(b.dtype is not DType.FP32 or b.space not in {MemorySpace.GLOBAL, MemorySpace.REGISTER}
           or b.mode is BufferMode.STATE or b.allocation is not None or b.byte_offset
           or b.stages != 1 or b.swizzle or b.scale_of or b.valid_extent
           for b in s.buffers):
        return _refuse('storage_domain', 'Require ordinary independent FP32 global/register buffers without state, aliases or validity relations.')
    allowed = {OperationKind.LOAD, OperationKind.ELEMENTWISE, OperationKind.REDUCE, OperationKind.STORE}
    if any(op.kind not in allowed for op in s.operations):
        return _refuse('operation_domain', 'Require ordinary loads, one multiply/sum contraction, pointwise arithmetic and one store.')
    folds = [op for op in s.operations if op.kind is OperationKind.REDUCE]
    stores = [op for op in s.operations if op.kind is OperationKind.STORE]
    if len(folds) != 1 or len(stores) != 1 or s.operations[-1] is not stores[0]:
        return _refuse('contraction_shape', 'Require one K sum and one final store; output-coupling reductions are outside this rewrite.')
    fold, store = folds[0], stores[0]
    if (fold.parameters.op is not ReduceOp.SUM or fold.parameters.scope.value != 'cta'
            or fold.parameters.across_loop or len(fold.reads) != 1 or len(fold.writes) != 1):
        return _refuse('contraction_shape', 'Require one non-loop-carried CTA sum of the product.')
    writers = {}
    for op in s.operations:
        for name in op.writes:
            if name in writers:
                return _refuse('value_ownership', 'Every value and output must have one writer.')
            writers[name] = op
    product = writers.get(fold.reads[0])
    if (product is None or product.kind is not OperationKind.ELEMENTWISE
            or product.parameters.op is not ElementwiseOp.MUL
            or len(product.reads) != 2 or len(product.writes) != 1
            or product.parameters.scalar is not None or product.parameters.instruction is not None
            or fold.parameters.axis not in (0, 1)
            or product.parameters.broadcast_axis != fold.parameters.axis):
        return _refuse('contraction_shape', 'Require a broadcast multiply followed by sum along the vector operand axis.')
    operands = [s.buffer(name) for name in product.reads]
    left = next((b for b in operands if len(b.shape) == 1), None)
    right = next((b for b in operands if len(b.shape) == 2), None)
    if left is None or right is None:
        return _refuse('contraction_shape', 'Require a row vector multiplied with a rank-two right operand.')
    a, b = writers.get(left.name), writers.get(right.name)
    if any(op is None or op.kind is not OperationKind.LOAD or len(op.reads) != 1 for op in (a, b)):
        return _refuse('operand_provenance', 'Both contraction operands must be direct global loads without intervening arithmetic or rounding.')
    ag, bg, output = s.buffer(a.reads[0]), s.buffer(b.reads[0]), s.buffer(store.writes[0])
    k_axis = fold.parameters.axis
    if (len(ag.shape) != 2 or len(bg.shape) != 2 or len(output.shape) != 2
            or ag.shape[1] != bg.shape[k_axis]
            or output.shape != (ag.shape[0], bg.shape[1 - k_axis])
            or len({ag.name, bg.name, output.name}) != 3
            or any(g.space is not MemorySpace.GLOBAL or g.mode is not BufferMode.INPUT for g in (ag, bg))
            or output.space is not MemorySpace.GLOBAL or output.mode is not BufferMode.OUTPUT
            or s.outputs != (output.name,)):
        return _refuse('storage_domain', 'Require independent input A[M,K], B[K,N] or B[N,K], and output C[M,N].')
    row = s.program_map.axes[0]
    anchor = s.buffer(row.buffer)
    if anchor.space is not MemorySpace.GLOBAL or anchor.shape[0] != ag.shape[0]:
        return _refuse('program_shape', 'The row program must use an unchanged global anchor spanning the left operand rows.')
    def whole(index, dimension):
        return (index.source is AccessIndexKind.DIMENSION and index.dimension == dimension
                and index.offset == 0 and index.extent is None)
    def row_index(index):
        return (index.source is AccessIndexKind.PROGRAM and index.name == row.name
                and index.offset == 0 and index.extent is None)
    def geometry(op, global_buffer, kind):
        access = s.access_map(op.op_id, global_buffer.name)
        if access is None or access.boundary.value != 'mask_tiled_axes':
            return False
        indices = access.indices
        if kind == 'whole':
            return len(indices) == len(global_buffer.shape) and all(whole(i, d) for d, i in enumerate(indices))
        return len(indices) == 2 and row_index(indices[0]) and whole(indices[1], 1)
    if (not geometry(a, ag, 'row') or not geometry(b, bg, 'whole')
            or not geometry(store, output, 'row')
            or left.shape != (ag.shape[1],) or right.shape != bg.shape):
        return _refuse('access_domain', 'Require whole row/whole matrix operand loads and a whole output-row store.')
    contraction_ids = {a.op_id, b.op_id, product.op_id, fold.op_id}
    # Contracted intermediates cannot also escape into the epilogue.
    for value, consumer in ((left.name, product), (right.name, product), (product.writes[0], fold)):
        if any(value in op.reads and op is not consumer for op in s.operations):
            return _refuse('value_ownership', 'Contraction loads and products must be private to this multiply/sum.')
    columns = output.shape[1]
    shapes = {fold.writes[0]: 'matrix'}
    for op in s.operations:
        if op.op_id in contraction_ids:
            continue
        if op is store:
            if len(op.reads) != 1 or shapes.get(op.reads[0]) != 'matrix':
                return _refuse('epilogue_domain', 'The store must depend on the contraction through pointwise values.')
            continue
        if len(op.writes) != 1 or s.buffer(op.writes[0]).shape != (columns,):
            return _refuse('epilogue_domain', 'Every epilogue value must be a full output-column vector before tiling.')
        if op.kind is OperationKind.LOAD:
            source = s.buffer(op.reads[0])
            if source.space is not MemorySpace.GLOBAL or source.mode is not BufferMode.INPUT:
                return _refuse('storage_domain', 'Epilogue loads must read immutable global inputs.')
            if source.shape == (columns,) and geometry(op, source, 'whole'):
                shapes[op.writes[0]] = 'column'
            elif source.shape == output.shape and geometry(op, source, 'row'):
                shapes[op.writes[0]] = 'matrix'
            else:
                return _refuse('access_domain', 'Epilogue inputs must be whole column vectors or matching output rows.')
        elif (op.kind is OperationKind.ELEMENTWISE and op.parameters.broadcast_axis is None
              and all(name in shapes for name in op.reads)):
            kinds = {shapes[name] for name in op.reads}
            if len(kinds) > 1 and len(op.reads) != 2:
                return _refuse('epilogue_domain', 'Mixed matrix/vector epilogue operands require an ordinary binary broadcast.')
            shapes[op.writes[0]] = 'matrix' if 'matrix' in kinds else 'column'
        else:
            return _refuse('epilogue_domain', 'Only aligned pointwise FP32 arithmetic may consume the contraction result.')
    # Pure data dependencies can be rebuilt after replacing the product. Extra
    # ordering commitments, dead values and unused public tensors are not matched.
    live = set(store.reads)
    for op in reversed(s.operations[:-1]):
        if not set(op.writes) <= live:
            return _refuse('value_ownership', 'Every original operation must contribute to the output.')
        live.update(op.reads)
    used_globals = {name for op in s.operations for name in op.reads + op.writes
                    if s.buffer(name).space is MemorySpace.GLOBAL}
    if used_globals != {g.name for g in s.buffers if g.space is MemorySpace.GLOBAL}:
        return _refuse('storage_domain', 'Every public tensor must participate in the matched graph.')
    for op in s.operations:
        data_dependencies = {writers[name].op_id for name in op.reads if name in writers}
        if not set(op.depends_on) <= data_dependencies:
            return _refuse('execution_commitments', 'Only data-producer dependencies may be reordered by this rewrite.')
    return _Contraction(deepcopy(dict(schedule)), a.op_id, b.op_id, fold.op_id,
                        product.op_id, k_axis, shapes)


def specialize_fp32_contraction(compiler, schedule, *, row_tile, column_tile, k_tile,
                                num_warps, num_stages, schedule_id, entry_point):
    """Construct one explicitly selected gfx938 IEEE FP32 MMA candidate.

    The tile domain is the retained rank-two dot family, with power-of-two tile
    extents of at least 16 and at least two K iterations. Global row/column
    extents may be smaller or have masked tails;
    this does not assert that scalar M is unrepresentable. No numerical or
    performance acceptance is inferred from Compiler admission.
    """
    matched = _match(compiler, schedule)
    if isinstance(matched, SpecializationResult):
        return matched
    d = matched.document
    if (not isinstance(schedule_id, str) or not schedule_id or schedule_id == d['schedule_id']
            or not isinstance(entry_point, str) or not entry_point.isidentifier()):
        return _refuse('result_identity', 'Require a fresh Schedule id and valid entry point.')
    for name, value in (('row_tile', row_tile), ('column_tile', column_tile), ('k_tile', k_tile)):
        if type(value) is not int or value < 16 or value & (value - 1):
            return _refuse('tile_extent', f'{name}: require a power of two of at least 16 in this rewrite domain.')
    if type(num_warps) is not int or num_warps <= 0 or num_warps & (num_warps - 1):
        return _refuse('warp_count', 'num_warps must be a positive power of two.')
    maximum = compiler._revision.targets[d['target']].resource_limits.maximum_warps_per_cta
    if num_warps > maximum:
        return _refuse('warp_count', f'The selected Target admits at most {maximum} execution groups per CTA.')
    if type(num_stages) is not int or num_stages < 1:
        return _refuse('stage_count', 'num_stages must be a positive integer.')
    ops = {op['id']: op for op in d['operations']}
    buffers = {b['name']: b for b in d['buffers']}
    left, right, fold, product = (ops[name] for name in (matched.left, matched.right, matched.fold, matched.product))
    ag, bg = left['reads'][0], right['reads'][0]
    if k_tile >= buffers[ag]['shape'][1]:
        return _refuse('tile_extent', 'k_tile must be smaller than K; this rewrite constructs a multi-trip K loop.')
    row = d['program_map']['axes'][0]
    names = set(buffers) | set(ops) | {row['name'], entry_point, d['roles'][0]['name']}
    def fresh(base):
        while base in names:
            base += '_'
        names.add(base)
        return base
    column, iterator, loop = fresh('mma_columns'), fresh('mma_k'), fresh('mma_k_loop')
    row['tile'] = row_tile
    d['program_map']['axes'].append(dict(name=column, axis=1, buffer=bg,
                                        dimension=1 - matched.k_axis, tile=column_tile))
    row_index = dict(source='program_tile', name=row['name'])
    column_index = dict(source='program_tile', name=column)
    k_index = dict(source='loop_tile', name=iterator)
    buffers[left['writes'][0]]['shape'] = [row_tile, k_tile]
    buffers[right['writes'][0]]['shape'] = ([k_tile, column_tile] if matched.k_axis == 0 else [column_tile, k_tile])
    body = [left, right]
    rhs = right['writes'][0]
    if matched.k_axis == 0:
        rhs = fresh('mma_right_nk')
        d['buffers'].append(dict(name=rhs, space='register', dtype='fp32', mode='scratch', shape=[column_tile, k_tile]))
        body.append(dict(id=fresh('mma_transpose_right'), kind='transpose', role=fold['role'],
                         reads=list(right['writes']), writes=[rhs], parameters={}))
    fold.update(kind='mma', reads=[left['writes'][0], rhs], parameters=dict(
        accumulator='fp32', instruction={'contract': 'triton.dot.fp32_ieee'},
        tile_shape=[row_tile, column_tile, k_tile]))
    body.append(fold)
    removed = {matched.left, matched.right, matched.product, matched.fold}
    epilogue = [op for op in d['operations'] if op['id'] not in removed]
    for name, kind in matched.epilogue_shapes.items():
        buffers[name]['shape'] = [row_tile, column_tile] if kind == 'matrix' else [column_tile]
    for op in epilogue:
        if op['kind'] == 'elementwise':
            kinds = {matched.epilogue_shapes[name] for name in op['reads']}
            if len(kinds) > 1:
                op['parameters']['broadcast_axis'] = 1
    for access in d['access_maps']:
        op = ops[access['operation']]
        if op is left:
            access['indices'] = [dict(row_index), dict(k_index)]
        elif op is right:
            access['indices'] = ([dict(k_index), dict(column_index)] if matched.k_axis == 0
                                 else [dict(column_index), dict(k_index)])
        else:
            access['indices'] = ([dict(column_index)] if len(buffers[access['buffer']]['shape']) == 1
                                 else [dict(row_index), dict(column_index)])
    d['buffers'] = [b for b in d['buffers'] if b['name'] not in product['writes']]
    d['operations'] = body + epilogue
    producers = {}
    for op in d['operations']:
        op['depends_on'] = list(dict.fromkeys(producers[name] for name in op['reads'] if name in producers))
        producers.update({name: op['id'] for name in op['writes']})
    d['tile_loops'] = [dict(name=loop, iterator=iterator, buffer=ag, dimension=1, tile=k_tile,
        body=[op['id'] for op in body], range_options=dict(num_stages=num_stages,
        loop_unroll_factor=1, disallow_acc_multi_buffer=False, flatten=False,
        warp_specialize=False, disable_licm=False))]
    d['roles'][0]['execution_groups'] = list(range(num_warps))
    d['schedule_id'] = schedule_id
    d['lowering']['entry_point'] = entry_point
    try:
        assessment = compiler.assess(d)
        if not assessment.lowering_eligible:
            return _refuse('result_refused', '; '.join(f'{f.code}: {f.message}' for f in assessment.findings
                                                     if f.blocks_lowering or f.blocks_acceptance))
        compiler.lower(assessment)
    except (CompilerError, ScheduleParseError, ValueError, TypeError) as error:
        return _refuse('result_refused', str(error))
    return SpecializationResult(assessment, 'applied',
        'Replaced the FP32 multiply/sum with explicitly tiled triton.dot.fp32_ieee and retained the pointwise epilogue, public tensors and workload metadata. '
        'Accumulation order changes; bitwise equivalence is not claimed. The original external oracle and target measurement must qualify this candidate.')
