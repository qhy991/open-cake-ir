"""Guarded explicit tiling of a pure FP32 squared-difference reduction.

The rewrites share graph, arithmetic, storage and access admission. None adds
an instruction, target fact, physical register guarantee or performance claim.
"""
from copy import deepcopy
from dataclasses import dataclass
from .passes import SpecializationResult
from .errors import CompilerError


def _refuse(reason,message):
    return SpecializationResult(None,reason,message)


@dataclass
class _SquaredDifferenceInput:
    document: dict
    schedule: object
    point: dict
    centroids: dict
    output: dict


def _guard_squared_difference(schedule):
    refuse = _refuse
    from .ir import Schedule
    from .errors import CompilerError
    try:
        s = Schedule.from_dict(schedule)
    except (CompilerError, ValueError, TypeError) as error:
        return refuse('input_refused', str(error))
    d = deepcopy(dict(schedule))
    if s.lowering.backend.value != 'triton':
        return refuse('target_route', 'This rewrite emits an explicit Triton reduction loop.')
    for path, incompatible, requirement in (
        ('tile_loops', bool(s.tile_loops), 'an original loop-free reduction'),
        ('allocations', bool(s.allocations), 'ordinary unallocated storage'),
        ('pipelines', bool(s.pipelines), 'no existing pipeline'),
        ('barriers', bool(s.barriers), 'no existing barrier'),
        ('residency', bool(s.residency), 'no existing residency commitment'),
        ('roles', len(s.roles) != 1, 'one role'),
        ('roles[0].registers_per_thread', len(s.roles) == 1 and s.roles[0].registers_per_thread is not None,
         'no role register budget'),
        ('program_map', s.program_map is None or s.program_map.persistent
         or len(s.program_map.axes) != 1, 'one nonpersistent row axis'),
    ):
        if incompatible:
            return refuse('execution_commitments', f'{path}: this pass requires {requirement}.')
    for index, op in enumerate(s.operations):
        for field in ('waits', 'signals', 'pipeline'):
            if getattr(op, field):
                return refuse('execution_commitments',
                              f'operations[{index}].{field}: this pass requires no synchronization or pipeline commitment.')
    ops = d['operations']
    if len(ops) != 6 or [op['kind'] for op in ops] != ['load','load','elementwise','elementwise','reduce','store']:
        return refuse('operation_domain', 'Require two loads, subtraction, square, sum, and store only.')
    x,c,sub,square,fold,store = ops
    if (sub['parameters'] != {'op':'sub','broadcast_axis':1}
        or square['parameters'] != {'op':'square'}
        or fold['parameters'] != {'op':'sum','axis':1,'scope':'cta','across_loop':False}
        or any(len(op['writes']) != 1 for op in ops)
        or sub['reads'] != [c['writes'][0],x['writes'][0]]
        or square['reads'] != sub['writes'] or fold['reads'] != square['writes']
        or store['reads'] != fold['writes'] or len(x['reads']) != 1 or len(c['reads']) != 1):
        return refuse('arithmetic_domain', 'Require the canonical rounded difference then square and sum.')
    buffers = {b['name']:b for b in d['buffers']}
    if any(name not in buffers for op in ops for name in op['reads']+op['writes']):
        return refuse('storage_domain', 'Every operation operand must name a declared buffer.')
    xb,cb,ob = (buffers.get(op[key][0]) for op,key in ((x,'reads'),(c,'reads'),(store,'writes')))
    if (any(b is None for b in (xb,cb,ob)) or len(xb['shape']) != 2 or len(cb['shape']) != 2
        or ob['shape'] != [xb['shape'][0],cb['shape'][0]] or xb['shape'][1] != cb['shape'][1]
        or len({xb['name'],cb['name'],ob['name']}) != 3
        or any(b['dtype'] != 'fp32' for b in d['buffers'])
        or any(b.space.value not in {'global','register'} or b.allocation is not None or b.byte_offset
               or b.stages != 1 or b.swizzle or b.scale_of or b.valid_extent for b in s.buffers)
        or {b['name'] for b in d['buffers'] if b['space']=='global'} != {xb['name'],cb['name'],ob['name']}
        or xb['mode'] != 'input' or cb['mode'] != 'input' or ob['mode'] != 'output'):
        return refuse('storage_domain', 'Require ordinary FP32 x[R,K], c[N,K], out[R,N].')
    K,N = xb['shape'][1],cb['shape'][0]
    axis = d['program_map']['axes'][0]
    row = {'source':'program','name':axis['name']}
    dim = lambda n: {'source':'dimension','dimension':n}
    expected_access = [dict(operation=x['id'],buffer=xb['name'],indices=[row,dim(1)],boundary='mask_tiled_axes'),
                       dict(operation=c['id'],buffer=cb['name'],indices=[dim(0),dim(1)],boundary='mask_tiled_axes'),
                       dict(operation=store['id'],buffer=ob['name'],indices=[row,dim(1)],boundary='mask_tiled_axes')]
    if (axis != dict(name=axis['name'],axis=0,buffer=xb['name'],dimension=0,tile=1)
        or sorted(d['access_maps'],key=lambda a:a['operation']) != sorted(expected_access,key=lambda a:a['operation'])
        or d['outputs'] != [ob['name']]
        or any(buffers[op['writes'][0]]['shape'] != shape for op,shape in
               ((x,[K]),(c,[N,K]),(sub,[N,K]),(square,[N,K]),(fold,[N])))):
        return refuse('access_domain', 'Require a whole-row load, whole centroid load and whole-row output.')
    return _SquaredDifferenceInput(d,s,xb,cb,ob)


def _result_identity(input,schedule_id,entry_point):
    if (not isinstance(schedule_id,str) or not schedule_id or schedule_id==input.schedule.schedule_id
        or not isinstance(entry_point,str) or not entry_point.isidentifier()):
        return _refuse('result_identity','Choose a fresh Schedule identity and valid entry point.')
    return None


def _assess_result(compiler,document,message):
    try:
        assessment=compiler.assess(document)
        if not assessment.lowering_eligible:
            return _refuse('result_refused', '; '.join(
                f'{f.category.value}: {f.code} at {f.path}: {f.message}'
                for f in assessment.findings if f.blocks_lowering or f.blocks_acceptance))
        compiler.lower(assessment)
    except (CompilerError,ValueError,TypeError) as error:
        return _refuse('result_refused',str(error))
    return SpecializationResult(assessment,'applied',message)


def _fresh_name(base, names):
    while base in names:
        base += '_'
    names.add(base)
    return base


def _build_candidate(compiler, input, *, output_tile, k_tile, loop_unroll_factor,
                     schedule_id, entry_point):
    """One constructor for the admitted canonical graph; no implicit selection."""
    d = input.document
    K, N = input.point['shape'][1], input.centroids['shape'][0]
    ops = d['operations']
    x, c, sub, square, fold, store = ops
    buffers = {b['name']: b for b in d['buffers']}
    row = d['program_map']['axes'][0]
    names = set(buffers) | {op['id'] for op in ops} | {row['name']}
    d['schedule_id'] = schedule_id
    d['lowering']['entry_point'] = entry_point
    if output_tile < N:
        column = _fresh_name('output_columns', names)
        row['axis'] = 1
        d['program_map']['axes'].insert(0, dict(name=column, axis=0,
            buffer=input.centroids['name'], dimension=0, tile=output_tile))
        for op in (c, sub, square, fold):
            buffers[op['writes'][0]]['shape'][0] = output_tile
        for access in d['access_maps']:
            if access['operation'] == c['id']:
                access['indices'][0] = {'source': 'program_tile', 'name': column}
            elif access['operation'] == store['id']:
                access['indices'][1] = {'source': 'program_tile', 'name': column}
    if k_tile < K:
        iterator = _fresh_name('contracted_tile', names)
        loop_name = _fresh_name(iterator + '_loop', names)
        for op in (x, c, sub, square):
            buffers[op['writes'][0]]['shape'][-1] = k_tile
        for access in d['access_maps']:
            if access['operation'] in {x['id'], c['id']}:
                access['indices'][-1] = {'source': 'loop_tile', 'name': iterator}
        del fold['parameters']['across_loop']
        d['tile_loops'] = [dict(name=loop_name, iterator=iterator,
            buffer=input.point['name'], dimension=1, tile=k_tile,
            body=[op['id'] for op in ops[:-1]], range_options=dict(num_stages=1,
            loop_unroll_factor=loop_unroll_factor, disallow_acc_multi_buffer=False,
            flatten=False, warp_specialize=False, disable_licm=False))]
    columns_per_row = (N + output_tile - 1) // output_tile
    trips = (K + k_tile - 1) // k_tile
    rows = input.point['shape'][0]
    message = (
        f'Intermediate tile [{N}, {K}] -> [{output_tile}, {k_tile}]; '
        f'logical elements {N*K} -> {output_tile*k_tile}. '
        f'Static CTAs {rows} -> {rows*columns_per_row}; '
        f'row-input load multiplicity 1 -> {columns_per_row}. '
        f'K trips {trips}, unroll factor {loop_unroll_factor}. '
        'FP32 subtraction precedes square; K tiling can change reduction rounding. '
        'Logical sizes and repeated loads do not predict physical allocation, memory traffic or latency; '
        'external-oracle and target measurement are required.'
    )
    return _assess_result(compiler, d, message)


def specialize_squared_difference(compiler, schedule, *, output_tile, k_tile,
                                  loop_unroll_factor, schedule_id, entry_point):
    """Build a jointly selected N/K/unroll candidate from the canonical seed.

    An extent-sized tile preserves that dimension. Smaller tiles are powers of
    two; a single-stage K loop has a fixed, evenly unrolled ceiling trip count.
    This is a candidate constructor, not a performance ranking or acceptance act.
    """
    input = _guard_squared_difference(schedule)
    if isinstance(input, SpecializationResult):
        return input
    identity = _result_identity(input, schedule_id, entry_point)
    if identity is not None:
        return identity
    K, N = input.point['shape'][1], input.centroids['shape'][0]
    for name, tile, extent, reason in (('output_tile', output_tile, N, 'output_extent'),
                                       ('k_tile', k_tile, K, 'tile_extent')):
        if (type(tile) is not int or tile <= 0 or tile > extent
            or (tile < extent and tile & (tile - 1))):
            return _refuse(reason, f'{name}: require the full extent {extent} or a smaller positive power of two.')
    if type(loop_unroll_factor) is not int or loop_unroll_factor < 1:
        return _refuse('unroll_extent', 'loop_unroll_factor: require a positive integer.')
    trips = (K + k_tile - 1) // k_tile
    if loop_unroll_factor > trips or trips % loop_unroll_factor:
        return _refuse('unroll_extent',
                       f'loop_unroll_factor: {loop_unroll_factor} must divide the fixed K trip count {trips}; full K requires factor 1.')
    return _build_candidate(compiler, input, output_tile=output_tile, k_tile=k_tile,
        loop_unroll_factor=loop_unroll_factor, schedule_id=schedule_id, entry_point=entry_point)


def tile_squared_difference(compiler, schedule, *, k_tile, schedule_id, entry_point):
    if type(k_tile) is not int or k_tile <= 0 or k_tile & (k_tile - 1):
        return _refuse('tile_extent', 'k_tile must be a positive power of two.')
    input = _guard_squared_difference(schedule)
    if isinstance(input, SpecializationResult):
        return input
    identity = _result_identity(input, schedule_id, entry_point)
    if identity is not None:
        return identity
    if k_tile >= input.point['shape'][1]:
        return _refuse('tile_extent', 'k_tile must be smaller than the contracted extent.')
    return _build_candidate(compiler, input, output_tile=input.centroids['shape'][0],
        k_tile=k_tile, loop_unroll_factor=1, schedule_id=schedule_id, entry_point=entry_point)


def tile_squared_difference_outputs(compiler, schedule, *, output_tile, schedule_id, entry_point):
    """Partition independent N outputs while retaining the complete K contraction."""
    if type(output_tile) is not int or output_tile <= 0 or output_tile & (output_tile - 1):
        return _refuse('output_extent', 'output_tile must be a positive power of two.')
    input = _guard_squared_difference(schedule)
    if isinstance(input, SpecializationResult):
        return input
    identity = _result_identity(input, schedule_id, entry_point)
    if identity is not None:
        return identity
    if output_tile >= input.centroids['shape'][0]:
        return _refuse('output_extent', 'output_tile must be smaller than the output-column extent.')
    return _build_candidate(compiler, input, output_tile=output_tile,
        k_tile=input.point['shape'][1], loop_unroll_factor=1,
        schedule_id=schedule_id, entry_point=entry_point)
