"""Bounded explicit K tiling of an FP32 squared-difference reduction.

The rewrite changes residency and reduction grouping, retaining subtraction before
square. It adds no instruction, target fact or performance qualification.
"""
from copy import deepcopy

from .passes import SpecializationResult


def tile_squared_difference(compiler, schedule, *, k_tile, schedule_id, entry_point):
    def refuse(reason, message):
        return SpecializationResult(None, reason, message)
    from .ir import Schedule
    try:
        s = Schedule.from_dict(schedule)
    except (ValueError, TypeError) as error:
        return refuse('input_refused', str(error))
    d = deepcopy(dict(schedule))
    if s.lowering.backend.value != 'triton':
        return refuse('target_route', 'This rewrite emits an explicit Triton reduction loop.')
    if (type(k_tile) is not int or k_tile <= 0 or k_tile & (k_tile-1)):
        return refuse('tile_extent', 'k_tile must be a positive power of two.')
    if (not isinstance(schedule_id, str) or not schedule_id or schedule_id == s.schedule_id
        or not isinstance(entry_point, str) or not entry_point.isidentifier()):
        return refuse('result_identity', 'Choose a fresh Schedule identity and valid entry point.')
    if (s.tile_loops or s.allocations or s.pipelines or s.barriers or s.residency
        or len(s.roles) != 1 or s.roles[0].registers_per_thread is not None
        or s.program_map is None or s.program_map.persistent
        or len(s.program_map.axes) != 1
        or any(op.waits or op.signals or op.pipeline for op in s.operations)):
        return refuse('execution_commitments', 'Require a pure loop-free row reduction with one role.')
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
    xb,cb,ob = (buffers.get(op[key][0]) for op,key in ((x,'reads'),(c,'reads'),(store,'writes')))
    if (any(b is None for b in (xb,cb,ob)) or len(xb['shape']) != 2 or len(cb['shape']) != 2
        or ob['shape'] != [xb['shape'][0],cb['shape'][0]] or xb['shape'][1] != cb['shape'][1]
        or any(b['dtype'] != 'fp32' for b in d['buffers'])
        or any(b.space.value not in {'global','register'} or b.allocation is not None or b.byte_offset
               or b.stages != 1 or b.swizzle or b.scale_of or b.valid_extent for b in s.buffers)
        or {b['name'] for b in d['buffers'] if b['space']=='global'} != {xb['name'],cb['name'],ob['name']}
        or xb['mode'] != 'input' or cb['mode'] != 'input' or ob['mode'] != 'output'):
        return refuse('storage_domain', 'Require ordinary FP32 x[R,K], c[N,K], out[R,N].')
    K,N = xb['shape'][1],cb['shape'][0]
    if k_tile >= K:
        return refuse('tile_extent', 'k_tile must be smaller than the contracted extent.')
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
    names = {b.name for b in s.buffers} | {op.op_id for op in s.operations} | {axis['name']}
    iterator = 'contracted_tile'
    while iterator in names: iterator += '_'
    d['schedule_id'] = schedule_id
    d['lowering']['entry_point'] = entry_point
    for op in (x,c,sub,square):
        buffers[op['writes'][0]]['shape'][-1] = k_tile
    for access in d['access_maps']:
        if access['operation'] in {x['id'],c['id']}:
            access['indices'][-1] = {'source':'loop_tile','name':iterator}
    fold['parameters']['across_loop'] = True
    d['tile_loops'] = [dict(name=iterator+'_loop',iterator=iterator,buffer=xb['name'],dimension=1,
        tile=k_tile,body=[op['id'] for op in ops[:-1]],range_options=dict(num_stages=1,
        loop_unroll_factor=1,disallow_acc_multi_buffer=False,flatten=False,warp_specialize=False,disable_licm=False))]
    try:
        result = compiler.assess(d)
        if not result.lowering_eligible:
            return refuse('result_refused', ', '.join(f.code for f in result.findings if f.blocks_lowering or f.blocks_acceptance))
        compiler.lower(result)
    except (ValueError, TypeError) as error:
        return refuse('result_refused', str(error))
    return SpecializationResult(result,'applied','Explicit K tiling preserves FP32 subtraction before square; measure reduction grouping, latency and resources independently.')
