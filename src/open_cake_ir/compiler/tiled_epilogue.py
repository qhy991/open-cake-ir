"""Explicit fusion across private rounded tiles, preserving producer loop ownership."""
from copy import deepcopy
from typing import Mapping

from .errors import CompilerError
from .ir import (AccessIndexKind, BoundaryPolicy, BufferMode, DType,
                 LoweringBackend, MemorySpace, OperationKind, ScheduleParseError)
from .passes import FusionResult, _refuse


def _geometry(schedule, access, shape):
    """Own a rank-two Cartesian output tile by its two launch coordinates."""
    mapping = schedule.program_map
    if (mapping is None or mapping.persistent or len(mapping.axes) != 2
            or access is None or len(access.indices) != 2
            or access.boundary is not BoundaryPolicy.MASK_TILED_AXES):
        return None
    result = []
    for dimension, index in enumerate(access.indices):
        if index.source is not AccessIndexKind.PROGRAM_TILE or index.offset != 0:
            return None
        axis = mapping.axis(index.name)
        if axis is None or axis.tile <= 1:
            return None
        owner = schedule.buffer(axis.buffer)
        if owner is None or owner.shape[axis.dimension] != shape[dimension]:
            return None
        result.append((axis.axis, axis.tile, shape[dimension]))
    if {item[0] for item in result} != {0, 1}:
        return None
    return tuple(result)


def fuse_tiled_epilogue(compiler, producer: Mapping, epilogue: Mapping, *,
                        seams: Mapping[str, str], schedule_id: str,
                        entry_point: str) -> FusionResult:
    """Forward private BF16/FP16 cast tiles to a matching pointwise consumer.

    Producer loops stay unchanged; every removed store and rounding cast is outside
    those loops. Consumer has no loop/reduction/state. This API proves only the
    supplied composition; Program proves public visibility and sole consumption.
    """
    documents, typed = [], []
    for label, document in [('producer', producer), ('epilogue', epilogue)]:
        try:
            copied = deepcopy(dict(document));a = compiler.assess(copied)
            if not a.lowering_eligible:
                return _refuse('input_refused', label + ': ' + ', '.join(
                    f.code for f in a.findings if f.blocks_lowering or f.blocks_acceptance))
            documents.append(copied);typed.append(a.typed_schedule)
        except (CompilerError, ScheduleParseError, TypeError, ValueError) as error:
            return _refuse('input_refused', str(error))
    p, e = typed
    if (p.target != e.target or p.lowering.backend is not LoweringBackend.TRITON
            or e.lowering.backend is not LoweringBackend.TRITON):
        return _refuse('target_route', 'Both stages require the same exact Target and Triton route.')
    if (not isinstance(schedule_id, str) or not schedule_id or schedule_id in {p.schedule_id,e.schedule_id}
            or not isinstance(entry_point, str) or not entry_point.isidentifier()):
        return _refuse('result_identity', 'Use a fresh Schedule id and valid entry point.')
    for label, s in [('producer',p),('epilogue',e)]:
        if (len(s.roles) != 1 or s.allocations or s.pipelines or s.barriers
                or any(op.waits or op.signals or op.pipeline for op in s.operations)
                or any(b.space not in {MemorySpace.GLOBAL,MemorySpace.REGISTER}
                       or b.mode is BufferMode.STATE or b.allocation is not None or b.byte_offset
                       or b.stages != 1 or b.swizzle or b.scale_of or b.valid_extent for b in s.buffers)
                or any(b.space is MemorySpace.GLOBAL and b.mode not in {BufferMode.INPUT,BufferMode.OUTPUT}
                       for b in s.buffers)):
            return _refuse('unsupported_effects', label + ': require one role with ordinary global/register values and no state or storage/synchronization commitments.')
    if (p.roles[0].execution_groups != e.roles[0].execution_groups
            or p.roles[0].registers_per_thread != e.roles[0].registers_per_thread
            or p.residency != e.residency):
        return _refuse('execution_controls', 'Role and residency commitments must be identical.')
    if e.tile_loops:
        return _refuse('epilogue_domain', 'The consumer must be a loop-free pointwise region.')
    p_kinds={OperationKind.LOAD,OperationKind.MMA,OperationKind.REDUCE,
             OperationKind.CAST,OperationKind.ELEMENTWISE,OperationKind.STORE}
    e_kinds={OperationKind.LOAD,OperationKind.CAST,OperationKind.ELEMENTWISE,OperationKind.STORE}
    if (any(op.kind not in p_kinds for op in p.operations)
            or any(op.kind not in e_kinds for op in e.operations)):
        return _refuse('operation_domain', 'Require a pure producer and pointwise consumer.')
    inputs={b.name for b in e.buffers if b.space is MemorySpace.GLOBAL and b.mode is BufferMode.INPUT}
    if (not seams or set(seams) != set(p.outputs) or set(seams.values()) != inputs
            or len(set(seams.values())) != len(seams) or len(e.outputs) != 1):
        return _refuse('composition_boundary', 'Map all sole-consumer private producer outputs to exactly the consumer inputs.')
    stores=[op for op in p.operations if op.kind is OperationKind.STORE]
    final_stores=[op for op in e.operations if op.kind is OperationKind.STORE]
    loads=[op for op in e.operations if op.kind is OperationKind.LOAD]
    if (len(stores) != len(seams) or len(final_stores) != 1 or len(loads) != len(seams)
            or tuple(p.operations[-len(stores):]) != tuple(stores) or e.operations[-1] != final_stores[0]):
        return _refuse('operation_domain', 'Producer materialization stores and consumer output store must be terminal.')
    loop_body={name for loop in p.tile_loops for name in loop.body}
    bridges, definitions = {}, {}
    geometry, shape = None, None
    for output, input_name in seams.items():
        middle=p.buffer(output);incoming=e.buffer(input_name)
        store=next((op for op in stores if op.writes==(output,)),None)
        load=next((op for op in loads if op.reads==(input_name,)),None)
        if (middle is None or incoming is None or len(middle.shape)!=2
                or middle.shape!=incoming.shape or middle.dtype!=incoming.dtype
                or store is None or load is None or len(store.reads)!=1 or len(load.writes)!=1):
            return _refuse('intermediate_abi', 'Each seam requires matching rank-two shape, dtype and one store/load edge.')
        if middle.dtype not in {DType.BF16,DType.FP16}:
            return _refuse('rounding_boundary', 'Arithmetic tiles require an explicit BF16/FP16 rounding seam; FP32 is not admitted.')
        g=_geometry(p,p.access_map(store.op_id,output),middle.shape)
        if g is None or _geometry(e,e.access_map(load.op_id,input_name),incoming.shape)!=g:
            return _refuse('tile_ownership', 'Producer and consumer must address the identical masked Cartesian tile.')
        if geometry is not None and (geometry!=g or shape!=middle.shape):
            return _refuse('tile_ownership', 'All forwarded tiles require one matching Cartesian domain.')
        geometry,shape=g,middle.shape
        bridge=p.buffer(store.reads[0]);loaded=e.buffer(load.writes[0])
        defs=[op for op in p.operations if bridge is not None and bridge.name in op.writes]
        tile=tuple(item[1] for item in g)
        if (bridge is None or loaded is None or len(defs)!=1 or defs[0].kind is not OperationKind.CAST
                or defs[0].parameters.to!=middle.dtype or bridge.dtype!=middle.dtype
                or loaded.dtype!=bridge.dtype or bridge.shape!=tile or loaded.shape!=tile
                or defs[0].op_id in loop_body or store.op_id in loop_body):
            return _refuse('rounding_boundary', 'Forward only an explicit cast tile materialized after the producer loops.')
        if any(output in op.reads for op in p.operations):
            return _refuse('intermediate_consumers', 'The producer must not reload a removed output.')
        bridges[loaded.name]=bridge.name;definitions[load.op_id]=defs[0].op_id
    final=final_stores[0];output=e.buffer(e.outputs[0])
    if (output is None or output.shape!=shape or final.writes!=(output.name,)
            or _geometry(e,e.access_map(final.op_id,output.name),shape)!=geometry):
        return _refuse('tile_ownership', 'Final output must keep the same masked domain.')
    if any(axis.buffer in seams for axis in p.program_map.axes):
        return _refuse('coordinate_owner', 'A removed intermediate cannot own producer launch coordinates.')
    out=documents[0];out['schedule_id']=schedule_id;out['lowering']={'backend':'triton','entry_point':entry_point};out['metadata']={}
    removed_stores={op.op_id for op in stores};removed_loads=set(definitions)
    out['buffers']=[b for b in out['buffers'] if b['name'] not in seams]
    out['operations']=[op for op in out['operations'] if op['id'] not in removed_stores]
    out['access_maps']=[a for a in out.get('access_maps',[]) if a['operation'] not in removed_stores]
    used={b.name for b in p.buffers}|{op.op_id for op in p.operations}|{axis.name for axis in p.program_map.axes}
    used|={loop.name for loop in p.tile_loops}|{loop.iterator for loop in p.tile_loops}|{p.roles[0].name,entry_point}
    def fresh(name):
        candidate='ep_'+name
        while candidate in used:candidate='ep_'+candidate
        used.add(candidate);return candidate
    names={b.name:fresh(b.name) for b in e.buffers if b.name not in inputs and b.name not in bridges}
    names.update(bridges)
    operations={op.op_id:fresh(op.op_id) for op in e.operations if op.op_id not in removed_loads}
    operations.update(definitions)
    axes={axis.name:next(a.name for a in p.program_map.axes if a.axis==axis.axis) for axis in e.program_map.axes}
    for b in documents[1]['buffers']:
        if b['name'] in inputs or b['name'] in bridges:continue
        b['name']=names[b['name']];out['buffers'].append(b)
    for op in documents[1]['operations']:
        if op['id'] in removed_loads:continue
        op['id']=operations[op['id']];op['role']=p.roles[0].name
        op['reads']=[names[n] for n in op['reads']];op['writes']=[names[n] for n in op['writes']]
        op['depends_on']=[operations[n] for n in op.get('depends_on',[])];out['operations'].append(op)
    for access in documents[1].get('access_maps',[]):
        if access['operation'] in removed_loads:continue
        access['operation']=operations[access['operation']];access['buffer']=names[access['buffer']]
        for index in access['indices']:
            if index['source'] in {'program','program_tile'}:index['name']=axes[index['name']]
        out['access_maps'].append(access)
    out['outputs']=[names[e.outputs[0]]]
    try:a=compiler.assess(out)
    except (CompilerError,ScheduleParseError,ValueError,TypeError) as error:return _refuse('result_refused',str(error))
    if not a.lowering_eligible:
        return _refuse('result_refused',', '.join(f.code for f in a.findings if f.blocks_lowering or f.blocks_acceptance))
    return FusionResult(a,'applied','Removed private rounded tile stores/reloads; producer loops, casts and execution controls retained. Task qualification remains required.')
