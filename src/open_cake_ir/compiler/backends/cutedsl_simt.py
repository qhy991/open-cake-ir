"""Compositional FP32 CuTe SIMT lowering for one-warp program tiles.

Value i belongs to lane i % 32, slot i // 32. Cross-lane broadcasts iterate source
slots uniformly; reductions fold local slots then a full-warp shuffle tree. The
mapping is compiler-owned and inspectable, not a new author-visible layout language.
No tensor-core acceleration or physical register residency is implied.
"""
from __future__ import annotations

import math

from .common import Emission, EmitError, python_name_findings, refusal, vocabulary_findings
from .simt import slots, private_values_per_thread
from ..diagnostics import Finding, FindingCategory, FindingSeverity
from ..ir import (AccessIndexKind, BufferMode, DType, ElementwiseOp, LoadMovement,
                  LoweringBackend, MemorySpace, OperationKind, ReduceOp, ReductionScope, Schedule)
from ..target import Target
from ..verifier import verify

SUPPORTED_DTYPES = frozenset({DType.FP32})
SUPPORTED_OPERATION_KINDS = frozenset({OperationKind.LOAD, OperationKind.ELEMENTWISE,
                                      OperationKind.REDUCE, OperationKind.STORE})
_BINARY = {ElementwiseOp.ADD: '+', ElementwiseOp.SUB: '-', ElementwiseOp.MUL: '*', ElementwiseOp.DIV: '/'}
_UNARY = {
    ElementwiseOp.SQUARE: '({x} * {x})',
    ElementwiseOp.RELU: 'cute.arch.fmax({x}, cutlass.Float32(0.0))',
    ElementwiseOp.RSQRT: 'cute.math.rsqrt({x}, fastmath=False)',
    ElementwiseOp.EXP: 'cute.math.exp({x}, fastmath=False)',
    ElementwiseOp.EXP2: 'cute.math.exp2({x}, fastmath=False)',
    ElementwiseOp.RECIPROCAL: '(cutlass.Float32(1.0) / {x})',
    ElementwiseOp.TANH: 'cute.math.tanh({x}, fastmath=False)',
}


def applies(schedule: Schedule) -> bool:
    return not any(op.kind is OperationKind.MMA for op in schedule.operations)


def requirements(schedule: Schedule) -> tuple[Finding, ...]:
    return vocabulary_findings(schedule, SUPPORTED_DTYPES, SUPPORTED_OPERATION_KINDS) + python_name_findings(schedule, register_route=True)


def preflight(schedule: Schedule, target: Target) -> tuple[Finding, ...]:
    findings = list(requirements(schedule))
    if findings:
        return tuple(findings)
    def check(condition, code, path, message):
        if not condition:
            findings.append(refusal(code, path, message))
    check(schedule.lowering.backend is LoweringBackend.CUTLASS_CUTE_DSL
          and schedule.target == target.target_id
          and target.target_id in {'sm_100a','sm_103a'}
          and target.compute_capability == {'sm_100a': (10,0), 'sm_103a': (10,3)}.get(target.target_id),
          'CUTE_SIMT_TARGET', 'target', 'CuTe SIMT requires the exact SM100a or SM103a Target.')
    check(len(schedule.roles)==1 and schedule.roles[0].warps==(0,), 'CUTE_SIMT_ROLE','roles',
          'CuTe SIMT currently assigns one role to one complete warp, warps=[0].')
    check(schedule.residency is None and all(r.registers_per_thread is None for r in schedule.roles),
          'CUTE_SIMT_RESIDENCY','residency','CuTe SIMT does not implement residency/register caps.')
    for field in ('allocations','pipelines','barriers','tile_loops'):
        check(not getattr(schedule,field),'CUTE_SIMT_DECLARATION',field,
              f'CuTe SIMT does not implement {field}; declared controls cannot be ignored.')
    pm=schedule.program_map
    check(pm is not None and not pm.persistent and all(a.tile==1 for a in pm.axes),
          'CUTE_SIMT_PROGRAM_MAP','program_map','CuTe SIMT uses scalar nonpersistent program indices.')
    pressure=private_values_per_thread(schedule)
    check(pressure<=1024,'CUTE_SIMT_PRIVATE_STORAGE','buffers',
          f'CuTe SIMT supports at most 1024 live declared-value slots per lane, got {pressure}; this is an implementation limit, not physical register capacity.')
    for index,b in enumerate(schedule.buffers):
        path=f'buffers[{index}]'
        check(b.space in {MemorySpace.GLOBAL,MemorySpace.REGISTER}
              and b.allocation is None and b.byte_offset==0 and b.stages==1
              and b.swizzle is None and b.scale_of is None and b.valid_extent is None,
              'CUTE_SIMT_STORAGE',path,'CuTe SIMT accepts ordinary global/private buffers without alias/storage/extent declarations.')
        check(b.mode in ({BufferMode.INPUT,BufferMode.OUTPUT} if b.space is MemorySpace.GLOBAL else {BufferMode.SCRATCH}),
              'CUTE_SIMT_BUFFER_MODE',path+'.mode','CuTe SIMT requires global input/output and private scratch values.')
        check(b.elements<=2**31-1,'CUTE_SIMT_INDEX_RANGE',path+'.shape','CuTe SIMT addressing must fit signed32 element indices.')
    for index,a in enumerate(schedule.access_maps):
        check(all(p.source in {AccessIndexKind.PROGRAM,AccessIndexKind.DIMENSION} for p in a.indices),
              'CUTE_SIMT_ACCESS',f'access_maps[{index}]','CuTe SIMT accepts scalar program coordinates and contiguous dimension slices.')
    for index,op in enumerate(schedule.operations):
        path=f'operations[{index}]'
        check(not op.waits and not op.signals and op.pipeline is None,'CUTE_SIMT_SYNC',path,
              'CuTe SIMT operations execute in program order without asynchronous effects.')
        params=op.parameters
        if op.kind in {OperationKind.ELEMENTWISE,OperationKind.REDUCE}:
            check(len(op.writes)==1 and all(
                schedule.buffer(name) is not None and schedule.buffer(name).space is MemorySpace.REGISTER
                for name in (*op.reads,*op.writes)),
                'CUTE_SIMT_ARITHMETIC_STORAGE',path,'CuTe SIMT arithmetic reads private values and writes one private result.')
        if op.kind is OperationKind.LOAD:
            check(params.movement is LoadMovement.GLOBAL and params.reuse is None,
                  'CUTE_SIMT_LOAD',path,'CuTe SIMT supports direct global loads without cache/reuse declarations.')
            check(len(op.reads)==len(op.writes)==1
                  and getattr(schedule.buffer(op.reads[0]),'space',None) is MemorySpace.GLOBAL
                  and getattr(schedule.buffer(op.reads[0]),'mode',None) is BufferMode.INPUT
                  and getattr(schedule.buffer(op.writes[0]),'space',None) is MemorySpace.REGISTER,
                  'CUTE_SIMT_LOAD_STORAGE',path,'CuTe SIMT loads one global input into one private value.')
        elif op.kind is OperationKind.STORE:
            check(not params.coalesced,'CUTE_SIMT_STORE',path+'.parameters.coalesced',
                  'CuTe SIMT scalar stores require coalesced=false; instruction width is not promised.')
            check(len(op.reads)==len(op.writes)==1
                  and getattr(schedule.buffer(op.reads[0]),'space',None) is MemorySpace.REGISTER
                  and getattr(schedule.buffer(op.writes[0]),'space',None) is MemorySpace.GLOBAL,
                  'CUTE_SIMT_STORE_STORAGE',path,'CuTe SIMT stores one private value into one global output.')
        elif op.kind is OperationKind.ELEMENTWISE:
            check(params.op in _BINARY or params.op in _UNARY,'CUTE_SIMT_ELEMENTWISE',path,
                  f'CuTe SIMT does not implement {params.op.value}.')
            check((params.op is ElementwiseOp.TANH and params.instruction is not None
                   and params.instruction.contract=='libdevice.tanh.f32')
                  or (params.op is not ElementwiseOp.TANH and params.instruction is None),
                  'CUTE_SIMT_INSTRUCTION',path+'.parameters.instruction',
                  'CuTe SIMT names accurate CUDA math; only the libdevice.tanh.f32 instruction contract is admitted.')
            check(params.scalar is None or math.isfinite(params.scalar) and abs(params.scalar)<=3.4028234663852886e38,
                  'CUTE_SIMT_SCALAR',path+'.parameters.scalar','Scalar must be finite FP32.')
        elif op.kind is OperationKind.REDUCE:
            check(params.op in {ReduceOp.SUM,ReduceOp.MAX} and params.scope is ReductionScope.CTA and not params.across_loop,
                  'CUTE_SIMT_REDUCTION',path+'.parameters','CuTe SIMT folds one private tile by sum/max in its single-warp CTA.')
            check(params.op is not ReduceOp.MAX or (
                  schedule.lowering.entry_point != 'float' and not any(
                      b.name == 'float' and b.space is MemorySpace.GLOBAL for b in schedule.buffers)),
                  'CUTE_SIMT_NAME',path,'Max reduction requires the unshadowed float builtin for its negative-infinity identity.')
        if op.kind in {OperationKind.LOAD,OperationKind.STORE} and len(op.reads)==len(op.writes)==1:
            glob,private=((op.reads[0],op.writes[0]) if op.kind is OperationKind.LOAD else (op.writes[0],op.reads[0]))
            maps=[a for a in schedule.access_maps if (a.operation,a.buffer)==(op.op_id,glob)]
            check(len(maps)==1,'CUTE_SIMT_ACCESS_MAP',path,'Global memory operations require one exact access map.')
            if len(maps)!=1:continue
            a=maps[0];g=schedule.buffer(glob);v=schedule.buffer(private)
            if g is None or v is None:continue
            if all(c.source is AccessIndexKind.PROGRAM or c.source is AccessIndexKind.DIMENSION
                   and c.dimension is not None and c.dimension<len(g.shape) for c in a.indices):
                shape=tuple(c.span(g.shape[c.dimension]) for c in a.indices if c.source is AccessIndexKind.DIMENSION) or (1,)
                check(v.shape==shape,'CUTE_SIMT_VALUE_SHAPE',path,
                      'Access value domain and private tile shape must match without reshape/truncation.')
            if op.kind is OperationKind.STORE and pm is not None:
                owned={c.name for c in a.indices if c.source is AccessIndexKind.PROGRAM}
                missing=[axis.name for axis in pm.axes if (owner:=schedule.buffer(axis.buffer)) is not None
                         and axis.dimension<len(owner.shape) and axis.tile_count(owner.shape[axis.dimension])>1 and axis.name not in owned]
                check(not missing,'CUTE_SIMT_STORE_OWNERSHIP',path,f'Store omits varying program axes {missing}.')
    if not findings:
        findings.append(Finding('CUTE_SIMT_EXECUTION','lowering',
            f'FP32 SIMT stripe: value i belongs to lane i%32, slot i//32; {pressure} peak live declared slots per lane. Uniform warp shuffle/reduction, scalar global copies, no Tensor Core claim; allocation/spills and latency need compiled/device evidence.',
            FindingCategory.HARDWARE_CONFORMANCE,FindingSeverity.REPORT))
    return tuple(findings)


def _coordinates(flat,shape):
    return [f'(({flat} // {math.prod(shape[i+1:])}) % {extent})' for i,extent in enumerate(shape)]


def _flat(coords,shape):
    return ' + '.join(f'({x}) * {math.prod(shape[i+1:])}' for i,x in enumerate(coords)) or '0'


def emit(schedule: Schedule, target: Target, *, entry_point: str | None=None) -> Emission:
    failures=[f for f in (*verify(schedule,target),*preflight(schedule,target)) if f.blocks_lowering or f.blocks_acceptance]
    if failures:raise EmitError(f'{failures[0].path}: {failures[0].message}')
    if entry_point is not None and entry_point!=schedule.lowering.entry_point:
        raise EmitError('Entry point differs from the canonical route.')
    entry=schedule.lowering.entry_point
    buffers={b.name:b for b in schedule.buffers}
    globals_=tuple(b for mode in (BufferMode.INPUT,BufferMode.OUTPUT) for b in schedule.buffers if b.space is MemorySpace.GLOBAL and b.mode is mode)
    prefix='_cake_'
    suffix=0
    while any(b.name.startswith(prefix) for b in globals_) or entry.startswith(prefix):
        suffix+=1
        prefix=f'_cake{suffix}_'
    def v(name):return prefix+name
    names={b.name:(b.name if b.space is MemorySpace.GLOBAL else v(f'v{i}')) for i,b in enumerate(schedule.buffers)}
    axes={a.name:a for a in schedule.program_map.axes}
    grid=[1,1,1]
    for a in axes.values():grid[a.axis]=buffers[a.buffer].shape[a.dimension]
    lines=[f'# Generated by open-cake-ir from {schedule.schedule_id!r}; DO NOT EDIT.',
           '# schedule_sha256=__SCHEDULE_SHA256__','import cutlass','import cutlass.cute as cute',
           'from cutlass.cute.nvgpu import warp','','@cute.kernel',
           f'def {entry}('+', '.join(b.name+': cute.Pointer' for b in globals_)+'):']
    def line(text,level=1):lines.append('    '*level+text)
    lane=v('lane');slot=v('slot');idx=v('index')
    line(f'{lane}, {v("ty")}, {v("tz")} = cute.arch.thread_idx()')
    programs=[v('px'),v('py'),v('pz')]
    line(', '.join(programs)+' = cute.arch.block_idx()')
    for b in schedule.buffers:
        if b.space is MemorySpace.REGISTER:
            line(f'{names[b.name]} = cute.make_rmem_tensor(({slots(b)},), cutlass.Float32)')
            line(f'{names[b.name]}.fill(0.0)')
    def address(op,g,b):
        a=schedule.access_map(op.op_id,g.name);coords=iter(_coordinates(idx,b.shape));out=[]
        for c in a.indices:
            out.append(programs[axes[c.name].axis] if c.source is AccessIndexKind.PROGRAM else f'({next(coords)} + {c.offset})')
        return _flat(out,g.shape)
    for oi,op in enumerate(schedule.operations):
        line(f'# CAKE_OP:{op.op_id}')
        src,dst=buffers[op.reads[0]],buffers[op.writes[0]]
        if op.kind is OperationKind.REDUCE:
            extent=src.shape[op.parameters.axis];inner=math.prod(src.shape[op.parameters.axis+1:])
            identity='0.0' if op.parameters.op is ReduceOp.SUM else "float('-inf')"
            # Loops are uniform across the warp, including lanes with padded values.
            for output in range(dst.elements):
                partial=v(f'p{oi}_{output}');begin=(output//inner)*extent*inner;end=begin+extent*inner
                line(f'{partial} = cutlass.Float32({identity})')
                for s in range(begin//32,(end+31)//32):
                    line(f'{idx} = {s*32} + {lane}')
                    line(f'if ({idx} >= {begin}) and ({idx} < {end}) and ({idx} % {inner} == {output%inner}):')
                    val=f'{names[src.name]}[{s}]'
                    expr=f'{partial} + {val}' if op.parameters.op is ReduceOp.SUM else f'cute.arch.fmax({partial}, {val})'
                    line(f'{partial} = {expr}',2)
                intrinsic='warp_reduction_sum' if op.parameters.op is ReduceOp.SUM else 'warp_reduction_max'
                line(f'{partial} = cute.arch.{intrinsic}({partial})')
                line(f'if {lane} == {output%32}:')
                line(f'{names[dst.name]}[{output//32}] = {partial}',2)
            continue
        count=src.elements if op.kind is OperationKind.STORE else dst.elements
        line(f'for {slot} in cutlass.range_constexpr({(count+31)//32}):')
        line(f'{idx} = {slot} * 32 + {lane}',2)
        if op.kind is OperationKind.LOAD:
            line(f'if {idx} < {count}:',2)
            ptr=v('ptr');line(f'{ptr} = cute.make_tensor({names[src.name]} + {address(op,src,dst)}, cute.make_layout((1,)))',3)
            line(f'{names[dst.name]}[{slot}] = {ptr}[0]',3)
        elif op.kind is OperationKind.STORE:
            line(f'if {idx} < {count}:',2)
            ptr=v('ptr');line(f'{ptr} = cute.make_tensor({names[dst.name]} + {address(op,dst,src)}, cute.make_layout((1,)))',3)
            line(f'{ptr}[0] = {names[src.name]}[{slot}]',3)
        else:
            operands=[]
            for ri,read in enumerate(op.reads):
                b=buffers[read]
                if b.shape==dst.shape:operands.append(f'{names[read]}[{slot}]')
                elif b.is_scalar:
                    name=v(f'scalar{oi}_{ri}');line(f'{name} = cute.arch.shuffle_sync({names[read]}[0], 0)',2);operands.append(name)
                else:
                    coords=_coordinates(idx,dst.shape);start=op.parameters.broadcast_axis
                    sub=_flat(coords[start:start+len(b.shape)],b.shape)
                    wanted=v(f'wanted{oi}_{ri}');value=v(f'operand{oi}_{ri}')
                    line(f'{wanted} = {sub}',2);line(f'{value} = cutlass.Float32(0.0)',2)
                    for s in range(slots(b)):
                        exchanged=v(f'exchange{oi}_{ri}_{s}')
                        line(f'{exchanged} = cute.arch.shuffle_sync({names[read]}[{s}], {wanted} % 32)',2)
                        line(f'if {wanted} // 32 == {s}:',2);line(f'{value} = {exchanged}',3)
                    operands.append(value)
            if op.parameters.scalar is not None:operands.append(f'cutlass.Float32({float(op.parameters.scalar)!r})')
            expr=f'({operands[0]} {_BINARY[op.parameters.op]} {operands[1]})' if op.parameters.op in _BINARY else _UNARY[op.parameters.op].format(x=operands[0])
            line(f'if {idx} < {count}:',2);line(f'{names[dst.name]}[{slot}] = {expr}',3)
    lines.append('# CAKE_KERNEL_END')
    return Emission('\n'.join(lines)+'\n',entry,{}, {'compiler':'cutlass_cute_dsl','source_language':'python',
        'target':target.target_id,'kernel_entry_point':entry,
        'signature':[{'name':b.name,'dtype':b.dtype.value} for b in globals_],
        'grid':grid,'block':[32,1,1],'dynamic_shared_memory_bytes':0})
