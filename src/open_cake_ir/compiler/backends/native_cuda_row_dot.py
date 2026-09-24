"""One-warp BF16 row dot lowered from ordinary Cake operations.

One CTA owns one weight row. Its lanes load BF16 operands, convert to FP32,
multiply and fold the declared SUM with a fixed warp tree. This is a bounded
expert-projection leaf, not a MoE operation or a dynamic expert dispatcher.
"""
from __future__ import annotations

import re

from .common import Emission, refusal
from .native_cuda import _Emitter, _TYPES
from ..diagnostics import Finding
from ..ir import (
    AccessIndexKind, BoundaryPolicy, BufferMode, DType, ElementwiseOp,
    LoadMovement, LoweringBackend, MemorySpace, OperationKind, ReduceOp,
    ReductionScope, Schedule,
)
from ..target import CodeObject, Target


_BODY = (OperationKind.LOAD, OperationKind.LOAD, OperationKind.CAST,
         OperationKind.CAST, OperationKind.ELEMENTWISE, OperationKind.REDUCE,
         OperationKind.STORE)
_SELECTED_BODY = (OperationKind.LOAD,) + _BODY
_MIXED_SELECTED_BODY = (OperationKind.LOAD, OperationKind.LOAD,
                        OperationKind.LOAD, OperationKind.CAST,
                        OperationKind.ELEMENTWISE, OperationKind.REDUCE,
                        OperationKind.STORE)
_IDENTIFIER = re.compile(r'[A-Za-z_][A-Za-z_0-9]*\Z')


def preflight(schedule: Schedule, target: Target) -> tuple[Finding, ...]:
    findings: list[Finding] = []

    def check(ok, code, path, message):
        if not ok:
            findings.append(refusal(code, path, message))

    check(schedule.lowering.backend is LoweringBackend.NATIVE_CUDA
          and schedule.target == target.target_id and target.code_object is CodeObject.CUBIN
          and target.compute_capability is not None and target.warp_size == 32,
          'NATIVE_ROW_DOT_ROUTE', 'lowering',
          'BF16 row dot needs the exact native CUDA cubin target with a 32-lane warp')
    check(_IDENTIFIER.fullmatch(schedule.lowering.entry_point) is not None,
          'NATIVE_ROW_DOT_ENTRY', 'lowering.entry_point',
          'the generated entry point must be one ASCII C identifier')
    check(not schedule.allocations and not schedule.pipelines and not schedule.barriers
          and not schedule.tile_loops and schedule.grid is None and schedule.residency is None,
          'NATIVE_ROW_DOT_RESOURCES', 'allocations',
          'one-warp row dot has no staged resources, loop, grid or residency commitment')
    check(len(schedule.roles) == 1 and schedule.roles[0].execution_groups == (0,)
          and schedule.roles[0].registers_per_thread is None,
          'NATIVE_ROW_DOT_ROLE', 'roles', 'one execution group owns the row dot')
    ops = schedule.operations
    kinds = tuple(op.kind for op in ops)
    mixed = kinds == _MIXED_SELECTED_BODY
    selected = kinds in (_SELECTED_BODY, _MIXED_SELECTED_BODY)
    mapping = schedule.program_map
    check(mapping is not None and len(mapping.axes) == 1
          and not mapping.persistent and not mapping.cooperative
          and mapping.axes[0].axis == 0
          and mapping.axes[0].dimension == (1 if selected else 0)
          and mapping.axes[0].tile == 1,
          'NATIVE_ROW_DOT_MAP', 'program_map',
          'one nonpersistent scalar program axis owns each output row')
    if kinds not in (_BODY, _SELECTED_BODY, _MIXED_SELECTED_BODY):
        check(False, 'NATIVE_ROW_DOT_BODY', 'operations',
              'row dot needs its explicit loads, BF16 casts, multiply, SUM and store')
        return tuple(findings)
    load_expert = ops[0] if selected else None
    if mixed:
        load_x, load_w, cast_w, mul, reduce, store = ops[1:]
        cast_x = None
    else:
        load_x, load_w, cast_x, cast_w, mul, reduce, store = ops[1:] if selected else ops
    if (any(len(op.writes) != 1 for op in ops)
            or any(len(op.reads) != (2 if op is mul or selected and op is load_w else 1)
                   for op in ops)):
        check(False, 'NATIVE_ROW_DOT_BODY', 'operations',
              'loads, casts, multiply, SUM and store must have their exact operand counts')
        return tuple(findings)
    buffers = {b.name: b for b in schedule.buffers}
    names = (load_x.reads[0], load_w.reads[0], store.writes[0],
             load_x.writes[0], load_w.writes[0])
    if not mixed:
        names += (cast_x.writes[0],)
    names += (cast_w.writes[0], mul.writes[0], reduce.writes[0])
    if selected:
        names += (load_expert.reads[0], load_expert.writes[0])
    expected_buffers = (10 if mixed else 11 if selected else 9)
    if len(set(names)) != expected_buffers or set(names) != set(buffers):
        check(False, 'NATIVE_ROW_DOT_BUFFERS', 'buffers',
              'the row dot owns only its distinct inputs, output and register temporaries')
        return tuple(findings)
    x, weight, y, rx, rw = (buffers[n] for n in names[:5])
    fx = None if mixed else buffers[names[5]]
    fw, products, total = (buffers[n] for n in names[5 if mixed else 6:
                                                      8 if mixed else 9])
    expert_global, expert_index = ((buffers[names[-2]], buffers[names[-1]])
                                   if selected else (None, None))
    dependencies = (((), (), (load_expert.op_id,), (load_w.op_id,),
                     (load_x.op_id, cast_w.op_id), (mul.op_id,), (reduce.op_id,))
                    if mixed else
                    (((), (load_expert.op_id,)) if selected else ((), ()))
                    + ((load_x.op_id,), (load_w.op_id,),
                       (cast_x.op_id, cast_w.op_id), (mul.op_id,), (reduce.op_id,)))
    if selected and not mixed:
        dependencies = ((),) + dependencies
    check(tuple(op.depends_on for op in ops) == dependencies,
          'NATIVE_ROW_DOT_DEPENDENCIES', 'operations',
          'the declared graph must order each cast, multiply, sum and store')
    check(all('\n' not in op.op_id and '\r' not in op.op_id
              and '\\' not in op.op_id for op in ops),
          'NATIVE_ROW_DOT_OP_NAME', 'operations',
          'source-map operation names cannot contain newlines or backslashes')
    for buffer in schedule.buffers:
        check(buffer.allocation is None and buffer.byte_offset == 0
              and buffer.stages == 1 and buffer.swizzle is None
              and buffer.scale_of is None and buffer.valid_extent is None,
              'NATIVE_ROW_DOT_REFINEMENT', f'buffers.{buffer.name}',
              'the SIMT leaf has no staged or refined storage')
        if buffer.space is MemorySpace.GLOBAL:
            check(buffer.elements <= 2147483647,
                  'NATIVE_ROW_DOT_INDEX_RANGE', f'buffers.{buffer.name}',
                  'contiguous native addressing requires a signed-32-bit element domain')
    check((cast_w.reads, mul.reads, reduce.reads, store.reads)
          == ((rw.name,), ((rx if mixed else fx).name, fw.name),
              (products.name,), (total.name,))
          and (mixed or cast_x.reads == (rx.name,))
          and (not selected or load_w.reads == (weight.name, expert_index.name)),
          'NATIVE_ROW_DOT_DATAFLOW', 'operations',
          'every read must consume the preceding load, cast, multiply or sum result')
    weight_rank = 3 if selected else 2
    rows = weight.shape[-2] if len(weight.shape) == weight_rank else 0
    width = weight.shape[-1] if len(weight.shape) == weight_rank else 0
    check(width in (16, 32) and x.shape == (width,) and y.shape == (rows,)
          and rows > 0 and rows <= target.resource_limits.maximum_grid[0]
          and (not selected or weight.shape[0] > 0),
          'NATIVE_ROW_DOT_SHAPE', 'buffers',
          'weight [rows, width] or [experts, rows, width], x [width] and y [rows] must agree')
    check((x.space, weight.space, y.space) == (MemorySpace.GLOBAL,) * 3
          and (x.mode, weight.mode, y.mode)
          == (BufferMode.INPUT, BufferMode.INPUT, BufferMode.OUTPUT)
          and (x.dtype, weight.dtype, y.dtype)
          == ((DType.FP32 if mixed else DType.BF16), DType.BF16, DType.FP32),
          'NATIVE_ROW_DOT_GLOBALS', 'buffers',
          'BF16 weights and BF16 or FP32 input produce one FP32 output')
    if selected:
        check(expert_global.space is MemorySpace.GLOBAL
              and expert_global.mode is BufferMode.INPUT
              and expert_global.dtype is DType.INT32 and expert_global.shape == (1,),
              'NATIVE_ROW_DOT_EXPERT_INPUT', 'buffers',
              'selected expert id is one public INT32 scalar')
    registers = ((rx, DType.FP32 if mixed else DType.BF16, (width,)),
                 (rw, DType.BF16, (width,)))
    if not mixed:
        registers += ((fx, DType.FP32, (width,)),)
    registers += ((fw, DType.FP32, (width,)),
                  (products, DType.FP32, (width,)),
                  (total, DType.FP32, (1,)))
    for buffer, dtype, shape in registers:
        check(buffer.space is MemorySpace.REGISTER and buffer.mode is BufferMode.SCRATCH
              and buffer.dtype is dtype and buffer.shape == shape,
              'NATIVE_ROW_DOT_REGISTER', f'buffers.{buffer.name}',
              'each temporary has one declared lane or scalar register type')
    if selected:
        check(expert_index.space is MemorySpace.REGISTER
              and expert_index.mode is BufferMode.SCRATCH
              and expert_index.dtype is DType.INT32 and expert_index.shape == (1,),
              'NATIVE_ROW_DOT_EXPERT_REGISTER', f'buffers.{expert_index.name}',
              'the selected expert coordinate is one INT32 register')
    if mapping is not None and len(mapping.axes) == 1:
        check(mapping.axes[0].buffer == weight.name,
              'NATIVE_ROW_DOT_OWNER', 'program_map.axes[0]',
              'weight rows own the output program axis')
    check(schedule.outputs == (y.name,), 'NATIVE_ROW_DOT_OUTPUT', 'outputs',
          'the FP32 row result is the sole output')
    check(all(op.role == schedule.roles[0].name and not op.signals and not op.waits
              and op.pipeline is None for op in ops) if schedule.roles else False,
          'NATIVE_ROW_DOT_EFFECTS', 'operations',
          'all operations execute synchronously in the owning warp')
    check(all(op.parameters.movement is LoadMovement.GLOBAL
              and op.parameters.reuse is None for op in
              ((load_expert,) if selected else ()) + (load_x, load_w)),
          'NATIVE_ROW_DOT_LOAD', 'operations',
          'the two operands use ordinary global loads with no cache override')
    check((mixed or cast_x.parameters.to is DType.FP32)
          and cast_w.parameters.to is DType.FP32,
          'NATIVE_ROW_DOT_CAST', 'operations', 'each BF16 operand converts explicitly to FP32')
    check(mul.parameters.op is ElementwiseOp.MUL and mul.parameters.scalar is None
          and mul.parameters.broadcast_axis is None and mul.parameters.instruction is None,
          'NATIVE_ROW_DOT_MUL', 'operations', 'multiply matching FP32 lanes')
    check(reduce.parameters.op is ReduceOp.SUM and reduce.parameters.axis == 0
          and reduce.parameters.scope is ReductionScope.CTA
          and not reduce.parameters.across_loop,
          'NATIVE_ROW_DOT_REDUCE', 'operations',
          'fold the resident lane vector with one CTA-scoped SUM')
    check(not store.parameters.coalesced, 'NATIVE_ROW_DOT_STORE', 'operations',
          'one lane stores each completed row; coalesced=false is required')

    def mapped(operation, buffer, expected):
        access = schedule.access_map(operation.op_id, buffer.name)
        if access is None or access.boundary is not BoundaryPolicy.MASK_TILED_AXES:
            return False
        if len(access.indices) != len(expected):
            return False
        for index, (kind, dimension) in zip(access.indices, expected, strict=True):
            if kind is AccessIndexKind.PROGRAM:
                if mapping is None or index.source is not kind or index.name != mapping.axes[0].name:
                    return False
            elif kind is AccessIndexKind.SCALAR_BUFFER:
                if index.source is not kind or index.name != dimension:
                    return False
            elif (index.source is not kind or index.dimension != dimension
                  or index.offset != 0 or index.extent is not None):
                return False
        return True

    weight_indices = (((AccessIndexKind.SCALAR_BUFFER, expert_index.name),)
                      if selected else ()) + ((AccessIndexKind.PROGRAM, None),
                                              (AccessIndexKind.DIMENSION, 2 if selected else 1))
    check(len(schedule.access_maps) == (4 if selected else 3)
          and (not selected or mapped(load_expert, expert_global,
                                      ((AccessIndexKind.DIMENSION, 0),)))
          and mapped(load_x, x, ((AccessIndexKind.DIMENSION, 0),))
          and mapped(load_w, weight, weight_indices)
          and mapped(store, y, ((AccessIndexKind.PROGRAM, None),)),
          'NATIVE_ROW_DOT_ACCESS', 'access_maps',
          'input vector, weight row and scalar output need exact masked coordinates')
    return tuple(findings)


class Emitter(_Emitter):
    def emit(self) -> Emission:
        kinds = tuple(op.kind for op in self.s.operations)
        mixed = kinds == _MIXED_SELECTED_BODY
        selected = kinds in (_SELECTED_BODY, _MIXED_SELECTED_BODY)
        load_expert = self.s.operations[0] if selected else None
        if mixed:
            load_x, load_w, cast_w, mul, reduce, store = self.s.operations[1:]
        else:
            load_x, load_w, cast_x, cast_w, mul, reduce, store = (
                self.s.operations[1:] if selected else self.s.operations)
        x, weight, y = (self.b(name) for name in
                        (load_x.reads[0], load_w.reads[0], store.writes[0]))
        rows, width = weight.shape[-2:]
        self.line('// Generated by Open-Cake native CUDA; schedule_sha256=__SCHEDULE_SHA256__')
        self.line('#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>'
                  '\n#include <cstdint>\n#include <new>\n#include <cstring>')
        major, minor = self.target.compute_capability
        arch = major * 100 + minor * 10
        self.line(f'#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != {arch}\n'
                  f'#error "Schedule requires exact {self.s.target}"\n#endif')
        params = [f'{_TYPES[b.dtype]}* {self.names[b.name]}' for b in self.globals]
        self.begin(f'extern "C" __global__ void {self.entry}_kernel('+', '.join(params)+')')
        self.line('const int cake_lane = int(threadIdx.x);')
        self.line('const int cake_row = int(blockIdx.x);')
        self.begin(f'if (cake_row < {rows} && cake_lane < {width})')
        for buffer in self.s.buffers:
            if buffer.space is MemorySpace.REGISTER:
                self.line(f'{_TYPES[buffer.dtype]} {self.names[buffer.name]};')
        for op in self.s.operations:
            self.line(f'// CAKE_OP: {op.op_id}')
            if op.kind is OperationKind.LOAD:
                src = self.b(op.reads[0])
                if selected and op is load_expert:
                    address, _ = self.address(op, src, ['0'])
                    self.line(f'{self.names[op.writes[0]]} = {address};')
                elif selected and op is load_w:
                    index = self.names[load_expert.writes[0]]
                    address = (f'{self.names[weight.name]}[({index}) * '
                               f'{rows * width} + cake_row * {width} + cake_lane]')
                    self.line(f'{self.names[op.writes[0]]} = '
                              f'({index} >= 0 && {index} < {weight.shape[0]}) ? '
                              f'{address} : __float2bfloat16(0.0f);')
                else:
                    address, _ = self.address(op, src, ['cake_lane'])
                    self.line(f'{self.names[op.writes[0]]} = {address};')
            elif op.kind is OperationKind.CAST:
                self.line(f'{self.names[op.writes[0]]} = '
                          f'__bfloat162float({self.names[op.reads[0]]});')
            elif op.kind is OperationKind.ELEMENTWISE:
                self.line(f'{self.names[op.writes[0]]} = '
                          f'{self.names[op.reads[0]]} * {self.names[op.reads[1]]};')
            elif op.kind is OperationKind.REDUCE:
                dst = self.names[op.writes[0]]
                mask = '0xffffffffu' if width == 32 else '0x0000ffffu'
                self.line(f'{dst} = {self.names[op.reads[0]]};')
                self.line('#pragma unroll')
                self.begin(f'for (int delta = {width // 2}; delta > 0; delta >>= 1)')
                self.line(f'{dst} += __shfl_down_sync({mask}, {dst}, delta, {width});')
                self.end()
            else:
                address, _ = self.address(op, y, [])
                self.line(f'if (cake_lane == 0) {address} = {self.names[op.reads[0]]};')
        self.end()
        self.end()
        self.line('// CAKE_KERNEL_END')
        self.host()
        mapping = ('one FP32/BF16 row dot per CTA; masked expert select and FP32 warp-tree SUM'
                   if mixed else
                   'one BF16 row dot per CTA; masked expert select and FP32 warp-tree SUM'
                   if selected else 'one BF16 row dot per CTA; FP32 warp-tree SUM')
        return Emission('\n'.join(self.lines) + '\n', self.entry, {'shared_bytes': 0},
                        self.metadata(mapping))
