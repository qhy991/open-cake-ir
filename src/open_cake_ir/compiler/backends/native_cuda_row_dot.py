"""One-warp BF16 row dot lowered from ordinary Cake operations.

One CTA owns one weight row. Its lanes load BF16 operands, convert to FP32,
multiply and fold the declared SUM with a fixed warp tree. This is a bounded
expert-projection leaf, not a MoE operation or a dynamic expert dispatcher.
"""
from __future__ import annotations

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
    check(not schedule.allocations and not schedule.pipelines and not schedule.barriers
          and not schedule.tile_loops and schedule.grid is None and schedule.residency is None,
          'NATIVE_ROW_DOT_RESOURCES', 'allocations',
          'one-warp row dot has no staged resources, loop, grid or residency commitment')
    check(len(schedule.roles) == 1 and schedule.roles[0].execution_groups == (0,)
          and schedule.roles[0].registers_per_thread is None,
          'NATIVE_ROW_DOT_ROLE', 'roles', 'one execution group owns the row dot')
    mapping = schedule.program_map
    check(mapping is not None and len(mapping.axes) == 1
          and not mapping.persistent and not mapping.cooperative
          and mapping.axes[0].axis == mapping.axes[0].dimension == 0
          and mapping.axes[0].tile == 1,
          'NATIVE_ROW_DOT_MAP', 'program_map',
          'one nonpersistent scalar program axis owns each output row')
    ops = schedule.operations
    if tuple(op.kind for op in ops) != _BODY or any(
            len(op.reads) != (2 if op.kind is OperationKind.ELEMENTWISE else 1)
            or len(op.writes) != 1 for op in ops):
        check(False, 'NATIVE_ROW_DOT_BODY', 'operations',
              'row dot is two loads, two casts, multiply, SUM and store')
        return tuple(findings)
    load_x, load_w, cast_x, cast_w, mul, reduce, store = ops
    buffers = {b.name: b for b in schedule.buffers}
    names = (load_x.reads[0], load_w.reads[0], store.writes[0],
             load_x.writes[0], load_w.writes[0], cast_x.writes[0],
             cast_w.writes[0], mul.writes[0], reduce.writes[0])
    if len(set(names)) != 9 or set(names) != set(buffers):
        check(False, 'NATIVE_ROW_DOT_BUFFERS', 'buffers',
              'the row dot owns exactly three global and six distinct register buffers')
        return tuple(findings)
    x, weight, y, rx, rw, fx, fw, products, total = (buffers[n] for n in names)
    check((cast_x.reads, cast_w.reads, mul.reads, reduce.reads, store.reads)
          == ((rx.name,), (rw.name,), (fx.name, fw.name), (products.name,), (total.name,)),
          'NATIVE_ROW_DOT_DATAFLOW', 'operations',
          'every read must consume the preceding load, cast, multiply or sum result')
    rows = weight.shape[0] if len(weight.shape) == 2 else 0
    width = weight.shape[1] if len(weight.shape) == 2 else 0
    check(width in (16, 32) and x.shape == (width,) and y.shape == (rows,)
          and rows > 0 and rows <= target.resource_limits.maximum_grid[0],
          'NATIVE_ROW_DOT_SHAPE', 'buffers',
          'weight [rows, 16|32], x [width] and y [rows] must share exact extents')
    check((x.space, weight.space, y.space) == (MemorySpace.GLOBAL,) * 3
          and (x.mode, weight.mode, y.mode)
          == (BufferMode.INPUT, BufferMode.INPUT, BufferMode.OUTPUT)
          and (x.dtype, weight.dtype, y.dtype) == (DType.BF16, DType.BF16, DType.FP32),
          'NATIVE_ROW_DOT_GLOBALS', 'buffers',
          'two BF16 global inputs produce one FP32 global output')
    for buffer, dtype, shape in ((rx, DType.BF16, (width,)),
                                 (rw, DType.BF16, (width,)),
                                 (fx, DType.FP32, (width,)),
                                 (fw, DType.FP32, (width,)),
                                 (products, DType.FP32, (width,)),
                                 (total, DType.FP32, (1,))):
        check(buffer.space is MemorySpace.REGISTER and buffer.mode is BufferMode.SCRATCH
              and buffer.dtype is dtype and buffer.shape == shape,
              'NATIVE_ROW_DOT_REGISTER', f'buffers.{buffer.name}',
              'each temporary has one declared lane or scalar register type')
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
              and op.parameters.reuse is None for op in (load_x, load_w)),
          'NATIVE_ROW_DOT_LOAD', 'operations',
          'the two operands use ordinary global loads with no cache override')
    check(cast_x.parameters.to is DType.FP32 and cast_w.parameters.to is DType.FP32,
          'NATIVE_ROW_DOT_CAST', 'operations', 'both BF16 operands convert to FP32')
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
            elif (index.source is not kind or index.dimension != dimension
                  or index.offset != 0 or index.extent is not None):
                return False
        return True

    check(len(schedule.access_maps) == 3
          and mapped(load_x, x, ((AccessIndexKind.DIMENSION, 0),))
          and mapped(load_w, weight, ((AccessIndexKind.PROGRAM, None),
                                      (AccessIndexKind.DIMENSION, 1)))
          and mapped(store, y, ((AccessIndexKind.PROGRAM, None),)),
          'NATIVE_ROW_DOT_ACCESS', 'access_maps',
          'input vector, weight row and scalar output need exact masked coordinates')
    return tuple(findings)


class Emitter(_Emitter):
    def emit(self) -> Emission:
        load_x, load_w, cast_x, cast_w, mul, reduce, store = self.s.operations
        x, weight, y = (self.b(name) for name in
                        (load_x.reads[0], load_w.reads[0], store.writes[0]))
        rows, width = weight.shape
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
        return Emission('\n'.join(self.lines) + '\n', self.entry, {'shared_bytes': 0},
                        self.metadata('one BF16 row dot per CTA; FP32 warp-tree SUM'))
