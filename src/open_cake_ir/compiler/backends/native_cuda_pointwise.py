"""One-warp FP32 FMA Schedule lowering for the native CUDA route.

The bounded path emits exactly the IR's three global loads, one register FMA
and one global store per row. It is also the first leaf form a worker Program
can embed without asking Triton to cross a CUDA CTA boundary.
"""
from __future__ import annotations

from .common import Emission, refusal
from .native_cuda import _Emitter, _launch_grid
from ..diagnostics import Finding
from ..ir import (
    AccessIndexKind, BoundaryPolicy, BufferMode, DType, ElementwiseOp,
    LoadMovement, LoweringBackend, MemorySpace, OperationKind, Schedule,
)
from ..target import CodeObject, Target


def preflight(schedule: Schedule, target: Target) -> tuple[Finding, ...]:
    findings: list[Finding] = []

    def check(ok, code, path, message):
        if not ok:
            findings.append(refusal(code, path, message))

    check(schedule.lowering.backend is LoweringBackend.NATIVE_CUDA
          and schedule.target == target.target_id and target.code_object is CodeObject.CUBIN,
          'NATIVE_POINTWISE_ROUTE', 'lowering',
          'pointwise FP32 FMA needs the exact native CUDA cubin route')
    check('ptx.fma.rn.f32' in target.instruction_contracts,
          'NATIVE_POINTWISE_FMA_CONTRACT', 'target',
          'Target must declare the PTX correctly rounded FP32 FMA contract')
    check(target.compute_capability is not None,
          'NATIVE_POINTWISE_ARCH', 'target', 'native CUDA needs a declared compute capability')
    check(not schedule.allocations and not schedule.pipelines and not schedule.barriers
          and not schedule.tile_loops and schedule.grid is None,
          'NATIVE_POINTWISE_RESOURCES', 'allocations',
          'this one-warp path admits no shared/tensor storage, pipeline, loop or explicit grid')
    check(len(schedule.roles) == 1 and schedule.roles[0].execution_groups == (0,)
          and schedule.roles[0].registers_per_thread is None,
          'NATIVE_POINTWISE_ROLE', 'roles', 'one execution group must own every operation')

    mapping = schedule.program_map
    check(mapping is not None and len(mapping.axes) == 1
          and mapping.axes[0].axis == mapping.axes[0].dimension == 0
          and mapping.axes[0].tile == 1,
          'NATIVE_POINTWISE_MAP', 'program_map',
          'one scalar row axis must determine the tile domain')
    check(mapping is None or not mapping.cooperative or target.cooperative_grid is True,
          'NATIVE_POINTWISE_COOPERATIVE_TARGET', 'program_map.cooperative',
          'cooperative launch needs the exact Target declaration')
    persistent = mapping is not None and mapping.persistent
    check(not persistent or (target.occupancy is not None and schedule.residency is not None
          and schedule.residency.ctas_per_multiprocessor is not None),
          'NATIVE_POINTWISE_PERSISTENCE', 'program_map.persistent',
          'persistent rows need observed SM count and declared CTA residency')
    check(schedule.residency is None or (persistent
          and schedule.residency.registers_per_thread is None),
          'NATIVE_POINTWISE_RESIDENCY', 'residency',
          'native pointwise admits only persistent CTA residency')

    globals_ = [b for b in schedule.buffers if b.space is MemorySpace.GLOBAL]
    registers = [b for b in schedule.buffers if b.space is MemorySpace.REGISTER]
    widths = {b.shape[0] for b in registers if len(b.shape) == 1}
    width = next(iter(widths), 0)
    rows = {b.shape[0] for b in globals_ if len(b.shape) == 2}
    check(len(widths) == len(rows) == 1 and 1 <= width <= target.warp_size,
          'NATIVE_POINTWISE_SHAPE', 'buffers',
          'all rank-two globals and rank-one registers must share one row and lane extent')
    for index, buffer in enumerate(schedule.buffers):
        valid = (buffer.dtype is DType.FP32 and buffer.valid_extent is None
                 and buffer.scale_of is None and buffer.stages == 1
                 and buffer.swizzle is None and buffer.byte_offset == 0
                 and ((buffer.space is MemorySpace.GLOBAL
                       and buffer.mode in (BufferMode.INPUT, BufferMode.OUTPUT)
                       and len(buffer.shape) == 2 and buffer.shape[1] == width)
                      or (buffer.space is MemorySpace.REGISTER
                          and buffer.mode is BufferMode.SCRATCH
                          and buffer.shape == (width,))))
        check(valid, 'NATIVE_POINTWISE_BUFFER', f'buffers[{index}]',
              'plain FP32 row globals and equal-width register lanes only')
    if mapping is not None and len(mapping.axes) == 1:
        owner = schedule.buffer(mapping.axes[0].buffer)
        check(owner is not None and owner.space is MemorySpace.GLOBAL
              and owner.mode is BufferMode.INPUT and len(owner.shape) == 2,
              'NATIVE_POINTWISE_AXIS_OWNER', 'program_map.axes[0]',
              'the row axis must name a rank-two global input')

    ops = schedule.operations
    body_valid = len(ops) == 5 and [op.kind for op in ops] == [
        OperationKind.LOAD, OperationKind.LOAD, OperationKind.LOAD,
        OperationKind.ELEMENTWISE, OperationKind.STORE]
    check(body_valid,
        'NATIVE_POINTWISE_BODY', 'operations',
        'one row consists of three loads, one FP32 FMA and one store')
    if not body_valid:
        return tuple(findings)
    by_name = {b.name: b for b in schedule.buffers}
    if any(op.role not in {role.name for role in schedule.roles}
           or op.signals or op.waits or op.pipeline is not None for op in ops):
        check(False, 'NATIVE_POINTWISE_OP_ROLE', 'operations',
              'all five operations must execute synchronously in the one warp')
    loaded: list[str] = []
    for index, op in enumerate(ops[:3]):
        src = by_name.get(op.reads[0]) if len(op.reads) == 1 else None
        dst = by_name.get(op.writes[0]) if len(op.writes) == 1 else None
        valid = (src is not None and dst is not None
                 and src.space is MemorySpace.GLOBAL and src.mode is BufferMode.INPUT
                 and dst.space is MemorySpace.REGISTER
                 and op.parameters.movement is LoadMovement.GLOBAL
                 and op.parameters.reuse is None)
        check(valid, 'NATIVE_POINTWISE_LOAD', f'operations[{index}]',
              'load one immutable FP32 global row without an unimplemented cache choice')
        if valid:
            loaded.append(dst.name)
    fma = ops[3]
    result = by_name.get(fma.writes[0]) if len(fma.writes) == 1 else None
    check(len(loaded) == 3 and tuple(fma.reads) == tuple(loaded)
          and result is not None and result.space is MemorySpace.REGISTER
          and fma.parameters.op is ElementwiseOp.FMA
          and fma.parameters.scalar is None
          and fma.parameters.broadcast_axis is None
          and fma.parameters.instruction is not None
          and fma.parameters.instruction.contract == 'ptx.fma.rn.f32',
          'NATIVE_POINTWISE_FMA', 'operations[3]',
          'FMA must consume the three loaded registers and name ptx.fma.rn.f32')
    store = ops[4]
    output = by_name.get(store.writes[0]) if len(store.writes) == 1 else None
    check(result is not None and tuple(store.reads) == (result.name,)
          and output is not None and output.space is MemorySpace.GLOBAL
          and output.mode is BufferMode.OUTPUT and store.parameters.coalesced
          and schedule.outputs == (output.name,),
          'NATIVE_POINTWISE_STORE', 'operations[4]',
          'store the FMA result to one fresh contiguous FP32 row')
    for index, op in (*enumerate(ops[:3]), (4, store)):
        global_name = op.reads[0] if op.kind is OperationKind.LOAD and op.reads else (
            op.writes[0] if op.writes else None)
        access = schedule.access_map(op.op_id, global_name) if global_name else None
        valid = (access is not None and access.boundary is BoundaryPolicy.MASK_TILED_AXES
                 and mapping is not None and len(mapping.axes) == 1
                 and len(access.indices) == 2
                 and access.indices[0].source is AccessIndexKind.PROGRAM
                 and access.indices[0].name == mapping.axes[0].name
                 and access.indices[1].source is AccessIndexKind.DIMENSION
                 and access.indices[1].dimension == 1
                 and access.indices[1].offset == 0
                 and access.indices[1].extent is None)
        check(valid, 'NATIVE_POINTWISE_ACCESS', f'operations[{index}]',
              'row/lane addressing must be fully masked and explicit')
    return tuple(findings)


class Emitter(_Emitter):
    def emit(self) -> Emission:
        self.line('// Generated by Open-Cake native CUDA; schedule_sha256=__SCHEDULE_SHA256__')
        self.line('#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cstdint>\n#include <new>\n#include <cstring>')
        major, minor = self.target.compute_capability
        arch = major * 100 + minor * 10
        self.line(f'#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != {arch}\n#error "Schedule requires exact {self.s.target}"\n#endif')
        params = [f'float* {self.names[b.name]}' for b in self.globals]
        self.begin(f'extern "C" __global__ void {self.entry}_kernel('+', '.join(params)+')')
        self.line('const int cake_lane = int(threadIdx.x);')
        owner = self.s.buffer(self.s.program_map.axes[0].buffer)
        rows, width = owner.shape
        if self.s.program_map.persistent:
            launched = _launch_grid(self.s, self.target)[0]
            self.line('#pragma unroll 1')
            self.begin(f'for (int cake_work=int(blockIdx.x); cake_work<{rows}; cake_work+={launched})')
        else:
            self.line('const int cake_work = int(blockIdx.x);')
        self.begin(f'if (cake_work < {rows} && cake_lane < {width})')
        for buffer in self.s.buffers:
            if buffer.space is MemorySpace.REGISTER:
                self.line(f'float {self.names[buffer.name]} = 0.0f;')
        address = f'cake_work * {width} + cake_lane'
        for op in self.s.operations:
            self.line(f'// CAKE_OP: {op.op_id}')
            if op.kind is OperationKind.LOAD:
                self.line(f'{self.names[op.writes[0]]} = {self.names[op.reads[0]]}[{address}];')
            elif op.kind is OperationKind.ELEMENTWISE:
                dst = self.names[op.writes[0]]
                a, b, c = (self.names[name] for name in op.reads)
                self.line(f'asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"({dst}) : '
                          f'"f"({a}), "f"({b}), "f"({c}));')
            else:
                self.line(f'{self.names[op.writes[0]]}[{address}] = {self.names[op.reads[0]]};')
        self.end()
        if self.s.program_map.persistent:
            self.end()
        self.end()
        self.line('// CAKE_KERNEL_END')
        self.host()
        return Emission('\n'.join(self.lines) + '\n', self.entry, {'shared_bytes': 0},
                        self.metadata('one FP32 row lane per thread; explicit PTX FMA'))
