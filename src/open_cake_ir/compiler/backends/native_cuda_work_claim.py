"""Native CUDA one-warp PTX atomic work-claim lowering.

Imported lazily by native_cuda after its tensor-core emitter is defined. This route
shares the exact native host ABI and grid derivation, while retaining its own narrow
admission and device instruction.
"""
from __future__ import annotations

import math

from .common import Emission, refusal
from .native_cuda import _Emitter, _launch_grid
from ..diagnostics import Finding
from ..ir import (
    AccessIndexKind, AtomicMemoryOrder, AtomicMemoryScope, AtomicOp,
    BoundaryPolicy, BufferMode, DType, LoadMovement, LoweringBackend,
    MemorySpace, OperationKind, Schedule,
)
from ..target import CodeObject, Target

_ATOMIC_CONTRACT = 'ptx.atom.relaxed.gpu.global.add.s32'


def preflight(s: Schedule, target: Target) -> tuple[Finding, ...]:
    """Bounded one-warp INT32 queue slice; all other native forms stay refused."""
    findings: list[Finding] = []

    def check(ok, code, path, message):
        if not ok:
            findings.append(refusal(code, path, message))

    check(s.lowering.backend is LoweringBackend.NATIVE_CUDA and s.target == target.target_id
          and target.code_object is CodeObject.CUBIN,
          'NATIVE_SIMT_ROUTE', 'lowering', 'SIMT work claims require the exact native CUDA cubin route')
    check(_ATOMIC_CONTRACT in target.instruction_contracts,
          'NATIVE_ATOMIC_CONTRACT_UNSUPPORTED', 'target',
          f'Target {target.target_id!r} does not declare {_ATOMIC_CONTRACT!r}')
    check(not s.allocations and not s.barriers and not s.pipelines and not s.tile_loops,
          'NATIVE_SIMT_RESOURCES', 'allocations',
          'one-warp work claims carry no shared, tensor, barrier or tile-loop resources')
    check(len(s.roles) == 1 and s.roles[0].execution_groups == (0,)
          and s.roles[0].registers_per_thread is None,
          'NATIVE_SIMT_ROLE', 'roles', 'the work-claim slice uses exactly one warp starting at zero')
    mapping = s.program_map
    check(mapping is not None and len(mapping.axes) == 1 and mapping.axes[0].axis == 0
          and mapping.axes[0].dimension == 0 and mapping.axes[0].tile == 1,
          'NATIVE_SIMT_PROGRAM_MAP', 'program_map',
          'one scalar token axis owns each work-claim CTA')
    persistent = mapping is not None and mapping.persistent
    check(not persistent or (target.occupancy is not None and s.residency is not None
          and s.residency.ctas_per_multiprocessor is not None),
          'NATIVE_SIMT_PERSISTENCE', 'program_map.persistent',
          'persistent claims require observed SM count and declared CTA residency')
    check(s.residency is None or (persistent and s.residency.registers_per_thread is None),
          'NATIVE_SIMT_RESIDENCY', 'residency',
          'native SIMT admits CTA residency only for a persistent grid, not a register cap')
    check(s.grid is None, 'NATIVE_SIMT_GRID', 'grid',
          'native SIMT claims derive their launch from ProgramMap')
    buffers = {buffer.name: buffer for buffer in s.buffers}
    if mapping is not None and len(mapping.axes) == 1:
        owner = buffers.get(mapping.axes[0].buffer)
        check(owner is not None and owner.space is MemorySpace.GLOBAL
              and owner.mode is BufferMode.INPUT and len(owner.shape) == 2,
              'NATIVE_SIMT_AXIS_OWNER', 'program_map.axes[0]',
              'the token axis names a rank-two global input')
    widths = {buffer.shape[0] for buffer in s.buffers
              if buffer.space is MemorySpace.REGISTER and len(buffer.shape) == 1}
    check(len(widths) == 1 and 1 <= next(iter(widths), 0) <= target.warp_size,
          'NATIVE_SIMT_WIDTH', 'buffers',
          'all register values share one lane extent no wider than the Target warp')
    width = next(iter(widths), 0)
    for index, buffer in enumerate(s.buffers):
        check(buffer.dtype is DType.INT32 and buffer.valid_extent is None
              and buffer.scale_of is None and buffer.stages == 1
              and buffer.swizzle is None and buffer.byte_offset == 0
              and ((buffer.space is MemorySpace.GLOBAL and len(buffer.shape) in (1, 2))
                   or (buffer.space is MemorySpace.REGISTER and buffer.mode is BufferMode.SCRATCH
                       and buffer.shape == (width,))),
              'NATIVE_SIMT_BUFFER', f'buffers[{index}]',
              'SIMT work claims use plain INT32 global arrays and equal-width register vectors')
    check(any(op.kind is OperationKind.ATOMIC_RMW for op in s.operations),
          'NATIVE_SIMT_ATOMIC_REQUIRED', 'operations',
          'this native route is reserved for returned-old-value atomic work claims')
    roles = {role.name for role in s.roles}
    for index, op in enumerate(s.operations):
        path = f'operations[{index}]'
        if op.kind not in {OperationKind.LOAD, OperationKind.ATOMIC_RMW, OperationKind.STORE}:
            check(False, 'NATIVE_SIMT_OPERATION_UNSUPPORTED', path,
                  'SIMT work claims admit load, atomic_rmw and store only')
            continue
        check(op.role in roles and not op.signals and not op.waits and op.pipeline is None,
              'NATIVE_SIMT_OPERATION_ROLE', path,
              'every operation executes in the one warp without an asynchronous handshake')
        memory_name = (op.reads[0] if op.kind is not OperationKind.STORE and op.reads
                       else op.writes[0] if op.writes else None)
        memory = buffers.get(memory_name)
        access = s.access_map(op.op_id, memory_name) if memory_name is not None else None
        if memory is None or access is None:
            check(False, 'NATIVE_SIMT_ACCESS_REQUIRED', path,
                  'each work-claim memory operation requires one known global AccessMap')
            continue
        check(memory.space is MemorySpace.GLOBAL and len(access.indices) == len(memory.shape)
              and access.boundary is BoundaryPolicy.MASK_TILED_AXES,
              'NATIVE_SIMT_ACCESS_FORM', path,
              'global addressing must cover every axis with masked bounds')
        for position, component in enumerate(access.indices):
            valid = (component.source is AccessIndexKind.PROGRAM
                     and mapping is not None and len(mapping.axes) == 1
                     and component.name == mapping.axes[0].name)
            valid |= (component.source is AccessIndexKind.DIMENSION
                      and component.dimension == position and component.offset == 0
                      and component.extent is None)
            valid |= (component.source is AccessIndexKind.BUFFER
                      and component.name in buffers
                      and buffers[component.name].space is MemorySpace.REGISTER)
            check(valid, 'NATIVE_SIMT_ACCESS_INDEX', f'{path}.access[{position}]',
                  'an address coordinate must be the token, lane or same-lane INT32 register')
        if op.kind is OperationKind.LOAD:
            destination = buffers.get(op.writes[0]) if len(op.writes) == 1 else None
            check(memory.mode is BufferMode.INPUT and destination is not None
                  and destination.space is MemorySpace.REGISTER
                  and op.parameters.movement is LoadMovement.GLOBAL
                  and op.parameters.reuse is None,
                  'NATIVE_SIMT_LOAD', path,
                  'load reads an immutable global INT32 array without an unimplemented cache commitment')
        elif op.kind is OperationKind.ATOMIC_RMW:
            result = buffers.get(op.writes[-1]) if len(op.writes) == 2 else None
            check(memory.mode is BufferMode.STATE and len(memory.shape) == 1
                  and len(op.reads) == len(op.writes) == 2
                  and op.writes[0] == memory.name and result is not None
                  and result.space is MemorySpace.REGISTER
                  and op.parameters.op is AtomicOp.ADD
                  and op.parameters.order is AtomicMemoryOrder.RELAXED
                  and op.parameters.scope is AtomicMemoryScope.DEVICE,
                  'NATIVE_SIMT_ATOMIC', path,
                  'PTX work claim is returned-old INT32 add with relaxed GPU scope')
        else:
            source = buffers.get(op.reads[0]) if len(op.reads) == 1 else None
            check(memory.mode is BufferMode.OUTPUT and len(memory.shape) == 2
                  and memory.shape[1] == width and source is not None
                  and source.space is MemorySpace.REGISTER
                  and len(access.indices) == 2
                  and access.indices[0].source is AccessIndexKind.PROGRAM
                  and access.indices[1].source is AccessIndexKind.DIMENSION
                  and op.parameters.coalesced,
                  'NATIVE_SIMT_STORE', path,
                  'one token and lane own each contiguous INT32 output element')
    return tuple(findings)


class Emitter(_Emitter):
    """One lane owns one routed slot and its returned-old-value work claim."""

    def address(self, op, buffer):
        access = self.s.access_map(op.op_id, buffer.name)
        assert access is not None  # Native SIMT preflight owns this admission.
        coords = []
        for component in access.indices:
            if component.source is AccessIndexKind.PROGRAM:
                coords.append('cake_work')
            elif component.source is AccessIndexKind.DIMENSION:
                coords.append('cake_lane')
            else:
                coords.append(self.names[component.name])
        linear = ' + '.join(f'({coord}) * {math.prod(buffer.shape[i+1:])}'
                            for i, coord in enumerate(coords))
        mask = ' && '.join(f'({coord}) >= 0 && ({coord}) < {extent}'
                            for coord, extent in zip(coords, buffer.shape, strict=True))
        return linear, mask

    def emit(self):
        self.line('// Generated by Open-Cake native CUDA; schedule_sha256=__SCHEDULE_SHA256__')
        self.line('#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cstdint>\n#include <new>\n#include <cstring>')
        major, minor = self.target.compute_capability
        arch = major*100 + minor*10
        self.line(f'#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != {arch}\n#error "Schedule requires exact {self.s.target}"\n#endif')
        params = [f'int32_t* {self.names[b.name]}' for b in self.globals]
        self.begin(f'extern "C" __global__ void {self.entry}_kernel('+', '.join(params)+')')
        self.line('const int cake_lane = int(threadIdx.x);')
        axis = self.s.program_map.axes[0]
        total = axis.tile_count(self.b(axis.buffer).shape[axis.dimension])
        if self.s.program_map.persistent:
            launched = _launch_grid(self.s, self.target)[0]
            self.line('#pragma unroll 1')
            self.begin(f'for (int cake_work=int(blockIdx.x); cake_work<{total}; cake_work+={launched})')
        else:
            self.line('const int cake_work = int(blockIdx.x);')
        width = next(buffer.shape[0] for buffer in self.s.buffers
                     if buffer.space is MemorySpace.REGISTER)
        self.begin(f'if (cake_lane < {width})')
        for buffer in self.s.buffers:
            if buffer.space is MemorySpace.REGISTER:
                self.line(f'int32_t {self.names[buffer.name]} = 0;')
        for op in self.s.operations:
            self.line(f'// CAKE_OP: {op.op_id}')
            if op.kind is OperationKind.LOAD:
                src = self.b(op.reads[0])
                dst = self.names[op.writes[0]]
                address, mask = self.address(op, src)
                self.line(f'{dst} = ({mask}) ? {self.names[src.name]}[{address}] : 0;')
            elif op.kind is OperationKind.ATOMIC_RMW:
                target = self.b(op.reads[0])
                old = self.names[op.writes[-1]]
                address, mask = self.address(op, target)
                self.begin(f'if ({mask})')
                self.line(f'asm volatile("atom.relaxed.gpu.global.add.s32 %0, [%1], %2;"'
                          f' : "=r"({old})'
                          f' : "l"(reinterpret_cast<unsigned long long>(&{self.names[target.name]}[{address}])), '
                          f'"r"(int32_t({op.parameters.value})) : "memory");')
                self.end()
            else:
                dst = self.b(op.writes[0])
                src = self.names[op.reads[0]]
                address, mask = self.address(op, dst)
                self.line(f'if ({mask}) {self.names[dst.name]}[{address}] = {src};')
        self.end()
        if self.s.program_map.persistent:
            self.end()
        self.end()
        self.line('// CAKE_KERNEL_END')
        self.host()
        return Emission('\n'.join(self.lines)+'\n', self.entry,
                        {'shared_bytes':0},
                        self.metadata('one lane per INT32 routed slot and work claim'))
