"""One-warp FP32 gated activation lowered from Cake's arithmetic graph."""
from __future__ import annotations

import re

from .common import Emission, refusal
from .native_cuda import _Emitter
from ..diagnostics import Finding
from ..ir import (
    AccessIndexKind, BoundaryPolicy, BufferMode, DType, ElementwiseOp,
    LoadMovement, LoweringBackend, MemorySpace, OperationKind, Schedule,
)
from ..target import CodeObject, Target


_BODY = (OperationKind.LOAD, OperationKind.LOAD,
         OperationKind.ELEMENTWISE, OperationKind.ELEMENTWISE,
         OperationKind.ELEMENTWISE, OperationKind.ELEMENTWISE,
         OperationKind.ELEMENTWISE, OperationKind.STORE)
_MODEL_BODY = _BODY[:-1] + (OperationKind.CAST, OperationKind.STORE)
_IDENTIFIER = re.compile(r'[A-Za-z_][A-Za-z_0-9]*\Z')


def preflight(schedule: Schedule, target: Target) -> tuple[Finding, ...]:
    findings: list[Finding] = []

    def check(ok, code, path, message):
        if not ok:
            findings.append(refusal(code, path, message))

    check(schedule.lowering.backend is LoweringBackend.NATIVE_CUDA
          and schedule.target == target.target_id and target.code_object is CodeObject.CUBIN
          and target.compute_capability is not None and target.warp_size == 32,
          'NATIVE_ACTIVATION_ROUTE', 'lowering',
          'gated activation needs the exact native CUDA cubin target and 32-lane warp')
    check(_IDENTIFIER.fullmatch(schedule.lowering.entry_point) is not None,
          'NATIVE_ACTIVATION_ENTRY', 'lowering.entry_point',
          'entry point must be one ASCII C identifier')
    check(not schedule.allocations and not schedule.pipelines and not schedule.barriers
          and not schedule.tile_loops and schedule.grid is None and schedule.residency is None,
          'NATIVE_ACTIVATION_RESOURCES', 'allocations',
          'this one-warp activation has no staged resources, loop, grid or residency')
    check(len(schedule.roles) == 1 and schedule.roles[0].execution_groups == (0,)
          and schedule.roles[0].registers_per_thread is None,
          'NATIVE_ACTIVATION_ROLE', 'roles', 'one execution group owns all arithmetic')
    mapping = schedule.program_map
    check(mapping is not None and len(mapping.axes) == 1
          and not mapping.persistent and not mapping.cooperative
          and mapping.axes[0].axis == mapping.axes[0].dimension == 0
          and mapping.axes[0].tile == 1,
          'NATIVE_ACTIVATION_MAP', 'program_map',
          'one scalar program axis owns the packed up/gate vector')
    ops = schedule.operations
    if tuple(op.kind for op in ops) != _BODY or any(len(op.writes) != 1 for op in ops):
        check(False, 'NATIVE_ACTIVATION_BODY', 'operations',
              'two loads, five explicit arithmetic operations and one store are required')
        return tuple(findings)
    load_up, load_gate, neg, exponential, denominator, silu, multiply, store = ops
    if any(len(op.reads) != expected for op, expected in zip(
            ops, (1, 1, 1, 1, 1, 2, 2, 1), strict=True)):
        check(False, 'NATIVE_ACTIVATION_BODY', 'operations',
              'each operation needs its exact operand count')
        return tuple(findings)
    names = (load_up.reads[0], store.writes[0],
             load_up.writes[0], load_gate.writes[0], neg.writes[0],
             exponential.writes[0], denominator.writes[0],
             silu.writes[0], multiply.writes[0])
    buffers = {buffer.name: buffer for buffer in schedule.buffers}
    if len(set(names)) != 9 or set(names) != set(buffers):
        check(False, 'NATIVE_ACTIVATION_BUFFERS', 'buffers',
              'two globals and seven distinct register values form the complete graph')
        return tuple(findings)
    up_gate, activated, up, gate, negative, exp_value, denom, silu_value, result = (
        buffers[name] for name in names)
    check(load_gate.reads == (up_gate.name,),
          'NATIVE_ACTIVATION_INPUT', 'operations[1].reads',
          'both source subranges must come from the same packed input')
    check((neg.reads, exponential.reads, denominator.reads, silu.reads,
           multiply.reads, store.reads)
          == ((gate.name,), (negative.name,), (exp_value.name,),
              (gate.name, denom.name), (up.name, silu_value.name), (result.name,)),
          'NATIVE_ACTIVATION_DATAFLOW', 'operations',
          'the gate and up values must feed the declared sigmoid and product graph')
    dependencies = ((), (), (load_gate.op_id,), (neg.op_id,),
                    (exponential.op_id,), (load_gate.op_id, denominator.op_id),
                    (load_up.op_id, silu.op_id), (multiply.op_id,))
    check(tuple(op.depends_on for op in ops) == dependencies,
          'NATIVE_ACTIVATION_DEPENDENCIES', 'operations',
          'all arithmetic and the store require their explicit producer edges')
    check(all('\n' not in op.op_id and '\r' not in op.op_id
              and '\\' not in op.op_id for op in ops),
          'NATIVE_ACTIVATION_OP_NAME', 'operations',
          'source-map operation ids cannot contain newlines or backslashes')
    check(up_gate.space is MemorySpace.GLOBAL and up_gate.mode is BufferMode.INPUT
          and up_gate.dtype is DType.FP32 and up_gate.shape == (1, 64)
          and activated.space is MemorySpace.GLOBAL
          and activated.mode is BufferMode.OUTPUT
          and activated.dtype is DType.FP32 and activated.shape == (1, 32),
          'NATIVE_ACTIVATION_GLOBALS', 'buffers',
          'packed [1,64] FP32 input produces [1,32] FP32 activation')
    check(all(buffer.space is MemorySpace.REGISTER
              and buffer.mode is BufferMode.SCRATCH
              and buffer.dtype is DType.FP32 and buffer.shape == (32,)
              for buffer in (up, gate, negative, exp_value, denom, silu_value, result)),
          'NATIVE_ACTIVATION_REGISTERS', 'buffers',
          'every intermediate is one FP32 lane per warp thread')
    for buffer in schedule.buffers:
        check(buffer.allocation is None and buffer.byte_offset == 0
              and buffer.stages == 1 and buffer.swizzle is None
              and buffer.scale_of is None and buffer.valid_extent is None,
              'NATIVE_ACTIVATION_REFINEMENT', f'buffers.{buffer.name}',
              'the SIMT leaf has no staged or refined storage')
    check(mapping is not None and len(mapping.axes) == 1
          and mapping.axes[0].buffer == up_gate.name,
          'NATIVE_ACTIVATION_OWNER', 'program_map.axes[0]',
          'packed input owns the scalar program axis')
    check(schedule.outputs == (activated.name,),
          'NATIVE_ACTIVATION_OUTPUT', 'outputs',
          'the FP32 activation is the sole output')
    check(all(op.role == schedule.roles[0].name and not op.signals and not op.waits
              and op.pipeline is None for op in ops) if schedule.roles else False,
          'NATIVE_ACTIVATION_EFFECTS', 'operations',
          'all operations execute synchronously in the owning warp')
    check(all(op.parameters.movement is LoadMovement.GLOBAL
              and op.parameters.reuse is None for op in (load_up, load_gate)),
          'NATIVE_ACTIVATION_LOAD', 'operations',
          'ordinary global loads have no undeclared cache choice')
    arithmetic = ((neg, ElementwiseOp.MUL, -1, None),
                  (exponential, ElementwiseOp.EXP, None, None),
                  (denominator, ElementwiseOp.ADD, 1, None),
                  (silu, ElementwiseOp.DIV, None, None),
                  (multiply, ElementwiseOp.MUL, None, None))
    for op, kind, scalar, broadcast in arithmetic:
        check(op.parameters.op is kind and op.parameters.scalar == scalar
              and op.parameters.broadcast_axis is broadcast
              and op.parameters.instruction is None,
              'NATIVE_ACTIVATION_ARITHMETIC', f'operations.{op.op_id}',
              'activation arithmetic must match the declared up * gate/(1+exp(-gate)) graph')
    check(store.parameters.coalesced,
          'NATIVE_ACTIVATION_STORE', 'operations',
          'all 32 lanes write one contiguous activation vector')

    def mapped(operation, buffer, *, offset: int, extent: int | None):
        access = schedule.access_map(operation.op_id, buffer.name)
        if (access is None or access.boundary is not BoundaryPolicy.MASK_TILED_AXES
                or len(access.indices) != 2 or mapping is None):
            return False
        row, lane = access.indices
        return (row.source is AccessIndexKind.PROGRAM
                and row.name == mapping.axes[0].name
                and lane.source is AccessIndexKind.DIMENSION
                and lane.dimension == 1 and lane.offset == offset
                and lane.extent == extent)

    check(len(schedule.access_maps) == 3
          and mapped(load_up, up_gate, offset=0, extent=32)
          and mapped(load_gate, up_gate, offset=32, extent=None)
          and mapped(store, activated, offset=0, extent=None),
          'NATIVE_ACTIVATION_ACCESS', 'access_maps',
          'the two disjoint input halves and one contiguous output need exact coordinates')
    return tuple(findings)


def preflight_model(schedule: Schedule, target: Target) -> tuple[Finding, ...]:
    """Admit one packed H2048/I768 SwiGLU row with an explicit BF16 round."""
    findings: list[Finding] = []

    def check(ok, code, path, message):
        if not ok:
            findings.append(refusal(code, path, message))

    check(schedule.lowering.backend is LoweringBackend.NATIVE_CUDA
          and schedule.target == target.target_id == 'sm_103a'
          and target.code_object is CodeObject.CUBIN
          and target.compute_capability is not None and target.warp_size == 32,
          'NATIVE_MODEL_ACTIVATION_ROUTE', 'lowering',
          'model activation needs the exact admitted B300 native CUDA target')
    check(_IDENTIFIER.fullmatch(schedule.lowering.entry_point) is not None,
          'NATIVE_MODEL_ACTIVATION_ENTRY', 'lowering.entry_point',
          'entry point must be one ASCII C identifier')
    check(not schedule.allocations and not schedule.pipelines and not schedule.barriers
          and not schedule.tile_loops and schedule.grid is None and schedule.residency is None,
          'NATIVE_MODEL_ACTIVATION_RESOURCES', 'allocations',
          'warp-striped activation has no staged resources or persistent grid')
    check(len(schedule.roles) == 1 and schedule.roles[0].execution_groups == (0,)
          and schedule.roles[0].registers_per_thread is None,
          'NATIVE_MODEL_ACTIVATION_ROLE', 'roles',
          'one execution group owns every activation operation')
    mapping = schedule.program_map
    check(mapping is not None and len(mapping.axes) == 1
          and not mapping.persistent and not mapping.cooperative
          and mapping.axes[0].axis == mapping.axes[0].dimension == 0
          and mapping.axes[0].tile == 1,
          'NATIVE_MODEL_ACTIVATION_MAP', 'program_map',
          'one program instance owns one complete packed token row')
    ops = schedule.operations
    if tuple(op.kind for op in ops) != _MODEL_BODY or any(len(op.writes) != 1 for op in ops):
        check(False, 'NATIVE_MODEL_ACTIVATION_BODY', 'operations',
              'two loads, five arithmetic ops, BF16 cast and store are required')
        return tuple(findings)
    load_up, load_gate, neg, exponential, denominator, silu, multiply, cast, store = ops
    if any(len(op.reads) != expected for op, expected in zip(
            ops, (1, 1, 1, 1, 1, 2, 2, 1, 1), strict=True)):
        check(False, 'NATIVE_MODEL_ACTIVATION_BODY', 'operations',
              'every operation needs its exact operand count')
        return tuple(findings)
    names = (load_up.reads[0], store.writes[0], load_up.writes[0],
             load_gate.writes[0], neg.writes[0], exponential.writes[0],
             denominator.writes[0], silu.writes[0], multiply.writes[0],
             cast.writes[0])
    buffers = {buffer.name: buffer for buffer in schedule.buffers}
    if len(set(names)) != 10 or set(names) != set(buffers):
        check(False, 'NATIVE_MODEL_ACTIVATION_BUFFERS', 'buffers',
              'packed input, BF16 output and eight distinct intermediates are required')
        return tuple(findings)
    up_gate, activated, up, gate, negative, exp_value, denom, silu_value, result, rounded = (
        buffers[name] for name in names)
    check(load_gate.reads == (up_gate.name,),
          'NATIVE_MODEL_ACTIVATION_INPUT', 'operations[1].reads',
          'both subranges must read the same packed gate/up input')
    check((neg.reads, exponential.reads, denominator.reads, silu.reads,
           multiply.reads, cast.reads, store.reads)
          == ((gate.name,), (negative.name,), (exp_value.name,),
              (gate.name, denom.name), (up.name, silu_value.name),
              (result.name,), (rounded.name,)),
          'NATIVE_MODEL_ACTIVATION_DATAFLOW', 'operations',
          'the explicit sigmoid/product/cast graph must feed the output')
    dependencies = ((), (), (load_gate.op_id,), (neg.op_id,),
                    (exponential.op_id,), (load_gate.op_id, denominator.op_id),
                    (load_up.op_id, silu.op_id), (multiply.op_id,), (cast.op_id,))
    check(tuple(op.depends_on for op in ops) == dependencies,
          'NATIVE_MODEL_ACTIVATION_DEPENDENCIES', 'operations',
          'each arithmetic, cast and store edge needs its producer')
    check(all('\n' not in op.op_id and '\r' not in op.op_id
              and '\\' not in op.op_id for op in ops),
          'NATIVE_MODEL_ACTIVATION_OP_NAME', 'operations',
          'source-map operation ids must occupy one line')
    check(up_gate.space is MemorySpace.GLOBAL and up_gate.mode is BufferMode.INPUT
          and up_gate.dtype is DType.FP32 and up_gate.shape == (128, 1536)
          and activated.space is MemorySpace.GLOBAL
          and activated.mode is BufferMode.OUTPUT
          and activated.dtype is DType.BF16 and activated.shape == (128, 768),
          'NATIVE_MODEL_ACTIVATION_GLOBALS', 'buffers',
          'packed [128,1536] FP32 input produces [128,768] BF16 output')
    check(all(buffer.space is MemorySpace.REGISTER
              and buffer.mode is BufferMode.SCRATCH
              and buffer.dtype is DType.FP32 and buffer.shape == (768,)
              for buffer in (up, gate, negative, exp_value, denom, silu_value, result))
          and rounded.space is MemorySpace.REGISTER
          and rounded.mode is BufferMode.SCRATCH
          and rounded.dtype is DType.BF16 and rounded.shape == (768,),
          'NATIVE_MODEL_ACTIVATION_REGISTERS', 'buffers',
          'each lane owns 24 FP32 values before one explicit BF16 cast')
    for buffer in schedule.buffers:
        check(buffer.allocation is None and buffer.byte_offset == 0
              and buffer.stages == 1 and buffer.swizzle is None
              and buffer.scale_of is None and buffer.valid_extent is None,
              'NATIVE_MODEL_ACTIVATION_REFINEMENT', f'buffers.{buffer.name}',
              'model activation has no staged or refined storage')
    check(mapping is not None and len(mapping.axes) == 1
          and mapping.axes[0].buffer == up_gate.name,
          'NATIVE_MODEL_ACTIVATION_OWNER', 'program_map.axes[0]',
          'packed input owns the token axis')
    check(schedule.outputs == (activated.name,),
          'NATIVE_MODEL_ACTIVATION_OUTPUT', 'outputs',
          'the rounded BF16 activation is the only output')
    check(all(op.role == schedule.roles[0].name and not op.signals and not op.waits
              and op.pipeline is None for op in ops) if schedule.roles else False,
          'NATIVE_MODEL_ACTIVATION_EFFECTS', 'operations',
          'all operations execute synchronously in the owning warp')
    check(all(op.parameters.movement is LoadMovement.GLOBAL
              and op.parameters.reuse is None for op in (load_up, load_gate)),
          'NATIVE_MODEL_ACTIVATION_LOAD', 'operations',
          'the two global loads have no undeclared cache choice')
    arithmetic = ((neg, ElementwiseOp.MUL, -1),
                  (exponential, ElementwiseOp.EXP, None),
                  (denominator, ElementwiseOp.ADD, 1),
                  (silu, ElementwiseOp.DIV, None),
                  (multiply, ElementwiseOp.MUL, None))
    for op, kind, scalar in arithmetic:
        check(op.parameters.op is kind and op.parameters.scalar == scalar
              and op.parameters.broadcast_axis is None
              and op.parameters.instruction is None,
              'NATIVE_MODEL_ACTIVATION_ARITHMETIC', f'operations.{op.op_id}',
              'activation must be up * gate/(1+exp(-gate))')
    check(cast.parameters.to is DType.BF16,
          'NATIVE_MODEL_ACTIVATION_CAST', f'operations.{cast.op_id}',
          'the down tile requires an explicit FP32-to-BF16 round')
    check(store.parameters.coalesced,
          'NATIVE_MODEL_ACTIVATION_STORE', f'operations.{store.op_id}',
          'all 32 lanes write a contiguous BF16 segment')

    def mapped(operation, buffer, *, offset: int, extent: int | None):
        access = schedule.access_map(operation.op_id, buffer.name)
        if (access is None or access.boundary is not BoundaryPolicy.MASK_TILED_AXES
                or len(access.indices) != 2 or mapping is None):
            return False
        row, feature = access.indices
        return (row.source is AccessIndexKind.PROGRAM
                and len(mapping.axes) == 1 and row.name == mapping.axes[0].name
                and feature.source is AccessIndexKind.DIMENSION
                and feature.dimension == 1 and feature.offset == offset
                and feature.extent == extent)

    check(len(schedule.access_maps) == 3
          and mapped(load_up, up_gate, offset=0, extent=768)
          and mapped(load_gate, up_gate, offset=768, extent=None)
          and mapped(store, activated, offset=0, extent=None),
          'NATIVE_MODEL_ACTIVATION_ACCESS', 'access_maps',
          'gate/up halves and BF16 output need exact contiguous coordinates')
    return tuple(findings)


class Emitter(_Emitter):
    def emit(self) -> Emission:
        self.line('// Generated by Open-Cake native CUDA; schedule_sha256=__SCHEDULE_SHA256__')
        self.line('#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cstdint>'
                  '\n#include <new>\n#include <cstring>\n#include <cmath>')
        major, minor = self.target.compute_capability
        arch = major * 100 + minor * 10
        self.line(f'#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != {arch}\n'
                  f'#error "Schedule requires exact {self.s.target}"\n#endif')
        params = [f'float* {self.names[buffer.name]}' for buffer in self.globals]
        self.begin(f'extern "C" __global__ void {self.entry}_kernel('+', '.join(params)+')')
        self.line('const int cake_lane = int(threadIdx.x);')
        self.begin('if (blockIdx.x < 1 && cake_lane < 32)')
        for buffer in self.s.buffers:
            if buffer.space is MemorySpace.REGISTER:
                self.line(f'float {self.names[buffer.name]};')
        for op in self.s.operations:
            self.line(f'// CAKE_OP: {op.op_id}')
            if op.kind is OperationKind.LOAD:
                source = self.b(op.reads[0])
                address, _ = self.address(op, source, ['cake_lane'])
                self.line(f'{self.names[op.writes[0]]} = {address};')
            elif op.kind is OperationKind.ELEMENTWISE:
                dst = self.names[op.writes[0]]
                left = self.names[op.reads[0]]
                kind = op.parameters.op
                if kind is ElementwiseOp.EXP:
                    self.line(f'{dst} = expf({left});')
                elif op.parameters.scalar is not None:
                    symbol = '*' if kind is ElementwiseOp.MUL else '+'
                    self.line(f'{dst} = {left} {symbol} {float(op.parameters.scalar):.1f}f;')
                else:
                    symbol = '/' if kind is ElementwiseOp.DIV else '*'
                    self.line(f'{dst} = {left} {symbol} {self.names[op.reads[1]]};')
            else:
                output = self.b(op.writes[0])
                address, _ = self.address(op, output, ['cake_lane'])
                self.line(f'{address} = {self.names[op.reads[0]]};')
        self.end()
        self.end()
        self.line('// CAKE_KERNEL_END')
        self.host()
        return Emission('\n'.join(self.lines) + '\n', self.entry, {'shared_bytes': 0},
                        self.metadata('one FP32 gated activation vector per CTA; explicit expf'))


class ModelEmitter(_Emitter):
    """One warp owns a whole model-width row, 32 features at a time."""

    def emit(self) -> Emission:
        self.line('// Generated by Open-Cake native CUDA; schedule_sha256=__SCHEDULE_SHA256__')
        self.line('#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>'
                  '\n#include <cstdint>\n#include <new>\n#include <cstring>\n#include <cmath>')
        major, minor = self.target.compute_capability
        arch = major * 100 + minor * 10
        self.line(f'#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != {arch}\n'
                  f'#error "Schedule requires exact {self.s.target}"\n#endif')
        params = [f'{"float" if buffer.dtype is DType.FP32 else "__nv_bfloat16"}* '
                  f'{self.names[buffer.name]}' for buffer in self.globals]
        self.begin(f'extern "C" __global__ void {self.entry}_kernel('+', '.join(params)+')')
        self.line('const int cake_lane = int(threadIdx.x);')
        self.begin('if (blockIdx.x < 128 && cake_lane < 32)')
        self.begin('for (int cake_feature=cake_lane; cake_feature<768; cake_feature+=32)')
        for buffer in self.s.buffers:
            if buffer.space is MemorySpace.REGISTER:
                ctype = 'float' if buffer.dtype is DType.FP32 else '__nv_bfloat16'
                self.line(f'{ctype} {self.names[buffer.name]};')
        for op in self.s.operations:
            self.line(f'// CAKE_OP: {op.op_id}')
            if op.kind is OperationKind.LOAD:
                source = self.b(op.reads[0])
                address, _ = self.address(op, source, ['cake_feature'])
                self.line(f'{self.names[op.writes[0]]} = {address};')
            elif op.kind is OperationKind.ELEMENTWISE:
                dst = self.names[op.writes[0]]
                left = self.names[op.reads[0]]
                kind = op.parameters.op
                if kind is ElementwiseOp.EXP:
                    self.line(f'{dst} = expf({left});')
                elif op.parameters.scalar is not None:
                    symbol = '*' if kind is ElementwiseOp.MUL else '+'
                    self.line(f'{dst} = {left} {symbol} {float(op.parameters.scalar):.1f}f;')
                else:
                    symbol = '/' if kind is ElementwiseOp.DIV else '*'
                    self.line(f'{dst} = {left} {symbol} {self.names[op.reads[1]]};')
            elif op.kind is OperationKind.CAST:
                self.line(f'{self.names[op.writes[0]]} = '
                          f'__float2bfloat16_rn({self.names[op.reads[0]]});')
            else:
                output = self.b(op.writes[0])
                address, _ = self.address(op, output, ['cake_feature'])
                self.line(f'{address} = {self.names[op.reads[0]]};')
        self.end()
        self.end()
        self.end()
        self.line('// CAKE_KERNEL_END')
        self.host()
        return Emission('\n'.join(self.lines) + '\n', self.entry, {'shared_bytes': 0},
                        self.metadata('one model-width SwiGLU row per CTA; explicit BF16 round'))
