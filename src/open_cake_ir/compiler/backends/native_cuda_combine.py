"""One-warp weighted route combine emitted from ordinary Cake operations."""
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


_BODY = (OperationKind.LOAD, OperationKind.LOAD, OperationKind.ELEMENTWISE,
         OperationKind.REDUCE, OperationKind.CAST, OperationKind.STORE)
_IDENTIFIER = re.compile(r'[A-Za-z_][A-Za-z_0-9]*\Z')


def preflight(schedule: Schedule, target: Target) -> tuple[Finding, ...]:
    findings: list[Finding] = []

    def check(ok, code, path, message):
        if not ok:
            findings.append(refusal(code, path, message))

    check(schedule.lowering.backend is LoweringBackend.NATIVE_CUDA
          and schedule.target == target.target_id and target.code_object is CodeObject.CUBIN
          and target.compute_capability is not None and target.warp_size == 32,
          'NATIVE_COMBINE_ROUTE', 'lowering',
          'weighted combine needs the exact native CUDA cubin target and 32-lane warp')
    check(_IDENTIFIER.fullmatch(schedule.lowering.entry_point) is not None,
          'NATIVE_COMBINE_ENTRY', 'lowering.entry_point',
          'entry point must be one ASCII C identifier')
    check(not schedule.allocations and not schedule.pipelines and not schedule.barriers
          and not schedule.tile_loops and schedule.grid is None and schedule.residency is None,
          'NATIVE_COMBINE_RESOURCES', 'allocations',
          'one-warp combine has no staged resources, loop, grid or residency')
    check(len(schedule.roles) == 1 and schedule.roles[0].execution_groups == (0,)
          and schedule.roles[0].registers_per_thread is None,
          'NATIVE_COMBINE_ROLE', 'roles', 'one execution group owns all operations')
    mapping = schedule.program_map
    check(mapping is not None and len(mapping.axes) == 1
          and not mapping.persistent and not mapping.cooperative
          and mapping.axes[0].axis == mapping.axes[0].dimension == 0
          and mapping.axes[0].tile == 1,
          'NATIVE_COMBINE_MAP', 'program_map',
          'one scalar program axis owns each source token')
    ops = schedule.operations
    if (tuple(op.kind for op in ops) != _BODY
            or any(len(op.reads) != (2 if op.kind is OperationKind.ELEMENTWISE else 1)
                   or len(op.writes) != 1 for op in ops)):
        check(False, 'NATIVE_COMBINE_BODY', 'operations',
              'two loads, multiply, SUM, BF16 cast and store are required')
        return tuple(findings)
    load_values, load_weights, multiply, reduce, cast, store = ops
    buffers = {buffer.name: buffer for buffer in schedule.buffers}
    names = (load_values.reads[0], load_weights.reads[0], store.writes[0],
             load_values.writes[0], load_weights.writes[0], multiply.writes[0],
             reduce.writes[0], cast.writes[0])
    if len(set(names)) != 8 or set(names) != set(buffers):
        check(False, 'NATIVE_COMBINE_BUFFERS', 'buffers',
              'three globals and five distinct register values form the full combine')
        return tuple(findings)
    contributions, weights, output, route_values, route_weights, weighted, summed, rounded = (
        buffers[name] for name in names)
    tokens = contributions.shape[0] if len(contributions.shape) == 3 else 0
    check(tokens > 0 and tokens <= target.resource_limits.maximum_grid[0]
          and contributions.shape == (tokens, 2, 16)
          and weights.shape == (tokens, 2)
          and output.shape == (tokens, 16),
          'NATIVE_COMBINE_SHAPE', 'buffers',
          'one token carries two FP32 routes of width 16 and one BF16 output')
    check((contributions.space, weights.space, output.space)
          == (MemorySpace.GLOBAL,) * 3
          and (contributions.mode, weights.mode, output.mode)
          == (BufferMode.INPUT, BufferMode.INPUT, BufferMode.OUTPUT)
          and (contributions.dtype, weights.dtype, output.dtype)
          == (DType.FP32, DType.FP32, DType.BF16),
          'NATIVE_COMBINE_GLOBALS', 'buffers',
          'FP32 route contributions and weights produce BF16 output')
    for buffer, dtype, shape in ((route_values, DType.FP32, (2, 16)),
                                 (route_weights, DType.FP32, (2,)),
                                 (weighted, DType.FP32, (2, 16)),
                                 (summed, DType.FP32, (16,)),
                                 (rounded, DType.BF16, (16,))):
        check(buffer.space is MemorySpace.REGISTER and buffer.mode is BufferMode.SCRATCH
              and buffer.dtype is dtype and buffer.shape == shape,
              'NATIVE_COMBINE_REGISTERS', f'buffers.{buffer.name}',
              'each route or output temporary has exact register extent and dtype')
    for buffer in schedule.buffers:
        check(buffer.allocation is None and buffer.byte_offset == 0
              and buffer.stages == 1 and buffer.swizzle is None
              and buffer.scale_of is None and buffer.valid_extent is None,
              'NATIVE_COMBINE_REFINEMENT', f'buffers.{buffer.name}',
              'the SIMT combine has no staged or refined storage')
        if buffer.space is MemorySpace.GLOBAL:
            check(buffer.elements <= 2147483647,
                  'NATIVE_COMBINE_INDEX_RANGE', f'buffers.{buffer.name}',
                  'contiguous native addressing requires signed-32-bit elements')
    check((multiply.reads, reduce.reads, cast.reads, store.reads)
          == ((route_values.name, route_weights.name), (weighted.name,),
              (summed.name,), (rounded.name,)),
          'NATIVE_COMBINE_DATAFLOW', 'operations',
          'weighted routes feed one sum, one BF16 rounding and one store')
    dependencies = ((), (), (load_values.op_id, load_weights.op_id),
                    (multiply.op_id,), (reduce.op_id,), (cast.op_id,))
    check(tuple(op.depends_on for op in ops) == dependencies,
          'NATIVE_COMBINE_DEPENDENCIES', 'operations',
          'producer edges must be explicit in the combine graph')
    check(all('\n' not in op.op_id and '\r' not in op.op_id
              and '\\' not in op.op_id for op in ops),
          'NATIVE_COMBINE_OP_NAME', 'operations',
          'source-map operation ids cannot contain newlines or backslashes')
    check(mapping is not None and len(mapping.axes) == 1
          and mapping.axes[0].buffer == contributions.name,
          'NATIVE_COMBINE_OWNER', 'program_map.axes[0]',
          'contribution token axis owns the output program coordinate')
    check(schedule.outputs == (output.name,),
          'NATIVE_COMBINE_OUTPUT', 'outputs', 'BF16 token tensor is sole output')
    check(all(op.role == schedule.roles[0].name and not op.signals and not op.waits
              and op.pipeline is None for op in ops) if schedule.roles else False,
          'NATIVE_COMBINE_EFFECTS', 'operations',
          'all operations execute synchronously in the owning warp')
    check(all(op.parameters.movement is LoadMovement.GLOBAL
              and op.parameters.reuse is None for op in (load_values, load_weights)),
          'NATIVE_COMBINE_LOAD', 'operations', 'global loads have no cache override')
    check(multiply.parameters.op is ElementwiseOp.MUL
          and multiply.parameters.scalar is None
          and multiply.parameters.broadcast_axis == 0
          and multiply.parameters.instruction is None,
          'NATIVE_COMBINE_MULTIPLY', 'operations',
          'per-route scalar weight broadcasts along route axis only')
    check(reduce.parameters.op is ReduceOp.SUM and reduce.parameters.axis == 0
          and reduce.parameters.scope is ReductionScope.CTA
          and not reduce.parameters.across_loop,
          'NATIVE_COMBINE_REDUCE', 'operations',
          'sum the two route contributions, preserving the feature axis')
    check(cast.parameters.to is DType.BF16,
          'NATIVE_COMBINE_CAST', 'operations', 'round FP32 sum once to BF16')
    check(store.parameters.coalesced,
          'NATIVE_COMBINE_STORE', 'operations',
          'the first 16 lanes store one contiguous BF16 token')

    def mapped(operation, buffer, dimensions):
        access = schedule.access_map(operation.op_id, buffer.name)
        if (access is None or access.boundary is not BoundaryPolicy.MASK_TILED_AXES
                or len(access.indices) != len(dimensions) + 1 or mapping is None):
            return False
        token, *tail = access.indices
        return (token.source is AccessIndexKind.PROGRAM
                and token.name == mapping.axes[0].name
                and all(index.source is AccessIndexKind.DIMENSION
                        and index.dimension == dimension
                        and index.offset == 0 and index.extent is None
                        for index, dimension in zip(tail, dimensions, strict=True)))

    check(len(schedule.access_maps) == 3
          and mapped(load_values, contributions, (1, 2))
          and mapped(load_weights, weights, (1,))
          and mapped(store, output, (1,)),
          'NATIVE_COMBINE_ACCESS', 'access_maps',
          'complete route and feature axes need exact masked global coordinates')
    return tuple(findings)


class Emitter(_Emitter):
    def emit(self) -> Emission:
        load_values, load_weights, multiply, reduce, cast, store = self.s.operations
        contributions = self.b(load_values.reads[0])
        weights = self.b(load_weights.reads[0])
        output = self.b(store.writes[0])
        tokens = contributions.shape[0]
        self.line('// Generated by Open-Cake native CUDA; schedule_sha256=__SCHEDULE_SHA256__')
        self.line('#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>'
                  '\n#include <cstdint>\n#include <new>\n#include <cstring>')
        major, minor = self.target.compute_capability
        arch = major * 100 + minor * 10
        self.line(f'#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != {arch}\n'
                  f'#error "Schedule requires exact {self.s.target}"\n#endif')
        params = [f'{_TYPES[buffer.dtype]}* {self.names[buffer.name]}'
                  for buffer in self.globals]
        self.begin(f'extern "C" __global__ void {self.entry}_kernel('+', '.join(params)+')')
        self.line('const int cake_lane = int(threadIdx.x);')
        self.line('const int cake_token = int(blockIdx.x);')
        self.begin(f'if (cake_token < {tokens} && cake_lane < 16)')
        values = self.names[load_values.writes[0]]
        route_weights = self.names[load_weights.writes[0]]
        weighted = self.names[multiply.writes[0]]
        summed = self.names[reduce.writes[0]]
        rounded = self.names[cast.writes[0]]
        self.line(f'float {values}[2], {route_weights}[2], {weighted}[2];')
        self.line(f'float {summed};')
        self.line(f'__nv_bfloat16 {rounded};')
        self.line(f'// CAKE_OP: {load_values.op_id}')
        self.begin('for (int route = 0; route < 2; ++route)')
        self.line(f'{values}[route] = {self.names[contributions.name]}'
                  '[cake_token * 32 + route * 16 + cake_lane];')
        self.end()
        self.line(f'// CAKE_OP: {load_weights.op_id}')
        self.begin('for (int route = 0; route < 2; ++route)')
        self.line(f'{route_weights}[route] = {self.names[weights.name]}'
                  '[cake_token * 2 + route];')
        self.end()
        self.line(f'// CAKE_OP: {multiply.op_id}')
        self.begin('for (int route = 0; route < 2; ++route)')
        self.line(f'{weighted}[route] = {values}[route] * {route_weights}[route];')
        self.end()
        self.line(f'// CAKE_OP: {reduce.op_id}')
        self.line(f'{summed} = {weighted}[0] + {weighted}[1];')
        self.line(f'// CAKE_OP: {cast.op_id}')
        self.line(f'{rounded} = __float2bfloat16_rn({summed});')
        self.line(f'// CAKE_OP: {store.op_id}')
        self.line(f'{self.names[output.name]}[cake_token * 16 + cake_lane] = {rounded};')
        self.end()
        self.end()
        self.line('// CAKE_KERNEL_END')
        self.host()
        return Emission('\n'.join(self.lines) + '\n', self.entry, {'shared_bytes': 0},
                        self.metadata('two weighted FP32 routes per token; one BF16 round'))
