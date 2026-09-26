"""B300 top-8 model-width weighted combine from ordinary Cake operations."""
from __future__ import annotations

from .common import Emission, refusal
from .native_cuda import _Emitter, _TYPES
from ..diagnostics import Finding
from ..ir import (AccessIndexKind, BoundaryPolicy, BufferMode, DType,
                  ElementwiseOp, LoadMovement, LoweringBackend, MemorySpace,
                  OperationKind, ReduceOp, ReductionScope, Schedule)
from ..target import CodeObject, Target


MODEL_COMBINE_ROUTE_EVIDENCE = frozenset({'sm_103a'})
_BODY = (OperationKind.LOAD, OperationKind.LOAD, OperationKind.ELEMENTWISE,
         OperationKind.REDUCE, OperationKind.CAST, OperationKind.STORE)


def preflight(s: Schedule, target: Target) -> tuple[Finding, ...]:
    findings: list[Finding] = []

    def check(ok, code, path, message):
        if not ok:
            findings.append(refusal(code, path, message))

    check(s.lowering.backend is LoweringBackend.NATIVE_CUDA
          and s.target == target.target_id
          and target.target_id in MODEL_COMBINE_ROUTE_EVIDENCE
          and target.code_object is CodeObject.CUBIN
          and target.compute_capability == (10, 3)
          and target.warp_size == 32 and bool(target.device_names),
          'NATIVE_MODEL_COMBINE_TARGET', 'target',
          'the model combine requires the evidenced exact B300 cubin route')
    check(not s.allocations and not s.pipelines and not s.barriers
          and not s.tile_loops and s.grid is None and s.residency is None,
          'NATIVE_MODEL_COMBINE_RESOURCES', 'allocations',
          'the 256-thread combine owns no staged resources or residency')
    check(len(s.roles) == 1 and s.roles[0].execution_groups == tuple(range(8))
          and s.roles[0].registers_per_thread is None,
          'NATIVE_MODEL_COMBINE_ROLE', 'roles',
          'eight execution groups own one 256-feature output tile')
    mapping = s.program_map
    check(mapping is not None and len(mapping.axes) == 2
          and not mapping.persistent and not mapping.cooperative
          and [(axis.axis, axis.dimension, axis.tile) for axis in mapping.axes]
          == [(0, 0, 1), (1, 2, 256)],
          'NATIVE_MODEL_COMBINE_MAP', 'program_map',
          'token and 256-feature axes own one output tile')
    ops = s.operations
    if (len(ops) != 6 or tuple(op.kind for op in ops) != _BODY
            or any(len(op.reads) != (2 if op.kind is OperationKind.ELEMENTWISE else 1)
                   or len(op.writes) != 1 for op in ops)):
        check(False, 'NATIVE_MODEL_COMBINE_BODY', 'operations',
              'two loads, multiply, SUM, BF16 cast and store are required')
        return tuple(findings)
    lv,lw,mul,reduce,cast,store=ops
    names=(lv.reads[0],lw.reads[0],store.writes[0],
           lv.writes[0],lw.writes[0],mul.writes[0],
           reduce.writes[0],cast.writes[0])
    buffers={name:s.buffer(name) for name in names}
    if (len(set(names))!=8 or set(names)!={b.name for b in s.buffers}
            or any(value is None for value in buffers.values())):
        check(False, 'NATIVE_MODEL_COMBINE_BUFFERS', 'buffers',
              'three globals and five distinct register values form the full combine')
        return tuple(findings)
    contributions,weights,output,route_values,route_weights,weighted,summed,rounded=(
        buffers[name] for name in names)
    tokens=contributions.shape[0] if len(contributions.shape)==3 else None
    check(tokens in (512,2048)
          and contributions.shape==(tokens,8,2048)
          and weights.shape==(tokens,8)
          and output.shape==(tokens,2048),
          'NATIVE_MODEL_COMBINE_SHAPE', 'buffers',
          'the evidenced combine has 512 or 2048 tokens, top-8 and H2048')
    check((contributions.space,weights.space,output.space)
          ==(MemorySpace.GLOBAL,)*3
          and (contributions.mode,weights.mode,output.mode)
          ==(BufferMode.INPUT,BufferMode.INPUT,BufferMode.OUTPUT)
          and (contributions.dtype,weights.dtype,output.dtype)
          ==(DType.FP32,DType.FP32,DType.BF16),
          'NATIVE_MODEL_COMBINE_GLOBALS', 'buffers',
          'two FP32 globals produce one BF16 global output')
    register_specs=((route_values,DType.FP32,(8,256)),
                    (route_weights,DType.FP32,(8,)),
                    (weighted,DType.FP32,(8,256)),
                    (summed,DType.FP32,(256,)),
                    (rounded,DType.BF16,(256,)))
    check(all(b.space is MemorySpace.REGISTER
              and b.mode is BufferMode.SCRATCH
              and b.dtype is dtype and b.shape == shape
              for b,dtype,shape in register_specs),
          'NATIVE_MODEL_COMBINE_REGISTERS', 'buffers',
          'route and feature register tiles must match the declared 8x256 reduction')
    check(all(b.allocation is None and b.byte_offset == 0 and b.stages == 1
              and b.swizzle is None and b.scale_of is None and b.valid_extent is None
              for b in s.buffers),
          'NATIVE_MODEL_COMBINE_REFINEMENT', 'buffers',
          'the combine has no hidden views or staged storage')
    check(mapping is not None and len(mapping.axes)==2
          and all(axis.buffer==contributions.name for axis in mapping.axes),
          'NATIVE_MODEL_COMBINE_OWNER', 'program_map',
          'both program axes derive from the contribution tensor')
    check(s.outputs==(output.name,),
          'NATIVE_MODEL_COMBINE_OUTPUT', 'outputs',
          'the BF16 token tensor is the sole output')
    check((mul.reads,reduce.reads,cast.reads,store.reads)
          ==((route_values.name,route_weights.name),(weighted.name,),
             (summed.name,),(rounded.name,)),
          'NATIVE_MODEL_COMBINE_DATAFLOW', 'operations',
          'weighted routes feed SUM, one BF16 cast and the final store')
    dependencies=((),(),(lv.op_id,lw.op_id),(mul.op_id,),
                  (reduce.op_id,),(cast.op_id,))
    check(tuple(op.depends_on for op in ops)==dependencies
          and all(op.role==s.roles[0].name and not op.signals and not op.waits
                  and op.pipeline is None for op in ops) if s.roles else False,
          'NATIVE_MODEL_COMBINE_DEPENDENCIES', 'operations',
          'all dataflow and role edges must be explicit')
    check(all(op.parameters.movement is LoadMovement.GLOBAL
              and op.parameters.reuse is None for op in (lv,lw))
          and mul.parameters.op is ElementwiseOp.MUL
          and mul.parameters.scalar is None
          and mul.parameters.broadcast_axis == 0
          and mul.parameters.instruction is None
          and reduce.parameters.op is ReduceOp.SUM
          and reduce.parameters.axis == 0
          and reduce.parameters.scope is ReductionScope.CTA
          and not reduce.parameters.across_loop
          and cast.parameters.to is DType.BF16
          and store.parameters.coalesced,
          'NATIVE_MODEL_COMBINE_OPERATIONS', 'operations',
          'direct loads, per-route FP32 multiply/SUM and one BF16 round are required')
    if mapping is not None and len(mapping.axes)==2:
        token_axis,feature_axis=(axis.name for axis in mapping.axes)

        def mapped(op,buffer,expected):
            access=s.access_map(op.op_id,buffer.name)
            return (access is not None
                    and access.boundary is BoundaryPolicy.MASK_TILED_AXES
                    and tuple((part.source,part.name,part.dimension)
                              for part in access.indices)==expected)

        token=(AccessIndexKind.PROGRAM,token_axis,None)
        feature=(AccessIndexKind.PROGRAM_TILE,feature_axis,None)
        route=(AccessIndexKind.DIMENSION,None,1)
        check(len(s.access_maps)==3
              and mapped(lv,contributions,(token,route,feature))
              and mapped(lw,weights,(token,route))
              and mapped(store,output,(token,feature)),
              'NATIVE_MODEL_COMBINE_ACCESS', 'access_maps',
              'all route and feature coordinates must be explicitly mapped')
    check(all('\n' not in op.op_id and '\r' not in op.op_id
              and '\\' not in op.op_id for op in ops),
          'NATIVE_MODEL_COMBINE_OP_NAME', 'operations',
          'source-map IDs cannot contain line breaks or backslashes')
    return tuple(findings)


class Emitter(_Emitter):
    def emit(self) -> Emission:
        lv,lw,mul,reduce,cast,store=self.s.operations
        contributions,weights,output=(self.b(lv.reads[0]),
                                      self.b(lw.reads[0]),
                                      self.b(store.writes[0]))
        tokens,routes,width=contributions.shape
        tile=self.s.program_map.axes[1].tile
        self.line('// Generated by Open-Cake native CUDA; schedule_sha256=__SCHEDULE_SHA256__')
        self.line('#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n'
                  '#include <cstdint>\n#include <cstring>\n#include <new>')
        major,minor=self.target.compute_capability
        arch=major*100+minor*10
        self.line(f'#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != {arch}\n'
                  f'#error "Schedule requires exact {self.s.target}"\n#endif')
        params=[f'{_TYPES[b.dtype]}* {self.names[b.name]}' for b in self.globals]
        self.begin(f'extern "C" __global__ void {self.entry}_kernel('+', '.join(params)+')')
        self.line('const int cake_token = int(blockIdx.x);')
        self.line(f'const int cake_feature = int(blockIdx.y) * {tile} + int(threadIdx.x);')
        self.begin(f'if (cake_token < {tokens} && cake_feature < {width})')
        self.line('float cake_sum = 0.0f;')
        self.line('#pragma unroll')
        self.begin(f'for (int route=0; route<{routes}; ++route)')
        self.line(f'// CAKE_OP: {lv.op_id}')
        self.line(f'const float cake_value = {self.names[contributions.name]}[(cake_token * {routes} + route) * {width} + cake_feature];')
        self.line(f'// CAKE_OP: {lw.op_id}')
        self.line(f'const float cake_weight = {self.names[weights.name]}[cake_token * {routes} + route];')
        self.line(f'// CAKE_OP: {mul.op_id}')
        self.line('const float cake_product = cake_value * cake_weight;')
        self.line(f'// CAKE_OP: {reduce.op_id}')
        self.line('cake_sum += cake_product;')
        self.end()
        self.line(f'// CAKE_OP: {cast.op_id}')
        self.line('const __nv_bfloat16 cake_rounded = __float2bfloat16_rn(cake_sum);')
        self.line(f'// CAKE_OP: {store.op_id}')
        self.line(f'{self.names[output.name]}[cake_token * {width} + cake_feature] = cake_rounded;')
        self.end();self.end();self.line('// CAKE_KERNEL_END')
        self.host()
        return Emission('\n'.join(self.lines)+'\n',self.entry,
                        {'shared_bytes':0},
                        self.metadata('one 256-thread CTA per token/feature tile; eight FP32 weighted routes'))
