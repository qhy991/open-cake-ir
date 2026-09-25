"""B300 native lowering for reservation-owned BF16 expert-bin row copies.

The Schedule uses ordinary global loads, a returned-old atomic reservation
and two indexed stores. The CPU host boundary checks the input domain that
bounds each expert to one row per source token before launching the kernel.
"""
from __future__ import annotations

from .common import Emission, refusal
from .native_cuda import _Emitter, _TYPES
from ..diagnostics import Finding
from ..ir import (AccessIndexKind, AtomicMemoryOrder, AtomicMemoryScope,
                  AtomicOp, BoundaryPolicy, BufferMode, DType, LoadMovement,
                  LoweringBackend, MemorySpace, OperationKind, Schedule)
from ..target import CodeObject, Target


BIN_PACK_ROUTE_EVIDENCE = frozenset({'sm_103a'})
_ATOM = 'ptx.atom.relaxed.gpu.global.add.s32'
_BODY = (OperationKind.LOAD, OperationKind.LOAD, OperationKind.LOAD,
         OperationKind.ATOMIC_RMW, OperationKind.LOAD,
         OperationKind.STORE, OperationKind.STORE)


def preflight(s: Schedule, target: Target) -> tuple[Finding, ...]:
    findings: list[Finding] = []

    def check(ok, code, path, message):
        if not ok:
            findings.append(refusal(code, path, message))

    check(s.lowering.backend is LoweringBackend.NATIVE_CUDA
          and s.target == target.target_id
          and target.target_id in BIN_PACK_ROUTE_EVIDENCE
          and target.code_object is CodeObject.CUBIN
          and target.compute_capability == (10, 3)
          and target.warp_size == 32 and bool(target.device_names)
          and _ATOM in target.instruction_contracts,
          'NATIVE_BIN_TARGET', 'target',
          'the exact B300 cubin Target must declare the GPU-scope PTX atomic')
    check(not s.allocations and not s.barriers and not s.pipelines
          and not s.tile_loops and s.grid is None and s.residency is None,
          'NATIVE_BIN_RESOURCES', 'allocations',
          'the bin pack uses one synchronous 256-thread CTA per route')
    check(len(s.roles) == 1 and s.roles[0].execution_groups == tuple(range(8))
          and s.roles[0].registers_per_thread is None,
          'NATIVE_BIN_ROLE', 'roles',
          'eight complete execution groups own the BF16 row copy')
    mapping = s.program_map
    check(mapping is not None and len(mapping.axes) == 2
          and not mapping.persistent and not mapping.cooperative
          and [(axis.axis, axis.dimension, axis.tile) for axis in mapping.axes]
          == [(0, 0, 1), (1, 1, 1)],
          'NATIVE_BIN_MAP', 'program_map',
          'scalar token and route axes own one independent reservation CTA')
    if len(s.operations) != 7 or tuple(op.kind for op in s.operations) != _BODY:
        check(False, 'NATIVE_BIN_BODY', 'operations',
              'three index loads, one atomic, one BF16 load and two stores are required')
        return tuple(findings)
    le, lt, lk, reserve, lh, sk, sh = s.operations
    if (any(len(op.writes) != 1 or len(op.reads) != 1
            for op in (le, lt, lk))
            or len(reserve.reads) != 2 or len(reserve.writes) != 2
            or len(lh.reads) != 2 or len(lh.writes) != 1
            or len(sk.reads) != 3 or len(sk.writes) != 1
            or len(sh.reads) != 3 or len(sh.writes) != 1):
        check(False, 'NATIVE_BIN_EDGES', 'operations',
              'indexed reads and reservation-owned stores have exact operand edges')
        return tuple(findings)
    names = (le.reads[0], lt.reads[0], lk.reads[0], lh.reads[0],
             reserve.reads[0], sk.writes[0], sh.writes[0],
             le.writes[0], lt.writes[0], lk.writes[0],
             reserve.writes[1], lh.writes[0])
    buffers = {name: s.buffer(name) for name in names}
    if (len(set(names)) != 12 or set(names) != {b.name for b in s.buffers}
            or any(value is None for value in buffers.values())):
        check(False, 'NATIVE_BIN_BUFFERS', 'buffers',
              'seven globals and five distinct register values form the complete pack')
        return tuple(findings)
    ids, token_index, key_input, hidden, counts, keys, packed, expert, token, key, row, values = (
        buffers[name] for name in names)
    tokens, routes = ids.shape if len(ids.shape) == 2 else (0, 0)
    experts = counts.shape[0] if len(counts.shape) == 1 else 0
    maximum = keys.shape[1] if len(keys.shape) == 2 else 0
    width = hidden.shape[1] if len(hidden.shape) == 2 else 0
    check((tokens, routes, experts, maximum, width)
          == (2048, 8, 32, 2048, 2048)
          and token_index.shape == key_input.shape == (tokens, routes)
          and keys.shape == (experts, maximum)
          and packed.shape == (experts, maximum, width),
          'NATIVE_BIN_GEOMETRY', 'buffers',
          'the evidenced B300 route/bin geometry is 2048x8, E32 and H2048')
    global_specs = ((ids, DType.INT32, BufferMode.INPUT),
                    (token_index, DType.INT32, BufferMode.INPUT),
                    (key_input, DType.INT32, BufferMode.INPUT),
                    (hidden, DType.BF16, BufferMode.INPUT),
                    (counts, DType.INT32, BufferMode.STATE),
                    (keys, DType.INT32, BufferMode.OUTPUT),
                    (packed, DType.BF16, BufferMode.OUTPUT))
    check(all(b.space is MemorySpace.GLOBAL and b.dtype is dtype and b.mode is mode
              for b, dtype, mode in global_specs),
          'NATIVE_BIN_GLOBALS', 'buffers',
          'global route metadata, state and BF16 rows must have exact dtypes and modes')
    check(all(b.space is MemorySpace.REGISTER and b.mode is BufferMode.SCRATCH
              and b.dtype is DType.INT32 and b.shape == (1,)
              for b in (expert, token, key, row))
          and values.space is MemorySpace.REGISTER
          and values.mode is BufferMode.SCRATCH
          and values.dtype is DType.BF16 and values.shape == (1, width),
          'NATIVE_BIN_REGISTERS', 'buffers',
          'scalar INT32 indices and one BF16 row are explicit register values')
    check(all(b.allocation is None and b.byte_offset == 0 and b.stages == 1
              and b.swizzle is None and b.scale_of is None and b.valid_extent is None
              for b in s.buffers),
          'NATIVE_BIN_REFINEMENT', 'buffers',
          'the bin pack has no staged or aliased storage refinements')
    check(mapping is not None and len(mapping.axes) == 2
          and all(axis.buffer == ids.name for axis in mapping.axes),
          'NATIVE_BIN_OWNER', 'program_map',
          'both scalar axes are owned by the route-id input')
    check(s.outputs == (keys.name, packed.name),
          'NATIVE_BIN_OUTPUT', 'outputs',
          'route keys and packed BF16 rows are the two observable outputs')
    check(reserve.reads == (counts.name, expert.name)
          and reserve.writes == (counts.name, row.name)
          and lh.reads == (hidden.name, token.name)
          and sk.reads == (key.name, expert.name, row.name)
          and sh.reads == (values.name, expert.name, row.name),
          'NATIVE_BIN_DATAFLOW', 'operations',
          'one returned-old row must own both indexed stores')
    dependencies = ((), (), (), (le.op_id,), (lt.op_id,),
                    (lk.op_id, reserve.op_id), (lh.op_id, reserve.op_id))
    check(tuple(op.depends_on for op in s.operations) == dependencies
          and all(op.role == s.roles[0].name and not op.signals and not op.waits
                  and op.pipeline is None for op in s.operations) if s.roles else False,
          'NATIVE_BIN_DEPENDENCIES', 'operations',
          'all producers and reservations must precede their stores in one CTA role')
    check(all(op.parameters.movement is LoadMovement.GLOBAL
              and op.parameters.reuse is None for op in (le, lt, lk, lh))
          and reserve.parameters.op is AtomicOp.ADD
          and reserve.parameters.value == 1
          and reserve.parameters.order is AtomicMemoryOrder.RELAXED
          and reserve.parameters.scope is AtomicMemoryScope.DEVICE
          and sk.parameters.coalesced and sh.parameters.coalesced,
          'NATIVE_BIN_OPERATIONS', 'operations',
          'direct loads, returned-old GPU atomic +1 and coalesced stores are required')
    if mapping is not None and len(mapping.axes) == 2:
        token_axis, route_axis = (axis.name for axis in mapping.axes)

        def mapped(op, buffer, expected):
            access = s.access_map(op.op_id, buffer.name)
            if access is None or access.boundary is not BoundaryPolicy.MASK_TILED_AXES:
                return False
            actual = tuple((part.source, part.name, part.dimension)
                           for part in access.indices)
            return actual == expected

        scalar = ((AccessIndexKind.PROGRAM, token_axis, None),
                  (AccessIndexKind.PROGRAM, route_axis, None))
        check(len(s.access_maps) == 7
              and mapped(le, ids, scalar)
              and mapped(lt, token_index, scalar)
              and mapped(lk, key_input, scalar)
              and mapped(reserve, counts,
                         ((AccessIndexKind.BUFFER, expert.name, None),))
              and mapped(lh, hidden,
                         ((AccessIndexKind.BUFFER, token.name, None),
                          (AccessIndexKind.DIMENSION, None, 1)))
              and mapped(sk, keys,
                         ((AccessIndexKind.BUFFER, expert.name, None),
                          (AccessIndexKind.BUFFER, row.name, None)))
              and mapped(sh, packed,
                         ((AccessIndexKind.BUFFER, expert.name, None),
                          (AccessIndexKind.BUFFER, row.name, None),
                          (AccessIndexKind.DIMENSION, None, 2))),
              'NATIVE_BIN_ACCESS', 'access_maps',
              'source token and atomic old row must own every dynamic address')
    check(all('\n' not in op.op_id and '\r' not in op.op_id
              and '\\' not in op.op_id for op in s.operations),
          'NATIVE_BIN_OP_NAME', 'operations',
          'source-map operation IDs cannot contain line breaks or backslashes')
    return tuple(findings)


class Emitter(_Emitter):
    def emit(self) -> Emission:
        le, lt, lk, reserve, lh, sk, sh = self.s.operations
        ids, token_index, key_input, hidden, counts, keys, packed = (
            self.b(name) for name in (le.reads[0], lt.reads[0], lk.reads[0],
                                      lh.reads[0], reserve.reads[0],
                                      sk.writes[0], sh.writes[0]))
        tokens, routes = ids.shape
        experts, maximum, width = packed.shape
        self.line('// Generated by Open-Cake native CUDA; schedule_sha256=__SCHEDULE_SHA256__')
        self.line('#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n'
                  '#include <cstdint>\n#include <cstring>\n#include <new>\n#include <vector>')
        major, minor = self.target.compute_capability
        arch = major*100 + minor*10
        self.line(f'#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != {arch}\n'
                  f'#error "Schedule requires exact {self.s.target}"\n#endif')
        params = [f'{_TYPES[b.dtype]}* {self.names[b.name]}' for b in self.globals]
        self.begin(f'extern "C" __global__ void {self.entry}_kernel('+', '.join(params)+')')
        self.line('const int cake_token = int(blockIdx.x);')
        self.line('const int cake_route = int(blockIdx.y);')
        self.line(f'const int cake_linear = cake_token * {routes} + cake_route;')
        self.line(f'// CAKE_OP: {le.op_id}')
        self.line(f'const int cake_expert = {self.names[ids.name]}[cake_linear];')
        self.line('if (cake_expert < 0) return;')
        self.line(f'// CAKE_OP: {lt.op_id}')
        self.line(f'const int cake_source = {self.names[token_index.name]}[cake_linear];')
        self.line(f'// CAKE_OP: {lk.op_id}')
        self.line(f'const int cake_key = {self.names[key_input.name]}[cake_linear];')
        self.line('if (cake_expert >= '+str(experts)+' || cake_source < 0 || cake_source >= '+str(tokens)+') return;')
        self.line('__shared__ int cake_row;')
        self.begin('if (threadIdx.x == 0)')
        self.line(f'// CAKE_OP: {reserve.op_id}')
        self.line(f'asm volatile("atom.relaxed.gpu.global.add.s32 %0, [%1], %2;"'
                  f' : "=r"(cake_row)'
                  f' : "l"(reinterpret_cast<unsigned long long>(&{self.names[counts.name]}[cake_expert])), '
                  '"r"(1) : "memory");')
        self.end()
        self.line('__syncthreads();')
        self.line(f'if (cake_row < 0 || cake_row >= {maximum}) return;')
        self.line(f'// CAKE_OP: {lh.op_id}')
        self.line(f'const uint16_t* cake_source_row = reinterpret_cast<const uint16_t*>({self.names[hidden.name]}) + size_t(cake_source) * {width};')
        self.begin('if (threadIdx.x == 0)')
        self.line(f'// CAKE_OP: {sk.op_id}')
        self.line(f'{self.names[keys.name]}[cake_expert * {maximum} + cake_row] = cake_key;')
        self.end()
        self.line(f'// CAKE_OP: {sh.op_id}')
        self.line(f'uint16_t* cake_destination = reinterpret_cast<uint16_t*>({self.names[packed.name]}) + (size_t(cake_expert) * {maximum} + cake_row) * {width};')
        self.line('#pragma unroll 1')
        self.begin(f'for (int feature = int(threadIdx.x); feature < {width}; feature += blockDim.x)')
        self.line('cake_destination[feature] = cake_source_row[feature];')
        self.end(); self.end(); self.line('// CAKE_KERNEL_END')
        self.host()
        metadata = self.metadata('one route per 256-thread CTA; bitwise BF16 expert-bin row copy')
        metadata['input_domain_runtime_check'] = True
        metadata['state_reset'] = 'zero_counts_before_launch'
        metadata['bin_capacity_rows_per_expert'] = maximum
        return Emission('\n'.join(self.lines)+'\n', self.entry, {'shared_bytes': 0}, metadata)

    def host(self):
        le, lt, lk, reserve, lh, sk, sh = self.s.operations
        ids, token_index, key_input, counts = (self.b(name) for name in
            (le.reads[0], lt.reads[0], lk.reads[0], reserve.reads[0]))
        tokens, routes = ids.shape
        experts = counts.shape[0]
        handle = self.entry + '_handle'
        self.begin(f'struct {handle}')
        self.line('int device;')
        for buffer in self.globals:
            self.line(f'{_TYPES[buffer.dtype]}* {self.names[buffer.name]};')
        self.end(); self.lines[-1] += ';'
        self.begin(f'extern "C" int {self.entry}_create(void** buffers, void** result)')
        self.line('if (!buffers || !result) return int(cudaErrorInvalidValue);')
        self.line('*result = nullptr;')
        self.line('int device; cudaDeviceProp prop; cudaError_t error = cudaGetDevice(&device);')
        self.line('if (error != cudaSuccess) return int(error);')
        self.line('error = cudaGetDeviceProperties(&prop, device); if (error != cudaSuccess) return int(error);')
        major, minor = self.target.compute_capability
        names = ' && '.join(f'std::strcmp(prop.name, "{name}") != 0'
                             for name in self.target.device_names)
        self.line(f'if (prop.major != {major} || prop.minor != {minor} || ({names})) return int(cudaErrorInvalidDevice);')
        self.line(f'auto* h = new(std::nothrow) {handle}; if (!h) return int(cudaErrorMemoryAllocation);')
        self.line('h->device = device;')
        for index, buffer in enumerate(self.globals):
            self.line(f'if (!buffers[{index}]) {{ delete h; return int(cudaErrorInvalidValue); }}')
            self.line(f'h->{self.names[buffer.name]} = static_cast<{_TYPES[buffer.dtype]}*>(buffers[{index}]);')
        self.line('*result = h; return 0;'); self.end()
        self.begin(f'extern "C" int {self.entry}_launch(void* handle, void* stream)')
        self.line('if (!handle) return int(cudaErrorInvalidValue);')
        self.line(f'auto* h = static_cast<{handle}*>(handle);')
        self.line('int device; cudaError_t error = cudaGetDevice(&device);')
        self.line('if (error != cudaSuccess) return int(error);')
        self.line('if (device != h->device) return int(cudaErrorInvalidDevice);')
        self.line('cudaStream_t cake_stream = reinterpret_cast<cudaStream_t>(stream);')
        self.line('error = cudaStreamSynchronize(cake_stream);')
        self.line('if (error != cudaSuccess) return int(error);')
        for name in ('cake_ids', 'cake_tokens', 'cake_keys'):
            self.line(f'std::vector<int32_t> {name}({tokens * routes});')
        self.line(f'std::vector<int32_t> cake_counts({experts});')
        for variable, buffer, size in (('cake_ids', ids, tokens*routes),
                                       ('cake_tokens', token_index, tokens*routes),
                                       ('cake_keys', key_input, tokens*routes),
                                       ('cake_counts', counts, experts)):
            self.line(f'error = cudaMemcpy({variable}.data(), h->{self.names[buffer.name]}, '
                      f'{size} * sizeof(int32_t), cudaMemcpyDeviceToHost);')
            self.line('if (error != cudaSuccess) return int(error);')
        self.begin(f'for (int expert=0; expert<{experts}; ++expert)')
        self.line('if (cake_counts[expert] != 0) return int(cudaErrorInvalidValue);')
        self.end()
        self.begin(f'for (int token=0; token<{tokens}; ++token)')
        self.line('uint32_t seen = 0;')
        self.begin(f'for (int route=0; route<{routes}; ++route)')
        self.line(f'const int index = token * {routes} + route;')
        self.line('const int expert = cake_ids[index];')
        self.line(f'if (expert < -1 || expert >= {experts} || cake_tokens[index] != token || cake_keys[index] != index) return int(cudaErrorInvalidValue);')
        self.begin('if (expert >= 0)')
        self.line('const uint32_t bit = 1u << expert;')
        self.line('if (seen & bit) return int(cudaErrorInvalidValue);')
        self.line('seen |= bit;')
        self.end(); self.end(); self.end()
        args = [f'h->{self.names[b.name]}' for b in self.globals]
        self.line(f'{self.entry}_kernel<<<dim3({tokens},{routes},1), 256, 0, cake_stream>>>('+', '.join(args)+');')
        self.line('return int(cudaGetLastError());'); self.end()
        self.begin(f'extern "C" int {self.entry}_destroy(void* handle)')
        self.line(f'delete static_cast<{handle}*>(handle); return 0;'); self.end()
