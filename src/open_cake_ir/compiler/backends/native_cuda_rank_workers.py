"""Two-rank cooperative worker lowering for three complete FP32 FMA leaves.

This is an executable distributed control slice: rank 0 dispatches and
combines, rank 1 owns regular compute, and rank 0 may steal compute from the
same system-scope queue. It is not an MoE opcode or an EP4 implementation.
"""
from __future__ import annotations

import re

from ..ir import DType, HandoffScope, LoweringBackend, MemorySpace
from ..program import LoweredWorkerProgram
from ..target import CodeObject
from .native_cuda_pointwise import preflight as pointwise_preflight
from .native_cuda_workers import _stage_source


_IDENTIFIER = re.compile(r'[A-Za-z_][A-Za-z_0-9]*\Z')
_ATOMIC = 'ptx.atom.relaxed.sys.global.add.s32'
_HANDOFF = {'ptx.st.release.sys.global.s32',
            'ptx.ld.acquire.sys.global.s32'}


def _refuse(message):
    raise ValueError(f'rank-placed worker Program native lowering refused: {message}')


def _admit(compiler, program):
    execution = program.execution
    placement = execution.placement if execution is not None else None
    if (execution is None or placement is None
            or execution.lowering.backend is not LoweringBackend.NATIVE_CUDA):
        _refuse('version-3 rank placement needs the native_cuda route')
    if not _IDENTIFIER.fullmatch(execution.lowering.entry_point):
        _refuse('entry point must be one ASCII C identifier')
    target = compiler._revision.targets.get(program.target)
    if (target is None or target.code_object is not CodeObject.CUBIN
            or target.cooperative_grid is not True or target.occupancy is None
            or target.compute_capability is None or target.warp_size != 32
            or _ATOMIC not in target.instruction_contracts
            or not _HANDOFF <= target.synchronization_contracts):
        _refuse('exact Target needs cooperative occupancy and system-scope queue/handoff contracts')
    if len(program.stages) != 3 or len(execution.handoffs) != 2:
        _refuse('this slice needs three mathematical leaves and two handoffs')
    names = tuple(stage.name for stage in program.stages)
    first, second = execution.workers
    if (first.phases != (names[0], names[2]) or second.phases != (names[1],)
            or tuple(queue.workers for queue in execution.queues)
            != ((first.name,), (first.name, second.name), (first.name,))
            or (execution.steal.borrower, execution.steal.stage,
                execution.steal.after, execution.steal.before)
            != (first.name, names[1], names[0], names[2])
            or tuple((h.producer, h.consumer, h.scope) for h in execution.handoffs)
            != ((names[0], names[1], HandoffScope.SYSTEM),
                (names[1], names[2], HandoffScope.SYSTEM))):
        _refuse('two-rank phases, queue ownership, steal and system handoffs differ')
    if (placement.world_size != 2 or placement.state_rank != 0
            or tuple(placement.worker_ranks[w.name] for w in execution.workers) != (0, 1)
            or tuple(placement.stage_ranks[name] for name in names) != (0, 1, 0)):
        _refuse('first CTA class and state must be rank 0; regular compute must be rank 1')
    owners = {name: 0 for name in program.tensors}
    owners[execution.handoffs[0].payload] = 1
    if dict(placement.tensor_ranks) != owners:
        _refuse('this slice owns the first intermediate on rank 1 and other tensors on rank 0')
    tiles, width = set(), set()
    stage_documents = program.document['stages']
    for index, stage in enumerate(program.stages):
        schedule = stage.schedule
        if schedule.lowering.backend is not LoweringBackend.NATIVE_CUDA:
            _refuse(f'stage {stage.name!r} needs an explicit native CUDA leaf route')
        assessment = compiler.assess(stage_documents[index]['schedule'])
        if not assessment.lowering_eligible:
            _refuse(f'stage {stage.name!r} refused: '
                    f'{[finding.code for finding in assessment.findings if finding.blocks_lowering]}')
        failures = pointwise_preflight(schedule, target)
        if failures:
            _refuse(f'stage {stage.name!r} pointwise body refused: {[f.code for f in failures]}')
        mapping = schedule.program_map
        if (not mapping.persistent or not mapping.cooperative
                or schedule.residency.ctas_per_multiprocessor != 1
                or any(binding.singleton_view for binding in stage.bindings.values())):
            _refuse(f'stage {stage.name!r} needs one cooperative CTA tile and exact bindings')
        owner = schedule.buffer(mapping.axes[0].buffer)
        tiles.add(owner.shape[0]); width.add(owner.shape[1])
    if len(tiles) != 1 or len(width) != 1:
        _refuse('leaf stages have different row/lane tile domains')
    tile_count, lanes = tiles.pop(), width.pop()
    sms = target.occupancy.multiprocessor_count
    if tile_count < sms:
        _refuse('tile domain must keep the cooperative grid productive')
    if any(tensor.dtype is not DType.FP32 or tensor.shape != (tile_count, lanes)
           for name, tensor in program.tensors.items()
           if name not in execution.controls.values()):
        _refuse('mathematical tensors must share the FP32 tile domain')
    for index, handoff in enumerate(execution.handoffs):
        producer = program.stages[index]
        if producer.bindings[producer.schedule.outputs[0]].tensor != handoff.payload:
            _refuse(f'handoff {handoff.payload!r} must own the preceding stage output')
    return target, tile_count, lanes, sms


_DEVICE = r'''
__device__ __forceinline__ int cake_sys_claim(int32_t* pointer) {
  int old;
  asm volatile("atom.relaxed.sys.global.add.s32 %0, [%1], %2;" : "=r"(old) :
    "l"(reinterpret_cast<unsigned long long>(pointer)), "r"(1) : "memory");
  return old;
}
__device__ __forceinline__ int cake_sys_load(const int32_t* pointer) {
  int value;
  asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(value) :
    "l"(reinterpret_cast<unsigned long long>(pointer)) : "memory");
  return value;
}
__device__ __forceinline__ void cake_sys_publish(int32_t* pointer) {
  asm volatile("st.release.sys.global.s32 [%0], %1;" ::
    "l"(reinterpret_cast<unsigned long long>(pointer)), "r"(1) : "memory");
}
__device__ __forceinline__ int cake_warp_claim(int32_t* pointer, int lane) {
  int old = lane == 0 ? cake_sys_claim(pointer) : 0;
  return __shfl_sync(0xffffffffu, old, 0);
}
'''


_KERNEL = r'''
__device__ __forceinline__ void @ENTRY@_compute(@FLOAT_PARAMS@, int32_t* state,
                                                  int tile, int lane,
                                                  int per_chunk, bool stolen) {
  while (cake_sys_load(state + @READY0@ + tile) == 0) __nanosleep(64);
  @ENTRY@_stage1(@FLOAT_ARGS@, tile, lane);
  __syncwarp();
  if (lane == 0) {
    cake_sys_publish(state + @READY1@ + tile);
    cake_sys_claim(state + @CHUNK@ + tile / per_chunk);
    cake_sys_claim(state + 4);
    if (stolen) cake_sys_claim(state + 8);
  }
  __syncwarp();
}
extern "C" __global__ void @ENTRY@_kernel(@KERNEL_PARAMS@, int32_t* state,
                                             int rank) {
  const int lane = int(threadIdx.x);
  const int c = @C_PTR@[0], k = @K_PTR@[0], budget = @BUDGET_PTR@[0];
  if (c <= 0 || c >= @GRID@ || k <= 0 || k > @TILES@ ||
      @TILES@ % k != 0 || budget < 0 || budget > @TILES@) {
    if (rank == 0 && blockIdx.x == 0 && lane == 0) state[7] = -1;
    return;
  }
  const int per_chunk = @TILES@ / k;
  if (rank == 0 && int(blockIdx.x) < c) {
    while (true) {
      int tile = cake_warp_claim(state + 0, lane);
      if (tile >= @TILES@) break;
      @ENTRY@_stage0(@FLOAT_ARGS@, tile, lane);
      __syncwarp();
      if (lane == 0) {
        cake_sys_publish(state + @READY0@ + tile);
        cake_sys_claim(state + 3);
      }
      __syncwarp();
    }
    if (lane == 0)
      while (cake_sys_load(state + 3) < @TILES@) __nanosleep(64);
    __syncwarp();
    while (true) {
      int stop = 0;
      if (lane == 0)
        stop = cake_sys_load(state + @CHUNK@) >= per_chunk ||
               cake_sys_claim(state + 6) >= budget;
      stop = __shfl_sync(0xffffffffu, stop, 0);
      if (stop) break;
      int tile = cake_warp_claim(state + 1, lane);
      if (tile >= @TILES@) break;
      @ENTRY@_compute(@FLOAT_ARGS@, state, tile, lane, per_chunk, true);
    }
    if (lane == 0) cake_sys_claim(state + 5);
    __syncwarp();
    if (lane == 0)
      while (cake_sys_load(state + 5) < c) __nanosleep(64);
    __syncwarp();
    while (true) {
      int tile = cake_warp_claim(state + 2, lane);
      if (tile >= @TILES@) break;
      while (cake_sys_load(state + @CHUNK@ + tile / per_chunk) < per_chunk)
        __nanosleep(64);
      while (cake_sys_load(state + @READY1@ + tile) == 0) __nanosleep(64);
      if (lane == 0)
        atomicCAS(state + 9, 0, cake_sys_load(state + 4) + 1);
      @ENTRY@_stage2(@FLOAT_ARGS@, tile, lane);
      __syncwarp();
    }
  } else if (rank == 1 && int(blockIdx.x) >= c) {
    while (true) {
      int tile = cake_warp_claim(state + 1, lane);
      if (tile >= @TILES@) break;
      @ENTRY@_compute(@FLOAT_ARGS@, state, tile, lane, per_chunk, false);
    }
  }
}
// CAKE_KERNEL_END
'''


def _replace(template: str, values: dict[str, object]) -> str:
    for name, value in values.items():
        template = template.replace('@' + name + '@', str(value))
    if '@' in template:
        raise ValueError('native rank worker source has an unfilled placeholder')
    return template


def _emit(program, target, tiles: int, width: int, grid: int):
    execution = program.execution
    placement = execution.placement
    entry = execution.lowering.entry_point
    names = tuple(program.tensors)
    floats = tuple(name for name in names if name not in execution.controls.values())
    pointer = {name: f'p{index}' for index, name in enumerate(names)}
    float_params = ', '.join(f'float* {pointer[name]}' for name in floats)
    float_args = ', '.join(pointer[name] for name in floats)
    kernel_params = ', '.join(
        f'{"int32_t" if name in execution.controls.values() else "float"}* {pointer[name]}'
        for name in names)
    state_ints = 10 + 3 * tiles
    major, minor = target.compute_capability
    arch = major * 100 + minor * 10
    values = {
        'ENTRY': entry, 'FLOAT_PARAMS': float_params, 'FLOAT_ARGS': float_args,
        'KERNEL_PARAMS': kernel_params, 'READY0': 10,
        'READY1': 10 + tiles, 'CHUNK': 10 + 2 * tiles,
        'GRID': grid, 'TILES': tiles,
        'C_PTR': pointer[execution.controls['first_class_ctas']],
        'K_PTR': pointer[execution.controls['chunk_count']],
        'BUDGET_PTR': pointer[execution.controls['steal_budget']],
    }
    pieces = ['#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cstdint>'
              '\n#include <new>\n#include <cstring>',
              f'#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != {arch}\n'
              f'#error "Program requires exact {program.target}"\n#endif',
              _DEVICE,
              *(_stage_source(program, index, width, floats) for index in range(3)),
              _replace(_KERNEL, values)]
    handle = entry + '_handle'
    lines = [f'struct {handle} {{', '  int device, rank;', '  int32_t* state;']
    for name in names:
        dtype = 'int32_t' if name in execution.controls.values() else 'float'
        lines.append(f'  {dtype}* {pointer[name]};')
    lines += ['};',
              f'extern "C" int {entry}_create_rank(int rank, void** buffers, '
              'void* state, size_t state_bytes, void** result) {',
              f'  if (rank < 0 || rank >= 2 || !buffers || !state || !result || '
              f'state_bytes < {state_ints * 4}) return int(cudaErrorInvalidValue);',
              '  *result = nullptr;',
              '  int device; cudaDeviceProp prop;',
              '  cudaError_t error = cudaGetDevice(&device);',
              '  if (error != cudaSuccess) return int(error);',
              '  if (device != rank) return int(cudaErrorInvalidDevice);',
              '  error = cudaGetDeviceProperties(&prop, device);',
              '  if (error != cudaSuccess) return int(error);']
    device_names = ' && '.join(f'std::strcmp(prop.name, "{name}") != 0'
                                for name in target.device_names)
    lines += [f'  if (prop.major != {major} || prop.minor != {minor} || '
              f'prop.multiProcessorCount != {grid} || ({device_names})) '
              '{ return int(cudaErrorInvalidDevice); }',
              '  cudaPointerAttributes state_attrs{};',
              '  error = cudaPointerGetAttributes(&state_attrs, state);',
              '  if (error != cudaSuccess) return int(error);',
              '  if (state_attrs.type != cudaMemoryTypeDevice || '
              'state_attrs.device != 0) return int(cudaErrorInvalidDevicePointer);']
    raw_pointers = ', '.join(f'buffers[{index}]' for index in range(len(names)))
    expected_owners = ', '.join(str(placement.tensor_ranks[name]) for name in names)
    lines += [f'  void* raw[] = {{{raw_pointers}}};',
              f'  const int owners[] = {{{expected_owners}}};',
              f'  for (int index = 0; index < {len(names)}; ++index) {{',
              '    if (!raw[index]) return int(cudaErrorInvalidValue);',
              '    cudaPointerAttributes attrs{};',
              '    error = cudaPointerGetAttributes(&attrs, raw[index]);',
              '    if (error != cudaSuccess) return int(error);',
              '    if (attrs.type != cudaMemoryTypeDevice || attrs.device != owners[index]) '
              'return int(cudaErrorInvalidDevicePointer);',
              '    if (attrs.device != device) {',
              '      cudaDeviceProp peer{};',
              '      error = cudaGetDeviceProperties(&peer, attrs.device);',
              '      if (error != cudaSuccess) return int(error);']
    peer_names = ' && '.join(f'std::strcmp(peer.name, "{name}") != 0'
                              for name in target.device_names)
    lines += [f'      if (peer.major != {major} || peer.minor != {minor} || '
              f'peer.multiProcessorCount != {grid} || ({peer_names})) '
              'return int(cudaErrorInvalidDevice);',
              '      int access = 0, native = 0;',
              '      error = cudaDeviceCanAccessPeer(&access, device, attrs.device);',
              '      if (error != cudaSuccess) return int(error);',
              '      error = cudaDeviceGetP2PAttribute(&native, '
              'cudaDevP2PAttrNativeAtomicSupported, device, attrs.device);',
              '      if (error != cudaSuccess) return int(error);',
              '      if (access != 1 || native != 1) return int(cudaErrorNotSupported);',
              '      error = cudaDeviceEnablePeerAccess(attrs.device, 0);',
              '      if (error != cudaSuccess && error != cudaErrorPeerAccessAlreadyEnabled) '
              'return int(error);',
              '    }', '  }',
              f'  auto* h = new(std::nothrow) {handle};',
              '  if (!h) return int(cudaErrorMemoryAllocation);',
              '  h->device = device; h->rank = rank;',
              '  h->state = static_cast<int32_t*>(state);']
    for index, name in enumerate(names):
        dtype = 'int32_t' if name in execution.controls.values() else 'float'
        lines.append(f'  h->{pointer[name]} = static_cast<{dtype}*>(buffers[{index}]);')
    lines += ['  *result = h; return 0;', '}',
              f'extern "C" int {entry}_launch_two(void* first, void* second, '
              'void* stream0, void* stream1) {',
              '  if (!first || !second) return int(cudaErrorInvalidValue);',
              f'  auto* h0 = static_cast<{handle}*>(first);',
              f'  auto* h1 = static_cast<{handle}*>(second);',
              '  if (h0->rank != 0 || h1->rank != 1 || h0->device != 0 || '
              'h1->device != 1 || h0->state != h1->state) '
              'return int(cudaErrorInvalidDevice);']
    for name in names:
        lines.append(f'  if (h0->{pointer[name]} != h1->{pointer[name]}) '
                     'return int(cudaErrorInvalidValue);')
    lines += ['  cudaError_t error;',
              '  for (int rank = 0; rank < 2; ++rank) {',
              '    error = cudaSetDevice(rank);',
              '    if (error != cudaSuccess) return int(error);',
              '    int cooperative = 0;',
              '    error = cudaDeviceGetAttribute(&cooperative, '
              'cudaDevAttrCooperativeLaunch, rank);',
              '    if (error != cudaSuccess) return int(error);',
              '    if (cooperative != 1) return int(cudaErrorNotSupported);',
              '    int resident = 0;',
              f'    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor('
              f'&resident, {entry}_kernel, 32, 0);',
              '    if (error != cudaSuccess) return int(error);',
              '    if (resident < 1) return int(cudaErrorCooperativeLaunchTooLarge);',
              '  }',
              '  error = cudaSetDevice(0);',
              '  if (error != cudaSuccess) return int(error);',
              f'  error = cudaMemsetAsync(h0->state, 0, {state_ints * 4}, '
              'reinterpret_cast<cudaStream_t>(stream0));',
              '  if (error != cudaSuccess) return int(error);',
              '  error = cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream0));',
              '  if (error != cudaSuccess) return int(error);',
              '  int rank0 = 0, rank1 = 1;']
    args0 = [f'&h0->{pointer[name]}' for name in names] + ['&h0->state', '&rank0']
    args1 = [f'&h1->{pointer[name]}' for name in names] + ['&h1->state', '&rank1']
    lines += ['  void* args0[] = {' + ', '.join(args0) + '};',
              '  void* args1[] = {' + ', '.join(args1) + '};',
              f'  error = cudaLaunchCooperativeKernel(reinterpret_cast<const void*>('
              f'{entry}_kernel), dim3({grid}), dim3(32), args0, 0, '
              'reinterpret_cast<cudaStream_t>(stream0));',
              '  if (error != cudaSuccess) return int(error);',
              '  error = cudaSetDevice(1);',
              '  if (error != cudaSuccess) return int(error);',
              f'  error = cudaLaunchCooperativeKernel(reinterpret_cast<const void*>('
              f'{entry}_kernel), dim3({grid}), dim3(32), args1, 0, '
              'reinterpret_cast<cudaStream_t>(stream1));',
              '  return int(error);', '}',
              f'extern "C" int {entry}_destroy_rank(void* handle) {{',
              f'  delete static_cast<{handle}*>(handle); return 0;', '}',
    ]
    source = '\n'.join([*pieces, '\n'.join(lines)]) + '\n'
    requirements = {
        'source_language': 'cuda_cpp', 'target': program.target,
        'entry_point': entry, 'kernel_entry_point': entry + '_kernel',
        'world_size': 2, 'state_rank': 0,
        'tensor_ranks': dict(placement.tensor_ranks),
        'grid_per_rank': [[grid, 1, 1], [grid, 1, 1]],
        'block': [32, 1, 1], 'cooperative_grid': True,
        'state_bytes': state_ints * 4, 'state_status_offset_bytes': 28,
        'state_stolen_tiles_offset_bytes': 32,
        'state_first_combine_compute_done_plus_one_offset_bytes': 36,
        'state_reset': 'zero_before_both_launches_on_rank0_stream',
        'argument_order': list(names),
        'host_abi': {'create_rank': entry+'_create_rank',
                     'launch_two': entry+'_launch_two',
                     'destroy_rank': entry+'_destroy_rank'},
        'peer_pair_runtime_check': True,
        'nvcc_flags': ['-std=c++17', f'--gpu-architecture=compute_{major}{minor}a',
                       f'--gpu-code={program.target}', '-O3', '--fmad=false', '-lineinfo'],
        'link_libraries': ['cuda', 'cudart'],
    }
    return source, requirements


def lower_rank_worker_program(compiler, program) -> LoweredWorkerProgram:
    target, tiles, width, grid = _admit(compiler, program)
    if compiler.commit is None:
        raise ValueError('program compilation requires a clean Compiler commit')
    source, requirements = _emit(program, target, tiles, width, grid)
    result = LoweredWorkerProgram(program, compiler._revision.revision_id,
                                  source, requirements)
    result.validate_binding()
    return result
