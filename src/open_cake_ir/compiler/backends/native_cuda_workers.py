"""Bounded native worker Program lowering, composed from complete FMA leaves.

The Program supplies mathematics and bindings; WorkerExecution supplies CTA
classes, queues, device handoffs and stealing. This first executable slice is
one GPU, three same-tile FP32 leaves, one CTA per declared SM. It refuses any
different structure rather than replaying the Program as ordered kernels.
"""
from __future__ import annotations

import re

from ..ir import BufferMode, DType, HandoffScope, LoweringBackend, MemorySpace, OperationKind
from ..program import LoweredWorkerProgram
from ..target import CodeObject
from .native_cuda_pointwise import preflight as pointwise_preflight


_IDENTIFIER = re.compile(r'[A-Za-z_][A-Za-z_0-9]*\Z')


def _refuse(message: str):
    raise ValueError(f'cooperative worker Program requires dedicated native lowering: {message}')


def _admit(compiler, program):
    execution = program.execution
    if execution is None or execution.lowering.backend is not LoweringBackend.NATIVE_CUDA:
        _refuse('the execution route must be native_cuda')
    if not _IDENTIFIER.fullmatch(execution.lowering.entry_point):
        _refuse('the native entry point must be an ASCII C identifier')
    target = compiler._revision.targets.get(program.target)
    if (target is None or target.code_object is not CodeObject.CUBIN
            or target.cooperative_grid is not True or target.occupancy is None
            or target.compute_capability is None or target.warp_size != 32):
        _refuse('the exact cubin Target needs cooperative support and observed occupancy')
    if len(program.stages) != 3 or len(execution.handoffs) != 2:
        _refuse('this slice has exactly three leaf stages and two handoffs')
    names = tuple(stage.name for stage in program.stages)
    first, second = execution.workers
    if (first.phases != (names[0], names[2])
            or second.phases != (names[1], names[2])
            or tuple(queue.workers for queue in execution.queues)
            != ((first.name,), (first.name, second.name), (first.name, second.name))
            or (execution.steal.borrower, execution.steal.stage,
                execution.steal.after, execution.steal.before)
            != (first.name, names[1], names[0], names[2])):
        _refuse('CTA phases, queue owners and steal window must form one producer/compute/combine pipeline')
    if any(handoff.scope is not HandoffScope.DEVICE for handoff in execution.handoffs):
        _refuse('system-scope handoff needs an explicit multi-GPU placement and peer-pair admission')
    if tuple((h.producer, h.consumer) for h in execution.handoffs) != (
            (names[0], names[1]), (names[1], names[2])):
        _refuse('handoffs must connect adjacent stage tile domains')

    tile_counts = set()
    widths = set()
    for stage in program.stages:
        schedule = stage.schedule
        if schedule.lowering.backend is not LoweringBackend.NATIVE_CUDA:
            _refuse(f'stage {stage.name!r} does not own a native_cuda leaf route')
        assessment = compiler.assess(program.document['stages'][names.index(stage.name)]['schedule'])
        if not assessment.lowering_eligible:
            _refuse(f'stage {stage.name!r} refused: '
                    f'{[finding.code for finding in assessment.findings if finding.blocks_lowering]}')
        failures = pointwise_preflight(schedule, target)
        if failures:
            _refuse(f'stage {stage.name!r} pointwise leaf refused: {[f.code for f in failures]}')
        mapping = schedule.program_map
        if (not mapping.persistent or not mapping.cooperative
                or schedule.residency.ctas_per_multiprocessor != 1):
            _refuse(f'stage {stage.name!r} needs cooperative one-CTA-per-SM tile ownership')
        owner = schedule.buffer(mapping.axes[0].buffer)
        tile_counts.add(owner.shape[0])
        widths.add(owner.shape[1])
        if any(binding.singleton_view for binding in stage.bindings.values()):
            _refuse(f'stage {stage.name!r} cannot reinterpret a tile through a singleton view')
    if len(tile_counts) != 1 or len(widths) != 1:
        _refuse('all leaf stages must share one exact row and lane tile domain')
    tiles, width = tile_counts.pop(), widths.pop()
    grid = target.occupancy.multiprocessor_count
    if tiles < grid:
        _refuse('the tile domain must keep every cooperative CTA assigned work')
    if any(tensor.dtype is not DType.FP32 or tensor.shape != (tiles, width)
           for name, tensor in program.tensors.items()
           if name not in execution.controls.values()):
        _refuse('all mathematical tensors need the same FP32 tile domain')
    for index, handoff in enumerate(execution.handoffs):
        producer = program.stages[index]
        output_name = producer.bindings[producer.schedule.outputs[0]].tensor
        if handoff.payload != output_name:
            _refuse(f'handoff {handoff.payload!r} is not the stage output payload')
    return target, tiles, width, grid


def _stage_source(program, index: int, width: int, float_names: tuple[str, ...]) -> str:
    stage = program.stages[index]
    schedule = stage.schedule
    entry = program.execution.lowering.entry_point
    pointers = {name: f'p{position}' for position, name in enumerate(program.tensors)}
    args = ', '.join(f'float* {pointers[name]}' for name in float_names)
    lines = [f'__device__ __forceinline__ void {entry}_stage{index}({args}, int tile, int lane) {{',
             f'  if (lane < {width}) {{']
    registers = {b.name: f'r{position}' for position, b in enumerate(schedule.buffers)
                 if b.space is MemorySpace.REGISTER}
    for name in registers.values():
        lines.append(f'    float {name} = 0.0f;')
    for operation in schedule.operations:
        lines.append(f'    // CAKE_OP: {stage.name}.{operation.op_id}')
        if operation.kind is OperationKind.LOAD:
            tensor = stage.bindings[operation.reads[0]].tensor
            lines.append(f'    {registers[operation.writes[0]]} = '
                         f'{pointers[tensor]}[tile * {width} + lane];')
        elif operation.kind is OperationKind.ELEMENTWISE:
            dst = registers[operation.writes[0]]
            a, b, c = (registers[name] for name in operation.reads)
            lines.append(f'    asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"({dst}) : '
                         f'"f"({a}), "f"({b}), "f"({c}));')
        else:
            tensor = stage.bindings[operation.writes[0]].tensor
            lines.append(f'    {pointers[tensor]}[tile * {width} + lane] = '
                         f'{registers[operation.reads[0]]};')
    lines += ['  }', '}']
    return '\n'.join(lines)


def _emit(program, target, tiles: int, width: int, grid: int):
    execution = program.execution
    entry = execution.lowering.entry_point
    tensor_names = tuple(program.tensors)
    float_names = tuple(name for name in tensor_names if name not in execution.controls.values())
    pointers = {name: f'p{index}' for index, name in enumerate(tensor_names)}
    params = ', '.join(f'{"int32_t" if name in execution.controls.values() else "float"}* '
                       f'{pointers[name]}' for name in tensor_names)
    args = ', '.join(pointers[name] for name in float_names)
    c = pointers[execution.controls['first_class_ctas']]
    k = pointers[execution.controls['chunk_count']]
    budget = pointers[execution.controls['steal_budget']]
    state_ints = 8 + 3 * tiles
    ready0, ready1, chunk = 8, 8 + tiles, 8 + 2 * tiles
    major, minor = target.compute_capability
    arch = major * 100 + minor * 10
    lines = [
        '#include <cuda.h>', '#include <cuda_runtime.h>', '#include <cstdint>',
        '#include <new>', '#include <cstring>',
        f'#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != {arch}',
        f'#error "Program requires exact {program.target}"', '#endif',
        '__device__ __forceinline__ int cake_claim(int32_t* pointer) {',
        '  int old;',
        '  asm volatile("atom.relaxed.gpu.global.add.s32 %0, [%1], %2;" : "=r"(old) :',
        '    "l"(reinterpret_cast<unsigned long long>(pointer)), "r"(1) : "memory");',
        '  return old;', '}',
        '__device__ __forceinline__ int cake_relaxed(const int32_t* pointer) {',
        '  int value;',
        '  asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(value) :',
        '    "l"(reinterpret_cast<unsigned long long>(pointer)) : "memory");',
        '  return value;', '}',
        '__device__ __forceinline__ int cake_acquire(const int32_t* pointer) {',
        '  int value;',
        '  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(value) :',
        '    "l"(reinterpret_cast<unsigned long long>(pointer)) : "memory");',
        '  return value;', '}',
        '__device__ __forceinline__ void cake_publish(int32_t* pointer) {',
        '  asm volatile("st.release.gpu.global.s32 [%0], %1;" ::',
        '    "l"(reinterpret_cast<unsigned long long>(pointer)), "r"(1) : "memory");',
        '}',
        '__device__ __forceinline__ int cake_warp_claim(int32_t* pointer, int lane) {',
        '  int old = lane == 0 ? cake_claim(pointer) : 0;',
        '  return __shfl_sync(0xffffffffu, old, 0);', '}',
    ]
    lines.extend(_stage_source(program, index, width, float_names)
                 for index in range(3))
    lines += [
        f'extern "C" __global__ void {entry}_kernel({params}, int32_t* state) {{',
        '  const int lane = int(threadIdx.x);',
        f'  const int c = {c}[0], k = {k}[0], budget = {budget}[0];',
        f'  if (c <= 0 || c >= {grid} || k <= 0 || k > {tiles} || '
        f'{tiles} % k != 0 || budget < 0 || budget > {tiles}) {{',
        '    if (blockIdx.x == 0 && lane == 0) state[7] = -1;',
        '    return;', '  }',
        f'  const int per_chunk = {tiles} / k;',
        '  const bool first_worker = int(blockIdx.x) < c;',
        '  if (first_worker) {',
        '    while (true) {',
        '      int tile = cake_warp_claim(state + 0, lane);',
        f'      if (tile >= {tiles}) break;',
        f'      {entry}_stage0({args}, tile, lane);',
        '      __syncwarp();',
        f'      if (lane == 0) {{ cake_publish(state + {ready0} + tile); '
        'cake_claim(state + 3); }}',
        '      __syncwarp();', '    }',
        f'    if (lane == 0) while (cake_relaxed(state + 3) < {tiles}) __nanosleep(64);',
        '    __syncwarp();',
        '    while (true) {',
        '      int stop = 0;',
        '      if (lane == 0) stop = cake_relaxed(state + '+str(chunk)+') >= per_chunk '
        '|| cake_claim(state + 6) >= budget;',
        '      stop = __shfl_sync(0xffffffffu, stop, 0);',
        '      if (stop) break;',
        '      int tile = cake_warp_claim(state + 1, lane);',
        f'      if (tile >= {tiles}) break;',
        f'      while (cake_acquire(state + {ready0} + tile) == 0) __nanosleep(64);',
        f'      {entry}_stage1({args}, tile, lane);',
        '      __syncwarp();',
        f'      if (lane == 0) {{ cake_publish(state + {ready1} + tile); '
        f'cake_claim(state + {chunk} + tile / per_chunk); cake_claim(state + 4); }}',
        '      __syncwarp();', '    }',
        '    if (lane == 0) cake_claim(state + 5);',
        '    __syncwarp();',
        '  } else {',
        '    while (true) {',
        '      int tile = cake_warp_claim(state + 1, lane);',
        f'      if (tile >= {tiles}) break;',
        f'      while (cake_acquire(state + {ready0} + tile) == 0) __nanosleep(64);',
        f'      {entry}_stage1({args}, tile, lane);',
        '      __syncwarp();',
        f'      if (lane == 0) {{ cake_publish(state + {ready1} + tile); '
        f'cake_claim(state + {chunk} + tile / per_chunk); cake_claim(state + 4); }}',
        '      __syncwarp();', '    }', '  }',
        '  if (lane == 0) while (cake_relaxed(state + 5) < c) __nanosleep(64);',
        '  __syncwarp();',
        '  while (true) {',
        '    int tile = cake_warp_claim(state + 2, lane);',
        f'    if (tile >= {tiles}) break;',
        f'    while (cake_relaxed(state + {chunk} + tile / per_chunk) < per_chunk) '
        '__nanosleep(64);',
        f'    while (cake_acquire(state + {ready1} + tile) == 0) __nanosleep(64);',
        f'    {entry}_stage2({args}, tile, lane);',
        '    __syncwarp();', '  }',
        '}', '// CAKE_KERNEL_END',
    ]
    handle = entry + '_handle'
    lines += [f'struct {handle} {{', '  int device;', '  int32_t* state;']
    for name in tensor_names:
        dtype = 'int32_t' if name in execution.controls.values() else 'float'
        lines.append(f'  {dtype}* {pointers[name]};')
    lines += [
        '};',
        f'extern "C" int {entry}_create(void** buffers, void* state, '
        'size_t state_bytes, void** result) {',
        '  if (!buffers || !state || !result || state_bytes < '
        f'{state_ints * 4}) return int(cudaErrorInvalidValue);',
        '  *result = nullptr;',
        '  int device; cudaDeviceProp prop;',
        '  cudaError_t error = cudaGetDevice(&device);',
        '  if (error != cudaSuccess) return int(error);',
        '  error = cudaGetDeviceProperties(&prop, device);',
        '  if (error != cudaSuccess) return int(error);',
    ]
    names = ' && '.join(f'std::strcmp(prop.name, "{name}") != 0'
                         for name in target.device_names)
    lines.append(f'  if (prop.major != {major} || prop.minor != {minor} || '
                 f'prop.multiProcessorCount != {grid} || ({names})) '
                 'return int(cudaErrorInvalidDevice);')
    lines += [f'  auto* h = new(std::nothrow) {handle};',
              '  if (!h) return int(cudaErrorMemoryAllocation);',
              '  h->device = device; h->state = static_cast<int32_t*>(state);']
    for index, name in enumerate(tensor_names):
        dtype = 'int32_t' if name in execution.controls.values() else 'float'
        lines.append(f'  if (!buffers[{index}]) {{ delete h; return int(cudaErrorInvalidValue); }}')
        lines.append(f'  h->{pointers[name]} = static_cast<{dtype}*>(buffers[{index}]);')
    lines += ['  *result = h; return 0;', '}',
              f'extern "C" int {entry}_launch(void* handle, void* stream) {{',
              '  if (!handle) return int(cudaErrorInvalidValue);',
              f'  auto* h = static_cast<{handle}*>(handle);',
              '  int device; cudaError_t error = cudaGetDevice(&device);',
              '  if (error != cudaSuccess) return int(error);',
              '  if (device != h->device) return int(cudaErrorInvalidDevice);',
              '  int cooperative = 0;',
              '  error = cudaDeviceGetAttribute(&cooperative, '
              'cudaDevAttrCooperativeLaunch, device);',
              '  if (error != cudaSuccess) return int(error);',
              '  if (cooperative != 1) return int(cudaErrorNotSupported);',
              '  int resident_blocks = 0;',
              f'  error = cudaOccupancyMaxActiveBlocksPerMultiprocessor('
              f'&resident_blocks, {entry}_kernel, 32, 0);',
              '  if (error != cudaSuccess) return int(error);',
              '  if (resident_blocks < 1) return int(cudaErrorCooperativeLaunchTooLarge);',
              f'  error = cudaMemsetAsync(h->state, 0, {state_ints * 4}, '
              'reinterpret_cast<cudaStream_t>(stream));',
              '  if (error != cudaSuccess) return int(error);',
    ]
    launch_args = [f'h->{pointers[name]}' for name in tensor_names] + ['h->state']
    lines.append('  void* kernel_args[] = {' + ', '.join('&'+arg for arg in launch_args) + '};')
    lines += [f'  error = cudaLaunchCooperativeKernel(reinterpret_cast<const void*>('
              f'{entry}_kernel), dim3({grid}), dim3(32), kernel_args, 0, '
              'reinterpret_cast<cudaStream_t>(stream));',
              '  return int(error);', '}',
              f'extern "C" int {entry}_destroy(void* handle) {{',
              f'  delete static_cast<{handle}*>(handle); return 0;', '}',
    ]
    requirements = {
        'source_language': 'cuda_cpp', 'target': program.target,
        'entry_point': entry, 'kernel_entry_point': entry + '_kernel',
        'grid': [grid, 1, 1], 'block': [32, 1, 1],
        'cooperative_grid': True, 'state_bytes': state_ints * 4,
        'state_status_offset_bytes': 7 * 4,
        'state_reset': 'zero_before_each_launch_on_launch_stream',
        'argument_order': list(tensor_names),
        'host_abi': {'create': entry+'_create', 'launch': entry+'_launch',
                     'destroy': entry+'_destroy'},
        'nvcc_flags': ['-std=c++17', f'--gpu-architecture=compute_{major}{minor}a',
                       f'--gpu-code={program.target}', '-O3', '--fmad=false', '-lineinfo'],
        'link_libraries': ['cuda', 'cudart'],
    }
    return '\n'.join(lines) + '\n', requirements


def lower_worker_program(compiler, program) -> LoweredWorkerProgram:
    target, tiles, width, grid = _admit(compiler, program)
    if compiler.commit is None:
        raise ValueError('program compilation requires a clean Compiler commit')
    source, requirements = _emit(program, target, tiles, width, grid)
    result = LoweredWorkerProgram(program, compiler._revision.revision_id,
                                  source, requirements)
    result.validate_binding()
    return result
