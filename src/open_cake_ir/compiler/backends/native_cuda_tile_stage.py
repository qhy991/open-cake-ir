"""Schedule-driven tensor-core stage body for a persistent CUDA worker.

The worker owns task claims and the lifetime of TMEM. This emitter owns the
TMA/MMA/readout/store body and the shared/barrier offsets, using the same
lowering routines as a standalone native CUDA Schedule. Its exact admission
scope is the evidenced B300 model-width FFN tile.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from . import native_cuda
from .common import EmitError
from .native_cuda_activation import ModelEmitter
from ..ir import (AccessIndexKind, BoundaryPolicy, DType, LoadMovement,
                  LoweringBackend, MemorySpace, OperationKind, Program,
                  RankedTileEffects, Schedule)
from ..target import CodeObject, Target


_IDENTIFIER = re.compile(r'[A-Za-z_][A-Za-z_0-9]*\Z')
_ROUTE_EVIDENCE = frozenset({'sm_103a'})


class _WorkerTensorEmitter(native_cuda._Emitter):
    """Use the caller's checked N-subtile range for one complete store tile."""

    def __init__(self, schedule, target, entry, *, output_width: int):
        super().__init__(schedule, target, entry)
        self.output_width = output_width

    def operation(self, op):
        if op.kind is not OperationKind.STORE:
            return super().operation(op)
        source = self.b(op.reads[0])
        output = self.b(op.writes[0])
        self.line(f'// CAKE_OP: {op.op_id}')
        self.begin(f'if ({self.role_condition(op.role)})')
        self.line('#pragma unroll')
        self.begin(f'for (int col=0; col<{native_cuda._slots(self.s, source)}; ++col)')
        self.line(f'{self.names[output.name]}[{self.row(op)} * '
                  f'{self.output_width} + n_tile * 64 + col] = '
                  f'{self.names[source.name]}[col];')
        self.end()
        self.end()


@dataclass(frozen=True)
class TileStageEmission:
    source: str
    instruction_helpers: str
    function_name: str
    dynamic_shared_bytes: int
    tmem_columns: int
    mapped_operations: tuple[str, ...]


@dataclass(frozen=True)
class RankedTileStageComposition:
    """Cake math and safe queue bounds for the evidenced B300 tile worker.

    This is a stage composition, not a ranked-tile executable lowering. The
    caller still owes runtime route admission, peer ownership, tile publication,
    worker launch and source-keyed return before a complete Program exists.
    """

    stages: tuple[TileStageEmission, ...]
    stage_work_units: tuple[tuple[str, int], ...]
    safe_logical_tile_slots_per_rank: int
    safe_stage_task_slots_per_rank: int
    declared_shared_bytes: int
    emitted_shared_bytes: int
    tensor_bytes: int
    required_execution_groups: int
    target_id: str
    compute_capability: tuple[int, int]
    device_names: tuple[str, ...]
    warp_size: int
    target_multiprocessors: int


def emit_tensor_tile_stage(schedule, target, *, function_name: str) -> TileStageEmission:
    """Emit one CTA work unit from a complete Cake TMA/MMA Schedule.

    ``maps_a`` addresses the current logical tile's descriptor and ``map_b``
    the selected expert weight descriptor. The caller owns both bindings,
    task publication, and one TMEM allocation for the persistent CTA.
    """
    if (_IDENTIFIER.fullmatch(function_name) is None):
        raise EmitError('worker stage function needs an ASCII C identifier')
    if (schedule.target != target.target_id or target.target_id not in _ROUTE_EVIDENCE
            or schedule.lowering.backend is not LoweringBackend.NATIVE_CUDA):
        raise EmitError('worker tensor tile requires the evidenced exact B300 native CUDA route')
    for check in (native_cuda.verify, native_cuda.preflight):
        failures = [finding for finding in check(schedule, target)
                    if finding.blocks_lowering or finding.blocks_acceptance]
        if failures:
            raise EmitError('; '.join(f'{f.code} at {f.path}: {f.message}'
                                      for f in failures))
    mapping = schedule.program_map
    if (mapping is None or mapping.persistent or len(mapping.axes) != 2
            or tuple(axis.tile for axis in mapping.axes) != (128, 64)
            or tuple(axis.axis for axis in mapping.axes) != (0, 1)):
        raise EmitError('worker tensor tile needs M128/N64 finite program axes')
    globals_ = [b for b in schedule.buffers if b.space is MemorySpace.GLOBAL]
    loads = [op for op in schedule.operations if op.kind is OperationKind.LOAD
             and op.parameters.movement is LoadMovement.TMA]
    other = [op for op in schedule.operations if op not in loads]
    if (len(globals_) != 3 or len(loads) != 2
            or tuple(op.kind for op in other)
            != (OperationKind.MMA, OperationKind.LOAD, OperationKind.STORE)
            or len(schedule.outputs) != 1):
        raise EmitError('worker tensor tile needs exactly two TMA loads, MMA, TMEM read and store')
    a = schedule.buffer(loads[0].reads[0])
    b = schedule.buffer(loads[1].reads[0])
    output = schedule.buffer(schedule.outputs[0])
    if (a not in globals_ or b not in globals_ or output not in globals_
            or a.dtype is not DType.BF16 or b.dtype is not DType.BF16
            or output.dtype is not DType.FP32
            or len(a.shape) != 2 or len(b.shape) != 2
            or a.shape[0] != 128 or b.shape[1] != a.shape[1]
            or output.shape != (128, b.shape[0])
            or b.shape[0] % 64 or a.shape[1] % 64):
        raise EmitError('worker tensor tile BF16 M128/K/N global shapes differ')
    store = other[-1]
    store_input = schedule.buffer(store.reads[0])
    access = schedule.access_map(store.op_id, output.name)
    epilogue = next(role for role in schedule.roles if role.name == store.role)
    if (store_input.shape != (128, 64)
            or epilogue.execution_groups != (0, 1, 2, 3)
            or access is None or access.boundary is not BoundaryPolicy.MASK_TILED_AXES
            or len(access.indices) != 2
            or tuple((item.source, item.name, item.offset)
                     for item in access.indices)
            != ((AccessIndexKind.PROGRAM_TILE, mapping.axes[0].name, 0),
                (AccessIndexKind.PROGRAM_TILE, mapping.axes[1].name, 0))):
        raise EmitError('worker unmasked store needs full row/column tile ownership')
    tensor = [alloc for alloc in schedule.allocations
              if alloc.space is MemorySpace.TENSOR]
    if len(tensor) != 1 or tensor[0].tensor_columns != 64:
        raise EmitError('worker tensor tile needs one externally owned 64-column TMEM allocation')
    emitter = _WorkerTensorEmitter(schedule, target, function_name,
                                   output_width=output.shape[1])
    if emitter.shared_bytes > 49200:
        raise EmitError('worker tensor tile exceeds the evidenced CTA shared-memory footprint')
    emitter.axisvars = {mapping.axes[0].name: '0', mapping.axes[1].name: 'n_tile'}
    emitter.mapnames = {loads[0].op_id: 'maps_a[0]',
                        loads[1].op_id: 'map_b[0]'}
    emitter.tmemvars = {tensor[0].name: 'tm1'}
    emitter.names[output.name] = 'output'
    emitter.begin(f'__device__ __forceinline__ void {function_name}('
                  'unsigned char* smem, uint32_t* tm1, int warp, float* output, '
                  'const CUtensorMap* maps_a, const CUtensorMap* map_b, int n_tile)')
    emitter.line(f'if (n_tile < 0 || n_tile >= {output.shape[1] // 64}) '
                 'asm volatile("trap;");')
    for barrier in schedule.barriers:
        key = 'barrier:' + barrier.name
        emitter.line(f'uint64_t* {emitter.barvars[barrier.name]} = '
                     f'reinterpret_cast<uint64_t*>(smem + {emitter.offsets[key]});')
    for index, pipeline in enumerate(schedule.pipelines):
        key = 'free:' + pipeline.name
        emitter.line(f'uint64_t* free{index} = '
                     f'reinterpret_cast<uint64_t*>(smem + {emitter.offsets[key]});')
    for register in (b for b in schedule.buffers if b.space is MemorySpace.REGISTER):
        slots = native_cuda._slots(schedule, register)
        emitter.line(f'{native_cuda._TYPES[register.dtype]} '
                     f'{emitter.names[register.name]}[{slots}];')
    emitter.sequence(None)
    emitter.line('__syncthreads();')
    emitter.invalidate_completions(None)
    emitter.end()
    source = '\n'.join(emitter.lines) + '\n'
    mapped = tuple(line.split('// CAKE_OP: ', 1)[1].strip()
                   for line in emitter.lines if '// CAKE_OP: ' in line)
    required = tuple(op.op_id for op in schedule.operations)
    if mapped != required:
        raise EmitError('worker stage source map differs from its Schedule operations')
    return TileStageEmission(source, native_cuda._INSTRUCTIONS,
                             function_name, emitter.shared_bytes,
                             tensor[0].tensor_columns, mapped)


def emit_model_activation_stage(schedule, target, *, function_name: str) -> TileStageEmission:
    """Emit the model SwiGLU row from its Cake Schedule inside the worker."""
    if _IDENTIFIER.fullmatch(function_name) is None:
        raise EmitError('worker activation function needs an ASCII C identifier')
    if (schedule.target != target.target_id
            or target.target_id not in _ROUTE_EVIDENCE
            or schedule.lowering.backend is not LoweringBackend.NATIVE_CUDA):
        raise EmitError('worker activation needs the evidenced exact B300 native CUDA route')
    for check in (native_cuda.verify, native_cuda.preflight):
        failures = [finding for finding in check(schedule, target)
                    if finding.blocks_lowering or finding.blocks_acceptance]
        if failures:
            raise EmitError('; '.join(f'{f.code} at {f.path}: {f.message}'
                                      for f in failures))
    emitter = ModelEmitter(schedule, target, function_name)
    globals_ = [b for b in schedule.buffers if b.space is MemorySpace.GLOBAL]
    if (len(globals_) != 2 or len(schedule.program_map.axes) != 1
            or globals_[0].shape != (128, 1536)
            or globals_[1].shape != (128, 768)):
        raise EmitError('worker activation needs the exact model-width row domain')
    emitter.axisvars = {schedule.program_map.axes[0].name: 'row_tile'}
    emitter.names[globals_[0].name] = 'tile_input'
    emitter.names[globals_[1].name] = 'tile_output'
    emitter.begin(f'__device__ __forceinline__ void {function_name}('
                  'const float* up_gate, __nv_bfloat16* activated, '
                  'int logical_tile, int row_tile)')
    emitter.line('const float* tile_input = up_gate + logical_tile * 128 * 1536;')
    emitter.line('__nv_bfloat16* tile_output = activated + logical_tile * 128 * 768;')
    emitter.line('const int cake_warp_row = int(threadIdx.x) / 32;')
    emitter.begin('if (cake_warp_row < 6 && row_tile * 6 + cake_warp_row < 128)')
    emitter.line('const int cake_lane = int(threadIdx.x) % 32;')
    emitter.begin('for (int cake_feature=cake_lane; cake_feature<768; cake_feature+=32)')
    emitter.emit_model_graph()
    emitter.end()
    emitter.line('asm volatile("fence.proxy.async.global;" ::: "memory");')
    emitter.end()
    emitter.line('__syncthreads();')
    emitter.end()
    source = '\n'.join(emitter.lines) + '\n'
    mapped = tuple(line.split('// CAKE_OP: ', 1)[1].strip()
                   for line in emitter.lines if '// CAKE_OP: ' in line)
    if mapped != tuple(op.op_id for op in schedule.operations):
        raise EmitError('worker activation source map differs from its Schedule operations')
    return TileStageEmission(source, '', function_name, 0, 0, mapped)


def compose_model_ranked_tile_stages(
        effects: RankedTileEffects, local_program: Program,
        combine_schedule: Schedule, target: Target) -> RankedTileStageComposition:
    """Bind schema-2 effects to the three evidenced Cake FFN stage bodies.

    The 255-tile bound is intentionally retained beside the 64-tile synthetic
    experiment. A future complete backend must either allocate the safe bound
    or make a checked runtime route-plan admission part of its launch ABI.
    """
    effects = RankedTileEffects.from_dict(effects.document)
    local_program = Program.from_dict(local_program.document)
    if (target.target_id != 'sm_103a' or target.code_object is not CodeObject.CUBIN
            or target.compute_capability != (10, 3)
            or target.warp_size != 32 or target.cooperative_grid is not True
            or target.occupancy is None):
        raise EmitError('ranked tile stages require exact observed B300 resources')
    if (effects.lowering.backend is not LoweringBackend.NATIVE_CUDA
            or effects.lowering.entry_point != 'cake_ranked_tile_b300'
            or (effects.world_size, effects.experts, effects.tile_rows,
                effects.maximum_chunks_per_rank, effects.partial_threshold_rows)
            != (4, 128, 128, 4, 64)
            or local_program.target != target.target_id
            or combine_schedule.target != target.target_id
            or tuple(stage.name for stage in local_program.stages)
            != ('up_gate', 'activation', 'down')):
        raise EmitError('ranked tile stage composition needs the exact model EP4 domain')
    analysis = effects.analyze(local_program, combine_schedule)
    if (analysis.items_per_rank != 512 or analysis.routes_per_item != 8
            or analysis.feature_width != 2048
            or analysis.stage_work_units
            != (('up_gate', 24), ('activation', 22), ('down', 32))
            or analysis.logical_tile_slots_per_rank != 255
            or analysis.stage_task_slots_per_rank != 19890
            or analysis.required_execution_groups != 6):
        raise EmitError('ranked tile safe capacity or Cake stage work units differ')
    from .native_cuda_model_combine import preflight as combine_preflight
    failures = [finding for check in (native_cuda.verify, combine_preflight)
                for finding in check(combine_schedule, target)
                if finding.blocks_lowering or finding.blocks_acceptance]
    if failures:
        raise EmitError('; '.join(f'{f.code} at {f.path}: {f.message}'
                                  for f in failures))
    stages = tuple(
        (emit_model_activation_stage if index == 1 else emit_tensor_tile_stage)(
            stage.schedule, target, function_name=function_name)
        for index, (stage, function_name) in enumerate(zip(
            local_program.stages,
            ('cake_upgate_stage_work', 'cake_activation_stage_work',
             'cake_down_stage_work'), strict=True))
    )
    if stages[0].instruction_helpers != stages[2].instruction_helpers:
        raise EmitError('ranked tile tensor stages disagree on PTX helpers')
    shared = max(stage.dynamic_shared_bytes for stage in stages)
    if (shared < analysis.maximum_shared_bytes
            or shared > target.resource_limits.maximum_shared_memory_bytes
            or analysis.maximum_tensor_bytes
            > (target.resource_limits.maximum_tensor_memory_bytes or 0)):
        raise EmitError('ranked tile emitted worker exceeds declared B300 resources')
    return RankedTileStageComposition(
        stages, analysis.stage_work_units,
        analysis.logical_tile_slots_per_rank,
        analysis.stage_task_slots_per_rank,
        analysis.maximum_shared_bytes, shared,
        analysis.maximum_tensor_bytes,
        analysis.required_execution_groups,
        target.target_id,
        target.compute_capability,
        target.device_names,
        target.warp_size,
        target.occupancy.multiprocessor_count)
