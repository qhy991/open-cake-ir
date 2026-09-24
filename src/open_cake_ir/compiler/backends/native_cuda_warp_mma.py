"""Bounded B300 TMA plus warp-MMA lowering from declared Schedule commitments.

The K64 TMA boxes, four K256 partials, roles, barriers and output mapping come from the
Schedule. PTX spelling, descriptor encoding and scratch addresses belong to this emitter.
No operator name or target-id dispatch table selects it: the instruction plus ordered
K partitions select this native CUDA mechanism, and preflight refuses unqualified shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

from .common import Emission, EmitError, refusal
from ..diagnostics import Finding
from ..ir import (
    AccessIndexKind, BarrierMechanism, BoundaryPolicy, BufferMode, DType,
    ElementwiseOp, LoadMovement, LoweringBackend, MemorySpace, OperandMajorMode,
    OperandSource, OperationKind, Schedule, Swizzle,
)
from ..target import Target


CONTRACT = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
# Qualification is per exact public fixture. Other shapes need their own bitwise peer
# and paired device evidence before this route may claim them.
SHAPE_EVIDENCE = frozenset({(1, 128, 720), (16, 1024, 1024)})
PARTITIONS = ((0, 256), (256, 512), (512, 768), (768, 1024))
_TEMPLATE = Path(__file__).with_name("native_cuda_warp_mma.cu.in")


@dataclass(frozen=True)
class _Plan:
    schedule: Schedule
    target: Target
    activation: object
    weight: object
    bias: object
    output: object
    activation_stage: object
    weight_stage: object
    allocation: object
    load_activation: object
    load_weight: object
    load_bias: object
    mma: object
    bias_cast: object
    add: object
    output_cast: object
    store: object
    partial_offset: int
    barrier_offset: int
    shared_bytes: int


def applies(schedule: Schedule) -> bool:
    return any(op.kind is OperationKind.MMA and op.parameters.k_partitions is not None
               for op in schedule.operations)


def _plan(schedule: Schedule, target: Target) -> tuple[_Plan | None, tuple[Finding, ...]]:
    findings: list[Finding] = []

    def check(condition: object, code: str, path: str, message: str) -> None:
        if not condition:
            findings.append(refusal(code, path, message))

    check(schedule.target == target.target_id and target.target_id == "sm_103a"
          and schedule.lowering.backend is LoweringBackend.NATIVE_CUDA,
          "NATIVE_WARP_TARGET", "target",
          "the TMA warp-MMA route is qualified only for native_cuda on exact sm_103a")
    check(schedule.residency is None and schedule.grid is None and not schedule.tile_loops,
          "NATIVE_WARP_SCOPE", "tile_loops",
          "this route owns one complete K1024 tile per CTA without a carried loop or residency cap")
    check(len(schedule.operations) == 8 and tuple(op.kind for op in schedule.operations) == (
        OperationKind.LOAD, OperationKind.LOAD, OperationKind.LOAD, OperationKind.MMA,
        OperationKind.CAST, OperationKind.ELEMENTWISE, OperationKind.CAST, OperationKind.STORE),
        "NATIVE_WARP_DAG", "operations",
        "this route requires three loads, one partitioned MMA, bias cast/add, output cast and store")
    check(len(schedule.roles) == 3 and tuple(role.execution_groups for role in schedule.roles) == (
        (0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11))
        and all(role.registers_per_thread is None for role in schedule.roles),
        "NATIVE_WARP_ROLES", "roles",
        "four compute and two four-warp TMA producer roles must be explicitly declared")
    check(len(schedule.pipelines) == 1 and schedule.pipelines[0].stages == 1,
          "NATIVE_WARP_PIPELINE", "pipelines",
          "the fixed short-K route has one complete TMA/MMA stage")
    check(len(schedule.allocations) == 1
          and schedule.allocations[0].space is MemorySpace.SHARED,
          "NATIVE_WARP_STORAGE", "allocations",
          "one declared shared allocation must own both TMA operands")
    if findings:
        return None, tuple(findings)

    load_bias, load_a, load_b, mma, bias_cast, add, output_cast, store = schedule.operations
    arities = (
        (load_bias, 1, 1), (load_a, 1, 1), (load_b, 1, 1), (mma, 2, 1),
        (bias_cast, 1, 1), (add, 2, 1), (output_cast, 1, 1), (store, 1, 1),
    )
    if any(len(op.reads) != reads or len(op.writes) != writes
           for op, reads, writes in arities):
        return None, (refusal("NATIVE_WARP_ARITY", "operations",
                              "every operation must preserve the declared load/MMA/epilogue arity"),)
    activation_stage, weight_stage = (schedule.buffer(name) for name in mma.reads)
    activation = schedule.buffer(load_a.reads[0]) if len(load_a.reads) == 1 else None
    weight = schedule.buffer(load_b.reads[0]) if len(load_b.reads) == 1 else None
    bias = schedule.buffer(load_bias.reads[0]) if len(load_bias.reads) == 1 else None
    output = schedule.buffer(store.writes[0]) if len(store.writes) == 1 else None
    if any(value is None for value in (activation_stage, weight_stage, activation, weight, bias, output)):
        return None, (refusal("NATIVE_WARP_DATAFLOW", "operations",
                              "every operand, bias and output must resolve to a declared Buffer"),)
    assert activation_stage is not None and weight_stage is not None
    assert activation is not None and weight is not None and bias is not None and output is not None
    batch, depth = activation.shape if len(activation.shape) == 2 else (0, 0)
    features = weight.shape[0] if len(weight.shape) == 2 else 0
    check((batch, features, depth) in SHAPE_EVIDENCE,
          "NATIVE_WARP_SHAPE_EVIDENCE", "buffers",
          "this route has B300 bitwise and paired evidence only for the declared short-K fixtures")
    check(weight.shape == (features, depth) and bias.shape == (features,)
          and output.shape == (batch, features),
          "NATIVE_WARP_GLOBAL_SHAPES", "buffers",
          "the global contract is activation[B,K], weight[N,K], bias[N], output[B,N]")
    check(all(buffer.space is MemorySpace.GLOBAL and buffer.dtype is DType.BF16
              for buffer in (activation, weight, bias, output))
          and all(buffer.mode is BufferMode.INPUT for buffer in (activation, weight, bias))
          and output.mode is BufferMode.OUTPUT and schedule.outputs == (output.name,),
          "NATIVE_WARP_GLOBAL_ABI", "buffers",
          "the exact BF16 input/output modes and sole public output must be declared")
    allocation = schedule.allocations[0]
    check(all(stage.space is MemorySpace.SHARED and stage.dtype is DType.BF16
              and stage.mode is BufferMode.SCRATCH and stage.allocation == allocation.name
              and stage.stages == 1 and stage.swizzle is Swizzle.B128
              for stage in (activation_stage, weight_stage))
          and activation_stage.shape == (16, 1024)
          and weight_stage.shape == (8, 1024),
          "NATIVE_WARP_STAGES", "buffers",
          "K1024 shared operands must declare 16x1024 and 8x1024 BF16 views with 128B swizzle")
    a_end = activation_stage.byte_offset + activation_stage.elements * 2
    b_end = weight_stage.byte_offset + weight_stage.elements * 2
    check(activation_stage.byte_offset >= 0 and weight_stage.byte_offset >= 0
          and activation_stage.byte_offset % 128 == weight_stage.byte_offset % 128 == 0
          and (a_end <= weight_stage.byte_offset or b_end <= activation_stage.byte_offset)
          and max(a_end, b_end) <= allocation.size_bytes,
          "NATIVE_WARP_STAGE_BOUNDS", "allocations",
          "TMA views must be nonoverlapping, 128B aligned and inside their declared allocation")
    check(load_a.writes == (activation_stage.name,) and load_b.writes == (weight_stage.name,)
          and load_a.role == schedule.roles[2].name and load_b.role == schedule.roles[1].name
          and all(op.parameters.movement is LoadMovement.TMA for op in (load_a, load_b))
          and load_a.parameters.descriptor_box == (16, 64)
          and load_b.parameters.descriptor_box == (8, 64)
          and load_a.pipeline == load_b.pipeline == mma.pipeline == schedule.pipelines[0].name,
          "NATIVE_WARP_TMA", "operations[1]",
          "producer roles must fill the declared shared tiles with repeated K64 TMA boxes")
    instruction = mma.parameters.instruction
    check(mma.role == schedule.roles[0].name
          and mma.parameters.accumulator is DType.FP32
          and mma.parameters.tile_shape == (16, 8, 1024)
          and mma.parameters.k_partitions == PARTITIONS
          and instruction is not None and instruction.contract == CONTRACT
          and instruction.shape == (16, 8, 16) and instruction.cta_group == 1
          and instruction.operand_source is OperandSource.REGISTER
          and instruction.operand_major == (OperandMajorMode.K, OperandMajorMode.K),
          "NATIVE_WARP_MMA", "operations[3].parameters",
          "four ordered K256 partials must use the declared BF16 register MMA atom")
    acc = schedule.buffer(mma.writes[0]) if len(mma.writes) == 1 else None
    bias_tile = schedule.buffer(load_bias.writes[0]) if len(load_bias.writes) == 1 else None
    bias32 = schedule.buffer(bias_cast.writes[0]) if len(bias_cast.writes) == 1 else None
    summed = schedule.buffer(add.writes[0]) if len(add.writes) == 1 else None
    rounded = schedule.buffer(output_cast.writes[0]) if len(output_cast.writes) == 1 else None
    check(acc is not None and acc.space is MemorySpace.REGISTER and acc.dtype is DType.FP32
          and acc.shape == (16, 8)
          and bias_tile is not None and bias_tile.space is MemorySpace.REGISTER
          and bias_tile.dtype is DType.BF16 and bias_tile.shape == (8,)
          and bias32 is not None and bias32.space is MemorySpace.REGISTER
          and bias32.dtype is DType.FP32 and bias32.shape == (8,)
          and summed is not None and summed.space is MemorySpace.REGISTER
          and summed.dtype is DType.FP32 and summed.shape == (16, 8)
          and rounded is not None and rounded.space is MemorySpace.REGISTER
          and rounded.dtype is DType.BF16 and rounded.shape == (16, 8),
          "NATIVE_WARP_REGISTER_VALUES", "buffers",
          "bias, partial sum and rounded output need their declared register types and shapes")
    check(load_bias.writes == bias_cast.reads
          and bias_cast.writes[0] == add.reads[1]
          and mma.writes[0] == add.reads[0]
          and add.writes == output_cast.reads
          and output_cast.writes == store.reads
          and load_bias.role == bias_cast.role == add.role == output_cast.role == store.role == mma.role
          and load_bias.parameters.movement is LoadMovement.GLOBAL
          and bias_cast.parameters.to is DType.FP32
          and add.parameters.op is ElementwiseOp.ADD and add.parameters.broadcast_axis == 1
          and add.parameters.scalar is None and add.parameters.instruction is None
          and output_cast.parameters.to is DType.BF16
          and store.parameters.coalesced is False,
          "NATIVE_WARP_EPILOGUE", "operations[4]",
          "the explicit FP32 bias addition, BF16 cast and noncoalesced store must be preserved")
    check(all(buffer.valid_extent is None and buffer.scale_of is None for buffer in schedule.buffers)
          and len(schedule.buffers) == 11,
          "NATIVE_WARP_BUFFER_OPTIONS", "buffers",
          "the bounded route supports only the eleven declared dense buffers")
    if findings:
        return None, tuple(findings)

    assert acc is not None and bias_tile is not None and bias32 is not None
    assert summed is not None and rounded is not None
    pm = schedule.program_map
    check(pm is not None and not pm.persistent and len(pm.axes) == 3,
          "NATIVE_WARP_PROGRAM_MAP", "program_map",
          "batch, feature and K axes must derive the exact CTA grid")
    if pm is not None and len(pm.axes) == 3:
        axes = sorted(pm.axes, key=lambda axis: axis.axis)
        check([(axis.buffer, axis.dimension, axis.tile) for axis in axes] == [
            (activation.name, 0, 16), (weight.name, 0, 8), (activation.name, 1, 1024)],
            "NATIVE_WARP_PROGRAM_MAP", "program_map.axes",
            "axis 0 tiles batch16, axis 1 tiles feature8 and axis 2 covers K1024")
    expected_access = (
        (load_a, activation, ("batch", "k")),
        (load_b, weight, ("feature", "k")),
        (load_bias, bias, ("feature",)),
        (store, output, ("batch", "feature")),
    )
    check(len(schedule.access_maps) == 4, "NATIVE_WARP_ACCESS", "access_maps",
          "each global load and the public store need one explicit access map")
    for operation, buffer, names in expected_access:
        access = schedule.access_map(operation.op_id, buffer.name)
        check(access is not None and access.boundary is BoundaryPolicy.MASK_TILED_AXES
              and tuple((index.source, index.name) for index in access.indices) == tuple(
                  (AccessIndexKind.PROGRAM_TILE, name) for name in names),
              "NATIVE_WARP_ACCESS", f"access_maps.{operation.op_id}",
              "the global address must follow its declared batch, feature and K tile axes")
    check(len(schedule.barriers) == 9 and len(load_a.signals) == len(load_b.signals) == 4
          and mma.waits == load_a.signals + load_b.signals
          and len(mma.signals) == 1 and add.waits == mma.signals,
          "NATIVE_WARP_SYNC", "barriers",
          "each producer needs four ordered ready barriers and the four compute warps need one named merge")
    barriers = {barrier.name: barrier for barrier in schedule.barriers}
    for load in (load_a, load_b):
        for name in load.signals:
            barrier = barriers.get(name)
            check(barrier is not None and barrier.mechanism is BarrierMechanism.MBARRIER
                  and barrier.count == 1 and barrier.pipeline == schedule.pipelines[0].name
                  and barrier.producers == (load.role,) and barrier.consumers == (mma.role,),
                  "NATIVE_WARP_READY_BARRIER", f"barriers.{name}",
                  "one K256 producer slot must publish through one mbarrier")
    partial = barriers.get(mma.signals[0]) if len(mma.signals) == 1 else None
    check(partial is not None and partial.mechanism is BarrierMechanism.NAMED
          and partial.count == 128 and partial.pipeline is None
          and partial.producers == partial.consumers == (mma.role,),
          "NATIVE_WARP_MERGE_BARRIER", "barriers",
          "four compute warps must meet at the declared 128-thread named barrier")
    if findings:
        return None, tuple(findings)
    partial_offset = (allocation.size_bytes + 15) // 16 * 16
    barrier_offset = (partial_offset + 4 * 32 * 4 * 4 + 7) // 8 * 8
    shared_bytes = (barrier_offset + 8 * 8 + 1023) // 1024 * 1024
    check(shared_bytes <= target.resource_limits.maximum_shared_memory_bytes,
          "NATIVE_WARP_SHARED_CAP", "allocations",
          "operand, merge and barrier storage must fit the Target's declared shared-memory cap")
    if findings:
        return None, tuple(findings)
    return _Plan(schedule, target, activation, weight, bias, output, activation_stage,
                 weight_stage, allocation, load_a, load_b, load_bias, mma, bias_cast,
                 add, output_cast, store, partial_offset, barrier_offset, shared_bytes), ()


def preflight(schedule: Schedule, target: Target) -> tuple[Finding, ...]:
    return _plan(schedule, target)[1]


def emit(schedule: Schedule, target: Target, *, entry_point: str | None = None) -> Emission:
    plan, findings = _plan(schedule, target)
    if findings or plan is None:
        raise EmitError('; '.join(f'{f.code} at {f.path}: {f.message}' for f in findings))
    entry = entry_point or schedule.lowering.entry_point
    if entry != schedule.lowering.entry_point or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', entry):
        raise EmitError('native warp MMA requires the canonical C entry point')
    batch, depth = plan.activation.shape
    features = plan.weight.shape[0]
    globals_ = [buffer for buffer in schedule.buffers if buffer.space is MemorySpace.GLOBAL]
    names = target.device_names
    device_refusal = ' && '.join(
        f'std::strcmp(properties.name, {json.dumps(name)}) != 0'
        for name in names
    )
    substitutions = {
        'ENTRY': entry,
        'A_OFFSET': str(plan.activation_stage.byte_offset),
        'B_OFFSET': str(plan.weight_stage.byte_offset),
        'PARTIAL_OFFSET': str(plan.partial_offset),
        'BARRIER_OFFSET': str(plan.barrier_offset),
        'SHARED_BYTES': str(plan.shared_bytes),
        'CUDA_ARCH': str(target.compute_capability[0] * 100 + target.compute_capability[1] * 10),
        'INPUT_INDEX': str(globals_.index(plan.activation)),
        'WEIGHT_INDEX': str(globals_.index(plan.weight)),
        'BIAS_INDEX': str(globals_.index(plan.bias)),
        'OUTPUT_INDEX': str(globals_.index(plan.output)),
        'MAJOR': str(target.compute_capability[0]),
        'MINOR': str(target.compute_capability[1]),
        'DEVICE_NAME_REFUSAL': device_refusal,
        'BATCH': str(batch), 'FEATURES': str(features), 'DEPTH': str(depth),
        'GRID_X': str(math.ceil(batch / 16)), 'GRID_Y': str(math.ceil(features / 8)),
        'LOAD_A_OP': plan.load_activation.op_id, 'LOAD_B_OP': plan.load_weight.op_id,
        'MMA_OP': plan.mma.op_id, 'BIAS_OP': plan.load_bias.op_id,
        'ADD_OP': plan.add.op_id, 'CAST_OP': plan.output_cast.op_id,
        'STORE_OP': plan.store.op_id,
    }
    source = _TEMPLATE.read_text()
    for key, value in substitutions.items():
        token = f'@{key}@'
        if token not in source:
            raise EmitError(f'native warp MMA template has no {token} slot')
        source = source.replace(token, value)
    if re.search(r'@[A-Z_]+@', source):
        raise EmitError('native warp MMA template has an unbound slot')
    major, minor = target.compute_capability
    flags = ['-std=c++17', f'--gpu-architecture=compute_{major}{minor}a',
             f'--gpu-code=sm_{major}{minor}a', '-O3', '--fmad=false', '-lineinfo', '-Xptxas=-v']
    requirements = {
        'kernel_entry_point': entry + '_kernel',
        'threads_per_cta': 12 * target.warp_size,
        'dynamic_shared_bytes': plan.shared_bytes,
        'grid': [math.ceil(batch / 16), math.ceil(features / 8), 1],
        'nvcc_flags': flags,
        'argument_order': [buffer.name for buffer in globals_],
        'arguments': [dict(name=buffer.name, dtype=buffer.dtype.value, shape=list(buffer.shape),
                           mode=buffer.mode.value) for buffer in globals_],
        'host_abi': dict(create=entry + '_create', launch=entry + '_launch',
                         destroy=entry + '_destroy'),
        'link_libraries': ['cuda', 'cudart'],
        'target_device_names': list(target.device_names),
        'register_mapping': 'four ordered K256 warp partials, named 128-thread merge',
        'source_language': 'cuda_cpp',
        'signature': {buffer.name: '*' + buffer.dtype.value for buffer in globals_},
        'block': [12 * target.warp_size, 1, 1],
    }
    return Emission(source, entry, {'shared_bytes': plan.shared_bytes}, requirements)
