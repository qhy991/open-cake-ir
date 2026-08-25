"""Emit one finite Apple Metal Flash-KMeans specialist from a Schedule.

This is intentionally not a generic MSL framework.  It is the first real Metal Adapter:
the fixed four-buffer Workload shape is narrow, while the token/centroid macro tiles and
SIMDgroup count are Schedule decisions that change work division and emitted source.
The 256x64 logical score tile is streamed through one 8x8 FP32 fragment per SIMDgroup;
it is never materialized as a 64 KiB threadgroup allocation.
"""

from __future__ import annotations

from .emit import Emission, EmissionConstraint, EmitError
from .ir import (
    AccessIndexKind,
    ArgminTieBreak,
    BoundaryPolicy,
    BufferMode,
    DType,
    ElementwiseOp,
    LoadMovement,
    LoadParameters,
    MemorySpace,
    MmaParameters,
    NaNPolicy,
    Operation,
    OperationKind,
    Schedule,
    StoreParameters,
)
from .target import InstructionPlacement, InstructionShape, Target


METAL_ARCHITECTURE = "apple_gpu_family9"
METAL_MMA_CONTRACT = "metal.simdgroup_mma.m8n8k8.bf16_bf16_fp32"
METAL_SYNCHRONIZATION_CONTRACT = "simdgroup_barrier"
SUPPORTED_DTYPES = frozenset({DType.BF16, DType.FP32, DType.INT32})
SUPPORTED_OPERATION_KINDS = frozenset(
    {
        OperationKind.LOAD,
        OperationKind.MMA,
        OperationKind.ELEMENTWISE,
        OperationKind.REDUCE_ARGMIN,
        OperationKind.STORE,
    }
)

_EXPECTED_ACCESS_MAPS = {
    ("load_tokens", "tokens"): (
        (
            (AccessIndexKind.PROGRAM, "batch", None),
            (AccessIndexKind.PROGRAM_TILE, "token_block", None),
            (AccessIndexKind.DIMENSION, None, 2),
        ),
        BoundaryPolicy.MASK_TILED_AXES,
    ),
    ("load_centroids", "centroids"): (
        (
            (AccessIndexKind.PROGRAM, "batch", None),
            (AccessIndexKind.LOOP_TILE, "centroid_start", None),
            (AccessIndexKind.DIMENSION, None, 2),
        ),
        BoundaryPolicy.MASK_TILED_AXES,
    ),
    ("load_norm", "centroid_sq"): (
        (
            (AccessIndexKind.PROGRAM, "batch", None),
            (AccessIndexKind.LOOP_TILE, "centroid_start", None),
        ),
        BoundaryPolicy.MASK_TILED_AXES,
    ),
    ("store_assignment", "assignments"): (
        (
            (AccessIndexKind.PROGRAM, "batch", None),
            (AccessIndexKind.PROGRAM_TILE, "token_block", None),
        ),
        BoundaryPolicy.MASK_TILED_AXES,
    ),
}

_EXPECTED_DEPENDENCIES = {
    "load_tokens": (),
    "load_centroids": (),
    "load_norm": (),
    "distance_mma": ("load_tokens", "load_centroids"),
    "scale_cross": ("distance_mma",),
    "distance": ("scale_cross", "load_norm"),
    "argmin": ("distance_mma",),
    "store_assignment": ("argmin",),
}


def _constraint(code: str, path: str, message: str) -> EmissionConstraint:
    return EmissionConstraint(code, path, message)


def _operation(
    schedule: Schedule,
    kind: OperationKind,
    *,
    reads: tuple[str, ...] | None = None,
    writes: tuple[str, ...] | None = None,
) -> Operation | None:
    matches = [
        item
        for item in schedule.operations
        if item.kind is kind
        and (reads is None or item.reads == reads)
        and (writes is None or item.writes == writes)
    ]
    return matches[0] if len(matches) == 1 else None


def _shape(schedule: Schedule, name: str) -> tuple[int, ...] | None:
    buffer = schedule.buffer(name)
    return None if buffer is None else buffer.shape


def _topology(schedule: Schedule) -> dict[str, Operation] | None:
    operations = {
        "load_tokens": _operation(
            schedule,
            OperationKind.LOAD,
            reads=("tokens",),
            writes=("token_tile",),
        ),
        "load_centroids": _operation(
            schedule,
            OperationKind.LOAD,
            reads=("centroids",),
            writes=("centroid_tile",),
        ),
        "load_norm": _operation(
            schedule,
            OperationKind.LOAD,
            reads=("centroid_sq",),
            writes=("norm_tile",),
        ),
        "mma": _operation(
            schedule,
            OperationKind.MMA,
            reads=("token_tile", "centroid_tile"),
            writes=("cross",),
        ),
        "scale": _operation(
            schedule,
            OperationKind.ELEMENTWISE,
            reads=("cross",),
            writes=("scaled_cross",),
        ),
        "distance": _operation(
            schedule,
            OperationKind.ELEMENTWISE,
            reads=("norm_tile", "scaled_cross"),
            writes=("distance_tile",),
        ),
        "argmin": _operation(
            schedule,
            OperationKind.REDUCE_ARGMIN,
            reads=("distance_tile",),
            writes=("best_index_tile",),
        ),
        "store": _operation(
            schedule,
            OperationKind.STORE,
            reads=("best_index_tile",),
            writes=("assignments",),
        ),
    }
    return None if any(item is None for item in operations.values()) else operations  # type: ignore[return-value]


def constraints(schedule: Schedule, target: Target) -> tuple[EmissionConstraint, ...]:
    """Return every known reason this finite Adapter cannot honor a Schedule."""

    findings: list[EmissionConstraint] = []
    if schedule.profile != "flash_kmeans_b32_smoke":
        findings.append(
            _constraint(
                "METAL_PROFILE_UNSUPPORTED",
                "metadata.profile",
                "the first Metal Adapter implements only flash_kmeans_b32_smoke",
            )
        )
    if schedule.target != target.target_id:
        findings.append(
            _constraint(
                "METAL_TARGET_MISMATCH",
                "target",
                "the Schedule and Metal Adapter Target differ",
            )
        )
    if target.architecture != METAL_ARCHITECTURE:
        findings.append(
            _constraint(
                "METAL_ARCHITECTURE_UNSUPPORTED",
                "target.architecture",
                "the Apple GPU family 9 Adapter requires its exact architecture contract",
            )
        )
    if (
        schedule.allocations
        or schedule.pipelines
        or schedule.barriers
        or schedule.residency is not None
    ):
        findings.append(
            _constraint(
                "METAL_TOPOLOGY_UNSUPPORTED",
                "allocations",
                "the first Metal Adapter admits no declared allocations, pipelines, barriers or residency override",
            )
        )
    if schedule.program_map is None or schedule.program_map.persistent:
        findings.append(
            _constraint(
                "METAL_PROGRAM_MAP_UNSUPPORTED",
                "program_map",
                "the first Metal Adapter requires a non-persistent token/batch program map",
            )
        )
    if len(schedule.roles) != 1 or not schedule.roles[0].warps:
        findings.append(
            _constraint(
                "METAL_EXECUTION_GROUPS_UNSUPPORTED",
                "roles",
                "the first Metal Adapter requires exactly one non-empty execution-group role",
            )
        )
    else:
        role = schedule.roles[0]
        groups = role.warps
        if role.name != "compute" or role.registers_per_thread is not None:
            findings.append(
                _constraint(
                    "METAL_EXECUTION_GROUPS_UNSUPPORTED",
                    "roles[0]",
                    "the Metal Adapter requires the unpartitioned compute role",
                )
            )
        if groups != tuple(range(len(groups))):
            findings.append(
                _constraint(
                    "METAL_EXECUTION_GROUPS_UNSUPPORTED",
                    "roles[0].warps",
                    "Metal SIMDgroup indices must form one zero-based contiguous range",
                )
            )
        if len(groups) not in {4, 8}:
            findings.append(
                _constraint(
                    "METAL_EXECUTION_GROUPS_UNSUPPORTED",
                    "roles[0].warps",
                    "the witnessed Metal Adapter admits exactly four or eight SIMDgroups",
                )
            )
        if len(groups) > target.resource_limits.maximum_execution_groups_per_workgroup:
            findings.append(
                _constraint(
                    "METAL_EXECUTION_GROUPS_UNSUPPORTED",
                    "roles[0].warps",
                    "declared SIMDgroups exceed the Target workgroup capacity",
                )
            )

    if target.execution_group_width != 32:
        findings.append(
            _constraint(
                "METAL_EXECUTION_GROUPS_UNSUPPORTED",
                "target.execution_group_width",
                "Metal SIMDgroup matrix lowering requires a 32-wide execution group",
            )
        )
    if METAL_SYNCHRONIZATION_CONTRACT not in target.synchronization_contracts:
        findings.append(
            _constraint(
                "METAL_SYNCHRONIZATION_CONTRACT_UNSUPPORTED",
                "target.synchronization_contracts",
                "Metal Flash-KMeans requires the simdgroup_barrier Target contract",
            )
        )
    if MemorySpace.SHARED not in target.memory_spaces:
        findings.append(
            _constraint(
                "METAL_THREADGROUP_MEMORY_CONTRACT_UNSUPPORTED",
                "target.memory_spaces",
                "Metal Flash-KMeans requires Target support for bounded threadgroup scratch",
            )
        )
    target_mma = target.instruction(METAL_MMA_CONTRACT)
    if target_mma is None or (
        target_mma.operand_dtypes != frozenset({DType.BF16})
        or target_mma.accumulator_dtype is not DType.FP32
        or target_mma.atom_shapes != ((8, 8, 8),)
        or target_mma.placement is not InstructionPlacement.BACKEND
        or target_mma.shape is not InstructionShape.EXPLICIT
    ):
        findings.append(
            _constraint(
                "METAL_TARGET_INSTRUCTION_CONTRACT_UNSUPPORTED",
                "target.instruction_contracts",
                "the Metal Adapter requires the exact backend-placed BF16 8x8x8 Target instruction contract",
            )
        )

    expected_globals = {
        "tokens": (DType.BF16, BufferMode.INPUT, (32, 512, 128)),
        "centroids": (DType.BF16, BufferMode.INPUT, (32, 1024, 128)),
        "centroid_sq": (DType.FP32, BufferMode.INPUT, (32, 1024)),
        "assignments": (DType.INT32, BufferMode.OUTPUT, (32, 512)),
    }
    for name, (dtype, mode, shape) in expected_globals.items():
        buffer = schedule.buffer(name)
        if buffer is None or (
            buffer.dtype,
            buffer.space,
            buffer.mode,
            buffer.shape,
            buffer.allocation,
            buffer.byte_offset,
            buffer.stages,
            buffer.swizzle,
        ) != (dtype, MemorySpace.GLOBAL, mode, shape, None, 0, 1, None):
            findings.append(
                _constraint(
                    "METAL_ABI_UNSUPPORTED",
                    f"buffers.{name}",
                    f"{name} differs from the fixed contiguous Metal ABI",
                )
            )

    topology = _topology(schedule)
    if topology is None or len(schedule.operations) != 8:
        findings.append(
            _constraint(
                "METAL_OPERATION_TOPOLOGY_UNSUPPORTED",
                "operations",
                "the first Metal Adapter requires the composed Flash-KMeans operation graph",
            )
        )
    else:
        ordered_ids = tuple(operation.op_id for operation in schedule.operations)
        if ordered_ids != tuple(_EXPECTED_DEPENDENCIES):
            findings.append(
                _constraint(
                    "METAL_OPERATION_TOPOLOGY_UNSUPPORTED",
                    "operations",
                    "Metal Flash-KMeans requires the canonical operation identities and order",
                )
            )
        if any(
            operation.role != "compute"
            or operation.waits
            or operation.signals
            or operation.pipeline is not None
            or operation.depends_on
            != _EXPECTED_DEPENDENCIES.get(operation.op_id, ())
            for operation in schedule.operations
        ):
            findings.append(
                _constraint(
                    "METAL_OPERATION_TOPOLOGY_UNSUPPORTED",
                    "operations",
                    "Metal Flash-KMeans requires the canonical role and dependency graph",
                )
            )
        load_operations = (
            topology["load_tokens"],
            topology["load_centroids"],
            topology["load_norm"],
        )
        if any(
            not isinstance(operation.parameters, LoadParameters)
            or operation.parameters.movement is not LoadMovement.GLOBAL
            or operation.parameters.descriptor_box is not None
            or operation.parameters.reuse is not None
            for operation in load_operations
        ):
            findings.append(
                _constraint(
                    "METAL_LOAD_CONTRACT_UNSUPPORTED",
                    "operations",
                    "the first Metal Adapter implements plain global loads without TMA or cache hints",
                )
            )
        store_parameters = topology["store"].parameters
        if (
            not isinstance(store_parameters, StoreParameters)
            or not store_parameters.coalesced
        ):
            findings.append(
                _constraint(
                    "METAL_STORE_CONTRACT_UNSUPPORTED",
                    "operations.store_assignment.parameters",
                    "the first Metal Adapter implements the declared coalesced assignment store",
                )
            )
        mma = topology["mma"]
        instruction = getattr(mma.parameters, "instruction", None)
        tile_shape = getattr(mma.parameters, "tile_shape", None)
        if (
            not isinstance(mma.parameters, MmaParameters)
            or mma.parameters.accumulator is not DType.FP32
            or instruction is None
            or instruction.contract != METAL_MMA_CONTRACT
            or instruction.shape != (8, 8, 8)
            or instruction.cta_group is not None
            or instruction.operand_source is not None
            or instruction.operand_major is not None
            or tile_shape is None
        ):
            findings.append(
                _constraint(
                    "METAL_MMA_CONTRACT_UNSUPPORTED",
                    "operations.mma.parameters.instruction",
                    "Metal Flash-KMeans requires the explicit BF16 8x8x8 SIMDgroup atom",
                )
            )
        scale = topology["scale"].parameters
        distance = topology["distance"].parameters
        argmin = topology["argmin"].parameters
        if (
            getattr(scale, "op", None) is not ElementwiseOp.MUL
            or getattr(scale, "scalar", None) != 2.0
            or getattr(scale, "broadcast_axis", None) is not None
            or getattr(distance, "op", None) is not ElementwiseOp.SUB
            or getattr(distance, "scalar", None) is not None
            or getattr(distance, "broadcast_axis", None) != 1
            or getattr(argmin, "tie_break", None) is not ArgminTieBreak.LOWEST_INDEX
            or getattr(argmin, "nan_policy", None) is not NaNPolicy.REJECT_INPUT
            or not getattr(argmin, "across_loop", False)
        ):
            findings.append(
                _constraint(
                    "METAL_REDUCTION_SEMANTICS_UNSUPPORTED",
                    "operations",
                    "Metal Flash-KMeans requires norm - 2*dot and a global lowest-index argmin",
                )
            )

    if len(schedule.tile_loops) != 1:
        findings.append(
            _constraint(
                "METAL_TILE_LOOP_UNSUPPORTED",
                "tile_loops",
                "the first Metal Adapter requires one centroid macro-tile loop",
            )
        )
    else:
        loop = schedule.tile_loops[0]
        options = loop.range_options
        if (
            options.num_stages != 1
            or options.loop_unroll_factor != 1
            or options.flatten
            or options.warp_specialize
            or options.disallow_acc_multi_buffer
            or options.disable_licm
        ):
            findings.append(
                _constraint(
                    "METAL_RANGE_OPTION_UNSUPPORTED",
                    "tile_loops[0].range_options",
                    "the first Metal Adapter implements one unstaged, un-specialized centroid loop",
                )
            )
        centroid_shape = _shape(schedule, "centroids")
        if (
            loop.name != "centroid_loop"
            or loop.iterator != "centroid_start"
            or loop.buffer != "centroids"
            or loop.dimension != 1
            or loop.body
            != (
                "load_centroids",
                "load_norm",
                "distance_mma",
                "scale_cross",
                "distance",
                "argmin",
            )
            or loop.tile % 8
            or centroid_shape is None
            or len(centroid_shape) < 2
            or centroid_shape[1] % loop.tile
        ):
            findings.append(
                _constraint(
                    "METAL_TILE_LOOP_UNSUPPORTED",
                    "tile_loops[0]",
                    "the canonical centroid loop must divide its extent into whole 8-column MMA atoms",
                )
            )

    if schedule.program_map is not None:
        axes = schedule.program_map.axes
        token_axes = [
            axis
            for axis in axes
            if axis.buffer == "tokens" and axis.dimension == 1 and axis.axis == 0
        ]
        batch_axes = [
            axis
            for axis in axes
            if axis.buffer == "tokens" and axis.dimension == 0 and axis.axis == 1
        ]
        if (
            schedule.program_map.traversal is not None
            or len(axes) != 2
            or len(token_axes) != 1
            or len(batch_axes) != 1
            or token_axes[0].name != "token_block"
            or batch_axes[0].name != "batch"
            or batch_axes[0].tile != 1
        ):
            findings.append(
                _constraint(
                    "METAL_PROGRAM_MAP_UNSUPPORTED",
                    "program_map.axes",
                    "Metal Flash-KMeans maps token macro tiles to x and batch to y",
                )
            )
        elif schedule.roles:
            token_tile = token_axes[0].tile
            groups = len(schedule.roles[0].warps)
            if token_tile % (groups * 8) or 512 % token_tile:
                findings.append(
                    _constraint(
                        "METAL_TOKEN_TILE_UNSUPPORTED",
                        "program_map.axes",
                        "the token tile must divide 512 and distribute whole 8-row atoms to every SIMDgroup",
                    )
                )
        maximum_grid = target.resource_limits.maximum_grid
        tokens = schedule.buffer("tokens")
        if (
            maximum_grid is not None
            and len(token_axes) == 1
            and tokens is not None
            and len(tokens.shape) >= 2
        ):
            required_grid = (
                (tokens.shape[1] + token_axes[0].tile - 1) // token_axes[0].tile,
                tokens.shape[0],
                1,
            )
            if any(
                required > maximum
                for required, maximum in zip(required_grid, maximum_grid)
            ):
                findings.append(
                    _constraint(
                        "METAL_GRID_LIMIT",
                        "target.resource_limits.maximum_grid",
                        "the derived Metal threadgroup grid exceeds the Target limit",
                    )
                )

    access_maps = {
        (access.operation, access.buffer): (
            tuple(
                (index.source, index.name, index.dimension)
                for index in access.indices
            ),
            access.boundary,
        )
        for access in schedule.access_maps
    }
    if (
        len(schedule.access_maps) != len(_EXPECTED_ACCESS_MAPS)
        or access_maps != _EXPECTED_ACCESS_MAPS
    ):
        findings.append(
            _constraint(
                "METAL_ACCESS_MAP_UNSUPPORTED",
                "access_maps",
                "the Metal Adapter requires the canonical contiguous token, centroid, norm and assignment maps",
            )
        )

    if schedule.outputs != ("assignments",):
        findings.append(
            _constraint(
                "METAL_ABI_UNSUPPORTED",
                "outputs",
                "the fixed Metal ABI exposes assignments as its sole output",
            )
        )

    if schedule.program_map is not None and schedule.tile_loops:
        token_axis = next(
            (
                axis
                for axis in schedule.program_map.axes
                if axis.name == "token_block"
                and axis.buffer == "tokens"
                and axis.dimension == 1
                and axis.axis == 0
            ),
            None,
        )
        if token_axis is not None:
            token_tile = token_axis.tile
            centroid_tile = schedule.tile_loops[0].tile
            expected_scratch = {
                "token_tile": (DType.BF16, (token_tile, 128)),
                "centroid_tile": (DType.BF16, (centroid_tile, 128)),
                "norm_tile": (DType.FP32, (centroid_tile,)),
                "cross": (DType.FP32, (token_tile, centroid_tile)),
                "scaled_cross": (DType.FP32, (token_tile, centroid_tile)),
                "distance_tile": (DType.FP32, (token_tile, centroid_tile)),
                "best_index_tile": (DType.INT32, (token_tile,)),
            }
            if (
                {buffer.name for buffer in schedule.buffers}
                != set(expected_globals) | set(expected_scratch)
                or any(
                buffer is None
                or (
                    buffer.dtype,
                    buffer.space,
                    buffer.mode,
                    buffer.shape,
                    buffer.allocation,
                    buffer.byte_offset,
                    buffer.stages,
                    buffer.swizzle,
                )
                != (
                    dtype,
                    MemorySpace.REGISTER,
                    BufferMode.SCRATCH,
                    shape,
                    None,
                    0,
                    1,
                    None,
                )
                for name, (dtype, shape) in expected_scratch.items()
                for buffer in (schedule.buffer(name),)
                )
            ):
                findings.append(
                    _constraint(
                        "METAL_SCRATCH_CONTRACT_UNSUPPORTED",
                        "buffers",
                        "logical scratch buffers must track the emitted token and centroid macro tiles",
                    )
                )

    if topology is not None and schedule.tile_loops and schedule.program_map is not None:
        token_axis = next(
            (
                axis
                for axis in schedule.program_map.axes
                if axis.buffer == "tokens" and axis.dimension == 1
            ),
            None,
        )
        tile_shape = getattr(topology["mma"].parameters, "tile_shape", None)
        token_shape = _shape(schedule, "tokens")
        if (
            token_axis is None
            or token_shape is None
            or len(token_shape) < 3
            or tile_shape
            != (token_axis.tile, schedule.tile_loops[0].tile, token_shape[2])
        ):
            findings.append(
                _constraint(
                    "METAL_MACRO_TILE_UNSUPPORTED",
                    "operations.mma.parameters.tile_shape",
                    "the MMA macro tile must match the token tile, centroid tile and feature extent",
                )
            )

    if schedule.roles:
        scratch_bytes = len(schedule.roles[0].warps) * 8 * 8 * DType.FP32.itemsize
        if scratch_bytes > target.resource_limits.maximum_threadgroup_memory_bytes:
            findings.append(
                _constraint(
                    "METAL_THREADGROUP_MEMORY_LIMIT",
                    "roles[0].warps",
                    "derived fragment scratch exceeds the Target threadgroup-memory limit",
                )
            )

    return tuple(findings)


def _derived(schedule: Schedule, target: Target) -> dict[str, int]:
    assert schedule.program_map is not None and len(schedule.tile_loops) == 1
    token_axis = next(
        axis
        for axis in schedule.program_map.axes
        if axis.buffer == "tokens" and axis.dimension == 1 and axis.axis == 0
    )
    tokens = schedule.buffer("tokens")
    centroids = schedule.buffer("centroids")
    assert tokens is not None and centroids is not None
    groups = len(schedule.roles[0].warps)
    return {
        "BATCHES": tokens.shape[0],
        "TOKENS": tokens.shape[1],
        "CENTROIDS": centroids.shape[1],
        "FEATURES": tokens.shape[2],
        "TOKEN_TILE": token_axis.tile,
        "CENTROID_TILE": schedule.tile_loops[0].tile,
        "SIMD_GROUPS": groups,
        "SIMD_WIDTH": target.execution_group_width,
        "ATOM_M": 8,
        "ATOM_N": 8,
        "ATOM_K": 8,
        "THREADGROUP_SCRATCH_BYTES": groups * 8 * 8 * DType.FP32.itemsize,
    }


def emit(
    schedule: Schedule, target: Target, *, entry_point: str | None = None
) -> Emission:
    """Emit deterministic MSL, or refuse every unsupported commitment first."""

    refused = constraints(schedule, target)
    if refused:
        first = refused[0]
        raise EmitError(f"{first.code} at {first.path}: {first.message}")
    topology = _topology(schedule)
    assert topology is not None
    values = _derived(schedule, target)
    entry = entry_point or "cake_flash_kmeans_assign"

    constants = "\n".join(
        f"constant constexpr uint {name} = {value}u;"
        for name, value in values.items()
        if name != "THREADGROUP_SCRATCH_BYTES"
    )
    source = f"""#include <metal_stdlib>

using namespace metal;

// Generated from Schedule __SCHEDULE_SHA256__; no runtime compilation or fallback.
{constants}

kernel void {entry}(
    device const bfloat *tokens [[buffer(0)]],
    device const bfloat *centroids [[buffer(1)]],
    device const float *centroid_sq [[buffer(2)]],
    device int *assignments [[buffer(3)]],
    uint3 threadgroup_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {{
  threadgroup float score_fragment[SIMD_GROUPS * ATOM_M * ATOM_N];
  const uint batch = threadgroup_id.y;
  const uint token_macro_start = threadgroup_id.x * TOKEN_TILE;

  for (uint token_local = simd_group * ATOM_M;
       token_local < TOKEN_TILE;
       token_local += SIMD_GROUPS * ATOM_M) {{
    const uint token_atom_start = token_macro_start + token_local;
    const uint token_index = token_atom_start + simd_lane;
    simdgroup_bfloat8x8 token_fragments[FEATURES / ATOM_K];

    // The Schedule keeps load_tokens outside the centroid loop. Retain all sixteen
    // distributed 8x8 fragments for this token atom and reuse them across centroids.
    for (uint feature = 0u; feature < FEATURES; feature += ATOM_K) {{
      simdgroup_barrier(mem_flags::mem_none);
      // CAKE_OP:{topology['load_tokens'].op_id}
      simdgroup_load(
          token_fragments[feature / ATOM_K],
          tokens + (batch * TOKENS + token_atom_start) * FEATURES + feature,
          FEATURES);
    }}

    float best_distance = INFINITY;
    uint best_index = 0xffffffffu;

    for (uint centroid_macro = 0u;
         centroid_macro < CENTROIDS;
         centroid_macro += CENTROID_TILE) {{
      for (uint centroid_atom = 0u;
           centroid_atom < CENTROID_TILE;
           centroid_atom += ATOM_N) {{
        const uint centroid_start = centroid_macro + centroid_atom;
        simdgroup_float8x8 dot =
            make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

        for (uint feature = 0u; feature < FEATURES; feature += ATOM_K) {{
          simdgroup_bfloat8x8 centroid_fragment;
          simdgroup_barrier(mem_flags::mem_none);
          // CAKE_OP:{topology['load_centroids'].op_id}
          simdgroup_load(
              centroid_fragment,
              centroids + (batch * CENTROIDS + centroid_start) * FEATURES + feature,
              FEATURES,
              ulong2(0, 0),
              true);
          simdgroup_barrier(mem_flags::mem_none);
          // CAKE_OP:{topology['mma'].op_id}
          simdgroup_multiply_accumulate(
              dot, token_fragments[feature / ATOM_K], centroid_fragment, dot);
        }}

        simdgroup_store(
            dot,
            score_fragment + simd_group * ATOM_M * ATOM_N,
            ATOM_N);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_lane < ATOM_M) {{
          for (uint column = 0u; column < ATOM_N; ++column) {{
            const uint candidate_index = centroid_start + column;
            // CAKE_OP:{topology['load_norm'].op_id}
            const float norm = centroid_sq[batch * CENTROIDS + candidate_index];
            const float dot_value = score_fragment[
                simd_group * ATOM_M * ATOM_N + simd_lane * ATOM_N + column];
            // CAKE_OP:{topology['scale'].op_id}
            const float scaled_dot = 2.0f * dot_value;
            // CAKE_OP:{topology['distance'].op_id}
            const float distance = norm - scaled_dot;
            // CAKE_OP:{topology['argmin'].op_id}
            const bool update = distance < best_distance ||
                (distance == best_distance && candidate_index < best_index);
            if (update) {{
              best_distance = distance;
              best_index = candidate_index;
            }}
          }}
        }}
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }}
    }}

    if (simd_lane < ATOM_M) {{
      // CAKE_OP:{topology['store'].op_id}
      assignments[batch * TOKENS + token_index] = int(best_index);
    }}
  }}
}}
// CAKE_KERNEL_END
"""
    return Emission(
        source=source,
        entry_point=entry,
        constants=values,
        toolchain={
            "threadgroups_per_grid": [
                values["TOKENS"] // values["TOKEN_TILE"],
                values["BATCHES"],
                1,
            ],
            "threads_per_threadgroup": [
                values["SIMD_GROUPS"] * values["SIMD_WIDTH"],
                1,
                1,
            ],
            "threadgroup_memory_bytes": values["THREADGROUP_SCRATCH_BYTES"],
            "language_standard": "metal3.2",
            "compiler_flags": [
                "-std=metal3.2",
                "-fmetal-math-mode=safe",
                "-ffp-contract=off",
            ],
        },
    )
