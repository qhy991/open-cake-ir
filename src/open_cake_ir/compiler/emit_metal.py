"""Emit the first finite Metal Shading Language subset from a Schedule.

The Module is an Adapter behind the generated-backend Seam.  It recognizes two
operation graphs rather than workload names: a runtime-indexed BF16 gather and the same
gather followed by FP32 weighting and reduction.  Every accepted commitment is checked
by :func:`preflight`; direct emitter users and ``Compiler.assess`` therefore refuse the
same unsupported Schedule before source generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .emit import BackendPrecondition, Emission, EmitError, source_comment_text
from .ir import (
    AccessIndexKind,
    BoundaryPolicy,
    BufferMode,
    DType,
    ElementwiseOp,
    ElementwiseParameters,
    LoadMovement,
    LoadParameters,
    LoadReuse,
    LoweringBackend,
    MemorySpace,
    OperationKind,
    ReduceOp,
    ReduceParameters,
    ReductionScope,
    Schedule,
    StoreParameters,
)
from .target import Target


METAL_ARCHITECTURE = "apple_gpu_family9"

SUPPORTED_DTYPES = frozenset({DType.BF16, DType.FP32, DType.INT32})
SUPPORTED_OPERATION_KINDS = frozenset(
    {
        OperationKind.LOAD,
        OperationKind.ELEMENTWISE,
        OperationKind.REDUCE,
        OperationKind.STORE,
    }
)


class _GraphKind(str, Enum):
    INDEXED_GATHER = "indexed_gather"
    WEIGHTED_COMBINE = "weighted_combine"


@dataclass(frozen=True)
class _Graph:
    kind: _GraphKind
    operation_ids: tuple[str, ...]
    execution_groups: int
    buffer_order: tuple[str, ...]


_INDEXED_GATHER = _Graph(
    _GraphKind.INDEXED_GATHER,
    (
        "load_expert_ids",
        "load_row_ids",
        "load_selected_rows",
        "store_gathered_rows",
    ),
    4,
    ("expert_rows", "expert_ids", "row_ids", "gathered_rows"),
)

_WEIGHTED_COMBINE = _Graph(
    _GraphKind.WEIGHTED_COMBINE,
    (
        "load_expert_ids",
        "load_row_ids",
        "load_route_weights",
        "load_selected_rows",
        "apply_route_weights",
        "combine_routes",
        "store_output",
    ),
    1,
    ("expert_rows", "expert_ids", "row_ids", "route_weights", "output"),
)


_COMMON_BUFFERS = {
    "expert_rows": (MemorySpace.GLOBAL, DType.BF16, (4, 8, 16), BufferMode.INPUT),
    "expert_ids": (MemorySpace.GLOBAL, DType.INT32, (8, 8), BufferMode.INPUT),
    "row_ids": (MemorySpace.GLOBAL, DType.INT32, (8, 8), BufferMode.INPUT),
    "expert_id_tile": (MemorySpace.REGISTER, DType.INT32, (8,), BufferMode.SCRATCH),
    "row_id_tile": (MemorySpace.REGISTER, DType.INT32, (8,), BufferMode.SCRATCH),
}

_GATHER_BUFFERS = {
    **_COMMON_BUFFERS,
    "gathered_rows": (
        MemorySpace.GLOBAL,
        DType.BF16,
        (8, 8, 16),
        BufferMode.OUTPUT,
    ),
    "gathered_tile": (
        MemorySpace.REGISTER,
        DType.BF16,
        (8, 16),
        BufferMode.SCRATCH,
    ),
}
_GATHER_BUFFER_ORDER = (
    "expert_rows",
    "expert_ids",
    "row_ids",
    "gathered_rows",
    "expert_id_tile",
    "row_id_tile",
    "gathered_tile",
)

_COMBINE_BUFFERS = {
    **_COMMON_BUFFERS,
    "route_weights": (MemorySpace.GLOBAL, DType.FP32, (8, 8), BufferMode.INPUT),
    "output": (MemorySpace.GLOBAL, DType.BF16, (8, 16), BufferMode.OUTPUT),
    "route_weight_tile": (
        MemorySpace.REGISTER,
        DType.FP32,
        (8,),
        BufferMode.SCRATCH,
    ),
    "selected_rows": (
        MemorySpace.REGISTER,
        DType.BF16,
        (8, 16),
        BufferMode.SCRATCH,
    ),
    "weighted_rows": (
        MemorySpace.REGISTER,
        DType.FP32,
        (8, 16),
        BufferMode.SCRATCH,
    ),
    "combined_row": (
        MemorySpace.REGISTER,
        DType.FP32,
        (16,),
        BufferMode.SCRATCH,
    ),
}
_COMBINE_BUFFER_ORDER = (
    "expert_rows",
    "expert_ids",
    "row_ids",
    "route_weights",
    "output",
    "expert_id_tile",
    "row_id_tile",
    "route_weight_tile",
    "selected_rows",
    "weighted_rows",
    "combined_row",
)


_GATHER_OPERATIONS = (
    (
        "load_expert_ids",
        OperationKind.LOAD,
        ("expert_ids",),
        ("expert_id_tile",),
        (),
    ),
    (
        "load_row_ids",
        OperationKind.LOAD,
        ("row_ids",),
        ("row_id_tile",),
        (),
    ),
    (
        "load_selected_rows",
        OperationKind.LOAD,
        ("expert_rows", "expert_id_tile", "row_id_tile"),
        ("gathered_tile",),
        ("load_expert_ids", "load_row_ids"),
    ),
    (
        "store_gathered_rows",
        OperationKind.STORE,
        ("gathered_tile",),
        ("gathered_rows",),
        ("load_selected_rows",),
    ),
)

_COMBINE_OPERATIONS = (
    (
        "load_expert_ids",
        OperationKind.LOAD,
        ("expert_ids",),
        ("expert_id_tile",),
        (),
    ),
    (
        "load_row_ids",
        OperationKind.LOAD,
        ("row_ids",),
        ("row_id_tile",),
        (),
    ),
    (
        "load_route_weights",
        OperationKind.LOAD,
        ("route_weights",),
        ("route_weight_tile",),
        (),
    ),
    (
        "load_selected_rows",
        OperationKind.LOAD,
        ("expert_rows", "expert_id_tile", "row_id_tile"),
        ("selected_rows",),
        ("load_expert_ids", "load_row_ids"),
    ),
    (
        "apply_route_weights",
        OperationKind.ELEMENTWISE,
        ("selected_rows", "route_weight_tile"),
        ("weighted_rows",),
        ("load_selected_rows", "load_route_weights"),
    ),
    (
        "combine_routes",
        OperationKind.REDUCE,
        ("weighted_rows",),
        ("combined_row",),
        ("apply_route_weights",),
    ),
    (
        "store_output",
        OperationKind.STORE,
        ("combined_row",),
        ("output",),
        ("combine_routes",),
    ),
)


def _index(
    source: AccessIndexKind, name: str | None = None, dimension: int | None = None
):
    return (source, name, dimension)


_COMMON_ACCESS = {
    ("load_expert_ids", "expert_ids"): (
        (
            _index(AccessIndexKind.PROGRAM, "token"),
            _index(AccessIndexKind.DIMENSION, dimension=1),
        ),
        BoundaryPolicy.MASK_TILED_AXES,
    ),
    ("load_row_ids", "row_ids"): (
        (
            _index(AccessIndexKind.PROGRAM, "token"),
            _index(AccessIndexKind.DIMENSION, dimension=1),
        ),
        BoundaryPolicy.MASK_TILED_AXES,
    ),
    ("load_selected_rows", "expert_rows"): (
        (
            _index(AccessIndexKind.BUFFER, "expert_id_tile"),
            _index(AccessIndexKind.BUFFER, "row_id_tile"),
            _index(AccessIndexKind.DIMENSION, dimension=2),
        ),
        BoundaryPolicy.MASK_TILED_AXES,
    ),
}

_GATHER_ACCESS = {
    **_COMMON_ACCESS,
    ("store_gathered_rows", "gathered_rows"): (
        (
            _index(AccessIndexKind.PROGRAM, "token"),
            _index(AccessIndexKind.DIMENSION, dimension=1),
            _index(AccessIndexKind.DIMENSION, dimension=2),
        ),
        BoundaryPolicy.MASK_TILED_AXES,
    ),
}

_COMBINE_ACCESS = {
    **_COMMON_ACCESS,
    ("load_route_weights", "route_weights"): (
        (
            _index(AccessIndexKind.PROGRAM, "token"),
            _index(AccessIndexKind.DIMENSION, dimension=1),
        ),
        BoundaryPolicy.MASK_TILED_AXES,
    ),
    ("store_output", "output"): (
        (
            _index(AccessIndexKind.PROGRAM, "token"),
            _index(AccessIndexKind.DIMENSION, dimension=1),
        ),
        BoundaryPolicy.MASK_TILED_AXES,
    ),
}


def _buffer_signature(schedule: Schedule) -> dict[str, tuple[object, ...]]:
    return {
        item.name: (item.space, item.dtype, item.shape, item.mode)
        for item in schedule.buffers
    }


def _graph_from_buffers(schedule: Schedule) -> _Graph | None:
    signature = _buffer_signature(schedule)
    buffer_order = tuple(item.name for item in schedule.buffers)
    if signature == _GATHER_BUFFERS and buffer_order == _GATHER_BUFFER_ORDER:
        return _INDEXED_GATHER
    if signature == _COMBINE_BUFFERS and buffer_order == _COMBINE_BUFFER_ORDER:
        return _WEIGHTED_COMBINE
    return None


def _operation_graph_matches(schedule: Schedule, graph: _Graph) -> bool:
    expected = (
        _GATHER_OPERATIONS
        if graph.kind is _GraphKind.INDEXED_GATHER
        else _COMBINE_OPERATIONS
    )
    observed = tuple(
        (item.op_id, item.kind, item.reads, item.writes, item.depends_on)
        for item in schedule.operations
    )
    if observed != expected:
        return False
    return all(
        item.role == "compute"
        and not item.waits
        and not item.signals
        and item.pipeline is None
        for item in schedule.operations
    )


def _operation_parameters_match(schedule: Schedule, graph: _Graph) -> bool:
    for operation in schedule.operations:
        parameters = operation.parameters
        if operation.kind is OperationKind.LOAD:
            if not (
                isinstance(parameters, LoadParameters)
                and parameters.movement is LoadMovement.GLOBAL
                and parameters.reuse is LoadReuse.STREAMED
                and parameters.descriptor_box is None
            ):
                return False
        elif operation.kind is OperationKind.STORE:
            if not isinstance(parameters, StoreParameters) or not parameters.coalesced:
                return False
        elif operation.kind is OperationKind.ELEMENTWISE:
            if not (
                graph.kind is _GraphKind.WEIGHTED_COMBINE
                and isinstance(parameters, ElementwiseParameters)
                and parameters.op is ElementwiseOp.MUL
                and parameters.scalar is None
                and parameters.broadcast_axis == 0
                and parameters.instruction is None
            ):
                return False
        elif operation.kind is OperationKind.REDUCE:
            if not (
                graph.kind is _GraphKind.WEIGHTED_COMBINE
                and isinstance(parameters, ReduceParameters)
                and parameters.op is ReduceOp.SUM
                and parameters.axis == 0
                and parameters.scope is ReductionScope.CTA
            ):
                return False
    return True


def _access_maps_match(schedule: Schedule, graph: _Graph) -> bool:
    expected = (
        _GATHER_ACCESS if graph.kind is _GraphKind.INDEXED_GATHER else _COMBINE_ACCESS
    )
    observed = {
        (item.operation, item.buffer): (
            tuple(
                (index.source, index.name, index.dimension) for index in item.indices
            ),
            item.boundary,
        )
        for item in schedule.access_maps
    }
    return observed == expected and len(observed) == len(schedule.access_maps)


def preflight(schedule: Schedule, target: Target) -> tuple[BackendPrecondition, ...]:
    """Return every backend-owned reason this finite Metal subset cannot emit."""

    findings: list[BackendPrecondition] = []

    def add(condition: object, code: str, path: str, message: str) -> None:
        if not condition:
            findings.append(BackendPrecondition(code, path, message))

    add(
        schedule.lowering.backend is LoweringBackend.METAL,
        "METAL_ROUTE_REQUIRED",
        "lowering.backend",
        "the Metal Adapter requires the metal lowering route",
    )
    add(
        schedule.target == target.target_id,
        "METAL_TARGET_MISMATCH",
        "target",
        "the Schedule and Metal Adapter Target differ",
    )
    add(
        target.architecture == METAL_ARCHITECTURE,
        "METAL_ARCHITECTURE_UNSUPPORTED",
        "target.architecture",
        "the Adapter requires the exact Apple GPU family 9 architecture contract",
    )
    add(
        target.execution_group_width == 32,
        "METAL_EXECUTION_WIDTH_UNSUPPORTED",
        "target.execution_group_width",
        "the witnessed Metal Adapter requires 32-wide SIMDgroups",
    )
    add(
        not schedule.allocations
        and not schedule.pipelines
        and not schedule.barriers
        and not schedule.tile_loops
        and schedule.residency is None,
        "METAL_TOPOLOGY_UNSUPPORTED",
        "allocations",
        "this Metal subset admits no allocations, pipelines, barriers, tile loops, or residency override",
    )

    program_map = schedule.program_map
    program_ok = False
    if (
        program_map is not None
        and not program_map.persistent
        and len(program_map.axes) == 1
    ):
        axis = program_map.axes[0]
        program_ok = (
            axis.name == "token"
            and axis.axis == 0
            and axis.buffer == "expert_ids"
            and axis.dimension == 0
            and axis.tile == 1
            and program_map.traversal is None
        )
    add(
        program_ok,
        "METAL_PROGRAM_MAP_UNSUPPORTED",
        "program_map",
        "this Metal subset requires one non-persistent token program axis",
    )

    graph = _graph_from_buffers(schedule)
    add(
        graph is not None,
        "METAL_OPERATION_GRAPH_UNSUPPORTED",
        "buffers",
        "the Metal Adapter implements only the declared indexed-gather and weighted-combine graphs",
    )
    if graph is None:
        return tuple(findings)

    role_ok = (
        len(schedule.roles) == 1
        and schedule.roles[0].name == "compute"
        and schedule.roles[0].registers_per_thread is None
        and schedule.roles[0].warps == tuple(range(graph.execution_groups))
    )
    threads = graph.execution_groups * target.execution_group_width
    add(
        role_ok and threads <= target.resource_limits.maximum_threads_per_workgroup,
        "METAL_EXECUTION_GROUPS_UNSUPPORTED",
        "roles",
        f"the {graph.kind.value} graph requires exactly {graph.execution_groups} contiguous SIMDgroup(s)",
    )
    add(
        _operation_graph_matches(schedule, graph),
        "METAL_OPERATION_GRAPH_UNSUPPORTED",
        "operations",
        "operation identities, dataflow, dependencies, or role placement differ from the admitted graph",
    )
    add(
        _operation_parameters_match(schedule, graph),
        "METAL_OPERATION_CONTRACT_UNSUPPORTED",
        "operations",
        "a load, arithmetic, reduction, or store commitment has no body in this Metal Adapter",
    )
    add(
        _access_maps_match(schedule, graph),
        "METAL_ACCESS_MAP_UNSUPPORTED",
        "access_maps",
        "the Metal Adapter cannot realize the declared access-map composition",
    )
    add(
        schedule.outputs
        == (
            ("gathered_rows",)
            if graph.kind is _GraphKind.INDEXED_GATHER
            else ("output",)
        ),
        "METAL_OUTPUT_UNSUPPORTED",
        "outputs",
        "the declared output differs from the admitted operation graph",
    )
    return tuple(findings)


def _dimensions(schedule: Schedule) -> dict[str, int]:
    expert_rows = schedule.buffer("expert_rows")
    expert_ids = schedule.buffer("expert_ids")
    assert expert_rows is not None and expert_ids is not None
    experts, rows, features = expert_rows.shape
    tokens, routes = expert_ids.shape
    return {
        "TOKENS": tokens,
        "EXPERTS": experts,
        "ROWS": rows,
        "ROUTES": routes,
        "FEATURES": features,
    }


def _preamble(schedule: Schedule) -> list[str]:
    dimensions = _dimensions(schedule)
    schedule_id = source_comment_text(schedule.schedule_id)
    return [
        "#include <metal_stdlib>",
        "",
        "using namespace metal;",
        "",
        f"// Schedule: {schedule_id}",
        "// Schedule SHA256: __SCHEDULE_SHA256__",
        *(f"constant uint {name} = {value}u;" for name, value in dimensions.items()),
        "",
    ]


def _emit_indexed_gather(schedule: Schedule, entry_point: str) -> str:
    lines = _preamble(schedule)
    lines.extend(
        [
            f"kernel void {entry_point}(",
            "    device const bfloat *expert_rows [[buffer(0)]],",
            "    device const int *expert_ids [[buffer(1)]],",
            "    device const int *row_ids [[buffer(2)]],",
            "    device bfloat *gathered_rows [[buffer(3)]],",
            "    uint3 threadgroup_position [[threadgroup_position_in_grid]],",
            "    uint thread_index [[thread_index_in_threadgroup]]) {",
            "  const uint token = threadgroup_position.x;",
            "  const uint route = thread_index / FEATURES;",
            "  const uint feature = thread_index % FEATURES;",
            "  if (token >= TOKENS || route >= ROUTES) return;",
            "  const uint route_offset = token * ROUTES + route;",
            "",
            "  // CAKE_OP:load_expert_ids",
            "  const int expert = expert_ids[route_offset];",
            "",
            "  // CAKE_OP:load_row_ids",
            "  const int row = row_ids[route_offset];",
            "",
            "  // CAKE_OP:load_selected_rows",
            "  bfloat selected = bfloat(0.0f);",
            "  if (expert >= 0 && expert < int(EXPERTS) && row >= 0 && row < int(ROWS)) {",
            "    const uint source = (uint(expert) * ROWS + uint(row)) * FEATURES + feature;",
            "    selected = expert_rows[source];",
            "  }",
            "",
            "  // CAKE_OP:store_gathered_rows",
            "  gathered_rows[route_offset * FEATURES + feature] = selected;",
            "  // CAKE_KERNEL_END",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def _emit_weighted_combine(schedule: Schedule, entry_point: str) -> str:
    lines = _preamble(schedule)
    lines.extend(
        [
            f"kernel void {entry_point}(",
            "    device const bfloat *expert_rows [[buffer(0)]],",
            "    device const int *expert_ids [[buffer(1)]],",
            "    device const int *row_ids [[buffer(2)]],",
            "    device const float *route_weights [[buffer(3)]],",
            "    device bfloat *output [[buffer(4)]],",
            "    uint3 threadgroup_position [[threadgroup_position_in_grid]],",
            "    uint thread_index [[thread_index_in_threadgroup]]) {",
            "  const uint token = threadgroup_position.x;",
            "  const uint feature = thread_index;",
            "  if (token >= TOKENS || feature >= FEATURES) return;",
            "",
            "  // CAKE_OP:load_expert_ids",
            "  int expert_id_tile[ROUTES];",
            "  for (uint route = 0; route < ROUTES; ++route)",
            "    expert_id_tile[route] = expert_ids[token * ROUTES + route];",
            "",
            "  // CAKE_OP:load_row_ids",
            "  int row_id_tile[ROUTES];",
            "  for (uint route = 0; route < ROUTES; ++route)",
            "    row_id_tile[route] = row_ids[token * ROUTES + route];",
            "",
            "  // CAKE_OP:load_route_weights",
            "  float route_weight_tile[ROUTES];",
            "  for (uint route = 0; route < ROUTES; ++route)",
            "    route_weight_tile[route] = route_weights[token * ROUTES + route];",
            "",
            "  // CAKE_OP:load_selected_rows",
            "  bfloat selected_rows[ROUTES];",
            "  for (uint route = 0; route < ROUTES; ++route) {",
            "    const int expert = expert_id_tile[route];",
            "    const int row = row_id_tile[route];",
            "    bfloat selected = bfloat(0.0f);",
            "    if (expert >= 0 && expert < int(EXPERTS) && row >= 0 && row < int(ROWS)) {",
            "      const uint source = (uint(expert) * ROWS + uint(row)) * FEATURES + feature;",
            "      selected = expert_rows[source];",
            "    }",
            "    selected_rows[route] = selected;",
            "  }",
            "",
            "  // CAKE_OP:apply_route_weights",
            "  float weighted_rows[ROUTES];",
            "  for (uint route = 0; route < ROUTES; ++route)",
            "    weighted_rows[route] = float(selected_rows[route]) * route_weight_tile[route];",
            "",
            "  // CAKE_OP:combine_routes",
            "  float combined = 0.0f;",
            "  for (uint route = 0; route < ROUTES; ++route)",
            "    combined += weighted_rows[route];",
            "",
            "  // CAKE_OP:store_output",
            "  output[token * FEATURES + feature] = bfloat(combined);",
            "  // CAKE_KERNEL_END",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def emit(
    schedule: Schedule,
    target: Target,
    entry_point: str | None = None,
) -> Emission:
    """Emit deterministic MSL after applying the same preflight as assessment."""

    failures = preflight(schedule, target)
    if failures:
        raise EmitError(failures[0].message)
    graph = _graph_from_buffers(schedule)
    assert graph is not None
    symbol = entry_point or schedule.lowering.entry_point
    source = (
        _emit_indexed_gather(schedule, symbol)
        if graph.kind is _GraphKind.INDEXED_GATHER
        else _emit_weighted_combine(schedule, symbol)
    )
    threads = graph.execution_groups * target.execution_group_width
    constants = {
        **_dimensions(schedule),
        "EXECUTION_GROUPS": graph.execution_groups,
    }
    toolchain = {
        "buffer_order": list(graph.buffer_order),
        "threadgroups_per_grid": [constants["TOKENS"], 1, 1],
        "threads_per_threadgroup": [threads, 1, 1],
        "threadgroup_memory_bytes": 0,
        "language_standard": "metal3.2",
        "compiler_flags": [
            "-std=metal3.2",
            "-fmetal-math-mode=safe",
            "-ffp-contract=off",
        ],
    }
    return Emission(source, symbol, constants, toolchain)
