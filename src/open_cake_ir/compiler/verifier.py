"""Schedule verifier: target-derived hard gates over the canonical typed IR.

Findings are grouped by the four contract classes the paper's harness reports
(arXiv:2608.12629v1 Table 1):

* ``SCHEDULE_SEMANTICS``    -- structural invariants of the declared schedule
* ``HARDWARE_CONFORMANCE``  -- supported resource, instruction and architecture contracts
* ``DATA_CONSISTENCY``      -- data flow and producer/consumer representation
* ``PROGRAM_SAFETY``        -- synchronization, ordering and memory-use hazards

Every rule is derived from the Schedule and its Target. None compares against a fixed
expected profile: the legacy `explicit_resource_verifier` reported "differs from the
selected closed profile", which tells an agent that its bytes are unfamiliar but not
what is wrong. A finding here names the offending path and the violated contract.

Verification is total -- it never raises for an admissible-but-wrong Schedule, it
returns findings. Structural admissibility is the IR's job and has already happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .ir import (
    TMEM_COLUMN_BYTES,
    AccessIndexKind,
    BufferMode,
    LoadMovement,
    MemorySpace,
    Operation,
    OperationKind,
    Schedule,
)
from .target import Target


class FindingCategory(str, Enum):
    SCHEDULE_SEMANTICS = "schedule_semantics"
    HARDWARE_CONFORMANCE = "hardware_conformance"
    DATA_CONSISTENCY = "data_consistency"
    PROGRAM_SAFETY = "program_safety"


class FindingSeverity(str, Enum):
    """The three dispositions the paper's harness returns.

    ``BLOCKING`` rejects the candidate with a localized reason, ``REPORT`` describes a
    likely limit, and ``HINT`` suggests a non-blocking improvement. Only ``BLOCKING``
    stops a Schedule from reaching the toolchain.
    """

    BLOCKING = "blocking"
    REPORT = "report"
    HINT = "hint"


@dataclass(frozen=True)
class Finding:
    """One localized violation: a repair target, not a backend error."""

    code: str
    path: str
    message: str
    category: FindingCategory
    severity: FindingSeverity = FindingSeverity.BLOCKING

    @property
    def blocks_lowering(self) -> bool:
        return self.severity is FindingSeverity.BLOCKING

    def __str__(self) -> str:
        return f"[{self.severity.value}] {self.code} at {self.path}: {self.message}"


_SEVERITY_ORDER = {
    FindingSeverity.BLOCKING: 0,
    FindingSeverity.REPORT: 1,
    FindingSeverity.HINT: 2,
}


class _Collector:
    def __init__(self) -> None:
        self._findings: list[Finding] = []

    def add(
        self,
        code: str,
        path: str,
        message: str,
        category: FindingCategory,
        severity: FindingSeverity = FindingSeverity.BLOCKING,
    ) -> None:
        self._findings.append(Finding(code, path, message, category, severity))

    def result(self) -> tuple[Finding, ...]:
        # Stable order: severity, then category, code and path. Deterministic across
        # runs so a Corpus Gate can pin the exact sequence.
        return tuple(
            sorted(
                self._findings,
                key=lambda f: (
                    _SEVERITY_ORDER[f.severity],
                    f.category.value,
                    f.code,
                    f.path,
                ),
            )
        )


def _duplicates(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    repeated: set[str] = set()
    for value in values:
        if value in seen:
            repeated.add(value)
        seen.add(value)
    return sorted(repeated)


def _cycle_members(graph: dict[str, tuple[str, ...]]) -> set[str]:
    """Nodes that participate in a dependency cycle (iterative Tarjan-free variant)."""

    WHITE, GREY, BLACK = 0, 1, 2
    colour = dict.fromkeys(graph, WHITE)
    members: set[str] = set()

    for root in graph:
        if colour[root] != WHITE:
            continue
        stack: list[tuple[str, int]] = [(root, 0)]
        path: list[str] = []
        while stack:
            node, index = stack.pop()
            if index == 0:
                if colour[node] != WHITE:
                    continue
                colour[node] = GREY
                path.append(node)
            edges = graph.get(node, ())
            if index < len(edges):
                stack.append((node, index + 1))
                nxt = edges[index]
                if nxt not in graph:
                    continue
                if colour[nxt] == GREY:
                    members.update(path[path.index(nxt):])
                elif colour[nxt] == WHITE:
                    stack.append((nxt, 0))
            else:
                colour[node] = BLACK
                if path and path[-1] == node:
                    path.pop()
    return members


# --------------------------------------------------------------- schedule semantics


def _verify_schedule_semantics(schedule: Schedule, out: _Collector) -> None:
    category = FindingCategory.SCHEDULE_SEMANTICS

    if not schedule.operations:
        out.add(
            "SCHEDULE_OPERATIONS_EMPTY",
            "operations",
            "a Schedule must declare at least one operation",
            category,
        )

    groups = {
        "roles": [item.name for item in schedule.roles],
        "allocations": [item.name for item in schedule.allocations],
        "buffers": [item.name for item in schedule.buffers],
        "pipelines": [item.name for item in schedule.pipelines],
        "barriers": [item.name for item in schedule.barriers],
        "operations": [item.op_id for item in schedule.operations],
        "tile_loops": [item.name for item in schedule.tile_loops],
    }
    for group, names in groups.items():
        for duplicate in _duplicates(names):
            out.add(
                "NAME_DUPLICATE",
                group,
                f"duplicate {group[:-1]} name {duplicate!r}",
                category,
            )

    _verify_loop_nest(schedule, out)

    for duplicate in _duplicates(loop.iterator for loop in schedule.tile_loops):
        out.add(
            "LOOP_DUPLICATE_ITERATOR",
            "tile_loops",
            f"duplicate loop iterator {duplicate!r}",
            category,
        )

    if schedule.program_map is not None:
        axes = schedule.program_map.axes
        for duplicate in _duplicates(axis.name for axis in axes):
            out.add(
                "PROGRAM_AXIS_DUPLICATE_NAME",
                "program_map.axes",
                f"duplicate program axis name {duplicate!r}",
                category,
            )
        numbers = [str(axis.axis) for axis in axes]
        for duplicate in _duplicates(numbers):
            out.add(
                "PROGRAM_AXIS_DUPLICATE_NUMBER",
                "program_map.axes",
                f"two axes claim program dimension {duplicate}",
                category,
            )

    seen_edges: set[tuple[str, str]] = set()
    for index, access in enumerate(schedule.access_maps):
        edge = (access.operation, access.buffer)
        if edge in seen_edges:
            out.add(
                "ACCESS_MAP_DUPLICATE",
                f"access_maps[{index}]",
                f"duplicate access map for {access.operation!r} -> {access.buffer!r}",
                category,
            )
        seen_edges.add(edge)


def _verify_loop_nest(schedule: Schedule, out: _Collector) -> None:
    """A loop body names operations and nested loops; both must resolve exactly once."""

    category = FindingCategory.SCHEDULE_SEMANTICS
    loops = {loop.name for loop in schedule.tile_loops}
    operations = {operation.op_id for operation in schedule.operations}
    order = {op.op_id: index for index, op in enumerate(schedule.operations)}

    scope_of: dict[str, str] = {}
    parent_of: dict[str, str] = {}
    for index, loop in enumerate(schedule.tile_loops):
        path = f"tile_loops[{index}].body"
        positions: list[int] = []
        for entry in loop.body:
            if entry == loop.name:
                out.add(
                    "LOOP_NEST_SELF",
                    path,
                    f"loop {loop.name!r} contains itself",
                    category,
                )
            elif entry in loops:
                if entry in parent_of:
                    out.add(
                        "LOOP_NEST_MULTIPLE_PARENTS",
                        path,
                        f"loop {entry!r} is nested in both {parent_of[entry]!r} and "
                        f"{loop.name!r}",
                        category,
                    )
                else:
                    parent_of[entry] = loop.name
            elif entry in operations:
                if entry in scope_of:
                    out.add(
                        "LOOP_OPERATION_MULTIPLE_SCOPE",
                        path,
                        f"operation {entry!r} is in both {scope_of[entry]!r} and "
                        f"{loop.name!r}",
                        category,
                    )
                else:
                    scope_of[entry] = loop.name
                positions.append(order[entry])
            else:
                out.add(
                    "LOOP_BODY_UNKNOWN",
                    path,
                    f"{entry!r} is neither a declared operation nor a declared loop",
                    category,
                )
        if positions != sorted(positions):
            out.add(
                "LOOP_BODY_ORDER",
                path,
                f"loop {loop.name!r} lists operations out of declaration order",
                category,
            )

    for name in sorted(loops):
        seen, cursor = {name}, name
        while cursor in parent_of:
            cursor = parent_of[cursor]
            if cursor in seen:
                out.add(
                    "LOOP_NEST_CYCLE",
                    "tile_loops",
                    f"loop nesting cycle includes {name!r}",
                    category,
                )
                break
            seen.add(cursor)


# ------------------------------------------------------------ hardware conformance


def _verify_hardware_conformance(
    schedule: Schedule, target: Target, out: _Collector
) -> None:
    category = FindingCategory.HARDWARE_CONFORMANCE
    limits = target.resource_limits

    if schedule.target != target.target_id:
        out.add(
            "TARGET_UNSUPPORTED",
            "target",
            f"Schedule targets {schedule.target!r} but the Compiler Revision binds "
            f"{target.target_id!r}; there is no silent architecture fallback",
            category,
        )
        return

    for index, role in enumerate(schedule.roles):
        for position, warp in enumerate(role.warps):
            if warp >= limits.maximum_warps_per_cta:
                out.add(
                    "ROLE_WARP_RANGE",
                    f"roles[{index}].warps[{position}]",
                    f"warp index {warp} is outside the Target CTA range "
                    f"[0, {limits.maximum_warps_per_cta})",
                    category,
                )

    # The budget is bounded by the highest warp index in use, not by how many warps
    # were declared: warps [0, 1, 2, 4096] is four entries but needs 4097 slots.
    extent = schedule.total_warp_extent
    if extent > limits.maximum_warps_per_cta:
        out.add(
            "TARGET_WARP_LIMIT",
            "roles",
            f"declared warps span {extent} slots, above the Target limit of "
            f"{limits.maximum_warps_per_cta}",
            category,
        )
    if extent * target.warp_size > limits.maximum_threads_per_cta:
        out.add(
            "TARGET_THREAD_LIMIT",
            "roles",
            f"declared warps span {extent * target.warp_size} threads, above the "
            f"Target limit of {limits.maximum_threads_per_cta}",
            category,
        )

    for index, operation in enumerate(schedule.operations):
        if operation.kind not in target.operation_kinds:
            out.add(
                "TARGET_OPERATION_UNSUPPORTED",
                f"operations[{index}].kind",
                f"operation kind {operation.kind.value!r} is not admitted by Target "
                f"{target.target_id!r}",
                category,
            )

    for index, allocation in enumerate(schedule.allocations):
        if allocation.space not in target.memory_spaces:
            out.add(
                "TARGET_MEMORY_SPACE_UNSUPPORTED",
                f"allocations[{index}].space",
                f"memory space {allocation.space.value!r} is not admitted by Target "
                f"{target.target_id!r}",
                category,
            )
        if allocation.space is MemorySpace.GLOBAL:
            out.add(
                "ALLOCATION_GLOBAL",
                f"allocations[{index}].space",
                "an Allocation declares a CTA-scoped region; global buffers are not "
                "allocated by the Schedule",
                category,
            )
        if allocation.space is MemorySpace.TENSOR:
            _verify_tensor_columns(allocation, index, limits, out)

    _verify_instruction_commitments(schedule, target, out)
    _verify_descriptor_commitments(schedule, out)
    _verify_swizzle_commitments(schedule, out)

    for index, buffer in enumerate(schedule.buffers):
        if buffer.space not in target.memory_spaces:
            out.add(
                "TARGET_MEMORY_SPACE_UNSUPPORTED",
                f"buffers[{index}].space",
                f"memory space {buffer.space.value!r} is not admitted by Target "
                f"{target.target_id!r}",
                category,
            )

    totals: dict[MemorySpace, int] = {}
    for allocation in schedule.allocations:
        totals[allocation.space] = totals.get(allocation.space, 0) + allocation.size_bytes
    for space, total in sorted(totals.items(), key=lambda item: item[0].value):
        capacity = limits.capacity(space)
        if capacity is not None and total > capacity:
            code = (
                "TARGET_SHARED_MEMORY_LIMIT"
                if space is MemorySpace.SHARED
                else "TARGET_TENSOR_MEMORY_LIMIT"
            )
            out.add(
                code,
                "allocations",
                f"{space.value} allocations total {total} bytes, above the Target "
                f"limit of {capacity}",
                category,
            )

    if schedule.grid is not None:
        for axis, (extent_value, maximum) in enumerate(
            zip(schedule.grid, limits.maximum_grid)
        ):
            if extent_value > maximum:
                out.add(
                    "TARGET_GRID_LIMIT",
                    f"grid[{axis}]",
                    f"grid extent {extent_value} exceeds the Target limit {maximum}",
                    category,
                )

    # `synchronization_contracts` names the handshakes the Target can actually lower.
    # A Schedule that declares barriers on a Target without one is not lowerable.
    if schedule.barriers and not (
        target.synchronization_contracts & {"mbarrier", "barrier.sync"}
    ):
        out.add(
            "TARGET_SYNCHRONIZATION_UNSUPPORTED",
            "barriers",
            f"Schedule declares barriers but Target {target.target_id!r} admits no "
            "barrier synchronization contract",
            category,
        )


def _verify_tensor_columns(allocation, index: int, limits, out: _Collector) -> None:
    """Tensor memory is allocated in columns; the Schedule sizes it in bytes.

    The artifact for the retained warp-specialized profile calls ``tmem.allocate(512)``
    while its Allocation is 131072 bytes, which is 256 columns. Nothing related the two,
    so the kernel reserved the whole array for a half-sized accumulator and halved TMEM
    occupancy. A declared column range makes the two comparable.
    """

    category = FindingCategory.HARDWARE_CONFORMANCE
    path = f"allocations[{index}]"
    implied = allocation.implied_tensor_columns

    if allocation.size_bytes % TMEM_COLUMN_BYTES:
        out.add(
            "ALLOCATION_TENSOR_GRANULARITY",
            f"{path}.size_bytes",
            f"{allocation.size_bytes} bytes is not a whole number of "
            f"{TMEM_COLUMN_BYTES}-byte tensor-memory columns",
            category,
        )
        return

    if allocation.tensor_columns is None:
        out.add(
            "ALLOCATION_TENSOR_COLUMNS_UNDECLARED",
            f"{path}.tensor_columns",
            f"tensor Allocation {allocation.name!r} does not commit to a column range; "
            f"{allocation.size_bytes} bytes implies {implied} columns, but the backend "
            "is free to reserve more",
            category,
            FindingSeverity.HINT,
        )
        return

    if allocation.tensor_columns != implied:
        out.add(
            "ALLOCATION_TENSOR_COLUMNS_MISMATCH",
            f"{path}.tensor_columns",
            f"Allocation {allocation.name!r} commits to {allocation.tensor_columns} "
            f"columns but its {allocation.size_bytes} bytes are {implied} columns",
            category,
        )

    capacity = limits.maximum_tensor_memory_bytes // TMEM_COLUMN_BYTES
    if allocation.tensor_columns > capacity:
        out.add(
            "TARGET_TENSOR_COLUMN_LIMIT",
            f"{path}.tensor_columns",
            f"{allocation.tensor_columns} columns exceeds the Target limit of "
            f"{capacity}",
            category,
        )


def _verify_instruction_commitments(
    schedule: Schedule, target: Target, out: _Collector
) -> None:
    category = FindingCategory.HARDWARE_CONFORMANCE
    buffers = {buffer.name: buffer for buffer in schedule.buffers}

    for index, operation in enumerate(schedule.operations):
        if operation.kind is not OperationKind.MMA:
            continue
        path = f"operations[{index}].parameters"
        instruction = getattr(operation.parameters, "instruction", None)
        tile = getattr(operation.parameters, "tile_shape", None)

        if instruction is None:
            out.add(
                "MMA_INSTRUCTION_UNDECLARED",
                f"{path}.instruction",
                f"operation {operation.op_id!r} does not name an instruction contract; "
                "the backend selects one and the choice is not inspectable",
                category,
                FindingSeverity.HINT,
            )
        elif instruction not in target.instruction_contracts:
            out.add(
                "TARGET_INSTRUCTION_UNSUPPORTED",
                f"{path}.instruction",
                f"instruction {instruction!r} is not admitted by Target "
                f"{target.target_id!r}",
                category,
            )

        if getattr(operation.parameters, "instruction_shape", None) is None:
            out.add(
                "MMA_INSTRUCTION_SHAPE_UNDECLARED",
                f"{path}.instruction_shape",
                f"operation {operation.op_id!r} does not commit to an atom M/N/K; the "
                "instruction shape is not the tile shape and the backend picks it",
                category,
                FindingSeverity.HINT,
            )

        if tile is None:
            out.add(
                "MMA_TILE_UNDECLARED",
                f"{path}.tile_shape",
                f"operation {operation.op_id!r} does not commit to an M/N/K tile",
                category,
                FindingSeverity.HINT,
            )
            continue

        for name in operation.writes:
            accumulator = buffers.get(name)
            if accumulator is None or len(accumulator.shape) != 2:
                continue
            if tuple(tile[:2]) != accumulator.shape:
                out.add(
                    "MMA_TILE_ACCUMULATOR_MISMATCH",
                    f"{path}.tile_shape",
                    f"tile M/N {tile[0]}x{tile[1]} does not match accumulator "
                    f"{name!r} shape {accumulator.shape[0]}x{accumulator.shape[1]}",
                    category,
                )


def _verify_descriptor_commitments(schedule: Schedule, out: _Collector) -> None:
    """`movement: tma` without a box is a flag, not a descriptor coordinate."""

    category = FindingCategory.HARDWARE_CONFORMANCE
    buffers = {buffer.name: buffer for buffer in schedule.buffers}
    for index, operation in enumerate(schedule.operations):
        if operation.kind is not OperationKind.LOAD:
            continue
        parameters = operation.parameters
        if getattr(parameters, "movement", None) is not LoadMovement.TMA:
            continue
        box = getattr(parameters, "descriptor_box", None)
        path = f"operations[{index}].parameters.descriptor_box"
        if box is None:
            out.add(
                "TMA_DESCRIPTOR_UNDECLARED",
                path,
                f"operation {operation.op_id!r} moves by TMA without committing to a "
                "descriptor box; the backend derives the tile it addresses",
                category,
                FindingSeverity.HINT,
            )
            continue
        for name in operation.writes:
            staged = buffers.get(name)
            if staged is None:
                continue
            if len(box) != len(staged.shape):
                out.add(
                    "TMA_DESCRIPTOR_RANK",
                    path,
                    f"descriptor box has {len(box)} extents for rank-"
                    f"{len(staged.shape)} destination {name!r}",
                    category,
                )
            elif tuple(box) != staged.shape:
                out.add(
                    "TMA_DESCRIPTOR_MISMATCH",
                    path,
                    f"descriptor box {box} does not match staging buffer {name!r} "
                    f"shape {staged.shape}",
                    category,
                )


def _verify_swizzle_commitments(schedule: Schedule, out: _Collector) -> None:
    """A shared operand feeding an MMA needs a declared swizzle to be reproducible."""

    category = FindingCategory.HARDWARE_CONFORMANCE
    buffers = {buffer.name: buffer for buffer in schedule.buffers}
    operands = {
        name
        for operation in schedule.operations
        if operation.kind is OperationKind.MMA
        for name in operation.reads
    }
    for index, buffer in enumerate(schedule.buffers):
        if (
            buffer.name in operands
            and buffer.space is MemorySpace.SHARED
            and buffer.swizzle is None
        ):
            out.add(
                "BUFFER_SWIZZLE_UNDECLARED",
                f"buffers[{index}].swizzle",
                f"shared MMA operand {buffer.name!r} does not commit to a swizzle; the "
                "backend picks one and the layout is neither inspectable nor verifiable",
                category,
                FindingSeverity.HINT,
            )


# --------------------------------------------------------------- data consistency

_ARITY = {
    OperationKind.LOAD: (1, 1, "load"),
    OperationKind.MMA: (2, 1, "mma"),
    OperationKind.EPILOGUE: (1, 1, "epilogue"),
    OperationKind.REDUCE_ARGMIN: (1, 1, "reduce_argmin"),
    OperationKind.REDUCE_SUM: (1, 1, "reduce_sum"),
    OperationKind.STORE: (1, 1, "store"),
}


def _verify_data_consistency(schedule: Schedule, out: _Collector) -> None:
    category = FindingCategory.DATA_CONSISTENCY

    buffers = {buffer.name: buffer for buffer in schedule.buffers}
    allocations = {item.name: item for item in schedule.allocations}
    roles = {role.name for role in schedule.roles}
    pipelines = {pipeline.name for pipeline in schedule.pipelines}

    # ---- buffer placement -------------------------------------------------
    for index, buffer in enumerate(schedule.buffers):
        path = f"buffers[{index}]"
        if buffer.mode in (BufferMode.INPUT, BufferMode.OUTPUT) and buffer.space is not MemorySpace.GLOBAL:
            out.add(
                "BUFFER_IO_SPACE",
                f"{path}.space",
                f"{buffer.mode.value} buffer {buffer.name!r} must live in global memory",
                category,
            )
        if buffer.space is MemorySpace.GLOBAL and buffer.allocation is not None:
            out.add(
                "BUFFER_GLOBAL_ALLOCATION",
                f"{path}.allocation",
                f"global buffer {buffer.name!r} must not claim a CTA Allocation",
                category,
            )
        if buffer.space is MemorySpace.REGISTER and buffer.allocation is not None:
            out.add(
                "BUFFER_REGISTER_ALLOCATION",
                f"{path}.allocation",
                f"register buffer {buffer.name!r} must not claim a CTA Allocation",
                category,
            )
        if buffer.space in (MemorySpace.SHARED, MemorySpace.TENSOR):
            if buffer.allocation is None:
                out.add(
                    "BUFFER_ALLOCATION_MISSING",
                    f"{path}.allocation",
                    f"{buffer.space.value} buffer {buffer.name!r} must name its Allocation",
                    category,
                )
            elif buffer.allocation not in allocations:
                out.add(
                    "BUFFER_ALLOCATION_UNKNOWN",
                    f"{path}.allocation",
                    f"unknown Allocation {buffer.allocation!r}",
                    category,
                )
            else:
                owner = allocations[buffer.allocation]
                if owner.space is not buffer.space:
                    out.add(
                        "BUFFER_SPACE_MISMATCH",
                        f"{path}.allocation",
                        f"buffer {buffer.name!r} is {buffer.space.value} but Allocation "
                        f"{owner.name!r} is {owner.space.value}",
                        category,
                    )
                elif buffer.byte_extent[1] > owner.size_bytes:
                    out.add(
                        "BUFFER_ALLOCATION_OVERFLOW",
                        f"{path}.byte_offset",
                        f"buffer {buffer.name!r} ends at byte {buffer.byte_extent[1]}, "
                        f"past the {owner.size_bytes}-byte Allocation {owner.name!r}",
                        category,
                    )

    # ---- aliasing inside one allocation -----------------------------------
    # Two buffers sharing an Allocation must not overlap unless the Schedule says so.
    # Nothing else in the pipeline detects this: the offsets are hand-authored and a
    # silent overlap corrupts one of the two tiles at runtime.
    by_allocation: dict[str, list[tuple[int, object]]] = {}
    for index, buffer in enumerate(schedule.buffers):
        if buffer.allocation in allocations:
            by_allocation.setdefault(buffer.allocation, []).append((index, buffer))
    for name, entries in sorted(by_allocation.items()):
        ordered = sorted(entries, key=lambda item: item[1].byte_extent)
        for (_, earlier), (later_index, later) in zip(ordered, ordered[1:]):
            if later.byte_extent[0] < earlier.byte_extent[1]:
                out.add(
                    "BUFFER_VIEW_OVERLAP",
                    f"buffers[{later_index}].byte_offset",
                    f"buffer {later.name!r} occupies bytes "
                    f"[{later.byte_extent[0]}, {later.byte_extent[1]}) of Allocation "
                    f"{name!r}, overlapping {earlier.name!r} at "
                    f"[{earlier.byte_extent[0]}, {earlier.byte_extent[1]})",
                    category,
                )

    # ---- operation wiring --------------------------------------------------
    writers: dict[str, list[str]] = {}
    readers: dict[str, list[str]] = {}
    for index, operation in enumerate(schedule.operations):
        path = f"operations[{index}]"
        if operation.role not in roles:
            out.add(
                "OP_ROLE_UNKNOWN",
                f"{path}.role",
                f"unknown role {operation.role!r}",
                category,
            )
        if operation.pipeline is not None and operation.pipeline not in pipelines:
            out.add(
                "OP_PIPELINE_UNKNOWN",
                f"{path}.pipeline",
                f"unknown pipeline {operation.pipeline!r}",
                category,
            )
        for name in operation.reads:
            if name not in buffers:
                out.add(
                    "OP_BUFFER_UNKNOWN",
                    f"{path}.reads",
                    f"unknown buffer {name!r}",
                    category,
                )
            else:
                readers.setdefault(name, []).append(operation.op_id)
        for name in operation.writes:
            if name not in buffers:
                out.add(
                    "OP_BUFFER_UNKNOWN",
                    f"{path}.writes",
                    f"unknown buffer {name!r}",
                    category,
                )
            else:
                writers.setdefault(name, []).append(operation.op_id)
        overlap = sorted(set(operation.reads) & set(operation.writes))
        for name in overlap:
            out.add(
                "OP_SELF_READ_WRITE",
                path,
                f"operation {operation.op_id!r} both reads and writes {name!r}",
                category,
            )
        _verify_operation_shape(operation, path, buffers, out)

    # ---- buffer roles ------------------------------------------------------
    # An input buffer that is written is the static form of the candidate mutating its
    # own inputs; downstream correctness compares against a reference recomputed from
    # those same buffers, so a writer here is a silent oracle bypass.
    for name, ops in sorted(writers.items()):
        buffer = buffers[name]
        if buffer.mode is BufferMode.INPUT:
            out.add(
                "INPUT_WRITTEN",
                f"buffers[{schedule.buffers.index(buffer)}].mode",
                f"input buffer {name!r} is written by {', '.join(sorted(ops))}; inputs "
                "are read-only for the whole Schedule",
                category,
            )
        if len(ops) > 1:
            out.add(
                "BUFFER_MULTIPLE_WRITERS",
                f"buffers[{schedule.buffers.index(buffer)}]",
                f"buffer {name!r} is written by {', '.join(sorted(ops))}",
                category,
            )
    for name, ops in sorted(readers.items()):
        buffer = buffers[name]
        if buffer.mode is BufferMode.SCRATCH and name not in writers:
            out.add(
                "BUFFER_UNPRODUCED",
                f"buffers[{schedule.buffers.index(buffer)}]",
                f"scratch buffer {name!r} is read by {', '.join(sorted(ops))} but never "
                "written",
                category,
            )

    for position, name in enumerate(schedule.outputs):
        if name not in buffers:
            out.add(
                "OUTPUT_UNKNOWN", f"outputs[{position}]", f"unknown buffer {name!r}", category
            )
            continue
        if buffers[name].mode is not BufferMode.OUTPUT:
            out.add(
                "OUTPUT_MODE",
                f"outputs[{position}]",
                f"buffer {name!r} is declared {buffers[name].mode.value}, not output",
                category,
            )
        if name not in writers:
            out.add(
                "OUTPUT_UNWRITTEN",
                f"outputs[{position}]",
                f"declared output {name!r} is never written",
                category,
            )
    for index, buffer in enumerate(schedule.buffers):
        if buffer.mode is BufferMode.OUTPUT and buffer.name not in schedule.outputs:
            out.add(
                "OUTPUT_NOT_EXPORTED",
                f"buffers[{index}].mode",
                f"buffer {buffer.name!r} is an output but is missing from outputs",
                category,
            )

    # ---- dependency graph ---------------------------------------------------
    op_ids = {operation.op_id for operation in schedule.operations}
    graph: dict[str, tuple[str, ...]] = {}
    for index, operation in enumerate(schedule.operations):
        path = f"operations[{index}].depends_on"
        edges = []
        for name in operation.depends_on:
            if name == operation.op_id:
                out.add(
                    "OP_SELF_DEPENDENCY", path, "operation depends on itself", category
                )
                continue
            if name not in op_ids:
                out.add(
                    "OP_DEPENDENCY_UNKNOWN", path, f"unknown operation {name!r}", category
                )
                continue
            edges.append(name)
        graph[operation.op_id] = tuple(edges)
    for node in sorted(_cycle_members(graph)):
        out.add(
            "OP_DEPENDENCY_CYCLE",
            "operations",
            f"dependency cycle includes {node!r}",
            category,
        )

    _verify_access_maps(schedule, buffers, out)


def _verify_operation_shape(operation, path: str, buffers, out: _Collector) -> None:
    category = FindingCategory.DATA_CONSISTENCY
    expected = _ARITY.get(operation.kind)
    if expected is not None:
        reads, writes, label = expected
        if len(operation.reads) < reads or len(operation.writes) < writes:
            out.add(
                "OP_ARITY",
                path,
                f"{label} requires at least {reads} read(s) and {writes} write(s), got "
                f"{len(operation.reads)} and {len(operation.writes)}",
                category,
            )
    if operation.kind is OperationKind.LOAD:
        for name in operation.reads:
            buffer = buffers.get(name)
            if buffer is not None and buffer.space is not MemorySpace.GLOBAL:
                out.add(
                    "OP_LOAD_SOURCE",
                    f"{path}.reads",
                    f"load source {name!r} is {buffer.space.value}; a load moves from "
                    "global memory",
                    category,
                )
    if operation.kind is OperationKind.STORE:
        for name in operation.writes:
            buffer = buffers.get(name)
            if buffer is not None and buffer.mode is not BufferMode.OUTPUT:
                out.add(
                    "OP_STORE_DESTINATION",
                    f"{path}.writes",
                    f"store destination {name!r} is {buffer.mode.value}, not an output",
                    category,
                )


def _verify_access_maps(schedule: Schedule, buffers, out: _Collector) -> None:
    category = FindingCategory.DATA_CONSISTENCY
    op_by_id = {operation.op_id: operation for operation in schedule.operations}
    axis_names = (
        {axis.name for axis in schedule.program_map.axes}
        if schedule.program_map is not None
        else set()
    )
    tiled_axes = (
        {axis.name for axis in schedule.program_map.axes if axis.is_tiled}
        if schedule.program_map is not None
        else set()
    )
    loop_iterators = {loop.iterator for loop in schedule.tile_loops}

    for index, access in enumerate(schedule.access_maps):
        path = f"access_maps[{index}]"
        operation = op_by_id.get(access.operation)
        if operation is None:
            out.add(
                "ACCESS_OPERATION_UNKNOWN",
                f"{path}.operation",
                f"unknown operation {access.operation!r}",
                category,
            )
            continue
        buffer = buffers.get(access.buffer)
        if buffer is None:
            out.add(
                "ACCESS_BUFFER_UNKNOWN",
                f"{path}.buffer",
                f"unknown buffer {access.buffer!r}",
                category,
            )
            continue
        if access.buffer not in set(operation.reads) | set(operation.writes):
            out.add(
                "ACCESS_EDGE_UNKNOWN",
                f"{path}.buffer",
                f"operation {access.operation!r} neither reads nor writes "
                f"{access.buffer!r}",
                category,
            )
        if buffer.space is not MemorySpace.GLOBAL:
            out.add(
                "ACCESS_BUFFER_LOCAL",
                f"{path}.buffer",
                f"buffer {access.buffer!r} is {buffer.space.value}; access maps address "
                "global memory",
                category,
            )
        if len(access.indices) != len(buffer.shape):
            out.add(
                "ACCESS_RANK",
                f"{path}.indices",
                f"{len(access.indices)} index components for rank-{len(buffer.shape)} "
                f"buffer {access.buffer!r}",
                category,
            )
        for position, component in enumerate(access.indices):
            component_path = f"{path}.indices[{position}]"
            if component.source is AccessIndexKind.DIMENSION:
                if component.dimension is not None and component.dimension >= len(buffer.shape):
                    out.add(
                        "ACCESS_DIMENSION_MISMATCH",
                        component_path,
                        f"dimension {component.dimension} is outside rank-"
                        f"{len(buffer.shape)} buffer {access.buffer!r}",
                        category,
                    )
            elif component.source is AccessIndexKind.LOOP_TILE:
                if component.name not in loop_iterators:
                    out.add(
                        "ACCESS_LOOP_UNKNOWN",
                        component_path,
                        f"unknown loop iterator {component.name!r}",
                        category,
                    )
            else:
                if component.name not in axis_names:
                    out.add(
                        "ACCESS_PROGRAM_AXIS_UNKNOWN",
                        component_path,
                        f"unknown program axis {component.name!r}",
                        category,
                    )
                elif (
                    component.source is AccessIndexKind.PROGRAM_TILE
                    and component.name not in tiled_axes
                ):
                    out.add(
                        "ACCESS_PROGRAM_AXIS_UNTILED",
                        component_path,
                        f"program axis {component.name!r} has tile 1 and carries no "
                        "offset vector; use source 'program'",
                        category,
                    )

    # Every global buffer an operation touches needs an addressing rule, otherwise
    # lowering has to invent one.
    if schedule.access_maps:
        declared = {(item.operation, item.buffer) for item in schedule.access_maps}
        for index, operation in enumerate(schedule.operations):
            for name in sorted(set(operation.reads) | set(operation.writes)):
                buffer = buffers.get(name)
                if buffer is None or buffer.space is not MemorySpace.GLOBAL:
                    continue
                if (operation.op_id, name) not in declared:
                    out.add(
                        "ACCESS_MAP_MISSING",
                        f"operations[{index}]",
                        f"operation {operation.op_id!r} touches global buffer {name!r} "
                        "without an access map",
                        category,
                    )


# ----------------------------------------------------------------- program safety


def _verify_program_safety(schedule: Schedule, out: _Collector) -> None:
    category = FindingCategory.PROGRAM_SAFETY

    roles = {role.name for role in schedule.roles}
    pipelines = {pipeline.name for pipeline in schedule.pipelines}
    barriers = {barrier.name: barrier for barrier in schedule.barriers}

    for index, barrier in enumerate(schedule.barriers):
        path = f"barriers[{index}]"
        for role in barrier.producers:
            if role not in roles:
                out.add(
                    "BARRIER_ROLE_UNKNOWN",
                    f"{path}.producers",
                    f"unknown role {role!r}",
                    category,
                )
        for role in barrier.consumers:
            if role not in roles:
                out.add(
                    "BARRIER_ROLE_UNKNOWN",
                    f"{path}.consumers",
                    f"unknown role {role!r}",
                    category,
                )
        if not barrier.producers or not barrier.consumers:
            out.add(
                "BARRIER_ENDPOINTS_EMPTY",
                path,
                f"barrier {barrier.name!r} needs at least one producer and one consumer",
                category,
            )
        if barrier.pipeline is not None and barrier.pipeline not in pipelines:
            out.add(
                "BARRIER_PIPELINE_UNKNOWN",
                f"{path}.pipeline",
                f"unknown pipeline {barrier.pipeline!r}",
                category,
            )

    signallers: dict[str, list[Operation]] = {}
    waiters: dict[str, list[Operation]] = {}
    for index, operation in enumerate(schedule.operations):
        path = f"operations[{index}]"
        for name in operation.signals:
            barrier = barriers.get(name)
            if barrier is None:
                out.add(
                    "OP_BARRIER_UNKNOWN",
                    f"{path}.signals",
                    f"unknown barrier {name!r}",
                    category,
                )
                continue
            signallers.setdefault(name, []).append(operation)
            if operation.role not in barrier.producers:
                out.add(
                    "OP_BARRIER_PRODUCER",
                    f"{path}.signals",
                    f"role {operation.role!r} signals {name!r} but is not one of its "
                    f"declared producers ({', '.join(barrier.producers) or 'none'})",
                    category,
                )
        for name in operation.waits:
            barrier = barriers.get(name)
            if barrier is None:
                out.add(
                    "OP_BARRIER_UNKNOWN",
                    f"{path}.waits",
                    f"unknown barrier {name!r}",
                    category,
                )
                continue
            waiters.setdefault(name, []).append(operation)
            if operation.role not in barrier.consumers:
                out.add(
                    "OP_BARRIER_CONSUMER",
                    f"{path}.waits",
                    f"role {operation.role!r} waits on {name!r} but is not one of its "
                    f"declared consumers ({', '.join(barrier.consumers) or 'none'})",
                    category,
                )

    # A declared handshake with no operation on one side never completes: the consumer
    # waits forever, or the producer's signal is never observed.
    for index, barrier in enumerate(schedule.barriers):
        path = f"barriers[{index}]"
        if barrier.name not in signallers:
            out.add(
                "BARRIER_UNUSED_PRODUCER",
                path,
                f"barrier {barrier.name!r} is never signalled by any operation",
                category,
            )
        if barrier.name not in waiters:
            out.add(
                "BARRIER_UNUSED_CONSUMER",
                path,
                f"barrier {barrier.name!r} is never waited on by any operation",
                category,
            )

    # A cross-role read-after-write is a race unless a barrier orders it.
    #
    # `depends_on` states program order, which only means anything inside one role: the
    # roles are distinct warp groups running concurrently, so nothing makes the
    # producer's store visible to the consumer's load except a barrier handshake. A
    # schedule that expresses a cross-role producer/consumer edge with `depends_on`
    # alone compiles, runs, and yields nondeterministic garbage.
    writer_of: dict[str, Operation] = {}
    for operation in schedule.operations:
        for name in operation.writes:
            writer_of[name] = operation
    order = {operation.op_id: index for index, operation in enumerate(schedule.operations)}
    for index, operation in enumerate(schedule.operations):
        for name in operation.reads:
            producer = writer_of.get(name)
            if producer is None or producer.op_id == operation.op_id:
                continue
            if producer.role == operation.role:
                # Same role: program order is real ordering. It must still run forwards.
                if producer.op_id in operation.depends_on and order.get(
                    producer.op_id, -1
                ) > index:
                    out.add(
                        "BARRIER_WAIT_UNORDERED",
                        f"operations[{index}].depends_on",
                        f"operation {operation.op_id!r} depends on {producer.op_id!r}, "
                        "which is declared later",
                        category,
                    )
                continue
            if set(operation.waits) & set(producer.signals):
                continue
            declared = (
                f"; {operation.op_id!r} declares depends_on {producer.op_id!r}, which "
                "orders operations inside one role but does not synchronize warps"
                if producer.op_id in operation.depends_on
                else ""
            )
            out.add(
                "OP_CROSS_ROLE_RACE",
                f"operations[{index}].reads",
                f"operation {operation.op_id!r} (role {operation.role!r}) reads "
                f"{name!r} written by {producer.op_id!r} (role {producer.role!r}) with "
                f"no barrier both sides use{declared}",
                category,
            )


def verify(schedule: Schedule, target: Target) -> tuple[Finding, ...]:
    """Return every contract violation, most-structural first. Never raises."""

    out = _Collector()
    _verify_schedule_semantics(schedule, out)
    _verify_hardware_conformance(schedule, target, out)
    _verify_data_consistency(schedule, out)
    _verify_program_safety(schedule, out)
    return out.result()
