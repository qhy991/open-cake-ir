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
    OperandSource,
    AccessIndexKind,
    BarrierMechanism,
    BufferMode,
    LoadMovement,
    MemorySpace,
    Operation,
    DType,
    OperationKind,
    Schedule,
)
from .analysis import (
    logical_registers_per_thread_lower_bound,
    residency_upper_bound,
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

    owner_by_warp: dict[int, str] = {}
    for index, role in enumerate(schedule.roles):
        for warp in role.warps:
            previous = owner_by_warp.get(warp)
            if previous is not None:
                out.add(
                    "ROLE_WARP_OVERLAP",
                    f"roles[{index}].warps",
                    f"warp {warp} belongs to both role {previous!r} and "
                    f"role {role.name!r}",
                    category,
                )
            else:
                owner_by_warp[warp] = role.name

    _verify_loop_nest(schedule, out)

    for duplicate in _duplicates(loop.iterator for loop in schedule.tile_loops):
        out.add(
            "LOOP_DUPLICATE_ITERATOR",
            "tile_loops",
            f"duplicate loop iterator {duplicate!r}",
            category,
        )

    if schedule.program_map is not None:
        program_map = schedule.program_map
        # A persistent grid launches multiprocessor_count * ctas_per_multiprocessor CTAs,
        # so without the commitment there is no count to launch. Requiring the
        # declaration keeps the grid size derived from one fact instead of restated.
        if program_map.persistent and (
            schedule.residency is None
            or schedule.residency.ctas_per_multiprocessor is None
        ):
            out.add(
                "PERSISTENT_WITHOUT_RESIDENCY",
                "program_map.persistent",
                "a persistent grid is sized from the residency this Schedule commits to; "
                "declare residency.ctas_per_multiprocessor",
                category,
            )
        if program_map.traversal is not None:
            declared = [item.name for item in program_map.axes]
            if sorted(program_map.traversal) != sorted(declared):
                out.add(
                    "TRAVERSAL_NOT_A_PERMUTATION",
                    "program_map.traversal",
                    f"traversal {list(program_map.traversal)} must order every declared "
                    f"axis exactly once; the axes are {declared}",
                    category,
                )
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
        # A loop that runs once is the same program as no loop, and the IR now has both
        # spellings -- a Schedule whose axis is already resident declares no tile loop.
        # Shapes here are static, so the trip count is knowable and one of the two forms
        # has to be the form. This is the one that carries an iterator nothing reads.
        buffer = schedule.buffer(loop.buffer)
        if buffer is not None and loop.dimension < len(buffer.shape):
            extent = buffer.shape[loop.dimension]
            if (extent + loop.tile - 1) // loop.tile < 2:
                out.add(
                    "TILE_LOOP_SINGLE_TRIP",
                    f"tile_loops[{index}].tile",
                    f"loop {loop.name!r} walks {extent} in tiles of {loop.tile}, which is "
                    f"one trip; a Schedule that does not iterate declares no tile loop",
                    category,
                )
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

    # A role interval may begin above zero, so the budget is bounded by the highest
    # warp index in use rather than by how many warps were declared.
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
    _verify_epilogue_commitments(schedule, out)
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
                f"operation {operation.op_id!r} does not commit to an MMA atom: "
                "contract, shape, CTA group, operand source and operand major mode are "
                "all left to the backend and none of them is inspectable",
                category,
                FindingSeverity.HINT,
            )
        else:
            # A contract names the dtypes the hardware will read and accumulate in. The
            # Target admits contracts by name and nothing checked the name against the
            # operands, so a Schedule could declare a bf16 contract over fp32 buffers:
            # accepted, lowered, and 0.06 off on a B200, because `tl.dot` quietly picks
            # TF32 for fp32 inputs. Adding a contract adds a row here.
            admitted = _CONTRACT_DTYPES.get(instruction.contract)
            if admitted is not None:
                operands, accumulate = admitted
                for name in operation.reads:
                    buffer = buffers.get(name)
                    if buffer is not None and buffer.dtype not in operands:
                        out.add(
                            "MMA_OPERAND_DTYPE_DIFFERS",
                            f"{path}.instruction.contract",
                            f"contract {instruction.contract!r} reads "
                            f"{'/'.join(sorted(d.value for d in operands))} but "
                            f"{name!r} is {buffer.dtype.value}",
                            category,
                        )
                written = buffers.get(operation.writes[0]) if operation.writes else None
                if written is not None and written.dtype is not accumulate:
                    out.add(
                        "MMA_ACCUMULATOR_DTYPE_DIFFERS",
                        f"{path}.instruction.contract",
                        f"contract {instruction.contract!r} accumulates in "
                        f"{accumulate.value} but {written.name!r} is "
                        f"{written.dtype.value}",
                        category,
                    )
            if instruction.contract not in target.instruction_contracts:
                out.add(
                    "TARGET_INSTRUCTION_UNSUPPORTED",
                    f"{path}.instruction.contract",
                    f"instruction {instruction.contract!r} is not admitted by Target "
                    f"{target.target_id!r}",
                    category,
                )
            _verify_atom_placement(operation, instruction, path, out)
            if (
                instruction.shape is not None
                and tile is not None
                and tile[0] != instruction.shape[0]
            ):
                out.add(
                    "MMA_TILE_INSTRUCTION_MISMATCH",
                    f"{path}.tile_shape",
                    f"tile M {tile[0]} differs from atom M {instruction.shape[0]}",
                    category,
                )
            if (
                instruction.shape is not None
                and tile is not None
                and tile[2] % instruction.shape[2]
            ):
                out.add(
                    "MMA_TILE_INSTRUCTION_MISMATCH",
                    f"{path}.tile_shape",
                    f"tile K {tile[2]} is not a whole number of atom K steps "
                    f"({instruction.shape[2]})",
                    category,
                )
            if instruction.operand_source is OperandSource.SHARED:
                for name in operation.reads:
                    operand = buffers.get(name)
                    if operand is not None and operand.space is not MemorySpace.SHARED:
                        out.add(
                            "MMA_OPERAND_SOURCE_MISMATCH",
                            f"{path}.instruction.operand_source",
                            f"atom reads operands from shared memory but {name!r} is "
                            f"{operand.space.value}",
                            category,
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


def _verify_epilogue_commitments(schedule: Schedule, out: _Collector) -> None:
    """An epilogue's sub-tiling and its accumulator copy atom are scheduling choices."""

    category = FindingCategory.HARDWARE_CONFORMANCE
    buffers = {buffer.name: buffer for buffer in schedule.buffers}
    for index, operation in enumerate(schedule.operations):
        if operation.kind is not OperationKind.EPILOGUE:
            continue
        path = f"operations[{index}].parameters"
        subtile = getattr(operation.parameters, "subtile", None)
        atom = getattr(operation.parameters, "source_atom", None)

        if subtile is None:
            out.add(
                "EPILOGUE_SUBTILE_UNDECLARED",
                f"{path}.subtile",
                f"operation {operation.op_id!r} does not commit to a sub-tiling of its "
                "accumulator; the backend chooses one and its register cost is hidden",
                category,
                FindingSeverity.HINT,
            )
        else:
            for name in operation.reads:
                source = buffers.get(name)
                if source is None or source.space is not MemorySpace.TENSOR:
                    continue
                if len(source.shape) == 2 and any(
                    source.shape[axis] % subtile[axis] for axis in (0, 1)
                ):
                    out.add(
                        "EPILOGUE_SUBTILE_MISMATCH",
                        f"{path}.subtile",
                        f"sub-tile {subtile[0]}x{subtile[1]} does not divide "
                        f"accumulator {name!r} shape "
                        f"{source.shape[0]}x{source.shape[1]}",
                        category,
                    )

        reads_tensor = any(
            (buffers.get(name) is not None)
            and buffers[name].space is MemorySpace.TENSOR
            for name in operation.reads
        )
        if reads_tensor and atom is None:
            out.add(
                "EPILOGUE_SOURCE_ATOM_UNDECLARED",
                f"{path}.source_atom",
                f"operation {operation.op_id!r} reads tensor memory without committing "
                "to a copy atom",
                category,
                FindingSeverity.HINT,
            )


# A tensor-core atom places its operands explicitly; a tile-level dot leaves that to the
# backend. Requiring both to say the same things would force one of them to invent an
# answer, so the requirement follows the contract.
_PLACED_CONTRACT_PREFIXES = ("tcgen05.", "mma.sync.", "wgmma.")
_PLACEMENT_FIELDS = ("shape", "cta_group", "operand_source", "operand_major")


def _verify_atom_placement(operation, instruction, path: str, out: _Collector) -> None:
    category = FindingCategory.HARDWARE_CONFORMANCE
    placed = instruction.contract.startswith(_PLACED_CONTRACT_PREFIXES)
    declared = [
        field
        for field in _PLACEMENT_FIELDS
        if getattr(instruction, field, None) is not None
    ]
    if placed:
        missing = [f for f in _PLACEMENT_FIELDS if f not in declared]
        if missing:
            out.add(
                "MMA_PLACEMENT_UNDECLARED",
                f"{path}.instruction",
                f"instruction {instruction.contract!r} places its operands explicitly "
                f"but {operation.op_id!r} leaves {', '.join(missing)} to the backend",
                category,
                FindingSeverity.HINT,
            )
    elif declared:
        out.add(
            "MMA_PLACEMENT_UNSUPPORTED",
            f"{path}.instruction",
            f"instruction {instruction.contract!r} does not place its operands, so "
            f"{', '.join(declared)} would be a commitment the backend cannot honour",
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

# What each admitted instruction contract reads and accumulates in. The Target names the
# contracts it admits; this is what those names mean, and it is here rather than in the
# Target because a Target describes hardware and this is a property of the instruction.
_CONTRACT_DTYPES = {
    "tcgen05.mma.cta_group::1.kind::f16": ({DType.BF16, DType.FP16}, DType.FP32),
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32": ({DType.BF16}, DType.FP32),
    "triton.dot.bf16_fp32": ({DType.BF16}, DType.FP32),
}


_ARITY = {
    OperationKind.LOAD: (1, 1, "load"),
    OperationKind.MMA: (2, 1, "mma"),
    OperationKind.EPILOGUE: (1, 1, "epilogue"),
    OperationKind.REDUCE_ARGMIN: (1, 1, "reduce_argmin"),
    OperationKind.REDUCE: (1, 1, "reduce"),
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

    # ---- loop-carried lifetime --------------------------------------------
    # A register value produced inside a tile loop does not survive it: registers hold
    # the loop body's live values and the next trip overwrites them. Shared and tensor
    # memory are allocations that outlive the loop, which is exactly how a warp-
    # specialized Schedule hands a staged tile from a producer role to a consumer that
    # reads it outside the loop, so the rule follows the declared space rather than the
    # nesting alone. The exception in registers is a reduction's result, which the loop
    # carries by construction.
    for loop in schedule.tile_loops:
        body = set(loop.body)
        carried = {
            name
            for op_id in loop.body
            if (op := schedule.operation(op_id)) is not None
            and (
                op.kind in (OperationKind.REDUCE, OperationKind.REDUCE_ARGMIN)
                # A contraction whose loop walks K is a reduction too, and its result is
                # carried by the same construction. Which loops do that is derived from
                # the operands rather than declared, so a Schedule that tiles the output
                # axis instead still hits the rule -- and should, because that shape has
                # no accumulator to carry.
                or (
                    op.kind is OperationKind.MMA
                    and schedule.mma_accumulates_over(op, loop)
                )
            )
            for name in op.writes
        }
        for name, producers in sorted(writers.items()):
            if name in carried or not set(producers) <= body:
                continue
            if buffers[name].space is not MemorySpace.REGISTER:
                continue
            escaping = sorted(set(readers.get(name, ())) - body)
            if escaping:
                # A contraction accumulated across the loop is the case an author is
                # most likely to expect to work, because that is what a GEMM is. Saying
                # only "a reduction result" sends them looking for a rule they broke
                # rather than telling them a register accumulator is not one -- an
                # accumulator that outlives its loop has to live in a space that does.
                contraction = any(
                    (op := schedule.operation(op_id)) is not None
                    and op.kind is OperationKind.MMA
                    for op_id in producers
                )
                reason = (
                    "a contraction accumulated across a loop needs an accumulator in a "
                    "space that outlives it, and a register buffer does not"
                    if contraction
                    else "only a reduction result is carried out of a loop"
                )
                out.add(
                    "BUFFER_ESCAPES_LOOP",
                    f"buffers[{schedule.buffers.index(buffers[name])}]",
                    f"buffer {name!r} is written only inside {loop.name!r} but read by "
                    f"{', '.join(escaping)} outside it; {reason}",
                    category,
                )

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
    # One cycle is one defect. Reporting a finding per member gave an author six blocking
    # findings for one mistake, none of them pointing anywhere: the path was `operations`
    # every time. The paper asks feedback to be localized diagnostics, and a set of names
    # with no index is the other way to fail that -- noise without a place to look.
    members = sorted(_cycle_members(graph))
    if members:
        position = {op.op_id: index for index, op in enumerate(schedule.operations)}
        first = min(members, key=lambda name: position.get(name, 0))
        out.add(
            "OP_DEPENDENCY_CYCLE",
            f"operations[{position[first]}].depends_on",
            f"dependency cycle through {', '.join(repr(name) for name in members)}",
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
    # A contraction reads its two operands and nothing else. It used to be able to name
    # a third, which a formula then folded in, so the arithmetic after the dot lived
    # inside the same operation. With that arithmetic declared separately a third read
    # has no meaning, and an emitter that simply contracts the first two would drop it
    # and produce a kernel that computes something else without saying so.
    if operation.kind is OperationKind.MMA:
        staged = [
            name
            for name in operation.reads
            if (buffer := buffers.get(name)) is not None
            and buffer.space is not MemorySpace.GLOBAL
        ]
        if len(operation.reads) != 2 or len(staged) != 2:
            out.add(
                "MMA_OPERAND_COUNT",
                f"{path}.reads",
                f"a contraction reads exactly two staged operands, got "
                f"{list(operation.reads)}; arithmetic over its result is a separate "
                "operation",
                category,
            )

    # An arithmetic primitive takes what its op says it takes. A binary op reads two
    # buffers, or one buffer and a declared scalar; anything else is a Schedule asking
    # for arithmetic whose operands are not all named.
    if operation.kind is OperationKind.ELEMENTWISE:
        parameters = operation.parameters
        supplied = len(operation.reads) + (parameters.scalar is not None)
        if supplied != parameters.arity_needed:
            out.add(
                "ELEMENTWISE_ARITY",
                f"{path}.reads",
                f"{parameters.op.value} takes {parameters.arity_needed} operand(s); "
                f"{len(operation.reads)} read(s) and "
                f"{'a' if parameters.scalar is not None else 'no'} scalar were given",
                category,
            )
        elif len(operation.writes) == 1:
            result = buffers.get(operation.writes[0])
            reads = [buffers[name] for name in operation.reads if name in buffers]
            if result is not None and len(reads) == len(operation.reads):
                widest = max((r.shape for r in reads), key=len, default=())
                if tuple(result.shape) != tuple(widest):
                    out.add(
                        "ELEMENTWISE_SHAPE_MISMATCH",
                        f"{path}.writes",
                        f"{parameters.op.value} over "
                        f"{[list(r.shape) for r in reads]} yields {list(widest)}, but "
                        f"{result.name!r} is {list(result.shape)}",
                        category,
                    )
                for read in reads:
                    if len(read.shape) == len(widest):
                        if tuple(read.shape) != tuple(widest):
                            out.add(
                                "ELEMENTWISE_SHAPE_MISMATCH",
                                f"{path}.reads",
                                f"operand {read.name!r} {list(read.shape)} differs from "
                                f"{list(widest)} and is not narrower",
                                category,
                            )
                        continue
                    axis = parameters.broadcast_axis
                    if axis is None:
                        out.add(
                            "ELEMENTWISE_BROADCAST_UNDECLARED",
                            f"{path}.parameters.broadcast_axis",
                            f"operand {read.name!r} {list(read.shape)} is narrower than "
                            f"{list(widest)}; the axis it spans must be declared",
                            category,
                        )
                    elif len(read.shape) != 1 or axis >= len(widest) or read.shape[0] != widest[axis]:
                        out.add(
                            "ELEMENTWISE_BROADCAST",
                            f"{path}.parameters.broadcast_axis",
                            f"operand {read.name!r} {list(read.shape)} does not span "
                            f"axis {axis} of {list(widest)}",
                            category,
                        )

    # A sum collapses one axis, so its result is its input with that axis dropped. The
    # extent used to be restated in the operation, which put the same fact in two places
    # and left the pair uncheckable; deriving it from the buffers makes disagreement
    # between them a Finding instead of a kernel that reduces the wrong number of values.
    if operation.kind is OperationKind.REDUCE and operation.reads and operation.writes:
        source = buffers.get(operation.reads[0])
        result = buffers.get(operation.writes[0])
        axis = operation.parameters.axis
        if source is not None and result is not None:
            if axis >= len(source.shape):
                out.add(
                    "REDUCE_AXIS_OUT_OF_RANGE",
                    f"{path}.parameters.axis",
                    f"axis {axis} is outside {source.name!r}, which has "
                    f"{len(source.shape)} dimension(s)",
                    category,
                )
            else:
                collapsed = source.shape[:axis] + source.shape[axis + 1 :]
                if tuple(result.shape) != tuple(collapsed):
                    out.add(
                        "REDUCE_SHAPE_MISMATCH",
                        f"{path}.writes",
                        f"summing axis {axis} of {source.name!r} {list(source.shape)} "
                        f"yields {list(collapsed)}, but {result.name!r} is "
                        f"{list(result.shape)}",
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

    # A tile axis produces a staged extent. If the axis says 128 and the buffer it
    # stages into says 256, the two disagree about the same tile and one of them is
    # wrong; nothing downstream can tell which.
    for index, access in enumerate(schedule.access_maps):
        operation = op_by_id.get(access.operation)
        if operation is None or not operation.writes:
            continue
        # The staged side, not the written side. A load writes its tile and a store reads
        # it, so taking `writes[0]` compared a store's *global* output against the tile
        # axes. Every store in the corpus escaped that by a rank coincidence -- the global
        # buffer had one more dimension than the access map had vector components, so the
        # check below skipped it -- and the first rank-2 output made it a false positive
        # on a Schedule that was correct.
        staged = next(
            (
                buffer
                for name in list(operation.writes) + list(operation.reads)
                if (buffer := buffers.get(name)) is not None
                and buffer.space is not MemorySpace.GLOBAL
            ),
            None,
        )
        source = buffers.get(access.buffer)
        if staged is None or source is None or source.space is not MemorySpace.GLOBAL:
            continue
        if len(staged.shape) != sum(1 for c in access.indices if c.is_vector):
            continue
        position = 0
        for component in access.indices:
            if not component.is_vector:
                continue
            if component.source is AccessIndexKind.PROGRAM_TILE:
                axis = (
                    schedule.program_map.axis(component.name)
                    if schedule.program_map is not None
                    else None
                )
                expected = axis.tile if axis is not None else None
            elif component.source is AccessIndexKind.LOOP_TILE:
                loop = next(
                    (l for l in schedule.tile_loops if l.iterator == component.name), None
                )
                expected = loop.tile if loop is not None else None
            else:
                expected = source.shape[component.dimension] if component.dimension is not None else None
            if expected is not None and staged.shape[position] != expected:
                out.add(
                    "ACCESS_TILE_MISMATCH",
                    f"access_maps[{index}].indices[{position}]",
                    f"tile axis {component.name or component.dimension} carries "
                    f"{expected} but {staged.name!r} stages {staged.shape[position]}",
                    category,
                )
            position += 1

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


TARGET_SYNC_HINT = {"mbarrier", "barrier.sync"}


def _verify_program_safety(schedule: Schedule, out: _Collector) -> None:
    category = FindingCategory.PROGRAM_SAFETY

    roles = {role.name for role in schedule.roles}
    pipelines = {pipeline.name for pipeline in schedule.pipelines}
    barriers = {barrier.name: barrier for barrier in schedule.barriers}
    active = {operation.role for operation in schedule.operations}

    for index, allocation in enumerate(schedule.allocations):
        role = allocation.allocating_role
        if role is None:
            continue
        path = f"allocations[{index}].allocating_role"
        if role not in roles:
            out.add("ALLOCATION_ROLE_UNKNOWN", path, f"unknown role {role!r}", category)
        elif role not in active:
            # The allocation and its release are instructions, so they need a role with
            # a body to put them in. A role with no operations is not dispatched at all,
            # and the allocation would never be taken out.
            out.add(
                "ALLOCATION_ROLE_INACTIVE",
                path,
                f"role {role!r} has no operations, so there is no body to allocate in",
                category,
            )

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
        if barrier.mechanism is None:
            out.add(
                "BARRIER_MECHANISM_UNDECLARED",
                f"{path}.mechanism",
                f"barrier {barrier.name!r} does not say how it is realized; the Target "
                f"admits {', '.join(sorted(TARGET_SYNC_HINT))} and the backend chooses",
                category,
                FindingSeverity.HINT,
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
        if barrier.mechanism is not BarrierMechanism.MBARRIER:
            continue
        producers = signallers.get(barrier.name, [])
        unsupported = [
            operation
            for operation in producers
            if operation.produced_pipeline_kind is None
        ]
        for operation in unsupported:
            operation_index = schedule.operations.index(operation)
            out.add(
                "BARRIER_PIPELINE_PRODUCER_UNSUPPORTED",
                f"operations[{operation_index}].signals",
                f"operation {operation.op_id!r} cannot drive an mbarrier pipeline; "
                "the implemented producer kinds are TMA load and MMA",
                category,
            )
        kinds = {
            operation.produced_pipeline_kind
            for operation in producers
            if operation.produced_pipeline_kind is not None
        }
        if len(kinds) > 1:
            out.add(
                "BARRIER_PIPELINE_KIND_AMBIGUOUS",
                path,
                f"mbarrier {barrier.name!r} is signalled by incompatible pipeline "
                f"kinds {sorted(kind.value for kind in kinds)}",
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


def _verify_role_register_split(schedule: Schedule, target: Target, out: _Collector) -> None:
    """Hold a role-by-role register split to the CTA budget it divides.

    `setmaxnreg` redistributes a CTA's launch-time allocation; it does not create
    registers. So a Schedule whose roles ask for more in total than the CTA was given is
    asking the hardware for something it cannot do, and one that asks for less is leaving
    registers unreachable rather than choosing to.

    The instruction is warpgroup-wide, so a role that names a budget must occupy whole
    warpgroups. A role sharing a warpgroup with another role cannot have its own budget --
    they would issue conflicting `setmaxnreg` from the same warpgroup.
    """

    category = FindingCategory.HARDWARE_CONFORMANCE
    budgeted = [role for role in schedule.roles if role.registers_per_thread is not None]
    if not budgeted:
        return

    warps_per_group = target.warps_per_warpgroup
    for index, role in enumerate(schedule.roles):
        if role.registers_per_thread is None:
            continue
        first, count = role.warps[0], len(role.warps)
        if first % warps_per_group or count % warps_per_group:
            out.add(
                "ROLE_REGISTERS_NOT_WARPGROUP_ALIGNED",
                f"roles[{index}].registers_per_thread",
                f"role {role.name!r} holds warps {list(role.warps)}; a register budget is "
                f"issued per warpgroup, so it must start on and span whole groups of "
                f"{warps_per_group}",
                category,
            )

    if len(budgeted) != len(schedule.roles):
        unbudgeted = [role.name for role in schedule.roles if role.registers_per_thread is None]
        out.add(
            "ROLE_REGISTERS_PARTIAL",
            "roles",
            f"roles {unbudgeted} declare no register budget while others do; the split "
            "divides one CTA allocation, so it is stated for every role or for none",
            category,
        )
        return

    commitment = schedule.residency
    total = commitment.registers_per_thread if commitment is not None else None
    if total is None:
        out.add(
            "ROLE_REGISTERS_WITHOUT_TOTAL",
            "residency.registers_per_thread",
            "a role-by-role register split divides the CTA's allocation; declare the "
            "allocation it divides",
            category,
        )
        return

    threads = schedule.total_warp_extent * target.warp_size
    distributed = sum(
        len(role.warps) * target.warp_size * role.registers_per_thread
        for role in schedule.roles
    )
    if distributed != total * threads:
        out.add(
            "ROLE_REGISTERS_NOT_CONSERVED",
            "roles",
            f"the roles distribute {distributed} registers but the CTA was allocated "
            f"{total * threads} ({total} per thread across {threads} threads); "
            "setmaxnreg redistributes an allocation, it does not change its size",
            category,
        )


def _verify_residency_commitment(
    schedule, target: Target, upper_bound, out: _Collector
) -> None:
    """Hold a Schedule to the residency it declared.

    Without a declaration the derivation is only a report. With one it is a check, and a
    Schedule that asks for two resident CTAs while its own buffers admit one is refused
    with the resource that stopped it named -- which is the difference between telling an
    author its occupancy and telling it which declaration to change.
    """

    _verify_role_register_split(schedule, target, out)
    commitment = schedule.residency
    if commitment is None:
        return
    category = FindingCategory.HARDWARE_CONFORMANCE

    wanted = commitment.ctas_per_multiprocessor
    if wanted is not None:
        maximum = upper_bound.ctas_per_multiprocessor or 0
        if maximum < wanted:
            binding = upper_bound.binding
            out.add(
                "RESIDENCY_UNMET",
                "residency.ctas_per_multiprocessor",
                f"this Schedule commits to {wanted} CTA per multiprocessor but its "
                f"declarations admit at most {maximum}; {binding.resource} is the "
                f"tightest static upper bound ({binding.per_cta} of "
                f"{binding.per_multiprocessor} {binding.unit})",
                category,
            )

    budget = commitment.registers_per_thread
    if budget is not None:
        needed = logical_registers_per_thread_lower_bound(schedule, target)
        if needed is not None and needed > budget and not commitment.allow_spill:
            out.add(
                "REGISTER_BUDGET_EXCEEDED",
                "residency.registers_per_thread",
                f"even with optimistic same-shape aliasing, declared logical register "
                f"storage needs at least {needed} registers per thread against a cap of "
                f"{budget}; it cannot fit without spill, which this Schedule did not "
                "admit",
                category,
            )


def _report_residency(schedule: Schedule, target: Target, out: _Collector) -> None:
    """Which declared resource bounds residency, and at what cost.

    This is the attribution half of the paper's `performance analysis` report. There is
    no cost estimate: a Target declares no clock and no bandwidth, so a predicted time
    would be invented rather than analysed.
    """

    category = FindingCategory.HARDWARE_CONFORMANCE
    if schedule.target != target.target_id:
        return  # nothing to analyse against a Target this Schedule does not name
    upper_bound = residency_upper_bound(schedule, target)
    if upper_bound is None or upper_bound.binding is None:
        return
    _verify_residency_commitment(schedule, target, upper_bound, out)
    binding = upper_bound.binding
    if binding.ctas == 0:
        # A resource that admits no CTA at all is not a performance report. The Schedule
        # asks a multiprocessor for more than it has, so it cannot run as declared.
        out.add(
            "RESIDENCY_IMPOSSIBLE",
            "allocations" if binding.resource.endswith("memory") else "buffers",
            f"declared {binding.resource} need {binding.per_cta} {binding.unit} per CTA "
            f"but a multiprocessor has {binding.per_multiprocessor}; no CTA is resident",
            category,
        )
        return
    others = ", ".join(
        f"{b.resource} {b.ctas}"
        for b in sorted(upper_bound.bounds, key=lambda b: b.ctas)
        if b.resource != binding.resource
    )
    out.add(
        "RESIDENCY_BOUND",
        "allocations" if binding.resource.endswith("memory") else "roles",
        f"{binding.resource} bounds maximum possible residency to {binding.ctas} CTA "
        "per multiprocessor "
        f"({binding.per_cta} of {binding.per_multiprocessor} {binding.unit})"
        + (f"; the next bounds are {others}" if others else ""),
        category,
        FindingSeverity.REPORT,
    )

    per_thread = logical_registers_per_thread_lower_bound(schedule, target)
    if per_thread and binding.resource == "logical_register_storage":
        out.add(
            "REGISTER_PRESSURE",
            "buffers",
            f"declared logical register storage has an optimistic lower bound of "
            f"{per_thread} registers per thread across "
            f"{schedule.total_warp_extent * target.warp_size} threads; this bounds "
            "maximum possible residency but is not ptxas-measured allocation",
            category,
            FindingSeverity.REPORT,
        )


def verify(schedule: Schedule, target: Target) -> tuple[Finding, ...]:
    """Return every contract violation, most-structural first. Never raises."""

    out = _Collector()
    _verify_schedule_semantics(schedule, out)
    _verify_hardware_conformance(schedule, target, out)
    _verify_data_consistency(schedule, out)
    _verify_program_safety(schedule, out)
    _report_residency(schedule, target, out)
    return out.result()
