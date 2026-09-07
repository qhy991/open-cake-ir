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

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from .ir import (
    PLACED_CONTRACT_PREFIXES,
    PLACEMENT_FIELDS,
    TMEM_COLUMN_BYTES,
    OperandSource,
    AccessIndexKind,
    BarrierMechanism,
    BufferMode,
    PackedBlockFormat,
    ReductionAlgorithm,
    ReductionScope,
    RoundingMode,
    OverflowPolicy,
    LoadMovement,
    LoweringBackend,
    MemorySpace,
    Operation,
    DType,
    ElementwiseOp,
    OperationKind,
    Schedule,
)
from .analysis import (
    logical_register_pressure_per_thread,
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
    """One typed Compiler diagnostic, retained unchanged through Assessment.

    Severity owns whether lowering is blocked. Blocking findings normally also reject
    the Schedule; a backend capability refusal may explicitly leave acceptance intact.
    Reports and hints are advisory and cannot authorize acceptance or performance.

    The constructor keeps the verifier's typed category/severity signature. Callers of
    the former core-only boolean constructor must name the category and express lowering
    disposition through severity; no category is inferred from a diagnostic code.
    Both blocking dispositions remain dataclass fields for existing ``asdict`` consumers,
    but ``blocks_lowering`` is derived and cannot be independently supplied.
    """

    code: str
    path: str
    message: str
    category: FindingCategory
    severity: FindingSeverity = FindingSeverity.BLOCKING
    blocks_acceptance: bool | None = field(default=None, kw_only=True)
    blocks_lowering: bool = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.category, FindingCategory) or not isinstance(
            self.severity, FindingSeverity
        ):
            raise ValueError("Finding category and severity must be typed")
        object.__setattr__(self, "blocks_lowering", self.severity is FindingSeverity.BLOCKING)
        if self.blocks_acceptance is None:
            object.__setattr__(self, "blocks_acceptance", self.blocks_lowering)
        elif not isinstance(self.blocks_acceptance, bool):
            raise ValueError("Finding blocks_acceptance must be a boolean")
        if self.blocks_acceptance and not self.blocks_lowering:
            raise ValueError("non-blocking Findings cannot block acceptance")

    def to_dict(self) -> dict[str, object]:
        """Project the public diagnostic without losing either blocking disposition."""

        return {
            "code": self.code,
            "path": self.path,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "blocks_acceptance": self.blocks_acceptance,
            "blocks_lowering": self.blocks_lowering,
        }

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
        for index, axis in enumerate(axes):
            owner = schedule.buffer(axis.buffer)
            if axis.axis >= 3:
                out.add(
                    "PROGRAM_AXIS_NUMBER_RANGE",
                    f"program_map.axes[{index}].axis",
                    f"program dimension {axis.axis} is outside the supported range [0, 2]",
                    category,
                )
            if owner is None:
                out.add(
                    "PROGRAM_AXIS_BUFFER_UNKNOWN",
                    f"program_map.axes[{index}].buffer",
                    f"program axis {axis.name!r} names unknown buffer {axis.buffer!r}",
                    category,
                )
            elif axis.dimension >= len(owner.shape):
                out.add(
                    "PROGRAM_AXIS_DIMENSION_RANGE",
                    f"program_map.axes[{index}].dimension",
                    f"dimension {axis.dimension} is outside rank-{len(owner.shape)} "
                    f"buffer {axis.buffer!r}",
                    category,
                )
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
        if loop.stop is not None:
            program_map = schedule.program_map
            axis = (
                None
                if program_map is None
                else program_map.axis(loop.stop.program)
            )
            if axis is None:
                out.add(
                    "LOOP_STOP_PROGRAM_UNKNOWN",
                    f"tile_loops[{index}].stop.program",
                    f"query-derived loop stop names unknown program axis "
                    f"{loop.stop.program!r}",
                    category,
                )
            elif axis.tile != 1:
                out.add(
                    "LOOP_STOP_PROGRAM_TILED",
                    f"tile_loops[{index}].stop.program",
                    f"query-derived loop stop requires one scalar program coordinate, "
                    f"but {axis.name!r} has tile {axis.tile}",
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

    # Child-loop placement is part of declaration order, just like direct operations.
    # A non-contiguous body would move a root/outer invariant across a loop boundary.
    for index, loop in enumerate(schedule.tile_loops):
        positions = [order[op.op_id] for op in schedule.loop_operations(loop)]
        if positions and positions != list(range(min(positions), max(positions) + 1)):
            out.add(
                "LOOP_BODY_SCOPE_ORDER",
                f"tile_loops[{index}].body",
                f"expanded body of {loop.name!r} must be a contiguous sequence in "
                "operation declaration order",
                category,
            )


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

    _verify_role_register_split(schedule, target, out)

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

    capacity_bytes = limits.maximum_tensor_memory_bytes
    if capacity_bytes is None:
        # The Target-space finding already says tensor memory is unsupported.  Its
        # absence is not a zero-byte budget and must not be turned into one here.
        return
    capacity = capacity_bytes // TMEM_COLUMN_BYTES
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
        path = f"operations[{index}].parameters"
        instruction = getattr(operation.parameters, "instruction", None)
        if (
            instruction is not None
            and instruction.contract not in target.instruction_contracts
        ):
            out.add(
                "TARGET_INSTRUCTION_UNSUPPORTED",
                f"{path}.instruction.contract",
                f"instruction {instruction.contract!r} is not admitted by Target "
                f"{target.target_id!r}",
                category,
            )

        if operation.kind is OperationKind.ELEMENTWISE:
            contract_semantics = _ELEMENTWISE_INSTRUCTIONS.get(
                instruction.contract if instruction is not None else ""
            )
            if (
                instruction is not None
                and instruction.contract in target.instruction_contracts
                and (
                    contract_semantics is None
                    or contract_semantics[0] is not operation.parameters.op
                )
            ):
                out.add(
                    "ELEMENTWISE_INSTRUCTION_KIND_DIFFERS",
                    f"{path}.instruction.contract",
                    f"contract {instruction.contract!r} is admitted by the Target but "
                    f"does not implement elementwise {operation.parameters.op.value}",
                    category,
                )
            elif contract_semantics is not None:
                expected = contract_semantics[1]
                for name in (*operation.reads, *operation.writes):
                    buffer = buffers.get(name)
                    if buffer is not None and buffer.dtype is not expected:
                        out.add(
                            "ELEMENTWISE_INSTRUCTION_DTYPE_DIFFERS",
                            f"{path}.instruction.contract",
                            f"contract {instruction.contract!r} consumes and produces "
                            f"{expected.value}, but {name!r} is {buffer.dtype.value}",
                            category,
                        )
            continue

        if operation.kind is not OperationKind.MMA:
            continue
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
                data_reads = operation.reads[:2]
                for name in data_reads:
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
                if instruction.contract == _BLOCK_SCALE_MMA_CONTRACT:
                    for name in operation.reads[2:]:
                        buffer = buffers.get(name)
                        if buffer is not None and buffer.dtype is not DType.FP32:
                            out.add(
                                "MMA_SCALE_DTYPE_DIFFERS",
                                f"{path}.instruction.contract",
                                f"contract {instruction.contract!r} reads fp32 scales "
                                f"but {name!r} is {buffer.dtype.value}",
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
                for name in operation.reads[:2]:
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
# answer, so the requirement follows the contract. The lists live in `ir` because the
# authoring Schema projects the same fact.
_PLACED_CONTRACT_PREFIXES = PLACED_CONTRACT_PREFIXES
_PLACEMENT_FIELDS = PLACEMENT_FIELDS


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
        for name in operation.reads[:2]
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
    "triton.dot.fp32_ieee": ({DType.FP32}, DType.FP32),
    "triton.dot.fp32_tf32": ({DType.FP32}, DType.FP32),
    "triton.dot.fp8e4m3_block_scale_fp32": ({DType.FP8_E4M3}, DType.FP32),
}

_BLOCK_SCALE_MMA_CONTRACT = "triton.dot.fp8e4m3_block_scale_fp32"

_ELEMENTWISE_INSTRUCTIONS = {
    "libdevice.tanh.f32": (ElementwiseOp.TANH, DType.FP32),
    "ptx.fma.rn.f32": (ElementwiseOp.FMA, DType.FP32),
}


def _verify_packed_block_relations(schedule: Schedule, out: _Collector) -> None:
    """Hold raw Buffer storage to its declared packed-record ABI."""

    category = FindingCategory.DATA_CONSISTENCY
    for index, buffer in enumerate(schedule.buffers):
        relation = buffer.packed_block
        if relation is None:
            continue
        path = f"buffers[{index}]"
        contract = relation.contract
        if buffer.dtype is not DType.UINT8:
            out.add(
                "PACKED_BLOCK_DTYPE",
                f"{path}.dtype",
                f"packed record {buffer.name!r} must use raw uint8 storage, not "
                f"{buffer.dtype.value}",
                category,
            )
        if relation.record_axis >= len(buffer.shape):
            out.add(
                "PACKED_BLOCK_RECORD_AXIS",
                f"{path}.packed_block.record_axis",
                f"packed record axis {relation.record_axis} is outside "
                f"{buffer.name!r}, which has {len(buffer.shape)} dimension(s)",
                category,
            )
        elif relation.record_axis != len(buffer.shape) - 1:
            out.add(
                "PACKED_BLOCK_RECORD_AXIS",
                f"{path}.packed_block.record_axis",
                f"packed record bytes must occupy the contiguous last axis of "
                f"{buffer.name!r}, not axis {relation.record_axis}",
                category,
            )
        elif buffer.shape[relation.record_axis] != contract.record_bytes:
            out.add(
                "PACKED_BLOCK_RECORD_EXTENT",
                f"{path}.shape[{relation.record_axis}]",
                f"{relation.format.value} requires {contract.record_bytes} record "
                f"bytes, but {buffer.name!r} declares "
                f"{buffer.shape[relation.record_axis]}",
                category,
            )
        if buffer.stages != 1:
            out.add(
                "PACKED_BLOCK_STAGES",
                f"{path}.stages",
                f"packed raw storage has one physical record per index, but "
                f"{buffer.name!r} declares {buffer.stages} stages",
                category,
            )
        if buffer.byte_offset % contract.record_alignment_bytes:
            out.add(
                "PACKED_BLOCK_ALIGNMENT",
                f"{path}.byte_offset",
                f"{relation.format.value} requires {contract.record_alignment_bytes}-byte "
                f"record alignment, but {buffer.name!r} starts at {buffer.byte_offset}",
                category,
            )
        if buffer.scale_of is not None:
            out.add(
                "PACKED_BLOCK_SCALE_CONFLICT",
                f"{path}.packed_block",
                f"packed record {buffer.name!r} already owns its metadata fields and "
                "cannot also be an FP8 scale relation",
                category,
            )


_ARITY = {
    OperationKind.LOAD: (1, 1, "load"),
    OperationKind.MMA: (2, 1, "mma"),
    OperationKind.EPILOGUE: (1, 1, "epilogue"),
    OperationKind.REDUCE_ARGMIN: (1, 1, "reduce_argmin"),
    OperationKind.REDUCE: (1, 1, "reduce"),
    OperationKind.SCAN: (1, 1, "scan"),
    OperationKind.STORE: (1, 1, "store"),
}


def _verify_scale_relations(schedule: Schedule, buffers, out: _Collector) -> None:
    """Hold scale storage to the FP8 tensor relation that gives it meaning."""

    category = FindingCategory.DATA_CONSISTENCY
    for index, scale in enumerate(schedule.buffers):
        relation = scale.scale_of
        if relation is None:
            continue
        path = f"buffers[{index}].scale_of"
        data = buffers.get(relation.buffer)
        if data is None:
            out.add(
                "SCALE_TARGET_UNKNOWN",
                f"{path}.buffer",
                f"scale buffer {scale.name!r} names unknown data buffer "
                f"{relation.buffer!r}",
                category,
            )
            continue
        if data is scale:
            out.add(
                "SCALE_TARGET_SELF",
                f"{path}.buffer",
                f"scale buffer {scale.name!r} cannot scale itself",
                category,
            )
            continue
        if data.dtype is not DType.FP8_E4M3:
            out.add(
                "SCALE_DATA_DTYPE",
                f"{path}.buffer",
                f"block scales describe fp8_e4m3 data, but {data.name!r} is "
                f"{data.dtype.value}",
                category,
            )
        if scale.dtype is not DType.FP32:
            out.add(
                "SCALE_DTYPE",
                f"buffers[{index}].dtype",
                f"block scale {scale.name!r} must be fp32, not {scale.dtype.value}",
                category,
            )
        rank = len(data.shape)
        if len(relation.granularity) != rank:
            out.add(
                "SCALE_GRANULARITY_RANK",
                f"{path}.granularity",
                f"scale granularity has rank {len(relation.granularity)} but "
                f"{data.name!r} has rank {rank}",
                category,
            )
        if tuple(sorted(relation.axis_order)) != tuple(range(rank)):
            out.add(
                "SCALE_AXIS_ORDER",
                f"{path}.axis_order",
                f"scale axis order {list(relation.axis_order)} is not a permutation "
                f"of all {rank} axes of {data.name!r}",
                category,
            )
        if len(relation.granularity) == rank and tuple(
            sorted(relation.axis_order)
        ) == tuple(range(rank)):
            grouped = tuple(
                (extent + granularity - 1) // granularity
                for extent, granularity in zip(data.shape, relation.granularity)
            )
            expected = tuple(grouped[axis] for axis in relation.axis_order)
            if scale.shape != expected:
                out.add(
                    "SCALE_SHAPE_MISMATCH",
                    f"buffers[{index}].shape",
                    f"{scale.name!r} shape {list(scale.shape)} differs from the "
                    f"derived grouped shape {list(expected)} for {data.name!r}",
                    category,
                )

    # A load changes storage, not meaning. The scale load and the data load together
    # establish which staged tile the staged scale describes.
    loads = [
        operation
        for operation in schedule.operations
        if operation.kind is OperationKind.LOAD
    ]
    load_edges = {
        (operation.reads[0], operation.writes[0])
        for operation in loads
        if len(operation.reads) == 1 and len(operation.writes) == 1
    }
    for operation_index, operation in enumerate(schedule.operations):
        if operation.kind is not OperationKind.LOAD:
            continue
        if len(operation.reads) != 1 or len(operation.writes) != 1:
            continue
        source = buffers.get(operation.reads[0])
        destination = buffers.get(operation.writes[0])
        if source is None or destination is None:
            continue
        before = source.scale_of
        after = destination.scale_of
        if (before is None) != (after is None):
            out.add(
                "SCALE_RELATION_DROPPED",
                f"operations[{operation_index}]",
                f"load {operation.op_id!r} changes whether {source.name!r}/"
                f"{destination.name!r} is a scale buffer",
                category,
            )
            continue
        if before is None or after is None:
            continue
        if (
            before.granularity != after.granularity
            or before.axis_order != after.axis_order
            or (before.buffer, after.buffer) not in load_edges
        ):
            out.add(
                "SCALE_RELATION_DRIFT",
                f"operations[{operation_index}]",
                f"load {operation.op_id!r} does not preserve the scale relation and "
                "the corresponding data-load edge",
                category,
            )


def _verify_valid_extents(schedule: Schedule, buffers, out: _Collector) -> None:
    """Type-check the one runtime authority for a padded Buffer's valid prefix."""

    category = FindingCategory.DATA_CONSISTENCY
    for index, data in enumerate(schedule.buffers):
        relation = data.valid_extent
        if relation is None:
            continue
        path = f"buffers[{index}].valid_extent"
        if data.space is not MemorySpace.GLOBAL:
            out.add(
                "VALID_EXTENT_DATA_SPACE",
                path,
                f"runtime valid extents apply to global padded storage, but "
                f"{data.name!r} is {data.space.value}",
                category,
            )
        extent = buffers.get(relation.buffer)
        if extent is None:
            out.add(
                "VALID_EXTENT_BUFFER_UNKNOWN",
                f"{path}.buffer",
                f"{data.name!r} names unknown extent buffer {relation.buffer!r}",
                category,
            )
        elif extent is data:
            out.add(
                "VALID_EXTENT_SELF",
                f"{path}.buffer",
                f"{data.name!r} cannot supply its own runtime extent",
                category,
            )
        else:
            if extent.dtype is not DType.INT32:
                out.add(
                    "VALID_EXTENT_DTYPE",
                    f"buffers[{schedule.buffers.index(extent)}].dtype",
                    f"extent buffer {extent.name!r} must be int32, not "
                    f"{extent.dtype.value}",
                    category,
                )
            if extent.space is not MemorySpace.GLOBAL or extent.mode is not BufferMode.INPUT:
                out.add(
                    "VALID_EXTENT_BUFFER_CONTRACT",
                    f"buffers[{schedule.buffers.index(extent)}]",
                    f"extent buffer {extent.name!r} must be a global input",
                    category,
                )

        rank = len(data.shape)
        axes_valid = True
        if relation.dimension >= rank:
            axes_valid = False
            out.add(
                "VALID_EXTENT_DIMENSION",
                f"{path}.dimension",
                f"valid dimension {relation.dimension} is outside rank-{rank} "
                f"buffer {data.name!r}",
                category,
            )
        for position, axis in enumerate(relation.indexed_by):
            if axis >= rank:
                axes_valid = False
                out.add(
                    "VALID_EXTENT_INDEX_AXIS",
                    f"{path}.indexed_by[{position}]",
                    f"index axis {axis} is outside rank-{rank} buffer {data.name!r}",
                    category,
                )
        if relation.dimension in relation.indexed_by:
            axes_valid = False
            out.add(
                "VALID_EXTENT_INDEX_AXIS",
                f"{path}.indexed_by",
                f"valid dimension {relation.dimension} cannot also index its lengths",
                category,
            )
        if extent is not None and extent is not data and axes_valid:
            expected = tuple(data.shape[axis] for axis in relation.indexed_by)
            if extent.shape != expected:
                out.add(
                    "VALID_EXTENT_SHAPE_MISMATCH",
                    f"buffers[{schedule.buffers.index(extent)}].shape",
                    f"extent buffer {extent.name!r} shape {list(extent.shape)} differs "
                    f"from indexed data extents {list(expected)}",
                    category,
                )


def _verify_data_consistency(schedule: Schedule, out: _Collector) -> None:
    category = FindingCategory.DATA_CONSISTENCY

    buffers = {buffer.name: buffer for buffer in schedule.buffers}
    allocations = {item.name: item for item in schedule.allocations}
    roles = {role.name for role in schedule.roles}
    pipelines = {pipeline.name for pipeline in schedule.pipelines}

    _verify_packed_block_relations(schedule, out)
    _verify_scale_relations(schedule, buffers, out)
    _verify_valid_extents(schedule, buffers, out)

    # ---- buffer placement -------------------------------------------------
    for index, buffer in enumerate(schedule.buffers):
        path = f"buffers[{index}]"
        if (
            buffer.mode in (BufferMode.INPUT, BufferMode.OUTPUT, BufferMode.STATE)
            and buffer.space is not MemorySpace.GLOBAL
        ):
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
    loop_operations = {
        entry
        for loop in schedule.tile_loops
        for entry in loop.body
        if schedule.operation(entry) is not None
    }
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
            buffer = buffers.get(name)
            if (
                operation.kind is OperationKind.ATOMIC_RMW
                and operation.reads
                and operation.writes
                and name == operation.reads[0] == operation.writes[0]
                and buffer is not None
                and buffer.mode is BufferMode.STATE
            ):
                # Reading and writing one state location is the defined effect of RMW,
                # not an accidental alias. Every other overlap remains illegal.
                continue
            out.add(
                "OP_SELF_READ_WRITE",
                path,
                f"operation {operation.op_id!r} both reads and writes {name!r}",
                category,
            )
        _verify_operation_shape(
            schedule,
            operation,
            path,
            buffers,
            out,
            inside_tile_loop=operation.op_id in loop_operations,
        )

    # ---- loop-carried lifetime --------------------------------------------
    # A register value produced inside a tile loop does not survive it: registers hold
    # the loop body's live values and the next trip overwrites them. Shared and tensor
    # memory are allocations that outlive the loop, which is exactly how a warp-
    # specialized Schedule hands a staged tile from a producer role to a consumer that
    # reads it outside the loop, so the rule follows the declared space rather than the
    # nesting alone. The exception in registers is a reduction's result, which the loop
    # carries by construction.
    for loop in schedule.tile_loops:
        body = {op.op_id for op in schedule.loop_operations(loop)}
        for op_id in loop.body:
            operation = schedule.operation(op_id)
            if (
                operation is None
                or operation.kind is not OperationKind.TOP_K
                or not operation.parameters.across_loop
                or not operation.reads
            ):
                continue
            source = buffers.get(operation.reads[0])
            if source is not None and source.shape != (loop.tile,):
                out.add(
                    "TOP_K_LOOP_TILE_MISMATCH",
                    f"operations[{schedule.operations.index(operation)}].reads",
                    f"loop-carried top_k consumes one score per {loop.name!r} position, "
                    f"so {source.name!r} must have shape [{loop.tile}], got "
                    f"{list(source.shape)}",
                    category,
                )
            if operation.parameters.source_tiles_per_merge != 2:
                continue

            for reader_id in sorted(
                {
                    reader
                    for result in operation.writes
                    for reader in readers.get(result, ())
                    if reader in body
                }
            ):
                reader = schedule.operation(reader_id)
                if reader is None:
                    continue
                observed = sorted(set(reader.reads) & set(operation.writes))
                out.add(
                    "TOP_K_MERGE_CADENCE_OUTPUT_READ_IN_LOOP",
                    f"operations[{schedule.operations.index(reader)}].reads",
                    f"{reader.op_id!r} reads delayed top_k result(s) "
                    f"{', '.join(repr(name) for name in observed)} inside {loop.name!r}; "
                    "a two-source-tile merge exposes only the finalized state after "
                    "the loop",
                    category,
                )

            if schedule.lowering.backend is LoweringBackend.TRITON:
                options = loop.range_options
                unsupported = (
                    ("loop_unroll_factor", options.loop_unroll_factor, 1),
                    ("warp_specialize", options.warp_specialize, False),
                    ("flatten", options.flatten, False),
                )
                for field, actual, admitted in unsupported:
                    if actual == admitted:
                        continue
                    out.add(
                        "TRITON_TOP_K_TWO_TILE_CONTROL_FLOW_UNSUPPORTED",
                        f"tile_loops[{schedule.tile_loops.index(loop)}].range_options.{field}",
                        f"the current Triton two-source-tile top_k control flow "
                        f"requires {field}={admitted!r}, got {actual!r}; the Compiler "
                        "does not silently rewrite a declared loop option",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
        for op_id in loop.body:
            operation = schedule.operation(op_id)
            if operation is None or operation.kind is not OperationKind.ONLINE_SOFTMAX:
                continue
            if operation.reads:
                logits = buffers.get(operation.reads[0])
                values = buffers.get(operation.reads[1]) if len(operation.reads) > 1 else None
                if (
                    logits is not None
                    and len(logits.shape) == 2
                    and logits.shape[1] != loop.tile
                ) or (
                    values is not None
                    and len(values.shape) == 2
                    and values.shape[0] != loop.tile
                ):
                    out.add(
                        "ONLINE_SOFTMAX_LOOP_TILE_MISMATCH",
                        f"operations[{schedule.operations.index(operation)}].reads",
                        f"online_softmax selected extent must equal {loop.name!r} tile "
                        f"{loop.tile}",
                        category,
                    )
            if len(operation.writes) == 4:
                normalized = operation.writes[3]
                in_loop_readers = sorted(set(readers.get(normalized, ())) & body)
                if in_loop_readers:
                    out.add(
                        "ONLINE_SOFTMAX_FINAL_READ_IN_LOOP",
                        f"operations[{schedule.operations.index(operation)}].writes",
                        f"normalized output is finalized after {loop.name!r}, but is read "
                        f"inside it by {', '.join(in_loop_readers)}",
                        category,
                    )
        carried = {
            name
            for op_id in loop.body
            if (op := schedule.operation(op_id)) is not None
            and (
                (
                    op.kind is OperationKind.REDUCE
                    and op.parameters.across_loop
                )
                or op.kind is OperationKind.REDUCE_ARGMIN
                or (
                    op.kind is OperationKind.TOP_K
                    and op.parameters.across_loop
                )
                or op.kind is OperationKind.ONLINE_SOFTMAX
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

    loop_bodies = {op_id for loop in schedule.tile_loops for op_id in loop.body}
    for operation in schedule.operations:
        if (
            operation.kind is OperationKind.TOP_K
            and operation.parameters.across_loop
            and operation.op_id not in loop_bodies
        ):
            out.add(
                "TOP_K_LOOP_REQUIRED",
                f"operations[{schedule.operations.index(operation)}].parameters.across_loop",
                "loop-carried top_k must be declared inside one tile loop",
                category,
            )
        if (
            operation.kind is OperationKind.ONLINE_SOFTMAX
            and operation.op_id not in loop_bodies
        ):
            out.add(
                "ONLINE_SOFTMAX_LOOP_REQUIRED",
                f"operations[{schedule.operations.index(operation)}]",
                "online_softmax must be declared inside one selected-token tile loop",
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
    for buffer in schedule.buffers:
        if buffer.mode is not BufferMode.STATE:
            continue
        position = schedule.buffers.index(buffer)
        if buffer.name not in readers:
            out.add(
                "STATE_NOT_READ",
                f"buffers[{position}].mode",
                f"state buffer {buffer.name!r} is never read; use output for a "
                "write-only result",
                category,
            )
        if buffer.name not in writers:
            out.add(
                "STATE_NOT_WRITTEN",
                f"buffers[{position}].mode",
                f"state buffer {buffer.name!r} is never written; use input for "
                "read-only data",
                category,
            )

    if not schedule.outputs and not any(
        buffer.mode is BufferMode.STATE for buffer in schedule.buffers
    ):
        out.add(
            "OUTPUTS_EMPTY_WITHOUT_STATE",
            "outputs",
            "a Schedule with no returned output must update caller-owned state",
            category,
        )

    for position, name in enumerate(schedule.outputs):
        if name in schedule.outputs[:position]:
            # `outputs` is the order the caller receives its tensors in, so a name
            # appearing twice asks for one buffer in two slots. The host wrapper can
            # only allocate it once, and the count it reports would not match.
            out.add(
                "OUTPUT_DUPLICATE",
                f"outputs[{position}]",
                f"buffer {name!r} is exported more than once",
                category,
            )
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


def _verify_block_scaled_mma(operation, path: str, buffers, out: _Collector) -> None:
    """Prove scale association, then gate the first backend's exact static subset."""

    if len(operation.reads) != 4:
        return
    a, b, scale_a, scale_b = (buffers.get(name) for name in operation.reads)
    if any(buffer is None for buffer in (a, b, scale_a, scale_b)):
        return
    association_ok = True
    for position, (scale, data) in enumerate(((scale_a, a), (scale_b, b)), start=2):
        relation = scale.scale_of
        if relation is None or relation.buffer != data.name:
            association_ok = False
            out.add(
                "MMA_SCALE_ASSOCIATION",
                f"{path}.reads[{position}]",
                f"scale operand {scale.name!r} does not declare that it scales "
                f"data operand {data.name!r}",
                FindingCategory.DATA_CONSISTENCY,
            )
    if not association_ok:
        return

    relation_a = scale_a.scale_of
    relation_b = scale_b.scale_of
    assert relation_a is not None and relation_b is not None
    tile = operation.parameters.tile_shape
    supported = (
        len(a.shape) == len(b.shape) == 2
        and a.space is not MemorySpace.GLOBAL
        and b.space is not MemorySpace.GLOBAL
        and scale_a.space is not MemorySpace.GLOBAL
        and scale_b.space is not MemorySpace.GLOBAL
        and a.shape[1] == b.shape[1]
        and len(relation_a.granularity) == len(relation_b.granularity) == 2
        and relation_a.granularity[0] == 1
        and relation_a.axis_order == (1, 0)
        and relation_b.axis_order == (0, 1)
        and relation_b.granularity[0] == b.shape[0]
        and relation_a.granularity[1] == relation_b.granularity[1]
        and a.shape[1] % relation_a.granularity[1] == 0
        and a.shape[1] // relation_a.granularity[1] == 2
        and tile == (a.shape[0], b.shape[0], a.shape[1])
    )
    if not supported:
        out.add(
            "MMA_BLOCK_SCALE_UNLOWERABLE",
            f"{path}.parameters",
            "the Triton block-scale lowering requires rank-2 staged A/B, A "
            "granularity [1, block_k] stored [K-block, M], B granularity "
            "[N, block_k] stored [N-block, K-block], equal divisible K blocks, "
            "exactly two K blocks, and a tile equal to the staged contraction",
            FindingCategory.HARDWARE_CONFORMANCE,
        )


_ELEMENTWISE_FLOAT_DTYPES = frozenset({DType.BF16, DType.FP16, DType.FP32})


def _elementwise_result_dtype(operation, buffers) -> DType | None:
    """The one admitted arithmetic promotion relation.

    Same-typed floating operands preserve their dtype. FP32 mixed with one 16-bit
    floating format produces FP32; mixing BF16 with FP16 has no implicit answer. This
    is deliberately smaller than a framework promotion table because the Schedule needs
    one target-independent spelling, not every conversion a frontend happens to accept.
    """

    reads = [buffers.get(name) for name in operation.reads]
    if any(buffer is None for buffer in reads):
        return None
    dtypes = {buffer.dtype for buffer in reads if buffer is not None}
    if not dtypes or not dtypes <= _ELEMENTWISE_FLOAT_DTYPES:
        return None
    if len(dtypes) == 1:
        return next(iter(dtypes))
    if DType.FP32 in dtypes and len(dtypes) == 2:
        return DType.FP32
    return None


def _verify_reshape(operation, path: str, buffers, out: _Collector) -> None:
    """Verify the one admitted spelling of a shape-only register view."""

    category = FindingCategory.DATA_CONSISTENCY
    if len(operation.reads) != 1 or len(operation.writes) != 1:
        out.add(
            "RESHAPE_EDGE_COUNT",
            path,
            "reshape reads exactly one register view and writes exactly one register "
            f"view, got {len(operation.reads)} read(s) and "
            f"{len(operation.writes)} write(s)",
            category,
        )
        return

    source = buffers.get(operation.reads[0])
    result = buffers.get(operation.writes[0])
    if source is None or result is None:
        return
    if source.space is not MemorySpace.REGISTER or result.space is not MemorySpace.REGISTER:
        out.add(
            "RESHAPE_REGISTER_ONLY",
            path,
            f"reshape is a register view, but {source.name!r} is "
            f"{source.space.value} and {result.name!r} is {result.space.value}",
            category,
        )
    if source.dtype is not result.dtype:
        out.add(
            "RESHAPE_DTYPE_MISMATCH",
            f"{path}.writes",
            f"reshape preserves dtype, but {source.name!r} is {source.dtype.value} "
            f"and {result.name!r} is {result.dtype.value}",
            category,
        )
    if source.elements != result.elements:
        out.add(
            "RESHAPE_ELEMENT_COUNT",
            f"{path}.writes",
            f"reshape preserves element count, but {source.name!r} has "
            f"{source.elements} and {result.name!r} has {result.elements}",
            category,
        )
    if source.packed_block is not None or result.packed_block is not None:
        out.add(
            "RESHAPE_PACKED_BLOCK_UNSUPPORTED",
            path,
            "reshape has no packed-record interpretation; load raw record bytes into "
            "an ordinary register buffer before forming typed fields",
            category,
        )


def _verify_cast_policy(operation, path: str, source, result, out: _Collector) -> None:
    """Gate explicit representation policies without changing existing float casts."""

    category = FindingCategory.DATA_CONSISTENCY
    parameters = operation.parameters
    policy = {
        DType.INT8: (RoundingMode.TOWARD_ZERO, OverflowPolicy.FORBID),
        DType.FP16: (RoundingMode.NEAREST_EVEN, OverflowPolicy.IEEE),
    }
    expected = policy.get(result.dtype) if source.dtype is DType.FP32 else None
    if expected is None:
        out.add(
            "CAST_DTYPE_UNSUPPORTED",
            f"{path}.writes",
            f"the first cast slice admits fp32 to int8 and fp32 to fp16, not "
            f"{source.dtype.value} to {result.dtype.value}",
            category,
        )
        return
    rounding, overflow = expected
    if parameters.rounding is not rounding:
        out.add(
            "CAST_ROUNDING_UNSUPPORTED",
            f"{path}.parameters.rounding",
            f"fp32 to {result.dtype.value} requires {rounding.value}, not "
            f"{parameters.rounding.value if parameters.rounding is not None else 'omitted'}",
            category,
        )
    if parameters.overflow is not overflow:
        out.add(
            "CAST_OVERFLOW_UNSUPPORTED",
            f"{path}.parameters.overflow",
            f"fp32 to {result.dtype.value} requires {overflow.value}, not "
            f"{parameters.overflow.value if parameters.overflow is not None else 'omitted'}",
            category,
        )


def _verify_packed_store(
    schedule: Schedule,
    operation,
    path: str,
    destination,
    buffers,
    out: _Collector,
) -> None:
    """Distinguish byte copies from the one typed packed-record encoding contract."""

    category = FindingCategory.DATA_CONSISTENCY
    relation = destination.packed_block
    assert relation is not None

    if len(operation.writes) != 1:
        out.add(
            "PACKED_STORE_ARITY",
            path,
            "a packed-record store writes exactly one packed destination",
            category,
        )
        return

    if len(operation.reads) == 1:
        source = buffers.get(operation.reads[0])
        if source is not None and source.dtype is not DType.UINT8:
            out.add(
                "PACKED_STORE_RAW_DTYPE",
                f"{path}.reads[0]",
                f"one-input packed store is an identity copy of uint8 bytes, but "
                f"{source.name!r} is {source.dtype.value}",
                category,
            )
        return

    if relation.format is PackedBlockFormat.GGML_Q4_0_V1:
        if len(operation.reads) == len(relation.contract.fields):
            out.add(
                "PACKED_STORE_Q4_TYPED_UNSUPPORTED",
                f"{path}.reads",
                "typed Q4_0 encoding is not in the first packed-store slice; only raw "
                "uint8 identity copies are admitted for ggml_q4_0_v1",
                category,
            )
        else:
            out.add(
                "PACKED_STORE_ARITY",
                f"{path}.reads",
                "ggml_q4_0_v1 store reads one raw uint8 byte tile; its two-field "
                f"typed encoder is explicitly unsupported, got {len(operation.reads)} "
                "inputs",
                category,
            )
        return

    if len(operation.reads) != 3:
        out.add(
            "PACKED_STORE_ARITY",
            f"{path}.reads",
            "ggml_q8_1_v1 store reads either one raw uint8 byte tile or exactly "
            f"three typed fields in registry order, got {len(operation.reads)}",
            category,
        )
        return

    if len(destination.shape) != 2 or relation.record_axis != 1:
        out.add(
            "PACKED_STORE_LAYOUT_UNSUPPORTED",
            f"{path}.writes[0]",
            "the first typed ggml_q8_1_v1 encoder writes rank-two "
            "[record, 36] storage with record_axis 1",
            FindingCategory.HARDWARE_CONFORMANCE,
        )
        return

    accesses = [
        access
        for access in schedule.access_maps
        if access.operation == operation.op_id and access.buffer == destination.name
    ]
    access_ok = (
        len(accesses) == 1
        and len(accesses[0].indices) == 1
        and accesses[0].indices[0].source is AccessIndexKind.DIMENSION
        and accesses[0].indices[0].dimension == 0
    )
    if not access_ok:
        out.add(
            "PACKED_STORE_ACCESS_MAP",
            f"{path}.writes[0]",
            "typed ggml_q8_1_v1 store requires one destination AccessMap with the "
            "single record-prefix index dimension 0",
            category,
        )

    # This first encoder spans every record. Repeating it in another program or a
    # tile loop would race on the same bytes, even if every field shape matched.
    program_count = 1
    if schedule.program_map is not None:
        for axis in schedule.program_map.axes:
            owner = buffers.get(axis.buffer)
            if owner is not None and axis.dimension < len(owner.shape):
                program_count *= axis.tile_count(owner.shape[axis.dimension])
    elif schedule.grid is not None:
        for extent in schedule.grid:
            program_count *= extent
    if program_count != 1 or any(operation.op_id in loop.body for loop in schedule.tile_loops):
        out.add("PACKED_STORE_PROGRAM_OWNERSHIP", path,
                "a full-prefix typed packed Store requires one program and no tile-loop repetition",
                FindingCategory.PROGRAM_SAFETY)

    prefix = destination.shape[:1]
    record_count_mismatches: list[str] = []
    field_shape_mismatches: list[str] = []
    for position, (name, field) in enumerate(
        zip(operation.reads, relation.contract.fields)
    ):
        source = buffers.get(name)
        if source is None:
            continue
        if source.space is not MemorySpace.REGISTER:
            out.add(
                "PACKED_STORE_FIELD_SPACE",
                f"{path}.reads[{position}]",
                f"typed registry field {field.name!r} must be register-resident, but "
                f"{source.name!r} is in {source.space.value}",
                FindingCategory.HARDWARE_CONFORMANCE,
            )
        if source.dtype is not field.dtype:
            out.add(
                "PACKED_STORE_FIELD_DTYPE",
                f"{path}.reads[{position}]",
                f"registry field {field.name!r} is {field.dtype.value}, but "
                f"{source.name!r} is {source.dtype.value}",
                category,
            )
        expected_shape = prefix + (() if field.elements == 1 else (field.elements,))
        if source.shape != expected_shape:
            suffix = () if field.elements == 1 else (field.elements,)
            has_field_layout = (
                len(source.shape) == 1 + len(suffix)
                and source.shape[1:] == suffix
            )
            mismatch = (
                f"{field.name} requires {list(expected_shape)}, "
                f"{source.name} is {list(source.shape)}"
            )
            if has_field_layout:
                record_count_mismatches.append(mismatch)
            else:
                field_shape_mismatches.append(mismatch)
    if record_count_mismatches and not field_shape_mismatches:
        out.add(
            "PACKED_BLOCK_STORE_RECORD_COUNT",
            f"{path}.reads",
            "typed fields and destination must own the same record prefix: "
            + "; ".join(record_count_mismatches),
            category,
        )
    elif record_count_mismatches or field_shape_mismatches:
        out.add(
            "PACKED_STORE_FIELD_SHAPE",
            f"{path}.reads",
            "typed fields must share the destination record prefix and use registry "
            "payload extents: "
            + "; ".join(record_count_mismatches + field_shape_mismatches),
            category,
        )


def _verify_operation_shape(
    schedule: Schedule,
    operation,
    path: str,
    buffers,
    out: _Collector,
    *,
    inside_tile_loop: bool = False,
) -> None:
    category = FindingCategory.DATA_CONSISTENCY
    if operation.kind is OperationKind.RESHAPE:
        _verify_reshape(operation, path, buffers, out)
    if operation.kind is OperationKind.REDUCE_ARGMIN:
        # `reduce_argmin` is not a generic numeric reduction with an incidental
        # output type.  The admitted operation compares fp32 distances and returns
        # int32 source positions; making that contract explicit prevents a backend
        # gaining a new storage dtype from silently widening every argmin use.
        if operation.reads:
            source = buffers.get(operation.reads[0])
            if source is not None and source.dtype is not DType.FP32:
                out.add(
                    "REDUCE_ARGMIN_VALUE_DTYPE",
                    f"{path}.reads",
                    f"reduce_argmin compares fp32 values, but {source.name!r} is "
                    f"{source.dtype.value}",
                    category,
                )
        if operation.writes:
            indices = buffers.get(operation.writes[0])
            if indices is not None and indices.dtype is not DType.INT32:
                out.add(
                    "REDUCE_ARGMIN_INDEX_DTYPE",
                    f"{path}.writes",
                    f"reduce_argmin returns int32 source positions, but "
                    f"{indices.name!r} is {indices.dtype.value}",
                    category,
                )
    if operation.kind is OperationKind.CAST:
        if len(operation.reads) != 1 or len(operation.writes) != 1:
            out.add(
                "CAST_ARITY",
                path,
                "cast reads one numeric tile and writes one converted tile",
                category,
            )
        else:
            source = buffers.get(operation.reads[0])
            output = buffers.get(operation.writes[0])
            if source is not None and output is not None:
                if source.shape != output.shape:
                    out.add(
                        "CAST_SHAPE_MISMATCH",
                        f"{path}.writes",
                        f"cast preserves shape {list(source.shape)}, got "
                        f"{list(output.shape)}",
                        category,
                    )
                allowed = {DType.BF16, DType.FP16, DType.FP32}
                if operation.parameters.rounding is not None or output.dtype is DType.INT8:
                    _verify_cast_policy(operation, path, source, output, out)
                elif source.dtype not in allowed or output.dtype not in allowed:
                    out.add(
                        "CAST_DTYPE_UNSUPPORTED",
                        path,
                        "the admitted cast converts among bf16, fp16, and fp32",
                        category,
                    )
                if output.dtype is not operation.parameters.to:
                    out.add(
                        "CAST_RESULT_DTYPE",
                        f"{path}.writes",
                        f"cast(to={operation.parameters.to.value}) writes "
                        f"{output.dtype.value}",
                        category,
                    )
                if source.dtype is operation.parameters.to:
                    out.add(
                        "CAST_IDENTITY",
                        f"{path}.parameters.to",
                        "an identity cast is a second spelling of the unchanged tile",
                        category,
                    )
                if (
                    source.space is not MemorySpace.REGISTER
                    or output.space is not MemorySpace.REGISTER
                ):
                    out.add(
                        "CAST_SPACE",
                        path,
                        "cast operands must be resident in registers",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
    if operation.kind is OperationKind.INDEX_EXPAND:
        if len(operation.reads) != 1 or len(operation.writes) != 1:
            out.add(
                "INDEX_EXPAND_ARITY",
                path,
                "index_expand reads selected int32 indices and writes one flat int32 run",
                category,
            )
        else:
            source = buffers.get(operation.reads[0])
            output = buffers.get(operation.writes[0])
            if source is not None:
                if len(source.shape) != 1:
                    out.add(
                        "INDEX_EXPAND_SOURCE_SHAPE",
                        f"{path}.reads",
                        "index_expand consumes one rank-one selected-index tile",
                        category,
                    )
                if source.dtype is not DType.INT32:
                    out.add(
                        "INDEX_EXPAND_DTYPE",
                        f"{path}.reads",
                        f"index_expand input must be int32, got {source.dtype.value}",
                        category,
                    )
                if source.space is not MemorySpace.REGISTER:
                    out.add(
                        "INDEX_EXPAND_SPACE",
                        f"{path}.reads",
                        "index_expand input must be resident in registers",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
                if len(source.shape) == 1 and source.shape[0] & (source.shape[0] - 1):
                    out.add(
                        "INDEX_EXPAND_SOURCE_UNLOWERABLE",
                        f"{path}.reads",
                        "the Triton index_expand input extent must be a power of two",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
            if output is not None:
                expected = (
                    source.shape[0] * operation.parameters.extent
                    if source is not None and len(source.shape) == 1
                    else None
                )
                if expected is not None and output.shape != (expected,):
                    out.add(
                        "INDEX_EXPAND_RESULT_SHAPE",
                        f"{path}.writes",
                        f"index_expand must write [{expected}], got {list(output.shape)}",
                        category,
                    )
                if output.dtype is not DType.INT32:
                    out.add(
                        "INDEX_EXPAND_DTYPE",
                        f"{path}.writes",
                        f"index_expand output must be int32, got {output.dtype.value}",
                        category,
                    )
                if output.space is not MemorySpace.REGISTER:
                    out.add(
                        "INDEX_EXPAND_SPACE",
                        f"{path}.writes",
                        "index_expand output must stay in registers before store",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
            extent = operation.parameters.extent
            if extent & (extent - 1):
                out.add(
                    "INDEX_EXPAND_EXTENT_UNLOWERABLE",
                    f"{path}.parameters.extent",
                    "the Triton index_expand extent must be a power of two",
                    FindingCategory.HARDWARE_CONFORMANCE,
                )
            if not -(1 << 31) <= operation.parameters.sentinel < (1 << 31):
                out.add(
                    "INDEX_EXPAND_SENTINEL_RANGE",
                    f"{path}.parameters.sentinel",
                    "index_expand sentinel must fit signed int32",
                    category,
                )
    if operation.kind is OperationKind.ONLINE_SOFTMAX:
        if len(operation.reads) not in {2, 3} or len(operation.writes) != 4:
            out.add(
                "ONLINE_SOFTMAX_ARITY",
                path,
                "online_softmax reads logits, values, optional validity indices and "
                "writes running maximum, "
                "normalizer, weighted accumulator, and normalized output",
                category,
            )
        else:
            logits = buffers.get(operation.reads[0])
            values = buffers.get(operation.reads[1])
            validity = (
                buffers.get(operation.reads[2]) if len(operation.reads) == 3 else None
            )
            maximum = buffers.get(operation.writes[0])
            normalizer = buffers.get(operation.writes[1])
            accumulator = buffers.get(operation.writes[2])
            normalized = buffers.get(operation.writes[3])
            axis = operation.parameters.axis
            if logits is not None:
                if len(logits.shape) != 2 or axis != 1:
                    out.add(
                        "ONLINE_SOFTMAX_LOGITS_SHAPE",
                        f"{path}.reads",
                        "the admitted online_softmax lowering consumes rank-two "
                        "[rows, selected] logits and reduces axis 1",
                        category,
                    )
                if logits.dtype is not DType.FP32:
                    out.add(
                        "ONLINE_SOFTMAX_LOGITS_DTYPE",
                        f"{path}.reads",
                        f"online_softmax logits must be fp32, got {logits.dtype.value}",
                        category,
                    )
                if logits.space is not MemorySpace.REGISTER:
                    out.add(
                        "ONLINE_SOFTMAX_SPACE",
                        f"{path}.reads",
                        "online_softmax logits must be resident in registers",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
            if values is not None:
                if len(values.shape) != 2:
                    out.add(
                        "ONLINE_SOFTMAX_VALUE_SHAPE",
                        f"{path}.reads",
                        "online_softmax values must have shape [selected, value_dim]",
                        category,
                    )
                if values.dtype not in {DType.BF16, DType.FP16, DType.FP32}:
                    out.add(
                        "ONLINE_SOFTMAX_VALUE_DTYPE",
                        f"{path}.reads",
                        f"online_softmax values must be floating, got {values.dtype.value}",
                        category,
                    )
                if values.space is not MemorySpace.REGISTER:
                    out.add(
                        "ONLINE_SOFTMAX_SPACE",
                        f"{path}.reads",
                        "online_softmax values must be resident in registers",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
            if len(operation.reads) == 3:
                if operation.parameters.sentinel is None:
                    out.add(
                        "ONLINE_SOFTMAX_SENTINEL_REQUIRED",
                        f"{path}.parameters.sentinel",
                        "validity indices require one explicit sentinel",
                        category,
                    )
                if validity is not None:
                    selected = (
                        logits.shape[1]
                        if logits is not None and len(logits.shape) == 2
                        else None
                    )
                    if selected is not None and validity.shape != (selected,):
                        out.add(
                            "ONLINE_SOFTMAX_VALIDITY_SHAPE",
                            f"{path}.reads",
                            f"validity indices must have shape [{selected}], got "
                            f"{list(validity.shape)}",
                            category,
                        )
                    if validity.dtype is not DType.INT32:
                        out.add(
                            "ONLINE_SOFTMAX_VALIDITY_DTYPE",
                            f"{path}.reads",
                            f"validity indices must be int32, got {validity.dtype.value}",
                            category,
                        )
                    if validity.space is not MemorySpace.REGISTER:
                        out.add(
                            "ONLINE_SOFTMAX_SPACE",
                            f"{path}.reads",
                            "validity indices must be resident in registers",
                            FindingCategory.HARDWARE_CONFORMANCE,
                        )
            elif operation.parameters.sentinel is not None:
                out.add(
                    "ONLINE_SOFTMAX_SENTINEL_UNUSED",
                    f"{path}.parameters.sentinel",
                    "online_softmax sentinel has no validity-index input",
                    category,
                )
            if operation.parameters.sentinel is not None and not (
                -(1 << 31) <= operation.parameters.sentinel < (1 << 31)
            ):
                out.add(
                    "ONLINE_SOFTMAX_SENTINEL_RANGE",
                    f"{path}.parameters.sentinel",
                    "online_softmax sentinel must fit signed int32",
                    category,
                )
            if (
                logits is not None
                and values is not None
                and len(logits.shape) == 2
                and len(values.shape) == 2
                and logits.shape[1] != values.shape[0]
            ):
                out.add(
                    "ONLINE_SOFTMAX_SELECTED_MISMATCH",
                    f"{path}.reads",
                    f"logits select {logits.shape[1]} values but the value tile has "
                    f"{values.shape[0]} rows",
                    category,
                )
            rows = logits.shape[0] if logits is not None and len(logits.shape) == 2 else None
            columns = (
                values.shape[1] if values is not None and len(values.shape) == 2 else None
            )
            expected = ((rows,), (rows,), (rows, columns), (rows, columns))
            for buffer, shape, label in zip(
                (maximum, normalizer, accumulator, normalized),
                expected,
                ("maximum", "normalizer", "accumulator", "normalized output"),
            ):
                if buffer is None or rows is None or columns is None:
                    continue
                if buffer.shape != shape:
                    out.add(
                        "ONLINE_SOFTMAX_STATE_SHAPE",
                        f"{path}.writes",
                        f"online_softmax {label} must have shape {list(shape)}, got "
                        f"{list(buffer.shape)}",
                        category,
                    )
                if buffer.dtype is not DType.FP32:
                    out.add(
                        "ONLINE_SOFTMAX_STATE_DTYPE",
                        f"{path}.writes",
                        f"online_softmax {label} must be fp32, got {buffer.dtype.value}",
                        category,
                    )
                if buffer.space is not MemorySpace.REGISTER:
                    out.add(
                        "ONLINE_SOFTMAX_SPACE",
                        f"{path}.writes",
                        f"online_softmax {label} must stay in registers",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
    if operation.kind is OperationKind.TOP_K:
        if (
            operation.parameters.source_tiles_per_merge == 2
            and not operation.parameters.across_loop
        ):
            out.add(
                "TOP_K_MERGE_CADENCE_REQUIRES_ACROSS_LOOP",
                f"{path}.parameters.source_tiles_per_merge",
                "source_tiles_per_merge=2 changes a loop-carried merge cadence, but "
                "this top_k does not declare across_loop=true",
                category,
            )
        if len(operation.reads) != 1 or len(operation.writes) != 2:
            out.add(
                "TOP_K_ARITY",
                path,
                "top_k reads exactly one score tile and writes values then int32 "
                f"indices, got {len(operation.reads)} read(s) and "
                f"{len(operation.writes)} write(s)",
                category,
            )
        else:
            source = buffers.get(operation.reads[0])
            values = buffers.get(operation.writes[0])
            indices = buffers.get(operation.writes[1])
            k = operation.parameters.k
            if source is not None:
                if len(source.shape) != 1:
                    out.add(
                        "TOP_K_SOURCE_RANK",
                        f"{path}.reads",
                        f"top_k selects from one resident rank-one tile, but "
                        f"{source.name!r} has shape {list(source.shape)}",
                        category,
                    )
                elif k > source.shape[0] and not operation.parameters.across_loop:
                    out.add(
                        "TOP_K_K_OUT_OF_RANGE",
                        f"{path}.parameters.k",
                        f"top_k asks for {k} values from {source.name!r} with extent "
                        f"{source.shape[0]}",
                        category,
                    )
                if len(source.shape) == 1 and source.shape[0] & (source.shape[0] - 1):
                    out.add(
                        "TOP_K_SOURCE_UNLOWERABLE",
                        f"{path}.reads",
                        "the admitted SM100 Triton top_k merge uses power-of-two "
                        f"resident vectors, but {source.name!r} has extent "
                        f"{source.shape[0]}",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
                if source.space is not MemorySpace.REGISTER:
                    out.add(
                        "TOP_K_SOURCE_SPACE",
                        f"{path}.reads",
                        f"top_k reduces a resident register tile, but {source.name!r} "
                        f"is in {source.space.value}",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
                if source.dtype not in {DType.FP32, DType.INT32}:
                    out.add(
                        "TOP_K_VALUE_DTYPE",
                        f"{path}.reads",
                        f"the admitted top_k lowering orders fp32 or int32 values, but "
                        f"{source.name!r} is {source.dtype.value}",
                        category,
                    )
                if (
                    source.dtype is DType.INT32
                    and operation.parameters.across_loop
                ):
                    out.add(
                        "TOP_K_INT32_ACROSS_LOOP_UNLOWERABLE",
                        f"{path}.parameters.across_loop",
                        "the admitted signed-int32 top_k lowering orders one resident "
                        "tile; loop-carried int32 state is not implemented",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
            if values is not None:
                if values.shape != (k,):
                    out.add(
                        "TOP_K_SHAPE_MISMATCH",
                        f"{path}.writes",
                        f"top_k with k={k} writes values[{k}], but {values.name!r} has "
                        f"shape {list(values.shape)}",
                        category,
                    )
                if source is not None and values.dtype is not source.dtype:
                    out.add(
                        "TOP_K_VALUE_DTYPE",
                        f"{path}.writes",
                        f"top_k values preserve {source.name!r}'s {source.dtype.value} "
                        f"dtype, but {values.name!r} is {values.dtype.value}",
                        category,
                    )
                if values.space is not MemorySpace.REGISTER:
                    out.add(
                        "TOP_K_RESULT_SPACE",
                        f"{path}.writes",
                        f"top_k values stay in registers before an explicit store, but "
                        f"{values.name!r} is in {values.space.value}",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
            if indices is not None:
                if indices.shape != (k,):
                    out.add(
                        "TOP_K_SHAPE_MISMATCH",
                        f"{path}.writes",
                        f"top_k with k={k} writes indices[{k}], but {indices.name!r} "
                        f"has shape {list(indices.shape)}",
                        category,
                    )
                if indices.dtype is not DType.INT32:
                    out.add(
                        "TOP_K_INDEX_DTYPE",
                        f"{path}.writes",
                        f"top_k source positions are int32, but {indices.name!r} is "
                        f"{indices.dtype.value}",
                        category,
                    )
                if indices.space is not MemorySpace.REGISTER:
                    out.add(
                        "TOP_K_RESULT_SPACE",
                        f"{path}.writes",
                        f"top_k indices stay in registers before an explicit store, but "
                        f"{indices.name!r} is in {indices.space.value}",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
            if k & (k - 1):
                out.add(
                    "TOP_K_K_UNLOWERABLE",
                    f"{path}.parameters.k",
                    f"the admitted SM100 Triton top_k lowering requires power-of-two k, "
                    f"but k is {k}",
                    FindingCategory.HARDWARE_CONFORMANCE,
                )
    if operation.kind is OperationKind.SCAN and operation.reads and operation.writes:
        source = buffers.get(operation.reads[0])
        result = buffers.get(operation.writes[0])
        axis = operation.parameters.axis
        if source is not None and result is not None:
            if (
                source.dtype not in _ELEMENTWISE_FLOAT_DTYPES
                or result.dtype is not DType.FP32
            ):
                out.add(
                    "SCAN_DTYPE_MISMATCH",
                    f"{path}.writes",
                    f"a {operation.parameters.op.value} scan accumulates bf16/fp16/fp32 "
                    f"into fp32, but {source.name!r} is {source.dtype.value} and "
                    f"{result.name!r} is {result.dtype.value}",
                    category,
                )
            if axis >= len(source.shape):
                out.add(
                    "SCAN_AXIS_OUT_OF_RANGE",
                    f"{path}.parameters.axis",
                    f"axis {axis} is outside {source.name!r}, which has "
                    f"{len(source.shape)} dimension(s)",
                    category,
                )
            elif tuple(result.shape) != tuple(source.shape):
                out.add(
                    "SCAN_SHAPE_MISMATCH",
                    f"{path}.writes",
                    f"scanning axis {axis} of {source.name!r} {list(source.shape)} "
                    f"keeps that shape, but {result.name!r} is {list(result.shape)}",
                    category,
                )

    packed_destinations = [buffers[name] for name in operation.writes if name in buffers and buffers[name].packed_block is not None] if operation.kind is OperationKind.STORE else []
    expected = _ARITY.get(operation.kind)
    if expected is not None and not packed_destinations:
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
        # The first read is the data source. Later reads are permitted only as
        # AccessMap-owned runtime coordinates; _verify_access_maps proves their exact
        # set, placement and dtype. Treating every read as a second data source made the
        # operation graph unable to name those real dependencies.
        for name in operation.reads[:1]:
            buffer = buffers.get(name)
            if buffer is not None and buffer.space is not MemorySpace.GLOBAL:
                out.add(
                    "OP_LOAD_SOURCE",
                    f"{path}.reads",
                    f"load source {name!r} is {buffer.space.value}; a load moves from "
                    "global memory",
                    category,
                )
    if operation.kind is OperationKind.ATOMIC_RMW:
        if not -(1 << 31) <= operation.parameters.value < (1 << 31):
            out.add(
                "ATOMIC_VALUE_RANGE",
                f"{path}.parameters.value",
                "the admitted int32 atomic add scalar must be in signed 32-bit range",
                FindingCategory.HARDWARE_CONFORMANCE,
            )
        if len(operation.reads) != 2 or len(operation.writes) != 2:
            out.add(
                "ATOMIC_EDGE_COUNT",
                path,
                "the admitted atomic_rmw reads target then one runtime index and "
                "writes the same target then one returned-old-value buffer",
                category,
            )
        if operation.reads and operation.writes:
            target = buffers.get(operation.reads[0])
            result = buffers.get(operation.writes[-1])
            if operation.writes[0] != operation.reads[0]:
                out.add(
                    "ATOMIC_TARGET_EDGE",
                    f"{path}.writes",
                    "atomic_rmw must write the same state buffer it reads first",
                    category,
                )
            if target is not None:
                if target.mode is not BufferMode.STATE:
                    out.add(
                        "ATOMIC_TARGET_MODE",
                        f"{path}.reads",
                        f"atomic target {target.name!r} is {target.mode.value}, not state",
                        category,
                    )
                if target.dtype is not DType.INT32:
                    out.add(
                        "ATOMIC_TARGET_DTYPE",
                        f"{path}.reads",
                        f"the admitted atomic add targets int32, but {target.name!r} is "
                        f"{target.dtype.value}",
                        category,
                    )
            if result is not None:
                if result.space is not MemorySpace.REGISTER:
                    out.add(
                        "ATOMIC_RESULT_SPACE",
                        f"{path}.writes",
                        f"atomic old values stay in registers before an explicit store, "
                        f"but {result.name!r} is in {result.space.value}",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
                if result.dtype is not DType.INT32:
                    out.add(
                        "ATOMIC_RESULT_DTYPE",
                        f"{path}.writes",
                        f"int32 atomic add returns int32, but {result.name!r} is "
                        f"{result.dtype.value}",
                        category,
                    )
    if operation.kind is OperationKind.STORE:
        if len(operation.writes) != 1:
            out.add(
                "STORE_EDGE_COUNT",
                f"{path}.writes",
                f"store declares exactly one destination edge, got "
                f"{len(operation.writes)}",
                category,
            )
        for name in operation.writes:
            buffer = buffers.get(name)
            if buffer is not None and buffer.mode not in {
                BufferMode.OUTPUT,
                BufferMode.STATE,
            }:
                out.add(
                    "OP_STORE_DESTINATION",
                    f"{path}.writes",
                    f"store destination {name!r} is {buffer.mode.value}, not output/state",
                    category,
                )
        if packed_destinations:
            _verify_packed_store(schedule, operation, path, packed_destinations[0], buffers, out)
        if operation.reads and operation.writes and len(operation.reads) == 1:
            source = buffers.get(operation.reads[0])
            destination = buffers.get(operation.writes[0])
            if source is not None and destination is not None:
                supported = source.dtype is destination.dtype or (
                    source.dtype is DType.FP32
                    and destination.dtype in {DType.BF16, DType.FP16}
                )
                if not supported:
                    out.add(
                        "STORE_DTYPE_UNSUPPORTED",
                        f"{path}.writes",
                        f"store cannot convert {source.name!r} from "
                        f"{source.dtype.value} to {destination.name!r} "
                        f"{destination.dtype.value}; the admitted conversions are "
                        "identity and fp32 to bf16/fp16",
                        category,
                    )
    # An ordinary contraction reads two data operands. The one admitted block-scale
    # contract reads the same pair followed by their two related scales; the contract,
    # rather than a flag, is the single owner of that arity and meaning.
    if operation.kind is OperationKind.MMA:
        instruction = operation.parameters.instruction
        contract = instruction.contract if instruction is not None else None
        expected_reads = 4 if contract == _BLOCK_SCALE_MMA_CONTRACT else 2
        staged = [
            name
            for name in operation.reads
            if (buffer := buffers.get(name)) is not None
            and buffer.space is not MemorySpace.GLOBAL
        ]
        if len(operation.reads) != expected_reads or len(staged) != expected_reads:
            out.add(
                "MMA_OPERAND_COUNT",
                f"{path}.reads",
                f"contract {contract!r} reads exactly {expected_reads} staged "
                f"operands, got {list(operation.reads)}",
                category,
            )
        elif contract == _BLOCK_SCALE_MMA_CONTRACT:
            _verify_block_scaled_mma(operation, path, buffers, out)

    # An arithmetic primitive takes what its op says it takes. A binary op reads two
    # buffers, or one buffer and a declared scalar; anything else is a Schedule asking
    # for arithmetic whose operands are not all named.
    if operation.kind is OperationKind.ELEMENTWISE:
        parameters = operation.parameters
        if parameters.op is ElementwiseOp.FMA:
            if len(operation.writes) != 1:
                out.add(
                    "ELEMENTWISE_FMA_RESULT_COUNT", f"{path}.writes",
                    "fma writes exactly one FP32 register result", category,
                )
            for name in (*operation.reads, *operation.writes):
                buffer = buffers.get(name)
                if buffer is not None and buffer.space is not MemorySpace.REGISTER:
                    out.add(
                        "ELEMENTWISE_FMA_SPACE", path,
                        f"fma operand/result {name!r} must be register-resident",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
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
                inferred_dtype = _elementwise_result_dtype(operation, buffers)
                if inferred_dtype is None:
                    out.add(
                        "ELEMENTWISE_DTYPE_UNSUPPORTED",
                        f"{path}.reads",
                        f"{parameters.op.value} has no admitted promotion for "
                        f"{[read.dtype.value for read in reads]}",
                        category,
                    )
                elif result.dtype is not inferred_dtype:
                    out.add(
                        "ELEMENTWISE_RESULT_DTYPE",
                        f"{path}.writes",
                        f"{parameters.op.value} over "
                        f"{[read.dtype.value for read in reads]} produces "
                        f"{inferred_dtype.value}, but {result.name!r} is "
                        f"{result.dtype.value}",
                        category,
                    )
                # A canonical scalar [1] broadcasts without selecting a result axis.
                # Other singleton dimensions keep their existing explicit shape rules;
                # the instruction-specific FMA contract still requires equal shapes.
                shape_reads = [r for r in reads if not r.is_scalar] if parameters.op is not ElementwiseOp.FMA else reads
                widest = max((r.shape for r in (shape_reads or reads)), key=len, default=())
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
                    if read.is_scalar and parameters.op is not ElementwiseOp.FMA:
                        axis = parameters.broadcast_axis
                        if axis is not None and (axis >= len(widest) or widest[axis] != 1):
                            out.add(
                                "ELEMENTWISE_BROADCAST", f"{path}.parameters.broadcast_axis",
                                f"scalar operand {read.name!r} needs no broadcast axis; "
                                f"declared axis {axis} does not have extent one", category,
                            )
                        continue
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
        parameters = operation.parameters
        axis = parameters.axis
        if source is not None and result is not None:
            if parameters.algorithm is ReductionAlgorithm.XOR_TREE_32:
                if source.dtype is not DType.FP32 or result.dtype is not DType.FP32:
                    out.add(
                        "REDUCE_XOR_DTYPE",
                        f"{path}.writes",
                        "xor_tree_32 preserves the source-ordered fp32 additions and "
                        f"therefore requires fp32 input and output, but {source.name!r} "
                        f"is {source.dtype.value} and {result.name!r} is "
                        f"{result.dtype.value}",
                        category,
                    )
                if parameters.scope is not ReductionScope.CTA:
                    out.add(
                        "REDUCE_XOR_SCOPE",
                        f"{path}.parameters.scope",
                        "xor_tree_32 is the declared 32-lane CTA reduction contract",
                        category,
                    )
                if inside_tile_loop:
                    out.add(
                        "REDUCE_XOR_ACROSS_LOOP",
                        path,
                        "xor_tree_32 reduces one resident 32-value axis and cannot be "
                        "carried or repeated across a tile loop",
                        category,
                    )
            elif (
                source.dtype not in _ELEMENTWISE_FLOAT_DTYPES
                or result.dtype is not DType.FP32
            ):
                # Keep the historical backend-selected reduction contract unchanged.
                out.add(
                    "REDUCE_DTYPE_MISMATCH",
                    f"{path}.writes",
                    f"{parameters.op.value} reduces bf16/fp16/fp32 into fp32, but "
                    f"{source.name!r} is {source.dtype.value} and {result.name!r} is "
                    f"{result.dtype.value}",
                    category,
                )
            if axis >= len(source.shape):
                out.add(
                    "REDUCE_AXIS_OUT_OF_RANGE",
                    f"{path}.parameters.axis",
                    f"axis {axis} is outside {source.name!r}, which has "
                    f"{len(source.shape)} dimension(s)",
                    category,
                )
            else:
                if parameters.algorithm is ReductionAlgorithm.XOR_TREE_32:
                    if axis != len(source.shape) - 1:
                        out.add("REDUCE_XOR_AXIS", f"{path}.parameters.axis", "xor_tree_32 reduces the contiguous last axis", category)
                    if source.shape[axis] != 32:
                        out.add("REDUCE_XOR_EXTENT", f"{path}.parameters.axis", "xor_tree_32 requires extent 32", category)
                # Buffer scalars use [1], as scalar loads and the Python frontend do;
                # a rank-zero Buffer is not part of the IR.
                collapsed = source.shape[:axis] + source.shape[axis + 1 :] or (1,)
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
        typed_packed_store = (
            operation.kind is OperationKind.STORE
            and len(operation.reads) == 3
            and access.buffer in operation.writes
            and buffer.packed_block is not None
            and buffer.packed_block.format is PackedBlockFormat.GGML_Q8_1_V1
        )
        expected_access_rank = 1 if typed_packed_store else len(buffer.shape)
        if len(access.indices) != expected_access_rank:
            out.add(
                "ACCESS_RANK",
                f"{path}.indices",
                f"{len(access.indices)} index components for "
                + (
                    "the one-axis typed packed-record prefix"
                    if typed_packed_store
                    else f"rank-{len(buffer.shape)} buffer {access.buffer!r}"
                ),
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
                elif component.dimension is not None:
                    # A sub-range that runs past the axis would read whatever follows it
                    # in the allocation, so the bound is checked here rather than left
                    # for the emitter to mask. An offset at or past the end selects
                    # nothing, which is a declaration no operation can have meant.
                    size = buffer.shape[component.dimension]
                    if component.offset >= size or component.offset + component.span(size) > size:
                        out.add(
                            "ACCESS_DIMENSION_SUBRANGE",
                            component_path,
                            f"sub-range [{component.offset}, "
                            f"{component.offset + component.span(size)}) leaves dimension "
                            f"{component.dimension} of size {size} in buffer "
                            f"{access.buffer!r}",
                            category,
                        )
                    elif component.extent == size - component.offset:
                        # A range reaching the end of the axis is written by omitting
                        # `extent`. Spelling it out names the same elements a second way,
                        # and the two spellings lower to different source -- so the same
                        # access would pin two different digests.
                        out.add(
                            "ACCESS_SUBRANGE_NONCANONICAL",
                            component_path,
                            f"extent {component.extent} reaches the end of dimension "
                            f"{component.dimension}; omit it to name the same range",
                            category,
                        )
                    if (
                        position < len(buffer.shape)
                        and component.dimension != position
                        and component.offset < size
                        and component.offset + component.span(size) <= size
                        and component.offset + component.span(size) > buffer.shape[position]
                    ):
                        out.add(
                            "ACCESS_DIMENSION_COORDINATE_RANGE",
                            component_path,
                            f"dimension vector reaches {component.offset + component.span(size)}, "
                            f"but addressed dimension {position} of {buffer.name!r} "
                            f"has extent {buffer.shape[position]}",
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
                elif component.name not in {
                    loop.iterator for loop in schedule.enclosing_loops(operation)
                }:
                    out.add(
                        "ACCESS_LOOP_SCOPE",
                        component_path,
                        f"loop coordinate {component.name!r} is not visible in "
                        f"operation {operation.op_id!r}'s lexical scope",
                        category,
                    )
            elif component.source is AccessIndexKind.BUFFER:
                index_buffer = buffers.get(component.name)
                if index_buffer is None:
                    out.add(
                        "ACCESS_INDEX_BUFFER_UNKNOWN",
                        component_path,
                        f"unknown runtime index buffer {component.name!r}",
                        category,
                    )
                else:
                    if index_buffer.space is not MemorySpace.REGISTER:
                        out.add(
                            "ACCESS_INDEX_BUFFER_SPACE",
                            component_path,
                            f"runtime index buffer {component.name!r} is in "
                            f"{index_buffer.space.value}, not registers",
                            category,
                        )
                    if index_buffer.dtype is not DType.INT32:
                        out.add(
                            "ACCESS_INDEX_BUFFER_DTYPE",
                            component_path,
                            f"runtime index buffer {component.name!r} is "
                            f"{index_buffer.dtype.value}, not int32",
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
                elif component.source is AccessIndexKind.PROGRAM:
                    axis = schedule.program_map.axis(component.name)
                    owner = buffers.get(axis.buffer) if axis is not None else None
                    if (
                        axis is not None
                        and owner is not None
                        and axis.dimension < len(owner.shape)
                        and position < len(buffer.shape)
                    ):
                        program_extent = axis.tile_count(owner.shape[axis.dimension])
                        if program_extent > buffer.shape[position]:
                            out.add(
                                "ACCESS_PROGRAM_EXTENT_MISMATCH",
                                component_path,
                                f"program axis {component.name!r} spans "
                                f"{program_extent} coordinate(s), but dimension "
                                f"{position} of {access.buffer!r} has extent "
                                f"{buffer.shape[position]}",
                                category,
                            )

        indirect = [
            component
            for component in access.indices
            if component.source is AccessIndexKind.BUFFER
        ]
        # AccessMap, not the ordering of Operation.reads, owns the global source.
        # Keep one dtype relation and one Finding spelling for direct and runtime-
        # indexed loads alike.
        if operation.kind is OperationKind.LOAD and len(operation.writes) == 1:
            staged = buffers.get(operation.writes[0])
            if staged is not None and staged.dtype is not buffer.dtype:
                out.add(
                    "LOAD_DTYPE_MISMATCH",
                    f"operations[{schedule.operations.index(operation)}].writes",
                    f"load preserves {buffer.name!r}'s {buffer.dtype.value}, but "
                    f"{staged.name!r} is {staged.dtype.value}",
                    category,
                )
        if (
            operation.kind is OperationKind.STORE
            and not indirect
            and not typed_packed_store
            and len(operation.reads) != 1
        ):
            out.add(
                "STORE_EDGE_COUNT",
                f"operations[{schedule.operations.index(operation)}].reads",
                f"direct store declares exactly one value edge, got "
                f"{len(operation.reads)}",
                category,
            )
        if indirect:
            index_names = tuple(dict.fromkeys(component.name for component in indirect))
            if operation.kind not in {
                OperationKind.LOAD,
                OperationKind.ATOMIC_RMW,
                OperationKind.STORE,
            }:
                out.add(
                    "ACCESS_INDEXED_OPERATION_UNLOWERABLE",
                    path,
                    "the admitted runtime-indexed subset applies to global loads and "
                    "reservation-owned stores and atomic_rmw only",
                    FindingCategory.HARDWARE_CONFORMANCE,
                )
            elif operation.kind is OperationKind.LOAD:
                if operation.parameters.movement is not LoadMovement.GLOBAL:
                    out.add(
                        "ACCESS_INDEXED_MOVEMENT_UNLOWERABLE",
                        path,
                        "runtime-indexed loads use direct global movement; TMA transfer "
                        "semantics are not admitted",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
                expected_reads = (access.buffer,) + index_names
                if operation.reads != expected_reads:
                    out.add(
                        "ACCESS_INDEX_BUFFER_READS",
                        f"operations[{schedule.operations.index(operation)}].reads",
                        f"runtime-indexed load reads data then its first-use ordered "
                        f"index buffers {list(expected_reads)}, got "
                        f"{list(operation.reads)}",
                        category,
                    )
            elif operation.kind is OperationKind.ATOMIC_RMW:
                if len(access.indices) != 1 or len(indirect) != 1:
                    out.add(
                        "ATOMIC_INDEX_FORM",
                        f"{path}.indices",
                        "the admitted atomic target is rank one and has exactly one "
                        "runtime INT32 index",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
                expected_reads = (access.buffer,) + index_names
                if operation.reads != expected_reads:
                    out.add(
                        "ATOMIC_INDEX_BUFFER_READS",
                        f"operations[{schedule.operations.index(operation)}].reads",
                        f"atomic_rmw reads target then its first-use ordered index "
                        f"buffers {list(expected_reads)}, got {list(operation.reads)}",
                        category,
                    )
            else:
                expected_reads = operation.reads[:1] + index_names
                if operation.reads != expected_reads:
                    out.add(
                        "STORE_INDEX_BUFFER_READS",
                        f"operations[{schedule.operations.index(operation)}].reads",
                        "runtime-indexed store reads its value then its first-use "
                        f"ordered index buffers {list(expected_reads)}, got "
                        f"{list(operation.reads)}",
                        category,
                    )

                canonical_form = (
                    len(indirect) == 2
                    and len(access.indices) >= 2
                    and all(
                        component.source is AccessIndexKind.BUFFER
                        for component in access.indices[:2]
                    )
                    and all(
                        component.source is not AccessIndexKind.BUFFER
                        for component in access.indices[2:]
                    )
                )
                if not canonical_form:
                    out.add(
                        "STORE_INDEX_RESERVATION_FORM",
                        f"{path}.indices",
                        "the admitted indexed store begins with exactly two runtime "
                        "coordinates: atomic target index then returned old value",
                        FindingCategory.PROGRAM_SAFETY,
                    )
                else:
                    state_index = access.indices[0].name
                    position = access.indices[1].name
                    producers = [
                        candidate
                        for candidate in schedule.operations
                        if candidate.kind is OperationKind.ATOMIC_RMW
                        and len(candidate.writes) == 2
                        and candidate.writes[1] == position
                    ]
                    if len(producers) != 1:
                        out.add(
                            "STORE_INDEX_RESERVATION_UNPROVEN",
                            f"{path}.indices[1]",
                            f"runtime position {position!r} is not the unique returned "
                            "old value of one atomic_rmw",
                            FindingCategory.PROGRAM_SAFETY,
                        )
                    else:
                        reservation = producers[0]
                        if reservation.role != operation.role:
                            out.add(
                                "STORE_INDEX_RESERVATION_ROLE",
                                f"operations[{schedule.operations.index(operation)}].role",
                                "the atomic result is register-resident, so reservation "
                                "and indexed store must execute in the same role",
                                FindingCategory.PROGRAM_SAFETY,
                            )
                        target_access = (
                            schedule.access_map(
                                reservation.op_id, reservation.reads[0]
                            )
                            if reservation.reads
                            else None
                        )
                        coordinates_match = (
                            target_access is not None
                            and len(target_access.indices) == 1
                            and target_access.indices[0].source
                            is AccessIndexKind.BUFFER
                            and target_access.indices[0].name == state_index
                        )
                        if not coordinates_match:
                            out.add(
                                "STORE_INDEX_RESERVATION_COORDINATES",
                                f"{path}.indices",
                                "the indexed store must pair the atomic target's exact "
                                "runtime index with that atomic's returned old value",
                                FindingCategory.PROGRAM_SAFETY,
                            )
                        target = (
                            buffers.get(reservation.reads[0])
                            if reservation.reads
                            else None
                        )
                        if (
                            target is not None
                            and target.shape
                            and buffer.shape
                            and target.shape[0] != buffer.shape[0]
                        ):
                            out.add(
                                "STORE_INDEX_RESERVATION_DOMAIN",
                                f"{path}.indices[0]",
                                "the store's reservation coordinate must have the same "
                                "extent as the atomic state target so their masks agree",
                                FindingCategory.PROGRAM_SAFETY,
                            )
                        if reservation.parameters.value != 1:
                            out.add(
                                "STORE_INDEX_RESERVATION_INCREMENT",
                                f"operations[{schedule.operations.index(reservation)}]."
                                "parameters.value",
                                "only atomic increment by one proves distinct returned "
                                "positions for an ordinary indexed store",
                                FindingCategory.PROGRAM_SAFETY,
                            )
                        reservation_index = schedule.operations.index(reservation)
                        loop_bodies = {
                            op_id for loop in schedule.tile_loops for op_id in loop.body
                        }
                        if (
                            reservation.op_id in loop_bodies
                            or operation.op_id in loop_bodies
                        ):
                            out.add(
                                "STORE_INDEX_RESERVATION_LOOP",
                                f"operations[{reservation_index}]",
                                "the first reservation-owned store subset executes its "
                                "atomic and store exactly once per program, outside tile "
                                "loops",
                                FindingCategory.PROGRAM_SAFETY,
                            )
                        program_instances = 1
                        program_domain_known = schedule.program_map is not None
                        if schedule.program_map is not None:
                            for axis in schedule.program_map.axes:
                                owner = buffers.get(axis.buffer)
                                if owner is None or axis.dimension >= len(owner.shape):
                                    program_domain_known = False
                                    break
                                program_instances *= axis.tile_count(
                                    owner.shape[axis.dimension]
                                )
                        result = buffers.get(position)
                        if result is None:
                            program_domain_known = False
                        else:
                            for extent in result.shape:
                                program_instances *= extent
                        if program_domain_known and program_instances > 1 << 32:
                            out.add(
                                "STORE_INDEX_RESERVATION_WRAP",
                                f"operations[{reservation_index}]",
                                "the launch may execute more than 2^32 reservations, "
                                "so an int32 returned position can repeat after wrap",
                                FindingCategory.PROGRAM_SAFETY,
                            )

            index_buffers = [buffers.get(name) for name in index_names]
            known = [item for item in index_buffers if item is not None]
            if len(known) == len(index_buffers):
                shapes = {item.shape for item in known}
                if any(len(item.shape) != 1 for item in known):
                    out.add(
                        "ACCESS_INDEX_DOMAIN_RANK",
                        path,
                        "the admitted runtime index domain is rank one",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
                elif len(shapes) != 1:
                    out.add(
                        "ACCESS_INDEX_DOMAIN_MISMATCH",
                        path,
                        f"runtime index buffers must share one zipped domain, got "
                        f"{[list(item.shape) for item in known]}",
                        category,
                    )

            # The local value has the zipped index domain once, plus every independent
            # tile/full-dimension domain in access order. That is the exact shape the
            # Triton address branch emits for a load result, atomic result or store
            # value.
            result_name = None
            if operation.kind is OperationKind.LOAD and len(operation.writes) == 1:
                result_name = operation.writes[0]
            elif operation.kind is OperationKind.ATOMIC_RMW and len(operation.writes) == 2:
                result_name = operation.writes[1]
            elif operation.kind is OperationKind.STORE and operation.reads:
                result_name = operation.reads[0]
            if result_name is not None:
                staged = buffers.get(result_name)
                source = buffer
                expected_shape: list[int] = []
                added_index_domain = False
                shape_known = len(known) == len(index_buffers) and bool(known)
                for component in access.indices:
                    if component.source is AccessIndexKind.PROGRAM:
                        continue
                    if component.source is AccessIndexKind.BUFFER:
                        if not added_index_domain and shape_known:
                            expected_shape.extend(known[0].shape)
                            added_index_domain = True
                        continue
                    if component.source is AccessIndexKind.PROGRAM_TILE:
                        axis = (
                            schedule.program_map.axis(component.name)
                            if schedule.program_map is not None
                            else None
                        )
                        if axis is not None:
                            expected_shape.append(axis.tile)
                    elif component.source is AccessIndexKind.LOOP_TILE:
                        loop = next(
                            (
                                item
                                for item in schedule.tile_loops
                                if item.iterator == component.name
                            ),
                            None,
                        )
                        if loop is not None:
                            expected_shape.append(loop.tile)
                    elif (
                        component.dimension is not None
                        and component.dimension < len(source.shape)
                    ):
                        # The value domain is what the access covers. Reading the axis
                        # size here demanded a staged buffer sized for elements a
                        # sub-range never addresses, which blocked a gather the emitter
                        # already lowered correctly.
                        expected_shape.append(
                            component.span(source.shape[component.dimension])
                        )
                if staged is not None and shape_known:
                    if staged.shape != tuple(expected_shape):
                        out.add(
                            "ACCESS_INDEXED_VALUE_SHAPE",
                            f"operations[{schedule.operations.index(operation)}]."
                            + (
                                "writes"
                                if operation.kind is not OperationKind.STORE
                                else "reads"
                            ),
                            f"runtime-indexed access has value domain {expected_shape}, but "
                            f"{staged.name!r} has shape {list(staged.shape)}",
                            category,
                        )

        relation = buffer.valid_extent
        if relation is not None and relation.indexed_by:
            supported = (
                len(relation.indexed_by) == 1
                and relation.indexed_by[0] < len(access.indices)
                and access.indices[relation.indexed_by[0]].source
                is AccessIndexKind.PROGRAM
            )
            if not supported:
                out.add(
                    "VALID_EXTENT_ACCESS_UNLOWERABLE",
                    path,
                    "the Triton valid-extent lowering requires one extent axis "
                    "indexed by one scalar program axis",
                    FindingCategory.HARDWARE_CONFORMANCE,
                )

    # A tile axis produces a staged extent. If the axis says 128 and the buffer it
    # stages into says 256, the two disagree about the same tile and one of them is
    # wrong; nothing downstream can tell which.
    for index, access in enumerate(schedule.access_maps):
        operation = op_by_id.get(access.operation)
        if operation is None or not operation.writes:
            continue
        if (
            operation.kind is OperationKind.STORE
            and len(operation.reads) == 3
            and any(
                (destination := buffers.get(name)) is not None
                and destination.packed_block is not None
                and destination.packed_block.format
                is PackedBlockFormat.GGML_Q8_1_V1
                for name in operation.writes
            )
        ):
            # The registry-field verifier above owns all three staged shapes and their
            # shared record prefix. Selecting the first field here would restate that
            # relation as an ordinary one-read Store and duplicate one record-count
            # drift as ACCESS_TILE_MISMATCH.
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
        # Buffer-valued accesses have one zipped domain no matter how many coordinates
        # it supplies. The dedicated check above derives that shape; the legacy loop
        # below intentionally remains byte-for-byte the path for all existing cases.
        if any(
            component.source is AccessIndexKind.BUFFER
            for component in access.indices
        ):
            continue
        vectors = []
        for component_index, component in enumerate(access.indices):
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
                # A sub-range stages what it covers, not the whole axis. Reading the
                # axis size here would demand a tile the access never addresses.
                expected = (
                    component.span(source.shape[component.dimension])
                    if component.dimension is not None and component.dimension < len(source.shape)
                    else None
                )
            vectors.append((component_index, component, expected))
        # A direct register load produces exactly the vector axes in its address.
        # Scalar addresses use the canonical one-value register shape (1,).
        # A rank mismatch must not bypass this relation: otherwise a scalar can be
        # mislabeled as a vector and silently broadcast by a later FMA.
        shape_known = all(extent is not None for _, _, extent in vectors)
        expected_shape = tuple(extent for _, _, extent in vectors) or (1,)
        if (
            operation.kind is OperationKind.LOAD
            and operation.parameters.movement is LoadMovement.GLOBAL
            and staged.space is MemorySpace.REGISTER
            and shape_known
            and (not vectors or len(staged.shape) != len(vectors))
            and staged.shape != expected_shape
        ):
            out.add(
                "LOAD_ACCESS_SHAPE_MISMATCH",
                f"operations[{schedule.operations.index(operation)}].writes[0]",
                f"load address produces register shape {list(expected_shape)}, but "
                f"{staged.name!r} declares {list(staged.shape)}; a load does not splat or reshape",
                category,
            )
        if len(staged.shape) != len(vectors):
            continue
        for staged_dimension, (component_index, component, expected) in enumerate(vectors):
            if expected is not None and staged.shape[staged_dimension] != expected:
                out.add(
                    "ACCESS_TILE_MISMATCH",
                    f"access_maps[{index}].indices[{component_index}]",
                    f"tile axis {component.name or component.dimension} carries "
                    f"{expected} but {staged.name!r} stages {staged.shape[staged_dimension]}",
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


TARGET_SYNC_HINT = {"mbarrier", "barrier.sync"}


def _verify_state_store_ownership(schedule: Schedule, out: _Collector) -> None:
    """Prove the first direct, single-writer ordinary store to caller-owned state."""

    category = FindingCategory.PROGRAM_SAFETY
    buffers = {buffer.name: buffer for buffer in schedule.buffers}
    loop_bodies = {op_id for loop in schedule.tile_loops for op_id in loop.body}

    for operation_index, operation in enumerate(schedule.operations):
        if operation.kind is not OperationKind.STORE or not operation.writes:
            continue
        state = buffers.get(operation.writes[0])
        if state is None or state.mode is not BufferMode.STATE:
            continue
        path = f"operations[{operation_index}]"
        access = schedule.access_map(operation.op_id, state.name)
        if access is None:
            # The generic AccessMap rule owns this malformed edge.
            continue
        if schedule.program_map is None:
            out.add(
                "STATE_STORE_PROGRAM_MAP_REQUIRED",
                path,
                "an ordinary state store requires a ProgramMap ownership proof",
                category,
            )
            continue
        if operation.op_id in loop_bodies:
            out.add(
                "STATE_STORE_LOOP_UNSUPPORTED",
                path,
                "the first single-writer state-store subset executes outside TileLoop",
                category,
            )

        used_axes: list[str] = []
        if len(access.indices) != len(state.shape):
            # The generic rank rule localizes the malformed AccessMap.
            continue
        for dimension, component in enumerate(access.indices):
            component_path = (
                f"access_maps[{schedule.access_maps.index(access)}].indices[{dimension}]"
            )
            if component.source in {
                AccessIndexKind.PROGRAM,
                AccessIndexKind.PROGRAM_TILE,
            }:
                axis = schedule.program_map.axis(component.name or "")
                if axis is None:
                    # The generic AccessMap rule owns an unknown axis.
                    continue
                used_axes.append(axis.name)
                if axis.buffer != state.name or axis.dimension != dimension:
                    out.add(
                        "STATE_STORE_PROGRAM_OWNER",
                        component_path,
                        f"program axis {axis.name!r} is owned by {axis.buffer!r} "
                        f"dimension {axis.dimension}, not state {state.name!r} "
                        f"dimension {dimension}",
                        category,
                    )
                if (
                    component.source is AccessIndexKind.PROGRAM
                    and axis.tile != 1
                ) or (
                    component.source is AccessIndexKind.PROGRAM_TILE
                    and axis.tile == 1
                ):
                    out.add(
                        "STATE_STORE_PROGRAM_COORDINATE",
                        component_path,
                        f"axis {axis.name!r} tile {axis.tile} must use "
                        f"{'program' if axis.tile == 1 else 'program_tile'}",
                        category,
                    )
                continue
            if component.source is AccessIndexKind.DIMENSION:
                if (
                    component.dimension != dimension
                    or component.offset != 0
                    or component.extent is not None
                ):
                    out.add(
                        "STATE_STORE_DIMENSION_COVERAGE",
                        component_path,
                        f"state dimension {dimension} must be covered once in full",
                        category,
                    )
                continue
            out.add(
                "STATE_STORE_COORDINATE_UNPROVEN",
                component_path,
                "state store coordinates admit only program/program_tile and full "
                "dimension components",
                category,
            )

        for axis_index, axis in enumerate(schedule.program_map.axes):
            count = used_axes.count(axis.name)
            if count != 1:
                out.add(
                    "STATE_STORE_PROGRAM_AXIS_COVERAGE",
                    f"program_map.axes[{axis_index}]",
                    f"state store must consume axis {axis.name!r} exactly once, got {count}",
                    category,
                )


def _verify_program_safety(schedule: Schedule, out: _Collector) -> None:
    category = FindingCategory.PROGRAM_SAFETY

    buffers = {buffer.name: buffer for buffer in schedule.buffers}
    roles = {role.name for role in schedule.roles}
    pipelines = {pipeline.name for pipeline in schedule.pipelines}
    barriers = {barrier.name: barrier for barrier in schedule.barriers}
    active = {operation.role for operation in schedule.operations}

    _verify_state_store_ownership(schedule, out)

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
                if order.get(producer.op_id, -1) > index:
                    buffer = buffers.get(name)
                    if (
                        buffer is not None
                        and buffer.mode is BufferMode.STATE
                        and producer.kind is OperationKind.STORE
                    ):
                        # State exists before the Schedule. A same-role load may read that
                        # initial value before the one admitted ordinary update.
                        continue
                    out.add(
                        "OP_READ_BEFORE_WRITE",
                        f"operations[{index}].reads",
                        f"operation {operation.op_id!r} reads {name!r} before its "
                        f"producer {producer.op_id!r} executes",
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
    if warps_per_group is None:
        out.add(
            "ROLE_REGISTERS_TARGET_UNSUPPORTED",
            "roles",
            f"Target {target.target_id!r} declares no execution-group scope for "
            "per-role register budgets",
            category,
        )
        return
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

def _report_residency(schedule: Schedule, target: Target, out: _Collector) -> None:
    """Which declared resource bounds residency, and at what cost.

    This is the attribution half of the paper's `performance analysis` report. There is
    no cost estimate: a Target declares no clock and no bandwidth, so a predicted time
    would be invented rather than analysed.
    """

    category = FindingCategory.HARDWARE_CONFORMANCE
    if schedule.target != target.target_id:
        return  # nothing to analyse against a Target this Schedule does not name
    _verify_role_register_split(schedule, target, out)
    upper_bound = residency_upper_bound(schedule, target)
    if upper_bound is None:
        if target.target_id != "gfx1151":
            return
        out.add(
            "RESIDENCY_TARGET_UNMODELED",
            "target.occupancy",
            f"Target {target.target_id!r} declares no per-multiprocessor occupancy facts; "
            "the Compiler cannot attribute a residency bound and GPU measurement remains "
            "the authority",
            category,
            FindingSeverity.REPORT,
        )
        return
    if upper_bound.binding is None:
        return
    _verify_residency_commitment(schedule, target, upper_bound, out)
    binding = upper_bound.binding
    others = ", ".join(
        f"{b.resource} {b.ctas}"
        for b in sorted(upper_bound.bounds, key=lambda b: b.ctas)
        if b.resource != binding.resource
    )
    # A resource the Schedule declares nothing for produces no bound, and omitting it
    # reads as "does not constrain" when it means "was not examined". Measurement made
    # that concrete: the GEMM declares no shared memory, Triton allocates it for the dot
    # anyway, and it bound residency exactly as tightly as the registers this does model.
    modelled = {b.resource for b in upper_bound.bounds}
    unmodelled = " or ".join(
        sorted(
            resource
            for resource in ("shared_memory", "tensor_memory")
            if resource not in modelled
        )
    )
    out.add(
        "RESIDENCY_BOUND",
        "allocations" if binding.resource.endswith("memory") else "roles",
        f"{binding.resource} bounds maximum possible residency to {binding.ctas} CTA "
        "per multiprocessor "
        f"({binding.per_cta} of {binding.per_multiprocessor} {binding.unit})"
        + (f"; the next bounds are {others}" if others else "")
        + (
            f"; this Schedule declares no {unmodelled}, so nothing here bounds it and a "
            "backend may still allocate some"
            if unmodelled
            else ""
        )
        + "; physical register allocation is backend evidence and is not part of this bound",
        category,
        FindingSeverity.REPORT,
    )

    per_thread = logical_register_pressure_per_thread(schedule, target)
    facts = target.occupancy
    proxy_ctas = (
        facts.registers_per_multiprocessor
        // (per_thread * schedule.total_warp_extent * target.warp_size)
        if per_thread and facts is not None
        else None
    )
    if proxy_ctas is not None and proxy_ctas <= binding.ctas:
        out.add(
            "REGISTER_PRESSURE",
            "buffers",
            f"declared register Buffers have logical pressure {per_thread} per CTA "
            f"thread; if physical allocation tracked that proxy it would imply "
            f"{proxy_ctas} CTA per multiprocessor, but B200 evidence shows the proxy "
            "can lie on either side of ptxas allocation, so it is not a bound or gate",
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
