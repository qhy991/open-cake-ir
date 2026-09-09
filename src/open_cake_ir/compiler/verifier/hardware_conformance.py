"""Target capabilities, concrete instruction commitments and resource bounds."""

from __future__ import annotations

from ..ir import (
    PLACED_CONTRACT_PREFIXES,
    PLACEMENT_FIELDS,
    TMEM_COLUMN_BYTES,
    OperandSource,
    LoadMovement,
    MemorySpace,
    DType,
    ElementwiseOp,
    OperationKind,
    Schedule,
)
from ..performance.residency import (
    logical_register_pressure_per_thread,
    residency_upper_bound,
)
from ..target import (
    Target,
)
from ..diagnostics import FindingCategory, FindingSeverity
from ._collector import _Collector


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
_REGISTER_MMA_CONTRACT = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"

_ELEMENTWISE_INSTRUCTIONS = {
    "libdevice.tanh.f32": (ElementwiseOp.TANH, DType.FP32),
    # Metal's own named-precision spelling. tanh requires a contract, so without one
    # here no Metal Schedule could reach the emitter's precise::tanh at all.
    "metal.precise.tanh.f32": (ElementwiseOp.TANH, DType.FP32),
    "ptx.fma.rn.f32": (ElementwiseOp.FMA, DType.FP32),
}


def resolve_grid(schedule: Schedule) -> tuple[int, int, int]:
    """Project the launch grid; semantic findings own invalid ProgramMap axes.

    This shared projection is total for structurally admissible schedules so
    Compiler can return localized findings even when an axis cannot be resolved.
    """

    if schedule.grid is not None:
        return schedule.grid
    resolved = [1, 1, 1]
    if schedule.program_map is not None:
        seen_axes: set[int] = set()
        for axis in schedule.program_map.axes:
            if not 0 <= axis.axis < 3 or axis.axis in seen_axes:
                continue
            buffer = schedule.buffer(axis.buffer)
            if buffer is None or axis.dimension >= len(buffer.shape):
                continue
            resolved[axis.axis] = axis.tile_count(buffer.shape[axis.dimension])
            seen_axes.add(axis.axis)
    return (resolved[0], resolved[1], resolved[2])


def verify(
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

    for axis, (extent_value, maximum) in enumerate(
        zip(resolve_grid(schedule), limits.maximum_grid)
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

    _report_residency(schedule, target, out)


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

    if allocation.tensor_columns < 32 or allocation.tensor_columns > 512 or allocation.tensor_columns & (allocation.tensor_columns - 1):
        out.add("ALLOCATION_TENSOR_COLUMNS_ILLEGAL", f"{path}.tensor_columns",
            "tcgen05 allocation requires a power-of-two column count in [32, 512]", category)

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
            _verify_register_mma(operation, instruction, buffers, path, out)
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
            if (instruction.shape is not None and tile is not None
                    and tile[1] % instruction.shape[1]):
                out.add("MMA_TILE_INSTRUCTION_MISMATCH", f"{path}.tile_shape",
                    f"tile N {tile[1]} is not a whole number of atom N steps ({instruction.shape[1]})",
                    category)
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


def _verify_register_mma(operation, instruction, buffers, path: str, out: _Collector) -> None:
    """Warp MMA consumes register fragments; declaring them constrains storage.

    This modeled instruction has a fixed 16x8x16 atom, K-major operands and one
    CTA. Other instruction families do not inherit its register-source contract.
    Omitted placement retains its coverage hint but cannot erase the hardware
    semantics of an explicitly named instruction.
    """
    category = FindingCategory.HARDWARE_CONFORMANCE
    if instruction.contract != _REGISTER_MMA_CONTRACT:
        if instruction.operand_source is OperandSource.REGISTER:
            out.add("MMA_REGISTER_CONTRACT_UNSUPPORTED", f"{path}.instruction.operand_source",
                    "register operand placement is modeled only for the admitted BF16 warp MMA",
                    category)
        return
    if instruction.operand_source not in (None, OperandSource.REGISTER):
        out.add("MMA_OPERAND_SOURCE_MISMATCH", f"{path}.instruction.operand_source",
                "warp MMA reads register fragments, not shared or tensor memory",
                category)
    for name in (*operation.reads, *operation.writes):
        buffer = buffers.get(name)
        if buffer is not None and buffer.space is not MemorySpace.REGISTER:
            out.add("MMA_OPERAND_SOURCE_MISMATCH", f"{path}.instruction.operand_source",
                    f"warp MMA operands and accumulator must be register-resident; "
                    f"{name!r} is {buffer.space.value}", category)
    if (instruction.shape is not None and instruction.shape != (16, 8, 16)):
        out.add("MMA_REGISTER_ATOM_MISMATCH", f"{path}.instruction.shape",
                "the declared BF16 warp MMA atom has shape 16x8x16", category)
    if instruction.cta_group is not None and instruction.cta_group != 1:
        out.add("MMA_REGISTER_ATOM_MISMATCH", f"{path}.instruction.cta_group",
                "warp MMA belongs to one CTA", category)
    if (instruction.operand_major is not None
            and tuple(mode.value for mode in instruction.operand_major) != ("k", "k")):
        out.add("MMA_REGISTER_ATOM_MISMATCH", f"{path}.instruction.operand_major",
                "the row/column BF16 warp MMA requires K-major A and B", category)


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
    cap = schedule.residency.registers_per_thread if schedule.residency is not None else None
    limit = target.resource_limits.maximum_registers_per_thread
    if cap is not None:
        if limit is None:
            out.add("TARGET_REGISTER_CAP_UNMODELED", "residency.registers_per_thread",
                "Target does not declare a per-thread register-cap limit; this limit was not checked",
                category, FindingSeverity.REPORT)
        elif cap > limit:
            out.add("TARGET_REGISTER_CAP_LIMIT", "residency.registers_per_thread",
                f"register cap {cap} exceeds Target per-thread capacity {limit}", category)
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
    if upper_bound is None or upper_bound.binding is None:
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
