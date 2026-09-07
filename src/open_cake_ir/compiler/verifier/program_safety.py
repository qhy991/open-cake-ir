"""Synchronization, ordering and declared ownership hazards."""

from __future__ import annotations

from ..ir import (
    AccessIndexKind,
    BarrierMechanism,
    BufferMode,
    Operation,
    OperationKind,
    Schedule,
)
from ..diagnostics import FindingCategory, FindingSeverity
from ._collector import _Collector


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


def verify(schedule: Schedule, out: _Collector) -> None:
    category = FindingCategory.PROGRAM_SAFETY

    buffers = {buffer.name: buffer for buffer in schedule.buffers}
    roles = {role.name for role in schedule.roles}
    pipelines = {pipeline.name for pipeline in schedule.pipelines}
    barriers = {barrier.name: barrier for barrier in schedule.barriers}
    active = {operation.role for operation in schedule.operations}

    _verify_state_store_ownership(schedule, out)

    # Acyclicity alone does not put dependencies before their consumers. This
    # also covers ordering edges that name no read/write buffer. Unknown and
    # self dependencies already have localized data-consistency findings.
    positions = {operation.op_id: index for index, operation in enumerate(schedule.operations)}
    for index, operation in enumerate(schedule.operations):
        for dependency in operation.depends_on:
            position = positions.get(dependency)
            if position is not None and position > index:
                out.add(
                    "OPERATION_DEPENDENCY_ORDER",
                    f"operations[{index}].depends_on",
                    f"dependency {dependency!r} appears later in the Schedule",
                    category,
                )

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
