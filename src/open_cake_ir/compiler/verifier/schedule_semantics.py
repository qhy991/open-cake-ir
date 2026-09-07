"""Structural relationships of declared schedule regions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..ir import Schedule
from ..diagnostics import FindingCategory
from ._collector import _Collector


@dataclass(frozen=True)
class NameConflict:
    """First repeated occurrence of one name in a declared collection.

    These are Target-independent facts, not an exception policy. In particular,
    the facade must interleave operation-id conflicts with its existing per-op
    input checks rather than raise every conflict before inspecting operations.
    """

    collection: str
    index: int
    field: str
    name: str


def _duplicate_positions(values: Iterable[str]) -> list[tuple[int, str]]:
    seen: set[str] = set()
    repeated: set[str] = set()
    positions: list[tuple[int, str]] = []
    for index, value in enumerate(values):
        if value in seen and value not in repeated:
            repeated.add(value)
            positions.append((index, value))
        seen.add(value)
    return positions


def _duplicates(values: Iterable[str]) -> list[str]:
    return sorted(name for _, name in _duplicate_positions(values))


def name_conflicts(schedule: Schedule) -> tuple[NameConflict, ...]:
    """Return named-collection conflicts without consulting a Target.

    Collections follow the legacy declaration/operation phases; positions follow
    source order within a collection. Each collection/name occurs once, at its
    first repeated position. Tile-loop names are the final, verifier-only group.
    Consumers choose their diagnostic or exception projection from these facts.
    """

    groups = (
        ("roles", "name", (item.name for item in schedule.roles)),
        ("allocations", "name", (item.name for item in schedule.allocations)),
        ("buffers", "name", (item.name for item in schedule.buffers)),
        ("pipelines", "name", (item.name for item in schedule.pipelines)),
        ("barriers", "name", (item.name for item in schedule.barriers)),
        ("operations", "id", (item.op_id for item in schedule.operations)),
        ("tile_loops", "name", (item.name for item in schedule.tile_loops)),
    )
    return tuple(
        NameConflict(collection, index, field, name)
        for collection, field, names in groups
        for index, name in _duplicate_positions(names)
    )


def verify(schedule: Schedule, out: _Collector) -> None:
    category = FindingCategory.SCHEDULE_SEMANTICS

    if not schedule.operations:
        out.add(
            "SCHEDULE_OPERATIONS_EMPTY",
            "operations",
            "a Schedule must declare at least one operation",
            category,
        )

    # Preserve the prior per-name diagnostic order. Exception consumers instead
    # use the helper's source positions to preserve their own first-error phase.
    for conflict in sorted(name_conflicts(schedule), key=lambda item: (item.collection, item.name)):
        out.add(
            "NAME_DUPLICATE",
            conflict.collection,
            f"duplicate {conflict.collection[:-1]} name {conflict.name!r}",
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
