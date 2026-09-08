"""Top-level Schedule assembly, input loading and derived structural queries."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ._parse import (
    ScheduleParseError,
    _enum,
    _object_list,
    _positive_int,
    _strict_object,
    _string,
    _string_tuple,
)
from .mapping import AccessMap, ProgramMap, TileLoop
from .operations import Operation
from .resources import (
    Allocation,
    Barrier,
    Buffer,
    Pipeline,
    Residency,
    Role,
)
from .vocabulary import (
    AccessIndexKind,
    LoweringBackend,
    MemorySpace,
    OperationKind,
    _CONTRACTION_AXIS,
)


_SCHEDULE_REQUIRED = {
    "schema_version",
    "schedule_id",
    "target",
    "lowering",
    "roles",
    "allocations",
    "buffers",
    "pipelines",
    "barriers",
    "operations",
    "outputs",
    "metadata",
}
_SCHEDULE_OPTIONAL = {"grid", "program_map", "residency", "tile_loops", "access_maps"}


@dataclass(frozen=True)
class LoweringRoute:
    """The only lowering choice a Schedule writes explicitly.

    The emitter derives the argument signature from global Buffers. Keeping an ABI label
    here would restate that signature and had already produced a false four-tensor label
    for a two-tensor Softmax Schedule.
    """

    backend: LoweringBackend
    entry_point: str

    @classmethod
    def from_dict(cls, value: Any, context: str = "schedule.lowering") -> "LoweringRoute":
        obj = _strict_object(
            value,
            required={"backend", "entry_point"},
            context=context,
        )
        entry_point = _string(obj["entry_point"], f"{context}.entry_point")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", entry_point) is None:
            raise ScheduleParseError(f"{context}.entry_point must be an identifier")
        return cls(
            backend=_enum(LoweringBackend, obj["backend"], f"{context}.backend"),
            entry_point=entry_point,
        )


def _metadata(value: Any) -> Mapping[str, Any]:
    obj = _strict_object(
        value,
        required=set(),
        optional={"workload_contract_sha256", "legacy_source"},
        context="schedule.metadata",
    )
    workload = obj.get("workload_contract_sha256")
    if workload is not None and (
        not isinstance(workload, str)
        or len(workload) != 64
        or any(character not in "0123456789abcdef" for character in workload)
    ):
        raise ScheduleParseError(
            "schedule.metadata.workload_contract_sha256 must be a lowercase SHA256 digest"
        )
    legacy = obj.get("legacy_source")
    if legacy is not None:
        source = _strict_object(
            legacy,
            required={"revision", "path", "canonical_json_sha256"},
            context="schedule.metadata.legacy_source",
        )
        _string(source["revision"], "schedule.metadata.legacy_source.revision")
        _string(source["path"], "schedule.metadata.legacy_source.path")
        digest = source["canonical_json_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ScheduleParseError(
                "schedule.metadata.legacy_source.canonical_json_sha256 must be a "
                "lowercase SHA256 digest"
            )
    return dict(obj)


@dataclass(frozen=True)
class Schedule:
    schema_version: int
    schedule_id: str
    target: str
    lowering: LoweringRoute
    grid: tuple[int, int, int] | None
    program_map: ProgramMap | None
    residency: Residency | None
    roles: tuple[Role, ...]
    allocations: tuple[Allocation, ...]
    buffers: tuple[Buffer, ...]
    pipelines: tuple[Pipeline, ...]
    barriers: tuple[Barrier, ...]
    tile_loops: tuple[TileLoop, ...]
    access_maps: tuple[AccessMap, ...]
    operations: tuple[Operation, ...]
    outputs: tuple[str, ...]
    metadata: Mapping[str, Any]

    # ---- derived views used by the verifier and the emitters -----------------

    def buffer(self, name: str) -> Buffer | None:
        return next((item for item in self.buffers if item.name == name), None)

    def operation(self, op_id: str) -> Operation | None:
        return next((item for item in self.operations if item.op_id == op_id), None)

    def argmin_domain(self, operation: Operation) -> int | None:
        """Known candidate extent for the supported loop-carried value-flow domain.

        The loop supplies candidate coordinates; the producing LOAD/MMA/arithmetic
        chain must preserve that coordinate and have no runtime candidate validity.
        A physical buffer capacity is not proof of a runtime valid prefix. Unknown
        producers, indirect domains or candidate-dependent extent lookups abstain.
        """
        chain = self.enclosing_loops(operation)
        if (operation.kind is not OperationKind.REDUCE_ARGMIN or len(operation.reads) != 1
                or len(chain) != 1 or not operation.parameters.across_loop):
            return None
        loop = chain[0]
        source = self.buffer(operation.reads[0])
        owner = self.buffer(loop.buffer)
        if (source is None or owner is None or len(source.shape) != 2
                or source.shape[1] != loop.tile or loop.stop is not None):
            return None

        def static_axis(name: str, axis: int, before: int) -> bool:
            value = self.buffer(name)
            if value is None or not 0 <= axis < len(value.shape):
                return False
            producers = [(i, op) for i, op in enumerate(self.operations[:before]) if name in op.writes]
            if not producers:
                return False
            position, producer = producers[-1]
            if producer.kind is OperationKind.LOAD and producer.reads:
                loaded = self.buffer(producer.reads[0])
                if loaded is None:
                    return False
                if loaded.space is not MemorySpace.GLOBAL:
                    return loaded.shape == value.shape and static_axis(loaded.name, axis, position)
                access = self.access_map(producer.op_id, loaded.name)
                if access is None or any(c.source is AccessIndexKind.BUFFER for c in access.indices):
                    return False
                vectors = [(i, c) for i, c in enumerate(access.indices) if c.is_vector]
                if len(vectors) != len(value.shape):
                    return False
                dimension, component = vectors[axis]
                relation = loaded.valid_extent
                if relation is not None and (relation.dimension == dimension or dimension in relation.indexed_by):
                    return False
                return component.source is AccessIndexKind.LOOP_TILE and component.name == loop.iterator
            if producer.kind is OperationKind.MMA and len(producer.reads) >= 2:
                # A(M,K) x B(N,K) preserves A's row and B's column domains.
                return static_axis(producer.reads[axis], 0, position)
            if producer.kind is OperationKind.CAST and len(producer.reads) == 1:
                return static_axis(producer.reads[0], axis, position)
            if producer.kind is OperationKind.ELEMENTWISE:
                varying = []
                for read in producer.reads:
                    operand = self.buffer(read)
                    if operand is None:
                        return False
                    if operand.is_scalar:
                        continue
                    if len(operand.shape) == len(value.shape):
                        read_axis = axis
                    elif len(operand.shape) == 1 and len(value.shape) == 2:
                        broadcast_axis = producer.parameters.broadcast_axis
                        if broadcast_axis is None:
                            broadcast_axis = 1 if operand.shape[0] == value.shape[1] else 0
                        read_axis = 0 if broadcast_axis == axis else None
                    else:
                        read_axis = axis - (len(value.shape) - len(operand.shape))
                    if read_axis is None or read_axis < 0 or operand.shape[read_axis] == 1:
                        continue
                    varying.append(static_axis(read, read_axis, position))
                return bool(varying) and all(varying)
            return False

        if static_axis(source.name, 1, self.operations.index(operation)):
            return owner.shape[loop.dimension]
        return None

    def _staged_axis_filled_by(self, operand: str, loop: "TileLoop") -> int | None:
        """Which axis of a staged operand this loop's tile index fills, if any.

        A `program` component is a scalar and removes the dimension it indexes, so the
        staged tile's axes are the remaining components in order. An operand the loop does
        not index at all -- loaded once outside it -- answers None.
        """

        producer = next((op for op in self.operations if operand in op.writes), None)
        if producer is None:
            return None
        access = next(
            (m for m in self.access_maps if m.operation == producer.op_id), None
        )
        if access is None:
            return None
        axis = 0
        saw_buffer_domain = False
        for component in access.indices:
            if component.source is AccessIndexKind.PROGRAM:
                continue
            if component.source is AccessIndexKind.BUFFER:
                # All buffer-valued coordinates in one access are zipped over one
                # common domain. Counting each coordinate separately would turn
                # [expert[k], row[k]] into a k-by-k product and shift every later axis.
                if saw_buffer_domain:
                    continue
                saw_buffer_domain = True
            if (
                component.source is AccessIndexKind.LOOP_TILE
                and component.name == loop.iterator
            ):
                return axis
            axis += 1
        return None

    def mma_accumulates_over(self, operation: Operation, loop: "TileLoop") -> bool:
        """Whether this contraction sums across the loop rather than starting again.

        A contraction is `out[M, N] = sum over K of a[M, K] * b[N, K]`, so K is axis 1 of
        both staged operands. A loop that fills K is walking the sum and its results have
        to accumulate; a loop that fills M or N is walking the output and each iteration
        computes a different part of it.

        Both shapes are in the corpus. Flash-KMeans tiles the centroid axis, which is the
        output's N, and each iteration produces a fresh block. The Blackwell assignment
        kernel tiles K, and tcgen05 accumulates in tensor memory. Deriving which is which
        is what lets one emitter serve both without the Schedule declaring a mode.

        An operand the loop never indexes does not vote: it is loop-invariant, which is
        true of either shape.
        """

        if operation.kind is not OperationKind.MMA or len(operation.reads) != 2:
            return False
        filled = [
            self._staged_axis_filled_by(name, loop) for name in operation.reads
        ]
        indexed = [axis for axis in filled if axis is not None]
        return bool(indexed) and all(axis == _CONTRACTION_AXIS for axis in indexed)

    def tile_loop(self, name: str) -> TileLoop | None:
        return next((item for item in self.tile_loops if item.name == name), None)

    def loop_parent(self) -> dict[str, str]:
        """Child loop name -> enclosing loop name.

        A `tile_loops` body lists what is inside the loop in order: operation ids, and
        the names of loops nested within it. The artifact for the warp-specialized
        profile is a two-deep nest, which a flat list of loops cannot carry.
        """

        names = {loop.name for loop in self.tile_loops}
        parent: dict[str, str] = {}
        for loop in self.tile_loops:
            for entry in loop.body:
                if entry in names:
                    parent[entry] = loop.name
        return parent

    def loop_depth(self, name: str) -> int:
        """Nesting depth of one loop, outermost being 0. Guards against cycles."""

        parent = self.loop_parent()
        depth, seen = 0, {name}
        while name in parent:
            name = parent[name]
            if name in seen:
                return depth
            seen.add(name)
            depth += 1
        return depth

    def enclosing_loops(self, operation: Operation) -> tuple[TileLoop, ...]:
        """Lexical loop chain, outermost first, derived from existing bodies.

        Ambiguous direct ownership and cycles remain Verifier findings; this query
        terminates on malformed schedules so assessment can localize those failures.
        """

        direct = [loop for loop in self.tile_loops if operation.op_id in loop.body]
        if len(direct) != 1:
            return ()
        chain = [direct[0]]
        parent = self.loop_parent()
        seen = {direct[0].name}
        while chain[-1].name in parent:
            name = parent[chain[-1].name]
            loop = self.tile_loop(name)
            if name in seen or loop is None:
                break
            seen.add(name)
            chain.append(loop)
        return tuple(reversed(chain))

    def loop_operations(self, loop: TileLoop) -> tuple[Operation, ...]:
        """Expand a loop body in order, including descendants, without new state."""

        result: list[Operation] = []
        seen: set[str] = {loop.name}
        pending = list(reversed(loop.body))
        while pending:
            entry = pending.pop()
            child = self.tile_loop(entry)
            operation = self.operation(entry)
            if child is not None:
                if child.name not in seen:
                    seen.add(child.name)
                    pending.extend(reversed(child.body))
            elif operation is not None:
                result.append(operation)
        return tuple(result)

    def access_map(self, operation: str, buffer: str) -> AccessMap | None:
        return next(
            (
                item
                for item in self.access_maps
                if item.operation == operation and item.buffer == buffer
            ),
            None,
        )

    @property
    def total_warp_extent(self) -> int:
        """One past the highest warp index used by any role."""

        return max((role.warp_extent for role in self.roles), default=0)

    # ---- parsing --------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "Schedule":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, value: Any) -> "Schedule":
        obj = _strict_object(
            value,
            required=_SCHEDULE_REQUIRED,
            optional=_SCHEDULE_OPTIONAL,
            context="schedule",
        )
        if obj["schema_version"] != 1:
            raise ScheduleParseError("schedule.schema_version must be 1")

        has_grid = "grid" in obj
        has_program_map = "program_map" in obj
        if has_grid == has_program_map:
            raise ScheduleParseError(
                "schedule must declare exactly one of grid or program_map"
            )

        grid: tuple[int, int, int] | None = None
        if has_grid:
            raw_grid = obj["grid"]
            if not isinstance(raw_grid, list) or len(raw_grid) != 3:
                raise ScheduleParseError(
                    "schedule.grid must contain exactly three dimensions"
                )
            grid = (
                _positive_int(raw_grid[0], "schedule.grid[0]"),
                _positive_int(raw_grid[1], "schedule.grid[1]"),
                _positive_int(raw_grid[2], "schedule.grid[2]"),
            )

        def parse_list(field: str, factory: Any) -> tuple[Any, ...]:
            items = _object_list(obj.get(field, []), f"schedule.{field}")
            return tuple(
                factory(item, f"schedule.{field}[{index}]")
                for index, item in enumerate(items)
            )

        return cls(
            schema_version=1,
            schedule_id=_string(obj["schedule_id"], "schedule.schedule_id"),
            target=_string(obj["target"], "schedule.target"),
            lowering=LoweringRoute.from_dict(obj["lowering"]),
            grid=grid,
            program_map=(
                ProgramMap.from_dict(obj["program_map"], "schedule.program_map")
                if has_program_map
                else None
            ),
            residency=(
                Residency.from_dict(obj["residency"], "schedule.residency")
                if "residency" in obj
                else None
            ),
            roles=parse_list("roles", Role.from_dict),
            allocations=parse_list("allocations", Allocation.from_dict),
            buffers=parse_list("buffers", Buffer.from_dict),
            pipelines=parse_list("pipelines", Pipeline.from_dict),
            barriers=parse_list("barriers", Barrier.from_dict),
            tile_loops=parse_list("tile_loops", TileLoop.from_dict),
            access_maps=parse_list("access_maps", AccessMap.from_dict),
            operations=parse_list("operations", Operation.from_dict),
            outputs=_string_tuple(obj["outputs"], "schedule.outputs"),
            metadata=_metadata(obj["metadata"]),
        )
