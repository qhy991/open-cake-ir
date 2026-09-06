"""Program coordinates, tile loops and per-operation memory access maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ._parse import (
    ScheduleParseError,
    _boolean,
    _enum,
    _nonnegative_int,
    _object_list,
    _positive_int,
    _strict_object,
    _string,
    _string_tuple,
)
from .vocabulary import (
    AccessIndexKind,
    BoundaryPolicy,
)


@dataclass(frozen=True)
class ProgramAxis:
    name: str
    axis: int
    buffer: str
    dimension: int
    tile: int

    @property
    def is_tiled(self) -> bool:
        """A tile of 1 is a scalar program index; anything wider carries an offset vector."""

        return self.tile > 1

    def tile_count(self, extent: int) -> int:
        """Number of program coordinates needed to cover an owned buffer extent."""

        return (extent + self.tile - 1) // self.tile

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "ProgramAxis":
        obj = _strict_object(
            value,
            required={"name", "axis", "buffer", "dimension", "tile"},
            context=context,
        )
        return cls(
            _string(obj["name"], f"{context}.name"),
            _nonnegative_int(obj["axis"], f"{context}.axis"),
            _string(obj["buffer"], f"{context}.buffer"),
            _nonnegative_int(obj["dimension"], f"{context}.dimension"),
            _positive_int(obj["tile"], f"{context}.tile"),
        )


@dataclass(frozen=True)
class ProgramMap:
    axes: tuple[ProgramAxis, ...]
    persistent: bool
    """Launch a fixed CTA count and walk the work, instead of one CTA per tile.

    The launched count is not a second number to declare: it is the Target's
    multiprocessor count times the residency the Schedule already commits to. Declaring
    it separately would let the grid and the commitment disagree.
    """

    traversal: tuple[str, ...] | None
    """Axis names, fastest-varying first.

    Only meaningful when persistent. Without persistence the axis-to-program-id
    assignment already decides which axis varies fastest, so a second way to say it would
    be an alternative spelling of a decision the Schedule has made. With a scheduler
    walking more tiles than there are CTAs, the walk order is a separate decision and
    this is where it goes.
    """

    def axis(self, name: str) -> ProgramAxis | None:
        return next((axis for axis in self.axes if axis.name == name), None)

    def walk_order(self) -> tuple[ProgramAxis, ...]:
        """The axes in the order a persistent scheduler decomposes its work index."""

        if self.traversal is None:
            return tuple(sorted(self.axes, key=lambda item: item.axis))
        # Unknown names are the verifier's to report; skipping them here keeps this a
        # view rather than a second place that decides what a legal traversal is.
        return tuple(
            axis for axis in (self.axis(name) for name in self.traversal) if axis is not None
        )

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "ProgramMap":
        obj = _strict_object(
            value,
            required={"axes"},
            optional={"persistent", "traversal"},
            context=context,
        )
        axes = _object_list(obj["axes"], f"{context}.axes", allow_empty=False)
        parsed = tuple(
            ProgramAxis.from_dict(item, f"{context}.axes[{index}]")
            for index, item in enumerate(axes)
        )
        persistent = _boolean(obj.get("persistent", False), f"{context}.persistent")
        traversal = obj.get("traversal")
        if traversal is not None:
            traversal = _string_tuple(traversal, f"{context}.traversal")
            if not persistent:
                raise ScheduleParseError(
                    f"{context}.traversal orders a persistent walk; without persistence "
                    "the axis numbering already decides which axis varies fastest"
                )
        return cls(parsed, persistent, traversal)


@dataclass(frozen=True)
class RangeOptions:
    num_stages: int
    loop_unroll_factor: int
    flatten: bool
    warp_specialize: bool
    disallow_acc_multi_buffer: bool
    disable_licm: bool

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "RangeOptions":
        obj = _strict_object(
            value,
            required={
                "num_stages",
                "loop_unroll_factor",
                "flatten",
                "warp_specialize",
                "disallow_acc_multi_buffer",
                "disable_licm",
            },
            context=context,
        )
        return cls(
            _positive_int(obj["num_stages"], f"{context}.num_stages"),
            _positive_int(obj["loop_unroll_factor"], f"{context}.loop_unroll_factor"),
            _boolean(obj["flatten"], f"{context}.flatten"),
            _boolean(obj["warp_specialize"], f"{context}.warp_specialize"),
            _boolean(
                obj["disallow_acc_multi_buffer"], f"{context}.disallow_acc_multi_buffer"
            ),
            _boolean(obj["disable_licm"], f"{context}.disable_licm"),
        )


@dataclass(frozen=True)
class LoopStop:
    """A query-derived exclusive loop bound in the loop buffer's coordinate space."""

    program: str
    add: int
    floor_div: int

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "LoopStop":
        obj = _strict_object(
            value,
            required={"program", "add", "floor_div"},
            context=context,
        )
        add = obj["add"]
        if not isinstance(add, int) or isinstance(add, bool):
            raise ScheduleParseError(f"{context}.add must be an integer")
        return cls(
            _string(obj["program"], f"{context}.program"),
            add,
            _positive_int(obj["floor_div"], f"{context}.floor_div"),
        )


@dataclass(frozen=True)
class TileLoop:
    name: str
    iterator: str
    buffer: str
    dimension: int
    tile: int
    body: tuple[str, ...]
    range_options: RangeOptions
    stop: LoopStop | None = None

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "TileLoop":
        obj = _strict_object(
            value,
            required={
                "name",
                "iterator",
                "buffer",
                "dimension",
                "tile",
                "body",
                "range_options",
            },
            optional={"stop"},
            context=context,
        )
        body = _string_tuple(obj["body"], f"{context}.body")
        if not body:
            raise ScheduleParseError(f"{context}.body must not be empty")
        return cls(
            _string(obj["name"], f"{context}.name"),
            _string(obj["iterator"], f"{context}.iterator"),
            _string(obj["buffer"], f"{context}.buffer"),
            _nonnegative_int(obj["dimension"], f"{context}.dimension"),
            _positive_int(obj["tile"], f"{context}.tile"),
            body,
            RangeOptions.from_dict(obj["range_options"], f"{context}.range_options"),
            None
            if obj.get("stop") is None
            else LoopStop.from_dict(obj["stop"], f"{context}.stop"),
        )


@dataclass(frozen=True)
class AccessIndex:
    source: AccessIndexKind
    name: str | None
    dimension: int | None
    # A `dimension` component covers the whole axis unless it says otherwise. `offset`
    # and `extent` narrow it to one contiguous sub-range, which is what lets two
    # operations address disjoint halves of the same buffer -- a producer writing the
    # layout its consumer wants, rather than a split materialized upstream first.
    # `extent` of None means "to the end of the axis", so the default spelling of a
    # whole dimension stays exactly what it was.
    offset: int = 0
    extent: int | None = None

    @property
    def is_vector(self) -> bool:
        """Whether this component contributes a tile axis rather than a scalar index."""

        return self.source is not AccessIndexKind.PROGRAM

    def span(self, size: int) -> int:
        """How many elements of an axis of `size` this component covers."""

        return size - self.offset if self.extent is None else self.extent

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "AccessIndex":
        if not isinstance(value, Mapping):
            raise ScheduleParseError(f"{context} must be an object")
        source = _enum(AccessIndexKind, value.get("source"), f"{context}.source")
        if source is AccessIndexKind.DIMENSION:
            obj = _strict_object(
                value,
                required={"source", "dimension"},
                optional={"offset", "extent"},
                context=context,
            )
            # Presence, not value: `obj.get` would read an explicit `null` as an absent
            # key, so `{"offset": null}` would parse as the omitted spelling. Same
            # program, different Schedule bytes -- a second identity for one kernel,
            # admitted by the parser even though the authoring Schema refuses it.
            return cls(
                source,
                None,
                _nonnegative_int(obj["dimension"], f"{context}.dimension"),
                # An absent offset is 0; a written 0 would be a second spelling of the
                # same thing, so only a real displacement may be written.
                _positive_int(obj["offset"], f"{context}.offset") if "offset" in obj else 0,
                _positive_int(obj["extent"], f"{context}.extent") if "extent" in obj else None,
            )
        obj = _strict_object(value, required={"source", "name"}, context=context)
        return cls(source, _string(obj["name"], f"{context}.name"), None)


@dataclass(frozen=True)
class AccessMap:
    operation: str
    buffer: str
    indices: tuple[AccessIndex, ...]
    boundary: BoundaryPolicy

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "AccessMap":
        obj = _strict_object(
            value,
            required={"operation", "buffer", "indices", "boundary"},
            context=context,
        )
        indices = _object_list(obj["indices"], f"{context}.indices", allow_empty=False)
        return cls(
            _string(obj["operation"], f"{context}.operation"),
            _string(obj["buffer"], f"{context}.buffer"),
            tuple(
                AccessIndex.from_dict(item, f"{context}.indices[{index}]")
                for index, item in enumerate(indices)
            ),
            _enum(BoundaryPolicy, obj["boundary"], f"{context}.boundary"),
        )
