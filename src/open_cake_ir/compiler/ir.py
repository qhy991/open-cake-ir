"""Canonical typed Schedule representation.

This is the single owner of Schedule semantics. It resolves the four legacy models
(`ir.py`, `schedule_v2_ir.py`, `explicit_resource_ir.py`, `production_resource_ir.py`)
into one vocabulary, as required by `docs/MIGRATION_PLAN.md`.

The legacy split forced two workarounds that do not survive here:

* `explicit_resource_ir` and `production_resource_ir` each had to rewrite an operation
  kind into a base-legal one (`epilogue` -> `store`, `reduce_sum` -> `reduce_argmin`),
  parse through the frozen base parser, then restore the real kind afterwards. One
  `OperationKind` removes the rewrite entirely.
* `EpilogueFormula` and `EpilogueParameters` were declared twice with disjoint members.
  They are unified here; a formula is admitted, not invented, per operation.

Parsing is strict: unknown fields are rejected, every closed vocabulary is an `Enum`,
and every parameter set is bound to its operation kind. Structural admissibility only
-- semantic gates (resource limits, synchronization, profile rules) belong to the
verifier, not to this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence, Union


class ScheduleParseError(ValueError):
    """One Schedule document is not structurally admissible."""


# --------------------------------------------------------------------------- enums


class DType(str, Enum):
    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"
    FP8_E4M3 = "fp8_e4m3"
    INT32 = "int32"
    INT64 = "int64"

    @property
    def itemsize(self) -> int:
        return _DTYPE_ITEMSIZE[self]


_DTYPE_ITEMSIZE = {
    DType.BF16: 2,
    DType.FP16: 2,
    DType.FP32: 4,
    DType.FP8_E4M3: 1,
    DType.INT32: 4,
    DType.INT64: 8,
}


class MemorySpace(str, Enum):
    GLOBAL = "global"
    SHARED = "shared"
    TENSOR = "tensor"
    REGISTER = "register"


class BufferMode(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    SCRATCH = "scratch"


class OperationKind(str, Enum):
    """The unified operation vocabulary.

    `EPILOGUE` came from the explicit-resource model and `REDUCE_SUM` from the
    production-resource model; both were separate enums that had to be smuggled
    through the base parser as `store` / `reduce_argmin`.
    """

    LOAD = "load"
    MMA = "mma"
    EPILOGUE = "epilogue"
    REDUCE_ARGMIN = "reduce_argmin"
    REDUCE_SUM = "reduce_sum"
    STORE = "store"
    FENCE_PROXY = "fence_proxy"


class LoadMovement(str, Enum):
    GLOBAL = "global"
    TMA = "tma"


class ArgminTieBreak(str, Enum):
    LOWEST_INDEX = "lowest_index"


class NaNPolicy(str, Enum):
    REJECT_INPUT = "reject_input"


class MmaFormula(str, Enum):
    SQUARED_EUCLIDEAN_XSQ_ELIDED = "squared_euclidean_xsq_elided"


class EpilogueFormula(str, Enum):
    """Union of the two legacy `EpilogueFormula` enums, which had disjoint members."""

    CENTROID_SQ_MINUS_TWO_DOT = "centroid_sq_minus_two_dot"
    BIAS_ADD_BF16_ROUND = "bias_add_bf16_round"


class ReductionScope(str, Enum):
    CTA = "cta"


class AccessIndexKind(str, Enum):
    """Legacy declared this as `class AccessIndexKind(str)` with a `VALUES` set."""

    PROGRAM = "program"
    PROGRAM_TILE = "program_tile"
    LOOP_TILE = "loop_tile"
    DIMENSION = "dimension"


class BoundaryPolicy(str, Enum):
    MASK_TILED_AXES = "mask_tiled_axes"


# ----------------------------------------------------------------------- primitives


def _strict_object(
    value: Any,
    *,
    required: set[str],
    optional: set[str] | None = None,
    context: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScheduleParseError(f"{context} must be an object")
    missing = required - value.keys()
    extra = value.keys() - required - (optional or set())
    if missing:
        raise ScheduleParseError(f"{context} missing fields: {sorted(missing)}")
    if extra:
        raise ScheduleParseError(f"{context} unknown fields: {sorted(extra)}")
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ScheduleParseError(f"{context} must be a non-empty string")
    return value


def _positive_int(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ScheduleParseError(f"{context} must be a positive integer")
    return value


def _nonnegative_int(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ScheduleParseError(f"{context} must be a non-negative integer")
    return value


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise ScheduleParseError(f"{context} must be a boolean")
    return value


def _string_tuple(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ScheduleParseError(f"{context} must be a list of strings")
    return tuple(_string(item, f"{context}[{index}]") for index, item in enumerate(value))


def _enum(enum_type: type[Enum], value: Any, context: str) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        admitted = ", ".join(sorted(member.value for member in enum_type))
        raise ScheduleParseError(
            f"{context} is unsupported; admitted values are {admitted}"
        ) from error


def _object_list(value: Any, context: str, *, allow_empty: bool = True) -> list[Any]:
    if not isinstance(value, list):
        raise ScheduleParseError(f"{context} must be a list")
    if not value and not allow_empty:
        raise ScheduleParseError(f"{context} must not be empty")
    return value


# ---------------------------------------------------------------------- declarations


@dataclass(frozen=True)
class Role:
    name: str
    warps: tuple[int, ...]

    @property
    def warp_extent(self) -> int:
        """One past the highest declared warp index.

        This is the quantity a CTA warp budget must bound. Legacy resource checks
        compared `len(warps)`, which admits sparse or out-of-range warp ids.
        """

        return max(self.warps) + 1

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "Role":
        obj = _strict_object(value, required={"name", "warps"}, context=context)
        warps = _object_list(obj["warps"], f"{context}.warps", allow_empty=False)
        parsed = tuple(
            _nonnegative_int(warp, f"{context}.warps[{index}]")
            for index, warp in enumerate(warps)
        )
        if len(set(parsed)) != len(parsed):
            raise ScheduleParseError(f"{context}.warps repeats a warp index")
        return cls(_string(obj["name"], f"{context}.name"), parsed)


@dataclass(frozen=True)
class Allocation:
    name: str
    space: MemorySpace
    size_bytes: int

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "Allocation":
        obj = _strict_object(
            value, required={"name", "space", "size_bytes"}, context=context
        )
        return cls(
            _string(obj["name"], f"{context}.name"),
            _enum(MemorySpace, obj["space"], f"{context}.space"),
            _positive_int(obj["size_bytes"], f"{context}.size_bytes"),
        )


@dataclass(frozen=True)
class Buffer:
    name: str
    space: MemorySpace
    dtype: DType
    shape: tuple[int, ...]
    mode: BufferMode
    allocation: str | None
    byte_offset: int
    stages: int

    @property
    def elements(self) -> int:
        total = 1
        for dimension in self.shape:
            total *= dimension
        return total

    @property
    def size_bytes(self) -> int:
        return self.elements * self.dtype.itemsize * self.stages

    @property
    def byte_extent(self) -> tuple[int, int]:
        """Half-open [start, stop) byte range inside the owning Allocation."""

        return (self.byte_offset, self.byte_offset + self.size_bytes)

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "Buffer":
        obj = _strict_object(
            value,
            required={"name", "space", "dtype", "shape", "mode"},
            optional={"allocation", "byte_offset", "stages"},
            context=context,
        )
        shape = _object_list(obj["shape"], f"{context}.shape", allow_empty=False)
        allocation = obj.get("allocation")
        return cls(
            _string(obj["name"], f"{context}.name"),
            _enum(MemorySpace, obj["space"], f"{context}.space"),
            _enum(DType, obj["dtype"], f"{context}.dtype"),
            tuple(
                _positive_int(dimension, f"{context}.shape[{index}]")
                for index, dimension in enumerate(shape)
            ),
            _enum(BufferMode, obj["mode"], f"{context}.mode"),
            None if allocation is None else _string(allocation, f"{context}.allocation"),
            _nonnegative_int(obj.get("byte_offset", 0), f"{context}.byte_offset"),
            _positive_int(obj.get("stages", 1), f"{context}.stages"),
        )


@dataclass(frozen=True)
class Pipeline:
    name: str
    stages: int

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "Pipeline":
        obj = _strict_object(value, required={"name", "stages"}, context=context)
        return cls(
            _string(obj["name"], f"{context}.name"),
            _positive_int(obj["stages"], f"{context}.stages"),
        )


@dataclass(frozen=True)
class Barrier:
    name: str
    count: int
    producers: tuple[str, ...]
    consumers: tuple[str, ...]
    pipeline: str | None

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "Barrier":
        obj = _strict_object(
            value,
            required={"name", "count", "producers", "consumers"},
            optional={"pipeline"},
            context=context,
        )
        pipeline = obj.get("pipeline")
        return cls(
            _string(obj["name"], f"{context}.name"),
            _positive_int(obj["count"], f"{context}.count"),
            _string_tuple(obj["producers"], f"{context}.producers"),
            _string_tuple(obj["consumers"], f"{context}.consumers"),
            None if pipeline is None else _string(pipeline, f"{context}.pipeline"),
        )


# ------------------------------------------------------------------ physical mapping


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

    def axis(self, name: str) -> ProgramAxis | None:
        return next((axis for axis in self.axes if axis.name == name), None)

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "ProgramMap":
        obj = _strict_object(value, required={"axes"}, context=context)
        axes = _object_list(obj["axes"], f"{context}.axes", allow_empty=False)
        return cls(
            tuple(
                ProgramAxis.from_dict(item, f"{context}.axes[{index}]")
                for index, item in enumerate(axes)
            )
        )


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
class TileLoop:
    name: str
    iterator: str
    buffer: str
    dimension: int
    tile: int
    body: tuple[str, ...]
    range_options: RangeOptions

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
        )


@dataclass(frozen=True)
class AccessIndex:
    source: AccessIndexKind
    name: str | None
    dimension: int | None

    @property
    def is_vector(self) -> bool:
        """Whether this component contributes a tile axis rather than a scalar index."""

        return self.source is not AccessIndexKind.PROGRAM

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "AccessIndex":
        if not isinstance(value, Mapping):
            raise ScheduleParseError(f"{context} must be an object")
        source = _enum(AccessIndexKind, value.get("source"), f"{context}.source")
        if source is AccessIndexKind.DIMENSION:
            obj = _strict_object(
                value, required={"source", "dimension"}, context=context
            )
            return cls(
                source, None, _nonnegative_int(obj["dimension"], f"{context}.dimension")
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


# ------------------------------------------------------------- operation parameters


@dataclass(frozen=True)
class LoadParameters:
    movement: LoadMovement


@dataclass(frozen=True)
class MmaParameters:
    accumulator: DType
    formula: MmaFormula | None


@dataclass(frozen=True)
class EpilogueParameters:
    formula: EpilogueFormula
    coalesced: bool


@dataclass(frozen=True)
class ReduceArgminParameters:
    tie_break: ArgminTieBreak
    nan_policy: NaNPolicy
    across_loop: bool


@dataclass(frozen=True)
class ReduceSumParameters:
    parts: int
    scope: ReductionScope


@dataclass(frozen=True)
class StoreParameters:
    coalesced: bool


@dataclass(frozen=True)
class FenceProxyParameters:
    pass


OperationParameters = Union[
    LoadParameters,
    MmaParameters,
    EpilogueParameters,
    ReduceArgminParameters,
    ReduceSumParameters,
    StoreParameters,
    FenceProxyParameters,
]


def _operation_parameters(
    kind: OperationKind, value: Any, context: str
) -> OperationParameters:
    if kind is OperationKind.LOAD:
        obj = _strict_object(value, required={"movement"}, context=context)
        return LoadParameters(_enum(LoadMovement, obj["movement"], f"{context}.movement"))

    if kind is OperationKind.MMA:
        obj = _strict_object(
            value, required={"accumulator"}, optional={"formula"}, context=context
        )
        accumulator = _enum(DType, obj["accumulator"], f"{context}.accumulator")
        if accumulator is not DType.FP32:
            raise ScheduleParseError(f"{context}.accumulator must be fp32")
        formula = obj.get("formula")
        return MmaParameters(
            accumulator,
            None if formula is None else _enum(MmaFormula, formula, f"{context}.formula"),
        )

    if kind is OperationKind.EPILOGUE:
        obj = _strict_object(
            value, required={"formula", "coalesced"}, context=context
        )
        return EpilogueParameters(
            _enum(EpilogueFormula, obj["formula"], f"{context}.formula"),
            _boolean(obj["coalesced"], f"{context}.coalesced"),
        )

    if kind is OperationKind.REDUCE_ARGMIN:
        obj = _strict_object(
            value,
            required={"tie_break", "nan_policy"},
            optional={"across_loop"},
            context=context,
        )
        return ReduceArgminParameters(
            _enum(ArgminTieBreak, obj["tie_break"], f"{context}.tie_break"),
            _enum(NaNPolicy, obj["nan_policy"], f"{context}.nan_policy"),
            _boolean(obj.get("across_loop", False), f"{context}.across_loop"),
        )

    if kind is OperationKind.REDUCE_SUM:
        obj = _strict_object(value, required={"parts", "scope"}, context=context)
        return ReduceSumParameters(
            _positive_int(obj["parts"], f"{context}.parts"),
            _enum(ReductionScope, obj["scope"], f"{context}.scope"),
        )

    if kind is OperationKind.STORE:
        obj = _strict_object(value, required={"coalesced"}, context=context)
        return StoreParameters(_boolean(obj["coalesced"], f"{context}.coalesced"))

    _strict_object(value, required=set(), context=context)
    return FenceProxyParameters()


@dataclass(frozen=True)
class Operation:
    op_id: str
    kind: OperationKind
    role: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    waits: tuple[str, ...]
    signals: tuple[str, ...]
    depends_on: tuple[str, ...]
    pipeline: str | None
    parameters: OperationParameters

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "Operation":
        obj = _strict_object(
            value,
            required={"id", "kind", "role", "reads", "writes", "parameters"},
            optional={"waits", "signals", "depends_on", "pipeline"},
            context=context,
        )
        kind = _enum(OperationKind, obj["kind"], f"{context}.kind")
        pipeline = obj.get("pipeline")
        return cls(
            _string(obj["id"], f"{context}.id"),
            kind,
            _string(obj["role"], f"{context}.role"),
            _string_tuple(obj["reads"], f"{context}.reads"),
            _string_tuple(obj["writes"], f"{context}.writes"),
            _string_tuple(obj.get("waits", []), f"{context}.waits"),
            _string_tuple(obj.get("signals", []), f"{context}.signals"),
            _string_tuple(obj.get("depends_on", []), f"{context}.depends_on"),
            None if pipeline is None else _string(pipeline, f"{context}.pipeline"),
            _operation_parameters(kind, obj["parameters"], f"{context}.parameters"),
        )


# -------------------------------------------------------------------------- schedule


_SCHEDULE_REQUIRED = {
    "schema_version",
    "schedule_id",
    "target",
    "roles",
    "allocations",
    "buffers",
    "pipelines",
    "barriers",
    "operations",
    "outputs",
    "metadata",
}
_SCHEDULE_OPTIONAL = {"grid", "program_map", "tile_loops", "access_maps"}


@dataclass(frozen=True)
class Schedule:
    schema_version: int
    schedule_id: str
    target: str
    grid: tuple[int, int, int] | None
    program_map: ProgramMap | None
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
    def profile(self) -> str | None:
        value = self.metadata.get("profile")
        return value if isinstance(value, str) else None

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

        metadata = obj["metadata"]
        if not isinstance(metadata, Mapping):
            raise ScheduleParseError("schedule.metadata must be an object")

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
            grid=grid,
            program_map=(
                ProgramMap.from_dict(obj["program_map"], "schedule.program_map")
                if has_program_map
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
            metadata=dict(metadata),
        )
