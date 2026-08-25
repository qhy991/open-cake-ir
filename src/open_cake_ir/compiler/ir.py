"""Canonical typed Schedule representation.

This is the single owner of Schedule semantics. It resolves the four legacy models
(`ir.py`, `schedule_v2_ir.py`, `explicit_resource_ir.py`, `production_resource_ir.py`)
into one vocabulary, as required by `docs/MIGRATION_PLAN.md`.

The legacy split forced two workarounds that do not survive here:

* `explicit_resource_ir` and `production_resource_ir` each had to rewrite an operation
  kind into a base-legal one (`epilogue` -> `store`, `reduce` -> `reduce_argmin`),
  parse through the frozen base parser, then restore the real kind afterwards. One
  `OperationKind` removes the rewrite entirely.
* `EpilogueFormula` and `EpilogueParameters` were declared twice with disjoint members.
  They are unified here; a formula is admitted, not invented, per operation.

Parsing is strict: unknown fields are rejected, every closed vocabulary is an `Enum`,
and every parameter set is bound to its operation kind. Structural admissibility only
-- semantic gates (resource limits, synchronization, backend preflight) belong to the
verifier, not to this module.
"""

from __future__ import annotations

import json
import re
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

    @property
    def itemsize(self) -> int:
        return _DTYPE_ITEMSIZE[self]


_DTYPE_ITEMSIZE = {
    DType.BF16: 2,
    DType.FP16: 2,
    DType.FP32: 4,
    DType.FP8_E4M3: 1,
    DType.INT32: 4,
}


class MemorySpace(str, Enum):
    GLOBAL = "global"
    SHARED = "shared"
    TENSOR = "tensor"
    REGISTER = "register"


class BufferMode(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    STATE = "state"
    SCRATCH = "scratch"


# `tl.dot(a, trans(b))` and tcgen05 alike contract the last axis of both staged operands,
# so a rank-2 operand carries K at axis 1. Named because two modules reason about it.
_CONTRACTION_AXIS = 1


class OperationKind(str, Enum):
    """The unified operation vocabulary.

    `EPILOGUE` came from the explicit-resource model and `REDUCE` from the
    production-resource model; both were separate enums that had to be smuggled
    through the base parser as `store` / `reduce_argmin`.

    `REDUCE` is one kind carrying an operator rather than one kind per operator, which is
    the shape `ELEMENTWISE` already had. A second reduction was the moment to pick: a
    `reduce_max` kind beside `reduce_sum` would have been two spellings of collapsing an
    axis. `REDUCE_ARGMIN` stays separate because it collapses a tiled search to one
    index; `TOP_K` instead preserves a selected axis and returns both values and indices.
    They therefore have different result types even when k is one.
    """

    LOAD = "load"
    MMA = "mma"
    EPILOGUE = "epilogue"
    REDUCE_ARGMIN = "reduce_argmin"
    REDUCE = "reduce"
    TOP_K = "top_k"
    ATOMIC_RMW = "atomic_rmw"
    ELEMENTWISE = "elementwise"
    STORE = "store"


class LoweringBackend(str, Enum):
    """Mechanism that materializes source, independent of operator or Workload."""

    TRITON = "triton"
    CUTLASS_CUTE_DSL = "cutlass_cute_dsl"
    CHECKED_CUDA_ASSET = "checked_cuda_asset"


class ReduceOp(str, Enum):
    """The associative operator a reduction folds with.

    Each one implies its own identity, which the backend needs before a loop that carries
    the reduction across iterations. Naming the operator here is what lets that identity
    be derived instead of assumed.
    """

    SUM = "sum"
    MAX = "max"


class LoadMovement(str, Enum):
    GLOBAL = "global"
    TMA = "tma"


class LoadReuse(str, Enum):
    """Whether an operand is read again, which decides what it should do to the cache.

    The surveyed PTX work toggles exactly this per operand: the activation every warp in
    the CTA reads is asked to stay resident, and the weight each warp reads once is asked
    not to displace it. Stated as intent rather than as a cache modifier, because the
    modifier that expresses it differs per backend while the fact about the operand does
    not (P8, and P2 -- the decision stays visible).
    """

    REUSED = "reused"
    STREAMED = "streamed"


class AtomicOp(str, Enum):
    ADD = "add"


class AtomicMemoryOrder(str, Enum):
    RELAXED = "relaxed"


class AtomicMemoryScope(str, Enum):
    DEVICE = "device"


class IndexTieBreak(str, Enum):
    """Deterministic ordering when an indexed selection sees equal values."""

    LOWEST_INDEX = "lowest_index"


class NaNPolicy(str, Enum):
    REJECT_INPUT = "reject_input"


class EpilogueFormula(str, Enum):
    """Union of the two legacy `EpilogueFormula` enums, which had disjoint members."""

    CENTROID_SQ_MINUS_TWO_DOT = "centroid_sq_minus_two_dot"
    BIAS_ADD_BF16_ROUND = "bias_add_bf16_round"


class ElementwiseOp(str, Enum):
    """The closed arithmetic vocabulary a Schedule composes.

    `MmaFormula` and `EpilogueFormula` name a whole operator's math in one token, so each
    backend hardcodes one operator's arithmetic and a new operator needs a new member and
    a new emitted body per backend. These are the pieces those formulas are built from:
    every admitted operator's epilogue decomposes into them, so an emitter implements
    each once and a Schedule composes rather than asking for a new name.

    Still closed, because the verifier gates on it. What changed is the granularity.
    """

    SQUARE = "square"
    RSQRT = "rsqrt"
    EXP = "exp"
    TANH = "tanh"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"

    @property
    def arity(self) -> int:
        return (
            1
            if self
            in (
                ElementwiseOp.SQUARE,
                ElementwiseOp.RSQRT,
                ElementwiseOp.EXP,
                ElementwiseOp.TANH,
            )
            else 2
        )


class ReductionScope(str, Enum):
    CTA = "cta"


class BarrierMechanism(str, Enum):
    """How a declared handshake is realized.

    A Target admits several; `synchronization_contracts` names them. Writing the
    emitter surfaced the gap: the retained artifact uses an mbarrier for its pipelined
    handoffs and `cute.arch.sync_threads()` for the epilogue-to-reduce one, and nothing
    in the Schedule said which.
    """

    MBARRIER = "mbarrier"
    NAMED = "barrier.sync"


class PipelineKind(str, Enum):
    """The asynchronous agent pair a producer operation requires.

    Pipeline depth is a declaration, but the two currently lowerable pipeline kinds are
    already fixed by producer semantics: TMA feeds UMMA, or UMMA releases to threads.
    Keeping that derivation typed gives the verifier and emitter one closed vocabulary.
    """

    TMA_TO_UMMA = "tma_to_umma"
    UMMA_TO_THREAD = "umma_to_thread"


class OperandSource(str, Enum):
    """Where an MMA reads its operands from."""

    SHARED = "shared"
    TENSOR = "tensor"


class OperandMajorMode(str, Enum):
    """Operand major axis. `tcgen05.OperandMajorMode` in the retained artifact."""

    K = "k"
    MN = "mn"


class Swizzle(str, Enum):
    """Shared-memory swizzle commitment.

    One of the concrete hardware commitments the paper's IR requires the agent to write
    down (arXiv:2608.12629v1 S2). Without it the backend picks a swizzle and the choice
    is neither inspectable nor verifiable.
    """

    NONE = "none"
    B32 = "swizzle_32b"
    B64 = "swizzle_64b"
    B128 = "swizzle_128b"


# Blackwell tensor memory is addressed as columns of `TMEM_LANES` 4-byte words. An
# Allocation's byte size and its column range must agree; the two are declared
# separately because the artifact allocates columns while buffers are sized in bytes.
TMEM_LANES = 128
TMEM_WORD_BYTES = 4
TMEM_COLUMN_BYTES = TMEM_LANES * TMEM_WORD_BYTES


class AccessIndexKind(str, Enum):
    """Legacy declared this as `class AccessIndexKind(str)` with a `VALUES` set."""

    PROGRAM = "program"
    PROGRAM_TILE = "program_tile"
    LOOP_TILE = "loop_tile"
    DIMENSION = "dimension"
    BUFFER = "buffer"


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
    registers_per_thread: int | None
    """What this role's threads hold, when the roles divide the CTA's budget.

    `Residency.registers_per_thread` is the whole CTA's allocation, fixed at launch.
    `setmaxnreg` redistributes *within* that: a role that loads can drop to a small budget
    so a role that accumulates can take more, and the total is conserved. So this is the
    distribution and residency is the total, and the verifier holds the two to each other
    rather than letting them drift apart.

    All four surveyed libraries do this and none of them can say it in a Schedule today --
    CUTLASS deallocates a transform role to 48 and allocates its accumulator role to 256,
    FlashInfer holds an entirely `Empty` role at 24 so its softmax roles can have 192, and
    the megakernel splits 64 against 104.
    """

    @property
    def warp_extent(self) -> int:
        """One past the highest declared warp index.

        This is the quantity a CTA warp budget must bound. Legacy resource checks
        compared `len(warps)`, which admits sparse or out-of-range warp ids.
        """

        return max(self.warps) + 1

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "Role":
        obj = _strict_object(
            value,
            required={"name", "warps"},
            optional={"registers_per_thread"},
            context=context,
        )
        warps = _object_list(obj["warps"], f"{context}.warps", allow_empty=False)
        parsed = tuple(
            _nonnegative_int(warp, f"{context}.warps[{index}]")
            for index, warp in enumerate(warps)
        )
        if len(set(parsed)) != len(parsed):
            raise ScheduleParseError(f"{context}.warps repeats a warp index")
        expected = tuple(range(parsed[0], parsed[0] + len(parsed)))
        if parsed != expected:
            raise ScheduleParseError(
                f"{context}.warps must be one ascending contiguous interval; "
                f"got {list(parsed)}"
            )
        registers = obj.get("registers_per_thread")
        if registers is not None:
            registers = _positive_int(registers, f"{context}.registers_per_thread")
            # setmaxnreg takes a multiple of eight in [24, 256]; a value outside that is
            # not a tuning choice the backend can decline, it is an illegal instruction.
            if registers % 8 or not 24 <= registers <= 256:
                raise ScheduleParseError(
                    f"{context}.registers_per_thread must be a multiple of 8 in [24, 256]"
                )
        return cls(_string(obj["name"], f"{context}.name"), parsed, registers)


@dataclass(frozen=True)
class Allocation:
    """A region of one memory space, and -- where the space has one -- who takes it out.

    `allocating_role` names the role that issues the allocation and its release. Tensor
    memory is the only space here that has that protocol: `tcgen05.alloc` is issued by one
    warp, the address reaches the rest of the CTA through shared memory, and the matching
    `relinquish_alloc_permit` has to come from the same warp. Which role that is changes
    when the release becomes visible to the next CTA, so it is a decision the Schedule
    makes rather than one a backend derives from the order operations happen to be
    written in.

    A space without an allocation protocol may not name a role, because there would be no
    instruction for it to place.
    """

    name: str
    space: MemorySpace
    size_bytes: int
    tensor_columns: int | None
    allocating_role: str | None

    @property
    def implied_tensor_columns(self) -> int:
        """Columns `size_bytes` corresponds to, for a tensor-memory Allocation."""

        return self.size_bytes // TMEM_COLUMN_BYTES

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "Allocation":
        obj = _strict_object(
            value,
            required={"name", "space", "size_bytes"},
            optional={"tensor_columns", "allocating_role"},
            context=context,
        )
        columns = obj.get("tensor_columns")
        space = _enum(MemorySpace, obj["space"], f"{context}.space")
        role = obj.get("allocating_role")
        if space is MemorySpace.TENSOR and role is None:
            raise ScheduleParseError(
                f"{context}.allocating_role is required for a tensor-memory Allocation"
            )
        if space is not MemorySpace.TENSOR and role is not None:
            raise ScheduleParseError(
                f"{context}.allocating_role names a space with no allocation protocol"
            )
        return cls(
            _string(obj["name"], f"{context}.name"),
            space,
            _positive_int(obj["size_bytes"], f"{context}.size_bytes"),
            None if columns is None else _positive_int(columns, f"{context}.tensor_columns"),
            None if role is None else _string(role, f"{context}.allocating_role"),
        )


@dataclass(frozen=True)
class ScaleRelation:
    """How one scale buffer partitions and names its FP8 data buffer.

    ``granularity`` is written in data-axis order. ``axis_order`` is the permutation
    that turns those grouped axes into the scale buffer's physical shape.  The target
    buffer supplies the rank and extents, so the verifier owns the dependent checks.
    """

    buffer: str
    granularity: tuple[int, ...]
    axis_order: tuple[int, ...]

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "ScaleRelation":
        obj = _strict_object(
            value,
            required={"buffer", "granularity", "axis_order"},
            context=context,
        )
        granularity = _object_list(
            obj["granularity"], f"{context}.granularity", allow_empty=False
        )
        axis_order = _object_list(
            obj["axis_order"], f"{context}.axis_order", allow_empty=False
        )
        parsed_order = tuple(
            _nonnegative_int(axis, f"{context}.axis_order[{index}]")
            for index, axis in enumerate(axis_order)
        )
        if len(set(parsed_order)) != len(parsed_order):
            raise ScheduleParseError(f"{context}.axis_order repeats a data axis")
        return cls(
            _string(obj["buffer"], f"{context}.buffer"),
            tuple(
                _positive_int(extent, f"{context}.granularity[{index}]")
                for index, extent in enumerate(granularity)
            ),
            parsed_order,
        )


@dataclass(frozen=True)
class ValidExtentRelation:
    """One padded data axis whose runtime-valid region is a device prefix.

    ``indexed_by`` maps extent-buffer axes to data-buffer axes.  AccessMap remains the
    coordinate authority; lowering reuses those coordinates to load the one extent.
    """

    dimension: int
    buffer: str
    indexed_by: tuple[int, ...]

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "ValidExtentRelation":
        obj = _strict_object(
            value,
            required={"dimension", "buffer", "indexed_by"},
            context=context,
        )
        axes = _object_list(obj["indexed_by"], f"{context}.indexed_by", allow_empty=False)
        indexed_by = tuple(
            _nonnegative_int(axis, f"{context}.indexed_by[{index}]")
            for index, axis in enumerate(axes)
        )
        if len(set(indexed_by)) != len(indexed_by):
            raise ScheduleParseError(f"{context}.indexed_by repeats a data axis")
        return cls(
            _nonnegative_int(obj["dimension"], f"{context}.dimension"),
            _string(obj["buffer"], f"{context}.buffer"),
            indexed_by,
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
    swizzle: Swizzle | None
    scale_of: ScaleRelation | None
    valid_extent: ValidExtentRelation | None

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
            optional={
                "allocation",
                "byte_offset",
                "stages",
                "swizzle",
                "scale_of",
                "valid_extent",
            },
            context=context,
        )
        shape = _object_list(obj["shape"], f"{context}.shape", allow_empty=False)
        allocation = obj.get("allocation")
        swizzle = obj.get("swizzle")
        scale_of = obj.get("scale_of")
        valid_extent = obj.get("valid_extent")
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
            None if swizzle is None else _enum(Swizzle, swizzle, f"{context}.swizzle"),
            None
            if scale_of is None
            else ScaleRelation.from_dict(scale_of, f"{context}.scale_of"),
            None
            if valid_extent is None
            else ValidExtentRelation.from_dict(
                valid_extent, f"{context}.valid_extent"
            ),
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
    mechanism: BarrierMechanism | None

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "Barrier":
        obj = _strict_object(
            value,
            required={"name", "count", "producers", "consumers"},
            optional={"pipeline", "mechanism"},
            context=context,
        )
        pipeline = obj.get("pipeline")
        mechanism = obj.get("mechanism")
        return cls(
            _string(obj["name"], f"{context}.name"),
            _positive_int(obj["count"], f"{context}.count"),
            _string_tuple(obj["producers"], f"{context}.producers"),
            _string_tuple(obj["consumers"], f"{context}.consumers"),
            None if pipeline is None else _string(pipeline, f"{context}.pipeline"),
            None
            if mechanism is None
            else _enum(BarrierMechanism, mechanism, f"{context}.mechanism"),
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
            extent = obj.get("extent")
            offset = obj.get("offset")
            return cls(
                source,
                None,
                _nonnegative_int(obj["dimension"], f"{context}.dimension"),
                # An absent offset is 0; a written 0 would be a second spelling of the
                # same thing, so only a real displacement may be written.
                0 if offset is None else _positive_int(offset, f"{context}.offset"),
                None if extent is None else _positive_int(extent, f"{context}.extent"),
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
    descriptor_box: tuple[int, ...] | None
    reuse: LoadReuse | None
    """TMA descriptor box extents -- the coordinate commitment the paper requires.

    Without it `movement: tma` is a flag: it says a bulk-tensor copy happens but not
    what tile the descriptor addresses, so the backend derives a box and the choice is
    not inspectable. Meaningless for `movement: global`.
    """


# Whether an instruction places its own operands is a fact about the contract, so the
# contract vocabulary owns it. The Verifier reads it to decide which declarations are
# legal; the authoring Schema projects the same fact so an agent cannot spend a Turn
# discovering it. Two spellings of this list would be the defect it exists to prevent.
PLACED_CONTRACT_PREFIXES = ("tcgen05.", "mma.sync.", "wgmma.")
PLACEMENT_FIELDS = ("shape", "cta_group", "operand_source", "operand_major")


@dataclass(frozen=True)
class MmaInstruction:
    """One MMA atom commitment.

    The retained artifact spreads this across five module-level decisions:
    MMA_INSTRUCTION_SHAPE, the contract selected by MmaF16BF16Op, tcgen05.CtaGroup.ONE,
    tcgen05.OperandSource.SMEM and OperandMajorMode.K for both operands. They are one
    choice and belong in one object; naming them separately is how they drifted out of
    the IR in the first place.
    """

    contract: str
    shape: tuple[int, int, int] | None
    cta_group: int | None
    operand_source: OperandSource | None
    operand_major: tuple[OperandMajorMode, OperandMajorMode] | None
    """Placement details a tensor-core atom commits to and a tile-level dot does not.

    `tcgen05` takes a CTA group, an operand source and a major mode per operand;
    `triton.dot` takes none of them, because the backend owns that placement. Which
    contract needs which is a semantic question, so the verifier asks it -- the same
    split that keeps an epilogue formula's operator family out of the parser.
    """

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "MmaInstruction":
        obj = _strict_object(
            value,
            required={"contract"},
            optional={"shape", "cta_group", "operand_source", "operand_major"},
            context=context,
        )
        shape = obj.get("shape")
        if shape is not None:
            extents = _object_list(shape, f"{context}.shape")
            if len(extents) != 3:
                raise ScheduleParseError(f"{context}.shape must declare exactly M, N and K")
            shape = tuple(
                _positive_int(extent, f"{context}.shape[{index}]")
                for index, extent in enumerate(extents)
            )
        major = obj.get("operand_major")
        if major is not None:
            modes = _object_list(major, f"{context}.operand_major")
            if len(modes) != 2:
                raise ScheduleParseError(
                    f"{context}.operand_major must declare a mode for A and for B"
                )
            major = tuple(
                _enum(OperandMajorMode, mode, f"{context}.operand_major[{index}]")
                for index, mode in enumerate(modes)
            )
        cta_group = obj.get("cta_group")
        if cta_group is not None:
            cta_group = _positive_int(cta_group, f"{context}.cta_group")
            if cta_group not in (1, 2):
                raise ScheduleParseError(f"{context}.cta_group must be 1 or 2")
        source = obj.get("operand_source")
        return cls(
            _string(obj["contract"], f"{context}.contract"),
            shape,
            cta_group,
            None if source is None else _enum(OperandSource, source, f"{context}.operand_source"),
            major,
        )


@dataclass(frozen=True)
class MmaParameters:
    accumulator: DType
    instruction: MmaInstruction | None
    tile_shape: tuple[int, int, int] | None


@dataclass(frozen=True)
class CopyAtom:
    """One copy-atom commitment, e.g. the TMEM load an epilogue uses.

    The retained artifact writes `tcgen05.Ld32x32bOp(tcgen05.Repetition.x64)`. The op
    and its repetition determine how many accumulator elements each thread moves per
    step, so they belong to the schedule rather than to the backend.
    """

    op: str
    repetition: int

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "CopyAtom":
        obj = _strict_object(value, required={"op", "repetition"}, context=context)
        return cls(
            _string(obj["op"], f"{context}.op"),
            _positive_int(obj["repetition"], f"{context}.repetition"),
        )


@dataclass(frozen=True)
class EpilogueParameters:
    formula: EpilogueFormula
    coalesced: bool
    subtile: tuple[int, int] | None
    """The epilogue's sub-tiling of the accumulator.

    The artifact derives it as `(size(acc, [0, 0]), size(acc, [0, 1]) // 4)`. The
    divisor is a scheduling choice with a register-pressure consequence, not an
    implementation detail, so it is declared rather than hidden.
    """
    source_atom: CopyAtom | None


@dataclass(frozen=True)
class ReduceArgminParameters:
    tie_break: IndexTieBreak
    nan_policy: NaNPolicy
    across_loop: bool


@dataclass(frozen=True)
class ReduceParameters:
    """A fold that collapses one declared axis of its input.

    This declared how many partial accumulators to combine, which is the split-K use it
    was written for and also a second statement of a fact the read buffer's shape already
    carried. Declaring the axis instead makes the extent follow from that shape, so a fold
    over any axis of any input is expressible, and the verifier gains an invariant it
    could not state before: the written shape is the read shape with this axis removed.

    The invariant holds whatever the operator is, which is the argument for `op` living
    here rather than in the kind: every rule written for the sum is a rule about the axis.
    """

    op: ReduceOp
    axis: int
    scope: ReductionScope


@dataclass(frozen=True)
class TopKParameters:
    """Greatest values and source positions from one resident rank-one tile.

    Descending result order is part of the operation rather than an optional spelling.
    Group formation and batched routing remain separate operations.
    """

    k: int
    tie_break: IndexTieBreak
    nan_policy: NaNPolicy


@dataclass(frozen=True)
class AtomicRmwParameters:
    """One atomic read-modify-write that returns the value preceding its effect.

    The target address stays in AccessMap.  These parameters own the update and the
    memory model, so neither an emitter nor a workload name chooses them implicitly.
    """

    op: AtomicOp
    value: int
    order: AtomicMemoryOrder
    scope: AtomicMemoryScope


@dataclass(frozen=True)
class ElementwiseInstruction:
    """The target instruction selected for arithmetic with multiple realizations.

    Most elementwise primitives have no separately admitted instruction in the current
    vocabulary. Tanh does: the KDA corpus relies on the approximate PTX instruction, whose
    cost and numerical behaviour differ from a libdevice call. Leaving it to the backend
    would make those two physical schedules have one spelling.
    """

    contract: str

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "ElementwiseInstruction":
        obj = _strict_object(value, required={"contract"}, context=context)
        return cls(_string(obj["contract"], f"{context}.contract"))


@dataclass(frozen=True)
class ElementwiseParameters:
    """One arithmetic primitive over the operation's reads.

    A binary op may name a `scalar` instead of a second read, which is what lets a
    Schedule write `x * 2.0` or `x + eps` without declaring a buffer to hold a constant.
    """

    op: ElementwiseOp
    scalar: float | None
    broadcast_axis: int | None
    instruction: ElementwiseInstruction | None
    """Which axis of the result a narrower operand spans.

    Trailing-axis alignment is the array convention, but it only covers half the cases
    here: a per-column bias spans the last axis of its accumulator while a per-row scale
    spans the first. Inferring one from the shapes would pick wrong whenever they happen
    to be equal, so the Schedule states it, as it states every other placement fact.
    """

    @property
    def arity_needed(self) -> int:
        return self.op.arity


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
    ReduceParameters,
    TopKParameters,
    AtomicRmwParameters,
    ElementwiseParameters,
    StoreParameters,
    FenceProxyParameters,
]


def _operation_parameters(
    kind: OperationKind, value: Any, context: str
) -> OperationParameters:
    if kind is OperationKind.LOAD:
        obj = _strict_object(
            value,
            required={"movement"},
            optional={"descriptor_box", "reuse"},
            context=context,
        )
        movement = _enum(LoadMovement, obj["movement"], f"{context}.movement")
        box = obj.get("descriptor_box")
        if box is not None:
            if movement is not LoadMovement.TMA:
                raise ScheduleParseError(
                    f"{context}.descriptor_box applies to tma movement only"
                )
            extents = _object_list(box, f"{context}.descriptor_box", allow_empty=False)
            box = tuple(
                _positive_int(extent, f"{context}.descriptor_box[{index}]")
                for index, extent in enumerate(extents)
            )
        reuse = obj.get("reuse")
        return LoadParameters(
            movement,
            box,
            None if reuse is None else _enum(LoadReuse, reuse, f"{context}.reuse"),
        )

    if kind is OperationKind.MMA:
        obj = _strict_object(
            value,
            required={"accumulator"},
            optional={"instruction", "tile_shape"},
            context=context,
        )
        accumulator = _enum(DType, obj["accumulator"], f"{context}.accumulator")
        if accumulator is not DType.FP32:
            raise ScheduleParseError(f"{context}.accumulator must be fp32")
        instruction = obj.get("instruction")
        def mnk(field: str):
            raw = obj.get(field)
            if raw is None:
                return None
            extents = _object_list(raw, f"{context}.{field}")
            if len(extents) != 3:
                raise ScheduleParseError(
                    f"{context}.{field} must declare exactly M, N and K"
                )
            return tuple(
                _positive_int(extent, f"{context}.{field}[{index}]")
                for index, extent in enumerate(extents)
            )

        return MmaParameters(
            accumulator,
            None
            if instruction is None
            else MmaInstruction.from_dict(instruction, f"{context}.instruction"),
            mnk("tile_shape"),
        )

    if kind is OperationKind.EPILOGUE:
        obj = _strict_object(
            value,
            required={"formula", "coalesced"},
            optional={"subtile", "source_atom"},
            context=context,
        )
        subtile = obj.get("subtile")
        if subtile is not None:
            extents = _object_list(subtile, f"{context}.subtile")
            if len(extents) != 2:
                raise ScheduleParseError(f"{context}.subtile must declare two extents")
            subtile = tuple(
                _positive_int(extent, f"{context}.subtile[{index}]")
                for index, extent in enumerate(extents)
            )
        atom = obj.get("source_atom")
        return EpilogueParameters(
            _enum(EpilogueFormula, obj["formula"], f"{context}.formula"),
            _boolean(obj["coalesced"], f"{context}.coalesced"),
            subtile,
            None if atom is None else CopyAtom.from_dict(atom, f"{context}.source_atom"),
        )

    if kind is OperationKind.REDUCE_ARGMIN:
        obj = _strict_object(
            value,
            required={"tie_break", "nan_policy"},
            optional={"across_loop"},
            context=context,
        )
        return ReduceArgminParameters(
            _enum(IndexTieBreak, obj["tie_break"], f"{context}.tie_break"),
            _enum(NaNPolicy, obj["nan_policy"], f"{context}.nan_policy"),
            _boolean(obj.get("across_loop", False), f"{context}.across_loop"),
        )

    if kind is OperationKind.REDUCE:
        obj = _strict_object(value, required={"op", "axis", "scope"}, context=context)
        return ReduceParameters(
            _enum(ReduceOp, obj["op"], f"{context}.op"),
            _nonnegative_int(obj["axis"], f"{context}.axis"),
            _enum(ReductionScope, obj["scope"], f"{context}.scope"),
        )

    if kind is OperationKind.TOP_K:
        obj = _strict_object(
            value,
            required={"k", "tie_break", "nan_policy"},
            context=context,
        )
        return TopKParameters(
            _positive_int(obj["k"], f"{context}.k"),
            _enum(IndexTieBreak, obj["tie_break"], f"{context}.tie_break"),
            _enum(NaNPolicy, obj["nan_policy"], f"{context}.nan_policy"),
        )

    if kind is OperationKind.ATOMIC_RMW:
        obj = _strict_object(
            value,
            required={"op", "value", "order", "scope"},
            context=context,
        )
        update = obj["value"]
        if not isinstance(update, int) or isinstance(update, bool):
            raise ScheduleParseError(f"{context}.value must be an integer")
        return AtomicRmwParameters(
            _enum(AtomicOp, obj["op"], f"{context}.op"),
            update,
            _enum(AtomicMemoryOrder, obj["order"], f"{context}.order"),
            _enum(AtomicMemoryScope, obj["scope"], f"{context}.scope"),
        )

    if kind is OperationKind.ELEMENTWISE:
        obj = _strict_object(
            value,
            required={"op"},
            optional={"scalar", "broadcast_axis", "instruction"},
            context=context,
        )
        op = _enum(ElementwiseOp, obj["op"], f"{context}.op")
        instruction = obj.get("instruction")
        if op is ElementwiseOp.TANH and instruction is None:
            raise ScheduleParseError(
                f"{context}.instruction is required for tanh so the backend does not "
                "choose its numerical and performance contract"
            )
        if op is not ElementwiseOp.TANH and instruction is not None:
            raise ScheduleParseError(
                f"{context}.instruction has no defined effect for {op.value}"
            )
        scalar = obj.get("scalar")
        if scalar is not None and (
            not isinstance(scalar, (int, float)) or isinstance(scalar, bool)
        ):
            raise ScheduleParseError(f"{context}.scalar must be a number")
        axis = obj.get("broadcast_axis")
        return ElementwiseParameters(
            op,
            float(scalar) if scalar is not None else None,
            _nonnegative_int(axis, f"{context}.broadcast_axis") if axis is not None else None,
            None
            if instruction is None
            else ElementwiseInstruction.from_dict(
                instruction, f"{context}.instruction"
            ),
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

    @property
    def produced_pipeline_kind(self) -> PipelineKind | None:
        """The pipeline kind this producer can drive in the implemented subset."""

        if (
            self.kind is OperationKind.LOAD
            and isinstance(self.parameters, LoadParameters)
            and self.parameters.movement is LoadMovement.TMA
        ):
            return PipelineKind.TMA_TO_UMMA
        if self.kind is OperationKind.MMA:
            return PipelineKind.UMMA_TO_THREAD
        return None

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
class Residency:
    """What a Schedule commits to holding, rather than what it happens to need.

    The analysis derives an upper bound on resident CTAs and an optimistic lower bound on
    logical register storage, and reports both. It does not claim ptxas's eventual register
    allocation. The surveyed kernel work writes resource bounds -- at most forty-eight
    registers with no spill, two CTAs resident, sixty-four TMEM columns so both fit -- in
    task prose and template assertions because the Schedule otherwise has nowhere to put
    them.

    Declaring one turns the same derivation into a gate: the Schedule states the residency
    it needs and the verifier holds it to it.

    `registers_per_thread` is a cap the backend enforces, which is a real choice -- capping
    below the logical storage lower bound cannot hold the declared values without a spill.
    `allow_spill` says whether that trade was intended. Actual allocation and spill counts
    remain toolchain evidence rather than facts inferred by the Schedule verifier.
    """

    ctas_per_multiprocessor: int | None
    registers_per_thread: int | None
    allow_spill: bool

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "Residency":
        obj = _strict_object(
            value,
            required=set(),
            optional={"ctas_per_multiprocessor", "registers_per_thread", "allow_spill"},
            context=context,
        )
        ctas = obj.get("ctas_per_multiprocessor")
        registers = obj.get("registers_per_thread")
        if ctas is None and registers is None:
            raise ScheduleParseError(f"{context} declares no commitment")
        return cls(
            None if ctas is None else _positive_int(ctas, f"{context}.ctas_per_multiprocessor"),
            None if registers is None else _positive_int(registers, f"{context}.registers_per_thread"),
            _boolean(obj.get("allow_spill", False), f"{context}.allow_spill"),
        )


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
