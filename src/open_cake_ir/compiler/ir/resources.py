"""Roles, storage declarations, buffer relations, synchronization and residency."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from ._parse import (
    ScheduleParseError,
    _enum,
    _nonnegative_int,
    _object_list,
    _positive_int,
    _strict_object,
    _string,
    _string_tuple,
)
from .vocabulary import (
    BarrierMechanism,
    BufferMode,
    ByteOrder,
    PackedBlockFormat,
    DType,
    MemorySpace,
    Swizzle,
    TMEM_COLUMN_BYTES,
)


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
class PackedBlockField:
    """One typed, non-overlapping field in a packed record."""

    name: str
    byte_offset: int
    dtype: DType
    elements: int

    @property
    def size_bytes(self) -> int:
        return self.dtype.itemsize * self.elements


@dataclass(frozen=True)
class PackedBlockContract:
    """Canonical mechanical ABI for one closed packed-block format."""

    logical_extent: int
    record_bytes: int
    record_alignment_bytes: int
    byte_order: ByteOrder
    fields: tuple[PackedBlockField, ...]
    nibble_logical_order: tuple[int, ...] = ()


# One authority for the bytes shared by Workload materialization, Schedule validation
# and future decode/encode operations.  Q4's tuple is physical nibble order: low then
# high for each payload byte maps to logical j then 16+j.
PACKED_BLOCK_FORMATS: Mapping[PackedBlockFormat, PackedBlockContract] = MappingProxyType(
    {
        PackedBlockFormat.GGML_Q4_0_V1: PackedBlockContract(
            logical_extent=32,
            record_bytes=18,
            record_alignment_bytes=2,
            byte_order=ByteOrder.LITTLE,
            fields=(
                PackedBlockField("d", 0, DType.FP16, 1),
                PackedBlockField("qs", 2, DType.UINT8, 16),
            ),
            nibble_logical_order=tuple(
                value for byte in range(16) for value in (byte, byte + 16)
            ),
        ),
        PackedBlockFormat.GGML_Q8_1_V1: PackedBlockContract(
            logical_extent=32,
            record_bytes=36,
            record_alignment_bytes=4,
            byte_order=ByteOrder.LITTLE,
            fields=(
                PackedBlockField("d", 0, DType.FP16, 1),
                PackedBlockField("s", 2, DType.FP16, 1),
                PackedBlockField("qs", 4, DType.INT8, 32),
            ),
        ),
    }
)


@dataclass(frozen=True)
class PackedBlockRelation:
    """Bind one UINT8 Buffer axis to a closed packed-record ABI."""

    format: PackedBlockFormat
    record_axis: int

    @property
    def contract(self) -> PackedBlockContract:
        return PACKED_BLOCK_FORMATS[self.format]

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "PackedBlockRelation":
        obj = _strict_object(
            value,
            required={"format", "record_axis"},
            context=context,
        )
        return cls(
            _enum(PackedBlockFormat, obj["format"], f"{context}.format"),
            _nonnegative_int(obj["record_axis"], f"{context}.record_axis"),
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
    packed_block: PackedBlockRelation | None = None

    @property
    def is_scalar(self) -> bool:
        """The canonical one-value shape, also used by scalar loads/reductions."""
        return self.shape == (1,)

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
                "packed_block",
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
            PackedBlockRelation.from_dict(obj["packed_block"], f"{context}.packed_block")
            if "packed_block" in obj else None,
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


@dataclass(frozen=True)
class Residency:
    """What a Schedule commits to holding, rather than what it happens to need.

    The analysis derives an upper bound on resident CTAs from exact threads and explicit
    shared/tensor allocations. Logical register-Buffer pressure is reported separately;
    it is not ptxas's eventual allocation and has no sound direction against it. The
    surveyed kernel work writes resource choices -- a ``maxnreg`` cap, two CTAs resident,
    sixty-four TMEM columns so both fit -- in task prose and template assertions because
    the Schedule otherwise has nowhere to put them.

    Declaring one turns the same derivation into a gate: the Schedule states the residency
    it needs and the verifier holds it to it.

    `registers_per_thread` is a cap the backend receives, which is a real compilation
    choice. Actual allocation and spill counts remain toolchain evidence rather than facts
    inferred by the Schedule verifier.
    """

    ctas_per_multiprocessor: int | None
    registers_per_thread: int | None

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "Residency":
        obj = _strict_object(
            value,
            required=set(),
            optional={"ctas_per_multiprocessor", "registers_per_thread"},
            context=context,
        )
        ctas = obj.get("ctas_per_multiprocessor")
        registers = obj.get("registers_per_thread")
        if ctas is None and registers is None:
            raise ScheduleParseError(f"{context} declares no commitment")
        return cls(
            None if ctas is None else _positive_int(ctas, f"{context}.ctas_per_multiprocessor"),
            None if registers is None else _positive_int(registers, f"{context}.registers_per_thread"),
        )
