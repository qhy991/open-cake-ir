"""Typed operations and their kind-specific parameter parsing.

Parameter structure stays beside its operation; cross-object legality and
backend capability remain the verifier's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

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
    AtomicMemoryOrder,
    AtomicMemoryScope,
    AtomicOp,
    DType,
    ElementwiseOp,
    EpilogueFormula,
    IndexTieBreak,
    LoadMovement,
    LoadReuse,
    NaNPolicy,
    OperandMajorMode,
    OperandSource,
    OperationKind,
    PipelineKind,
    ReduceOp,
    ReductionScope,
    ScanDirection,
    ScanOp,
)


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
    across_loop: bool = True


@dataclass(frozen=True)
class ScanParameters:
    """An inclusive running prefix along one declared axis."""

    op: ScanOp
    axis: int
    direction: ScanDirection


@dataclass(frozen=True)
class TopKParameters:
    """Greatest FP32 or signed-INT32 values and positions from rank-one tiles.

    Descending result order is part of the operation rather than an optional spelling.
    ``across_loop`` carries the same values/indices state across tiles of one declared
    loop for FP32 scores.  ``source_tiles_per_merge`` makes the one admitted delayed
    state-update cadence visible: one is the historical per-tile merge and two batches
    exactly two source tiles before each merge.  Resident INT32 ordering is admitted
    while carried INT32 state remains an explicit backend exclusion. Group formation and
    batched routing remain separate operations.
    """

    k: int
    tie_break: IndexTieBreak
    nan_policy: NaNPolicy
    across_loop: bool = False
    source_tiles_per_merge: int = 1


@dataclass(frozen=True)
class OnlineSoftmaxParameters:
    """Stable weighted softmax reduction carried across one tile loop.

    The operation consumes one FP32 logits tile and one value tile.  Its four outputs
    are running maximum, normalizer, FP32 weighted accumulator, and the final normalized
    accumulator.  Making the state explicit lets verification reason about dtype,
    lifetime, and placement while lowering owns only the recurrence mechanics.
    """

    axis: int
    scope: ReductionScope
    sentinel: int | None = None


@dataclass(frozen=True)
class IndexExpandParameters:
    """Expand each selected source group into a flat run of source positions."""

    scale: int
    extent: int
    sentinel: int


@dataclass(frozen=True)
class CastParameters:
    """One explicit numeric representation conversion."""

    to: DType


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
    vocabulary. FMA binds one RN-even, non-FTZ ternary operation. Tanh's approximate
    PTX instruction differs in cost and numerical behaviour from a libdevice call.
    Leaving it to the backend
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
    FMA instead requires three same-shaped register reads and an instruction contract.
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
    ScanParameters,
    TopKParameters,
    IndexExpandParameters,
    OnlineSoftmaxParameters,
    AtomicRmwParameters,
    CastParameters,
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
        obj = _strict_object(
            value,
            required={"op", "axis", "scope"},
            optional={"across_loop"},
            context=context,
        )
        if obj.get("across_loop") is True:
            raise ScheduleParseError(
                f"{context}.across_loop=true is the historical omitted spelling"
            )
        return ReduceParameters(
            _enum(ReduceOp, obj["op"], f"{context}.op"),
            _nonnegative_int(obj["axis"], f"{context}.axis"),
            _enum(ReductionScope, obj["scope"], f"{context}.scope"),
            (
                _boolean(obj["across_loop"], f"{context}.across_loop")
                if "across_loop" in obj
                else True
            ),
        )

    if kind is OperationKind.SCAN:
        obj = _strict_object(
            value,
            required={"op", "axis"},
            optional={"direction"},
            context=context,
        )
        return ScanParameters(
            _enum(ScanOp, obj["op"], f"{context}.op"),
            _nonnegative_int(obj["axis"], f"{context}.axis"),
            _enum(
                ScanDirection,
                obj.get("direction", ScanDirection.FORWARD.value),
                f"{context}.direction",
            ),
        )

    if kind is OperationKind.TOP_K:
        obj = _strict_object(
            value,
            required={"k", "tie_break", "nan_policy"},
            optional={"across_loop", "source_tiles_per_merge"},
            context=context,
        )
        source_tiles_per_merge = (
            obj["source_tiles_per_merge"]
            if "source_tiles_per_merge" in obj
            else None
        )
        if "source_tiles_per_merge" in obj:
            source_tiles_per_merge = _positive_int(
                source_tiles_per_merge,
                f"{context}.source_tiles_per_merge",
            )
            if source_tiles_per_merge != 2:
                raise ScheduleParseError(
                    f"{context}.source_tiles_per_merge must be omitted for the "
                    "canonical one-source-tile cadence or be exactly 2"
                )
        return TopKParameters(
            _positive_int(obj["k"], f"{context}.k"),
            _enum(IndexTieBreak, obj["tie_break"], f"{context}.tie_break"),
            _enum(NaNPolicy, obj["nan_policy"], f"{context}.nan_policy"),
            _boolean(obj.get("across_loop", False), f"{context}.across_loop"),
            1 if source_tiles_per_merge is None else source_tiles_per_merge,
        )

    if kind is OperationKind.ONLINE_SOFTMAX:
        obj = _strict_object(
            value,
            required={"axis", "scope"},
            optional={"sentinel"},
            context=context,
        )
        sentinel = obj.get("sentinel")
        if sentinel is not None and (
            not isinstance(sentinel, int) or isinstance(sentinel, bool)
        ):
            raise ScheduleParseError(f"{context}.sentinel must be an integer")
        return OnlineSoftmaxParameters(
            _nonnegative_int(obj["axis"], f"{context}.axis"),
            _enum(ReductionScope, obj["scope"], f"{context}.scope"),
            sentinel,
        )

    if kind is OperationKind.INDEX_EXPAND:
        obj = _strict_object(
            value,
            required={"scale", "extent", "sentinel"},
            context=context,
        )
        sentinel = obj["sentinel"]
        if not isinstance(sentinel, int) or isinstance(sentinel, bool):
            raise ScheduleParseError(f"{context}.sentinel must be an integer")
        return IndexExpandParameters(
            _positive_int(obj["scale"], f"{context}.scale"),
            _positive_int(obj["extent"], f"{context}.extent"),
            sentinel,
        )

    if kind is OperationKind.CAST:
        obj = _strict_object(value, required={"to"}, context=context)
        return CastParameters(_enum(DType, obj["to"], f"{context}.to"))

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
        if op in (ElementwiseOp.TANH, ElementwiseOp.FMA) and instruction is None:
            raise ScheduleParseError(
                f"{context}.instruction is required for {op.value} so the backend does not "
                "choose its numerical and performance contract"
            )
        if op not in (ElementwiseOp.TANH, ElementwiseOp.FMA) and instruction is not None:
            raise ScheduleParseError(
                f"{context}.instruction has no defined effect for {op.value}"
            )
        if op is ElementwiseOp.FMA and ({"scalar", "broadcast_axis"} & obj.keys()):
            raise ScheduleParseError(
                f"{context}: fma requires three same-shaped register operands; "
                "scalar and broadcast_axis are not admitted"
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
