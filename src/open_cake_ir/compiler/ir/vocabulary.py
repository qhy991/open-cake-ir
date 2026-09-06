"""Closed Schedule vocabularies and shared hardware constants."""

from __future__ import annotations

from enum import Enum


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
    INDEX_EXPAND = "index_expand"
    ONLINE_SOFTMAX = "online_softmax"
    ATOMIC_RMW = "atomic_rmw"
    CAST = "cast"
    ELEMENTWISE = "elementwise"
    SCAN = "scan"
    STORE = "store"


class LoweringBackend(str, Enum):
    """Mechanism that materializes source, independent of operator or Workload."""

    TRITON = "triton"
    CUTLASS_CUTE_DSL = "cutlass_cute_dsl"
    CHECKED_CUDA_ASSET = "checked_cuda_asset"


class ScanOp(str, Enum):
    """The associative operator a prefix scan accumulates with."""

    SUM = "sum"


class ScanDirection(str, Enum):
    """Which end of the scanned axis the prefix accumulates from."""

    FORWARD = "forward"
    REVERSE = "reverse"


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
    RELU = "relu"
    TANH = "tanh"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    FMA = "fma"

    @property
    def arity(self) -> int:
        if self is ElementwiseOp.FMA:
            return 3
        return (
            1
            if self
            in (
                ElementwiseOp.SQUARE,
                ElementwiseOp.RSQRT,
                ElementwiseOp.EXP,
                ElementwiseOp.RELU,
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
