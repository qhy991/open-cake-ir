"""Canonical typed Schedule IR and its stable import boundary.

Each definition has one owner: vocabulary, resources, mapping, operations or
schedule. This package re-exports those same objects for Compiler consumers;
internal modules import their dependencies directly, never through this entry.

Parsing admits structure. Cross-object semantics, resource limits, synchronization
and backend capability belong to the verifier. Python authoring and JSON both
reach this one model; the former legacy models are not separate parsing routes.

See docs/IR_GUIDE.md for the data model, dependency direction and extension guide.
"""

from ._parse import (
    ScheduleParseError,
    _enum,
    _string,
)

from .vocabulary import (
    AccessIndexKind,
    AtomicMemoryOrder,
    AtomicMemoryScope,
    AtomicOp,
    BarrierMechanism,
    BoundaryPolicy,
    BufferMode,
    DType,
    ElementwiseOp,
    EpilogueFormula,
    IndexTieBreak,
    LoadMovement,
    LoadReuse,
    LoweringBackend,
    MemorySpace,
    NaNPolicy,
    OperandMajorMode,
    OperandSource,
    OperationKind,
    PipelineKind,
    ReduceOp,
    ReductionScope,
    ScanDirection,
    ScanOp,
    Swizzle,
    TMEM_COLUMN_BYTES,
    TMEM_LANES,
    TMEM_WORD_BYTES,
)

from .resources import (
    Allocation,
    Barrier,
    Buffer,
    Pipeline,
    Residency,
    Role,
    ScaleRelation,
    ValidExtentRelation,
)

from .mapping import (
    AccessIndex,
    AccessMap,
    LoopStop,
    ProgramAxis,
    ProgramMap,
    RangeOptions,
    TileLoop,
)

from .operations import (
    AtomicRmwParameters,
    CastParameters,
    CopyAtom,
    ElementwiseInstruction,
    ElementwiseParameters,
    EpilogueParameters,
    FenceProxyParameters,
    IndexExpandParameters,
    LoadParameters,
    MmaInstruction,
    MmaParameters,
    OnlineSoftmaxParameters,
    Operation,
    OperationParameters,
    PLACED_CONTRACT_PREFIXES,
    PLACEMENT_FIELDS,
    ReduceArgminParameters,
    ReduceParameters,
    ScanParameters,
    StoreParameters,
    TopKParameters,
)

from .schedule import (
    LoweringRoute,
    Schedule,
    _SCHEDULE_OPTIONAL,
    _SCHEDULE_REQUIRED,
)
