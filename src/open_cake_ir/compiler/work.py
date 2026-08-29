"""Declared work: the arithmetic a Schedule commits to and the global bytes it must move.

`analysis` bounds how much of one multiprocessor a CTA occupies and `ranking` orders
candidates by how much of the device the grid fills. Neither says how much *work* the
program does, so a measured time had nothing to be measured against: a kernel that ran in
forty microseconds was fast or slow relative to nothing.

This derives that missing half from the same declarations, and it is not the number the
analysis has refused from the start. A predicted time would divide work by a rate the
Target does not declare. These are counts. `flops` is what the declared operations commit
to computing and `compulsory_bytes` is what the declared global Buffers commit to moving.
Neither mentions a clock, a bandwidth or a second, and nothing here is divided by one.

What a caller may then do with a *measured* time and a separately declared peak rate is
divide, and the useful property of that quotient is that it cannot exceed one: no kernel
performs more arithmetic than the arithmetic peak, and none moves compulsory bytes faster
than the memory system moves any bytes. A quotient above one is therefore a refutation of
the work model or of the declared peak rather than an unfalsifiable estimate. That
ceiling is the whole reason this is worth deriving; the refuted wave-count term never had
one. Supplying the peak is not this module's business and neither is the measurement.

Two error directions, and they are stated rather than hidden:

* `flops` is exact when `uncounted_arithmetic` is empty and a lower bound otherwise. An
  operation whose arithmetic the IR does not decompose is named instead of guessed at.
* `compulsory_bytes` is exact when `partially_addressed` is empty and an upper bound
  otherwise, because it charges every global Buffer an operation touches in full.

So the quotient of the two is a lower bound on arithmetic intensity under either error,
which is the direction that keeps a memory-bound Schedule from reading as compute-bound.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ir import (
    AccessIndexKind,
    ElementwiseOp,
    ElementwiseParameters,
    MemorySpace,
    MmaParameters,
    Operation,
    OperationKind,
    ReduceOp,
    ReduceParameters,
    Schedule,
    StoreParameters,
)

MULTIPLY_ADD_FLOPS = 2
"""A contraction's inner step is one multiply and one add, which is the convention every
published FLOP count for a matmul uses. Naming it is cheaper than explaining a 2."""

_TRANSCENDENTAL = frozenset(
    {ElementwiseOp.RSQRT, ElementwiseOp.EXP, ElementwiseOp.TANH}
)
"""Primitives with no defensible operation count.

`libdevice.tanh.f32` and `tanh.approx.f32` are both admissible spellings of one declared
primitive and they do not cost the same, which is exactly why v15 made the Schedule name
the contract. A module that charged both the same number would be inventing the agreement
the IR refuses to assume. These abstain instead, and the caller is told which ones did.
"""

_NON_ARITHMETIC_KINDS = frozenset(
    {
        OperationKind.LOAD,
        OperationKind.STORE,
        OperationKind.REDUCE_ARGMIN,
        OperationKind.TOP_K,
        OperationKind.ATOMIC_RMW,
    }
)
"""Kinds that perform no floating-point arithmetic, counted as zero rather than abstained.

A load moves bytes, an argmin and a top-k compare and carry indices, and the one admitted
atomic is an INT32 add. Each consumes issue slots, and none of it is arithmetic a
floating-point peak is the ceiling for. Zero is a claim here, not an omission: abstaining
would make every Schedule that stores its result report an unknown FLOP count.
"""

_UNCOUNTABLE_KINDS = frozenset({OperationKind.EPILOGUE, OperationKind.SCAN})
"""Kinds whose arithmetic the Schedule does not decompose far enough to count.

An `epilogue` names a whole formula in one token -- which is the granularity
`ElementwiseOp` exists to replace -- so counting one would mean hardcoding that formula's
arithmetic here, in a third place after the two emitters. A `scan` declares an associative
operator and an axis and leaves the parallel decomposition to the backend, and a
sequential prefix and a work-inefficient tree do not perform the same number of adds. The
Schedule does not say which, so neither does this.
"""


@dataclass(frozen=True)
class WorkBound:
    """What one Schedule's declarations commit to computing and to moving.

    Every field is a whole-grid quantity: per-operation work multiplied by the tile loops
    that repeat it and by the program tiles that spread it, so the numbers are comparable
    against one kernel launch.
    """

    program_tiles: int
    flops: int
    mma_flops: int
    arithmetic_contracts: tuple[str, ...]
    uncounted_arithmetic: tuple[str, ...]
    compulsory_read_bytes: int
    compulsory_written_bytes: int
    partially_addressed: tuple[str, ...]

    @property
    def contended_contract(self) -> str | None:
        """The one instruction contract this Schedule's counted contractions issue.

        None when there is no contraction, and None again when there are several
        different ones: a Schedule that issues two instructions has no single rate to be
        compared against, and picking either would decide by which appeared first.
        """

        contracts = set(self.arithmetic_contracts)
        return contracts.pop() if len(contracts) == 1 else None

    @property
    def flops_exact(self) -> bool:
        """Whether every arithmetic operation in the Schedule was counted."""

        return not self.uncounted_arithmetic

    @property
    def compulsory_bytes(self) -> int:
        return self.compulsory_read_bytes + self.compulsory_written_bytes

    @property
    def compulsory_bytes_exact(self) -> bool:
        """Whether every global Buffer charged here is provably addressed in full."""

        return not self.partially_addressed

    @property
    def mma_fraction(self) -> float | None:
        """Share of counted arithmetic that a contraction performs.

        The question this answers is whether comparing this Schedule against an
        arithmetic peak means anything. A GEMM answers near one and the comparison is
        about the tensor cores; an rmsnorm answers zero and the same comparison is a
        statement about a kernel that was never trying to do arithmetic.
        """

        return None if not self.flops else self.mma_flops / self.flops

    @property
    def arithmetic_intensity(self) -> float | None:
        """Counted FLOPs per compulsory byte, or None when nothing must move.

        A lower bound whenever either half is inexact, since uncounted arithmetic can
        only raise the numerator and an over-charged Buffer can only lower the quotient.
        """

        return None if not self.compulsory_bytes else self.flops / self.compulsory_bytes


def program_tiles(schedule: Schedule) -> int | None:
    """How many program tiles of work the Schedule declares, or None when it cannot say.

    This is the work domain, not the launched CTA count, and on a persistent
    `program_map` those differ: persistence launches a fixed number of CTAs that walk
    this many tiles. Work is what is being counted here, so the walk is what matters and
    the launch is `ranking`'s business.
    """

    if schedule.grid is not None:
        x, y, z = schedule.grid
        return x * y * z
    program_map = schedule.program_map
    if program_map is None:
        return None
    total = 1
    for axis in program_map.axes:
        buffer = schedule.buffer(axis.buffer)
        if buffer is None or axis.dimension >= len(buffer.shape):
            return None
        total *= axis.tile_count(buffer.shape[axis.dimension])
    return total


def _elements(schedule: Schedule, name: str) -> int | None:
    buffer = schedule.buffer(name)
    return None if buffer is None else buffer.elements


def _loop_trips(schedule: Schedule, loop) -> int | None:
    buffer = schedule.buffer(loop.buffer)
    if buffer is None or loop.dimension >= len(buffer.shape):
        return None
    return -(-buffer.shape[loop.dimension] // loop.tile)


def _repetition(schedule: Schedule) -> dict[str, int] | None:
    """Operation id -> how many times one program tile executes it.

    A `tile_loops` body lists operation ids and the names of loops nested inside it, so
    an operation's repetition is the product of trip counts up its enclosing chain. The
    chain is walked with a seen-set because `loop_depth` already guards the same cycle
    and a second walk should not be the one that hangs.
    """

    trips: dict[str, int] = {}
    for loop in schedule.tile_loops:
        count = _loop_trips(schedule, loop)
        if count is None:
            return None
        trips[loop.name] = count

    innermost: dict[str, str] = {}
    for loop in schedule.tile_loops:
        for entry in loop.body:
            if entry not in trips and schedule.operation(entry) is not None:
                innermost[entry] = loop.name

    parent = schedule.loop_parent()
    factors: dict[str, int] = {}
    for operation in schedule.operations:
        name = innermost.get(operation.op_id)
        factor, seen = 1, set()
        while name is not None and name not in seen:
            seen.add(name)
            factor *= trips[name]
            name = parent.get(name)
        factors[operation.op_id] = factor
    return factors


def _operation_flops(schedule: Schedule, operation: Operation) -> int | None:
    """Floating-point operations one execution performs, or None when it is uncountable."""

    if operation.kind in _NON_ARITHMETIC_KINDS:
        return 0
    if operation.kind in _UNCOUNTABLE_KINDS:
        return None

    if operation.kind is OperationKind.MMA:
        parameters = operation.parameters
        if not isinstance(parameters, MmaParameters):
            return None
        # `tile_shape` is the contraction this operation performs; an instruction shape
        # is the atom a backend issues repeatedly to perform it. Preferring the atom
        # would count one MMA where the tile needs many.
        shape = parameters.tile_shape
        if shape is None and parameters.instruction is not None:
            shape = parameters.instruction.shape
        if shape is None:
            return None
        m, n, k = shape
        return MULTIPLY_ADD_FLOPS * m * n * k

    if operation.kind is OperationKind.OUTER:
        # Each result element is one explicit multiply. Shape legality belongs to the
        # verifier; work only needs the declared result extent once it is available.
        return (
            _elements(schedule, operation.writes[0])
            if len(operation.writes) == 1
            else None
        )

    if operation.kind is OperationKind.ELEMENTWISE:
        parameters = operation.parameters
        if not isinstance(parameters, ElementwiseParameters) or len(operation.writes) != 1:
            return None
        if parameters.op in _TRANSCENDENTAL:
            return None
        # One arithmetic operation per element written, whatever the arity: a broadcast
        # operand is narrower than the result and the result is what got computed.
        return _elements(schedule, operation.writes[0])

    if operation.kind is OperationKind.REDUCE:
        parameters = operation.parameters
        if not isinstance(parameters, ReduceParameters):
            return None
        if parameters.op is not ReduceOp.SUM:
            # A max is a compare and a select. It folds an axis without adding anything,
            # and softmax's is the reason no published FLOP count charges for it.
            return 0
        if len(operation.reads) != 1 or len(operation.writes) != 1:
            return None
        read = _elements(schedule, operation.reads[0])
        written = _elements(schedule, operation.writes[0])
        if read is None or written is None:
            return None
        # Folding an axis of length L into one element is L-1 additions, and the written
        # shape is the read shape with that axis removed, so the difference is the count.
        return max(read - written, 0)

    return None


def _is_partially_addressed(schedule: Schedule, name: str) -> bool:
    """Whether a declared coordinate makes 'the whole Buffer moves' unsafe to assume.

    Four declarations do. A runtime `buffer` coordinate selects rows the Schedule cannot
    name, a `valid_extent` says a padded axis has a shorter live prefix, an `offset` or
    `extent` narrows an access to a sub-range so a sibling can own the rest, and a guarded
    store makes its active rows runtime data. Each one means the traffic charged here is
    an over-count rather than a measurement of it.
    """

    buffer = schedule.buffer(name)
    if buffer is not None and buffer.valid_extent is not None:
        return True
    if any(
        operation.kind is OperationKind.STORE
        and name in operation.writes
        and isinstance(operation.parameters, StoreParameters)
        and operation.parameters.valid_if is not None
        for operation in schedule.operations
    ):
        return True
    for access in schedule.access_maps:
        if access.buffer != name:
            continue
        for component in access.indices:
            if component.source is AccessIndexKind.BUFFER:
                return True
            if component.offset or component.extent is not None:
                return True
    return False


def _compulsory_traffic(schedule: Schedule) -> tuple[int, int, tuple[str, ...]]:
    """Unique global bytes that must cross the memory system at least once, each way.

    Charged per Buffer rather than per operation: two loads of one input are one input,
    and how many times a backend actually re-reads it is a measurement rather than a
    declaration. `stages` is excluded deliberately -- it counts resident copies of a
    staged tile, which is a residency fact `analysis` already charges for, not a second
    copy of the bytes in memory.
    """

    read = written = 0
    partial: list[str] = []
    for buffer in schedule.buffers:
        if buffer.space is not MemorySpace.GLOBAL:
            continue
        is_read = any(buffer.name in op.reads for op in schedule.operations)
        is_written = any(buffer.name in op.writes for op in schedule.operations)
        if not (is_read or is_written):
            continue
        extent = buffer.elements * buffer.dtype.itemsize
        if is_read:
            read += extent
        if is_written:
            written += extent
        if _is_partially_addressed(schedule, buffer.name):
            partial.append(buffer.name)
    return read, written, tuple(partial)


def work_bound(schedule: Schedule) -> WorkBound | None:
    """Declared arithmetic and compulsory traffic, or None when the structure cannot be walked.

    None means a program axis or a tile loop names a Buffer dimension that is not there,
    which a gated Schedule cannot do. It is the same abstention `residency_upper_bound`
    makes when the Target says too little: a missing answer rather than a zero.
    """

    tiles = program_tiles(schedule)
    if tiles is None:
        return None
    repetition = _repetition(schedule)
    if repetition is None:
        return None

    flops = mma_flops = 0
    uncounted: list[str] = []
    contracts: list[str] = []
    for operation in schedule.operations:
        per_execution = _operation_flops(schedule, operation)
        if per_execution is None:
            uncounted.append(operation.op_id)
            continue
        total = per_execution * repetition[operation.op_id] * tiles
        flops += total
        if operation.kind is OperationKind.MMA:
            mma_flops += total
            # Which instruction performs the counted arithmetic, so a caller comparing
            # against a peak rate compares against the rate for *this* instruction
            # rather than for a dtype that several instructions share.
            parameters = operation.parameters
            if isinstance(parameters, MmaParameters) and parameters.instruction:
                contracts.append(parameters.instruction.contract)

    read, written, partial = _compulsory_traffic(schedule)
    return WorkBound(
        program_tiles=tiles,
        flops=flops,
        mma_flops=mma_flops,
        arithmetic_contracts=tuple(contracts),
        uncounted_arithmetic=tuple(uncounted),
        compulsory_read_bytes=read,
        compulsory_written_bytes=written,
        partially_addressed=partial,
    )
