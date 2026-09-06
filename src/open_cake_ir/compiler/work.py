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

What a caller may then do with a *measured* time and a separately declared rate is divide.
When that rate is a device-specification ceiling, the useful property of the quotient is
that a sound lower bound cannot exceed one: no kernel performs more arithmetic than the
architecture ceiling, and exact compulsory bytes do not cross faster than the specified
memory ceiling. A microbenchmark rate is instead a comparative reference and may be
exceeded. Supplying either rate is not this module's business and neither is the
measurement.

Two error directions, and they are stated rather than hidden:

* `flops` is exact when `uncounted_arithmetic` is empty and a lower bound otherwise. An
  operation whose arithmetic the IR does not decompose, or whose whole-grid repetition
  is unknown, is named instead of guessed at.
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
    TileLoop,
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
class LoopTripDistribution:
    """One loop's trips across its declared whole-grid program domain."""

    estimate_kind: str
    trips: tuple[int, ...] | None
    multiplicity: int | None
    missing: tuple[str, ...] = ()


@dataclass(frozen=True)
class OperationRepetition:
    """One operation's whole-grid execution count, or why it is unknown."""

    operation: str
    estimate_kind: str
    whole_grid: int | None
    missing: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScheduledTransfer:
    """Unmasked logical LOAD/STORE payload, not cache or DRAM transactions."""

    operation: str
    read_bytes_upper_bound: int | None
    written_bytes_upper_bound: int | None
    missing: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkBound:
    """What one Schedule's declarations commit to computing and to moving.

    Every field is a whole-grid quantity. ``operation_repetitions`` owns how the declared
    loops and program coordinates compose; arithmetic with an unknown repetition is
    omitted and named in ``uncounted_arithmetic``, preserving a lower bound rather than
    substituting a static maximum.
    """

    program_tiles: int
    flops: int
    mma_flops: int
    arithmetic_contracts: tuple[str, ...]
    uncounted_arithmetic: tuple[str, ...]
    operation_repetitions: tuple[OperationRepetition, ...]
    compulsory_read_bytes: int
    compulsory_written_bytes: int
    partially_addressed: tuple[str, ...]
    scheduled_transfers: tuple[ScheduledTransfer, ...] = ()

    @property
    def scheduled_read_bytes_upper_bound(self) -> int | None:
        values = [item.read_bytes_upper_bound for item in self.scheduled_transfers]
        return None if any(value is None for value in values) else sum(values)

    @property
    def scheduled_written_bytes_upper_bound(self) -> int | None:
        values = [item.written_bytes_upper_bound for item in self.scheduled_transfers]
        return None if any(value is None for value in values) else sum(values)

    @property
    def contended_contract(self) -> str | None:
        """The one instruction contract this Schedule's counted contractions issue.

        None when there is no counted contraction, and None again when there are several
        different ones: a Schedule that issues two instructions has no single rate to be
        compared against, and picking either would decide by which appeared first. An
        uncounted contraction must not lend its tensor rate to unrelated counted scalar
        arithmetic.
        """

        if not self.mma_flops:
            return None
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


def loop_trip_distribution(
    schedule: Schedule, loop: TileLoop
) -> LoopTripDistribution:
    """Derive exact trips after add, floor-divide, clamp, then tile ceiling.

    The order mirrors the Triton emitter.  A static loop has one trip count repeated for
    every program tile; a program-derived stop has one count per scalar coordinate and a
    multiplicity for all other program axes.
    """

    static_trips = _loop_trips(schedule, loop)
    tiles = program_tiles(schedule)
    if static_trips is None:
        return LoopTripDistribution(
            "unknown", None, None, ("loop extent is unavailable",)
        )
    if tiles is None:
        return LoopTripDistribution(
            "unknown", None, None, ("program tile domain is unavailable",)
        )
    if loop.stop is None:
        return LoopTripDistribution("exact", (static_trips,), tiles)

    program_map = schedule.program_map
    axis = None if program_map is None else program_map.axis(loop.stop.program)
    if axis is None or axis.tile != 1:
        return LoopTripDistribution(
            "unknown",
            None,
            None,
            ("loop stop lacks one scalar program axis",),
        )
    axis_buffer = schedule.buffer(axis.buffer)
    if axis_buffer is None or axis.dimension >= len(axis_buffer.shape):
        return LoopTripDistribution(
            "unknown", None, None, ("loop-stop program extent is unavailable",)
        )
    axis_extent = axis_buffer.shape[axis.dimension]
    if axis_extent <= 0 or tiles % axis_extent:
        return LoopTripDistribution(
            "unknown", None, None, ("loop-stop program multiplicity is unavailable",)
        )
    loop_buffer = schedule.buffer(loop.buffer)
    assert loop_buffer is not None
    loop_extent = loop_buffer.shape[loop.dimension]
    trips = []
    for coordinate in range(axis_extent):
        stop = (coordinate + loop.stop.add) // loop.stop.floor_div
        stop = min(max(stop, 0), loop_extent)
        trips.append((stop + loop.tile - 1) // loop.tile if stop else 0)
    return LoopTripDistribution("exact", tuple(trips), tiles // axis_extent)


def operation_repetitions(
    schedule: Schedule,
) -> tuple[OperationRepetition, ...] | None:
    """Return per-operation exact whole-grid repetitions without max-trip fallback."""

    tiles = program_tiles(schedule)
    if tiles is None:
        return None

    loops = {loop.name: loop for loop in schedule.tile_loops}
    scopes: dict[str, list[str]] = {}
    for loop in schedule.tile_loops:
        for entry in loop.body:
            if entry not in loops and schedule.operation(entry) is not None:
                scopes.setdefault(entry, []).append(loop.name)

    parent = schedule.loop_parent()
    rows: list[OperationRepetition] = []
    for operation in schedule.operations:
        direct = scopes.get(operation.op_id, [])
        if len(direct) > 1:
            rows.append(
                OperationRepetition(
                    operation.op_id,
                    "unknown",
                    None,
                    ("operation belongs to multiple loop scopes",),
                )
            )
            continue
        chain = []
        name = direct[0] if direct else None
        seen: set[str] = set()
        chain_error = None
        while name is not None:
            if name in seen or name not in loops:
                chain_error = "operation loop chain is cyclic or unresolved"
                break
            seen.add(name)
            chain.append(loops[name])
            name = parent.get(name)
        if chain_error is not None:
            rows.append(
                OperationRepetition(
                    operation.op_id,
                    "unknown",
                    None,
                    (chain_error,),
                )
            )
            continue
        dynamic = [loop for loop in chain if loop.stop is not None]
        if len(dynamic) > 1:
            rows.append(
                OperationRepetition(
                    operation.op_id,
                    "unknown",
                    None,
                    ("operation loop chain has more than one dynamic stop",),
                )
            )
            continue
        static_factor = 1
        missing = None
        for loop in chain:
            if loop.stop is not None:
                continue
            count = _loop_trips(schedule, loop)
            if count is None:
                missing = "static loop extent is unavailable"
                break
            static_factor *= count
        if missing is not None:
            rows.append(
                OperationRepetition(operation.op_id, "unknown", None, (missing,))
            )
            continue
        if not dynamic:
            rows.append(
                OperationRepetition(
                    operation.op_id, "exact", tiles * static_factor
                )
            )
            continue
        distribution = loop_trip_distribution(schedule, dynamic[0])
        if distribution.trips is None or distribution.multiplicity is None:
            rows.append(
                OperationRepetition(
                    operation.op_id,
                    "unknown",
                    None,
                    distribution.missing,
                )
            )
            continue
        rows.append(
            OperationRepetition(
                operation.op_id,
                "exact",
                sum(distribution.trips)
                * distribution.multiplicity
                * static_factor,
            )
        )
    return tuple(rows)


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

    if operation.kind is OperationKind.ELEMENTWISE:
        parameters = operation.parameters
        if not isinstance(parameters, ElementwiseParameters) or len(operation.writes) != 1:
            return None
        if parameters.op in _TRANSCENDENTAL:
            return None
        if parameters.op is ElementwiseOp.FMA:
            elements = _elements(schedule, operation.writes[0])
            return None if elements is None else MULTIPLY_ADD_FLOPS * elements
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
    name, a `valid_extent` says a padded axis has a shorter live prefix, a stopped loop may
    leave a suffix untouched across the whole program, and an `offset` or `extent` narrows
    an access to a sub-range so a sibling can own the rest. Each one means the traffic
    charged here is an over-count rather than a measurement of it.
    """

    buffer = schedule.buffer(name)
    if buffer is not None and buffer.valid_extent is not None:
        return True
    for loop in schedule.tile_loops:
        if loop.buffer != name or loop.stop is None:
            continue
        distribution = loop_trip_distribution(schedule, loop)
        if distribution.trips is None:
            return True
        assert buffer is not None and loop.dimension < len(buffer.shape)
        # The emitted loop starts every tile whose start is below `stop`, and the access
        # mask clips that tile only to the static Buffer extent. Whole-Program union
        # coverage is therefore the largest padded trip count, not the largest raw stop.
        padded_coverage = min(
            max(distribution.trips, default=0) * loop.tile,
            buffer.shape[loop.dimension],
        )
        if padded_coverage < buffer.shape[loop.dimension]:
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
    """Declared arithmetic and compulsory traffic over the walkable program domain.

    None means the program tile domain itself is unavailable. A localized unknown loop
    repetition instead leaves the affected arithmetic uncounted and carries its reason
    in ``operation_repetitions``; known operations remain a sound lower bound.
    """

    tiles = program_tiles(schedule)
    if tiles is None:
        return None
    repetitions = operation_repetitions(schedule)
    if repetitions is None:
        return None
    repetition_by_operation = {row.operation: row for row in repetitions}

    flops = mma_flops = 0
    uncounted: list[str] = []
    contracts: list[str] = []
    for operation in schedule.operations:
        if operation.kind is OperationKind.MMA:
            parameters = operation.parameters
            if isinstance(parameters, MmaParameters) and parameters.instruction:
                contracts.append(parameters.instruction.contract)
        per_execution = _operation_flops(schedule, operation)
        repetition = repetition_by_operation[operation.op_id]
        if repetition.whole_grid is None:
            if per_execution is None or per_execution > 0:
                uncounted.append(operation.op_id)
            continue
        if per_execution is None:
            uncounted.append(operation.op_id)
            continue
        total = per_execution * repetition.whole_grid
        flops += total
        if operation.kind is OperationKind.MMA:
            mma_flops += total

    read, written, partial = _compulsory_traffic(schedule)
    transfers = _scheduled_transfers(schedule, repetition_by_operation)
    return WorkBound(
        program_tiles=tiles,
        flops=flops,
        mma_flops=mma_flops,
        arithmetic_contracts=tuple(contracts),
        uncounted_arithmetic=tuple(uncounted),
        operation_repetitions=repetitions,
        compulsory_read_bytes=read,
        compulsory_written_bytes=written,
        partially_addressed=partial,
        scheduled_transfers=transfers,
    )


def _scheduled_transfers(
    schedule: Schedule, repetitions: dict[str, OperationRepetition]
) -> tuple[ScheduledTransfer, ...]:
    """Count values moved by explicit IR transfers, charging every loop execution.

    Tile payload includes lanes which an access mask may remove. Staging copies do not
    multiply transfers. Backend duplication/coalescing/caching is outside this quantity,
    so this is deliberately not a bound on physical DRAM/L2 traffic.
    """
    rows = []
    for operation in schedule.operations:
        reads = [schedule.buffer(name) for name in operation.reads]
        writes = [schedule.buffer(name) for name in operation.writes]
        global_reads = [item for item in reads if item is not None and item.space is MemorySpace.GLOBAL]
        global_writes = [item for item in writes if item is not None and item.space is MemorySpace.GLOBAL]
        if not global_reads and not global_writes:
            continue
        count = repetitions[operation.op_id]
        read_bytes = None if global_reads else 0
        written_bytes = None if global_writes else 0
        missing = []
        if count.whole_grid is None:
            missing.extend(count.missing or ("operation repetition is unavailable",))
        elif count.whole_grid == 0:
            read_bytes = written_bytes = 0
        elif (operation.kind is OperationKind.LOAD and len(global_reads) == 1
              and not global_writes and len(writes) == 1 and writes[0] is not None):
            read_bytes = writes[0].elements * global_reads[0].dtype.itemsize * count.whole_grid
        elif (operation.kind is OperationKind.STORE and len(global_writes) == 1
              and not global_reads and len(reads) == 1 and reads[0] is not None):
            written_bytes = reads[0].elements * global_writes[0].dtype.itemsize * count.whole_grid
        else:
            missing.append("global access is not one explicit LOAD or STORE payload")
        rows.append(ScheduledTransfer(operation.op_id, read_bytes, written_bytes, tuple(missing)))
    return tuple(rows)
