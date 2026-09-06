"""Static residency bounds over the typed IR.

The paper's harness reports a `performance analysis` row as a *report*: a cost estimate
plus bottleneck attribution. This module supplies the attribution half. It does not
estimate time, because a Target declares no clock and no bandwidth, and a number derived
from neither would be invented rather than analysed.

What it derives is an upper bound on residency from exact Schedule quantities and Target
facts. Threads and explicit shared/tensor allocations are in that domain. Register
Buffers are different: a backend may distribute, alias, recompute, spill, or place their
logical values elsewhere, so their live logical extent is only a pressure proxy. It is
reported separately and never used as a physical-register bound or legality gate.

Loop-carried ``top_k`` also declares a merge cadence whose pending ranked-key payload is
not an author-visible Buffer.  This module derives that logical pressure and exact
whole-grid update counts when the declared program/loop-stop domain is walkable; otherwise
it names the missing domain and abstains.
"""

from __future__ import annotations

from dataclasses import dataclass

from .compiled_resources import CompiledResources
from .ir import MemorySpace, Operation, OperationKind, Schedule, TopKParameters
from .target import Target
from .work import loop_trip_distribution, program_tiles

REGISTER_BYTES = 4
TOP_K_RANKED_KEY_BYTES = 8
"""The admitted lowering carries each pending FP32 value and INT32 source position as
one packed uint64 ordering key.  This is logical lowering state, not a claim about ptxas
physical-register allocation."""


@dataclass(frozen=True)
class ResidencyBound:
    """How many CTAs one multiprocessor can hold, and what stops it being more."""

    resource: str
    per_cta: int
    per_multiprocessor: int
    ctas: int

    @property
    def unit(self) -> str:
        return {
            "threads": "threads",
            "registers": "registers",
        }.get(self.resource, "bytes")


@dataclass(frozen=True)
class ResidencyUpperBound:
    """Per-resource upper bounds on resident CTAs for one declared Schedule."""

    bounds: tuple[ResidencyBound, ...]

    @property
    def binding(self) -> ResidencyBound | None:
        """The bound that stops residency being higher."""

        return min(self.bounds, key=lambda b: b.ctas) if self.bounds else None

    @property
    def ctas_per_multiprocessor(self) -> int | None:
        return self.binding.ctas if self.binding else None


@dataclass(frozen=True)
class TopKMergeStructure:
    """Schedule-derived state and cadence for one canonical ``top_k``.

    The byte counts describe logical lowering payloads, not backend physical registers. The
    whole-grid counts are exact only when this analysis can walk the declared program axis
    and containing loop, including a query-derived stop.  An unsupported shape is reported
    as ``unknown`` rather than silently falling back to the loop buffer's full extent.
    """

    source_tiles_per_merge: int
    source_extent: int
    source_elements_per_merge: int
    merge_width: int
    loop_carried_state_bytes: int
    pending_source_key_elements: int
    pending_source_state_bytes: int
    count_estimate_kind: str
    whole_grid_source_tile_update_count: int | None
    whole_grid_full_group_merge_count: int | None
    whole_grid_tail_flush_merge_count: int | None
    whole_grid_merge_update_count: int | None
    count_missing: tuple[str, ...] = ()


@dataclass(frozen=True)
class TopKSelectionStructure:
    """Triton 3.7.1 frontend comparison work for one merge selection.

    ``comparison_lane_work`` counts input lanes participating in compare/select work in
    the pinned ``standard.py`` network: both lanes of compare-exchange and both inputs
    of its pairwise top-k reduction.  It is a toolchain-specific structural model, not a
    public API guarantee, cycle estimate, instruction count, or physical resource fact.
    The optimized algorithm is admitted only when public ``sort`` plus
    ``bitonic_merge`` removes at least one third of that baseline work.
    """

    algorithm: str
    comparison_model: str
    baseline_comparison_lane_work: int | None
    selected_comparison_lane_work: int | None
    comparison_lane_reduction_fraction: float | None


def top_k_selection_structure(
    k: int,
    merge_width: int,
    source_tiles_per_merge: int,
) -> TopKSelectionStructure:
    """Choose the exact batched-source selector whose structural saving is material."""

    if k <= 0 or k & (k - 1) or merge_width != 2 * k:
        return TopKSelectionStructure(
            "triton_topk", "triton_3_7_1_standard_py", None, None, None
        )
    log_k = k.bit_length() - 1
    baseline = k * (log_k**2 + 2 * log_k + 2)
    half_selection = k * (log_k + 1) * (log_k + 4) // 2
    if source_tiles_per_merge != 2 or 3 * half_selection > 2 * baseline:
        return TopKSelectionStructure(
            "triton_topk",
            "triton_3_7_1_standard_py",
            baseline,
            baseline,
            0.0,
        )
    return TopKSelectionStructure(
        "sorted_source_half_bitonic_merge",
        "triton_3_7_1_standard_py",
        baseline,
        half_selection,
        (baseline - half_selection) / baseline,
    )


def _containing_loop(schedule: Schedule, operation: Operation):
    chain = schedule.enclosing_loops(operation)
    return chain[-1] if chain else None


def top_k_merge_structure(
    schedule: Schedule, operation: Operation
) -> TopKMergeStructure | None:
    """Return the canonical merge geometry and exact structural cadence when derivable."""

    if (
        operation.kind is not OperationKind.TOP_K
        or not isinstance(operation.parameters, TopKParameters)
        or not operation.reads
    ):
        return None
    source = schedule.buffer(operation.reads[0])
    if source is None or len(source.shape) != 1:
        return None

    parameters = operation.parameters
    source_extent = source.shape[0]
    source_tiles_per_merge = parameters.source_tiles_per_merge
    source_elements_per_merge = source_extent * source_tiles_per_merge
    merge_width = 2 * max(parameters.k, source_elements_per_merge)
    loop_carried_state_bytes = (
        parameters.k * TOP_K_RANKED_KEY_BYTES if parameters.across_loop else 0
    )
    pending_source_key_elements = (
        (source_tiles_per_merge - 1) * source_extent
        if parameters.across_loop
        else 0
    )
    pending_source_state_bytes = pending_source_key_elements * TOP_K_RANKED_KEY_BYTES

    def unknown(reason: str) -> TopKMergeStructure:
        return TopKMergeStructure(
            source_tiles_per_merge,
            source_extent,
            source_elements_per_merge,
            merge_width,
            loop_carried_state_bytes,
            pending_source_key_elements,
            pending_source_state_bytes,
            "unknown",
            None,
            None,
            None,
            None,
            (reason,),
        )

    loop = _containing_loop(schedule, operation)
    if loop is not None:
        if loop.name in schedule.loop_parent():
            return unknown("nested top_k loop cadence is not modeled")
        distribution = loop_trip_distribution(schedule, loop)
        if distribution.trips is None or distribution.multiplicity is None:
            return unknown(
                distribution.missing[0]
                if distribution.missing
                else "top_k loop cadence is unavailable"
            )
        trip_counts = distribution.trips
        factor = distribution.multiplicity
    else:
        if parameters.across_loop:
            return unknown("top_k has no unique containing loop")
        else:
            tile_count = program_tiles(schedule)
            if tile_count is None:
                return unknown("program tile domain is unavailable")
            trip_counts = (1,)
            factor = tile_count

    source_updates = sum(trip_counts) * factor
    full_groups = sum(
        trips // source_tiles_per_merge for trips in trip_counts
    ) * factor
    tail_flushes = sum(
        bool(trips % source_tiles_per_merge) for trips in trip_counts
    ) * factor
    return TopKMergeStructure(
        source_tiles_per_merge,
        source_extent,
        source_elements_per_merge,
        merge_width,
        loop_carried_state_bytes,
        pending_source_key_elements,
        pending_source_state_bytes,
        "exact",
        source_updates,
        full_groups,
        tail_flushes,
        full_groups + tail_flushes,
    )


def _pending_top_k_bytes_by_operation(schedule: Schedule) -> dict[str, int]:
    pending: dict[str, int] = {}
    for operation in schedule.operations:
        structure = top_k_merge_structure(schedule, operation)
        if structure is None or not structure.pending_source_state_bytes:
            continue
        loop = _containing_loop(schedule, operation)
        if loop is None:
            continue
        for scoped in schedule.loop_operations(loop):
            pending[scoped.op_id] = (
                pending.get(scoped.op_id, 0) + structure.pending_source_state_bytes
            )
    return pending


def _storage_classes(schedule: Schedule, registers) -> dict[str, str]:
    """Group same-shaped elementwise Buffers in the logical pressure proxy.

    An elementwise primitive reads a tile and writes a tile of the same shape, which a
    backend can often perform in place. Charging for each made composing arithmetic from
    primitives inflate the feature purely because the canonical form names each step
    (P3). A dot or reduction has a distinct result shape/lifetime, so only same-shaped
    elementwise chains are unioned. This is still a heuristic feature, not physical
    liveness analysis.
    """

    parent = {name: name for name in registers}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for operation in schedule.operations:
        if operation.kind is not OperationKind.ELEMENTWISE:
            continue
        if len(operation.writes) != 1:
            continue
        written = operation.writes[0]
        if written not in registers:
            continue
        for name in operation.reads:
            # Only a same-shaped operand can be overwritten in place; a broadcast
            # operand is narrower and outlives the step that reads it.
            if name in registers and registers[name].shape == registers[written].shape:
                parent[find(name)] = find(written)
                break
    return {name: find(name) for name in registers}


def _logical_register_pressure_bytes(schedule: Schedule) -> int:
    """Peak live logical bytes after simple aliasing and derived pending top-k state.

    Summing them charges a Schedule for every temporary it ever names, which reads the
    same whether two tiles overlap or one is dead before the other is written. That was
    close enough while an operation carried a whole formula and named few intermediates.
    Composing arithmetic from primitives names one buffer per step, and under a sum a
    decomposition that changes no kernel reports as no longer resident.

    A storage class is live from its first write to its last read in declared order.
    Anything the Schedule declares but never writes is charged for the whole program,
    since nothing here can say when it dies. This quantity has no sound direction against
    physical registers: a backend can add temporaries, but it can also distribute values
    across lanes, recompute them, alias them more aggressively, or realize them in another
    storage class. A two-source-tile top-k necessarily retains one packed key tile across
    loop trips, so that derived payload is charged across the containing loop body even
    though making an author declare an implementation Buffer would duplicate the operation
    contract. The compiled artifact remains the authority for physical allocation.
    """

    registers = {
        buffer.name: buffer
        for buffer in schedule.buffers
        if buffer.space is MemorySpace.REGISTER
    }
    if not registers:
        return 0

    classes = _storage_classes(schedule, registers)
    size: dict[str, int] = {}
    for name, buffer in registers.items():
        root = classes[name]
        size[root] = max(size.get(root, 0), buffer.size_bytes)

    first_write: dict[str, int] = {}
    last_read: dict[str, int] = {}
    for position, operation in enumerate(schedule.operations):
        for name in operation.writes:
            if name in registers:
                first_write.setdefault(classes[name], position)
        for name in operation.reads:
            if name in registers:
                last_read[classes[name]] = position

    total = len(schedule.operations)
    pending_top_k = _pending_top_k_bytes_by_operation(schedule)
    peak = 0
    for position in range(total):
        live = pending_top_k.get(schedule.operations[position].op_id, 0)
        for root, extent in size.items():
            birth = first_write.get(root)
            if birth is None:
                live += extent
                continue
            death = last_read.get(root, total)
            if birth <= position <= max(death, birth):
                live += extent
        peak = max(peak, live)
    return peak


def _allocation_bytes(schedule: Schedule, space: MemorySpace) -> int:
    return sum(
        allocation.size_bytes
        for allocation in schedule.allocations
        if allocation.space is space
    )


def residency_upper_bound(
    schedule: Schedule, target: Target, *, compiled_resources: CompiledResources | None = None
) -> ResidencyUpperBound | None:
    """Resource upper bounds; a compiled observation can replace logical declarations.

    Allocation granularity, barriers, carveout, and backend tensor-memory use may tighten
    these ceilings further. Physical registers are admitted only from compiled facts.
    """

    facts = target.occupancy
    if facts is None:
        return None

    if compiled_resources is not None and compiled_resources.target != target.target_id:
        raise ValueError("compiled resource target differs from residency Target")
    threads = (
        compiled_resources.threads_per_cta if compiled_resources is not None
        else schedule.total_warp_extent * target.warp_size
    )
    bounds: list[ResidencyBound] = []

    if threads:
        bounds.append(
            ResidencyBound(
                "threads",
                threads,
                facts.maximum_threads_per_multiprocessor,
                facts.maximum_threads_per_multiprocessor // threads,
            )
        )

    if compiled_resources is not None:
        registers = compiled_resources.registers_per_thread * threads
        bounds.append(ResidencyBound(
            "registers", registers, facts.registers_per_multiprocessor,
            facts.registers_per_multiprocessor // registers,
        ))
    shared = (
        compiled_resources.shared_bytes if compiled_resources is not None
        else _allocation_bytes(schedule, MemorySpace.SHARED)
    )
    if shared:
        bounds.append(
            ResidencyBound(
                "shared_memory",
                shared,
                facts.shared_memory_per_multiprocessor_bytes,
                facts.shared_memory_per_multiprocessor_bytes // shared,
            )
        )

    # Tensor memory is not shared between resident CTAs on this Target, so an
    # allocation that fills it admits one CTA and nothing else changes that.
    tensor = _allocation_bytes(schedule, MemorySpace.TENSOR)
    if tensor:
        capacity = target.resource_limits.maximum_tensor_memory_bytes
        bounds.append(
            ResidencyBound("tensor_memory", tensor, capacity, capacity // tensor)
        )

    return ResidencyUpperBound(tuple(bounds))


def logical_register_pressure_per_thread(
    schedule: Schedule, target: Target
) -> int | None:
    """Normalize live logical register-Buffer bytes across the CTA threads.

    This is a deterministic structural feature for diagnostics and future calibration,
    not a lower bound or estimate of ptxas's physical register allocation. B200 QSA
    evidence includes both proxy-below-measurement and proxy-above-measurement cases.
    """

    threads = schedule.total_warp_extent * target.warp_size
    if not threads:
        return None
    registers = _logical_register_pressure_bytes(schedule) // REGISTER_BYTES
    return (registers + threads - 1) // threads if registers else 0
