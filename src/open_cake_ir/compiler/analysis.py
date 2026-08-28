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
"""

from __future__ import annotations

from dataclasses import dataclass

from .ir import MemorySpace, OperationKind, Schedule
from .target import Target

REGISTER_BYTES = 4


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
    """Peak live bytes named by logical register Buffers after simple aliasing.

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
    storage class. The compiled artifact remains the authority for physical allocation.
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
    peak = 0
    for position in range(total):
        live = 0
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
    schedule: Schedule, target: Target
) -> ResidencyUpperBound | None:
    """Static resident-CTA upper bounds, or None without Target capacity facts."""

    facts = target.occupancy
    if facts is None:
        return None

    threads = schedule.total_warp_extent * target.warp_size
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

    shared = _allocation_bytes(schedule, MemorySpace.SHARED)
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
