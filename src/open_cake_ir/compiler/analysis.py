"""Static residency bounds over the typed IR.

The paper's harness reports a `performance analysis` row as a *report*: a cost estimate
plus bottleneck attribution. This module supplies the attribution half. It does not
estimate time, because a Target declares no clock and no bandwidth, and a number derived
from neither would be invented rather than analysed.

What it derives is an upper bound on residency from declared storage and Target facts.
Threads and explicit allocations are exact Schedule quantities. Register storage is an
optimistic logical lower bound: it assumes same-shaped elementwise values can alias and
does not claim to be ptxas's eventual allocation. Dividing Target capacity by those
per-CTA quantities therefore gives a safe upper bound, not measured occupancy.
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
            "logical_register_storage": "registers",
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
    """Group register buffers that share one physical set of registers.

    An elementwise primitive reads a tile and writes a tile of the same shape, which a
    backend performs in place -- there is no reason to hold both. Charging for each is
    what made composing arithmetic from primitives cost registers the kernel never uses,
    and it charged exactly the schedules that follow the canonical form (P3). A dot or a
    reduction is different: it genuinely needs its operands and its result at once, so
    only the elementwise chain is unioned.
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


def _logical_register_bytes_lower_bound(schedule: Schedule) -> int:
    """Optimistic lower bound on peak live logical register storage.

    Summing them charges a Schedule for every temporary it ever names, which reads the
    same whether two tiles overlap or one is dead before the other is written. That was
    close enough while an operation carried a whole formula and named few intermediates.
    Composing arithmetic from primitives names one buffer per step, and under a sum a
    decomposition that changes no kernel reports as no longer resident.

    A storage class is live from its first write to its last read in declared order.
    Anything the Schedule declares but never writes is charged for the whole program,
    since nothing here can say when it dies. Backend temporaries, allocation granularity,
    and failed aliasing can only increase the physical register allocation; ptxas remains
    the authority for that eventual number.
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

    registers = _logical_register_bytes_lower_bound(schedule) // REGISTER_BYTES
    if registers:
        bounds.append(
            ResidencyBound(
                "logical_register_storage",
                registers,
                facts.registers_per_multiprocessor,
                facts.registers_per_multiprocessor // registers,
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
        if capacity is not None:
            bounds.append(
                ResidencyBound("tensor_memory", tensor, capacity, capacity // tensor)
            )

    return ResidencyUpperBound(tuple(bounds))


def logical_registers_per_thread_lower_bound(
    schedule: Schedule, target: Target
) -> int | None:
    """Optimistic lower bound on registers needed by at least one CTA thread.

    This divides declared logical storage across every CTA thread and rounds up. It is a
    safe gate only when even that optimistic distribution exceeds a declared ``maxnreg``
    cap. It is not an estimate of ptxas's physical allocation.
    """

    threads = schedule.total_warp_extent * target.warp_size
    if not threads:
        return None
    registers = _logical_register_bytes_lower_bound(schedule) // REGISTER_BYTES
    return (registers + threads - 1) // threads if registers else 0
