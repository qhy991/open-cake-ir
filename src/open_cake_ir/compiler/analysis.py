"""Occupancy analysis over the typed IR.

The paper's harness reports a `performance analysis` row as a *report*: a cost estimate
plus bottleneck attribution. This module supplies the attribution half. It does not
estimate time, because a Target declares no clock and no bandwidth, and a number derived
from neither would be invented rather than analysed.

What it does derive is which declared resource bounds residency. A CTA budget bounds one
CTA; the Target's observed per-multiprocessor facts bound how many CTAs an SM can hold at
once, and the smallest of those bounds is the binding resource. Every input is a
declaration or a device observation recorded in the Target.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ir import MemorySpace, Schedule
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
        return {"threads": "threads", "registers": "registers"}.get(
            self.resource, "bytes"
        )


@dataclass(frozen=True)
class Occupancy:
    bounds: tuple[ResidencyBound, ...]

    @property
    def binding(self) -> ResidencyBound | None:
        """The bound that stops residency being higher."""

        return min(self.bounds, key=lambda b: b.ctas) if self.bounds else None

    @property
    def ctas_per_multiprocessor(self) -> int | None:
        return self.binding.ctas if self.binding else None


def _register_bytes(schedule: Schedule) -> int:
    """Peak concurrently-live register bytes, not the sum of every declared tile.

    Summing them charges a Schedule for every temporary it ever names, which reads the
    same whether two tiles overlap or one is dead before the other is written. That was
    close enough while an operation carried a whole formula and named few intermediates.
    Composing arithmetic from primitives names one buffer per step, and under a sum a
    decomposition that changes no kernel reports as no longer resident.

    A register buffer is live from its first write to its last read in declared order.
    Anything the Schedule declares but never writes is charged for the whole program,
    since nothing here can say when it dies.
    """

    registers = {
        buffer.name: buffer
        for buffer in schedule.buffers
        if buffer.space is MemorySpace.REGISTER
    }
    if not registers:
        return 0

    first_write: dict[str, int] = {}
    last_read: dict[str, int] = {}
    for position, operation in enumerate(schedule.operations):
        for name in operation.writes:
            if name in registers:
                first_write.setdefault(name, position)
        for name in operation.reads:
            if name in registers:
                last_read[name] = position

    total = len(schedule.operations)
    peak = 0
    for position in range(total):
        live = 0
        for name, buffer in registers.items():
            birth = first_write.get(name)
            if birth is None:
                live += buffer.size_bytes
                continue
            death = last_read.get(name, total)
            if birth <= position <= max(death, birth):
                live += buffer.size_bytes
        peak = max(peak, live)
    return peak


def _allocation_bytes(schedule: Schedule, space: MemorySpace) -> int:
    return sum(
        allocation.size_bytes
        for allocation in schedule.allocations
        if allocation.space is space
    )


def occupancy(schedule: Schedule, target: Target) -> Occupancy | None:
    """Residency bounds for one Schedule, or None when the Target declares no facts."""

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

    registers = _register_bytes(schedule) // REGISTER_BYTES
    if registers:
        bounds.append(
            ResidencyBound(
                "registers",
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
        bounds.append(
            ResidencyBound("tensor_memory", tensor, capacity, capacity // tensor)
        )

    return Occupancy(tuple(bounds))


def registers_per_thread(schedule: Schedule, target: Target) -> int | None:
    """Registers each thread holds, from the declared register buffers.

    Reported rather than gated: the per-thread architectural ceiling is not among the
    facts the Target declares, so a violation cannot be asserted from here.
    """

    threads = schedule.total_warp_extent * target.warp_size
    if not threads:
        return None
    registers = _register_bytes(schedule) // REGISTER_BYTES
    return registers // threads if registers else 0
