"""Normalize declared Schedule work by measured time and target-specific references.

Arithmetic peaks remain instruction-contract specific. The Schedule byte count measures
logical external-buffer work, even when exact; it does not establish DRAM traffic during
a warm-cache launch. By default its bandwidth ratio is only a logical-byte/reference
ratio and may exceed one without refuting a ceiling.

A caller may assert COMPULSORY_DRAM only when its measurement contract independently
establishes that the counted bytes must cross DRAM within the measured interval. Exact
bytes under that scope and specification-backed arithmetic may form roofline floors and
refutations. Microbenchmark rates remain comparative references, never ceilings. Neither
these ratios nor a partial floor predict latency or decide candidate acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..target import PeakRate, PeakSource, Target
from .work import WorkBound


class MemoryScope(str, Enum):
    """The caller's evidence about where counted bytes must move during timing.

    LOGICAL covers warm-cache or unspecified measurements. COMPULSORY_DRAM requires
    independent evidence for the complete counted traffic, including writes; merely
    knowing a buffer's size, or flushing a cache without matching the timed interval,
    does not establish it.
    """

    LOGICAL = "logical"
    COMPULSORY_DRAM = "compulsory_dram"


@dataclass(frozen=True)
class Utilization:
    """Declared work over a measured interval, with the memory interpretation attached."""

    seconds: float
    arithmetic: float | None
    arithmetic_exact: bool
    arithmetic_contract: str | None
    arithmetic_peak: PeakRate | None
    bandwidth: float | None
    bandwidth_exact: bool
    bandwidth_peak: PeakRate | None
    roofline_seconds: float | None
    memory_scope: MemoryScope = MemoryScope.LOGICAL

    @property
    def refuted(self) -> bool:
        """Whether a device-ceiling-backed lower-bound ratio exceeded one.

        Read it before quoting either number. It fires on a wrong work count, a wrong
        device specification, or a duration that is not this kernel's. Microbenchmark
        references, logical-byte ratios and upper-bound byte ratios cannot fire it.
        """

        arithmetic_refuted = (
            self.arithmetic_peak is not None
            and self.arithmetic_peak.source is PeakSource.DEVICE_SPECIFICATION
            and self.arithmetic is not None
            and self.arithmetic > 1
        )
        bandwidth_refuted = (
            self.memory_scope is MemoryScope.COMPULSORY_DRAM
            and self.bandwidth_exact
            and self.bandwidth_peak is not None
            and self.bandwidth_peak.source is PeakSource.DEVICE_SPECIFICATION
            and self.bandwidth is not None
            and self.bandwidth > 1
        )
        return arithmetic_refuted or bandwidth_refuted

    @property
    def roofline_efficiency(self) -> float | None:
        """Declared-work time over measured time: how close the launch came to its bound.

        This equals the larger sound lower-bound ratio. An upper-bound bandwidth ratio is
        omitted from the floor, so it may be numerically larger without becoming this
        efficiency. A value far below one means the answer is somewhere the declarations
        cannot see -- occupancy, latency, or a stall the profiler has a counter for.
        """

        if self.roofline_seconds is None or not self.seconds:
            return None
        return self.roofline_seconds / self.seconds


def roofline_seconds(
    bound: WorkBound, target: Target, *, memory_scope: MemoryScope = MemoryScope.LOGICAL
) -> float | None:
    """How long declared work must take at specification ceilings, or None without them.

    The one quantity in this repository that is a time and is not a measurement, and it
    is admissible for exactly one reason: it is a *bound*, in the same sense the residency
    figure is. A kernel cannot finish before its arithmetic is issued at the arithmetic
    specification ceiling, and it cannot finish before exact compulsory bytes have crossed
    DRAM at its specification ceiling only under COMPULSORY_DRAM scope. The larger
    admitted term is a floor under a measurement that satisfies those premises. A measured
    time below it refutes an input rather than beating the hardware.

    That is not a predicted time and must not be used as one. The gap between this floor
    and a real launch is everything the declarations cannot see -- occupancy, latency,
    tail effects, the instruction mix around the contraction -- and on the calibrated
    Schedules that gap has been large. `ranking` therefore still does not order on it;
    whether it separates candidates is a question for a measured calibration to answer,
    and until one does, this is a floor and nothing more.

    Terms with no device-specification ceiling are omitted rather than treated as zero,
    so a partial ceiling set yields a partial floor. Microbenchmark rates still produce
    comparative utilization ratios but do not enter this bound. Logical bytes never
    enter the memory floor, even when the Schedule byte count is exact.
    """

    memory_scope = MemoryScope(memory_scope)
    peak = target.peak
    if peak is None:
        return None
    floors: list[float] = []
    contract = bound.contended_contract
    arithmetic_peak = peak.for_contract(contract) if contract else None
    if (
        arithmetic_peak is not None
        and arithmetic_peak.source is PeakSource.DEVICE_SPECIFICATION
        and bound.flops
    ):
        floors.append(bound.flops / arithmetic_peak.value)
    if (
        memory_scope is MemoryScope.COMPULSORY_DRAM
        and peak.memory_bandwidth is not None
        and peak.memory_bandwidth.source is PeakSource.DEVICE_SPECIFICATION
        and bound.compulsory_bytes
        and bound.compulsory_bytes_exact
    ):
        floors.append(bound.compulsory_bytes / peak.memory_bandwidth.value)
    return max(floors) if floors else None


def utilization(
    bound: WorkBound, target: Target, seconds: float, *,
    memory_scope: MemoryScope = MemoryScope.LOGICAL,
) -> Utilization | None:
    """Divide declared work by a declared rate and a measured time, or abstain.

    None when the Target declares no peak at all: a Target without one is admissible and
    complete for every other purpose, and inventing a rate to avoid returning None here
    is the failure this whole module is arranged to prevent.

    A present peak that says nothing about *this* Schedule -- no bandwidth, or no rate
    for the contraction it issues -- yields a Utilization with that half None. The
    difference matters: the first means nobody has measured this device, the second means
    nobody has measured this instruction. The memory_scope records whether bandwidth
    is logical-byte normalization or has an independently established DRAM premise;
    bandwidth_exact describes the byte count only, not its physical traffic scope.
    """

    memory_scope = MemoryScope(memory_scope)
    peak = target.peak
    if peak is None or not seconds > 0:
        return None

    contract = bound.contended_contract
    arithmetic_peak = peak.for_contract(contract) if contract else None
    arithmetic = (
        bound.flops / (arithmetic_peak.value * seconds)
        if arithmetic_peak is not None and bound.flops
        else None
    )

    bandwidth_peak = peak.memory_bandwidth
    bandwidth = (
        bound.compulsory_bytes / (bandwidth_peak.value * seconds)
        if bandwidth_peak is not None and bound.compulsory_bytes
        else None
    )

    return Utilization(
        seconds=float(seconds),
        arithmetic=arithmetic,
        arithmetic_exact=bound.flops_exact,
        arithmetic_contract=contract if arithmetic is not None else None,
        arithmetic_peak=arithmetic_peak,
        bandwidth=bandwidth,
        bandwidth_exact=bound.compulsory_bytes_exact,
        bandwidth_peak=bandwidth_peak,
        # Derived from specification ceilings rather than blindly from the ratios above;
        # microbenchmark references and byte upper bounds are deliberately omitted.
        roofline_seconds=roofline_seconds(bound, target, memory_scope=memory_scope),
        memory_scope=memory_scope,
    )
