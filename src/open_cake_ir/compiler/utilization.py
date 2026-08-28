"""Turn a measured time into a utilisation, using only counts and declared rates.

`work` counts what a Schedule commits to doing. `Target.peak` declares how fast the
device does it. This is the one place the two meet a measurement, and it is deliberately
the only module in the Compiler that takes a number nothing here derived.

That is the whole design. The analysis has refused to predict a time since it was
written, and this does not start: no quantity here is computed *instead of* running the
kernel. A measured second is an argument, not an output, and what comes back is a
fraction of a declared ceiling that the kernel already reached.

Why it is worth having, when a predicted time was not:

**It is bounded by one, so a single launch can refute it.** Nothing performs arithmetic
faster than the arithmetic peak, and nothing moves its compulsory bytes faster than the
memory system moves any bytes. A utilisation above one is therefore a defect in the work
count or in the declared rate, and it says which by which of the two exceeded. The
refuted wave-count term had no such ceiling; it could only ever be less useful than
hoped, never wrong. `Utilization.refuted` is that check, and a caller that ignores it is
choosing to.

**It abstains loudly.** A Schedule with no contraction has no arithmetic utilisation,
because the only peaks a Target declares are per instruction contract and an rmsnorm
issues none of them. That is not a gap to be filled with an FP32 CUDA-core rate nobody
measured -- an rmsnorm was never trying to do arithmetic, and a number saying it achieved
two percent of the tensor cores describes the metric rather than the kernel.

**It inherits the work model's error directions.** `flops` is exact or a lower bound and
`compulsory_bytes` is exact or an upper bound, so the arithmetic utilisation is a lower
bound and the bandwidth utilisation is an upper bound. `exact` travels with each one; a
ratio printed without it is the estimate this repository keeps refusing to produce.

The roofline time is the same three inputs read the other way round -- how long the
declared work must take at the declared rates. It is a lower bound on the measured time,
which makes it the one term in this file that could order candidates before a GPU. That
is a claim for a calibration to test, not for this module to assert, so nothing here
ranks anything.
"""

from __future__ import annotations

from dataclasses import dataclass

from .target import PeakRate, Target
from .work import WorkBound


@dataclass(frozen=True)
class Utilization:
    """What one measured launch achieved against the rates its Target declares."""

    seconds: float
    arithmetic: float | None
    arithmetic_exact: bool
    arithmetic_contract: str | None
    arithmetic_peak: PeakRate | None
    bandwidth: float | None
    bandwidth_exact: bool
    bandwidth_peak: PeakRate | None
    roofline_seconds: float | None

    @property
    def refuted(self) -> bool:
        """Whether a ratio exceeded one, which no correct pair of inputs can do.

        Read it before quoting either number. It fires on a wrong work count, a peak
        declared below what the device actually does, or a duration that is not this
        kernel's -- and every one of those makes the utilisation beside it meaningless
        rather than merely imprecise.
        """

        return any(
            ratio is not None and ratio > 1 for ratio in (self.arithmetic, self.bandwidth)
        )

    @property
    def roofline_efficiency(self) -> float | None:
        """Declared-work time over measured time: how close the launch came to its bound.

        The larger of the two utilisations, and therefore the one that names the resource
        the kernel got closest to saturating. A value far below one on both means neither
        rate was the limit and the answer is somewhere the declarations cannot see --
        occupancy, latency, or a stall the profiler has a counter for.
        """

        if self.roofline_seconds is None or not self.seconds:
            return None
        return self.roofline_seconds / self.seconds


def roofline_seconds(bound: WorkBound, target: Target) -> float | None:
    """How long the declared work must take at the declared rates, or None without them.

    The one quantity in this repository that is a time and is not a measurement, and it
    is admissible for exactly one reason: it is a *bound*, in the same sense the residency
    figure is. A kernel cannot finish before its arithmetic is issued at the arithmetic
    peak, and it cannot finish before its compulsory bytes have crossed the memory system
    at the bandwidth peak, so the larger of those two is a floor under any measurement.
    A measured time below it refutes an input rather than beating the hardware.

    That is not a predicted time and must not be used as one. The gap between this floor
    and a real launch is everything the declarations cannot see -- occupancy, latency,
    tail effects, the instruction mix around the contraction -- and on the calibrated
    Schedules that gap has been large. `ranking` therefore still does not order on it;
    whether it separates candidates is a question for a measured calibration to answer,
    and until one does, this is a floor and nothing more.

    Terms with no declared rate are omitted rather than treated as zero, so a partial
    peak yields a partial floor rather than a whole one that quietly ignores a resource.
    """

    peak = target.peak
    if peak is None:
        return None
    floors: list[float] = []
    contract = bound.contended_contract
    arithmetic_peak = peak.for_contract(contract) if contract else None
    if arithmetic_peak is not None and bound.flops:
        floors.append(bound.flops / arithmetic_peak.value)
    if peak.memory_bandwidth is not None and bound.compulsory_bytes:
        floors.append(bound.compulsory_bytes / peak.memory_bandwidth.value)
    return max(floors) if floors else None


def utilization(
    bound: WorkBound, target: Target, seconds: float
) -> Utilization | None:
    """Divide declared work by a declared rate and a measured time, or abstain.

    None when the Target declares no peak at all: a Target without one is admissible and
    complete for every other purpose, and inventing a rate to avoid returning None here
    is the failure this whole module is arranged to prevent.

    A present peak that says nothing about *this* Schedule -- no bandwidth, or no rate
    for the contraction it issues -- yields a Utilization with that half None. The
    difference matters: the first means nobody has measured this device, the second means
    nobody has measured this instruction.
    """

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
        # Derived from the same rates rather than from the ratios above, so the floor
        # and the utilisations cannot disagree about which resource binds.
        roofline_seconds=roofline_seconds(bound, target),
    )
