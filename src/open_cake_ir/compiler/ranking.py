"""Rank candidate Schedules before any of them reaches a GPU.

The paper's loop filters a set of structurally distinct candidates with construction checks,
verifier gates and a cost-model ranking, and only then spends GPU time. The gates were here
and the ranking was not, so a set had no order and the stage had nothing to do.

This does not predict a time, and that is a decision rather than an omission. A Target
declares no clock and no bandwidth, so a predicted time would be derived from neither. The
same choice is visible in production: DeepGEMM's own layout comparison ranks on wave count
and last-wave utilisation, and its `num_cycles` field is hardwired to zero behind a TODO.

What it ranks on is what measurement supports, and it declines where measurement withdrew
support. This model used to sort on wave count first -- how many full rounds of resident
CTAs the grid takes -- on the reasoning that a partial final round is a round of dead time.
A sweep of one Schedule's grid across four predicted wave boundaries found latency linear
in CTA count with no step at any of them, and no step at any other wave size either: the
staircase the term describes is not there (`docs/ANALYSIS_CALIBRATION.md`). So the term is
gone, and with it the ability to order a grid that overfills the device -- `cost` returns
None for those rather than ordering them on a refuted basis.

What survives is device fill: among candidates that fit within one round of the device,
prefer the one that fills more of it. That is the whole model, and its support is one
kernel. The nine-tiling Flash-KMeans calibration came out concordant on twenty-nine of
thirty-six pairs; pointing the same instrument at RMSNorm found it beating a blind pick at
one workload scale and losing to one at the next. Read the order as advice from a model
that has been right about one kernel, not as a filter that is known to work.

Ranking is advisory by construction. It orders candidates that have already passed the
gates; it never admits or rejects one. On-device measurement remains the authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .analysis import ResidencyUpperBound, residency_upper_bound
from .ir import Schedule
from .target import Target


@dataclass(frozen=True)
class Cost:
    """What is known about one candidate before it runs.

    Every field is derived from the Schedule and the Target. `device_fill` is the fraction
    of the device's resident capacity the declared grid occupies -- CTAs over CTAs the
    device can hold at once. A Cost exists only where that fraction is at most one, so a
    grid that overfills the device has no Cost rather than a low-confidence one.

    That ceiling is itself optimistic. `ctas_per_multiprocessor` comes from an upper bound,
    so a grid the model believes fits in one round may not: the calibration predicted seven
    resident CTAs where the profiler measured five.
    """

    schedule_id: str
    ctas: int
    ctas_per_multiprocessor: int
    binding_resource: str
    device_fill: float

    @property
    def order(self) -> tuple:
        """Sort key, lower is better.

        A fuller device first, because every candidate here fits within one round and the
        unused capacity is idle multiprocessors. Then the higher residency ceiling, which
        breaks ties toward the candidate with more room to hide latency. `schedule_id` ends
        the key so the order is total and reproducible rather than dependent on input
        order -- a ranking that is not deterministic cannot be evidence.
        """

        return (-self.device_fill, -self.ctas_per_multiprocessor, self.schedule_id)


def _grid_ctas(schedule: Schedule) -> int | None:
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
        total *= (buffer.shape[axis.dimension] + axis.tile - 1) // axis.tile
    return total


def cost(schedule: Schedule, target: Target) -> Cost | None:
    """What is knowable about one candidate, or None when nothing rankable is.

    None means one of two things, and both are the model declining rather than failing:
    the Target does not say enough to derive a residency ceiling, or the declared grid
    overfills the device. The second is the measured limit of this model -- beyond one
    round of the device the only term it had was wave count, and that term is refuted.
    """

    bound: ResidencyUpperBound | None = residency_upper_bound(schedule, target)
    if bound is None or bound.binding is None:
        return None
    resident = bound.ctas_per_multiprocessor or 0
    ctas = _grid_ctas(schedule)
    facts = target.occupancy
    if not resident or ctas is None or facts is None:
        return None

    capacity = resident * facts.multiprocessor_count
    if ctas > capacity:
        return None
    return Cost(
        schedule_id=schedule.schedule_id,
        ctas=ctas,
        ctas_per_multiprocessor=resident,
        binding_resource=bound.binding.resource,
        device_fill=ctas / capacity,
    )


def rank(
    schedules: Iterable[Schedule], target: Target
) -> tuple[tuple[Cost, ...], tuple[str, ...]]:
    """Order candidates best-first, and name the ones nothing could be said about.

    The unranked list is returned rather than dropped. A candidate the model cannot score
    has not been judged inferior, and silently losing it would let a ranking report a
    complete order over an incomplete set. A workload large enough to overfill the device
    puts every candidate there, which is the honest state of this model rather than a bug:
    the term that would have separated them did not survive measurement.
    """

    scored: list[Cost] = []
    unscored: list[str] = []
    for schedule in schedules:
        item = cost(schedule, target)
        if item is None:
            unscored.append(schedule.schedule_id)
        else:
            scored.append(item)
    return tuple(sorted(scored, key=lambda c: c.order)), tuple(unscored)
