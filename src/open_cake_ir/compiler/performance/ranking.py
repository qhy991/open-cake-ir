"""Rank candidate Schedules before any of them reaches a GPU.

The paper's loop filters a set of structurally distinct candidates with construction checks,
verifier gates and a cost-model ranking, and only then spends GPU time. The gates were here
and the ranking was not, so a set had no order and the stage had nothing to do.

This does not predict a time, and that is a decision rather than an omission. A Target
declares no clock and no bandwidth, so a predicted time would be derived from neither. The
same choice is visible in production: DeepGEMM's own layout comparison leaves `num_cycles`
hardwired to zero behind a TODO and orders on structure instead.

The structure it orders on is wave count and last-wave utilisation. This model tried that
and measurement refused it; see below. The precedent that survives is the narrow one --
that a serious implementation declines to predict a time -- and not the particular terms
that implementation picked, which are its evidence to produce and not ours to inherit.

What this primitive ranks on is device fill, and it declines where measurement withdrew
support. It used to sort on wave count first -- how many full rounds of resident
CTAs the grid takes -- on the reasoning that a partial final round is a round of dead time.
A sweep of one Schedule's grid across four predicted wave boundaries found latency linear
in CTA count with no step at any of them, and no step at any other wave size either: the
staircase the term describes is not there (`docs/ANALYSIS_CALIBRATION.md`). So the term is
gone, and with it the ability to order a grid that overfills the device -- `cost` returns
None for those rather than ordering them on a refuted basis.

What survives inside this primitive is device fill: among candidates that fit within one
round of the device, prefer the one that fills more of it. Expanded declared-domain B200
calibrations on GEMM and Flash-KMeans did not support that key as a useful profile filter.
Consequently the released Compiler's `calibration_coverage` is empty and its public
`Compiler.rank` withholds every profile. Keeping the primitive separate preserves the
measured hypothesis without exposing it as calibrated product behavior.

When a later Revision has coverage, ranking remains advisory by construction: it orders
candidates that have already passed the gates and never admits or rejects one. On-device
measurement remains the authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .residency import ResidencyUpperBound, residency_upper_bound
from ..ir import Schedule
from ..target import Target


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
    def order(self) -> tuple[float, int]:
        """Performance-semantic preorder key, lower is better.

        A fuller device first, because every candidate here fits within one round and the
        unused capacity is idle multiprocessors. Then the higher residency ceiling, which
        breaks ties toward the candidate with more room to hide latency.

        `schedule_id` is deliberately absent. It can make serialization deterministic,
        but it says nothing about performance and therefore cannot justify putting one
        side of a tied group across a GPU-search cut.
        """

        return (-self.device_fill, -self.ctas_per_multiprocessor)


def rank_for_cut(costs: Sequence[Cost], survivor_count: int) -> tuple[Cost, ...] | None:
    """Return a stable order only when the requested cut does not split a tie.

    The structural model is a preorder, not an oracle. `schedule_id` makes rows within
    each equivalence class reproducible, but a boundary through such a class is a model
    abstention: choosing either tied member would add a performance claim the declarations
    do not contain. The caller must preserve its prior order when this returns None.
    """

    if not isinstance(survivor_count, int) or isinstance(survivor_count, bool):
        raise ValueError("survivor_count must be an integer")
    if survivor_count <= 0 or survivor_count > len(costs):
        raise ValueError("survivor_count must select a non-empty subset of costs")
    ordered = tuple(sorted(costs, key=lambda item: (*item.order, item.schedule_id)))
    if survivor_count < len(ordered):
        if ordered[survivor_count - 1].order == ordered[survivor_count].order:
            return None
    return ordered


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
    # `rank` exposes every row rather than making a selection. A stable spelling is useful
    # for evidence, but only `Cost.order` carries performance meaning; `rank_for_cut`
    # prevents this final serialization key from deciding who reaches a GPU.
    return tuple(sorted(scored, key=lambda c: (*c.order, c.schedule_id))), tuple(unscored)
