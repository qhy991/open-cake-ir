"""Rank candidate Schedules before any of them reaches a GPU.

The paper's loop filters a set of structurally distinct candidates with construction checks,
verifier gates and a cost-model ranking, and only then spends GPU time. The gates were here
and the ranking was not, so a set had no order and the stage had nothing to do.

This does not predict a time, and that is a decision rather than an omission. A Target
declares no clock and no bandwidth, so a predicted time would be derived from neither. The
same choice is visible in production: DeepGEMM's own layout comparison ranks on wave count
and last-wave utilisation, and its `num_cycles` field is hardwired to zero behind a TODO.

What it ranks on is what measurement supports. Profiling on a B200 found the predicted
*binding resource* correct on both kernels tested, while the magnitudes were loose -- so the
order here is built from wave structure and the residency ceiling, both of which follow from
declarations, and never from an estimated latency (`docs/ANALYSIS_CALIBRATION.md`).

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

    Every field is derived from the Schedule and the Target. `waves` is how many full
    rounds of resident CTAs the declared grid takes, and `last_wave_occupancy` is the
    fraction of a wave the final round fills -- a candidate that needs 2.05 waves wastes
    almost a whole round, which is the classic reason a slightly smaller tile wins.
    """

    schedule_id: str
    ctas: int
    ctas_per_multiprocessor: int
    binding_resource: str
    waves: int
    last_wave_occupancy: float

    @property
    def order(self) -> tuple:
        """Sort key, lower is better.

        Fewer waves first, because a wave is a round of the whole device. Then a fuller
        last wave, because the tail is dead time. Then the higher residency ceiling, which
        breaks ties toward the candidate with more room to hide latency. `schedule_id` ends
        the key so the order is total and reproducible rather than dependent on input
        order -- a ranking that is not deterministic cannot be evidence.
        """

        return (self.waves, -self.last_wave_occupancy, -self.ctas_per_multiprocessor,
                self.schedule_id)


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
    """What is knowable about one candidate, or None when the Target says too little."""

    bound: ResidencyUpperBound | None = residency_upper_bound(schedule, target)
    if bound is None or bound.binding is None:
        return None
    resident = bound.ctas_per_multiprocessor or 0
    ctas = _grid_ctas(schedule)
    facts = target.occupancy
    if not resident or ctas is None or facts is None:
        return None

    per_wave = resident * facts.multiprocessor_count
    waves = (ctas + per_wave - 1) // per_wave
    remainder = ctas - (waves - 1) * per_wave
    return Cost(
        schedule_id=schedule.schedule_id,
        ctas=ctas,
        ctas_per_multiprocessor=resident,
        binding_resource=bound.binding.resource,
        waves=waves,
        last_wave_occupancy=remainder / per_wave,
    )


def rank(
    schedules: Iterable[Schedule], target: Target
) -> tuple[tuple[Cost, ...], tuple[str, ...]]:
    """Order candidates best-first, and name the ones nothing could be said about.

    The unranked list is returned rather than dropped. A candidate the model cannot score
    has not been judged inferior, and silently losing it would let a ranking report a
    complete order over an incomplete set.
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
