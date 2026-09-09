"""Pure paired timing derivation from retained raw samples."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence, cast


def _canonical(value: float) -> float:
    return round(float(value), 15)


def summarize_cohort(samples: Sequence[float]) -> dict[str, object]:
    """Derive the canonical cohort summary from finite positive samples."""

    values = [float(value) for value in samples]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("timing samples must be finite and positive")
    median = statistics.median(values)
    mean = statistics.mean(values)
    stddev = statistics.pstdev(values)
    return {
        "sample_count": len(values),
        "median_ms": _canonical(median),
        "mean_ms": _canonical(mean),
        "min_ms": _canonical(min(values)),
        "max_ms": _canonical(max(values)),
        "population_stddev_ms": _canonical(stddev),
        "cv": _canonical(stddev / mean),
    }


@dataclass(frozen=True)
class PairedTimingProtocol:
    """Frozen common timing and decision rules for two launchable arms."""

    arms: tuple[str, str]
    pair_order: tuple[tuple[str, str], ...]
    samples_per_cohort: int
    route_calls_per_cohort: int
    maximum_cv: float
    materiality_ratio: float
    required_pair_wins: int
    # Dispatches encoded in each timed command buffer. One keeps the original
    # regime, where fixed command overhead is charged to every sample.
    dispatches_per_sample: int = 1
    # When set, cohort dispersion is gated on the relative interquartile range of the
    # raw samples instead of their coefficient of variation. Both describe spread; only
    # the first survives the isolated samples a shared GPU produces, and this assay
    # already records that external GPU activity is not excluded. None keeps CV.
    maximum_relative_iqr: float | None = None

    def __post_init__(self) -> None:
        if (
            len(set(self.arms)) != 2
            or not self.pair_order
            or any(set(pair) != set(self.arms) for pair in self.pair_order)
            or self.samples_per_cohort <= 0
            or self.route_calls_per_cohort <= 0
            or not math.isfinite(self.maximum_cv)
            or not 0 <= self.maximum_cv < 1
            or not math.isfinite(self.materiality_ratio)
            or self.materiality_ratio <= 1
            or not 1 <= self.required_pair_wins <= len(self.pair_order)
            or type(self.dispatches_per_sample) is not int
            or not 1 <= self.dispatches_per_sample <= 4096
            or self.maximum_relative_iqr is not None
            and (not isinstance(self.maximum_relative_iqr, (int, float))
                 or isinstance(self.maximum_relative_iqr, bool)
                 or not math.isfinite(self.maximum_relative_iqr)
                 or not 0 < self.maximum_relative_iqr < 1)
        ):
            raise ValueError("paired timing protocol is invalid")


@dataclass(frozen=True)
class PairedTimingObservation:
    """Common measurement-quality and directional observation for one fixed pair."""

    measurement_quality_passed: bool
    pair_wins: Mapping[str, int]
    tied_pairs: int
    pooled_sample_counts: Mapping[str, int]
    pooled_medians_ms: Mapping[str, float]
    speedup: float
    classification: str


def relative_iqr(samples: Sequence[float]) -> float:
    """Interquartile range over the median: the spread of the samples' middle half.

    Uses the same inclusive quartiles and relative form the retained local benchmark
    protocol already applies to its own command-buffer samples.
    """
    values = sorted(float(value) for value in samples)
    if len(values) < 2:
        raise ValueError("relative interquartile range needs at least two samples")
    first, _, third = statistics.quantiles(values, n=4, method="inclusive")
    return _canonical((third - first) / statistics.median(values))


def _object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return cast(Mapping[str, object], value)


def _samples(value: object, expected_count: int, context: str) -> list[float]:
    if not isinstance(value, list) or len(value) != expected_count:
        raise ValueError(f"{context} sample count differs")
    samples: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise ValueError(f"{context} samples must be numbers")
        samples.append(float(item))
    if any(not math.isfinite(item) or item <= 0 for item in samples):
        raise ValueError(f"{context} samples must be finite and positive")
    return samples


def derive_paired_timing(
    measurements: object,
    protocol: PairedTimingProtocol,
) -> PairedTimingObservation:
    """Recompute paired timing solely from raw cohorts and the frozen protocol."""

    if not isinstance(measurements, list) or len(measurements) != len(protocol.pair_order):
        raise ValueError("paired timing cohort count differs")
    first_arm, second_arm = protocol.arms
    pooled: dict[str, list[float]] = {arm: [] for arm in protocol.arms}
    pair_wins = {arm: 0 for arm in protocol.arms}
    measurement_quality_passed = True
    for pair_index, (value, expected_order) in enumerate(
        zip(measurements, protocol.pair_order, strict=True)
    ):
        measurement = _object(value, f"measurements[{pair_index}]")
        if measurement.get("pair_index") != pair_index or measurement.get("order") != list(expected_order):
            raise ValueError(f"measurements[{pair_index}] order differs")
        arms = _object(measurement.get("arms"), f"measurements[{pair_index}].arms")
        if set(arms) != set(protocol.arms):
            raise ValueError(f"measurements[{pair_index}] arm set differs")
        medians: dict[str, float] = {}
        for position, arm in enumerate(expected_order):
            record = _object(arms.get(arm), f"measurements[{pair_index}].arms.{arm}")
            if record.get("position") != position:
                raise ValueError(f"measurements[{pair_index}] arm position differs")
            values = _samples(
                record.get("samples_ms"),
                protocol.samples_per_cohort,
                f"measurements[{pair_index}].arms.{arm}",
            )
            summary = summarize_cohort(values)
            if record.get("summary") != summary:
                raise ValueError(f"measurements[{pair_index}] cohort summary differs")
            if record.get("route_calls") != protocol.route_calls_per_cohort:
                raise ValueError(f"measurements[{pair_index}] route calls differ")
            pooled[arm].extend(values)
            medians[arm] = cast(float, summary["median_ms"])
            if protocol.maximum_relative_iqr is None:
                stable = cast(float, summary["cv"]) <= protocol.maximum_cv
            else:
                stable = relative_iqr(values) <= protocol.maximum_relative_iqr
            measurement_quality_passed = measurement_quality_passed and stable
        if medians[first_arm] < medians[second_arm]:
            pair_wins[first_arm] += 1
        elif medians[second_arm] < medians[first_arm]:
            pair_wins[second_arm] += 1

    pooled_medians = {
        arm: _canonical(statistics.median(samples)) for arm, samples in pooled.items()
    }
    speedup = _canonical(pooled_medians[second_arm] / pooled_medians[first_arm])
    if not measurement_quality_passed:
        classification = "measurement_quality_failed"
    elif pair_wins[first_arm] >= protocol.required_pair_wins and speedup >= protocol.materiality_ratio:
        classification = "first_arm_faster"
    elif (
        pair_wins[second_arm] >= protocol.required_pair_wins
        and speedup <= 1.0 / protocol.materiality_ratio
    ):
        classification = "second_arm_faster"
    else:
        classification = "close_null"
    return PairedTimingObservation(
        measurement_quality_passed=measurement_quality_passed,
        pair_wins=MappingProxyType(dict(pair_wins)),
        tied_pairs=len(protocol.pair_order) - sum(pair_wins.values()),
        pooled_sample_counts=MappingProxyType(
            {arm: len(samples) for arm, samples in pooled.items()}
        ),
        pooled_medians_ms=MappingProxyType(pooled_medians),
        speedup=speedup,
        classification=classification,
    )
