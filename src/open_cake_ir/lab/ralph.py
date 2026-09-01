"""Deterministic budget and StateCard control for same-thread Ralph iterations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, cast


@dataclass(frozen=True)
class RalphBudget:
    provider_token_limit: int
    maximum_turns: int
    maximum_candidates_per_turn: int
    wall_time_seconds: float
    active_authoring_time_seconds: float
    search_evaluations: int
    confirmatory_evaluations: int
    attribution_evaluations: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "RalphBudget":
        evaluations = value.get("evaluation_limits")
        if not isinstance(evaluations, Mapping):
            raise ValueError("Ralph evaluation_limits must be an object")

        def positive_int(source: Mapping[str, object], name: str) -> int:
            item = source.get(name)
            if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
                raise ValueError(f"Ralph budget {name} must be a positive integer")
            return item

        def positive_number(name: str) -> float:
            item = value.get(name)
            if (
                not isinstance(item, (int, float))
                or isinstance(item, bool)
                or not math.isfinite(float(item))
                or float(item) <= 0
            ):
                raise ValueError(f"Ralph budget {name} must be positive and finite")
            return float(item)

        budget = cls(
            provider_token_limit=positive_int(value, "limit"),
            maximum_turns=positive_int(value, "maximum_turns"),
            maximum_candidates_per_turn=positive_int(
                value, "maximum_candidates_per_turn"
            ),
            wall_time_seconds=positive_number("wall_time_seconds"),
            active_authoring_time_seconds=positive_number(
                "active_authoring_time_seconds"
            ),
            search_evaluations=positive_int(evaluations, "search"),
            confirmatory_evaluations=positive_int(evaluations, "confirmatory"),
            attribution_evaluations=positive_int(evaluations, "attribution"),
        )
        if budget.active_authoring_time_seconds > budget.wall_time_seconds:
            raise ValueError("Ralph active authoring limit exceeds wall-time limit")
        return budget


def derive_ralph_stop_reason(
    budget: RalphBudget,
    *,
    turn: int,
    cumulative_provider_tokens: int,
    elapsed_wall_seconds: float,
    active_authoring_seconds: float,
    evaluation_counts: Mapping[str, int],
    searches_per_turn: int,
    profile_each_search_survivor: bool,
) -> str | None:
    """Pure terminal decision shared by live control and semantic replay."""

    if cumulative_provider_tokens >= budget.provider_token_limit:
        return "provider_token_limit"
    if turn > budget.maximum_turns:
        return "maximum_turns"
    if elapsed_wall_seconds >= budget.wall_time_seconds:
        return "wall_time_limit"
    if active_authoring_seconds >= budget.active_authoring_time_seconds:
        return "active_authoring_time_limit"
    required = {
        "search": searches_per_turn,
        "confirmatory": 1,
        "attribution": searches_per_turn if profile_each_search_survivor else 0,
    }
    limits = {
        "search": budget.search_evaluations,
        "confirmatory": budget.confirmatory_evaluations,
        "attribution": budget.attribution_evaluations,
    }
    if set(evaluation_counts) != set(limits) or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in evaluation_counts.values()
    ):
        raise ValueError("Ralph Evaluation counts differ")
    if any(
        evaluation_counts[name] + needed > limits[name]
        for name, needed in required.items()
    ):
        return "evaluation_budget"
    return None


class RalphController:
    """Own one Run's time, token, Turn, and Evaluation budgets."""

    def __init__(
        self,
        budget: RalphBudget,
        *,
        searches_per_turn: int,
        profile_each_search_survivor: bool,
        clock: Callable[[], float],
    ) -> None:
        if searches_per_turn <= 0:
            raise ValueError("Ralph searches_per_turn must be positive")
        self.budget = budget
        self.searches_per_turn = searches_per_turn
        self.profile_each_search_survivor = profile_each_search_survivor
        self._clock = clock
        self._started = clock()
        self._active_authoring = 0.0
        self._counts = {"search": 0, "confirmatory": 0, "attribution": 0}

    def _elapsed(self) -> float:
        return max(0.0, self._clock() - self._started)

    def stop_reason(self, *, turn: int, cumulative_provider_tokens: int) -> str | None:
        return derive_ralph_stop_reason(
            self.budget,
            turn=turn,
            cumulative_provider_tokens=cumulative_provider_tokens,
            elapsed_wall_seconds=self._elapsed(),
            active_authoring_seconds=self._active_authoring,
            evaluation_counts=self._counts,
            searches_per_turn=self.searches_per_turn,
            profile_each_search_survivor=self.profile_each_search_survivor,
        )

    def begin_authoring(self) -> float:
        return self._clock()

    def end_authoring(self, started: float) -> None:
        duration = self._clock() - started
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("Ralph authoring duration differs")
        self._active_authoring += duration

    def record_evaluation(self, purpose: str) -> None:
        if purpose not in self._counts:
            raise ValueError("Ralph Evaluation purpose differs")
        limit = cast(int, getattr(self.budget, f"{purpose}_evaluations"))
        if self._counts[purpose] >= limit:
            raise ValueError(f"Ralph {purpose} Evaluation budget exhausted")
        self._counts[purpose] += 1

    def state_card(
        self,
        *,
        turn: int,
        cumulative_provider_tokens: int,
        feedback: Mapping[str, object],
        terminal_reason: str | None = None,
    ) -> Mapping[str, object]:
        elapsed = self._elapsed()
        limits = {
            "search": self.budget.search_evaluations,
            "confirmatory": self.budget.confirmatory_evaluations,
            "attribution": self.budget.attribution_evaluations,
        }
        document = {
            "schema_version": 1,
            "kind": "ralph_state_v1",
            "iteration": turn,
            "cumulative_provider_tokens": cumulative_provider_tokens,
            "elapsed_wall_seconds": round(elapsed, 6),
            "active_authoring_seconds": round(self._active_authoring, 6),
            "evaluation_counts": dict(self._counts),
            "remaining": {
                "provider_tokens": max(
                    0, self.budget.provider_token_limit - cumulative_provider_tokens
                ),
                "turns": max(0, self.budget.maximum_turns - turn + 1),
                "wall_time_seconds": round(
                    max(0.0, self.budget.wall_time_seconds - elapsed), 6
                ),
                "active_authoring_seconds": round(
                    max(
                        0.0,
                        self.budget.active_authoring_time_seconds
                        - self._active_authoring,
                    ),
                    6,
                ),
                "evaluations": {
                    name: max(0, limits[name] - self._counts[name])
                    for name in self._counts
                },
            },
            "previous_feedback": dict(feedback),
            "terminal_reason": terminal_reason,
        }
        return MappingProxyType(document)
