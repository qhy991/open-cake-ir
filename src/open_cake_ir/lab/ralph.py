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
    maximum_compilations: int
    confirmation_wall_time_seconds: float

    @property
    def search_wall_time_seconds(self):
        return self.wall_time_seconds - self.confirmation_wall_time_seconds

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
            maximum_compilations=positive_int(value, 'maximum_compilations'),
            confirmation_wall_time_seconds=positive_number('confirmation_wall_time_seconds'),
        )
        if budget.confirmation_wall_time_seconds >= budget.wall_time_seconds:
            raise ValueError('confirmation reserve must leave a positive search wall budget')
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
    compilation_count: int,
) -> str | None:
    """Pure terminal decision shared by live control and semantic replay."""

    if cumulative_provider_tokens >= budget.provider_token_limit:
        return "provider_token_limit"
    if turn > budget.maximum_turns:
        return "maximum_turns"
    if elapsed_wall_seconds >= budget.search_wall_time_seconds:
        return "wall_time_limit"
    if active_authoring_seconds >= budget.active_authoring_time_seconds:
        return "active_authoring_time_limit"
    if type(compilation_count) is not int or not 0 <= compilation_count <= budget.maximum_compilations:
        raise ValueError('Run compilation count differs')
    if compilation_count == budget.maximum_compilations:
        return 'compilation_budget'
    required = {
        "search": searches_per_turn,
        "confirmatory": 0,
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


def exceeded_run_budgets(budget, *, search_state, terminal_state):
    """Actual overshoot never counts as success inside the frozen allocation.

    A search Turn and a common Evaluation are atomic for evidence collection.
    Native call timeouts remain with their adapters; these are scheduling and
    acceptance limits, not a claim to preempt an in-flight compiler or GPU job.
    """
    if search_state is None:
        return ()  # A fault before search closure has no normal endpoint.
    search = search_state['elapsed_wall_seconds']
    terminal = terminal_state['elapsed_wall_seconds']
    return tuple(name for name, exceeded in (
        ('provider_tokens',terminal_state['cumulative_provider_tokens'] > budget.provider_token_limit),
        ('active_authoring_time',terminal_state['active_authoring_seconds'] > budget.active_authoring_time_seconds),
        ('search_wall_time',search > budget.search_wall_time_seconds),
        ('confirmation_wall_time',round(terminal-search,6) > budget.confirmation_wall_time_seconds),
    ) if exceeded)


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
        self._compilations = 0
        self._search_closed_at = None

    @property
    def elapsed_wall_seconds(self) -> float:
        return self._elapsed()

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
            compilation_count=self._compilations,
        )

    def begin_authoring(self) -> float:
        if self._search_closed_at is not None:
            raise ValueError('authoring cannot resume after search completion')
        return self._clock()

    def complete_search(self, *, turn, cumulative_provider_tokens, feedback):
        if self._search_closed_at is not None:
            raise ValueError('search completion is unique')
        state = dict(self.state_card(turn=turn,cumulative_provider_tokens=cumulative_provider_tokens,feedback=feedback))
        reason = derive_ralph_stop_reason(self.budget,turn=turn,cumulative_provider_tokens=cumulative_provider_tokens,
            elapsed_wall_seconds=state['elapsed_wall_seconds'],active_authoring_seconds=state['active_authoring_seconds'],
            evaluation_counts=self._counts,compilation_count=self._compilations,
            searches_per_turn=self.searches_per_turn,profile_each_search_survivor=self.profile_each_search_survivor)
        if reason is None:
            raise ValueError('search ended without a declared budget stop')
        state['terminal_reason'] = reason
        self._search_closed_at = state['elapsed_wall_seconds']
        return MappingProxyType(state)

    def end_authoring(self, started: float) -> None:
        duration = self._clock() - started
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("Ralph authoring duration differs")
        self._active_authoring += duration

    def record_evaluation(self, purpose: str) -> None:
        if purpose not in self._counts:
            raise ValueError("Ralph Evaluation purpose differs")
        if (purpose == 'search' and self._search_closed_at is not None
            or purpose == 'confirmatory' and self._search_closed_at is None):
            raise ValueError('Evaluation purpose differs from the Run phase')
        limit = cast(int, getattr(self.budget, f"{purpose}_evaluations"))
        if self._counts[purpose] >= limit:
            raise ValueError(f"Ralph {purpose} Evaluation budget exhausted")
        self._counts[purpose] += 1

    def record_compilation(self):
        from .faults import CompilationBudgetExceeded
        if self._search_closed_at is not None:
            raise ValueError('native compilation cannot resume after search completion')
        if self._compilations >= self.budget.maximum_compilations:
            raise CompilationBudgetExceeded('Run native compilation budget exhausted')
        self._compilations += 1
        return self._compilations

    def state_card(
        self,
        *,
        turn: int,
        cumulative_provider_tokens: int,
        feedback: Mapping[str, object],
        terminal_reason: str | None = None,
    ) -> Mapping[str, object]:
        elapsed = round(self._elapsed(),6)
        search_elapsed = elapsed if self._search_closed_at is None else self._search_closed_at
        confirmation_elapsed = 0.0 if self._search_closed_at is None else round(elapsed-self._search_closed_at,6)
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
            "compilation_count": self._compilations,
            "remaining": {
                "compilations": self.budget.maximum_compilations - self._compilations,
                "provider_tokens": max(
                    0, self.budget.provider_token_limit - cumulative_provider_tokens
                ),
                "turns": max(0, self.budget.maximum_turns - turn + 1),
                "wall_time_seconds": round(
                    max(0.0, self.budget.wall_time_seconds - elapsed), 6
                ),
                'search_wall_time_seconds': round(max(0.0,self.budget.search_wall_time_seconds-search_elapsed),6),
                'confirmation_wall_time_seconds': round(max(0.0,self.budget.confirmation_wall_time_seconds-confirmation_elapsed),6),
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
