"""Run-local checkpoint projection with explicit reachability."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

_DIGEST = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class TurnObservation:
    """One completed search turn; this never confers final confirmation."""

    turn: int
    cumulative_provider_tokens: int
    candidate_sha256: str | None
    search_qualified: bool
    search_latency_ms: float | None

    def __post_init__(self) -> None:
        if (
            self.turn <= 0
            or self.cumulative_provider_tokens <= 0
            or (self.candidate_sha256 is not None and (not isinstance(self.candidate_sha256, str) or _DIGEST.fullmatch(self.candidate_sha256) is None))
            or (self.search_qualified and self.candidate_sha256 is None)
            or not isinstance(self.search_qualified, bool)
            or (
                self.search_qualified
                and (
                    self.search_latency_ms is None
                    or not math.isfinite(self.search_latency_ms)
                    or self.search_latency_ms <= 0
                )
            )
            or (not self.search_qualified and self.search_latency_ms is not None)
        ):
            raise ValueError("Turn observation is invalid")


@dataclass(frozen=True)
class CheckpointObservation:
    """One of the three valid checkpoint states."""

    provider_tokens: int
    state: str
    best_candidate_sha256: str | None
    best_search_latency_ms: float | None


def project_checkpoints(
    *,
    turns: Sequence[TurnObservation],
    checkpoints: Sequence[int],
    terminal_provider_tokens: int,
) -> tuple[CheckpointObservation, ...]:
    """Project reached budgets without future or crossing-turn backfill."""

    if terminal_provider_tokens < 0:
        raise ValueError("terminal provider tokens must be non-negative")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in checkpoints
    ):
        raise ValueError("checkpoint grid must contain positive integers")
    if list(checkpoints) != sorted(set(checkpoints)):
        raise ValueError("checkpoint grid must be strictly increasing")
    previous_tokens = -1
    for expected_turn, turn in enumerate(turns, start=1):
        if turn.turn != expected_turn or turn.cumulative_provider_tokens <= previous_tokens:
            raise ValueError("Turn order or cumulative provider tokens differ")
        if turn.cumulative_provider_tokens > terminal_provider_tokens:
            raise ValueError("Turn exceeds terminal provider-token count")
        previous_tokens = turn.cumulative_provider_tokens

    result: list[CheckpointObservation] = []
    for boundary in checkpoints:
        if terminal_provider_tokens < boundary:
            result.append(CheckpointObservation(boundary, "unreached", None, None))
            continue
        eligible = [turn for turn in turns if turn.cumulative_provider_tokens <= boundary]
        qualified = [turn for turn in eligible if turn.search_qualified]
        if not qualified:
            result.append(
                CheckpointObservation(boundary, "reached_no_search_candidate", None, None)
            )
            continue
        best = min(qualified, key=lambda turn: turn.search_latency_ms or math.inf)
        result.append(
            CheckpointObservation(
                boundary,
                "reached_with_search_candidate",
                best.candidate_sha256,
                best.search_latency_ms,
            )
        )
    return tuple(result)
