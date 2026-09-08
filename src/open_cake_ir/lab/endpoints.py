"""Prospective terminal observation policy, separate from token checkpoints."""
from __future__ import annotations

import math
from typing import Mapping, Sequence

from .checkpoints import CheckpointObservation, TurnObservation

NORMAL_BUDGET_TERMINAL = "normal_budget_terminal_v1"
_NORMAL_REASONS = frozenset({"provider_token_limit", "maximum_turns", "wall_time_limit",
                             "active_authoring_time_limit", "evaluation_budget"})


def endpoint_policy(analysis: Mapping[str, object]) -> str | None:
    if "endpoint_policy" not in analysis:
        return None
    if analysis["endpoint_policy"] != NORMAL_BUDGET_TERMINAL:
        raise ValueError("Analysis Plan endpoint policy differs")
    return NORMAL_BUDGET_TERMINAL


def analysis_without_endpoint_policy(analysis: Mapping[str, object]) -> dict[str, object]:
    """Validate the optional closed policy before checking the existing whole plan."""
    endpoint_policy(analysis)
    return {key: value for key, value in analysis.items() if key != "endpoint_policy"}


def matched_endpoint(*, checkpoint: CheckpointObservation, observations: Sequence[TurnObservation],
                     terminal_provider_tokens: int, protocol_adherence: str,
                     terminal_reason: str | None, analysis: Mapping[str, object]):
    from .selection import _matched_endpoint_from_checkpoint

    policy = endpoint_policy(analysis)
    if policy is None:
        return _matched_endpoint_from_checkpoint(checkpoint, protocol_adherence)
    if protocol_adherence != "adhered":
        return "missing", None
    if terminal_reason not in _NORMAL_REASONS:
        raise ValueError("normal-budget endpoint requires a normal Ralph stop")
    if type(terminal_provider_tokens) is not int or terminal_provider_tokens < 0:
        raise ValueError("terminal provider-token observation differs")
    if any(turn.cumulative_provider_tokens > terminal_provider_tokens for turn in observations):
        raise ValueError("terminal observation includes a future Turn")
    # A TurnObservation is appended only after the selected candidate's complete
    # search/confirmation path; a partly completed or faulted turn is not backfilled.
    qualified = [turn for turn in observations if turn.qualified]
    best = min(qualified, key=lambda turn: turn.confirmed_latency_ms or math.inf) if qualified else None
    endpoint = {"qualified_by_budget": best is not None, "budget": terminal_provider_tokens,
                "observation_basis": policy, "terminal_reason": terminal_reason}
    if best is not None:
        endpoint.update(best_candidate_sha256=best.candidate_sha256,
                        best_confirmed_latency_ms=best.confirmed_latency_ms)
    return ("qualified" if best is not None else "no_qualified_candidate"), endpoint
