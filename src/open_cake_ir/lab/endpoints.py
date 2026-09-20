"""Prospective terminal observation policy, separate from token checkpoints."""
from __future__ import annotations

from typing import Mapping, Sequence

from .checkpoints import CheckpointObservation, TurnObservation

NORMAL_BUDGET_TERMINAL = "normal_budget_terminal_v1"
_NORMAL_REASONS = frozenset({"provider_token_limit", "maximum_turns", "wall_time_limit",
                             "active_authoring_time_limit", "evaluation_budget", "compilation_budget"})


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
                     terminal_reason: str | None, analysis: Mapping[str, object], confirmation=None,
                     budget_exceeded=()):
    """Only the single terminal confirmation can qualify an endpoint."""
    policy = endpoint_policy(analysis)
    if protocol_adherence != 'adhered':
        return 'missing', None
    if terminal_reason not in _NORMAL_REASONS:
        raise ValueError('endpoint requires a normal Ralph search stop')
    if type(terminal_provider_tokens) is not int or terminal_provider_tokens < 0:
        raise ValueError('terminal provider-token observation differs')
    if any(turn.cumulative_provider_tokens > terminal_provider_tokens for turn in observations):
        raise ValueError('terminal observation includes a future Turn')
    if policy is None and checkpoint.state == 'unreached':
        return 'missing', None
    if confirmation is not None:
        source = next((row for row in observations if row.turn == confirmation.source_turn), None)
        if (source is None or not source.search_qualified
            or source.candidate_sha256 != confirmation.candidate_sha256
            or confirmation.provider_tokens != terminal_provider_tokens):
            raise ValueError('terminal confirmation differs from the nominated search origin')
    qualified = (confirmation is not None and confirmation.qualified
                 and terminal_provider_tokens <= checkpoint.provider_tokens and not budget_exceeded)
    endpoint = {'qualified_by_budget': qualified,
                'budget_exceeded':list(budget_exceeded),
                'budget': terminal_provider_tokens if policy else checkpoint.provider_tokens}
    if policy:
        endpoint.update(observation_basis=policy, terminal_reason=terminal_reason)
    if qualified:
        endpoint.update(best_candidate_sha256=confirmation.candidate_sha256,
                        best_confirmed_latency_ms=confirmation.confirmed_latency_ms)
    return ('qualified' if qualified else 'no_qualified_candidate'), endpoint
