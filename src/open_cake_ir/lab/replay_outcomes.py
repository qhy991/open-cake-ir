"""Derive retained diagnosis and terminal-outcome contracts."""

from __future__ import annotations

from typing import Mapping, Sequence, cast

from open_cake_ir.evaluation import EvaluationReceipt
from open_cake_ir.evidence import RunAudit
from open_cake_ir.serialization import canonical_json_bytes

from ._documents import _object
from .endpoints import endpoint_policy, matched_endpoint
from ._policies import _ATTRIBUTION_EVALUATION
from .checkpoints import TurnObservation, project_checkpoints
from .contracts import CampaignLock
from .ralph import RalphBudget, derive_ralph_stop_reason
from .selection import (
    _collapse_diagnosis,
    _matched_search_decision,
    _matched_search_plan,
)


def _expected_matched_diagnoses_v1(
    *,
    filters: Mapping[int, Mapping[str, object]],
    receipts: Mapping[tuple[int, str, str], EvaluationReceipt],
    receipt_order: list[tuple[int, str, str]],
    searches_per_turn: int,
    materiality_ratio: float,
) -> tuple[dict[int, list[dict[str, object]]], dict[int, list[str]]]:
    """Replay the same search-plan and diagnosis primitives over retained facts."""

    diagnoses: dict[int, list[dict[str, object]]] = {}
    expected_searches: dict[int, list[str]] = {}
    for turn, payload in sorted(filters.items()):
        rows = cast(list[Mapping[str, object]], payload["order"])
        planned, collapsed = _matched_search_plan(rows, searches_per_turn)
        expected_searches[turn] = planned
        projected = [item for item in (_collapse_diagnosis(turn, collapsed),) if item]
        search_keys = [
            key for key in receipt_order if key[0] == turn and key[1] == "search"
        ]
        if [key[2] for key in search_keys] == planned:
            _, _, cost_diagnosis = _matched_search_decision(
                turn,
                [(key[2], receipts[key]) for key in search_keys],
                cost_order_applied=(
                    payload["candidate_selection"]["order_applied"]
                    if "candidate_selection" in payload else all(
                        row["cost"] is not None
                        for row in rows if row["disposition"] == "launchable"
                    )
                ),
                materiality_ratio=materiality_ratio,
            )
            if cost_diagnosis is not None:
                projected.append(cost_diagnosis)
        diagnoses[turn] = projected
    return diagnoses, expected_searches


def _validate_matched_diagnoses_v1(
    events: tuple[Mapping[str, object], ...],
    *,
    expected: Mapping[int, list[dict[str, object]]],
    fault_turn: int | None,
) -> None:
    """Require every retained diagnosis to be the unique derived projection."""

    observed: dict[int, list[Mapping[str, object]]] = {}
    for event in events:
        if event.get("kind") != "diagnosis_routed":
            continue
        payload = _object(event.get("payload"), "diagnosis_routed.payload")
        turn = payload.get("turn")
        if not isinstance(turn, int) or isinstance(turn, bool) or turn <= 0:
            raise ValueError("diagnosis_routed Turn differs")
        observed.setdefault(turn, []).append(payload)
    if set(observed) - set(expected):
        raise ValueError("diagnosis_routed Turn is outside the filtered set")
    for turn, expected_payloads in expected.items():
        actual = observed.get(turn, [])
        admitted = (
            expected_payloads[: len(actual)]
            if turn == fault_turn
            else expected_payloads
        )
        if [dict(payload) for payload in actual] != admitted:
            raise ValueError("diagnosis_routed is not derived from retained evidence")


def _replay_terminal(
    *,
    attribution_evaluation: str | None,
    audit: RunAudit,
    checkpoint_events: Sequence[Mapping[str, object]],
    cumulative_by_turn: Mapping[int, int],
    fault_terminal_tokens: int | None,
    faults: Sequence[Mapping[str, object]],
    lock: CampaignLock,
    observations: Sequence[TurnObservation],
    receipts: Mapping[tuple[int, str, str], EvaluationReceipt],
    searches_per_turn: int,
    invocation_counts: Mapping[str, int] | None = None,
) -> bool:
    if not observations and not faults and endpoint_policy(lock.analysis_plan) is None:
        return False
    if [item.turn for item in observations] != list(range(1, len(observations) + 1)):
        return False
    resolved = _object(lock.document["resolved_inputs"], "resolved_inputs")
    budget = _object(resolved["budget"], "resolved_inputs.budget")
    terminal_tokens = fault_terminal_tokens if fault_terminal_tokens is not None else max(cumulative_by_turn.values(), default=0)
    projected = project_checkpoints(
        turns=observations,
        checkpoints=cast(list[int], budget["checkpoints"]),
        terminal_provider_tokens=terminal_tokens,
    )
    expected_projection = [
        {
            "provider_tokens": item.provider_tokens,
            "state": item.state,
            "best_candidate_sha256": item.best_candidate_sha256,
            "best_confirmed_latency_ms": item.best_confirmed_latency_ms,
        }
        for item in projected
    ]
    checkpoint_payload = _object(
        checkpoint_events[0].get("payload"), "checkpoints_projected.payload"
    )
    expected_checkpoint_fields = (
        {"checkpoints", "ralph"}
    )
    if (
        (set(checkpoint_payload) != expected_checkpoint_fields)
        or checkpoint_payload.get("checkpoints") != expected_projection
    ):
        return False
    ralph_state = _object(
        checkpoint_payload.get("ralph"), "checkpoints_projected.ralph"
    )
    expected_counts = dict(invocation_counts) if invocation_counts is not None else {
        purpose: sum(key[1] == purpose for key in receipts)
        for purpose in ("search", "confirmatory", "attribution")
    }
    state_turn = ralph_state.get("iteration")
    elapsed_wall = ralph_state.get("elapsed_wall_seconds")
    active_authoring = ralph_state.get("active_authoring_seconds")
    if (
        not isinstance(state_turn, int)
        or isinstance(state_turn, bool)
        or not isinstance(elapsed_wall, (int, float))
        or isinstance(elapsed_wall, bool)
        or not isinstance(active_authoring, (int, float))
        or isinstance(active_authoring, bool)
    ):
        return False
    if endpoint_policy(lock.analysis_plan) is not None and state_turn != min(
            budget["maximum_turns"] + 1, len(observations) + 1):
        return False
    expected_stop_reason = audit.protocol_adherence if faults else derive_ralph_stop_reason(
        RalphBudget.from_mapping(budget),
        turn=state_turn,
        cumulative_provider_tokens=terminal_tokens,
        elapsed_wall_seconds=float(elapsed_wall),
        active_authoring_seconds=float(active_authoring),
        evaluation_counts=expected_counts,
        searches_per_turn=searches_per_turn,
        profile_each_search_survivor=(
            attribution_evaluation == _ATTRIBUTION_EVALUATION
        ),
    )
    if (
        ralph_state.get("kind") != "ralph_state_v1"
        or ralph_state.get("cumulative_provider_tokens")
        != terminal_tokens
        or ralph_state.get("remaining", {}).get("provider_tokens") != max(0, budget["limit"] - terminal_tokens)
        or ralph_state.get("evaluation_counts") != expected_counts
        or ralph_state.get("terminal_reason") != expected_stop_reason
    ):
        return False
    expected_observation, expected_endpoint = matched_endpoint(
        checkpoint=projected[-1], observations=observations,
        terminal_provider_tokens=terminal_tokens, protocol_adherence=audit.protocol_adherence,
        terminal_reason=expected_stop_reason, analysis=lock.analysis_plan,
    )
    endpoint_matches = audit.endpoint == expected_endpoint
    if endpoint_policy(lock.analysis_plan) is not None:
        # Exact JSON types as well as fields: observed zero is not False, and a
        # typed terminal token count cannot be substituted by an equal float.
        endpoint_matches = canonical_json_bytes(dict(audit.endpoint) if audit.endpoint is not None else None) == canonical_json_bytes(expected_endpoint)
    return audit.endpoint_observation == expected_observation and endpoint_matches
