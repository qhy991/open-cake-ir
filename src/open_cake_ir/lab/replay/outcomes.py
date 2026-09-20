"""Derive retained diagnosis and terminal-outcome contracts."""

from __future__ import annotations

from typing import Mapping, Sequence, cast

from open_cake_ir.evaluation import EvaluationReceipt
from open_cake_ir.evidence import RunAudit
from open_cake_ir.serialization import canonical_json_bytes

from .._documents import _object
from ..endpoints import endpoint_policy, matched_endpoint
from .._policies import _ATTRIBUTION_EVALUATION
from ..checkpoints import TurnObservation, project_checkpoints
from ..contracts import CampaignLock
from ..ralph import RalphBudget, exceeded_run_budgets
from .refusals import event_location, refuse
from ..selection import (
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
    ordinal = 0
    for event in events:
        if event.get("kind") != "diagnosis_routed":
            continue
        payload = _object(event.get("payload"), "diagnosis_routed.payload")
        turn = payload.get("turn")
        if not isinstance(turn, int) or isinstance(turn, bool) or turn <= 0:
            refuse(event_location("diagnosis_routed", ordinal=ordinal) + ".payload.turn",
                   "not a positive integer", observed=turn)
        ordinal += 1
        observed.setdefault(turn, []).append(payload)
    if set(observed) - set(expected):
        refuse("diagnosis_routed", "a diagnosis in a Turn that was not filtered",
               observed=set(observed) - set(expected), expected=set(expected))
    for turn, expected_payloads in expected.items():
        actual = observed.get(turn, [])
        admitted = (
            expected_payloads[: len(actual)]
            if turn == fault_turn
            else expected_payloads
        )
        if [dict(payload) for payload in actual] != admitted:
            refuse(event_location("diagnosis_routed", turn=turn),
                   "differs from the diagnoses rederived from the retained filter and receipts",
                   observed=[dict(payload) for payload in actual], expected=admitted)


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
    confirmation=None,
    search_state=None,
    compilation_count=0,
) -> None:
    """Refuse unless the terminal, checkpoints and Ralph state rederive from the Run's facts."""
    if not observations and not faults and search_state is None:
        refuse("run_terminal", "a Run with no Turn observation and no fault has no terminal to derive")
    observed_turns = [item.turn for item in observations]
    if observed_turns != list(range(1, len(observations) + 1)):
        refuse("checkpoints_projected", "observed Turns are not 1..n", observed=observed_turns,
               expected=list(range(1, len(observations) + 1)))
    resolved = lock.document
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
            "best_search_latency_ms": item.best_search_latency_ms,
        }
        for item in projected
    ]
    checkpoint_payload = _object(
        checkpoint_events[0].get("payload"), "checkpoints_projected.payload"
    )
    expected_checkpoint_fields = (
        {"checkpoints", "ralph"}
    )
    if set(checkpoint_payload) != expected_checkpoint_fields:
        refuse("checkpoints_projected.payload", "fields differ", observed=set(checkpoint_payload),
               expected=expected_checkpoint_fields)
    if checkpoint_payload.get("checkpoints") != expected_projection:
        refuse("checkpoints_projected.payload.checkpoints",
               "differs from the projection rederived from the Turn observations",
               observed=checkpoint_payload.get("checkpoints"), expected=expected_projection)
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
    if not isinstance(state_turn, int) or isinstance(state_turn, bool):
        refuse("checkpoints_projected.payload.ralph.iteration", "not an integer", observed=state_turn)
    if not isinstance(elapsed_wall, (int, float)) or isinstance(elapsed_wall, bool):
        refuse("checkpoints_projected.payload.ralph.elapsed_wall_seconds", "not a number",
               observed=elapsed_wall)
    if not isinstance(active_authoring, (int, float)) or isinstance(active_authoring, bool):
        refuse("checkpoints_projected.payload.ralph.active_authoring_seconds", "not a number",
               observed=active_authoring)
    if endpoint_policy(lock.terminal_policy) is not None:
        expected_turn = min(budget["maximum_turns"] + 1, len(observations) + 1)
        if state_turn != expected_turn:
            refuse("checkpoints_projected.payload.ralph.iteration",
                   "differs from the Turn after the last observed or budgeted one",
                   observed=state_turn, expected=expected_turn)
    expected_stop_reason = audit.protocol_adherence if faults else (
        search_state['terminal_reason'] if search_state else None)
    if search_state is not None:
        if (elapsed_wall < search_state['elapsed_wall_seconds']
            or active_authoring != search_state['active_authoring_seconds']):
            refuse('checkpoints_projected.payload.ralph', 'terminal time predates search or adds authoring after nomination')
    time_limits = RalphBudget.from_mapping(budget)
    search_elapsed = search_state['elapsed_wall_seconds'] if search_state else elapsed_wall
    confirmation_elapsed = round(elapsed_wall-search_elapsed,6) if search_state else 0.
    remaining = _object(ralph_state.get('remaining'),'checkpoints_projected.payload.ralph.remaining')
    for name,expected in (
        ('wall_time_seconds',round(max(0.,time_limits.wall_time_seconds-elapsed_wall),6)),
        ('search_wall_time_seconds',round(max(0.,time_limits.search_wall_time_seconds-search_elapsed),6)),
        ('confirmation_wall_time_seconds',round(max(0.,time_limits.confirmation_wall_time_seconds-confirmation_elapsed),6)),
    ):
        if type(remaining.get(name)) not in {int,float} or remaining[name] != expected:
            refuse('checkpoints_projected.payload.ralph.remaining.'+name,'phase remainder differs from elapsed time')
    for field, observed, expected in (
        ('budget_exceeded',ralph_state.get('budget_exceeded'),
         list(exceeded_run_budgets(time_limits,search_state=search_state,terminal_state=ralph_state))),
        ("kind", ralph_state.get("kind"), "ralph_state_v1"),
        ("cumulative_provider_tokens", ralph_state.get("cumulative_provider_tokens"), terminal_tokens),
        ("remaining.provider_tokens", ralph_state.get("remaining", {}).get("provider_tokens"),
         max(0, budget["limit"] - terminal_tokens)),
        ("evaluation_counts", ralph_state.get("evaluation_counts"), expected_counts),
        ('compilation_count',ralph_state.get('compilation_count'),compilation_count),
        ('remaining.compilations',ralph_state.get('remaining',{}).get('compilations'),budget['maximum_compilations']-compilation_count),
        ("terminal_reason", ralph_state.get("terminal_reason"), expected_stop_reason),
    ):
        if observed != expected:
            refuse(f"checkpoints_projected.payload.ralph.{field}",
                   "differs from the Ralph state rederived from the Run's facts",
                   observed=observed, expected=expected)
    expected_observation, expected_endpoint = matched_endpoint(
        checkpoint=projected[-1], observations=observations,
        terminal_provider_tokens=terminal_tokens, protocol_adherence=audit.protocol_adherence,
        terminal_reason=expected_stop_reason, analysis=lock.terminal_policy, confirmation=confirmation,
        budget_exceeded=exceeded_run_budgets(time_limits,search_state=search_state,terminal_state=ralph_state),
    )
    if audit.endpoint_observation != expected_observation:
        refuse("run_terminal.payload.endpoint_observation",
               "differs from the observation rederived from the final checkpoint",
               observed=audit.endpoint_observation, expected=expected_observation)
    observed_endpoint = dict(audit.endpoint) if audit.endpoint is not None else None
    endpoint_matches = observed_endpoint == expected_endpoint
    if endpoint_policy(lock.terminal_policy) is not None:
        # Exact JSON types as well as fields: observed zero is not False, and a
        # typed terminal token count cannot be substituted by an equal float.
        endpoint_matches = canonical_json_bytes(observed_endpoint) == canonical_json_bytes(expected_endpoint)
    if not endpoint_matches:
        refuse("run_terminal.payload.endpoint", "differs from the endpoint rederived from the final checkpoint",
               observed=observed_endpoint, expected=expected_endpoint)
