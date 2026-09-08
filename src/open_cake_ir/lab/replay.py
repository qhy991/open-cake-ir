"""Independent reconstruction of Campaign outcomes from raw retained evidence."""

from __future__ import annotations

import json
import math
import re
from hashlib import sha256
from pathlib import Path
from typing import Callable, Mapping, Sequence, cast

from open_cake_ir.evaluation import EvaluationReceipt, LaunchableCandidate
from open_cake_ir.evidence import EvidenceStore, RunAudit

from ._documents import _DIGEST, _canonical_json_bytes, _object
from ._policies import (
    _ATTRIBUTION_EVALUATION,
    _LEGACY_ATTRIBUTION_EVALUATION,
    _MATCHED_EVENT_KINDS_V1,
    _matched_evidence_policy_version,
)
from .archive import (
    _replay_evaluation_attempt_event,
    _replay_evaluation_receipt,
    _replay_launchable_candidate,
)
from .checkpoints import TurnObservation, project_checkpoints
from .contracts import CampaignLock
from .executor import ExecutorRevision
from .pairing import comparison_arm
from .providers import (
    CANDIDATE_SET_ENVELOPE_V1,
    _project_candidate_submission,
    parse_codex_turn_events,
)
from .ralph import RalphBudget, derive_ralph_stop_reason
from .routing import route_rejection
from .selection import (
    _EmpiricalSelection,
    _collapse_diagnosis,
    _empirical_context,
    _empirical_filter,
    _matched_endpoint_from_checkpoint,
    _matched_search_decision,
    _matched_search_plan,
    _receipt_latency_ms,
    _receipt_qualifies,
)
from .task_package import TASK_AGENTS_RALPH_V1, TaskPackage


def _artifact_outcomes_are_closed(payload: Mapping[str, object]) -> bool:
    """Validate the optional retained/rejected artifact-role partition."""

    objects = payload.get("objects")
    rejected = payload.get("artifact_rejections")
    retained_roles: list[object] = []
    if objects is not None:
        if not isinstance(objects, list) or not objects:
            return False
        retained_roles = [
            item.get("role") if isinstance(item, Mapping) else None
            for item in objects
        ]
    if rejected is not None and (not isinstance(rejected, list) or not rejected):
        return False
    rejected_roles = rejected if isinstance(rejected, list) else []
    roles = [*retained_roles, *rejected_roles]
    if not all(
        isinstance(role, str)
        and re.fullmatch(r"[a-z][a-z0-9_]*", role) is not None
        for role in roles
    ):
        return False
    return len(roles) == len(set(roles))


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


def replay_matched_run(
    evidence: EvidenceStore,
    audit: RunAudit,
    lock: CampaignLock,
    *,
    project_root: Path,
    manifest_parser: Callable,
    task_package: Callable,
) -> bool:
    events = evidence.replay_events(audit.run_id)
    resolved_inputs = _object(
        lock.document["resolved_inputs"], "resolved_inputs"
    )
    event_vocabulary = _matched_evidence_policy_version(
        _object(
            resolved_inputs["evidence_policy"],
            "resolved_inputs.evidence_policy",
        ),
        "resolved_inputs.evidence_policy",
    )
    arm = audit.run_id.rsplit("-", 1)[0]
    kinds = [event.get("kind") for event in events]
    if (
        any(kind not in _MATCHED_EVENT_KINDS_V1 for kind in kinds)
        or kinds.count("run_started") != 1
        or kinds.count("checkpoints_projected") != 1
        or kinds.count("run_terminal") != 1
        or kinds[0] != "run_started"
        or kinds[-2:] != ["checkpoints_projected", "run_terminal"]
        or audit.run_id not in lock.run_order
    ):
        return False
    start_payload = _object(
        events[0].get("payload"), "run_started.payload"
    )
    if start_payload != {
        "sequence": lock.run_order.index(audit.run_id) + 1,
        "assigned_arm": arm,
        "automatic_retries": 0,
        "replacement_run": False,
    }:
        return False
    terminal_payload = _object(
        events[-1].get("payload"), "run_terminal.payload"
    )
    if terminal_payload != {
        "protocol_adherence": audit.protocol_adherence,
        "endpoint_observation": audit.endpoint_observation,
        "endpoint": dict(audit.endpoint) if audit.endpoint is not None else None,
    }:
        return False
    turn_events = [
        _object(event.get("payload"), f"event.{event.get('kind')}.payload")[
            "turn"
        ]
        for event in events[1:-2]
        if "turn"
        in _object(
            event.get("payload"), f"event.{event.get('kind')}.payload"
        )
    ]
    if (
        any(
            not isinstance(turn, int)
            or isinstance(turn, bool)
            or turn <= 0
            for turn in turn_events
        )
        or turn_events != sorted(turn_events)
    ):
        return False
    provider_events = [event for event in events if event.get("kind") == "provider_turn_completed"]
    checkpoint_events = [event for event in events if event.get("kind") == "checkpoints_projected"]
    if len(checkpoint_events) != 1:
        return False
    if not provider_events:
        return _replay_provider_fault(
            audit=audit,
            checkpoint_events=checkpoint_events,
            events=events,
            lock=lock,
        )
    replay_budget = _object(resolved_inputs["budget"], "resolved_inputs.budget")
    maximum_candidates_per_turn = int(
        replay_budget.get("maximum_candidates_per_turn", 1)
    )
    arm_environments = _object(
        resolved_inputs["arm_environments"], "resolved_inputs.arm_environments"
    )
    empirical_selection = None
    selection_binding = arm_environments[arm].get("candidate_selection")
    if selection_binding is not None:
        executor = ExecutorRevision.load_reference(project_root, lock.document["execution"]["executor_revision"], "execution.executor_revision")
        compiler_ref = lock.document["compiler_revision"]
        empirical_selection = _EmpiricalSelection(
            selection_binding,
            context=_empirical_context(
                executor, workload_sha256=lock.document["workload"]["canonical_sha256"],
                case_id=lock.document["evaluation_protocol"]["case_id"],
            ),
            compiler_revision_id=compiler_ref["revision_id"],
            compiler_revision_sha256=compiler_ref["canonical_sha256"],
            target=lock.document["execution"]["target"],
        )
    provider_authority = _object(
        _object(arm_environments[arm], f"arm_environments.{arm}")["provider"],
        f"arm_environments.{arm}.provider",
    )
    event_contract = str(
        provider_authority.get("event_contract", "closed_file_change_v1")
    )
    expected_task_package = (
        task_package(lock, audit.run_id)
    )
    provider = _replay_provider_turns(
        arm=arm,
        audit=audit,
        event_contract=event_contract,
        evidence=evidence,
        expected_task_package=expected_task_package,
        maximum_candidates_per_turn=maximum_candidates_per_turn,
        provider_authority=provider_authority,
        provider_events=provider_events,
    )
    if provider is None:
        return False
    cumulative_by_turn, provider_candidates_by_turn, provider_candidate_bytes, candidate_set_turns, prior_cumulative = provider
    workload_sha256 = str(_object(lock.document["workload"], "workload")["canonical_sha256"])
    protocol_sha256 = sha256(
        _canonical_json_bytes(lock.document["evaluation_protocol"])
    ).hexdigest()
    case_id = str(_object(lock.document["evaluation_protocol"], "protocol")["case_id"])
    faults = [event for event in events if event.get("kind") == "run_fault"]
    if len(faults) > 1:
        return False
    fault_turn: int | None = None
    fault_terminal_tokens: int | None = None
    if faults:
        fault_payload = _object(faults[0].get("payload"), "run_fault.payload")
        required_fault_fields = {
            "fault",
            "exception_type",
            "turn",
            "stage",
            "terminal_provider_tokens",
        }
        if (
            not required_fault_fields <= set(fault_payload)
            or set(fault_payload)
            - required_fault_fields
            - {"objects", "artifact_rejections"}
            or not isinstance(fault_payload.get("exception_type"), str)
            or not fault_payload.get("exception_type")
            or not _artifact_outcomes_are_closed(fault_payload)
        ):
            return False
        if fault_payload.get("fault") != audit.protocol_adherence:
            return False
        fault_turn_value = fault_payload.get("turn")
        fault_stage = fault_payload.get("stage")
        fault_terminal_value = fault_payload.get("terminal_provider_tokens")
        if (
            not isinstance(fault_turn_value, int)
            or isinstance(fault_turn_value, bool)
            or fault_turn_value <= 0
            or fault_stage not in {"provider", "environment", "evaluation"}
            or not isinstance(fault_terminal_value, int)
            or isinstance(fault_terminal_value, bool)
            or fault_terminal_value < prior_cumulative
            or (
                fault_stage == "provider"
                and fault_turn_value != len(provider_events) + 1
            )
            or (
                fault_stage in {"environment", "evaluation"}
                and (
                    fault_turn_value != len(provider_events)
                    or fault_terminal_value != prior_cumulative
                )
            )
        ):
            return False
        fault_turn = fault_turn_value
        fault_terminal_tokens = fault_terminal_value

    candidates = _replay_candidates(
        arm=arm,
        case_id=case_id,
        events=events,
        evidence=evidence,
        fault_turn=fault_turn,
        faults=faults,
        lock=lock,
        manifest_parser=manifest_parser,
        protocol_sha256=protocol_sha256,
        provider_candidates_by_turn=provider_candidates_by_turn,
        workload_sha256=workload_sha256,
    )
    if candidates is None:
        return False
    launchables, receipts, receipt_order, rejected = candidates
    selected = _replay_candidate_selection(
        candidate_set_turns=candidate_set_turns,
        cumulative_by_turn=cumulative_by_turn,
        empirical_selection=empirical_selection,
        events=events,
        fault_turn=fault_turn,
        launchables=launchables,
        lock=lock,
        provider_candidate_bytes=provider_candidate_bytes,
        provider_candidates_by_turn=provider_candidates_by_turn,
        receipt_order=receipt_order,
        receipts=receipts,
        rejected=rejected,
    )
    if selected is None:
        return False
    observations, searches_per_turn, attribution_evaluation = selected
    return _replay_terminal(
        attribution_evaluation=attribution_evaluation,
        audit=audit,
        checkpoint_events=checkpoint_events,
        cumulative_by_turn=cumulative_by_turn,
        fault_terminal_tokens=fault_terminal_tokens,
        faults=faults,
        lock=lock,
        observations=observations,
        receipts=receipts,
        searches_per_turn=searches_per_turn,
    )


def _replay_provider_fault(
    *,
    audit: RunAudit,
    checkpoint_events: Sequence[Mapping[str, object]],
    events: Sequence[Mapping[str, object]],
    lock: CampaignLock,
) -> bool:
    faults = [event for event in events if event.get("kind") == "run_fault"]
    if len(faults) != 1:
        return False
    fault_payload = _object(faults[0].get("payload"), "run_fault.payload")
    checkpoint_payload = _object(
        checkpoint_events[0].get("payload"), "checkpoints_projected.payload"
    )
    checkpoints = checkpoint_payload.get("checkpoints")
    required_fault_fields = {
        "fault",
        "exception_type",
        "turn",
        "stage",
        "terminal_provider_tokens",
    }
    if (
        [event.get("kind") for event in events]
        != [
            "run_started",
            "run_fault",
            "checkpoints_projected",
            "run_terminal",
        ]
        or not required_fault_fields <= set(fault_payload)
        or set(fault_payload)
        - required_fault_fields
        - {"objects", "artifact_rejections"}
        or not isinstance(fault_payload.get("exception_type"), str)
        or not fault_payload.get("exception_type")
        or not _artifact_outcomes_are_closed(fault_payload)
        or set(checkpoint_payload)
        != ({"checkpoints", "ralph"})
    ):
        return False
    terminal_tokens = fault_payload.get("terminal_provider_tokens")
    if (
        set(("turn", "stage", "terminal_provider_tokens"))
        - set(fault_payload)
        or fault_payload.get("turn") != 1
        or fault_payload.get("stage") != "provider"
        or not isinstance(terminal_tokens, int)
        or isinstance(terminal_tokens, bool)
        or terminal_tokens < 0
    ):
        return False
    replay_budget = _object(
        _object(lock.document["resolved_inputs"], "resolved_inputs")[
            "budget"
        ],
        "resolved_inputs.budget",
    )
    expected_checkpoints = [
        {
            "provider_tokens": item.provider_tokens,
            "state": item.state,
            "best_candidate_sha256": item.best_candidate_sha256,
            "best_confirmed_latency_ms": item.best_confirmed_latency_ms,
        }
        for item in project_checkpoints(
            turns=(),
            checkpoints=cast(list[int], replay_budget["checkpoints"]),
            terminal_provider_tokens=terminal_tokens,
        )
    ]
    return (
        fault_payload.get("fault") == audit.protocol_adherence
        and audit.endpoint_observation == "missing"
        and audit.endpoint is None
        and isinstance(checkpoints, list)
        and bool(checkpoints)
        and checkpoints == expected_checkpoints
    )


def _replay_provider_turns(
    *,
    arm: str,
    audit: RunAudit,
    event_contract: str,
    evidence: EvidenceStore,
    expected_task_package: TaskPackage,
    maximum_candidates_per_turn: int,
    provider_authority: Mapping[str, object],
    provider_events: Sequence[Mapping[str, object]],
) -> tuple[
    dict[int, int],
    dict[int, tuple[str, ...]],
    dict[tuple[int, str], bytes],
    set[int],
    int,
] | None:
    threads: set[str] = set()
    cumulative_by_turn: dict[int, int] = {}
    provider_candidates_by_turn: dict[int, tuple[str, ...]] = {}
    provider_candidate_bytes: dict[tuple[int, str], bytes] = {}
    candidate_set_turns: set[int] = set()
    prior_cumulative = 0
    for expected_turn, event in enumerate(provider_events, start=1):
        payload = _object(event.get("payload"), "provider_turn.payload")
        expected_provider_fields = {
            "turn",
            "thread_id",
            "turn_provider_tokens",
            "cumulative_provider_tokens",
            "normalization",
            "candidate_count",
            "objects",
        }
        if "event_contract" in provider_authority:
            expected_provider_fields.add("auxiliary_activity")
        if set(payload) != expected_provider_fields:
            return None
        if payload.get("turn") != expected_turn:
            return None
        thread_id = payload.get("thread_id")
        cumulative = payload.get("cumulative_provider_tokens")
        if (
            not isinstance(thread_id, str)
            or not isinstance(cumulative, int)
            or isinstance(cumulative, bool)
            or cumulative <= 0
        ):
            return None
        objects = payload.get("objects")
        if not isinstance(objects, list):
            return None
        event_references = [
            cast(Mapping[str, object], item)
            for item in objects
            if isinstance(item, Mapping) and item.get("role") == "provider_events"
        ]
        reference_bundle_references = [
            cast(Mapping[str, object], item)
            for item in objects
            if isinstance(item, Mapping)
            and item.get("role") == "provider_reference_bundle"
        ]
        submission_references = [
            cast(Mapping[str, object], item)
            for item in objects
            if isinstance(item, Mapping)
            and item.get("role") == "provider_submission_envelope"
        ]
        candidate_count = payload.get("candidate_count")
        candidate_set_turns.add(expected_turn)
        indexed_references: dict[int, Mapping[str, object]] = {}
        for item in objects:
            if not isinstance(item, Mapping):
                continue
            role = item.get("role")
            match = (
                re.fullmatch(r"candidate_submission_(\d{4})", role)
                if isinstance(role, str)
                else None
            )
            if match is not None:
                index = int(match.group(1))
                if index in indexed_references:
                    return None
                indexed_references[index] = item
        candidate_references = [
            indexed_references[index]
            for index in range(len(indexed_references))
        ]
        if (
            len(event_references) != 1
            or len(submission_references) != 1
            or len(reference_bundle_references) > 1
            or (
                len(reference_bundle_references) != 1
            )
            or not isinstance(candidate_count, int)
            or isinstance(candidate_count, bool)
            or candidate_count <= 0
            or candidate_count > maximum_candidates_per_turn
            or len(candidate_references) != candidate_count
            or len(objects)
            != candidate_count + 2 + len(reference_bundle_references)
        ):
            return None
        raw_events = evidence.read_object(event_references[0])
        reference_bundle = (
            evidence.read_object(reference_bundle_references[0])
            if reference_bundle_references
            else None
        )
        candidates = tuple(
            evidence.read_object(reference) for reference in candidate_references
        )
        try:
            projected_candidates = _project_candidate_submission(
                evidence.read_object(submission_references[0]),
                submission_contract=CANDIDATE_SET_ENVELOPE_V1,
                arm=arm,
                maximum_candidates_per_turn=maximum_candidates_per_turn,
            )
        except (UnicodeError, ValueError):
            return None
        if projected_candidates != candidates:
            return None
        candidate_digests = tuple(sha256(candidate).hexdigest() for candidate in candidates)
        if (
            sha256(raw_events).hexdigest() != event_references[0].get("sha256")
            or (
                reference_bundle is not None
                and (
                    not reference_bundle
                    or sha256(reference_bundle).hexdigest()
                    != reference_bundle_references[0].get("sha256")
                )
            )
            or any(
                digest != reference.get("sha256")
                for digest, reference in zip(candidate_digests, candidate_references)
            )
            or len(set(candidate_digests)) != len(candidate_digests)
        ):
            return None
        if reference_bundle is not None:
            try:
                decoded_reference = reference_bundle.decode("utf-8")
                bundle = _object(
                    json.loads(decoded_reference),
                    "provider Ralph task package",
                )
                state = _object(
                    bundle.get("state_card"),
                    "provider Ralph StateCard",
                )
                if (
                    set(bundle)
                    != {
                        "schema_version",
                        "kind",
                        "run_id",
                        "arm",
                        "task_markdown",
                        "agents_markdown",
                        "state_card",
                    }
                    or bundle.get("schema_version") != 1
                    or bundle.get("kind") != TASK_AGENTS_RALPH_V1
                    or bundle.get("run_id") != audit.run_id
                    or bundle.get("arm") != arm
                    or bundle.get("task_markdown")
                    != expected_task_package.task_markdown
                    or bundle.get("agents_markdown")
                    != expected_task_package.agents_markdown
                    or state.get("kind") != "ralph_state_v1"
                    or state.get("iteration") != expected_turn
                    or state.get("cumulative_provider_tokens")
                    != prior_cumulative
                    or state.get("terminal_reason") is not None
                ):
                    return None
            except (UnicodeError, json.JSONDecodeError, ValueError):
                return None
        terminal_document: dict[str, object] = {
            "arm": arm,
            "candidate_written": True,
            "kind": "open_cake_ir_turn",
            "turn": expected_turn,
        }
        if event_contract == "closed_file_change_v1":
            terminal_document["tool_calls"] = 1
        expected_terminal = json.dumps(
            terminal_document,
            sort_keys=True,
            separators=(",", ":"),
        )
        expected_change = "add" if expected_turn == 1 else "update"
        expected_name = (
            "candidate-set.json"
        )
        parsed = parse_codex_turn_events(
            raw_events,
            expected_terminal_message=expected_terminal,
            event_contract=event_contract,
        )
        turn_tokens = payload.get("turn_provider_tokens")
        if (
            parsed.thread_id != thread_id
            or turn_tokens != parsed.provider_tokens
            or cumulative != prior_cumulative + turn_tokens
        ):
            return None
        if parsed.candidate_path is not None and (
            parsed.change_kind != expected_change
            or Path(parsed.candidate_path).name != expected_name
        ):
            return None
        if parsed.candidate_path is None and event_contract != "tool_rich_candidate_v1":
            return None
        if payload.get("normalization") != parsed.normalization:
            return None
        if "event_contract" in provider_authority and payload.get(
            "auxiliary_activity"
        ) != [dict(activity.document) for activity in parsed.tool_activity]:
            return None
        threads.add(thread_id)
        cumulative_by_turn[expected_turn] = cumulative
        provider_candidates_by_turn[expected_turn] = candidate_digests
        provider_candidate_bytes.update(
            ((expected_turn, digest), payload) for digest, payload in zip(candidate_digests, candidates)
        )
        prior_cumulative = cumulative
    if len(threads) != 1 or list(cumulative_by_turn.values()) != sorted(
        cumulative_by_turn.values()
    ):
        return None
    return cumulative_by_turn, provider_candidates_by_turn, provider_candidate_bytes, candidate_set_turns, prior_cumulative


def _replay_candidates(
    *,
    arm: str,
    case_id: str,
    events: Sequence[Mapping[str, object]],
    evidence: EvidenceStore,
    fault_turn: int | None,
    faults: Sequence[Mapping[str, object]],
    lock: CampaignLock,
    manifest_parser: Callable,
    protocol_sha256: str,
    provider_candidates_by_turn: Mapping[int, tuple[str, ...]],
    workload_sha256: str,
) -> tuple[
    dict[tuple[int, str], LaunchableCandidate],
    dict[tuple[int, str, str], EvaluationReceipt],
    list[tuple[int, str, str]],
    dict[tuple[int, str], Mapping[str, object]],
] | None:
    attempt_events = [
        event for event in events if event.get("kind") == "evaluation_attempt_completed"
    ]
    attempt_payloads: dict[tuple[int, str, str], Mapping[str, object]] = {}
    for event in attempt_events:
        payload = _object(
            event.get("payload"), "evaluation_attempt_completed.payload"
        )
        turn = payload.get("turn")
        purpose = payload.get("purpose")
        candidate_sha256 = payload.get("candidate_sha256")
        if (
            set(payload) != {"turn", "purpose", "candidate_sha256", "objects"}
            or not isinstance(turn, int)
            or isinstance(turn, bool)
            or turn <= 0
            or purpose not in {"search", "confirmatory", "attribution"}
            or not isinstance(candidate_sha256, str)
            or _DIGEST.fullmatch(candidate_sha256) is None
            or not isinstance(payload.get("objects"), list)
        ):
            return None
        key = (turn, cast(str, purpose), candidate_sha256)
        if key in attempt_payloads:
            return None
        attempt_payloads[key] = payload
    replayed_attempts: set[tuple[int, str, str]] = set()
    launchable_events = [
        event for event in events if event.get("kind") == "launchable_candidate_sealed"
    ]
    launchables: dict[tuple[int, str], LaunchableCandidate] = {}
    for event in launchable_events:
        payload = _object(event.get("payload"), "launchable.payload")
        turn = payload.get("turn")
        candidate_sha256 = payload.get("candidate_sha256")
        if (
            not isinstance(turn, int)
            or isinstance(turn, bool)
            or turn <= 0
            or not isinstance(candidate_sha256, str)
            or _DIGEST.fullmatch(candidate_sha256) is None
        ):
            return None
        key = (turn, candidate_sha256)
        if key in launchables:
            return None
        launchables[key] = _replay_launchable_candidate(
            evidence,
            launchable_events,
            turn=turn,
            candidate_sha256=candidate_sha256,
            arm=arm,
            manifest_parser=manifest_parser,
        )

    receipts: dict[tuple[int, str, str], EvaluationReceipt] = {}
    receipt_order: list[tuple[int, str, str]] = []
    rejected: dict[tuple[int, str], Mapping[str, object]] = {}
    for event in events:
        kind = event.get("kind")
        payload = _object(event.get("payload"), f"event.{kind}.payload")
        if kind == "candidate_rejected":
            turn = payload.get("turn")
            candidate_sha256 = payload.get("candidate_sha256")
            feedback = payload.get("feedback")
            required_rejection_fields = {
                "turn",
                "candidate_sha256",
                "feedback",
                "routed_to",
                "routing_reason",
            }
            if (
                not required_rejection_fields <= set(payload)
                or set(payload)
                - required_rejection_fields
                - {"objects", "artifact_rejections"}
                or not isinstance(feedback, Mapping)
                or not _artifact_outcomes_are_closed(payload)
            ):
                return None
            decision = route_rejection(feedback)
            if (
                payload.get("routed_to") != decision.destination
                or payload.get("routing_reason") != decision.reason
            ):
                return None
            if (
                not isinstance(turn, int)
                or isinstance(turn, bool)
                or not isinstance(candidate_sha256, str)
                or _DIGEST.fullmatch(candidate_sha256) is None
                or (turn, candidate_sha256) in rejected
            ):
                return None
            rejected[(turn, candidate_sha256)] = payload
        elif kind == "candidate_evaluated":
            turn = payload.get("turn")
            purpose = payload.get("purpose")
            candidate_sha256 = payload.get("candidate_sha256")
            if (
                (set(payload) != ({
                    "turn", "purpose", "candidate_sha256", "objects",
                } | ({"elapsed_wall_seconds"} if purpose == "confirmatory" and
                     comparison_arm(lock.document["resolved_inputs"]["arm_environments"]) == "native_triton" else set())))
                or
                not isinstance(turn, int)
                or isinstance(turn, bool)
                or purpose not in {"search", "confirmatory", "attribution"}
                or not isinstance(candidate_sha256, str)
                or _DIGEST.fullmatch(candidate_sha256) is None
            ):
                return None
            if "elapsed_wall_seconds" in payload and (
                type(payload["elapsed_wall_seconds"]) not in {int, float}
                or not math.isfinite(payload["elapsed_wall_seconds"])
                or payload["elapsed_wall_seconds"] < 0
            ):
                return None
            launchable = launchables.get((turn, candidate_sha256))
            validated_receipt = _replay_evaluation_receipt(
                evidence, payload, launchable=launchable, candidate_sha256=candidate_sha256,
                workload_sha256=workload_sha256, protocol_sha256=protocol_sha256,
                case_id=case_id, purpose=purpose,
                evaluation_protocol=lock.document['evaluation_protocol'],
                fixed_baseline=lock.document['execution'].get('fixed_baseline', {}).get('candidate'),
            )
            if validated_receipt is None:
                return None
            receipt_key = (turn, cast(str, purpose), candidate_sha256)
            if receipt_key in receipts:
                return None
            receipts[receipt_key] = validated_receipt
            receipt_order.append(receipt_key)
            attempt_payload = attempt_payloads.get(receipt_key)
            if attempt_payload is None:
                return None
            replayed_attempts.add(receipt_key)
            _replay_evaluation_attempt_event(
                evidence,
                attempt_payload,
                candidate=launchable,
                protocol_sha256=protocol_sha256,
                final_receipt=validated_receipt,
            )
    unreplayed_attempts = set(attempt_payloads) - replayed_attempts
    if unreplayed_attempts:
        if len(unreplayed_attempts) != 1 or len(faults) != 1:
            return None
        turn, purpose, candidate_sha256 = next(iter(unreplayed_attempts))
        if turn != fault_turn:
            return None
        attempt_payload = attempt_payloads[(turn, purpose, candidate_sha256)]
        launchable = launchables.get((turn, candidate_sha256))
        if launchable is None:
            return None
        _replay_evaluation_attempt_event(
            evidence,
            attempt_payload,
            candidate=launchable,
            protocol_sha256=protocol_sha256,
            final_receipt=None,
        )

    if any(
        turn not in provider_candidates_by_turn
        or candidate_sha256 not in provider_candidates_by_turn[turn]
        for turn, candidate_sha256 in [
            *launchables,
            *rejected,
            *((key[0], key[2]) for key in receipts),
            *((key[0], key[2]) for key in attempt_payloads),
        ]
    ):
        return None
    return launchables, receipts, receipt_order, rejected


def _replay_candidate_selection(
    *,
    candidate_set_turns: set[int],
    cumulative_by_turn: Mapping[int, int],
    empirical_selection: _EmpiricalSelection | None,
    events: Sequence[Mapping[str, object]],
    fault_turn: int | None,
    launchables: Mapping[tuple[int, str], LaunchableCandidate],
    lock: CampaignLock,
    provider_candidate_bytes: Mapping[tuple[int, str], bytes],
    provider_candidates_by_turn: Mapping[int, tuple[str, ...]],
    receipt_order: Sequence[tuple[int, str, str]],
    receipts: Mapping[tuple[int, str, str], EvaluationReceipt],
    rejected: Mapping[tuple[int, str], Mapping[str, object]],
) -> tuple[list[TurnObservation], int, str | None] | None:
    filter_events = [
        event for event in events if event.get("kind") == "candidate_set_filtered"
    ]
    filters: dict[int, Mapping[str, object]] = {}
    filter_order: dict[int, tuple[str, ...]] = {}
    filter_disposition: dict[int, dict[str, str]] = {}
    for event in filter_events:
        payload = _object(event.get("payload"), "candidate_set_filtered.payload")
        turn = payload.get("turn")
        order = payload.get("order")
        if (
            set(payload) != {"turn", "submitted", "launchable", "order"} | (
                {"candidate_selection"} if empirical_selection is not None else set()
            )
            or not isinstance(turn, int)
            or isinstance(turn, bool)
            or turn in filters
            or turn not in candidate_set_turns
            or not isinstance(order, list)
        ):
            return None
        candidates: list[str] = []
        dispositions: dict[str, str] = {}
        for row in order:
            expected_row_fields = {
                "candidate_sha256",
                "disposition",
                "cost",
            }
            expected_row_fields.add("semantic_sha256")
            if empirical_selection is not None:
                expected_row_fields.add("empirical_cost")
            if not isinstance(row, Mapping) or set(row) != expected_row_fields:
                return None
            candidate_sha256 = row.get("candidate_sha256")
            disposition = row.get("disposition")
            cost = row.get("cost")
            semantic_sha256 = row.get("semantic_sha256")
            if (
                not isinstance(candidate_sha256, str)
                or _DIGEST.fullmatch(candidate_sha256) is None
                or candidate_sha256 in dispositions
                or disposition not in {"launchable", "rejected"}
                or (
                    cost is not None
                    and (
                        not isinstance(cost, Mapping)
                        or set(cost) != {"device_fill", "binding_resource"}
                        or not isinstance(cost.get("device_fill"), (int, float))
                        or isinstance(cost.get("device_fill"), bool)
                        or not math.isfinite(float(cost["device_fill"]))
                        or not 0 < float(cost["device_fill"]) <= 1
                        or not isinstance(cost.get("binding_resource"), str)
                        or not cost.get("binding_resource")
                    )
                )
                or (
                    semantic_sha256 is not None
                    and (
                        not isinstance(semantic_sha256, str)
                        or _DIGEST.fullmatch(semantic_sha256) is None
                    )
                )
            ):
                return None
            candidates.append(candidate_sha256)
            dispositions[candidate_sha256] = cast(str, disposition)
        provider_candidates = provider_candidates_by_turn.get(turn)
        if (
            provider_candidates is None
            or len(candidates) != len(provider_candidates)
            or set(candidates) != set(provider_candidates)
            or payload.get("submitted") != len(provider_candidates)
            or payload.get("launchable")
            != sum(value == "launchable" for value in dispositions.values())
        ):
            return None
        if empirical_selection is not None:
            rows_by_candidate = {row["candidate_sha256"]: row for row in order}
            expected_rows = []
            for candidate_sha256 in provider_candidates:
                retained = rows_by_candidate[candidate_sha256]
                expected = dict(retained)
                expected["empirical_cost"] = (
                    empirical_selection.estimate(json.loads(provider_candidate_bytes[(turn, candidate_sha256)]))
                    if retained["disposition"] == "launchable" else None
                )
                expected_rows.append(expected)
            expected_rows, expected_selection = _empirical_filter(expected_rows)
            if (
                _canonical_json_bytes(order) != _canonical_json_bytes(expected_rows)
                or _canonical_json_bytes(payload["candidate_selection"])
                != _canonical_json_bytes(expected_selection)
            ):
                return None
        filters[turn] = payload
        filter_order[turn] = tuple(candidates)
        filter_disposition[turn] = dispositions

    expected_rejections = {
        (turn, candidate_sha256)
        for turn, dispositions in filter_disposition.items()
        for candidate_sha256, disposition in dispositions.items()
        if disposition == "rejected"
    }
    if set(rejected) != expected_rejections:
        return None

    selection_events = [
        event for event in events if event.get("kind") == "candidate_selected"
    ]
    selections: dict[int, Mapping[str, object]] = {}
    for event in selection_events:
        payload = _object(event.get("payload"), "candidate_selected.payload")
        turn = payload.get("turn")
        candidate_sha256 = payload.get("candidate_sha256")
        qualified = payload.get("qualified_search_candidates")
        if (
            set(payload)
            != {
                "turn",
                "candidate_sha256",
                "qualified_search_candidates",
                "reason",
            }
            or not isinstance(turn, int)
            or isinstance(turn, bool)
            or turn in selections
            or turn not in candidate_set_turns
            or not isinstance(candidate_sha256, str)
            or _DIGEST.fullmatch(candidate_sha256) is None
            or not isinstance(qualified, list)
            or any(
                not isinstance(value, str) or _DIGEST.fullmatch(value) is None
                for value in qualified
            )
        ):
            return None
        selections[turn] = payload

    observations: list[TurnObservation] = []
    evaluation_protocol = _object(
        lock.document["evaluation_protocol"], "evaluation_protocol"
    )
    searches_per_turn = int(
        evaluation_protocol.get("searches_per_turn", 1)
    )
    attribution_evaluation = evaluation_protocol.get("attribution_evaluation")
    expected_searches: dict[int, list[str]] = {}
    expected_diagnoses, expected_searches = _expected_matched_diagnoses_v1(
        filters=filters,
        receipts=receipts,
        receipt_order=receipt_order,
        searches_per_turn=searches_per_turn,
        materiality_ratio=float(
            evaluation_protocol.get("search_materiality_ratio", math.inf)
        ),
    )
    _validate_matched_diagnoses_v1(
        events,
        expected=expected_diagnoses,
        fault_turn=fault_turn,
    )
    for turn, provider_candidates in sorted(provider_candidates_by_turn.items()):
        if turn in candidate_set_turns:
            if turn not in filters:
                if turn != fault_turn:
                    return None
                continue
            selection = selections.get(turn)
            if selection is None:
                if turn != fault_turn:
                    return None
                continue
            selected = cast(str, selection["candidate_sha256"])
            order = filter_order[turn]
            dispositions = filter_disposition[turn]
            if selected not in provider_candidates:
                return None
            search_keys = [
                key
                for key in receipt_order
                if key[0] == turn and key[1] == "search"
            ]
            if selection.get("reason") == "all_candidates_rejected":
                if (
                    selected != order[0]
                    or any(value == "launchable" for value in dispositions.values())
                    or search_keys
                    or selection.get("qualified_search_candidates") != []
                    or (turn, selected) not in rejected
                ):
                    return None
                if turn != fault_turn:
                    observations.append(
                        TurnObservation(
                            turn,
                            cumulative_by_turn[turn],
                            selected,
                            False,
                            None,
                        )
                    )
                continue
            if not search_keys or len(search_keys) > searches_per_turn:
                return None
            searched_candidates = [key[2] for key in search_keys]
            if (
                len(set(searched_candidates)) != len(searched_candidates)
                or any(dispositions.get(value) != "launchable" for value in searched_candidates)
                or [order.index(value) for value in searched_candidates]
                != sorted(order.index(value) for value in searched_candidates)
                or (
                    searched_candidates
                    != (
                        expected_searches[turn][
                            : len(searched_candidates)
                        ]
                        if turn == fault_turn
                        else expected_searches[turn]
                    )
                )
            ):
                return None
            qualified_search = [
                key[2] for key in search_keys if _receipt_qualifies(receipts[key])
            ]
            expected_selected = (
                min(
                    qualified_search,
                    key=lambda value: _receipt_latency_ms(
                        receipts[(turn, "search", value)]
                    )
                    or float("inf"),
                )
                if qualified_search
                else searched_candidates[0]
            )
            expected_reason = (
                "lowest_qualified_search_latency"
                if qualified_search
                else "no_qualified_search_candidate"
            )
            if (
                selection.get("qualified_search_candidates") != qualified_search
                or selected != expected_selected
                or selection.get("reason") != expected_reason
            ):
                return None
            confirms = [
                receipt
                for (candidate_turn, purpose, candidate), receipt in receipts.items()
                if candidate_turn == turn
                and purpose == "confirmatory"
                and candidate == selected
            ]
            foreign_confirms = [
                key
                for key in receipts
                if key[0] == turn
                and key[1] == "confirmatory"
                and key[2] != selected
            ]
            if foreign_confirms or (not qualified_search and confirms):
                return None
            if turn == fault_turn:
                continue
            if len(confirms) != (1 if qualified_search else 0):
                return None
            confirmed = confirms[0] if confirms else None
            qualified = confirmed is not None and _receipt_qualifies(confirmed)
            attributions = [
                key
                for key in receipts
                if key[0] == turn and key[1] == "attribution"
            ]
            expected_attributions = (
                [
                    candidate
                    for candidate in searched_candidates
                    if receipts[(turn, "search", candidate)].correctness_passed
                ]
                if attribution_evaluation == _ATTRIBUTION_EVALUATION
                else (
                    [selected]
                    if qualified
                    and attribution_evaluation
                    == _LEGACY_ATTRIBUTION_EVALUATION
                    else []
                )
            )
            if (
                len(attributions) != len(expected_attributions)
                or {key[2] for key in attributions}
                != set(expected_attributions)
            ):
                return None
            observations.append(
                TurnObservation(
                    turn,
                    cumulative_by_turn[turn],
                    selected,
                    qualified,
                    _receipt_latency_ms(confirmed) if qualified else None,
                )
            )
        else:
            # Historical evidence wrote exactly one candidate per Turn and had no
            # explicit filter/selection events. Keep that bounded spelling readable;
            # new evidence must use the candidate-set contract above.
            if len(provider_candidates) != 1:
                return None
            selected = provider_candidates[0]
            if turn == fault_turn:
                continue
            has_rejection = (turn, selected) in rejected
            has_evaluation = any(
                key[0] == turn and key[2] == selected for key in receipts
            )
            if has_rejection == has_evaluation:
                return None
            confirmed = receipts.get((turn, "confirmatory", selected))
            qualified = confirmed is not None and _receipt_qualifies(confirmed)
            observations.append(
                TurnObservation(
                    turn,
                    cumulative_by_turn[turn],
                    selected,
                    qualified,
                    _receipt_latency_ms(confirmed) if qualified else None,
                )
            )

    if set(filters) != candidate_set_turns - ({fault_turn} if fault_turn else set()):
        # A fault may happen after its filter was written, so the final Turn is the
        # sole allowed extra member on either side of this equality.
        if not (
            fault_turn in candidate_set_turns
            and set(filters) | {fault_turn} == candidate_set_turns
        ):
            return None
    if any(turn not in filters for turn in selections):
        return None
    for turn, candidate_sha256 in rejected:
        if turn in candidate_set_turns:
            # Rejection is a property of each set member, not of the Turn's
            # eventual selection.  A mixed set legitimately records rejected
            # members while selecting a different launchable member.  The exact
            # rejected-member set was checked against the filter dispositions
            # above; the all-rejected selection rule is checked in the selection
            # replay branch.
            if (
                filter_disposition.get(turn, {}).get(candidate_sha256)
                != "rejected"
            ):
                return None

    for turn, candidate_sha256 in launchables:
        if turn in candidate_set_turns and filter_disposition.get(turn, {}).get(
            candidate_sha256
        ) != "launchable":
            return None
    return observations, searches_per_turn, attribution_evaluation


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
) -> bool:
    if not observations and not faults:
        return False
    if [item.turn for item in observations] != list(range(1, len(observations) + 1)):
        return False
    resolved = _object(lock.document["resolved_inputs"], "resolved_inputs")
    budget = _object(resolved["budget"], "resolved_inputs.budget")
    projected = project_checkpoints(
        turns=observations,
        checkpoints=cast(list[int], budget["checkpoints"]),
        terminal_provider_tokens=(
            fault_terminal_tokens
            if fault_terminal_tokens is not None
            else max(cumulative_by_turn.values())
        ),
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
    expected_counts = {
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
    expected_stop_reason = derive_ralph_stop_reason(
        RalphBudget.from_mapping(budget),
        turn=state_turn,
        cumulative_provider_tokens=max(cumulative_by_turn.values()),
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
        != max(cumulative_by_turn.values())
        or ralph_state.get("evaluation_counts") != expected_counts
        or ralph_state.get("terminal_reason") != expected_stop_reason
    ):
        return False
    expected_observation, expected_endpoint = _matched_endpoint_from_checkpoint(
        projected[-1], audit.protocol_adherence
    )
    return (
        audit.endpoint_observation == expected_observation
        and audit.endpoint == expected_endpoint
    )
