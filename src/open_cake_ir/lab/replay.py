"""Independent reconstruction of Campaign outcomes from raw retained evidence."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Callable, Mapping, Sequence, cast

from open_cake_ir.evidence import EvidenceStore, RunAudit

from ._documents import _canonical_json_bytes, _object
from ._policies import _MATCHED_EVENT_KINDS_V1, _matched_evidence_policy_version
from .contracts import CampaignLock
from .executor import ExecutorRevision
from .selection import _EmpiricalSelection, _empirical_context
from .replay_candidates import _artifact_outcomes_are_closed, _replay_candidates
from .replay_outcomes import _replay_terminal
from .replay_provider import _replay_provider_turns, replay_fault_usage
from .replay_selection import _replay_candidate_selection


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
            evidence=evidence,
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
            - {"objects", "artifact_rejections", "provider_usage", "terminal_provider_tokens_scope", "provider_usage_witness_mismatch"}
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
        usage_delta = replay_fault_usage(payload=fault_payload, evidence=evidence,
            provider=provider_authority,
            expected_thread_id=provider_events[-1]["payload"]["thread_id"])
        if (usage_delta is None or fault_terminal_value != prior_cumulative + usage_delta):
            return False
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
    evidence: EvidenceStore,
) -> bool:
    faults = [event for event in events if event.get("kind") == "run_fault"]
    if len(faults) != 1:
        return False
    fault_payload = _object(faults[0].get("payload"), "run_fault.payload")
    checkpoint_payload = _object(
        checkpoint_events[0].get("payload"), "checkpoints_projected.payload"
    )
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
        - {"objects", "artifact_rejections", "provider_usage", "terminal_provider_tokens_scope", "provider_usage_witness_mismatch"}
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
    arm = audit.run_id.rsplit("-", 1)[0]
    provider = lock.document["resolved_inputs"]["arm_environments"][arm].get("provider", {})
    delta = replay_fault_usage(payload=fault_payload, evidence=evidence, provider=provider)
    if (delta is None or terminal_tokens != delta
            or fault_payload.get("fault") != audit.protocol_adherence
            or audit.endpoint_observation != "missing" or audit.endpoint is not None):
        return False
    protocol = lock.document.get("evaluation_protocol", {})
    return _replay_terminal(
        attribution_evaluation=protocol.get("attribution_evaluation"), audit=audit,
        checkpoint_events=checkpoint_events, cumulative_by_turn={},
        fault_terminal_tokens=terminal_tokens, faults=faults, lock=lock, observations=(), receipts={},
        searches_per_turn=protocol.get("searches_per_turn", 1),
    )
