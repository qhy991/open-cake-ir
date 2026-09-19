"""Independent reconstruction of Campaign outcomes from raw retained evidence."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Callable, Mapping, Sequence, cast

from open_cake_ir.evidence import EvidenceStore, RunAudit

from ._documents import _canonical_json_bytes, _object
from ._policies import _MATCHED_EVENT_KINDS_V1, _matched_evidence_policy_version
from .claude import CLAUDE_EVENT_CONTRACTS, candidate_write_declared_unwitnessed
from .contracts import CampaignLock
from .executor import ExecutorRevision
from .endpoints import endpoint_policy
from .evaluation_lifecycle import replay_evaluation_invocations
from .selection import _EmpiricalSelection, _empirical_context
from .replay_candidates import _artifact_outcomes_are_closed, _replay_candidates
from .replay_outcomes import _replay_terminal
from .replay_provider import (
    _expected_terminal_message,
    _replay_provider_turns,
    replay_fault_usage,
)
from .replay_refusals import ReplayRefusal, ReplayResult, refuse
from .replay_selection import _replay_candidate_selection

_REQUIRED_FAULT_FIELDS = frozenset({
    "fault", "exception_type", "turn", "stage", "terminal_provider_tokens",
})
# `exception_message` is optional, so evidence sealed before it replays unchanged.
# `observed_quota` likewise: replay rederives it from the retained provider stdout
# and rejects a value that differs, but evidence sealed without it replays unchanged.
_OPTIONAL_FAULT_FIELDS = frozenset({
    "objects", "artifact_rejections", "provider_usage", "terminal_provider_tokens_scope",
    "provider_usage_witness_mismatch", "exception_message", "observed_quota",
})


def _identity_reference(value: object, context: str) -> dict[str, object]:
    """A Compiler or Executor reference as its boundary reads it: the path and the id.

    A lock sealed before the identity reform also carries `canonical_sha256`. Replay
    reads that row as data and neither requires nor forwards it: the digest was a
    function of the commit the id already names, and the boundary is the exact check.
    """
    reference = _object(value, context)
    return {key: item for key, item in reference.items() if key != "canonical_sha256"}


def _fault_message_is_closed(payload) -> bool:
    """The harness's own account of a fault: absent, or bounded readable text."""
    if "exception_message" not in payload:
        return True
    message = payload["exception_message"]
    return message is None or isinstance(message, str) and 0 < len(message) <= 2048



def replay_matched_run(
    evidence: EvidenceStore,
    audit: RunAudit,
    lock: CampaignLock,
    *,
    project_root: Path,
    manifest_parser: Callable,
    task_package: Callable,
) -> ReplayResult:
    """Reconstruct one Run from its raw retained objects, independently of the writer.

    The result is truthy when the ledger rederives; otherwise it carries the refusal
    that stopped the reconstruction. A ledger-shape check names its location. A live
    decision function or document primitive refuses in its own words and at no
    ledger location, so its refusal is carried as `unlocated` with that text rather
    than dropped at the reporting boundary. Any other exception propagates.
    """
    try:
        _replay_matched_run(
            evidence, audit, lock, project_root=project_root,
            manifest_parser=manifest_parser, task_package=task_package,
        )
    except ReplayRefusal as refusal:
        return ReplayResult(audit.run_id, (refusal,))
    except ValueError as error:
        return ReplayResult(audit.run_id, (ReplayRefusal("unlocated", str(error)),))
    return ReplayResult(audit.run_id)


def _replay_matched_run(
    evidence: EvidenceStore,
    audit: RunAudit,
    lock: CampaignLock,
    *,
    project_root: Path,
    manifest_parser: Callable,
    task_package: Callable,
) -> None:
    from .bindings import load_compiler_reference
    compiler_ref = _identity_reference(lock.document["compiler_revision"], "compiler_revision")
    load_compiler_reference(project_root, compiler_ref, "replay.compiler_revision")
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
    if audit.run_id not in lock.run_order:
        refuse("run_id", "not a Run of this Campaign Lock", observed=audit.run_id,
               expected=lock.run_order)
    unknown_kinds = [
        f"events[{index}]" for index, kind in enumerate(kinds) if kind not in _MATCHED_EVENT_KINDS_V1
    ]
    if unknown_kinds:
        refuse("events", "event kinds outside the matched vocabulary", observed=unknown_kinds,
               expected=_MATCHED_EVENT_KINDS_V1)
    for kind in ("run_started", "checkpoints_projected", "run_terminal"):
        if kinds.count(kind) != 1:
            refuse("events", f"{kind} event count differs", observed=kinds.count(kind), expected=1)
    if kinds[0] != "run_started" or kinds[-2:] != ["checkpoints_projected", "run_terminal"]:
        refuse("events", "the ledger does not start with run_started and end with "
               "checkpoints_projected, run_terminal",
               observed=[kinds[0], *kinds[-2:]],
               expected=["run_started", "checkpoints_projected", "run_terminal"])
    start_payload = _object(
        events[0].get("payload"), "run_started.payload"
    )
    expected_start = {
        "sequence": lock.run_order.index(audit.run_id) + 1,
        "assigned_arm": arm,
        "automatic_retries": 0,
        "replacement_run": False,
    }
    if start_payload != expected_start:
        refuse("run_started.payload", "differs from the Run's place in the Campaign Lock",
               observed=start_payload, expected=expected_start)
    terminal_payload = _object(
        events[-1].get("payload"), "run_terminal.payload"
    )
    expected_terminal = {
        "protocol_adherence": audit.protocol_adherence,
        "endpoint_observation": audit.endpoint_observation,
        "endpoint": dict(audit.endpoint) if audit.endpoint is not None else None,
    }
    extra_terminal_fields = set(terminal_payload) - set(expected_terminal) - {"boundary_diagnostic"}
    if extra_terminal_fields:
        refuse("run_terminal.payload", "fields outside the terminal contract",
               observed=extra_terminal_fields, expected=set(expected_terminal) | {"boundary_diagnostic"})
    for field, expected in expected_terminal.items():
        if terminal_payload.get(field) != expected:
            refuse(f"run_terminal.payload.{field}", "differs from the sealed audit",
                   observed=terminal_payload.get(field), expected=expected)
    boundary_diagnostic = terminal_payload.get("boundary_diagnostic")
    fault_count = len([event for event in events if event.get("kind") == "run_fault"])
    if boundary_diagnostic is not None:
        # F-2026-09-16-002: the terminal names the retained provider fault its
        # settled checkpoint outlived. The marker is closed, requires exactly the
        # one fault observation it converts, and only an adhered terminal can
        # carry it; usage, quota and the unwitnessed shape are rederived below
        # from the retained fault stdout, never from this declaration.
        location = "run_terminal.payload.boundary_diagnostic"
        if not isinstance(boundary_diagnostic, Mapping) or set(boundary_diagnostic) != {"turn", "stage", "diagnostic"}:
            refuse(location, "not a closed boundary marker", observed=boundary_diagnostic,
                   expected={"turn", "stage", "diagnostic"})
        if boundary_diagnostic.get("stage") != "provider":
            refuse(f"{location}.stage", "only a provider fault converts", observed=boundary_diagnostic.get("stage"),
                   expected="provider")
        if boundary_diagnostic.get("diagnostic") != "candidate_write_declared_unwitnessed":
            refuse(f"{location}.diagnostic", "not the named boundary diagnostic",
                   observed=boundary_diagnostic.get("diagnostic"),
                   expected="candidate_write_declared_unwitnessed")
        if type(boundary_diagnostic.get("turn")) is not int or boundary_diagnostic["turn"] <= 0:
            refuse(f"{location}.turn", "not a positive integer", observed=boundary_diagnostic.get("turn"))
        if audit.protocol_adherence != "adhered":
            refuse("run_terminal.payload.protocol_adherence", "only an adhered terminal carries a boundary marker",
                   observed=audit.protocol_adherence, expected="adhered")
        if fault_count != 1:
            refuse("run_fault", "a boundary marker converts exactly one fault observation",
                   observed=fault_count, expected=1)
    turn_events = [
        (index, _object(event.get("payload"), f"event.{event.get('kind')}.payload")["turn"])
        for index, event in enumerate(events[1:-2], start=1)
        if "turn"
        in _object(
            event.get("payload"), f"event.{event.get('kind')}.payload"
        )
    ]
    for index, turn in turn_events:
        if not isinstance(turn, int) or isinstance(turn, bool) or turn <= 0:
            refuse(f"events[{index}].payload.turn", "not a positive integer", observed=turn)
    turns = [turn for _, turn in turn_events]
    if turns != sorted(turns):
        refuse("events", "Turn-bearing events are not in Turn order",
               observed=[f"events[{index}]:turn={turn}" for index, turn in turn_events])
    provider_events = [event for event in events if event.get("kind") == "provider_turn_completed"]
    checkpoint_events = [event for event in events if event.get("kind") == "checkpoints_projected"]
    if not provider_events:
        if (endpoint_policy(lock.analysis_plan) is not None and audit.protocol_adherence == "adhered" and
                kinds == ["run_started", "checkpoints_projected", "run_terminal"]):
            protocol = lock.document["evaluation_protocol"]
            _replay_terminal(
                attribution_evaluation=protocol.get("attribution_evaluation"), audit=audit,
                checkpoint_events=checkpoint_events, cumulative_by_turn={},
                fault_terminal_tokens=None, faults=(), lock=lock, observations=(), receipts={},
                searches_per_turn=protocol.get("searches_per_turn", 1),
            )
            return
        _replay_provider_fault(
            audit=audit,
            checkpoint_events=checkpoint_events,
            events=events,
            lock=lock,
            evidence=evidence,
        )
        return
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
        executor = ExecutorRevision.load_reference(
            project_root,
            _identity_reference(lock.document["execution"]["executor_revision"], "execution.executor_revision"),
            "execution.executor_revision",
        )
        empirical_selection = _EmpiricalSelection(
            selection_binding,
            context=_empirical_context(
                executor, workload_sha256=lock.document["workload"]["canonical_sha256"],
                case_id=lock.document["evaluation_protocol"]["case_id"],
            ),
            compiler_revision_id=compiler_ref["revision_id"],
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
    cumulative_by_turn, provider_candidates_by_turn, provider_candidate_bytes, candidate_set_turns, prior_cumulative = provider
    workload_sha256 = str(_object(lock.document["workload"], "workload")["canonical_sha256"])
    protocol_sha256 = sha256(
        _canonical_json_bytes(lock.document["evaluation_protocol"])
    ).hexdigest()
    case_id = str(_object(lock.document["evaluation_protocol"], "protocol")["case_id"])
    faults = [event for event in events if event.get("kind") == "run_fault"]
    if len(faults) > 1:
        refuse("run_fault", "a Run retains at most one fault", observed=len(faults), expected="0 or 1")
    fault_turn: int | None = None
    fault_terminal_tokens: int | None = None
    if faults:
        fault_payload = _object(faults[0].get("payload"), "run_fault.payload")
        _refuse_unless_fault_payload_is_closed(fault_payload)
        if (
            fault_payload.get("fault") != audit.protocol_adherence
            and boundary_diagnostic is None
        ):
            # The converted boundary terminal, validated against its fault
            # below, is the one adherent exception to fault == adherence.
            refuse("run_fault.payload.fault", "differs from the terminal's protocol adherence",
                   observed=fault_payload.get("fault"), expected=audit.protocol_adherence)
        fault_turn_value = fault_payload.get("turn")
        fault_stage = fault_payload.get("stage")
        fault_terminal_value = fault_payload.get("terminal_provider_tokens")
        usage_delta = replay_fault_usage(payload=fault_payload, evidence=evidence,
            provider=provider_authority,
            expected_thread_id=provider_events[-1]["payload"]["thread_id"])
        if fault_terminal_value != prior_cumulative + usage_delta:
            refuse("run_fault.payload.terminal_provider_tokens",
                   "differs from the completed Turns' total plus the usage rederived from the fault stdout",
                   observed=fault_terminal_value, expected=prior_cumulative + usage_delta)
        if not isinstance(fault_turn_value, int) or isinstance(fault_turn_value, bool) or fault_turn_value <= 0:
            refuse("run_fault.payload.turn", "not a positive integer", observed=fault_turn_value)
        if fault_stage not in {"provider", "environment", "evaluation"}:
            refuse("run_fault.payload.stage", "not a fault stage", observed=fault_stage,
                   expected={"provider", "environment", "evaluation"})
        if not isinstance(fault_terminal_value, int) or isinstance(fault_terminal_value, bool):
            refuse("run_fault.payload.terminal_provider_tokens", "not an integer", observed=fault_terminal_value)
        if fault_terminal_value < prior_cumulative:
            refuse("run_fault.payload.terminal_provider_tokens", "below the completed Turns' total",
                   observed=fault_terminal_value, expected=f">= {prior_cumulative}")
        expected_fault_turn = len(provider_events) + (1 if fault_stage == "provider" else 0)
        if fault_turn_value != expected_fault_turn:
            refuse("run_fault.payload.turn", f"a {fault_stage} fault lies in a different Turn",
                   observed=fault_turn_value, expected=expected_fault_turn)
        if fault_stage != "provider" and fault_terminal_value != prior_cumulative:
            refuse("run_fault.payload.terminal_provider_tokens",
                   f"a {fault_stage} fault adds no provider usage",
                   observed=fault_terminal_value, expected=prior_cumulative)
        if boundary_diagnostic is not None:
            # Scoped to exactly the named boundary fault: same stage, same Turn,
            # the fault type that carries the name, and a retained stdout that
            # reparses into the declared-but-unwitnessed shape under the fault
            # Turn's own terminal expectation.
            for field, expected in (
                ("fault", "provider_fault"),
                ("stage", "provider"),
                ("exception_type", "ProviderBoundaryDeclarationFault"),
            ):
                if fault_payload.get(field) != expected:
                    refuse(f"run_fault.payload.{field}", "not the fault a boundary marker converts",
                           observed=fault_payload.get(field), expected=expected)
            if boundary_diagnostic.get("turn") != fault_turn_value:
                refuse("run_terminal.payload.boundary_diagnostic.turn", "differs from the fault Turn",
                       observed=boundary_diagnostic.get("turn"), expected=fault_turn_value)
            stdout_references = [
                cast(Mapping[str, object], reference)
                for reference in fault_payload.get("objects", [])
                if isinstance(reference, Mapping)
                and reference.get("role") == "provider_stdout"
            ]
            if event_contract not in CLAUDE_EVENT_CONTRACTS:
                refuse("run_terminal.payload.boundary_diagnostic",
                       "the unwitnessed-write diagnosis exists only under a Claude event contract",
                       observed=event_contract, expected=CLAUDE_EVENT_CONTRACTS)
            if len(stdout_references) != 1:
                refuse("run_fault.payload.objects", "provider_stdout reference count differs",
                       observed=len(stdout_references), expected=1)
            if not candidate_write_declared_unwitnessed(
                evidence.read_object(stdout_references[0]),
                expected_terminal_message=_expected_terminal_message(
                    arm, fault_turn_value, event_contract
                ),
                event_contract=event_contract,
            ):
                refuse("run_fault.payload.objects.provider_stdout",
                       "retained stdout does not reparse into a declared-but-unwitnessed candidate write")
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
    launchables, receipts, receipt_order, rejected = candidates
    try:
        invocation_counts = replay_evaluation_invocations(events, receipts=receipts,
            budget=replay_budget, protocol=lock.document["evaluation_protocol"])
    except ValueError as error:
        # The lifecycle owner's own diagnosis, placed at the events it examined.
        refuse("evaluation_invocations", str(error))
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
    observations, searches_per_turn, attribution_evaluation = selected
    _replay_terminal(
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
        invocation_counts=invocation_counts,
        boundary_converted=boundary_diagnostic is not None,
    )

def _refuse_unless_fault_payload_is_closed(fault_payload: Mapping[str, object]) -> None:
    """The retained fault carries its required fields, no others, and closed text and roles."""
    missing = _REQUIRED_FAULT_FIELDS - set(fault_payload)
    if missing:
        refuse("run_fault.payload", "required fields are missing", observed=set(fault_payload),
               expected=_REQUIRED_FAULT_FIELDS)
    extra = set(fault_payload) - _REQUIRED_FAULT_FIELDS - _OPTIONAL_FAULT_FIELDS
    if extra:
        refuse("run_fault.payload", "fields outside the fault contract", observed=extra,
               expected=_REQUIRED_FAULT_FIELDS | _OPTIONAL_FAULT_FIELDS)
    if not isinstance(fault_payload.get("exception_type"), str) or not fault_payload.get("exception_type"):
        refuse("run_fault.payload.exception_type", "not a non-empty string",
               observed=fault_payload.get("exception_type"))
    if not _fault_message_is_closed(fault_payload):
        refuse("run_fault.payload.exception_message", "not absent, None or readable text of at most 2048 characters")
    if not _artifact_outcomes_are_closed(fault_payload):
        refuse("run_fault.payload", "retained/rejected artifact roles are not a closed partition",
               observed={"objects": fault_payload.get("objects"),
                         "artifact_rejections": fault_payload.get("artifact_rejections")})


def _replay_provider_fault(
    *,
    audit: RunAudit,
    checkpoint_events: Sequence[Mapping[str, object]],
    events: Sequence[Mapping[str, object]],
    lock: CampaignLock,
    evidence: EvidenceStore,
) -> None:
    """A Run whose first provider Turn faulted: exactly four events and a zero-Turn terminal."""
    kinds = [event.get("kind") for event in events]
    expected_kinds = ["run_started", "run_fault", "checkpoints_projected", "run_terminal"]
    if kinds != expected_kinds:
        refuse("events", "a Run with no completed provider Turn retains exactly its fault",
               observed=kinds, expected=expected_kinds)
    faults = events[1:2]
    fault_payload = _object(faults[0].get("payload"), "run_fault.payload")
    checkpoint_payload = _object(
        checkpoint_events[0].get("payload"), "checkpoints_projected.payload"
    )
    _refuse_unless_fault_payload_is_closed(fault_payload)
    if set(checkpoint_payload) != {"checkpoints", "ralph"}:
        refuse("checkpoints_projected.payload", "fields differ", observed=set(checkpoint_payload),
               expected={"checkpoints", "ralph"})
    terminal_tokens = fault_payload.get("terminal_provider_tokens")
    if fault_payload.get("turn") != 1:
        refuse("run_fault.payload.turn", "a fault before any completed Turn is in Turn 1",
               observed=fault_payload.get("turn"), expected=1)
    if fault_payload.get("stage") != "provider":
        refuse("run_fault.payload.stage", "a fault before any completed Turn is a provider fault",
               observed=fault_payload.get("stage"), expected="provider")
    if not isinstance(terminal_tokens, int) or isinstance(terminal_tokens, bool) or terminal_tokens < 0:
        refuse("run_fault.payload.terminal_provider_tokens", "not a non-negative integer",
               observed=terminal_tokens)
    arm = audit.run_id.rsplit("-", 1)[0]
    provider = lock.document["resolved_inputs"]["arm_environments"][arm].get("provider", {})
    delta = replay_fault_usage(payload=fault_payload, evidence=evidence, provider=provider)
    if terminal_tokens != delta:
        refuse("run_fault.payload.terminal_provider_tokens",
               "differs from the usage rederived from the fault stdout",
               observed=terminal_tokens, expected=delta)
    if fault_payload.get("fault") != audit.protocol_adherence:
        refuse("run_fault.payload.fault", "differs from the terminal's protocol adherence",
               observed=fault_payload.get("fault"), expected=audit.protocol_adherence)
    if audit.endpoint_observation != "missing":
        refuse("run_terminal.payload.endpoint_observation", "a faulted first Turn observes no endpoint",
               observed=audit.endpoint_observation, expected="missing")
    if audit.endpoint is not None:
        refuse("run_terminal.payload.endpoint", "a faulted first Turn has no endpoint",
               observed=dict(audit.endpoint), expected=None)
    protocol = lock.document.get("evaluation_protocol", {})
    _replay_terminal(
        attribution_evaluation=protocol.get("attribution_evaluation"), audit=audit,
        checkpoint_events=checkpoint_events, cumulative_by_turn={},
        fault_terminal_tokens=terminal_tokens, faults=faults, lock=lock, observations=(), receipts={},
        searches_per_turn=protocol.get("searches_per_turn", 1),
    )
