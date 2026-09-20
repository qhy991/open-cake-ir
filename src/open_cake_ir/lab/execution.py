"""Live Campaign execution, budget accounting and evidence writing."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, cast

from open_cake_ir.evaluation import EvaluationReceipt, LaunchableCandidate
from open_cake_ir.evidence import EvidenceStore

from ._documents import _canonical_json_bytes, _digest, _name, _object, differs
from ._policies import (
    _ATTRIBUTION_EVALUATION,
    _LEGACY_ATTRIBUTION_EVALUATION,
    _matched_evidence_policy_version,
)
from .archive import _arm_artifact_roles, _archive_provider_turn, _candidate_artifact_media_type
from .evaluation_writer import EvaluationWriter
from .execution_admission import validate_execution_bindings, validate_run_bindings
from .candidate_filter import _build_filter_candidates, record_candidate_rejections
from .diagnoses import rejected_peer_feedback
from .run_completion import _seal_run, record_run_fault
from .faults import RunProtocolFault, ReportedProviderUsage
from .provider_events import reported_provider_usage, provider_token_delta
from .checkpoints import TurnObservation
from .nomination import FinalConfirmation, nominate, nomination_document
from .run_spec import RunSpecification, RunRef
from .contracts import CampaignLock, CampaignRef, RunEvaluator, RunProvider, TurnRequest
from .custody import admit_new_campaign_path
from .environments import AuthoringEnvironment, CandidateSubmission, EnvironmentResult
from .executor import ExecutorRevision
from .pairing import comparison_arm, native_backend
from .ralph import RalphBudget, RalphController
from .selection import (
    _collapse_diagnosis,
    _matched_search_decision,
    _matched_search_plan,
    _receipt_latency_ms,
    _receipt_qualifies,
)


@dataclass(frozen=True)
class _SearchedCandidate:
    """One candidate kept intact from authored bytes through its search receipt."""

    submission: CandidateSubmission
    environment_result: EnvironmentResult
    launchable: LaunchableCandidate
    receipt: EvaluationReceipt
    attribution: EvaluationReceipt | None

from .pairing import matched_run_arms
from open_cake_ir.evaluation.paired import paired_protocol


def _baseline_comparison_feedback(
    lock: CampaignLock, timing: Mapping[str, object]
) -> dict[str, object]:
    """Project the exact black-box opponent into next-Turn feedback."""

    medians = timing.get("pooled_medians_ms")
    fixed = _object(lock.document["execution"], "campaign execution").get(
        "fixed_baseline"
    )
    selection = (
        _object(fixed, "fixed baseline").get("selection")
        if isinstance(fixed, Mapping)
        else None
    )
    return {
        "source": (
            _object(selection, "fixed baseline selection").get("source")
            if isinstance(selection, Mapping)
            else "campaign_fixed_baseline"
        ),
        "baseline_latency_ms": (
            _object(medians, "paired timing medians").get("baseline")
            if isinstance(medians, Mapping)
            else None
        ),
        "candidate_speedup": timing.get("speedup"),
        "measurement_quality_passed": timing.get(
            "measurement_quality_passed"
        ),
    }


def execute_campaign(
    lock: CampaignLock,
    evidence_root: str | Path,
    *,
    project_root: Path,
    workload_loader: Callable,
    validate_authoring: Callable,
    clock: Callable[[], float],
    provider: RunProvider,
    environments: Mapping[str, AuthoringEnvironment],
    evaluator: RunEvaluator,
) -> CampaignRef:
    """Own every Turn, budget, checkpoint, feedback and terminal decision."""

    root = admit_new_campaign_path(
        project_root,
        evidence_root,
        role="Campaign Evidence root",
    )
    matched_run_arms(environments, lock.claim_scope)
    if set(environments) != set(lock.document["resolved_inputs"]["arm_environments"]):
        raise differs(
            "Campaign Authoring Environment set",
            expected=sorted(lock.document["resolved_inputs"]["arm_environments"]),
            observed=sorted(environments),
        )
    if lock.study_kind != "matched_search":
        raise ValueError("Lab.execute matched-search path requires a matched Campaign Lock")
    ExecutorRevision.load_reference(
        project_root,
        _object(lock.document["execution"], "campaign_lock.execution").get("executor_revision"),
        "campaign_lock.execution.executor_revision",
    )
    resolved_inputs = _object(
        lock.document["resolved_inputs"], "campaign_lock.resolved_inputs"
    )
    _matched_evidence_policy_version(
        _object(
            resolved_inputs["evidence_policy"],
            "campaign_lock.resolved_inputs.evidence_policy",
        ),
        "campaign_lock.resolved_inputs.evidence_policy",
    )
    evaluation_protocol = _object(
        lock.document["evaluation_protocol"], "campaign_lock.evaluation_protocol"
    )
    expected_protocol_sha256 = sha256(
        _canonical_json_bytes(evaluation_protocol)
    ).hexdigest()
    validate_execution_bindings(
        lock=lock,
        resolved_inputs=resolved_inputs,
        evaluation_protocol=evaluation_protocol,
        expected_protocol_sha256=expected_protocol_sha256,
        provider=provider,
        evaluator=evaluator,
        environments=environments,
        project_root=project_root,
        workload_loader=workload_loader,
        validate_authoring=validate_authoring,
    )
    evidence = EvidenceStore.create(root)
    for run_id in lock.run_order:
        specification = lock.run_specification(run_id)
        _execute_run(specification, project_root=project_root, evidence=evidence, clock=clock, provider=provider,
                     environment=environments[specification.condition_id], evaluator=evaluator)

    return CampaignRef(lock=lock, evidence_root=evidence.root)


def execute_run(specification: RunSpecification, evidence_root, *, project_root,
                workload_loader, clock, provider, environment, evaluator, task_package, validate_run=None):
    """Execute a frozen engineering or Study-assigned Run through the same engine."""
    root = admit_new_campaign_path(project_root, evidence_root, role='Run Evidence root')
    specification = RunSpecification.from_dict(specification.document)
    if validate_run is not None:
        validate_run(specification)
    validate_run_bindings(specification, project_root=project_root,
                          workload_loader=workload_loader, provider=provider,
                          environment=environment, evaluator=evaluator, task_package=task_package)
    evidence = EvidenceStore.create(root)
    _execute_run(specification, project_root=project_root, evidence=evidence, clock=clock, provider=provider,
                 environment=environment, evaluator=evaluator)
    return RunRef(specification, evidence.root)


def _execute_run(specification: RunSpecification, *, project_root, evidence, clock, provider, environment, evaluator):
    """The one search/evaluation lifecycle for every frozen Run."""
    document = specification.document
    from open_cake_ir.compiler import Compiler
    from .actions import resolve_action_set
    action_compiler = None
    def compiler_factory():
        nonlocal action_compiler
        if action_compiler is None:
            action_compiler = Compiler.load(project_root, project_root / document['compiler_revision']['path'])
        return action_compiler
    prior_candidates = {}
    baselines = {name: _canonical_json_bytes(program) for name, program in
                 document['reference_inputs'].get('baseline_programs', {}).items()}
    run_id = specification.run_id
    sequence = document['sequence']
    budget = document['budget']
    checkpoints = budget['checkpoints']
    maximum_turns = budget['maximum_turns']
    maximum_candidates_per_turn = budget['maximum_candidates_per_turn']
    evaluation_protocol = document['evaluation_protocol']
    attribution_evaluation = evaluation_protocol.get('attribution_evaluation')
    profile_each_search_survivor = attribution_evaluation == _ATTRIBUTION_EVALUATION
    ralph_budget = RalphBudget.from_mapping(budget)
    expected_protocol_sha256 = sha256(_canonical_json_bytes(evaluation_protocol)).hexdigest()
    provider_document = document['authoring']['provider']
    case_id = evaluation_protocol['case_id']
    workload_sha256 = document['workload']['canonical_sha256']
    arm = specification.condition_id
    kind = specification.environment_kind
    empirical_enabled = "candidate_selection" in document["authoring"]
    ledger = evidence.start_run(
        run_id,
        authority_sha256=specification.canonical_sha256,
        authority=document,
    )
    ledger.append(
        "run_started",
        {
            "sequence": sequence,
            "assigned_arm": arm,
            "automatic_retries": 0,
            "replacement_run": False,
        },
    )
    protocol_adherence = "adhered"
    thread_id: str | None = None
    cumulative_tokens = 0
    feedback: Mapping[str, object] = MappingProxyType({"kind": "initial"})
    observations: list[TurnObservation] = []
    selected_by_turn = {}
    confirmation = None
    search_state = None
    confirmation_source_turn = None
    live_stage = "provider"
    provider_usage_accounted = False
    provider_turn = None
    ralph = RalphController(
        ralph_budget,
        searches_per_turn=int(evaluation_protocol.get("searches_per_turn", 1)),
        profile_each_search_survivor=profile_each_search_survivor,
        clock=clock,
    )
    ralph_stop_reason: str | None = None
    evaluation_writer = EvaluationWriter(
        evidence=evidence, ledger=ledger, evaluator=evaluator, ralph=ralph,
        case_id=case_id, workload_sha256=workload_sha256,
        protocol_sha256=expected_protocol_sha256, evaluation_protocol=evaluation_protocol,
        execution=document['execution'],
    )
    try:
        for turn_number in range(1, maximum_turns + 1):
            ralph_stop_reason = ralph.stop_reason(
                turn=turn_number,
                cumulative_provider_tokens=cumulative_tokens,
            )
            if ralph_stop_reason is not None:
                break
            state_card = ralph.state_card(
                turn=turn_number,
                cumulative_provider_tokens=cumulative_tokens,
                feedback=feedback,
            )
            live_stage = "provider"
            provider_usage_accounted = False
            provider_turn = None
            authoring_started = ralph.begin_authoring()
            try:
                provider_turn = provider.turn(
                    TurnRequest(
                        run_id,
                        arm,
                        turn_number,
                        cumulative_tokens,
                        thread_id,
                        feedback,
                        maximum_candidates_per_turn,
                        state_card,
                        environment_kind=kind,
                    )
                )
            finally:
                ralph.end_authoring(authoring_started)
            next_thread_id = provider_turn.thread_id
            if thread_id is not None and next_thread_id != thread_id:
                raise differs("provider resume thread identity", expected=thread_id, observed=next_thread_id)
            next_cumulative_tokens = cumulative_tokens + provider_turn.provider_tokens
            _archive_provider_turn(
                arm=arm,
                candidate_media_type=environment.media_type,
                environment_kind=kind,
                cumulative_tokens=next_cumulative_tokens,
                evidence=evidence,
                ledger=ledger,
                maximum_candidates_per_turn=maximum_candidates_per_turn,
                provider_document=provider_document,
                provider_turn=provider_turn,
                thread_id=next_thread_id,
                turn_number=turn_number,
            )
            # Completion owns the cumulative/session commit. A returned Turn
            # can still be refused by archive validation before any build.
            thread_id = next_thread_id
            cumulative_tokens = next_cumulative_tokens
            provider_usage_accounted = True
            # The pre-GPU filter runs on the whole set: every candidate is built,
            # which is the verifier and the toolchain but no device. Only then is
            # an order taken, and only the survivor reaches an Evaluation. This is
            # the stage the paper spends compile time on to avoid spending GPU
            # time, so building all of them is the point rather than a cost.
            live_stage = "environment"
            resolutions = resolve_action_set(provider_turn.candidates,
                environment_kind=kind, transformations=document['knowledge']['transformations'],
                candidates=prior_candidates, baselines=baselines, compiler_factory=compiler_factory,
                allow_python=document["authoring"].get("input_format") == "schedule_or_python_v1")
            action_rows = []
            resolved_candidates = {}
            for ordinal, resolution in enumerate(resolutions):
                row = {'ordinal': ordinal, **resolution.document, 'objects': []}
                if resolution.candidate is not None:
                    obj = evidence.put(resolution.candidate, media_type=environment.media_type)
                    row['objects'] = [obj.reference('resolved_candidate')]
                    resolved_candidates[obj.sha256] = resolution.candidate
                action_rows.append(row)
            ledger.append('author_actions_resolved', {'turn': turn_number, 'actions': action_rows})
            prior_candidates.update(resolved_candidates)
            action_feedback = [{key: value for key, value in row.items() if key != 'objects'} for row in action_rows]
            (
                built,
                launchable_first,
                cost_order_applied,
                filter_rows,
                selection_summary,
            ) = _build_filter_candidates(
                empirical_enabled=empirical_enabled,
                environment=environment,
                ledger=ledger,
                candidate_payloads=tuple(resolved_candidates.values()),
                turn_number=turn_number,
                ralph=ralph,
            )
            record_candidate_rejections(
                built=built, evidence=evidence, ledger=ledger, turn_number=turn_number, arm=kind,
            )
            if not built:
                ledger.append('candidate_selected', {'turn': turn_number, 'candidate_sha256': None,
                    'qualified_search_candidates': [], 'reason': 'no_candidate_produced'})
                observations.append(TurnObservation(turn_number, cumulative_tokens, None, False, None))
                feedback = {'stage': 'authoring', 'author_actions': action_feedback}
                continue
            submission, environment_result = built[launchable_first[0]]
            if environment_result.disposition == "rejected":
                ledger.append(
                    "candidate_selected",
                    {
                        "turn": turn_number,
                        "candidate_sha256": submission.sha256,
                        "qualified_search_candidates": [],
                        "reason": "all_candidates_rejected",
                    },
                )
                observations.append(
                    TurnObservation(
                        turn_number,
                        cumulative_tokens,
                        submission.sha256,
                        False,
                        None,
                    )
                )
                feedback = environment_result.feedback
            else:
                launchable = environment_result.launchable
                assert launchable is not None
                required_roles = _arm_artifact_roles(kind, launchable.target, program=launchable.is_program)
                if (
                    not required_roles <= set(launchable.artifact_roles)
                    or set(launchable.artifact_payloads)
                    != set(launchable.artifact_roles)
                    or launchable.artifact_roles.get("launch_manifest")
                    != launchable.launch_spec_sha256
                ):
                    raise ValueError("LaunchableCandidate artifact custody is incomplete")

                # Search-evaluate the candidates the filter kept, in its order.
                # Search is the assay that exists to choose; confirmatory stays
                # single because that one is the measurement a claim rests on.
                budget_k = int(
                    evaluation_protocol.get("searches_per_turn", 1)
                )
                searched: list[_SearchedCandidate] = []
                planned_searches, collapsed = _matched_search_plan(
                    filter_rows, budget_k
                )
                position_by_candidate = {
                    submission.sha256: index
                    for index, (submission, _) in enumerate(built)
                }
                for candidate_sha256 in planned_searches:
                    position = position_by_candidate[candidate_sha256]
                    entry_submission, entry_result = built[position]
                    entry_launchable = entry_result.launchable
                    assert entry_launchable is not None
                    artifact_references = []
                    for role, payload in sorted(
                        entry_launchable.artifact_payloads.items()
                    ):
                        artifact = evidence.put(
                            payload,
                            media_type=_candidate_artifact_media_type(role),
                        )
                        artifact_references.append(artifact.reference(role))
                    ledger.append(
                        "launchable_candidate_sealed",
                        {
                            "turn": turn_number,
                            "candidate_sha256": entry_launchable.candidate_sha256,
                            "candidate_record_sha256": entry_launchable.canonical_sha256,
                            "objects": artifact_references,
                        },
                    )
                    live_stage = "evaluation"
                    entry_search = evaluation_writer.evaluate(
                        entry_launchable, purpose="search", turn=turn_number,
                    )
                    entry_attribution = (
                        evaluation_writer.evaluate(entry_launchable, purpose="attribution", turn=turn_number)
                        if profile_each_search_survivor
                        and entry_search.correctness_passed
                        else None
                    )
                    searched.append(
                        _SearchedCandidate(
                            entry_submission,
                            entry_result,
                            entry_launchable,
                            entry_search,
                            entry_attribution,
                        )
                    )

                collapse_diagnosis = _collapse_diagnosis(
                    turn_number, collapsed
                )
                if collapse_diagnosis is not None:
                    # Not a measurement's finding, so it does not wait for
                    # materiality: two spellings of one program is a fact about
                    # the set, visible before any of it ran.
                    ledger.append(
                        "diagnosis_routed", collapse_diagnosis
                    )

                # Qualification and order diagnosis share this pure decision in
                # execution and replay, so the gate cannot manufacture a second
                # interpretation of the retained measurements.
                qualified_search, best, cost_diagnosis = (
                    _matched_search_decision(
                        turn_number,
                        [
                            (item.launchable.candidate_sha256, item.receipt)
                            for item in searched
                        ],
                        cost_order_applied=cost_order_applied,
                        materiality_ratio=float(
                            evaluation_protocol.get(
                                "search_materiality_ratio", math.inf
                            )
                        ),
                    )
                )
                if cost_diagnosis is not None:
                    ledger.append("diagnosis_routed", cost_diagnosis)
                selected = searched[best]
                submission = selected.submission
                environment_result = selected.environment_result
                launchable = selected.launchable
                search = selected.receipt
                ledger.append(
                    "candidate_selected",
                    {
                        "turn": turn_number,
                        "candidate_sha256": launchable.candidate_sha256,
                        "qualified_search_candidates": [
                            searched[index].launchable.candidate_sha256
                            for index in qualified_search
                        ],
                        "reason": (
                            "lowest_qualified_search_latency"
                            if qualified_search
                            else "no_qualified_search_candidate"
                        ),
                    },
                )

                qualified = bool(qualified_search)
                selected_by_turn[turn_number] = selected
                attribution = selected.attribution
                observations.append(TurnObservation(turn_number, cumulative_tokens,
                    launchable.candidate_sha256, qualified,
                    _receipt_latency_ms(search) if qualified else None))
                # A measurement says what this candidate cost; the Environment's
                # surviving findings say which declared resource is what bounds
                # it. Only the pair is actionable, so the next Turn gets both.
                feedback_document: dict[str, object] = {
                    "kind": "evaluation",
                    "candidate_disposition": search.candidate_disposition,
                    "measurement_quality": search.measurement_quality,
                    "search_qualified": qualified,
                    "search_latency_ms": _receipt_latency_ms(search),
                    "findings": environment_result.feedback.get("findings", []),
                }
                if isinstance(search.timing, Mapping):
                    feedback_document["baseline_comparison"] = (
                        _baseline_comparison_feedback(specification, search.timing)
                    )
                if "attribution_evaluation" in evaluation_protocol:
                    attribution_feedback = (
                        attribution.attribution_feedback
                        if attribution is not None
                        else None
                    )
                    feedback_document["profile"] = (
                        dict(attribution_feedback)
                        if attribution_feedback is not None
                        else None
                    )
                feedback = MappingProxyType(feedback_document)
            if any(row["kind"] != "submit" or row["action_sha256"] != row["candidate_sha256"] for row in action_rows):
                feedback = MappingProxyType({**feedback, "author_actions": action_feedback})
            rejected_peers = rejected_peer_feedback(built, arm=kind)
            if rejected_peers:
                feedback = MappingProxyType({**feedback, "rejected_candidates": rejected_peers})
            if empirical_enabled:
                feedback = MappingProxyType({
                    **feedback,
                    "candidate_selection": {**selection_summary, "order": filter_rows},
                })
            if cumulative_tokens >= cast(int, budget["limit"]):
                break
        # Search closes before nomination; no author/build activity follows this.
        search_state = dict(ralph.complete_search(
            turn=min(maximum_turns + 1, len(observations) + 1),
            cumulative_provider_tokens=cumulative_tokens, feedback=feedback))
        ralph_stop_reason = search_state['terminal_reason']
        ledger.append('search_completed', {'state': search_state})
        nominee = nominate(observations, provider_token_limit=budget['limit'])
        selected = selected_by_turn[nominee.turn] if nominee else None
        ledger.append('candidate_nominated', nomination_document(nominee, selected.launchable if selected else None))
        if nominee is not None:
            confirmation_source_turn = nominee.turn
            live_stage = 'evaluation'
            confirmed = evaluation_writer.evaluate(selected.launchable, purpose='confirmatory', source_turn=nominee.turn)
            qualified = _receipt_qualifies(confirmed)
            confirmation = FinalConfirmation(nominee.turn, nominee.candidate_sha256, cumulative_tokens,
                qualified, _receipt_latency_ms(confirmed) if qualified else None)
            if qualified and attribution_evaluation == _LEGACY_ATTRIBUTION_EVALUATION:
                evaluation_writer.evaluate(selected.launchable, purpose='attribution', source_turn=nominee.turn)
    except Exception as error:
        pending_usage = live_stage == "provider" and not provider_usage_accounted
        observed_usage = None
        declared_usage = None
        payloads = dict(error.artifact_payloads) if isinstance(error, RunProtocolFault) else {}
        if pending_usage:
            declared_usage = error.reported_usage if isinstance(error, RunProtocolFault) else None
            if declared_usage is not None:
                try:
                    declared_usage = replace(declared_usage, provider_tokens=provider_token_delta(
                        declared_usage.provider_tokens, provider=provider_document,
                        previous_tokens=cumulative_tokens))
                except ValueError:
                    declared_usage = None
            if provider_turn is not None:
                # Preserve the native statement even when the returned Turn's
                # bundle, identity or candidate envelope failed validation.
                raw_events = getattr(provider_turn, "raw_events", None)
                if isinstance(raw_events, bytes):
                    payloads.setdefault("provider_stdout", raw_events)
                if declared_usage is None:
                    try:
                        declared_usage = ReportedProviderUsage(
                            provider_document.get("event_contract", "closed_file_change_v1"),
                            provider_turn.thread_id, provider_turn.provider_tokens)
                    except (AttributeError, TypeError, ValueError):
                        declared_usage = None
            observed_usage = reported_provider_usage(payloads.get("provider_stdout", b""),
                provider=provider_document, expected_thread_id=thread_id,
                previous_tokens=cumulative_tokens)
            if observed_usage is not None:
                cumulative_tokens += observed_usage.provider_tokens
        fault = record_run_fault(
            error=error,
            live_stage=live_stage,
            turn_number=turn_number,
            source_turn=confirmation_source_turn,
            cumulative_tokens=cumulative_tokens,
            evidence=evidence,
            ledger=ledger,
            pending_provider_usage=pending_usage,
            observed_usage=observed_usage,
            declared_usage=declared_usage,
            artifact_payloads=payloads,
        )
        protocol_adherence = fault
        ralph_stop_reason = fault

    _seal_run(
        ralph_stop_reason=ralph_stop_reason,
        checkpoints=checkpoints,
        cumulative_tokens=cumulative_tokens,
        feedback=feedback,
        ledger=ledger,
        maximum_turns=maximum_turns,
        observations=observations,
        protocol_adherence=protocol_adherence,
        ralph=ralph,
        analysis=specification.terminal_policy,
        confirmation=confirmation, search_state=search_state,
    )
