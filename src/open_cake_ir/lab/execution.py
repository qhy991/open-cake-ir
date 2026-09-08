"""Live Campaign execution, budget accounting and evidence writing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, cast

from open_cake_ir.evaluation import EvaluationReceipt, LaunchableCandidate
from open_cake_ir.evidence import EvidenceStore

from ._documents import _canonical_json_bytes, _digest, _name, _object
from ._policies import (
    _ATTRIBUTION_EVALUATION,
    _LEGACY_ATTRIBUTION_EVALUATION,
    _matched_evidence_policy_version,
)
from .archive import _ARM_ARTIFACT_ROLES, _archive_provider_turn, _candidate_artifact_media_type
from .evaluation_writer import EvaluationWriter
from .execution_admission import validate_execution_bindings
from .candidate_filter import _build_filter_candidates, record_candidate_rejections
from .run_completion import _seal_run, record_run_fault
from .checkpoints import TurnObservation
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
    comparison_arm(environments)
    if set(environments) != set(lock.document["resolved_inputs"]["arm_environments"]):
        raise ValueError("Campaign Authoring Environment set differs")
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
    budget = _object(resolved_inputs["budget"], "campaign_lock.resolved_inputs.budget")
    checkpoints = cast(list[int], budget["checkpoints"])
    maximum_turns = cast(int, budget["maximum_turns"])
    maximum_candidates_per_turn = cast(
        int, budget.get("maximum_candidates_per_turn", 1)
    )
    evaluation_protocol = _object(
        lock.document["evaluation_protocol"], "campaign_lock.evaluation_protocol"
    )
    attribution_evaluation = evaluation_protocol.get("attribution_evaluation")
    profile_each_search_survivor = (
        attribution_evaluation == _ATTRIBUTION_EVALUATION
    )
    ralph_budget = RalphBudget.from_mapping(budget)
    expected_protocol_sha256 = sha256(
        _canonical_json_bytes(evaluation_protocol)
    ).hexdigest()
    arms, provider_document = validate_execution_bindings(
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
    case_id = _name(evaluation_protocol.get("case_id"), "evaluation_protocol.case_id")
    workload_sha256 = _digest(
        _object(lock.document["workload"], "campaign_lock.workload").get(
            "canonical_sha256"
        ),
        "campaign_lock.workload.canonical_sha256",
    )
    evidence = EvidenceStore.create(root)
    record_confirmation_time = native_backend(comparison_arm(environments)) is not None
    for sequence, run_id in enumerate(lock.run_order, start=1):
        run_started_at = clock() if record_confirmation_time else None
        arm = run_id.rsplit("-", 1)[0]
        environment = environments[arm]
        empirical_enabled = "candidate_selection" in arms[arm]
        ledger = evidence.start_run(
            run_id,
            authority_sha256=lock.canonical_sha256,
            authority=lock.document,
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
        live_stage = "provider"
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
            execution=lock.document['execution'],
            clock=clock, run_started_at=run_started_at,
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
                        )
                    )
                finally:
                    ralph.end_authoring(authoring_started)
                if thread_id is not None and provider_turn.thread_id != thread_id:
                    raise ValueError("provider resume thread identity differs")
                thread_id = provider_turn.thread_id
                cumulative_tokens += provider_turn.provider_tokens
                _archive_provider_turn(
                    arm=arm,
                    candidate_media_type=environment.media_type,
                    cumulative_tokens=cumulative_tokens,
                    evidence=evidence,
                    ledger=ledger,
                    maximum_candidates_per_turn=maximum_candidates_per_turn,
                    provider_document=provider_document,
                    provider_turn=provider_turn,
                    thread_id=thread_id,
                    turn_number=turn_number,
                )
                # The pre-GPU filter runs on the whole set: every candidate is built,
                # which is the verifier and the toolchain but no device. Only then is
                # an order taken, and only the survivor reaches an Evaluation. This is
                # the stage the paper spends compile time on to avoid spending GPU
                # time, so building all of them is the point rather than a cost.
                live_stage = "environment"
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
                    provider_turn=provider_turn,
                    turn_number=turn_number,
                )
                record_candidate_rejections(
                    built=built, evidence=evidence, ledger=ledger, turn_number=turn_number,
                )
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
                    required_roles = _ARM_ARTIFACT_ROLES[arm]
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

                    confirmed: EvaluationReceipt | None = None
                    if qualified_search:
                        confirmed = evaluation_writer.evaluate(
                            launchable, purpose="confirmatory", turn=turn_number,
                        )
                    qualified = confirmed is not None and _receipt_qualifies(confirmed)
                    latency = _receipt_latency_ms(confirmed) if qualified else None
                    # The current assay already profiled every correctness-passing
                    # search survivor. The selected profile is feedback, not an
                    # acceptance input. Frozen Studies retain the earlier
                    # selected-after-confirmation operation at this compatibility
                    # edge.
                    attribution = selected.attribution
                    if (
                        not profile_each_search_survivor
                        and qualified
                        and attribution_evaluation
                        == _LEGACY_ATTRIBUTION_EVALUATION
                    ):
                        attribution = evaluation_writer.evaluate(launchable, purpose="attribution", turn=turn_number)
                    observations.append(
                        TurnObservation(
                            turn_number,
                            cumulative_tokens,
                            launchable.candidate_sha256,
                            qualified,
                            latency,
                        )
                    )
                    # A measurement says what this candidate cost; the Environment's
                    # surviving findings say which declared resource is what bounds
                    # it. Only the pair is actionable, so the next Turn gets both.
                    feedback_document: dict[str, object] = {
                        "kind": "evaluation",
                        "candidate_disposition": search.candidate_disposition,
                        "measurement_quality": search.measurement_quality,
                        "confirmed": qualified,
                        "search_latency_ms": _receipt_latency_ms(search),
                        "confirmed_latency_ms": latency,
                        "findings": environment_result.feedback.get("findings", []),
                    }
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
                if empirical_enabled:
                    feedback = MappingProxyType({
                        **feedback,
                        "candidate_selection": {**selection_summary, "order": filter_rows},
                    })
                if cumulative_tokens >= cast(int, budget["limit"]):
                    break
        except Exception as error:
            fault = record_run_fault(
                error=error,
                live_stage=live_stage,
                turn_number=turn_number,
                cumulative_tokens=cumulative_tokens,
                evidence=evidence,
                ledger=ledger,
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
        )

    return CampaignRef(lock=lock, evidence_root=evidence.root)
