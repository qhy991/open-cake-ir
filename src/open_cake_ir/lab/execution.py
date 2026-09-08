"""Live Campaign execution, budget accounting and evidence writing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence, cast

from open_cake_ir.evaluation import EvaluationReceipt, LaunchableCandidate
from open_cake_ir.evaluation.paired import paired_protocol
from open_cake_ir.evidence import EvidenceStore
from open_cake_ir.evidence.store import RunLedger

from ._documents import _canonical_json_bytes, _digest, _name, _object
from ._policies import (
    _ATTRIBUTION_EVALUATION,
    _LEGACY_ATTRIBUTION_EVALUATION,
    _matched_evidence_policy_version,
)
from .archive import (
    _ARM_ARTIFACT_ROLES,
    _archive_evaluation_receipt,
    _archive_logical_attempt,
    _archive_provider_turn,
    _candidate_artifact_media_type,
    _validate_receipt_authority,
)
from .bindings import qualification_path as _qualification_path
from .checkpoints import TurnObservation, project_checkpoints
from .contracts import CampaignLock, CampaignRef, RunEvaluator, RunProvider, TurnRequest
from .custody import admit_new_campaign_path
from .environments import AuthoringEnvironment, CandidateSubmission, EnvironmentResult
from .executor import ExecutorRevision
from .faults import RunProtocolFault
from .pairing import comparison_arm, native_backend
from .providers import (
    CANDIDATE_SET_ENVELOPE_V1,
    CODEX_DISABLED_FEATURES,
    ProviderQualificationReceipt,
    ProviderTurn,
)
from .ralph import RalphBudget, RalphController
from .routing import route_rejection
from .selection import (
    _collapse_diagnosis,
    _empirical_filter,
    _matched_endpoint_from_checkpoint,
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
    if (
        getattr(evaluator, "protocol", None) != evaluation_protocol
        or getattr(evaluator, "protocol_sha256", None) != expected_protocol_sha256
    ):
        raise ValueError("Run Evaluator does not match the Campaign Lock")
    arms = _object(resolved_inputs["arm_environments"], "resolved_inputs.arm_environments")
    validate_authoring(workload_loader(project_root / str(lock.document["workload"]["path"])), arms)
    arm_hashes = _object(
        resolved_inputs["arm_environment_sha256"],
        "resolved_inputs.arm_environment_sha256",
    )
    provider_documents = {
        name: _object(
            _object(arms[name], f"arm_environments.{name}").get("provider"),
            f"arm_environments.{name}.provider",
        )
        for name in environments
    }
    provider_revisions = {
        _name(value.get("revision"), f"arm_environments.{name}.provider.revision")
        for name, value in provider_documents.items()
    }
    qualification_digests = {
        _digest(
            _object(
                value.get("qualification"),
                f"arm_environments.{name}.provider.qualification",
            ).get("canonical_sha256"),
            f"arm_environments.{name}.provider.qualification.sha256",
        )
        for name, value in provider_documents.items()
    }
    if (
        provider_revisions != {getattr(provider, "provider_revision", None)}
        or qualification_digests != {getattr(provider, "qualification_sha256", None)}
    ):
        raise ValueError("Run Provider does not match the Campaign Lock")
    first_arm = _object(arms["open_cake"], "arm_environments.open_cake")
    provider_document = _object(
        first_arm["provider"], "arm_environments.open_cake.provider"
    )
    qualification_ref = _object(
        provider_document["qualification"],
        "arm_environments.open_cake.provider_qualification",
    )
    _, qualification_path = _qualification_path(
        project_root,
        qualification_ref["path"],
        "arm_environments.open_cake.provider_qualification.path",
    )
    qualification = ProviderQualificationReceipt.load(qualification_path)
    if (native_backend(comparison_arm(arms)) is not None and qualification.scope != 'zero_gpu_contract_fixture_only'
        and (paired_protocol(evaluation_protocol) is None
             or provider_document['disabled_features'] != list(CODEX_DISABLED_FEATURES))):
        raise ValueError('new live native execution requires paired policy and current closed provider surface')
    if getattr(provider, "executable_sha256", None) != qualification.executable_sha256:
        raise ValueError("Run Provider executable does not match its qualification")
    output_schema = _object(
        provider_document["output_schema"],
        "arm_environments.open_cake.provider.output_schema",
    )
    expected_provider_configuration = {
        "model": provider_document["model"],
        "reasoning_effort": provider_document["reasoning_effort"],
        "service_tier": provider_document["service_tier"],
        "output_schema_sha256": output_schema["sha256"],
        "removed_environment": provider_document["removed_environment"],
        "sandbox": provider_document["sandbox"],
        "cwd_policy": provider_document["cwd_policy"],
        "reference_visibility": provider_document["reference_visibility"],
        "disabled_features": provider_document["disabled_features"],
        "code_mode_host": provider_document["code_mode_host"],
    }
    if "web_search" in provider_document:
        expected_provider_configuration["web_search"] = provider_document["web_search"]
    if "event_contract" in provider_document:
        expected_provider_configuration["event_contract"] = provider_document[
            "event_contract"
        ]
    expected_provider_configuration[
        "submission_contract"
    ] = CANDIDATE_SET_ENVELOPE_V1
    if (
        getattr(provider, "configuration", None) != expected_provider_configuration
        or qualification.canonical_sha256
        != qualification_ref["canonical_sha256"]
        or qualification.configuration_sha256
        != sha256(
            _canonical_json_bytes(expected_provider_configuration)
        ).hexdigest()
    ):
        raise ValueError("Run Provider configuration does not match the Campaign Lock")
    for name, environment in environments.items():
        if (
            getattr(environment, "authority_document", None) != arms[name]
            or getattr(environment, "canonical_sha256", None) != arm_hashes[name]
        ):
            raise ValueError(f"{name} Authoring Environment does not match the Campaign Lock")
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
        ralph = (
            RalphController(
                cast(RalphBudget, ralph_budget),
                searches_per_turn=int(
                    evaluation_protocol.get("searches_per_turn", 1)
                ),
                profile_each_search_survivor=profile_each_search_survivor,
                clock=clock,
            )
        )
        ralph_stop_reason: str | None = None
        try:
            for turn_number in range(1, maximum_turns + 1):
                state_card = None
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
                authoring_started = (
                    ralph.begin_authoring()
                )
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
                for rejected_submission, rejected_result in built:
                    if rejected_result.disposition != "rejected":
                        continue
                    decision = route_rejection(rejected_result.feedback)
                    rejection_payload: dict[str, object] = {
                        "turn": turn_number,
                        "candidate_sha256": rejected_submission.sha256,
                        "feedback": dict(rejected_result.feedback),
                        "routed_to": decision.destination,
                        "routing_reason": decision.reason,
                    }
                    if rejected_result.artifact_payloads:
                        references = []
                        rejected_roles = []
                        for role, payload in sorted(
                            rejected_result.artifact_payloads.items()
                        ):
                            try:
                                references.append(
                                    evidence.put(
                                        payload,
                                        media_type=_candidate_artifact_media_type(
                                            role.rsplit("_", 1)[-1]
                                        ),
                                    ).reference(role)
                                )
                            except (OSError, ValueError):
                                rejected_roles.append(role)
                        if references:
                            rejection_payload["objects"] = references
                        if rejected_roles:
                            rejection_payload["artifact_rejections"] = rejected_roles
                    ledger.append("candidate_rejected", rejection_payload)
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

                    def evaluate_attribution(
                        candidate: LaunchableCandidate,
                    ) -> EvaluationReceipt:
                        ralph.record_evaluation("attribution")
                        attempt = evaluator.evaluate(
                            candidate,
                            case_id=case_id,
                            purpose="attribution",
                        )
                        ledger.append(
                            "evaluation_attempt_completed",
                            {
                                "turn": turn_number,
                                "purpose": "attribution",
                                "candidate_sha256": candidate.candidate_sha256,
                                "objects": _archive_logical_attempt(
                                    evidence, attempt
                                ),
                            },
                        )
                        receipt = attempt.final_receipt
                        if receipt is None:
                            raise RuntimeError(
                                "attribution Evaluation has no final receipt"
                            )
                        _validate_receipt_authority(
                            receipt,
                            candidate=candidate,
                            workload_sha256=workload_sha256,
                            protocol_sha256=expected_protocol_sha256,
                            case_id=case_id,
                            purpose="attribution",
                            evaluation_protocol=evaluation_protocol,
                            fixed_baseline=lock.document['execution'].get('fixed_baseline', {}).get('candidate'),
                        )
                        ledger.append(
                            "candidate_evaluated",
                            {
                                "turn": turn_number,
                                "purpose": "attribution",
                                "candidate_sha256": candidate.candidate_sha256,
                                "objects": _archive_evaluation_receipt(
                                    evidence, receipt
                                ),
                            },
                        )
                        return receipt

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
                        ralph.record_evaluation("search")
                        entry_attempt = evaluator.evaluate(
                            entry_launchable,
                            case_id=case_id,
                            purpose="search",
                        )
                        entry_search = entry_attempt.final_receipt
                        ledger.append(
                            "evaluation_attempt_completed",
                            {
                                "turn": turn_number,
                                "purpose": "search",
                                "candidate_sha256": entry_launchable.candidate_sha256,
                                "objects": _archive_logical_attempt(
                                    evidence, entry_attempt
                                ),
                            },
                        )
                        if entry_search is None:
                            raise RuntimeError("search Evaluation has no final receipt")
                        _validate_receipt_authority(
                            entry_search,
                            candidate=entry_launchable,
                            workload_sha256=workload_sha256,
                            protocol_sha256=expected_protocol_sha256,
                            case_id=case_id,
                            purpose="search",
                            evaluation_protocol=evaluation_protocol,
                            fixed_baseline=lock.document['execution'].get('fixed_baseline', {}).get('candidate'),
                        )
                        ledger.append(
                            "candidate_evaluated",
                            {
                                "turn": turn_number,
                                "purpose": "search",
                                "candidate_sha256": entry_launchable.candidate_sha256,
                                "objects": _archive_evaluation_receipt(
                                    evidence, entry_search
                                ),
                            },
                        )
                        entry_attribution = (
                            evaluate_attribution(entry_launchable)
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
                        ralph.record_evaluation("confirmatory")
                        confirmed_attempt = evaluator.evaluate(
                            launchable,
                            case_id=case_id,
                            purpose="confirmatory",
                        )
                        confirmed = confirmed_attempt.final_receipt
                        confirmed_attempt_references = _archive_logical_attempt(
                            evidence, confirmed_attempt
                        )
                        ledger.append(
                            "evaluation_attempt_completed",
                            {
                                "turn": turn_number,
                                "purpose": "confirmatory",
                                "candidate_sha256": launchable.candidate_sha256,
                                "objects": confirmed_attempt_references,
                            },
                        )
                        if confirmed is None:
                            raise RuntimeError("confirmatory Evaluation has no final receipt")
                        _validate_receipt_authority(
                            confirmed,
                            candidate=launchable,
                            workload_sha256=workload_sha256,
                            protocol_sha256=expected_protocol_sha256,
                            case_id=case_id,
                            purpose="confirmatory",
                            evaluation_protocol=evaluation_protocol,
                            fixed_baseline=lock.document['execution'].get('fixed_baseline', {}).get('candidate'),
                        )
                        confirmed_references = _archive_evaluation_receipt(
                            evidence, confirmed
                        )
                        ledger.append(
                            "candidate_evaluated",
                            {
                                "turn": turn_number,
                                "purpose": "confirmatory",
                                "candidate_sha256": launchable.candidate_sha256,
                                "objects": confirmed_references,
                                **({"elapsed_wall_seconds": clock() - run_started_at}
                                   if run_started_at is not None else {}),
                            },
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
                        attribution = evaluate_attribution(launchable)
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
            fault = (
                error.protocol_adherence
                if isinstance(error, RunProtocolFault)
                else {
                    "provider": "provider_fault",
                    "environment": "harness_fault",
                    "evaluation": "broker_fault",
                }[live_stage]
            )
            fault_payload: dict[str, object] = {
                "fault": fault,
                "exception_type": type(error).__name__,
                "turn": turn_number,
                "stage": live_stage,
                "terminal_provider_tokens": cumulative_tokens,
            }
            if isinstance(error, RunProtocolFault) and error.artifact_payloads:
                references = []
                rejected_roles = []
                for role, payload in sorted(error.artifact_payloads.items()):
                    try:
                        references.append(
                            evidence.put(payload, media_type="text/plain").reference(role)
                        )
                    except (OSError, ValueError):
                        rejected_roles.append(role)
                if references:
                    fault_payload["objects"] = references
                if rejected_roles:
                    fault_payload["artifact_rejections"] = rejected_roles
            ledger.append("run_fault", fault_payload)
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


def _build_filter_candidates(
    *,
    empirical_enabled: bool,
    environment: AuthoringEnvironment,
    ledger: RunLedger,
    provider_turn: ProviderTurn,
    turn_number: int,
) -> tuple[
    list[tuple[CandidateSubmission, EnvironmentResult]],
    list[int],
    bool,
    list[dict[str, object]],
    dict[str, object] | None,
]:
    built = []
    for payload in provider_turn.candidates:
        entry = CandidateSubmission.seal(environment.media_type, payload)
        built.append((entry, environment.build(entry)))
    launchable_first = [
        index
        for index, (_, result) in enumerate(built)
        if result.disposition == "launchable"
    ]
    # A partial order is not an order over the candidate set. If the
    # model declines any launchable member, moving that unknown behind
    # scored members would let `searches_per_turn` silently reject it as
    # slower. Apply the cost order only when it covers the whole
    # launchable set; otherwise every member keeps provider order.
    cost_order_applied = bool(launchable_first) and all(
        built[index][1].cost is not None
        for index in launchable_first
    )
    if cost_order_applied and not empirical_enabled:
        def complete_cost_order(index: int) -> tuple[tuple, int]:
            cost = built[index][1].cost
            assert cost is not None
            return cost.order, index

        launchable_first.sort(key=complete_cost_order)
    launchable_first.extend(
        index
        for index, (_, result) in enumerate(built)
        if result.disposition != "launchable"
    )
    filter_rows = [
        {
            "candidate_sha256": built[index][0].sha256,
            "disposition": built[index][1].disposition,
            "cost": (
                {
                    "device_fill": round(
                        built[index][1].cost.device_fill, 6
                    ),
                    "binding_resource": built[
                        index
                    ][1].cost.binding_resource,
                }
                if built[index][1].cost is not None
                else None
            ),
            "semantic_sha256": built[index][1].semantic_sha256,
            **({"empirical_cost": (
                dict(built[index][1].empirical_cost)
                if built[index][1].empirical_cost is not None else None
            )} if empirical_enabled else {}),
        }
        for index in (range(len(built)) if empirical_enabled else launchable_first)
    ]
    selection_summary = None
    if empirical_enabled:
        filter_rows, selection_summary = _empirical_filter(filter_rows)
        cost_order_applied = selection_summary["order_applied"]
        by_submission = {entry.sha256: index for index, (entry, _) in enumerate(built)}
        launchable_first = [by_submission[row["candidate_sha256"]] for row in filter_rows]
    ledger.append(
        "candidate_set_filtered",
        {
            "turn": turn_number,
            "submitted": len(built),
            "launchable": sum(
                result.disposition == "launchable"
                for _, result in built
            ),
            "order": filter_rows,
            **({"candidate_selection": selection_summary} if empirical_enabled else {}),
        },
    )
    # A rejected member remains evidence even when another member is
    # launchable. Otherwise the archive would retain only a disposition
    # bit and lose the concrete feedback needed to improve the next set.
    return built, launchable_first, cost_order_applied, filter_rows, selection_summary


def _seal_run(
    *,
    ralph_stop_reason: str | None,
    checkpoints: Sequence[int],
    cumulative_tokens: int,
    feedback: Mapping[str, object],
    ledger: RunLedger,
    maximum_turns: int,
    observations: Sequence[TurnObservation],
    protocol_adherence: str,
    ralph: RalphController,
) -> None:
    if ralph_stop_reason is None:
        ralph_stop_reason = ralph.stop_reason(
            turn=min(maximum_turns + 1, len(observations) + 1),
            cumulative_provider_tokens=cumulative_tokens,
        ) or "maximum_turns"

    projected = project_checkpoints(
        turns=observations,
        checkpoints=checkpoints,
        terminal_provider_tokens=cumulative_tokens,
    )
    checkpoint_payload: dict[str, object] = {
        "checkpoints": [
                {
                    "provider_tokens": item.provider_tokens,
                    "state": item.state,
                    "best_candidate_sha256": item.best_candidate_sha256,
                    "best_confirmed_latency_ms": item.best_confirmed_latency_ms,
                }
                for item in projected
            ]
    }
    checkpoint_payload["ralph"] = dict(
        ralph.state_card(
            turn=min(maximum_turns + 1, len(observations) + 1),
            cumulative_provider_tokens=cumulative_tokens,
            feedback=feedback,
            terminal_reason=ralph_stop_reason,
        )
    )
    ledger.append("checkpoints_projected", checkpoint_payload)
    final_checkpoint = projected[-1]
    endpoint_observation, endpoint = _matched_endpoint_from_checkpoint(
        final_checkpoint, protocol_adherence
    )
    ledger.seal(
        protocol_adherence=protocol_adherence,
        endpoint_observation=endpoint_observation,
        endpoint=endpoint,
    )
