"""Study contracts and pure Campaign preflight resolution."""

from __future__ import annotations

import json
import math
import re
import statistics
import time
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, cast

from open_cake_ir.compiler import Compiler, CorpusGateReport
from open_cake_ir.compiler.empirical_cost import EmpiricalCostModel
from open_cake_ir.evaluation import EvaluationReceipt, LaunchableCandidate, LogicalEvaluationAttempt
from open_cake_ir.evaluation.core import _plain_json as _evaluation_plain_json
from open_cake_ir.evidence import EvidenceStore, RunAudit

from .checkpoints import TurnObservation, project_checkpoints
from .custody import admit_new_campaign_path
from .bindings import (
    qualification_path as _qualification_path, resolve_execution_bindings, load_baseline_bundle,
    resolve_executor, CURRENT_RELEASE_BINDING as _CURRENT_RELEASE_BINDING,
)
from .environments import (
    AuthoringEnvironment, CandidateSubmission, EnvironmentResult,
    _EMPIRICAL_SELECTION, _EmpiricalSelection, _empirical_context,
)
from .executor import ExecutorRevision
from .faults import RunProtocolFault
from .routing import CANDIDATE, COST_MODEL, route_rejection
from open_cake_ir.evaluation.paired import paired_protocol, validate_receipt_policy, candidate_from_identity, validate_paired_broker

from .pairing import comparison_arm, bind_baseline, native_baseline, triton_optimization_analysis_plan
from open_cake_ir.compiler.target import cuda_target
from .providers import (
    CANDIDATE_SET_ENVELOPE_V1,
    CODEX_DISABLED_FEATURES,
    ProviderQualificationReceipt,
    ProviderTurn,
    _project_candidate_submission,
    parse_codex_turn_events,
    required_live_provider_qualification_scope,
)
from .ralph import RalphBudget, RalphController, derive_ralph_stop_reason
from .task_package import TASK_AGENTS_RALPH_V1, render_task_package

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ARM_ARTIFACT_ROLES = {
    "open_cake": {
        "lowered_source",
        "compiler_expanded_source",
        "ptx",
        "cubin",
        "launch_manifest",
    },
    "direct_cuda": {"authored_source", "ptx", "cubin", "sass", "launch_manifest"},
    "native_triton": {"authored_source", "compiler_expanded_source", "ptx", "cubin", "launch_manifest"},
}
_RALPH_STUDY_FIELDS = {
    "schema_version",
    "study_id",
    "state",
    "kind",
    "claim_scope",
    "workload",
    "arms",
    "agent_interface",
    "allocation",
    "budget",
    "run_protocol",
    "evaluation_protocol",
    "execution",
    "analysis_plan",
    "evidence",
}
_PORTFOLIO_STUDY_FIELDS = {
    "schema_version",
    "study_id",
    "state",
    "kind",
    "claim_scope",
    "workload",
    "compiler_revision",
    "kernel_seed",
    "case_roles",
    "specialization_policy",
    "dispatch_policy",
    "evaluation_protocol",
    "execution",
    "analysis_plan",
    "evidence",
}
_SYSTEM_QUALIFICATION_ANALYSIS_PLAN = {
    "experimental_unit": "run",
    "estimand": None,
    "qualification_criterion": (
        "all_prescheduled_runs_complete_integrity_semantic_replay_"
        "adhered_and_evaluated"
    ),
    "comparative_statistics": "forbidden",
    "pooling": "forbidden",
}
_ARTIFACT_OPTIMIZATION_ANALYSIS_PLAN = {
    "experimental_unit": "run",
    "estimand": None,
    "artifact_promotion": {
        "scope": "per_run",
        "eligibility": "adhered_and_confirmatory_receipt_qualifies",
        "rank": "lowest_confirmed_latency_ms",
        "tie_break": "earliest_turn",
    },
    "comparative_statistics": "forbidden",
    "pooling": "forbidden",
    "scientific_inclusion": "forbidden",
}
_MATCHED_RALPH_EVENT_VOCABULARY_V1 = "matched_ralph_v1"
_MATCHED_EVENT_KINDS_V1 = frozenset(
    {
        "run_started",
        "provider_turn_completed",
        "candidate_set_filtered",
        "candidate_rejected",
        "launchable_candidate_sealed",
        "evaluation_attempt_completed",
        "candidate_evaluated",
        "diagnosis_routed",
        "candidate_selected",
        "run_fault",
        "checkpoints_projected",
        "run_terminal",
    }
)
_MATCHED_RALPH_EVIDENCE_POLICY_V1 = {
    "schema_version": 3,
    "terminal_archive_required_for_every_run": True,
    "event_vocabulary": _MATCHED_RALPH_EVENT_VOCABULARY_V1,
}
_SCIENTIFIC_MATCHED_ANALYSIS_PLAN_V2 = {
    "experimental_unit": "run",
    "target_population": "prescheduled_runs_under_exact_campaign_lock",
    "primary_endpoint": [
        "qualified_by_budget",
        "best_confirmed_latency_ms_if_qualified",
    ],
    "contrast": "two_part_open_cake_vs_direct_cuda",
    "estimand": (
        "terminal-budget qualification-rate difference and conditional confirmed "
        "performance"
    ),
    "missingness": {
        "candidate_failure": "observed_outcome",
        "external_fault": "missing",
        "replacement": "forbidden",
    },
    "pooling": "forbidden_without_successor_analysis_plan",
    "availability": "all_prescheduled_runs_observed_and_each_arm_has_qualified_run",
    "summary_statistics": {
        "qualification": "arm_rate",
        "qualification_contrast": "open_cake_rate_minus_direct_cuda_rate",
        "conditional_latency": "arm_median_ms",
        "contrast": "direct_cuda_median_divided_by_open_cake_median",
        "uncertainty": "per_arm_observed_range_ms",
    },
    "direction": "lower_latency_is_better",
}
_MATCHED_CLAIM_SCOPES = {
    "system_qualification_only",
    "artifact_optimization_only",
    "scientific_matched_search",
}
_ONE_RUN_PER_ARM_SCOPES = {
    "system_qualification_only",
    "artifact_optimization_only",
}
_LEGACY_ATTRIBUTION_EVALUATION = "correctness_then_profile"
_ATTRIBUTION_EVALUATION = "correctness_then_profile_each_search_survivor"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return cast(Mapping[str, object], value)


def _name(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


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


def _digest(value: object, context: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA256 digest")
    return value


def scientific_matched_analysis_plan_v2() -> Mapping[str, object]:
    """Return the sole current two-part scientific Analysis Plan projection."""

    return cast(
        Mapping[str, object],
        json.loads(_canonical_json_bytes(_SCIENTIFIC_MATCHED_ANALYSIS_PLAN_V2)),
    )


def _matched_evidence_policy_version(
    policy: Mapping[str, object], context: str
) -> str:
    """Admit only the Ralph evidence contract."""

    if policy == _MATCHED_RALPH_EVIDENCE_POLICY_V1:
        return _MATCHED_RALPH_EVENT_VOCABULARY_V1
    raise ValueError(f"{context} is unsupported")


def _scientific_analysis_plan_version(
    analysis: Mapping[str, object], context: str
) -> str:
    """Admit the current scientific plans for the two supported comparisons."""

    if analysis == triton_optimization_analysis_plan():
        return "triton_optimization_v1"
    if analysis == _SCIENTIFIC_MATCHED_ANALYSIS_PLAN_V2:
        return "two_part_v2"
    raise ValueError(f"{context} is unsupported")


def _project_path(root: Path, value: object, context: str) -> tuple[str, Path]:
    relative = _name(value, context)
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or ".." in parsed.parts or "\\" in relative:
        raise ValueError(f"{context} is unsafe")
    path = (root / relative).resolve(strict=True)
    if root not in path.parents:
        raise ValueError(f"{context} escapes project root")
    return relative, path


def _resolve_compiler_reference(
    root: Path,
    value: object,
    context: str,
    *,
    template: bool,
) -> tuple[CorpusGateReport, str, dict[str, object]]:
    """Resolve a template binding or verify one frozen Compiler reference."""

    reference = _object(value, context)
    if template:
        if reference != _CURRENT_RELEASE_BINDING:
            raise ValueError("Study template Compiler binding differs")
        relative = "compiler/revision.lock.json"
        path = (root / relative).resolve(strict=True)
    else:
        if reference == _CURRENT_RELEASE_BINDING:
            raise ValueError("frozen Study cannot follow the current Compiler")
        if set(reference) not in (
            {"path", "canonical_sha256"},
            {"path", "canonical_sha256", "revision_id"},
        ):
            raise ValueError("Compiler Revision reference fields differ")
        relative, path = _project_path(root, reference["path"], f"{context}.path")

    compiler = Compiler.load(root, path)
    gate = compiler.check_corpus()
    if compiler.state != "released" or not gate.passed:
        raise ValueError("Study Contract requires a released gated Compiler Revision")
    if not template and (
        gate.compiler_revision_sha256
        != _digest(reference["canonical_sha256"], f"{context}.canonical_sha256")
        or (
            "revision_id" in reference
            and reference["revision_id"] != gate.compiler_revision_id
        )
    ):
        raise ValueError("Study Contract Compiler Revision differs")
    exact = {
        "revision_id": gate.compiler_revision_id,
        "path": relative,
        "canonical_sha256": gate.compiler_revision_sha256,
    }
    return gate, relative, exact


def _evaluation_receipt_document(receipt: EvaluationReceipt) -> dict[str, object]:
    return {
        "candidate_sha256": receipt.candidate_sha256,
        "workload_sha256": receipt.workload_sha256,
        "evaluation_protocol_sha256": receipt.evaluation_protocol_sha256,
        "purpose": receipt.purpose,
        "case_id": receipt.case_id,
        "correctness_passed": receipt.correctness_passed,
        "correctness": _evaluation_plain_json(receipt.correctness),
        "kernel_calls": receipt.kernel_calls,
        "fallback_calls": receipt.fallback_calls,
        "launch_receipt_sha256": receipt.launch_receipt_sha256,
        "timing": _evaluation_plain_json(receipt.timing),
        "artifact_payload_sha256": {
            role: sha256(payload).hexdigest()
            for role, payload in sorted(receipt.artifact_payloads.items())
        },
    }


def _candidate_artifact_media_type(role: str) -> str:
    if role == "cubin":
        return "application/x-elf"
    if role == "launch_manifest":
        return "application/json"
    return "text/plain"


def _archive_evaluation_receipt(
    evidence: EvidenceStore,
    receipt: EvaluationReceipt,
) -> list[dict[str, object]]:
    required = (
        {"correctness_output", "launch_receipt", "profile"}
        if receipt.purpose == "attribution"
        else {"correctness_output", "launch_receipt", "timing_samples"}
    )
    if set(receipt.artifact_payloads) != required:
        raise ValueError("EvaluationReceipt artifact custody is incomplete")
    references: list[dict[str, object]] = []
    for role, payload in sorted(receipt.artifact_payloads.items()):
        item = evidence.put(payload, media_type="application/json")
        references.append(item.reference(role))
    receipt_object = evidence.put(
        _canonical_json_bytes(_evaluation_receipt_document(receipt)),
        media_type="application/json",
    )
    references.append(receipt_object.reference("evaluation_receipt"))
    return references


def _logical_attempt_document(attempt: LogicalEvaluationAttempt) -> dict[str, object]:
    return {
        "candidate_sha256": attempt.candidate_sha256,
        "attempts": [
            {
                "job_id": item.job_id,
                "mode": item.mode,
                "candidate_sha256": item.candidate_sha256,
                "manifest_sha256": item.manifest_sha256,
                "policy_sha256": item.policy_sha256,
                "evaluator_arguments_sha256": item.evaluator_arguments_sha256,
                "admitted": item.admitted,
                "error": item.error,
                "compiler_invocations": item.compiler_invocations,
                "module_loads": item.module_loads,
                "preflight_calls": item.preflight_calls,
                "kernel_calls": item.kernel_calls,
                "timing_samples": item.timing_samples,
                "fallback_calls": item.fallback_calls,
                "receipt_sha256": (
                    item.receipt.canonical_sha256 if item.receipt is not None else None
                ),
                "artifact_payload_sha256": {
                    role: sha256(payload).hexdigest()
                    for role, payload in sorted(item.artifact_payloads.items())
                },
            }
            for item in attempt.attempts
        ],
        "final_receipt_sha256": (
            attempt.final_receipt.canonical_sha256
            if attempt.final_receipt is not None
            else None
        ),
    }


def _archive_logical_attempt(
    evidence: EvidenceStore,
    attempt: LogicalEvaluationAttempt,
) -> list[dict[str, object]]:
    references: list[dict[str, object]] = []
    for index, broker_attempt in enumerate(attempt.attempts, start=1):
        if not {"broker_record", "evaluator_result", "stdout", "stderr"} <= set(
            broker_attempt.artifact_payloads
        ):
            raise ValueError("BrokerAttempt raw artifact custody is incomplete")
        for role, payload in sorted(broker_attempt.artifact_payloads.items()):
            media_type = "application/json" if role in {"broker_record", "evaluator_result"} else "text/plain"
            item = evidence.put(payload, media_type=media_type)
            references.append(item.reference(f"attempt_{index}_{role}"))
    document = evidence.put(
        _canonical_json_bytes(_logical_attempt_document(attempt)),
        media_type="application/json",
    )
    references.append(document.reference("broker_attempt_ledger"))
    return references


def _replay_broker_attempt_ledger(
    evidence: EvidenceStore,
    references: list[object],
    document: Mapping[str, object],
    *,
    candidate: LaunchableCandidate,
    protocol_sha256: str,
    final_receipt: EvaluationReceipt | None,
) -> None:
    """Rebuild every broker attempt from retained raw results and compare its ledger."""

    root_fields = {"candidate_sha256", "attempts", "final_receipt_sha256"}
    attempt_fields = {
        "job_id",
        "mode",
        "candidate_sha256",
        "manifest_sha256",
        "policy_sha256",
        "evaluator_arguments_sha256",
        "admitted",
        "error",
        "compiler_invocations",
        "module_loads",
        "preflight_calls",
        "kernel_calls",
        "timing_samples",
        "fallback_calls",
        "receipt_sha256",
        "artifact_payload_sha256",
    }
    result_fields = {
        "schema_version",
        "job_id",
        "mode",
        "admitted",
        "error",
        "failure_class",
        "counters",
        "receipt",
    }
    receipt_fields = {
        "correctness_passed",
        "correctness",
        "kernel_calls",
        "fallback_calls",
        "timing",
        "artifacts",
    }
    counter_fields = (
        "compiler_invocations",
        "module_loads",
        "preflight_calls",
        "kernel_calls",
        "timing_samples",
        "fallback_calls",
    )
    if set(document) != root_fields or document.get("candidate_sha256") != candidate.candidate_sha256:
        raise ValueError("broker attempt ledger authority differs")
    attempts = document.get("attempts")
    if not isinstance(attempts, list) or len(attempts) not in {1, 2}:
        raise ValueError("broker attempt ledger cardinality differs")
    expected_final_sha256 = (
        final_receipt.canonical_sha256 if final_receipt is not None else None
    )
    if document.get("final_receipt_sha256") != expected_final_sha256:
        raise ValueError("broker attempt ledger final receipt differs")

    by_role: dict[str, Mapping[str, object]] = {}
    for value in references:
        if not isinstance(value, Mapping):
            raise ValueError("broker attempt object reference differs")
        role = value.get("role")
        if not isinstance(role, str) or role in by_role:
            raise ValueError("broker attempt object roles differ")
        by_role[role] = cast(Mapping[str, object], value)

    expected_reference_roles = {"broker_attempt_ledger"}
    authorities: list[tuple[object, ...]] = []
    job_ids: set[object] = set()
    receipt_attempts: list[int] = []
    for index, value in enumerate(attempts, start=1):
        attempt = _object(value, f"broker_attempts[{index - 1}]")
        if set(attempt) != attempt_fields:
            raise ValueError("broker attempt ledger fields differ")
        artifact_digests = attempt.get("artifact_payload_sha256")
        receipt_sha256 = attempt.get("receipt_sha256")
        expected_artifact_roles = {
            "broker_record",
            "evaluator_result",
            "stdout",
            "stderr",
        }
        if receipt_sha256 is not None:
            receipt_attempts.append(index)
        if (
            not isinstance(artifact_digests, Mapping)
            or set(artifact_digests) != expected_artifact_roles
        ):
            raise ValueError("broker attempt raw artifact roles differ")
        raw_payloads: dict[str, bytes] = {}
        for artifact_role in sorted(expected_artifact_roles):
            role = f"attempt_{index}_{artifact_role}"
            expected_reference_roles.add(role)
            reference = by_role.get(role)
            expected_digest = artifact_digests.get(artifact_role)
            if reference is None or expected_digest != reference.get("sha256"):
                raise ValueError("broker attempt raw artifact reference differs")
            raw_payloads[artifact_role] = evidence.read_object(reference)
            if sha256(raw_payloads[artifact_role]).hexdigest() != expected_digest:
                raise ValueError("broker attempt raw artifact bytes differ")

        try:
            broker_result = json.loads(raw_payloads["broker_record"])
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("broker attempt raw result is not JSON") from error
        result = _object(broker_result, "broker_record")
        if set(result) != result_fields or result.get("schema_version") != 1:
            raise ValueError("broker attempt raw result fields differ")
        counters = _object(result.get("counters"), "broker_record.counters")
        if set(counters) != set(counter_fields) or any(
            not isinstance(counters.get(field), int)
            or isinstance(counters.get(field), bool)
            or cast(int, counters[field]) < 0
            for field in counter_fields
        ):
            raise ValueError("broker attempt raw counters differ")
        if (
            not isinstance(attempt.get("job_id"), str)
            or re.fullmatch(r"gpuq-[0-9a-f]{12}", cast(str, attempt["job_id"]))
            is None
            or not isinstance(attempt.get("admitted"), bool)
            or attempt.get("mode") != "exclusive"
            or (
                attempt.get("error") is not None
                and not isinstance(attempt.get("error"), str)
            )
            or any(
                not isinstance(attempt.get(field), int)
                or isinstance(attempt.get(field), bool)
                or cast(int, attempt[field]) < 0
                for field in counter_fields
            )
            or result.get("job_id") != attempt.get("job_id")
            or result.get("mode") != attempt.get("mode")
            or result.get("admitted") is not attempt.get("admitted")
            or result.get("error") != attempt.get("error")
            or any(counters.get(field) != attempt.get(field) for field in counter_fields)
        ):
            raise ValueError("broker attempt raw observation differs from ledger")
        if result.get("failure_class") is not None and not isinstance(
            result.get("failure_class"), str
        ):
            raise ValueError("broker attempt failure class differs")
        if result.get("error") is None and result.get("failure_class") is not None:
            raise ValueError("broker attempt failure class contradicts success")
        if (
            attempt.get("candidate_sha256") != candidate.candidate_sha256
            or attempt.get("manifest_sha256") != candidate.launch_spec_sha256
            or attempt.get("policy_sha256") != protocol_sha256
            or not isinstance(attempt.get("evaluator_arguments_sha256"), str)
            or _DIGEST.fullmatch(cast(str, attempt["evaluator_arguments_sha256"]))
            is None
        ):
            raise ValueError("broker attempt execution authority differs")

        raw_receipt = result.get("receipt")
        if raw_receipt is None:
            if receipt_sha256 is not None:
                raise ValueError("broker attempt receipt absence differs")
        else:
            if final_receipt is None or receipt_sha256 != final_receipt.canonical_sha256:
                raise ValueError("broker attempt receipt seal differs")
            receipt = _object(raw_receipt, "broker_record.receipt")
            artifacts = _object(receipt.get("artifacts"), "broker_record.receipt.artifacts")
            expected_receipt_artifacts = (
                {"correctness_output", "launch_receipt", "profile"}
                if final_receipt.purpose == "attribution"
                else {"correctness_output", "launch_receipt", "timing_samples"}
            )
            if (
                set(receipt) != receipt_fields
                or set(artifacts) != expected_receipt_artifacts
                or any(not isinstance(path, str) or not path for path in artifacts.values())
                or len(set(artifacts.values())) != len(expected_receipt_artifacts)
                or result.get("admitted") is not True
                or result.get("error") is not None
                or receipt.get("correctness_passed") is not final_receipt.correctness_passed
                or receipt.get("correctness") != final_receipt.correctness
                or receipt.get("kernel_calls") != final_receipt.kernel_calls
                or receipt.get("fallback_calls") != final_receipt.fallback_calls
                or receipt.get("timing") != final_receipt.timing
            ):
                raise ValueError("broker attempt raw receipt differs")
            validate_paired_broker(final_receipt, str(result["job_id"]), counters)
        try:
            evaluator_result = _object(
                json.loads(raw_payloads["evaluator_result"]),
                "evaluator_result",
            )
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("evaluator result is not JSON") from error
        if evaluator_result.get("job_id") not in {
            result.get("job_id"),
            "gpuq-000000000000",
        }:
            raise ValueError("worker and broker job identities differ")
        normalized_evaluator_result = dict(evaluator_result)
        normalized_evaluator_result["job_id"] = result["job_id"]
        if normalized_evaluator_result != broker_result:
            raise ValueError("broker and evaluator raw results differ")

        authority = tuple(
            attempt[field]
            for field in (
                "candidate_sha256",
                "manifest_sha256",
                "policy_sha256",
                "evaluator_arguments_sha256",
            )
        )
        authorities.append(authority)
        job_ids.add(attempt["job_id"])

    if set(by_role) != expected_reference_roles:
        raise ValueError("broker attempt archived object coverage differs")
    if len(job_ids) != len(attempts):
        raise ValueError("broker attempt job identity is duplicated")
    if len(attempts) == 2:
        first = _object(attempts[0], "broker_attempts[0]")
        if (
            first.get("admitted") is not False
            or first.get("error") != "gpu_admission_differs"
            or first.get("receipt_sha256") is not None
            or any(first.get(field) != 0 for field in counter_fields)
            or authorities[0] != authorities[1]
        ):
            raise ValueError("broker admission recovery differs")
    expected_receipt_attempts = [len(attempts)] if final_receipt is not None else []
    if receipt_attempts != expected_receipt_attempts:
        raise ValueError("broker attempt receipt placement differs")


def _replay_launchable_candidate(
    evidence: EvidenceStore,
    launchable_events: list[Mapping[str, object]],
    *,
    turn: int,
    candidate_sha256: str,
    arm: str,
    manifest_parser: Callable,
) -> LaunchableCandidate:
    """Rebuild one sealed launchable and enforce its arm-owned artifact contract."""

    matching = []
    for event in launchable_events:
        payload = _object(event.get("payload"), "launchable.payload")
        if (
            payload.get("turn") == turn
            and payload.get("candidate_sha256") == candidate_sha256
        ):
            matching.append(payload)
    if len(matching) != 1:
        raise ValueError("launchable candidate event coverage differs")
    payload = matching[0]
    if set(payload) != {
        "turn",
        "candidate_sha256",
        "candidate_record_sha256",
        "objects",
    } or payload.get("candidate_sha256") != candidate_sha256:
        raise ValueError("launchable candidate event authority differs")
    references = payload.get("objects")
    if not isinstance(references, list):
        raise ValueError("launchable candidate object references differ")
    artifact_payloads: dict[str, bytes] = {}
    artifact_roles: dict[str, str] = {}
    for reference in references:
        if not isinstance(reference, Mapping):
            raise ValueError("launchable candidate object reference differs")
        role = reference.get("role")
        digest = reference.get("sha256")
        if (
            not isinstance(role, str)
            or role in artifact_roles
            or not isinstance(digest, str)
        ):
            raise ValueError("launchable candidate object roles differ")
        artifact_payloads[role] = evidence.read_object(reference)
        artifact_roles[role] = digest
        if sha256(artifact_payloads[role]).hexdigest() != digest:
            raise ValueError("launchable candidate artifact bytes differ")
    if (
        arm not in _ARM_ARTIFACT_ROLES
        or not _ARM_ARTIFACT_ROLES[arm] <= set(artifact_roles)
        or set(artifact_payloads) != set(artifact_roles)
    ):
        raise ValueError("launchable candidate arm artifact roles differ")
    manifest = manifest_parser(json.loads(artifact_payloads["launch_manifest"]))
    if artifact_roles["launch_manifest"] != manifest.canonical_sha256:
        raise ValueError("launchable candidate launch manifest seal differs")
    candidate = LaunchableCandidate(
        candidate_sha256=candidate_sha256,
        target=manifest.target,
        entry_point=manifest.kernel_name,
        artifact_roles=artifact_roles,
        launch_spec_sha256=manifest.canonical_sha256,
        artifact_payloads=artifact_payloads,
    )
    if payload.get("candidate_record_sha256") != candidate.canonical_sha256:
        raise ValueError("launchable candidate record seal differs")
    return candidate


def _replay_evaluation_attempt_event(
    evidence: EvidenceStore,
    payload: Mapping[str, object],
    *,
    candidate: LaunchableCandidate,
    protocol_sha256: str,
    final_receipt: EvaluationReceipt | None,
) -> None:
    """Resolve one attempt event to its sole ledger and retained raw artifacts."""

    if payload.get("candidate_sha256") != candidate.candidate_sha256:
        raise ValueError("evaluation attempt candidate differs")
    references = payload.get("objects")
    if not isinstance(references, list):
        raise ValueError("evaluation attempt objects differ")
    ledger_references = [
        value
        for value in references
        if isinstance(value, Mapping) and value.get("role") == "broker_attempt_ledger"
    ]
    if len(ledger_references) != 1:
        raise ValueError("evaluation attempt ledger coverage differs")
    document = _object(
        json.loads(
            evidence.read_object(cast(Mapping[str, object], ledger_references[0]))
        ),
        "broker_attempt_ledger",
    )
    _replay_broker_attempt_ledger(
        evidence,
        cast(list[object], references),
        document,
        candidate=candidate,
        protocol_sha256=protocol_sha256,
        final_receipt=final_receipt,
    )


def _matched_endpoint_from_checkpoint(
    checkpoint: object,
    protocol_adherence: str,
) -> tuple[str, Mapping[str, object] | None]:
    state = getattr(checkpoint, "state")
    if protocol_adherence != "adhered" or state == "unreached":
        return "missing", None
    if state == "reached_with_best":
        return (
            "qualified",
            {
                "qualified_by_budget": True,
                "budget": getattr(checkpoint, "provider_tokens"),
                "best_candidate_sha256": getattr(checkpoint, "best_candidate_sha256"),
                "best_confirmed_latency_ms": getattr(
                    checkpoint, "best_confirmed_latency_ms"
                ),
            },
        )
    return (
        "no_qualified_candidate",
        {"qualified_by_budget": False, "budget": getattr(checkpoint, "provider_tokens")},
    )


def _receipt_qualifies(receipt: EvaluationReceipt) -> bool:
    return (
        receipt.correctness_passed
        and receipt.kernel_calls == 1
        and receipt.fallback_calls == 0
        and receipt.timing is not None
        and receipt.timing.get("measurement_quality_passed") is True
        and _receipt_latency_ms(receipt) is not None
    )


def _validate_receipt_authority(
    receipt: EvaluationReceipt,
    *,
    candidate: LaunchableCandidate,
    workload_sha256: str,
    protocol_sha256: str,
    case_id: str,
    purpose: str,
    evaluation_protocol: Mapping[str, object] | None = None,
    fixed_baseline: Mapping[str, object] | None = None,
) -> None:
    if (
        receipt.candidate_sha256 != candidate.candidate_sha256
        or receipt.workload_sha256 != workload_sha256
        or receipt.evaluation_protocol_sha256 != protocol_sha256
        or receipt.case_id != case_id
        or receipt.purpose != purpose
    ):
        raise ValueError("EvaluationReceipt does not match the Campaign Lock")
    if evaluation_protocol is not None:
        validate_receipt_policy(receipt, evaluation_protocol, fixed_baseline, candidate)


def _receipt_latency_ms(receipt: EvaluationReceipt | None) -> float | None:
    if receipt is None or receipt.timing is None:
        return None
    value = receipt.timing.get("pooled_median_ms")
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        return None
    return float(value)


def _matched_search_plan(
    order: list[Mapping[str, object]], searches_per_turn: int
) -> tuple[list[str], list[dict[str, str]]]:
    """Derive the unique programs to measure from the retained filter order."""

    first_by_semantic: dict[str, str] = {}
    searched: list[str] = []
    collapsed: list[dict[str, str]] = []
    for row in order:
        if len(searched) >= searches_per_turn or row["disposition"] != "launchable":
            break
        candidate = cast(str, row["candidate_sha256"])
        semantic = cast(str | None, row.get("semantic_sha256"))
        if semantic is not None and semantic in first_by_semantic:
            collapsed.append(
                {
                    "candidate_sha256": candidate,
                    "same_program_as": first_by_semantic[semantic],
                    "semantic_sha256": semantic,
                }
            )
        else:
            searched.append(candidate)
            if semantic is not None:
                first_by_semantic[semantic] = candidate
    return searched, collapsed


def _collapse_diagnosis(
    turn: int, collapsed: list[dict[str, str]]
) -> dict[str, object] | None:
    if not collapsed:
        return None
    return {
        "turn": turn,
        "routed_to": CANDIDATE,
        "routing_reason": (
            f"{len(collapsed)} of the ranked candidates this Turn are the same "
            "program as an earlier one under a different name"
        ),
        "collapsed": collapsed,
    }


def _matched_search_decision(
    turn: int,
    searched: list[tuple[str, EvaluationReceipt]],
    *,
    cost_order_applied: bool,
    materiality_ratio: float,
) -> tuple[list[int], int, dict[str, object] | None]:
    """Choose the measured winner and derive the sole cost-order diagnosis."""

    qualified = [
        index for index, (_, receipt) in enumerate(searched) if _receipt_qualifies(receipt)
    ]
    measured = sorted(
        qualified,
        key=lambda index: _receipt_latency_ms(searched[index][1]) or float("inf"),
    )
    best = measured[0] if measured else 0
    if not cost_order_applied or len(qualified) < 2 or best == qualified[0]:
        return qualified, best, None
    ranked_index = qualified[0]
    ranked = _receipt_latency_ms(searched[ranked_index][1])
    fastest = _receipt_latency_ms(searched[best][1])
    ratio = ranked / fastest if ranked is not None and fastest else None
    if ratio is None or ratio < materiality_ratio:
        return qualified, best, None
    ranked_sha = searched[ranked_index][0]
    fastest_sha = searched[best][0]
    return qualified, best, {
        "turn": turn,
        "routed_to": COST_MODEL,
        "routing_reason": (
            f"the filter ranked {ranked_sha} first and measurement put "
            f"{fastest_sha} {ratio:.3f}x ahead of it, which the Study counts as "
            f"material at {materiality_ratio}x"
        ),
        "ranked_first": ranked_sha,
        "measured_first": fastest_sha,
        "observed_ratio": round(ratio, 6),
        "materiality_ratio": materiality_ratio,
    }


def _empirical_filter(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Apply one complete advisory order; callers supply original provider order."""
    launchable = [row for row in rows if row["disposition"] == "launchable"]
    for row in launchable:
        estimate = _object(row.get("empirical_cost"), "candidate empirical cost")
        if estimate.get("covered") is True:
            value = estimate.get("predicted_kernel_us")
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError("candidate empirical prediction differs")
        elif estimate.get("covered") is not False:
            raise ValueError("candidate empirical coverage differs")
    applied = bool(launchable) and all(row["empirical_cost"]["covered"] for row in launchable)
    if applied:
        launchable.sort(key=lambda row: row["empirical_cost"]["predicted_kernel_us"])
    return (
        launchable + [row for row in rows if row["disposition"] != "launchable"],
        {
            "kind": _EMPIRICAL_SELECTION,
            "order_applied": applied,
            "reason": (
                "complete comparable point estimates; advisory only"
                if applied else "incomplete coverage or no launchable candidates; provider order retained"
            ),
        },
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


def _promoted_artifact(
    evidence: EvidenceStore,
    audit: RunAudit,
) -> Mapping[str, object] | None:
    """Select one per-Run confirmed artifact without constructing a treatment contrast."""

    if (
        not audit.archive_integrity
        or not audit.filesystem_custody_verified
        or audit.protocol_adherence != "adhered"
    ):
        return None
    eligible: list[tuple[float, int, str, str]] = []
    for event in evidence.replay_events(audit.run_id):
        if event.get("kind") != "candidate_evaluated":
            continue
        payload = _object(event.get("payload"), "candidate_evaluated.payload")
        if payload.get("purpose") != "confirmatory":
            continue
        turn = payload.get("turn")
        candidate_sha256 = payload.get("candidate_sha256")
        objects = payload.get("objects")
        if (
            not isinstance(turn, int)
            or isinstance(turn, bool)
            or turn <= 0
            or not isinstance(candidate_sha256, str)
            or _DIGEST.fullmatch(candidate_sha256) is None
            or not isinstance(objects, list)
        ):
            raise ValueError("artifact promotion Candidate evidence differs")
        receipt_refs = [
            cast(Mapping[str, object], item)
            for item in objects
            if isinstance(item, Mapping) and item.get("role") == "evaluation_receipt"
        ]
        if len(receipt_refs) != 1:
            raise ValueError("artifact promotion receipt evidence differs")
        receipt_bytes = evidence.read_object(receipt_refs[0])
        receipt = _object(json.loads(receipt_bytes), "artifact promotion receipt")
        timing = receipt.get("timing")
        latency = timing.get("pooled_median_ms") if isinstance(timing, Mapping) else None
        if (
            receipt.get("candidate_sha256") != candidate_sha256
            or receipt.get("correctness_passed") is not True
            or receipt.get("kernel_calls") != 1
            or receipt.get("fallback_calls") != 0
            or not isinstance(timing, Mapping)
            or timing.get("measurement_quality_passed") is not True
            or not isinstance(latency, (int, float))
            or isinstance(latency, bool)
            or not math.isfinite(float(latency))
            or float(latency) <= 0
        ):
            continue
        eligible.append(
            (
                float(latency),
                turn,
                candidate_sha256,
                sha256(receipt_bytes).hexdigest(),
            )
        )
    if not eligible:
        return None
    latency, turn, candidate_sha256, receipt_sha256 = min(eligible)
    return MappingProxyType(
        {
            "turn": turn,
            "candidate_sha256": candidate_sha256,
            "confirmed_latency_ms": latency,
            "evaluation_receipt_sha256": receipt_sha256,
        }
    )


@dataclass(frozen=True)
class StudyContract:
    """Frozen matched-search or Portfolio execution and data-use authority."""

    document: Mapping[str, object]
    source_path: Path
    study_id: str
    schema_version: int
    state: str
    canonical_sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "StudyContract":
        """Load the currently supported closed Study Contract variant."""

        source = Path(path).resolve(strict=True)
        document = _object(json.loads(source.read_text(encoding="utf-8")), "study")
        kind = document.get("kind")
        schema_version = document.get("schema_version")
        fields = (
            _RALPH_STUDY_FIELDS
            if kind == "matched_search" and schema_version == 2
            else _PORTFOLIO_STUDY_FIELDS
            if kind == "portfolio" and schema_version == 1
            else set()
        )
        if set(document) != fields:
            raise ValueError("study root fields or schema_version differ")
        state = document.get("state")
        if state not in {"template", "frozen"} or kind not in {
            "matched_search",
            "portfolio",
        }:
            raise ValueError("Study state or kind differs")
        study_id = _name(document.get("study_id"), "study.study_id")
        if schema_version == 2:
            interface = _object(document.get("agent_interface"), "study.agent_interface")
            if interface != {"schema_version": 1, "kind": TASK_AGENTS_RALPH_V1}:
                raise ValueError("Study Ralph agent interface differs")
        claim_scope = _name(document.get("claim_scope"), "study.claim_scope")
        if kind == "matched_search" and claim_scope not in _MATCHED_CLAIM_SCOPES:
            raise ValueError("matched Study Contract claim scope differs")
        object_fields = (
            (
                "workload",
                "arms",
                "allocation",
                "budget",
                "run_protocol",
                "evaluation_protocol",
                "execution",
                "analysis_plan",
                "evidence",
            )
            if kind == "matched_search"
            else (
                "workload",
                "compiler_revision",
                "kernel_seed",
                "case_roles",
                "specialization_policy",
                "dispatch_policy",
                "evaluation_protocol",
                "execution",
                "analysis_plan",
                "evidence",
            )
        )
        for field in object_fields:
            _object(document.get(field), f"study.{field}")
        detached = cast(Mapping[str, object], json.loads(_canonical_json_bytes(document)))
        return cls(
            document=detached,
            source_path=source,
            study_id=study_id,
            schema_version=cast(int, schema_version),
            state=cast(str, state),
            canonical_sha256=sha256(_canonical_json_bytes(document)).hexdigest(),
        )


@dataclass(frozen=True)
class CampaignLock:
    """Resolved immutable closure authorizing one Campaign execution instance."""

    document: Mapping[str, object]
    canonical_sha256: str
    study_id: str
    study_kind: str
    claim_scope: str
    agent_interface: str
    workload_id: str
    compiler_revision_id: str
    run_order: tuple[str, ...]
    experimental_unit: str
    estimand: str | None
    analysis_plan: Mapping[str, object]

    @classmethod
    def from_dict(cls, value: object) -> "CampaignLock":
        """Validate and reconstruct one resolved Campaign Lock."""

        document = _object(value, "campaign_lock")
        fields = {
            "schema_version",
            "study",
            "workload",
            "compiler_revision",
            "resolved_inputs",
            "run_order",
            "evaluation_protocol",
            "execution",
            "analysis_plan",
            "analysis_plan_sha256",
        }
        if set(document) != fields or document.get("schema_version") != 1:
            raise ValueError("Campaign Lock fields or schema_version differ")
        study = _object(document.get("study"), "campaign_lock.study")
        workload = _object(document.get("workload"), "campaign_lock.workload")
        compiler = _object(
            document.get("compiler_revision"), "campaign_lock.compiler_revision"
        )
        resolved = _object(document.get("resolved_inputs"), "campaign_lock.resolved_inputs")
        if set(study) != {"study_id", "kind", "claim_scope", "canonical_sha256"}:
            raise ValueError("Campaign Lock study fields differ")
        if set(workload) != {"workload_id", "path", "canonical_sha256"}:
            raise ValueError("Campaign Lock workload fields differ")
        if set(compiler) != {"revision_id", "path", "canonical_sha256"}:
            raise ValueError("Campaign Lock Compiler Revision fields differ")
        for context, digest in (
            ("study", study.get("canonical_sha256")),
            ("workload", workload.get("canonical_sha256")),
            ("compiler", compiler.get("canonical_sha256")),
            ("analysis_plan", document.get("analysis_plan_sha256")),
        ):
            _digest(digest, f"campaign_lock.{context}.sha256")
        run_order_value = document.get("run_order")
        if not isinstance(run_order_value, list) or not run_order_value:
            raise ValueError("Campaign Lock run_order must be non-empty")
        run_order = tuple(_name(item, "campaign_lock.run_order[]") for item in run_order_value)
        if len(run_order) != len(set(run_order)):
            raise ValueError("Campaign Lock run_order contains duplicates")
        study_kind = _name(study.get("kind"), "campaign_lock.study.kind")
        claim_scope = _name(study.get("claim_scope"), "campaign_lock.study.claim_scope")
        if study_kind == "matched_search":
            if claim_scope not in _MATCHED_CLAIM_SCOPES:
                raise ValueError("matched Campaign Lock claim scope differs")
            if set(resolved) != {
                "arm_environments", "arm_environment_sha256", "budget",
                "run_protocol", "evidence_policy", "agent_interface",
            }:
                raise ValueError("matched Campaign Lock requires the Ralph interface")
            interface = _object(resolved["agent_interface"], "campaign_lock.agent_interface")
            if interface != {"schema_version": 1, "kind": TASK_AGENTS_RALPH_V1}:
                raise ValueError("Campaign Lock Ralph agent interface differs")
            agent_interface = TASK_AGENTS_RALPH_V1
            RalphBudget.from_mapping(_object(resolved["budget"], "campaign_lock.budget"))
            arms = _object(
                resolved.get("arm_environments"),
                "campaign_lock.resolved_inputs.arm_environments",
            )
            arm_hashes = _object(
                resolved.get("arm_environment_sha256"),
                "campaign_lock.resolved_inputs.arm_environment_sha256",
            )
            comparison = comparison_arm(arms)
            if set(arm_hashes) != set(arms):
                raise ValueError("Campaign Lock Authoring Environment set differs")
            for arm_name in arms:
                environment = _object(
                    arms.get(arm_name),
                    f"campaign_lock.resolved_inputs.arm_environments.{arm_name}",
                )
                if "prompt_template" in environment:
                    raise ValueError("Ralph arms cannot contain prompt_template")
                digest = _digest(
                    arm_hashes.get(arm_name),
                    f"campaign_lock.resolved_inputs.{arm_name}.sha256",
                )
                if digest != sha256(_canonical_json_bytes(environment)).hexdigest():
                    raise ValueError(f"Campaign Lock {arm_name} environment bytes differ")
            budget = _object(
                resolved.get("budget"), "campaign_lock.resolved_inputs.budget"
            )
            selection = arms["open_cake"].get("candidate_selection")
            if "candidate_selection" in arms[comparison]:
                raise ValueError(f"{comparison} empirical selection is unsupported")
            if "candidate_selection" in arms["open_cake"]:
                if (
                    comparison != "direct_cuda"
                    or "input_format" in arms["open_cake"]
                ):
                    raise ValueError("empirical selection requires the complete-Schedule/direct-CUDA assay")
                if (
                    claim_scope != "artifact_optimization_only"
                    or "maximum_candidates_per_turn" not in budget
                    or not isinstance(selection, Mapping)
                    or set(selection) != {"kind", "model"}
                    or selection.get("kind") != _EMPIRICAL_SELECTION
                ):
                    raise ValueError("Campaign Lock empirical selection policy differs")
                EmpiricalCostModel(selection["model"])
            _object(resolved.get("run_protocol"), "campaign_lock.resolved_inputs.run_protocol")
            evidence_policy = _object(
                resolved.get("evidence_policy"),
                "campaign_lock.resolved_inputs.evidence_policy",
            )
            _matched_evidence_policy_version(
                evidence_policy,
                "campaign_lock.resolved_inputs.evidence_policy",
            )
            expected_arms = (
                sorted([comparison, "open_cake"])
                if claim_scope in _ONE_RUN_PER_ARM_SCOPES
                else sorted([comparison] * 3 + ["open_cake"] * 3)
            )
            if sorted(name.rsplit("-", 1)[0] for name in run_order) != expected_arms:
                raise ValueError("matched Campaign Lock Run allocation differs")
        elif study_kind == "portfolio":
            agent_interface = "portfolio_v1"
            if claim_scope != "bounded_local_b200_reconstruction":
                raise ValueError("portfolio Campaign Lock claim scope differs")
            if set(resolved) != {
                "kernel_seed",
                "case_roles",
                "specialization_policy",
                "dispatch_policy",
                "evidence_policy",
            }:
                raise ValueError("portfolio Campaign Lock inputs differ")
            seed = _object(resolved.get("kernel_seed"), "campaign_lock.resolved_inputs.kernel_seed")
            if set(seed) != {"seed_id", "path", "canonical_sha256"}:
                raise ValueError("portfolio Kernel Seed reference differs")
            _digest(seed.get("canonical_sha256"), "campaign_lock.kernel_seed.sha256")
            _object(resolved.get("case_roles"), "campaign_lock.resolved_inputs.case_roles")
            _object(
                resolved.get("specialization_policy"),
                "campaign_lock.resolved_inputs.specialization_policy",
            )
            _object(resolved.get("dispatch_policy"), "campaign_lock.resolved_inputs.dispatch_policy")
            _object(
                resolved.get("evidence_policy"),
                "campaign_lock.resolved_inputs.evidence_policy",
            )
        else:
            raise ValueError("Campaign Lock Study kind is unsupported")
        for field in ("evaluation_protocol", "execution"):
            _object(document.get(field), f"campaign_lock.{field}")
        if paired_protocol(document['evaluation_protocol']) is not None:
            execution = document['execution']
            if set(execution) != {'target', 'executor_revision', 'broker_execution_sha256',
                                  'gpu', 'sandbox', 'fixed_baseline', 'runtime_config'}:
                raise ValueError('paired Campaign execution fields differ')
            if study_kind != 'matched_search' or comparison != 'native_triton':
                raise ValueError('paired Campaign requires the native Triton comparison')
            _digest(execution['broker_execution_sha256'], 'execution.broker_execution_sha256')
            executor = _object(execution['executor_revision'], 'execution.executor_revision')
            if set(executor) != {'executor_id', 'path', 'canonical_sha256'}:
                raise ValueError('paired Campaign Executor must be resolved')
            _digest(executor['canonical_sha256'], 'execution.executor_revision.canonical_sha256')
            runtime = _object(execution['runtime_config'], 'execution.runtime_config')
            if (set(runtime) != {'path', 'sha256'} or not isinstance(runtime['path'], str)
                or not Path(runtime['path']).is_absolute() or '..' in Path(runtime['path']).parts):
                raise ValueError('paired Campaign runtime binding differs')
            _digest(runtime['sha256'], 'runtime_config.sha256')
            for arm in arms.values():
                provider = arm['provider']
                _digest(arm['toolchain_sha256'], 'arm.toolchain_sha256')
                _digest(provider['executable_sha256'], 'provider.executable_sha256')
                _name(provider['revision'], 'provider.revision')
                for field in ('qualification', 'qualification_anchor'):
                    reference = _object(provider[field], f'provider.{field}')
                    if set(reference) != {'path', 'canonical_sha256'} or not Path(str(reference['path'])).is_absolute():
                        raise ValueError('paired Campaign qualification binding differs')
                    _digest(reference['canonical_sha256'], f'provider.{field}.canonical_sha256')
            fixed = _object(document['execution'].get('fixed_baseline'), 'execution.fixed_baseline')
            if set(fixed) != {'bundle_path', 'candidate'} or not Path(str(fixed['bundle_path'])).is_absolute():
                raise ValueError('paired Campaign fixed baseline binding differs')
            bound_baseline = candidate_from_identity(fixed['candidate'])
            if bound_baseline.target != execution['target']:
                raise ValueError('paired Campaign baseline target differs')
        analysis = _object(document.get("analysis_plan"), "campaign_lock.analysis_plan")
        analysis_sha = sha256(_canonical_json_bytes(analysis)).hexdigest()
        if document.get("analysis_plan_sha256") != analysis_sha:
            raise ValueError("Campaign Lock Analysis Plan bytes differ")
        experimental_unit = _name(
            analysis.get("experimental_unit"), "campaign_lock.analysis_plan.experimental_unit"
        )
        expected_unit = "run" if study_kind == "matched_search" else "case_route"
        if experimental_unit != expected_unit:
            raise ValueError("Campaign Lock experimental unit differs from Study kind")
        raw_estimand = analysis.get("estimand")
        if claim_scope == "system_qualification_only":
            if analysis != _SYSTEM_QUALIFICATION_ANALYSIS_PLAN:
                raise ValueError("system qualification Campaign Lock Analysis Plan differs")
            estimand = None
        elif claim_scope == "artifact_optimization_only":
            if analysis != _ARTIFACT_OPTIMIZATION_ANALYSIS_PLAN:
                raise ValueError("artifact optimization Campaign Lock Analysis Plan differs")
            estimand = None
        elif study_kind == "matched_search":
            version = _scientific_analysis_plan_version(analysis, "campaign_lock.analysis_plan")
            if (comparison == "native_triton") != (version == "triton_optimization_v1"):
                raise ValueError("Campaign Lock treatment and analysis arms differ")
            estimand = _name(raw_estimand, "campaign_lock.analysis_plan.estimand")
        else:
            estimand = _name(raw_estimand, "campaign_lock.analysis_plan.estimand")
        detached = cast(Mapping[str, object], json.loads(_canonical_json_bytes(document)))
        return cls(
            document=detached,
            canonical_sha256=sha256(_canonical_json_bytes(document)).hexdigest(),
            study_id=_name(study.get("study_id"), "campaign_lock.study.study_id"),
            study_kind=study_kind,
            claim_scope=claim_scope,
            agent_interface=agent_interface,
            workload_id=_name(
                workload.get("workload_id"), "campaign_lock.workload.workload_id"
            ),
            compiler_revision_id=_name(
                compiler.get("revision_id"),
                "campaign_lock.compiler_revision.revision_id",
            ),
            run_order=run_order,
            experimental_unit=experimental_unit,
            estimand=estimand,
            analysis_plan=MappingProxyType(dict(analysis)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CampaignLock":
        """Load a Campaign Lock from canonical JSON."""

        source = Path(path).resolve(strict=True)
        return cls.from_dict(json.loads(source.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class CampaignRef:
    """Read-only reference to one Campaign's lock and evidence root."""

    lock: CampaignLock
    evidence_root: Path


@dataclass(frozen=True)
class StudyReport:
    """Policy-bound report over structurally and semantically audited Runs."""

    study_id: str
    claim_scope: str
    system_qualification_passed: bool | None
    estimand: str | None
    campaign_complete: bool
    archive_integrity_passed: bool
    filesystem_custody_verified: bool
    semantic_replay_passed: bool
    estimand_available: bool
    missing_run_count: int
    estimate: Mapping[str, object] | None
    uncertainty: Mapping[str, object] | None
    descriptive: Mapping[str, object]
    run_inclusion: tuple["AnalysisInclusion", ...]
    run_audits: tuple[RunAudit, ...]


@dataclass(frozen=True)
class AnalysisInclusion:
    """Preregistered inclusion decision kept separate from archive/adherence facts."""

    run_id: str
    qualification_endpoint_included: bool
    conditional_performance_included: bool
    reason: str


@dataclass(frozen=True)
class TurnRequest:
    """Lab-owned request for exactly one provider Turn."""

    run_id: str
    arm: str
    turn: int
    cumulative_provider_tokens: int
    thread_id: str | None
    feedback: Mapping[str, object]
    maximum_candidates_per_turn: int
    state_card: Mapping[str, object] | None = None


class RunProvider(Protocol):
    """Provider seam returning raw-normalized Turn evidence, never a Run terminal."""

    provider_revision: str
    qualification_sha256: str
    executable_sha256: str
    configuration: Mapping[str, object]

    def turn(self, request: TurnRequest) -> ProviderTurn:
        """Execute exactly one initial or resumed Turn."""


class RunEvaluator(Protocol):
    """Common Evaluation seam used identically by both treatments."""

    protocol_sha256: str
    protocol: Mapping[str, object]

    def evaluate(
        self,
        candidate: LaunchableCandidate,
        *,
        case_id: str,
        purpose: str,
    ) -> LogicalEvaluationAttempt:
        """Return one logical attempt retaining every bounded broker job."""


@dataclass(frozen=True)
class _SearchedCandidate:
    """One candidate kept intact from authored bytes through its search receipt."""

    submission: CandidateSubmission
    environment_result: EnvironmentResult
    launchable: LaunchableCandidate
    receipt: EvaluationReceipt
    attribution: EvaluationReceipt | None


class Lab:
    """Resolve, execute and audit preregistered studies over frozen dependencies."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        workload_loader: Callable,
        prepare_schedule: Callable,
        validate_authoring: Callable,
        manifest_parser: Callable,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._root = Path(project_root).resolve(strict=True)
        self._clock = clock
        self._load_workload = workload_loader
        self._prepare_schedule = prepare_schedule
        self._validate_authoring = validate_authoring
        self._parse_manifest = manifest_parser

    def task_package(self, lock, run_id):
        workload = self._load_workload(self._root / str(lock.document["workload"]["path"]))
        return render_task_package(self._root, lock, run_id,
            workload_contract=workload, prepare_schedule=self._prepare_schedule)

    def preflight(
        self, study_path: str | Path, *, empirical_cost_model_path: str | Path | None = None,
        execution_bindings_path: str | Path | None = None
    ) -> CampaignLock:
        """Resolve one Study Contract without provider, GPU or evidence side effects."""

        study = StudyContract.load(study_path)
        if study.document["kind"] == "portfolio":
            if empirical_cost_model_path is not None or execution_bindings_path is not None:
                raise ValueError("empirical selection requires artifact_optimization_only matched search")
            raise ValueError("portfolio preflight belongs to its task entrypoint")
        resolved_document, bound_executor = resolve_execution_bindings(
            self._root, study, execution_bindings_path
        )
        study = replace(study, document=resolved_document)
        workload_ref = _object(study.document.get("workload"), "study.workload")
        if set(workload_ref) != {"path", "canonical_sha256"}:
            raise ValueError("study workload reference fields differ")
        workload_relative, workload_path = _project_path(
            self._root, workload_ref.get("path"), "study.workload.path"
        )
        workload = self._load_workload(workload_path)
        if workload.canonical_sha256 != _digest(
            workload_ref.get("canonical_sha256"), "study.workload.canonical_sha256"
        ):
            raise ValueError("Study Contract workload bytes differ")

        arms = _object(study.document.get("arms"), "study.arms")
        comparison = comparison_arm(arms)
        paired_triton = comparison == "native_triton"
        open_cake = _object(arms.get("open_cake"), "study.arms.open_cake")
        direct_cuda = _object(arms.get(comparison), f"study.arms.{comparison}")
        self._validate_authoring(workload, arms, empirical_cost_model_path=empirical_cost_model_path)
        empirical_policy = open_cake.get("candidate_selection")
        has_empirical_policy = "candidate_selection" in open_cake
        if has_empirical_policy or empirical_cost_model_path is not None:
            if (
                study.document["claim_scope"] != "artifact_optimization_only"
                or empirical_policy != {"kind": _EMPIRICAL_SELECTION}
                or empirical_cost_model_path is None
                or "maximum_candidates_per_turn" not in study.document["budget"]
            ):
                raise ValueError("empirical selection requires artifact_optimization_only candidate-set policy and an explicit model")
        open_cake_fields = {
            "environment_kind",
            "provider",
            "scaffold",
            "compiler_revision",
            "lowering_route",
            "schedule_skeleton",
            "tool_surface",
            "feedback",
        }
        if has_empirical_policy:
            open_cake_fields.add("candidate_selection")
        direct_cuda_fields = {
            "environment_kind",
            "provider",
            "scaffold",
            "launch_contract",
            "candidate_skeleton",
            "toolchain_sha256",
            "tool_surface",
            "feedback",
        }
        if paired_triton:
            open_cake_fields.update({"input_format", "toolchain_sha256"})
            direct_cuda_fields -= {"launch_contract", "candidate_skeleton"}
            direct_cuda_fields.add("baseline")
            if (open_cake.get("input_format") != "schedule_or_python_v1"
                or direct_cuda.get("baseline") != {"binding": "open_cake_lowering"}
                or open_cake.get("toolchain_sha256") != direct_cuda.get("toolchain_sha256")):
                raise ValueError("paired Triton input, baseline or common toolchain binding differs")
        if set(open_cake) != open_cake_fields or set(direct_cuda) != direct_cuda_fields:
            raise ValueError("Study Contract Authoring Environment fields differ")
        if open_cake.get("environment_kind") != "open_cake" or direct_cuda.get(
            "environment_kind"
        ) != comparison:
            raise ValueError("Study Contract Authoring Environment kinds differ")
        route = open_cake.get("lowering_route")
        if (not isinstance(route, Mapping) or set(route) != {"backend", "entry_point"}
            or route.get("backend") != "triton" or not isinstance(route.get("entry_point"), str)
            or not route["entry_point"].isidentifier()):
            raise ValueError("Study Contract Open Cake lowering route differs")
        schedule_skeleton = _object(
            open_cake.get("schedule_skeleton"), "study.arms.open_cake.schedule_skeleton"
        )
        if set(schedule_skeleton) != {"path", "canonical_sha256"}:
            raise ValueError("Study Contract Schedule skeleton reference differs")
        _, schedule_skeleton_path = _project_path(
            self._root,
            schedule_skeleton.get("path"),
            "study.arms.open_cake.schedule_skeleton.path",
        )
        skeleton_document = _object(
            json.loads(schedule_skeleton_path.read_text(encoding="utf-8")),
            "study.arms.open_cake.schedule_skeleton",
        )
        if (
            skeleton_document.get("lowering") != open_cake.get("lowering_route")
            or _digest(
                schedule_skeleton.get("canonical_sha256"),
                "study.arms.open_cake.schedule_skeleton.canonical_sha256",
            )
            != sha256(_canonical_json_bytes(skeleton_document)).hexdigest()
        ):
            raise ValueError("Study Contract Schedule skeleton bytes or lowering route differ")
        if open_cake.get("provider") != direct_cuda.get(
            "provider"
        ) or open_cake.get("scaffold") != direct_cuda.get("scaffold"):
            raise ValueError("matched Authoring Environments differ in provider or scaffold")
        _digest(direct_cuda.get("toolchain_sha256"), "study.arms.direct_cuda.toolchain_sha256")
        provider = _object(open_cake.get("provider"), "study.arms.provider")
        provider_fields = {
            "revision",
            "qualification",
            "qualification_anchor",
            "executable_sha256",
            "model",
            "reasoning_effort",
            "service_tier",
            "output_schema",
            "removed_environment",
            "sandbox",
            "cwd_policy",
            "reference_visibility",
            "disabled_features",
            "code_mode_host",
        }
        if frozenset(provider) not in {
            frozenset(provider_fields | {"web_search"}),
            frozenset(provider_fields | {"event_contract"}),
        }:
            raise ValueError("Study Contract provider configuration fields differ")
        provider_revision = _name(provider.get("revision"), "study.arms.provider.revision")
        claim_scope = cast(str, study.document["claim_scope"])
        expected_disabled_features = (
            []
            if claim_scope == "artifact_optimization_only"
            else list(CODEX_DISABLED_FEATURES)
        )
        expected_event_contract = (
            "tool_rich_candidate_v1"
            if claim_scope == "artifact_optimization_only"
            else "closed_file_change_v1"
        )
        code_mode_host = _object(provider.get("code_mode_host"), "study.arms.provider.code_mode_host")
        if (set(code_mode_host) != {"path", "sha256"}
            or not isinstance(code_mode_host.get("path"), str)
            or not Path(code_mode_host["path"]).is_absolute()
            or ".." in Path(code_mode_host["path"]).parts):
            raise ValueError("Study Contract Code Mode host identity differs")
        _digest(code_mode_host.get("sha256"), "study.arms.provider.code_mode_host.sha256")
        _name(
            provider.get("reasoning_effort"),
            "study.arms.provider.reasoning_effort",
        )
        if (
            provider.get("model") != "gpt-5.6-sol"
            or provider.get("service_tier") != "default"
            or provider.get("sandbox") != "workspace-write"
            or provider.get("cwd_policy")
            != "independent_task_workspace"
            or provider.get("reference_visibility")
            != "workspace_task_files"
            or provider.get("disabled_features") != expected_disabled_features
            or (expected_event_contract == "closed_file_change_v1"
                and provider.get("web_search") != "disabled")
            or provider.get("event_contract", "closed_file_change_v1")
            != expected_event_contract
            or provider.get("removed_environment")
            != ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
        ):
            raise ValueError("Study Contract provider configuration differs")
        executable_sha256 = _digest(
            provider.get("executable_sha256"), "study.arms.provider.executable_sha256"
        )
        qualification_ref = _object(
            provider.get("qualification"),
            "study.arms.provider.qualification",
        )
        if set(qualification_ref) != {"path", "canonical_sha256"}:
            raise ValueError("provider qualification reference fields differ")
        _, qualification_path = _qualification_path(
            self._root,
            qualification_ref.get("path"),
            "study.arms.provider.qualification.path",
        )
        qualification = ProviderQualificationReceipt.load(qualification_path)
        if (
            qualification.provider_revision != provider_revision
            or qualification.executable_sha256 != executable_sha256
            or qualification.configuration_sha256
            != sha256(
                _canonical_json_bytes(
                    {
                        "model": provider["model"],
                        "reasoning_effort": provider["reasoning_effort"],
                        "service_tier": provider["service_tier"],
                        "output_schema_sha256": _object(
                            provider["output_schema"], "study.arms.provider.output_schema"
                        )["sha256"],
                        "removed_environment": provider["removed_environment"],
                        "sandbox": provider["sandbox"],
                        "cwd_policy": provider["cwd_policy"],
                        "reference_visibility": provider["reference_visibility"],
                        "disabled_features": provider["disabled_features"],
                        "code_mode_host": provider["code_mode_host"],
                        **({"web_search": provider["web_search"]} if "web_search" in provider else {}),
                        **(
                            {"event_contract": provider["event_contract"]}
                            if "event_contract" in provider
                            else {}
                        ),
                        **(
                            {
                                "submission_contract": CANDIDATE_SET_ENVELOPE_V1
                            }
                        ),
                    }
                )
            ).hexdigest()
            or not qualification.initial_and_resume_equivalent
            or not qualification.file_lifecycle_observed
            or not qualification.usage_observed
            or not qualification.qualified
            or qualification.scope
            not in {
                "zero_gpu_contract_fixture_only",
                required_live_provider_qualification_scope(claim_scope),
            }
            or qualification_ref.get("canonical_sha256") != qualification.canonical_sha256
        ):
            raise ValueError("provider qualification bytes or capability differs")
        if (paired_triton and qualification.scope != 'zero_gpu_contract_fixture_only'
            and paired_protocol(study.document['evaluation_protocol']) is None):
            raise ValueError('new live native Campaign requires explicit fixed-baseline paired policy')
        qualification_anchor = provider.get("qualification_anchor")
        if qualification.scope == "zero_gpu_contract_fixture_only":
            if qualification_anchor is not None:
                raise ValueError("fixture provider qualification anchor must be null")
        else:
            anchor_reference = _object(
                qualification_anchor,
                "study.arms.provider.qualification_anchor",
            )
            if set(anchor_reference) != {"path", "canonical_sha256"}:
                raise ValueError("provider qualification anchor reference differs")
            _, anchor_path = _qualification_path(
                self._root,
                anchor_reference.get("path"),
                "study.arms.provider.qualification_anchor.path",
            )
            anchor = _object(
                json.loads(anchor_path.read_text(encoding="utf-8")),
                "study.arms.provider.qualification_anchor.document",
            )
            if set(anchor) != {
                "schema_version",
                "kind",
                "run_id",
                "evidence_root",
                "authority_sha256",
                "qualification_receipt_sha256",
                "immediate_audit_integrity",
                "terminal_seal_sha256",
            }:
                raise ValueError("provider qualification anchor fields differ")
            if (
                anchor.get("schema_version") != 1
                or anchor.get("kind")
                != "codex_provider_qualification_evidence_anchor"
                or not isinstance(anchor.get("run_id"), str)
                or not anchor["run_id"]
                or not isinstance(anchor.get("evidence_root"), str)
                or not anchor["evidence_root"]
                or _digest(
                    anchor.get("authority_sha256"),
                    "provider qualification anchor authority",
                )
                != anchor.get("authority_sha256")
                or anchor.get("qualification_receipt_sha256")
                != qualification.canonical_sha256
                or anchor.get("immediate_audit_integrity") is not True
                or _digest(
                    anchor.get("terminal_seal_sha256"),
                    "provider qualification anchor terminal seal",
                )
                != anchor.get("terminal_seal_sha256")
                or _digest(
                    anchor_reference.get("canonical_sha256"),
                    "study.arms.provider.qualification_anchor.canonical_sha256",
                )
                != sha256(_canonical_json_bytes(anchor)).hexdigest()
            ):
                raise ValueError("provider qualification anchor evidence differs")
        for field in ("output_schema",):
            reference = _object(provider.get(field), f"study.arms.provider.{field}")
            if set(reference) != {"path", "sha256"}:
                raise ValueError(f"Study Contract provider {field} reference differs")
            _, path = _project_path(
                self._root, reference.get("path"), f"study.arms.provider.{field}.path"
            )
            if _digest(reference.get("sha256"), f"study.arms.provider.{field}.sha256") != sha256(
                path.read_bytes()
            ).hexdigest():
                raise ValueError(f"Study Contract provider {field} bytes differ")
        scaffold = _object(open_cake.get("scaffold"), "study.arms.scaffold")
        if set(scaffold) != {"path", "sha256"}:
            raise ValueError("Study Contract scaffold reference differs")
        _, scaffold_path = _project_path(
            self._root, scaffold.get("path"), "study.arms.scaffold.path"
        )
        if _digest(scaffold.get("sha256"), "study.arms.scaffold.sha256") != sha256(
            scaffold_path.read_bytes()
        ).hexdigest():
            raise ValueError("Study Contract scaffold bytes differ")
        if not paired_triton:
            launch_contract = _object(
                direct_cuda.get("launch_contract"), "study.arms.direct_cuda.launch_contract"
            )
            if set(launch_contract) != {"path", "sha256"}:
                raise ValueError("Study Contract direct launch contract reference differs")
            _, launch_contract_path = _project_path(
                self._root,
                launch_contract.get("path"),
                "study.arms.direct_cuda.launch_contract.path",
            )
            if _digest(
                launch_contract.get("sha256"),
                "study.arms.direct_cuda.launch_contract.sha256",
            ) != sha256(launch_contract_path.read_bytes()).hexdigest():
                raise ValueError("Study Contract direct launch contract bytes differ")
            candidate_skeleton = _object(
                direct_cuda.get("candidate_skeleton"),
                "study.arms.direct_cuda.candidate_skeleton",
            )
            if set(candidate_skeleton) != {"path", "sha256"}:
                raise ValueError("Study Contract direct candidate skeleton reference differs")
            _, candidate_skeleton_path = _project_path(
                self._root,
                candidate_skeleton.get("path"),
                "study.arms.direct_cuda.candidate_skeleton.path",
            )
            if _digest(
                candidate_skeleton.get("sha256"),
                "study.arms.direct_cuda.candidate_skeleton.sha256",
            ) != sha256(candidate_skeleton_path.read_bytes()).hexdigest():
                raise ValueError("Study Contract direct candidate skeleton bytes differ")
        if open_cake.get("tool_surface") != (["submit_schedule_or_python"] if paired_triton else ["submit_schedule"]) or direct_cuda.get(
            "tool_surface"
        ) != (["submit_triton_kernel"] if paired_triton else ["submit_cuda"]):
            raise ValueError("Study Contract Authoring Environment tool surfaces differ")
        attribution_evaluation = _object(
            study.document.get("evaluation_protocol"),
            "study.evaluation_protocol",
        ).get("attribution_evaluation")
        if attribution_evaluation not in {
            None,
            _LEGACY_ATTRIBUTION_EVALUATION,
            _ATTRIBUTION_EVALUATION,
        }:
            raise ValueError("Study Contract attribution Evaluation differs")
        profile_feedback = ["profile"] if attribution_evaluation is not None else []
        if open_cake.get("feedback") != [
            "findings",
            "correctness",
            "qualified_timing",
            *profile_feedback,
        ] or direct_cuda.get("feedback") != [
            "compile",
            "correctness",
            "qualified_timing",
            *profile_feedback,
        ]:
            raise ValueError("Study Contract Authoring Environment feedback differs")
        gate, compiler_relative, compiler_reference = (
            _resolve_compiler_reference(
                self._root,
                open_cake.get("compiler_revision"),
                "study.arms.open_cake.compiler_revision",
                template=study.state == "template",
            )
        )

        if paired_triton:
            case_id = str(_object(study.document["evaluation_protocol"], "evaluation_protocol")["case_id"])
            baseline = bind_baseline(skeleton_document, workload, case_id)
            baseline_compiler = Compiler.load(self._root, self._root / compiler_relative)
            assessment = baseline_compiler.assess(baseline)
            if not assessment.lowering_eligible:
                raise ValueError("paired optimization baseline is not lowerable")
            baseline_lowering = baseline_compiler.lower(assessment)
            native_baseline(baseline_lowering)

        allocation = _object(study.document.get("allocation"), "study.allocation")
        order = allocation.get("order")
        if allocation.get("method") != "predeclared_balanced_blocks" or not isinstance(order, list):
            raise ValueError("Study Contract allocation differs")
        run_order = tuple(_name(value, "study.allocation.order[]") for value in order)
        expected_arms = (
            sorted([comparison, "open_cake"])
            if claim_scope in _ONE_RUN_PER_ARM_SCOPES
            else sorted([comparison] * 3 + ["open_cake"] * 3)
        )
        if len(run_order) != len(set(run_order)) or sorted(
            name.rsplit("-", 1)[0] for name in run_order
        ) != expected_arms:
            required = "one" if claim_scope in _ONE_RUN_PER_ARM_SCOPES else "three"
            raise ValueError(
                f"Study Contract must predeclare {required} independent Run(s) per arm"
            )

        budget = _object(study.document.get("budget"), "study.budget")
        checkpoints = budget.get("checkpoints")
        limit = budget.get("limit")
        maximum_turns = budget.get("maximum_turns")
        maximum_candidates_per_turn = budget.get("maximum_candidates_per_turn", 1)
        budget_fields = {"unit", "limit", "checkpoints", "maximum_turns"}
        budget_fields.add("maximum_candidates_per_turn")
        budget_fields.update(
            {
                "wall_time_seconds",
                "active_authoring_time_seconds",
                "evaluation_limits",
            }
        )
        if (
            set(budget) != budget_fields
            or
            budget.get("unit") != "provider_tokens"
            or not isinstance(limit, int)
            or isinstance(limit, bool)
            or limit <= 0
            or not isinstance(checkpoints, list)
            or not checkpoints
            or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in checkpoints)
            or checkpoints != sorted(set(checkpoints))
            or checkpoints[-1] != limit
            or not isinstance(maximum_turns, int)
            or isinstance(maximum_turns, bool)
            or maximum_turns <= 0
            or not isinstance(maximum_candidates_per_turn, int)
            or isinstance(maximum_candidates_per_turn, bool)
            or maximum_candidates_per_turn <= 0
        ):
            raise ValueError("Study Contract budget grid differs")
        RalphBudget.from_mapping(budget)
        run_protocol = _object(study.document.get("run_protocol"), "study.run_protocol")
        expected_workspace = (
            run_protocol.get("workspace_seed") == "task_agents_only"
        )
        if (
            run_protocol.get("independent_thread") is not True
            or not expected_workspace
            or run_protocol.get("automatic_retries") != 0
            or run_protocol.get("replacement_runs") != 0
        ):
            raise ValueError("Study Contract Run Protocol differs")
        evaluation = _object(
            study.document.get("evaluation_protocol"), "study.evaluation_protocol"
        )
        workload.case(_name(evaluation.get("case_id"), "study.evaluation_protocol.case_id"))
        assay = paired_protocol(evaluation)
        if assay is not None and not paired_triton:
            raise ValueError('fixed-baseline assay requires the paired Triton Study')
        # How many candidates a Turn search-evaluates. Checked here because a Study that
        # asks for none, or for a word, would otherwise fault partway through a run --
        # and a run that faults has already spent the GPU time this Lab exists to gate.
        searches = evaluation.get("searches_per_turn", 1)
        if not isinstance(searches, int) or isinstance(searches, bool) or searches < 1:
            raise ValueError("Study Contract searches_per_turn differs")
        if searches > maximum_candidates_per_turn:
            raise ValueError(
                "Study Contract searches_per_turn exceeds maximum_candidates_per_turn"
            )
        ralph_limits = _object(
            budget.get("evaluation_limits"), "study.budget.evaluation_limits"
        )
        required_attribution = searches if attribution_evaluation == _ATTRIBUTION_EVALUATION else 0
        if (
            int(ralph_limits.get("search", 0)) < searches
            or int(ralph_limits.get("confirmatory", 0)) < 1
            or int(ralph_limits.get("attribution", 0)) < required_attribution
        ):
            raise ValueError("Ralph budget cannot admit one complete Turn")
        # How much faster the measurement has to be before the order counts as wrong.
        # A Study that searches more than one candidate has to say, because without it
        # every inversion inside the noise would be routed to the cost model as a defect
        # -- and the loss surface is a plateau, so most inversions are inside the noise
        # (`docs/ANALYSIS_CALIBRATION.md`).
        materiality = evaluation.get("search_materiality_ratio")
        if searches > 1:
            if (
                not isinstance(materiality, float)
                or not 1.0 < materiality < 100.0
            ):
                raise ValueError(
                    "a Study searching more than one candidate declares "
                    "search_materiality_ratio"
                )
        elif materiality is not None:
            # No second candidate to compare against, so a ratio here would state a
            # threshold nothing can cross.
            raise ValueError("search_materiality_ratio without searches_per_turn above one")
        execution = _object(study.document.get("execution"), "study.execution")
        expected_execution_fields = {'target', 'executor_revision', 'broker_execution_sha256', 'gpu', 'sandbox'}
        if assay is not None:
            expected_execution_fields.update({'fixed_baseline', 'runtime_config'})
        if set(execution) != expected_execution_fields:
            raise ValueError("Study Contract execution fields differ")
        if assay is not None:
            from open_cake_ir.evaluation.paired import candidate_identity, validate_pair_candidates
            fixed = _object(execution['fixed_baseline'], 'execution.fixed_baseline')
            sealed_baseline = load_baseline_bundle(self._root, fixed['bundle_path'])
            validate_pair_candidates(sealed_baseline, sealed_baseline, workload, str(evaluation['case_id']))
            import ast
            from open_cake_ir.compiler.toolchain import project_triton_kernel
            requirements = baseline_lowering.toolchain_requirements
            source = sealed_baseline.artifact_payloads.get('lowered_source')
            expected_source = project_triton_kernel(baseline_lowering.source.encode(), requirements)
            if source is None:
                raise ValueError('fixed baseline requires retained Compiler lowering source')
            observed_source = project_triton_kernel(source, requirements)
            manifest = self._parse_manifest(json.loads(sealed_baseline.artifact_payloads['launch_manifest']))
            if (fixed['candidate'] != candidate_identity(sealed_baseline)
                or ast.dump(ast.parse(observed_source)) != ast.dump(ast.parse(expected_source))
                or list(manifest.grid) != requirements['grid']
                or manifest.block != (requirements['compile_options']['num_warps'] * 32, 1, 1)):
                raise ValueError('fixed baseline differs from the frozen Compiler kernel or launch commitments')
        if (
            execution.get("target") != workload.target
            or execution.get("target") != skeleton_document.get("target")
            or execution.get("sandbox") != "workspace-write"
        ):
            raise ValueError("Study Contract execution authority differs")
        _digest(
            execution.get("broker_execution_sha256"),
            "study.execution.broker_execution_sha256",
        )
        # External binding has already applied the original template grammar and
        # verified this Executor. Preserve that resolution and the Study identity.
        executor = (
            bound_executor
            if bound_executor is not None
            else resolve_executor(
                self._root,
                execution.get("executor_revision"),
                "study.execution",
                template=study.state == "template",
            )
        )
        executor_reference = dict(executor.reference)
        gpu = _object(execution.get("gpu"), "study.execution.gpu")
        target = cuda_target(execution['target'])
        if (set(gpu) != {'name', 'count', 'mode'} or gpu.get('name') not in target.device_names
            or type(gpu.get('count')) is not int or gpu['count'] != 1 or gpu.get('mode') != 'exclusive'):
            raise ValueError("Study Contract GPU admission differs")
        analysis = _object(study.document.get("analysis_plan"), "study.analysis_plan")
        if claim_scope == "system_qualification_only":
            if analysis != _SYSTEM_QUALIFICATION_ANALYSIS_PLAN:
                raise ValueError("system qualification Analysis Plan differs")
            estimand = None
        elif claim_scope == "artifact_optimization_only":
            if analysis != _ARTIFACT_OPTIMIZATION_ANALYSIS_PLAN:
                raise ValueError("artifact optimization Analysis Plan differs")
            estimand = None
        else:
            version = _scientific_analysis_plan_version(analysis, "study.analysis_plan")
            if paired_triton != (version == "triton_optimization_v1"):
                raise ValueError("scientific treatment and analysis arm assignment differ")
            estimand = _name(analysis.get("estimand"), "study.analysis_plan.estimand")
        evidence_policy = _object(study.document.get("evidence"), "study.evidence")
        evidence_version = _matched_evidence_policy_version(
            evidence_policy, "study.evidence"
        )
        if evidence_version != _MATCHED_RALPH_EVENT_VOCABULARY_V1:
            raise ValueError("Study agent interface and Evidence policy differ")

        resolved_arms = cast(
            dict[str, object], json.loads(_canonical_json_bytes(arms))
        )
        _object(
            resolved_arms["open_cake"], "resolved open_cake arm"
        )["compiler_revision"] = compiler_reference
        if has_empirical_policy:
            model_path = Path(empirical_cost_model_path).resolve(strict=True)
            if self._root in model_path.parents:
                raise ValueError("external empirical model must stay outside the project checkout")
            model_document = json.loads(model_path.read_text(encoding="utf-8"))
            EmpiricalCostModel(model_document)
            resolved_arms["open_cake"]["candidate_selection"] = {
                "kind": _EMPIRICAL_SELECTION, "model": model_document,
            }
        resolved_execution = cast(
            dict[str, object], json.loads(_canonical_json_bytes(execution))
        )
        resolved_execution["executor_revision"] = executor_reference
        arm_digests = {
            name: sha256(_canonical_json_bytes(value)).hexdigest()
            for name, value in resolved_arms.items()
        }
        lock_document: dict[str, object] = {
            "schema_version": 1,
            "study": {
                "study_id": study.study_id,
                "kind": study.document["kind"],
                "claim_scope": claim_scope,
                "canonical_sha256": study.canonical_sha256,
            },
            "workload": {
                "workload_id": workload.workload_id,
                "path": workload_relative,
                "canonical_sha256": workload.canonical_sha256,
            },
            "compiler_revision": {
                "revision_id": gate.compiler_revision_id,
                "path": compiler_relative,
                "canonical_sha256": gate.compiler_revision_sha256,
            },
            "resolved_inputs": {
                "arm_environments": resolved_arms,
                "arm_environment_sha256": arm_digests,
                "budget": budget,
                "run_protocol": run_protocol,
                "evidence_policy": evidence_policy,
                **(
                    {"agent_interface": study.document["agent_interface"]}
                ),
            },
            "run_order": list(run_order),
            "evaluation_protocol": evaluation,
            "execution": resolved_execution,
            "analysis_plan": analysis,
            "analysis_plan_sha256": sha256(_canonical_json_bytes(analysis)).hexdigest(),
        }
        lock = CampaignLock.from_dict(lock_document)
        if (
            lock.study_id != study.study_id
            or lock.workload_id != workload.workload_id
            or lock.compiler_revision_id != gate.compiler_revision_id
            or lock.claim_scope != claim_scope
            or lock.estimand != estimand
        ):
            raise ValueError("resolved Campaign Lock projection differs")
        return lock


    def reference_campaign(
        self,
        lock: CampaignLock,
        evidence_root: str | Path,
    ) -> CampaignRef:
        """Create a read-only Campaign reference after execution or fixture construction."""

        root = Path(evidence_root).resolve(strict=True)
        if not (root / "runs").is_dir() or not (root / "objects/sha256").is_dir():
            raise ValueError("Campaign evidence root is incomplete")
        return CampaignRef(lock=lock, evidence_root=root)

    def execute(
        self,
        lock: CampaignLock,
        evidence_root: str | Path,
        *,
        provider: RunProvider,
        environments: Mapping[str, AuthoringEnvironment],
        evaluator: RunEvaluator,
    ) -> CampaignRef:
        """Own every Turn, budget, checkpoint, feedback and terminal decision."""

        root = admit_new_campaign_path(
            self._root,
            evidence_root,
            role="Campaign Evidence root",
        )
        comparison_arm(environments)
        if set(environments) != set(lock.document["resolved_inputs"]["arm_environments"]):
            raise ValueError("Campaign Authoring Environment set differs")
        if lock.study_kind != "matched_search":
            raise ValueError("Lab.execute matched-search path requires a matched Campaign Lock")
        ExecutorRevision.load_reference(
            self._root,
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
        self._validate_authoring(self._load_workload(self._root / str(lock.document["workload"]["path"])), arms)
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
            self._root,
            qualification_ref["path"],
            "arm_environments.open_cake.provider_qualification.path",
        )
        qualification = ProviderQualificationReceipt.load(qualification_path)
        if (comparison_arm(arms) == 'native_triton' and qualification.scope != 'zero_gpu_contract_fixture_only'
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
        record_confirmation_time = comparison_arm(environments) == "native_triton"
        for sequence, run_id in enumerate(lock.run_order, start=1):
            run_started_at = self._clock() if record_confirmation_time else None
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
                    clock=self._clock,
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
                    reference_bundle = provider_turn.reference_bundle
                    if reference_bundle is None:
                        raise RunProtocolFault(
                            "harness_fault",
                            "provider reference bundle observation is missing",
                        )
                    else:
                        if (
                            not isinstance(reference_bundle, bytes)
                            or not reference_bundle
                        ):
                            raise RunProtocolFault(
                                "harness_fault",
                                "provider reference bundle observation differs",
                            )
                        try:
                            reference_bundle.decode("utf-8")
                        except UnicodeError as error:
                            raise RunProtocolFault(
                                "harness_fault",
                                "provider reference bundle is not UTF-8",
                            ) from error
                        reference_object = evidence.put(
                            reference_bundle,
                            media_type="text/plain",
                        )
                    events_object = evidence.put(
                        provider_turn.raw_events,
                        media_type="application/x-ndjson",
                    )
                    submission_object = evidence.put(
                        provider_turn.raw_submission,
                        media_type="application/json",
                    )
                    projected_candidates = _project_candidate_submission(
                        provider_turn.raw_submission,
                        submission_contract=CANDIDATE_SET_ENVELOPE_V1,
                        arm=arm,
                        maximum_candidates_per_turn=maximum_candidates_per_turn,
                    )
                    if (
                        events_object.sha256 != provider_turn.raw_events_sha256
                        or projected_candidates != provider_turn.candidates
                        or not provider_turn.candidates
                        or len(provider_turn.candidates) > maximum_candidates_per_turn
                        or len(provider_turn.candidates)
                        != len(provider_turn.candidate_sha256s)
                        or len(set(provider_turn.candidate_sha256s))
                        != len(provider_turn.candidate_sha256s)
                    ):
                        raise ValueError("provider candidate set identity differs")
                    # Every candidate is sealed, not only the one that reaches a GPU.
                    # The set a Turn produced is what the pre-GPU filter acted on, so an
                    # evidence root that kept only the survivor could not show what was
                    # filtered or why the order was what it was.
                    candidate_objects = []
                    for payload, expected in zip(
                        provider_turn.candidates, provider_turn.candidate_sha256s
                    ):
                        sealed = evidence.put(payload, media_type=environment.media_type)
                        if sealed.sha256 != expected:
                            raise ValueError("provider candidate seal differs")
                        candidate_objects.append(sealed)
                    provider_payload: dict[str, object] = {
                        "turn": turn_number,
                        "thread_id": thread_id,
                        "turn_provider_tokens": provider_turn.provider_tokens,
                        "cumulative_provider_tokens": cumulative_tokens,
                        "normalization": provider_turn.normalization,
                        "candidate_count": len(candidate_objects),
                        "objects": [
                            *(
                                [
                                    reference_object.reference(
                                        "provider_reference_bundle"
                                    )
                                ]
                                if reference_object is not None
                                else []
                            ),
                            events_object.reference("provider_events"),
                            submission_object.reference("provider_submission_envelope"),
                            *(
                                item.reference(f"candidate_submission_{index:04d}")
                                for index, item in enumerate(candidate_objects)
                            ),
                        ],
                    }
                    if "event_contract" in provider_document:
                        provider_payload["auxiliary_activity"] = [
                            dict(activity.document)
                            for activity in provider_turn.tool_activity
                        ]
                    ledger.append("provider_turn_completed", provider_payload)
                    # The pre-GPU filter runs on the whole set: every candidate is built,
                    # which is the verifier and the toolchain but no device. Only then is
                    # an order taken, and only the survivor reaches an Evaluation. This is
                    # the stage the paper spends compile time on to avoid spending GPU
                    # time, so building all of them is the point rather than a cost.
                    live_stage = "environment"
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
                                    **({"elapsed_wall_seconds": self._clock() - run_started_at}
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
        return CampaignRef(lock=lock, evidence_root=evidence.root)


    def _replay_matched_run(
        self,
        evidence: EvidenceStore,
        audit: RunAudit,
        lock: CampaignLock,
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
        threads: set[str] = set()
        cumulative_by_turn: dict[int, int] = {}
        provider_candidates_by_turn: dict[int, tuple[str, ...]] = {}
        provider_candidate_bytes: dict[tuple[int, str], bytes] = {}
        candidate_set_turns: set[int] = set()
        prior_cumulative = 0
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
            executor = ExecutorRevision.load_reference(self._root, lock.document["execution"]["executor_revision"], "execution.executor_revision")
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
            self.task_package(lock, audit.run_id)
        )
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
                return False
            if payload.get("turn") != expected_turn:
                return False
            thread_id = payload.get("thread_id")
            cumulative = payload.get("cumulative_provider_tokens")
            if (
                not isinstance(thread_id, str)
                or not isinstance(cumulative, int)
                or isinstance(cumulative, bool)
                or cumulative <= 0
            ):
                return False
            objects = payload.get("objects")
            if not isinstance(objects, list):
                return False
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
                        return False
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
                return False
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
                return False
            if projected_candidates != candidates:
                return False
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
                return False
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
                        return False
                except (UnicodeError, json.JSONDecodeError, ValueError):
                    return False
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
                return False
            if parsed.candidate_path is not None and (
                parsed.change_kind != expected_change
                or Path(parsed.candidate_path).name != expected_name
            ):
                return False
            if parsed.candidate_path is None and event_contract != "tool_rich_candidate_v1":
                return False
            if payload.get("normalization") != parsed.normalization:
                return False
            if "event_contract" in provider_authority and payload.get(
                "auxiliary_activity"
            ) != [dict(activity.document) for activity in parsed.tool_activity]:
                return False
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
            return False
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
                return False
            key = (turn, cast(str, purpose), candidate_sha256)
            if key in attempt_payloads:
                return False
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
                return False
            key = (turn, candidate_sha256)
            if key in launchables:
                return False
            launchables[key] = _replay_launchable_candidate(
                evidence,
                launchable_events,
                turn=turn,
                candidate_sha256=candidate_sha256,
                arm=arm,
                manifest_parser=self._parse_manifest,
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
                    return False
                decision = route_rejection(feedback)
                if (
                    payload.get("routed_to") != decision.destination
                    or payload.get("routing_reason") != decision.reason
                ):
                    return False
                if (
                    not isinstance(turn, int)
                    or isinstance(turn, bool)
                    or not isinstance(candidate_sha256, str)
                    or _DIGEST.fullmatch(candidate_sha256) is None
                    or (turn, candidate_sha256) in rejected
                ):
                    return False
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
                    return False
                if "elapsed_wall_seconds" in payload and (
                    type(payload["elapsed_wall_seconds"]) not in {int, float}
                    or not math.isfinite(payload["elapsed_wall_seconds"])
                    or payload["elapsed_wall_seconds"] < 0
                ):
                    return False
                objects = payload.get("objects")
                if not isinstance(objects, list):
                    return False
                receipt_refs = [
                    item
                    for item in objects
                    if isinstance(item, Mapping) and item.get("role") == "evaluation_receipt"
                ]
                if len(receipt_refs) != 1:
                    return False
                receipt = _object(
                    json.loads(evidence.read_object(cast(Mapping[str, object], receipt_refs[0]))),
                    "evaluation_receipt",
                )
                if (
                    receipt.get("candidate_sha256") != candidate_sha256
                    or receipt.get("workload_sha256") != workload_sha256
                    or receipt.get("evaluation_protocol_sha256") != protocol_sha256
                    or receipt.get("case_id") != case_id
                    or receipt.get("purpose") != payload.get("purpose")
                ):
                    return False
                launchable = launchables.get((turn, candidate_sha256))
                if launchable is None:
                    return False
                expected_raw = receipt.get("artifact_payload_sha256")
                if not isinstance(expected_raw, Mapping) or (
                    purpose in {"search", "confirmatory"}
                    and set(expected_raw)
                    != {"correctness_output", "launch_receipt", "timing_samples"}
                ) or (
                    purpose == "attribution"
                    and set(expected_raw)
                    != {"correctness_output", "launch_receipt", "profile"}
                ):
                    return False
                raw_payloads: dict[str, bytes] = {}
                for role in sorted(expected_raw):
                    matching = [
                        item
                        for item in objects
                        if isinstance(item, Mapping) and item.get("role") == role
                    ]
                    if len(matching) != 1:
                        return False
                    raw_payloads[role] = evidence.read_object(
                        cast(Mapping[str, object], matching[0])
                    )
                if expected_raw != {
                    role: sha256(raw).hexdigest()
                    for role, raw in sorted(raw_payloads.items())
                }:
                    return False
                if {
                    item.get("role")
                    for item in objects
                    if isinstance(item, Mapping)
                } != {"evaluation_receipt", *expected_raw}:
                    return False
                validated_receipt = EvaluationReceipt(
                    candidate_sha256=candidate_sha256,
                    workload_sha256=workload_sha256,
                    evaluation_protocol_sha256=protocol_sha256,
                    purpose=str(payload["purpose"]),
                    case_id=case_id,
                    correctness_passed=receipt.get("correctness_passed") is True,
                    correctness=_object(receipt.get("correctness"), "receipt.correctness"),
                    kernel_calls=int(receipt["kernel_calls"]),
                    fallback_calls=int(receipt["fallback_calls"]),
                    launch_receipt_sha256=str(receipt["launch_receipt_sha256"]),
                    timing=(
                        _object(receipt["timing"], "receipt.timing")
                        if receipt.get("timing") is not None
                        else None
                    ),
                    artifact_payloads=raw_payloads,
                )
                validate_receipt_policy(validated_receipt, lock.document['evaluation_protocol'],
                    lock.document['execution'].get('fixed_baseline', {}).get('candidate'), launchable)
                if validated_receipt.canonical_sha256 != sha256(
                    _canonical_json_bytes(receipt)
                ).hexdigest():
                    return False
                receipt_key = (turn, cast(str, purpose), candidate_sha256)
                if receipt_key in receipts:
                    return False
                receipts[receipt_key] = validated_receipt
                receipt_order.append(receipt_key)
                attempt_payload = attempt_payloads.get(receipt_key)
                if attempt_payload is None:
                    return False
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
                return False
            turn, purpose, candidate_sha256 = next(iter(unreplayed_attempts))
            if turn != fault_turn:
                return False
            attempt_payload = attempt_payloads[(turn, purpose, candidate_sha256)]
            launchable = launchables.get((turn, candidate_sha256))
            if launchable is None:
                return False
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
            return False

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
                return False
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
                    return False
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
                    return False
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
                return False
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
                    return False
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
            return False

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
                return False
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
                        return False
                    continue
                selection = selections.get(turn)
                if selection is None:
                    if turn != fault_turn:
                        return False
                    continue
                selected = cast(str, selection["candidate_sha256"])
                order = filter_order[turn]
                dispositions = filter_disposition[turn]
                if selected not in provider_candidates:
                    return False
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
                        return False
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
                    return False
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
                    return False
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
                    return False
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
                    return False
                if turn == fault_turn:
                    continue
                if len(confirms) != (1 if qualified_search else 0):
                    return False
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
                    return False
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
                    return False
                selected = provider_candidates[0]
                if turn == fault_turn:
                    continue
                has_rejection = (turn, selected) in rejected
                has_evaluation = any(
                    key[0] == turn and key[2] == selected for key in receipts
                )
                if has_rejection == has_evaluation:
                    return False
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
                return False
        if any(turn not in filters for turn in selections):
            return False
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
                    return False

        for turn, candidate_sha256 in launchables:
            if turn in candidate_set_turns and filter_disposition.get(turn, {}).get(
                candidate_sha256
            ) != "launchable":
                return False

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

    def threshold_view(self, campaign: CampaignRef, latency_threshold_ms: float) -> Mapping[str, object]:
        """Describe first fresh confirmations from audited records; retain every Run.

        A caller-selected threshold is descriptive, never a new scientific estimand.
        Historical events without confirmation time retain unknown wall time.
        """
        if type(latency_threshold_ms) not in {int, float}:
            raise ValueError("latency threshold must be finite and positive")
        try:
            latency_threshold_ms = float(latency_threshold_ms)
        except (ValueError, OverflowError) as error:
            raise ValueError("latency threshold must be finite and positive") from error
        if not math.isfinite(latency_threshold_ms) or latency_threshold_ms <= 0:
            raise ValueError("latency threshold must be finite and positive")
        if campaign.lock.study_kind != "matched_search":
            raise ValueError("threshold view requires matched_search records")
        report = self.audit(campaign)
        evidence = EvidenceStore.open(campaign.evidence_root)
        audits = {audit.run_id: audit for audit in report.run_audits}
        limit = campaign.lock.document["resolved_inputs"]["budget"]["limit"]
        rows = []
        for run_id in campaign.lock.run_order:
            audit = audits.get(run_id)
            row = {"run_id": run_id, "endpoint": audit.endpoint_observation if audit else "missing",
                   "first_confirmation_turn": None, "provider_tokens": None,
                   "elapsed_wall_seconds": None, "candidate_sha256": None,
                   "confirmed_latency_ms": None, "status": "missing"}
            eligible = (audit is not None and audit.archive_integrity and audit.filesystem_custody_verified
                        and audit.protocol_adherence == "adhered" and report.semantic_replay_passed)
            if audit is not None and not eligible:
                row["status"] = "unverified_archive_or_protocol"
            elif eligible:
                row["status"] = "threshold_not_reached"
                tokens = {}
                for event in evidence.replay_events(run_id):
                    payload = event["payload"]
                    if event["kind"] == "provider_turn_completed":
                        tokens[payload["turn"]] = payload["cumulative_provider_tokens"]
                    if event["kind"] != "candidate_evaluated" or payload["purpose"] != "confirmatory":
                        continue
                    reference = next(value for value in payload["objects"] if value["role"] == "evaluation_receipt")
                    receipt = json.loads(evidence.read_object(reference))
                    timing = receipt["timing"]
                    if (tokens[payload["turn"]] <= limit and receipt["correctness_passed"] is True
                        and timing is not None and timing.get("measurement_quality_passed") is True
                        and timing["pooled_median_ms"] <= latency_threshold_ms):
                        row.update(status="reached_by_fresh_confirmation", first_confirmation_turn=payload["turn"],
                            provider_tokens=tokens[payload["turn"]], elapsed_wall_seconds=payload.get("elapsed_wall_seconds"),
                            candidate_sha256=payload["candidate_sha256"], confirmed_latency_ms=timing["pooled_median_ms"])
                        break
            rows.append(row)
        return {"audit": report, "scope": "descriptive_threshold_view", "latency_threshold_ms": latency_threshold_ms,
                "wall_time_definition": "run_start_to_archived_fresh_confirmation_including_authoring_build_and_evaluation",
                "missing_confirmation_timestamps": "unknown_never_inferred_from_search_or_terminal_time",
                "runs": rows}

    def audit(self, campaign: CampaignRef) -> StudyReport:
        """Audit terminal Runs, then apply the preregistered availability rule."""

        evidence = EvidenceStore.open(campaign.evidence_root)
        audits: list[RunAudit] = []
        campaign_complete = True
        for run_id in campaign.lock.run_order:
            try:
                audit = evidence.audit_run(run_id)
            except (OSError, ValueError, json.JSONDecodeError):
                campaign_complete = False
                continue
            if audit.authority_sha256 != campaign.lock.canonical_sha256:
                campaign_complete = False
            audits.append(audit)
        campaign_complete = campaign_complete and len(audits) == len(campaign.lock.run_order)
        archive_integrity_passed = campaign_complete and all(
            audit.archive_integrity for audit in audits
        )
        filesystem_custody_verified = campaign_complete and all(
            audit.filesystem_custody_verified for audit in audits
        )
        semantic_replay_passed = archive_integrity_passed
        evaluation_receipt_counts: dict[str, int] = {}
        if semantic_replay_passed:
            for audit in audits:
                try:
                    if not self._replay_matched_run(evidence, audit, campaign.lock):
                        semantic_replay_passed = False
                        break
                    evaluation_receipt_counts[audit.run_id] = sum(
                        event.get("kind") == "candidate_evaluated"
                        for event in evidence.replay_events(audit.run_id)
                    )
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    semantic_replay_passed = False
                    break
        missing_run_count = len(campaign.lock.run_order) - len(audits) + sum(
            not audit.archive_integrity
            or not audit.filesystem_custody_verified
            or audit.authority_sha256 != campaign.lock.canonical_sha256
            or audit.endpoint_observation == "missing"
            or audit.protocol_adherence != "adhered"
            for audit in audits
        )
        if campaign.lock.claim_scope == "artifact_optimization_only":
            promoted_artifacts = (
                {
                    audit.run_id: _promoted_artifact(evidence, audit)
                    for audit in audits
                }
                if semantic_replay_passed
                else {audit.run_id: None for audit in audits}
            )
            artifact_optimization_complete = (
                campaign_complete
                and archive_integrity_passed
                and filesystem_custody_verified
                and semantic_replay_passed
                and all(audit.protocol_adherence == "adhered" for audit in audits)
                and all(
                    promoted_artifacts.get(run_id) is not None
                    for run_id in campaign.lock.run_order
                )
            )
            inclusions = tuple(
                AnalysisInclusion(
                    audit.run_id,
                    False,
                    False,
                    "artifact_optimization_not_scientific_data",
                )
                for audit in audits
            )
            return StudyReport(
                study_id=campaign.lock.study_id,
                claim_scope=campaign.lock.claim_scope,
                system_qualification_passed=None,
                estimand=None,
                campaign_complete=campaign_complete,
                archive_integrity_passed=archive_integrity_passed,
                filesystem_custody_verified=filesystem_custody_verified,
                semantic_replay_passed=semantic_replay_passed,
                estimand_available=False,
                missing_run_count=sum(
                    not audit.archive_integrity
                    or not audit.filesystem_custody_verified
                    or audit.protocol_adherence != "adhered"
                    for audit in audits
                ),
                estimate=None,
                uncertainty=None,
                descriptive={
                    "artifact_optimization_complete": artifact_optimization_complete,
                    "promoted_artifacts": promoted_artifacts,
                },
                run_inclusion=inclusions,
                run_audits=tuple(audits),
            )
        if campaign.lock.claim_scope == "system_qualification_only":
            inclusions = tuple(
                AnalysisInclusion(
                    audit.run_id,
                    False,
                    False,
                    "system_qualification_not_scientific_data",
                )
                for audit in audits
            )
            system_qualification_passed = (
                campaign_complete
                and archive_integrity_passed
                and filesystem_custody_verified
                and semantic_replay_passed
                and all(audit.protocol_adherence == "adhered" for audit in audits)
                and all(
                    evaluation_receipt_counts.get(run_id, 0) >= 1
                    for run_id in campaign.lock.run_order
                )
            )
            descriptive: Mapping[str, object] = {
                "prescheduled_run_count": len(campaign.lock.run_order),
                "completed_run_count": len(audits),
                "runs": [
                    {
                        "run_id": audit.run_id,
                        "archive_integrity": audit.archive_integrity,
                        "filesystem_custody_verified": audit.filesystem_custody_verified,
                        "protocol_adherence": audit.protocol_adherence,
                        "endpoint_observation": audit.endpoint_observation,
                        "evaluation_receipt_count": evaluation_receipt_counts.get(
                            audit.run_id, 0
                        ),
                    }
                    for audit in audits
                ],
            }
            return StudyReport(
                study_id=campaign.lock.study_id,
                claim_scope=campaign.lock.claim_scope,
                system_qualification_passed=system_qualification_passed,
                estimand=None,
                campaign_complete=campaign_complete,
                archive_integrity_passed=archive_integrity_passed,
                filesystem_custody_verified=filesystem_custody_verified,
                semantic_replay_passed=semantic_replay_passed,
                estimand_available=False,
                missing_run_count=missing_run_count,
                estimate=None,
                uncertainty=None,
                descriptive=descriptive,
                run_inclusion=inclusions,
                run_audits=tuple(audits),
            )
        assigned_arms = campaign.lock.document["resolved_inputs"]["arm_environments"]
        comparison = comparison_arm(assigned_arms)
        arm_runs: dict[str, list[RunAudit]] = {name: [] for name in assigned_arms}
        inclusions: list[AnalysisInclusion] = []
        for audit in audits:
            arm = audit.run_id.rsplit("-", 1)[0]
            if arm not in arm_runs:
                raise ValueError(f"Run {audit.run_id!r} has an unknown assigned arm")
            arm_runs[arm].append(audit)
            if not audit.archive_integrity:
                inclusions.append(
                    AnalysisInclusion(audit.run_id, False, False, "archive_integrity")
                )
            elif audit.authority_sha256 != campaign.lock.canonical_sha256:
                inclusions.append(
                    AnalysisInclusion(audit.run_id, False, False, "campaign_authority")
                )
            elif not audit.filesystem_custody_verified:
                inclusions.append(
                    AnalysisInclusion(
                        audit.run_id,
                        False,
                        False,
                        "filesystem_custody_not_verified",
                    )
                )
            elif audit.protocol_adherence != "adhered":
                inclusions.append(
                    AnalysisInclusion(audit.run_id, False, False, "protocol_deviation")
                )
            elif audit.endpoint_observation == "missing":
                inclusions.append(
                    AnalysisInclusion(audit.run_id, False, False, "endpoint_missing")
                )
            elif audit.endpoint_observation == "no_qualified_candidate":
                inclusions.append(
                    AnalysisInclusion(
                        audit.run_id,
                        True,
                        False,
                        "observed_no_qualified_candidate_at_checkpoint",
                    )
                )
            else:
                inclusions.append(AnalysisInclusion(audit.run_id, True, True, "included"))
        analysis_version = _scientific_analysis_plan_version(
            campaign.lock.analysis_plan, "campaign_lock.analysis_plan"
        )
        inclusion_by_run = {item.run_id: item for item in inclusions}
        qualification_rate: dict[str, float | None] = {}
        endpoint_counts: dict[str, dict[str, int]] = {}
        medians: dict[str, float | None] = {}
        ranges: dict[str, list[float] | None] = {}
        latencies: dict[str, dict[str, float]] = {}
        for arm, arm_audits in arm_runs.items():
            prescheduled = sum(
                run_id.rsplit("-", 1)[0] == arm
                for run_id in campaign.lock.run_order
            )
            observed = [
                audit
                for audit in arm_audits
                if inclusion_by_run[audit.run_id].qualification_endpoint_included
            ]
            qualified = [
                audit
                for audit in observed
                if audit.endpoint_observation == "qualified"
            ]
            denominator = len(observed)
            qualification_rate[arm] = (
                len(qualified) / denominator
                if denominator
                else (None)
            )
            endpoint_counts[arm] = {
                "prescheduled": prescheduled,
                "observed": len(observed),
                "qualified": len(qualified),
                "missing": prescheduled - len(observed),
            }
            values: list[float] = []
            latencies[arm] = {}
            for audit in qualified:
                endpoint = audit.endpoint
                if endpoint is None or endpoint.get("qualified_by_budget") is not True:
                    raise ValueError(f"qualified Run {audit.run_id!r} lacks endpoint evidence")
                latency = endpoint.get("best_confirmed_latency_ms")
                if (
                    not isinstance(latency, (int, float))
                    or isinstance(latency, bool)
                    or not math.isfinite(float(latency))
                    or float(latency) <= 0
                ):
                    raise ValueError(f"qualified Run {audit.run_id!r} has an invalid latency")
                values.append(float(latency))
                latencies[arm][audit.run_id.rsplit("-", 1)[1]] = float(latency)
            medians[arm] = statistics.median(values) if values else None
            ranges[arm] = [min(values), max(values)] if values else None
        estimand_available = (
            archive_integrity_passed
            and filesystem_custody_verified
            and semantic_replay_passed
            and missing_run_count == 0
            and (
                all(medians[arm] is not None for arm in arm_runs)
            )
        )
        paired: list[dict[str, object]] = []
        for repetition in sorted({name.rsplit("-", 1)[1] for name in campaign.lock.run_order}):
            open_latency = latencies["open_cake"].get(repetition)
            cuda_latency = latencies[comparison].get(repetition)
            paired.append(
                {
                    "repetition": int(repetition) if repetition.isdigit() else repetition,
                    "open_cake_latency_ms": open_latency,
                    f"{comparison}_latency_ms": cuda_latency,
                    "open_cake_outcome": next((a.endpoint_observation for a in audits if a.run_id == f"open_cake-{repetition}"), "missing"),
                    f"{comparison}_outcome": next((a.endpoint_observation for a in audits if a.run_id == f"{comparison}-{repetition}"), "missing"),
                    "open_cake_speedup": (
                        cuda_latency / open_latency
                        if open_latency is not None and cuda_latency is not None
                        else None
                    ),
                }
            )
        descriptive = {
            "endpoint_counts": endpoint_counts,
            "qualification_rate_among_observed": qualification_rate,
            "qualification_rate_difference_among_observed": (
                cast(float, qualification_rate["open_cake"])
                - cast(float, qualification_rate[comparison])
                if all(value is not None for value in qualification_rate.values())
                else None
            ),
            "median_confirmed_latency_ms": medians,
            "paired_runs": paired,
        }
        estimate: Mapping[str, object] | None = None
        uncertainty: Mapping[str, object] | None = None
        if estimand_available:
            estimate = {
                "qualification_rate": qualification_rate,
                "median_confirmed_latency_ms": medians,
                "ratio_of_arm_medians": cast(float, medians[comparison])
                / cast(float, medians["open_cake"]),
            }
            if analysis_version in {"two_part_v2", "triton_optimization_v1"}:
                estimate = {
                    **estimate,
                    "qualification_rate_difference": (
                        cast(float, qualification_rate["open_cake"])
                        - cast(float, qualification_rate[comparison])
                    ),
                }
            uncertainty = {"latency_range_ms": ranges}
        return StudyReport(
            study_id=campaign.lock.study_id,
            claim_scope=campaign.lock.claim_scope,
            system_qualification_passed=None,
            estimand=campaign.lock.estimand,
            campaign_complete=campaign_complete,
            archive_integrity_passed=archive_integrity_passed,
            filesystem_custody_verified=filesystem_custody_verified,
            semantic_replay_passed=semantic_replay_passed,
            estimand_available=estimand_available,
            missing_run_count=missing_run_count,
            estimate=estimate,
            uncertainty=uncertainty,
            descriptive=descriptive,
            run_inclusion=tuple(inclusions),
            run_audits=tuple(audits),
        )
