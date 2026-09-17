"""Study contracts and pure Campaign preflight resolution."""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping, Protocol, cast

from open_cake_ir.compiler import Compiler
from open_cake_ir.evaluation import (
    CudaLaunchManifest,
    EvaluationReceipt,
    LaunchableCandidate,
    LogicalEvaluationAttempt,
    PortfolioArtifact,
    PortfolioEvaluationReceipt,
    WorkloadContract,
    replay_portfolio_receipt,
)
from open_cake_ir.evidence import EvidenceStore, RunAudit

from .checkpoints import TurnObservation, project_checkpoints
from .environments import AuthoringEnvironment, CandidateSubmission
from .executor import ExecutorRevision
from .faults import RunProtocolFault
from .portfolio import KernelSeed
from .providers import (
    CODEX_DISABLED_FEATURES,
    ProviderQualificationReceipt,
    ProviderTurn,
    parse_codex_turn_events,
)

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
}
_STUDY_FIELDS = {
    "schema_version",
    "study_id",
    "state",
    "kind",
    "claim_scope",
    "workload",
    "arms",
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


def _digest(value: object, context: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA256 digest")
    return value


def _project_path(root: Path, value: object, context: str) -> tuple[str, Path]:
    relative = _name(value, context)
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or ".." in parsed.parts or "\\" in relative:
        raise ValueError(f"{context} is unsafe")
    path = (root / relative).resolve(strict=True)
    if root not in path.parents:
        raise ValueError(f"{context} escapes project root")
    return relative, path


def _validate_executor_revision(
    root: Path,
    execution: Mapping[str, object],
    context: str,
) -> ExecutorRevision:
    reference = _object(execution.get("executor_revision"), f"{context}.executor_revision")
    if set(reference) != {"path", "canonical_sha256", "executor_id"}:
        raise ValueError(f"{context}.executor_revision fields differ")
    _, path = _project_path(root, reference["path"], f"{context}.executor_revision.path")
    revision = ExecutorRevision.load(root, path)
    if (
        revision.executor_id != reference["executor_id"]
        or revision.canonical_sha256
        != _digest(
            reference["canonical_sha256"],
            f"{context}.executor_revision.canonical_sha256",
        )
    ):
        raise ValueError(f"{context} Executor Revision differs")
    return revision


def _evaluation_receipt_document(receipt: EvaluationReceipt) -> dict[str, object]:
    return {
        "candidate_sha256": receipt.candidate_sha256,
        "workload_sha256": receipt.workload_sha256,
        "evaluation_protocol_sha256": receipt.evaluation_protocol_sha256,
        "purpose": receipt.purpose,
        "case_id": receipt.case_id,
        "correctness_passed": receipt.correctness_passed,
        "correctness": dict(receipt.correctness),
        "kernel_calls": receipt.kernel_calls,
        "fallback_calls": receipt.fallback_calls,
        "launch_receipt_sha256": receipt.launch_receipt_sha256,
        "timing": dict(receipt.timing) if receipt.timing is not None else None,
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
    required = {"correctness_output", "launch_receipt", "timing_samples"}
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
    if set(document) != root_fields or document.get("candidate_sha256") != (
        candidate.candidate_sha256
    ):
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
            if (
                set(receipt) != receipt_fields
                or set(artifacts)
                != {"correctness_output", "launch_receipt", "timing_samples"}
                or any(not isinstance(path, str) or not path for path in artifacts.values())
                or len(set(artifacts.values())) != 3
                or result.get("admitted") is not True
                or result.get("error") is not None
                or receipt.get("correctness_passed") is not final_receipt.correctness_passed
                or receipt.get("correctness") != final_receipt.correctness
                or receipt.get("kernel_calls") != final_receipt.kernel_calls
                or receipt.get("fallback_calls") != final_receipt.fallback_calls
                or receipt.get("timing") != final_receipt.timing
            ):
                raise ValueError("broker attempt raw receipt differs")
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
) -> LaunchableCandidate:
    """Rebuild one sealed launchable and enforce its arm-owned artifact contract."""

    matching = []
    for event in launchable_events:
        payload = _object(event.get("payload"), "launchable.payload")
        if payload.get("turn") == turn:
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
    manifest = CudaLaunchManifest.from_dict(json.loads(artifact_payloads["launch_manifest"]))
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


def _portfolio_endpoint(receipt: PortfolioEvaluationReceipt) -> dict[str, object]:
    return {
        "correctness_by_case": dict(receipt.correctness_by_case),
        "kernel_measurement_quality_by_case": dict(
            receipt.kernel_measurement_quality_by_case
        ),
        "dispatcher_measurement_quality_by_case": dict(
            receipt.dispatcher_measurement_quality_by_case
        ),
        "pooled_medians_ms": {
            case_id: dict(values) for case_id, values in receipt.pooled_medians_ms.items()
        },
        "unsupported_rejected_before_launch": receipt.unsupported_rejected_before_launch,
        "route_counts": dict(receipt.route_counts),
        "unsupported_kernel_call_delta": receipt.unsupported_kernel_call_delta,
        "receipt_sha256": receipt.canonical_sha256,
    }


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
) -> None:
    if (
        receipt.candidate_sha256 != candidate.candidate_sha256
        or receipt.workload_sha256 != workload_sha256
        or receipt.evaluation_protocol_sha256 != protocol_sha256
        or receipt.case_id != case_id
        or receipt.purpose != purpose
    ):
        raise ValueError("EvaluationReceipt does not match the Campaign Lock")


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


@dataclass(frozen=True)
class StudyContract:
    """Frozen matched-search or Portfolio execution and data-use authority."""

    document: Mapping[str, object]
    source_path: Path
    study_id: str
    canonical_sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "StudyContract":
        """Load the currently supported closed Study Contract variant."""

        source = Path(path).resolve(strict=True)
        document = _object(json.loads(source.read_text(encoding="utf-8")), "study")
        kind = document.get("kind")
        fields = _STUDY_FIELDS if kind == "matched_search" else _PORTFOLIO_STUDY_FIELDS
        if set(document) != fields or document.get("schema_version") != 1:
            raise ValueError("study root fields or schema_version differ")
        if document.get("state") != "frozen" or kind not in {"matched_search", "portfolio"}:
            raise ValueError("only frozen matched_search or portfolio studies are supported")
        study_id = _name(document.get("study_id"), "study.study_id")
        claim_scope = _name(document.get("claim_scope"), "study.claim_scope")
        if kind == "matched_search" and claim_scope not in {
            "system_qualification_only",
            "scientific_matched_search",
        }:
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
            if claim_scope not in {
                "system_qualification_only",
                "scientific_matched_search",
            }:
                raise ValueError("matched Campaign Lock claim scope differs")
            if set(resolved) != {
                "arm_environments",
                "arm_environment_sha256",
                "budget",
                "run_protocol",
                "evidence_policy",
            }:
                raise ValueError("matched Campaign Lock inputs differ")
            arms = _object(
                resolved.get("arm_environments"),
                "campaign_lock.resolved_inputs.arm_environments",
            )
            arm_hashes = _object(
                resolved.get("arm_environment_sha256"),
                "campaign_lock.resolved_inputs.arm_environment_sha256",
            )
            if set(arms) != {"open_cake", "direct_cuda"} or set(arm_hashes) != {
                "open_cake",
                "direct_cuda",
            }:
                raise ValueError("Campaign Lock Authoring Environment set differs")
            for arm_name in ("open_cake", "direct_cuda"):
                environment = _object(
                    arms.get(arm_name),
                    f"campaign_lock.resolved_inputs.arm_environments.{arm_name}",
                )
                digest = _digest(
                    arm_hashes.get(arm_name),
                    f"campaign_lock.resolved_inputs.{arm_name}.sha256",
                )
                if digest != sha256(_canonical_json_bytes(environment)).hexdigest():
                    raise ValueError(f"Campaign Lock {arm_name} environment bytes differ")
            budget = _object(
                resolved.get("budget"), "campaign_lock.resolved_inputs.budget"
            )
            _object(resolved.get("run_protocol"), "campaign_lock.resolved_inputs.run_protocol")
            _object(
                resolved.get("evidence_policy"),
                "campaign_lock.resolved_inputs.evidence_policy",
            )
            expected_arms = (
                ["direct_cuda", "open_cake"]
                if claim_scope == "system_qualification_only"
                else ["direct_cuda"] * 3 + ["open_cake"] * 3
            )
            if sorted(name.rsplit("-", 1)[0] for name in run_order) != expected_arms:
                raise ValueError("matched Campaign Lock Run allocation differs")
            if claim_scope == "system_qualification_only" and (
                budget.get("checkpoints") != [budget.get("limit")]
                or budget.get("maximum_turns") != 1
            ):
                raise ValueError("system qualification Campaign Lock budget differs")
        elif study_kind == "portfolio":
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
        else:
            estimand = _name(raw_estimand, "campaign_lock.analysis_plan.estimand")
        detached = cast(Mapping[str, object], json.loads(_canonical_json_bytes(document)))
        return cls(
            document=detached,
            canonical_sha256=sha256(_canonical_json_bytes(document)).hexdigest(),
            study_id=_name(study.get("study_id"), "campaign_lock.study.study_id"),
            study_kind=study_kind,
            claim_scope=claim_scope,
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


class PortfolioAssay(Protocol):
    """Common Evaluation implementation for the frozen portfolio sequence."""

    protocol_sha256: str

    def prepare(self, campaign_lock: CampaignLock) -> PortfolioArtifact:
        """Build/load the Portfolio after the Terminal Archive already exists."""

    def evaluate(self, campaign_lock: CampaignLock) -> PortfolioEvaluationReceipt:
        """Return typed observations; Lab still owns the Run terminal."""


@dataclass(frozen=True)
class ClaimView:
    """Scope-limited claims derived from one audited Portfolio terminal."""

    heldout_correctness_supported: bool
    dispatcher_correctness_supported: bool
    stable_kernel_performance_supported: bool
    stable_heldout_dispatcher_performance_supported: bool
    arbitrary_shape_generalization_supported: bool
    serving_supported: bool
    paper_result_reproduced: bool


@dataclass(frozen=True)
class PortfolioStudyReport:
    """Portfolio audit with correctness and measurement quality kept orthogonal."""

    study_id: str
    campaign_complete: bool
    archive_integrity_passed: bool
    semantic_replay_passed: bool
    semantic_replay_error: str | None
    protocol_adherence: str
    endpoint: Mapping[str, object] | None
    claim_view: ClaimView
    run_audit: RunAudit | None



class Lab:
    """Resolve, execute and audit preregistered studies over frozen dependencies."""

    def __init__(self, project_root: str | Path) -> None:
        self._root = Path(project_root).resolve(strict=True)

    def preflight(self, study_path: str | Path) -> CampaignLock:
        """Resolve one Study Contract without provider, GPU or evidence side effects."""

        study = StudyContract.load(study_path)
        if study.document["kind"] == "portfolio":
            return self._preflight_portfolio(study)
        workload_ref = _object(study.document.get("workload"), "study.workload")
        if set(workload_ref) != {"path", "canonical_sha256"}:
            raise ValueError("study workload reference fields differ")
        workload_relative, workload_path = _project_path(
            self._root, workload_ref.get("path"), "study.workload.path"
        )
        workload = WorkloadContract.load(workload_path)
        if workload.canonical_sha256 != _digest(
            workload_ref.get("canonical_sha256"), "study.workload.canonical_sha256"
        ):
            raise ValueError("Study Contract workload bytes differ")

        arms = _object(study.document.get("arms"), "study.arms")
        if set(arms) != {"open_cake", "direct_cuda"}:
            raise ValueError("matched_search requires open_cake and direct_cuda arms")
        open_cake = _object(arms.get("open_cake"), "study.arms.open_cake")
        direct_cuda = _object(arms.get("direct_cuda"), "study.arms.direct_cuda")
        if set(open_cake) != {
            "environment_kind",
            "provider",
            "scaffold",
            "prompt_template",
            "compiler_revision",
            "schedule_profile",
            "schedule_skeleton",
            "tool_surface",
            "feedback",
        } or set(direct_cuda) != {
            "environment_kind",
            "provider",
            "scaffold",
            "prompt_template",
            "launch_contract",
            "candidate_skeleton",
            "toolchain_sha256",
            "tool_surface",
            "feedback",
        }:
            raise ValueError("Study Contract Authoring Environment fields differ")
        if open_cake.get("environment_kind") != "open_cake" or direct_cuda.get(
            "environment_kind"
        ) != "direct_cuda":
            raise ValueError("Study Contract Authoring Environment kinds differ")
        if open_cake.get("schedule_profile") != "flash_kmeans_b32_smoke":
            raise ValueError("Study Contract Open Cake Schedule profile differs")
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
            skeleton_document.get("metadata", {}).get("profile")
            != open_cake.get("schedule_profile")
            or _digest(
                schedule_skeleton.get("canonical_sha256"),
                "study.arms.open_cake.schedule_skeleton.canonical_sha256",
            )
            != sha256(_canonical_json_bytes(skeleton_document)).hexdigest()
        ):
            raise ValueError("Study Contract Schedule skeleton bytes or profile differ")
        if open_cake.get("provider") != direct_cuda.get(
            "provider"
        ) or open_cake.get("scaffold") != direct_cuda.get("scaffold"):
            raise ValueError("matched Authoring Environments differ in provider or scaffold")
        _digest(direct_cuda.get("toolchain_sha256"), "study.arms.direct_cuda.toolchain_sha256")
        provider = _object(open_cake.get("provider"), "study.arms.provider")
        if set(provider) != {
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
        }:
            raise ValueError("Study Contract provider configuration fields differ")
        provider_revision = _name(provider.get("revision"), "study.arms.provider.revision")
        if (
            provider.get("model") != "gpt-5.6-sol"
            or provider.get("reasoning_effort") != "max"
            or provider.get("service_tier") != "default"
            or provider.get("sandbox") != "workspace-write"
            or provider.get("cwd_policy") != "independent_empty_workspace"
            or provider.get("reference_visibility") != "embedded_frozen_bundle"
            or provider.get("disabled_features") != list(CODEX_DISABLED_FEATURES)
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
        _, qualification_path = _project_path(
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
                    }
                )
            ).hexdigest()
            or not qualification.initial_and_resume_equivalent
            or not qualification.file_lifecycle_observed
            or not qualification.usage_observed
            or not qualification.qualified
            or qualification_ref.get("canonical_sha256") != qualification.canonical_sha256
        ):
            raise ValueError("provider qualification bytes or capability differs")
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
            _, anchor_path = _project_path(
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
        for arm_name, environment in (("open_cake", open_cake), ("direct_cuda", direct_cuda)):
            prompt = _object(
                environment.get("prompt_template"),
                f"study.arms.{arm_name}.prompt_template",
            )
            if set(prompt) != {"path", "sha256"}:
                raise ValueError("Study Contract prompt reference differs")
            _, prompt_path = _project_path(
                self._root,
                prompt.get("path"),
                f"study.arms.{arm_name}.prompt_template.path",
            )
            if _digest(
                prompt.get("sha256"), f"study.arms.{arm_name}.prompt_template.sha256"
            ) != sha256(prompt_path.read_bytes()).hexdigest():
                raise ValueError("Study Contract prompt bytes differ")
        if open_cake.get("tool_surface") != ["submit_schedule"] or direct_cuda.get(
            "tool_surface"
        ) != ["submit_cuda"]:
            raise ValueError("Study Contract Authoring Environment tool surfaces differ")
        if open_cake.get("feedback") != [
            "findings",
            "correctness",
            "qualified_timing",
        ] or direct_cuda.get("feedback") != [
            "compile",
            "correctness",
            "qualified_timing",
        ]:
            raise ValueError("Study Contract Authoring Environment feedback differs")
        compiler_ref = _object(
            open_cake.get("compiler_revision"), "study.arms.open_cake.compiler_revision"
        )
        if set(compiler_ref) != {"path", "canonical_sha256"}:
            raise ValueError("Compiler Revision reference fields differ")
        compiler_relative, compiler_path = _project_path(
            self._root,
            compiler_ref.get("path"),
            "study.arms.open_cake.compiler_revision.path",
        )
        compiler = Compiler.load(self._root, compiler_path)
        if compiler.state != "released":
            raise ValueError("Study Contract requires a released Compiler Revision")
        gate = compiler.check_corpus()
        compiler_sha = _digest(
            compiler_ref.get("canonical_sha256"),
            "study.arms.open_cake.compiler_revision.canonical_sha256",
        )
        if not gate.passed or gate.compiler_revision_sha256 != compiler_sha:
            raise ValueError("Study Contract Compiler Revision differs or fails its Corpus Gate")

        allocation = _object(study.document.get("allocation"), "study.allocation")
        order = allocation.get("order")
        if allocation.get("method") != "predeclared_balanced_blocks" or not isinstance(order, list):
            raise ValueError("Study Contract allocation differs")
        run_order = tuple(_name(value, "study.allocation.order[]") for value in order)
        claim_scope = cast(str, study.document["claim_scope"])
        expected_arms = (
            ["direct_cuda", "open_cake"]
            if claim_scope == "system_qualification_only"
            else ["direct_cuda"] * 3 + ["open_cake"] * 3
        )
        if len(run_order) != len(set(run_order)) or sorted(
            name.rsplit("-", 1)[0] for name in run_order
        ) != expected_arms:
            required = "one" if claim_scope == "system_qualification_only" else "three"
            raise ValueError(
                f"Study Contract must predeclare {required} independent Run(s) per arm"
            )

        budget = _object(study.document.get("budget"), "study.budget")
        checkpoints = budget.get("checkpoints")
        limit = budget.get("limit")
        maximum_turns = budget.get("maximum_turns")
        if (
            set(budget) != {"unit", "limit", "checkpoints", "maximum_turns"}
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
        ):
            raise ValueError("Study Contract budget grid differs")
        if claim_scope == "system_qualification_only" and (
            checkpoints != [limit] or maximum_turns != 1
        ):
            raise ValueError("system qualification requires one bounded Turn and checkpoint")

        run_protocol = _object(study.document.get("run_protocol"), "study.run_protocol")
        if (
            run_protocol.get("independent_thread") is not True
            or run_protocol.get("empty_workspace") is not True
            or run_protocol.get("automatic_retries") != 0
            or run_protocol.get("replacement_runs") != 0
        ):
            raise ValueError("Study Contract Run Protocol differs")
        evaluation = _object(
            study.document.get("evaluation_protocol"), "study.evaluation_protocol"
        )
        workload.case(_name(evaluation.get("case_id"), "study.evaluation_protocol.case_id"))
        execution = _object(study.document.get("execution"), "study.execution")
        if set(execution) != {
            "target",
            "executor_revision",
            "broker_execution_sha256",
            "gpu",
            "sandbox",
        }:
            raise ValueError("Study Contract execution fields differ")
        if (
            execution.get("target") != "sm_100a"
            or execution.get("sandbox") != "workspace-write"
        ):
            raise ValueError("Study Contract execution authority differs")
        _digest(
            execution.get("broker_execution_sha256"),
            "study.execution.broker_execution_sha256",
        )
        _validate_executor_revision(self._root, execution, "study.execution")
        gpu = _object(execution.get("gpu"), "study.execution.gpu")
        if gpu != {"name": "NVIDIA B200", "count": 1, "mode": "exclusive"}:
            raise ValueError("Study Contract GPU admission differs")
        analysis = _object(study.document.get("analysis_plan"), "study.analysis_plan")
        if claim_scope == "system_qualification_only":
            if analysis != _SYSTEM_QUALIFICATION_ANALYSIS_PLAN:
                raise ValueError("system qualification Analysis Plan differs")
            estimand = None
        else:
            if set(analysis) != {
                "experimental_unit",
                "target_population",
                "primary_endpoint",
                "contrast",
                "estimand",
                "missingness",
                "pooling",
                "availability",
                "summary_statistics",
                "direction",
            }:
                raise ValueError("Study Contract Analysis Plan fields differ")
            if (
                analysis.get("experimental_unit") != "run"
                or analysis.get("primary_endpoint")
                != ["qualified_by_budget", "best_confirmed_latency_ms_if_qualified"]
                or analysis.get("contrast")
                != "open_cake_minus_direct_cuda_descriptive"
                or analysis.get("missingness")
                != {
                    "candidate_failure": "observed_outcome",
                    "external_fault": "missing",
                    "replacement": "forbidden",
                }
                or analysis.get("pooling")
                != "forbidden_without_successor_analysis_plan"
                or analysis.get("availability")
                != "all_prescheduled_runs_qualified_at_final_checkpoint"
                or analysis.get("summary_statistics")
                != {
                    "qualification": "arm_rate",
                    "conditional_latency": "arm_median_ms",
                    "contrast": "direct_cuda_median_divided_by_open_cake_median",
                    "uncertainty": "per_arm_observed_range_ms",
                }
                or analysis.get("direction") != "lower_latency_is_better"
            ):
                raise ValueError("Study Contract Analysis Plan is unsupported")
            _name(analysis.get("target_population"), "study.analysis_plan.target_population")
            estimand = _name(analysis.get("estimand"), "study.analysis_plan.estimand")
        evidence_policy = _object(study.document.get("evidence"), "study.evidence")
        if evidence_policy != {
            "schema_version": 2,
            "terminal_archive_required_for_every_run": True,
        }:
            raise ValueError("Study Contract Evidence policy differs")

        arm_digests = {
            name: sha256(_canonical_json_bytes(value)).hexdigest()
            for name, value in (("open_cake", open_cake), ("direct_cuda", direct_cuda))
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
                "arm_environments": arms,
                "arm_environment_sha256": arm_digests,
                "budget": budget,
                "run_protocol": run_protocol,
                "evidence_policy": evidence_policy,
            },
            "run_order": list(run_order),
            "evaluation_protocol": evaluation,
            "execution": execution,
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

    def _preflight_portfolio(self, study: StudyContract) -> CampaignLock:
        """Resolve the real r43-r45 second use case without adding a runtime mode."""

        workload_ref = _object(study.document["workload"], "study.workload")
        if set(workload_ref) != {"path", "canonical_sha256"}:
            raise ValueError("portfolio Workload reference differs")
        workload_relative, workload_path = _project_path(
            self._root, workload_ref["path"], "study.workload.path"
        )
        workload = WorkloadContract.load(workload_path)
        if workload.canonical_sha256 != _digest(
            workload_ref["canonical_sha256"], "study.workload.canonical_sha256"
        ):
            raise ValueError("portfolio Workload bytes differ")

        compiler_ref = _object(study.document["compiler_revision"], "study.compiler_revision")
        if set(compiler_ref) != {"path", "canonical_sha256"}:
            raise ValueError("portfolio Compiler Revision reference differs")
        compiler_relative, compiler_path = _project_path(
            self._root, compiler_ref["path"], "study.compiler_revision.path"
        )
        compiler = Compiler.load(self._root, compiler_path)
        gate = compiler.check_corpus()
        compiler_sha = _digest(
            compiler_ref["canonical_sha256"], "study.compiler_revision.canonical_sha256"
        )
        if compiler.state != "released" or not gate.passed or gate.compiler_revision_sha256 != compiler_sha:
            raise ValueError("portfolio requires the exact released Compiler Revision")

        seed_ref = _object(study.document["kernel_seed"], "study.kernel_seed")
        if set(seed_ref) != {"path", "canonical_sha256"}:
            raise ValueError("portfolio Kernel Seed reference differs")
        seed_relative, seed_path = _project_path(
            self._root, seed_ref["path"], "study.kernel_seed.path"
        )
        seed = KernelSeed.load(self._root, seed_path)
        if seed.canonical_sha256 != _digest(
            seed_ref["canonical_sha256"], "study.kernel_seed.canonical_sha256"
        ) or seed.workload_sha256 != workload.canonical_sha256:
            raise ValueError("portfolio Kernel Seed bytes or Workload binding differ")

        case_roles = _object(study.document["case_roles"], "study.case_roles")
        if set(case_roles) != {"anchor", "held_out"}:
            raise ValueError("portfolio case roles differ")
        anchor = case_roles.get("anchor")
        held_out = case_roles.get("held_out")
        if (
            not isinstance(anchor, list)
            or len(anchor) != 1
            or not isinstance(held_out, list)
            or len(held_out) != 2
            or any(not isinstance(case, str) or not case for case in anchor + held_out)
            or len(set(anchor + held_out)) != 3
        ):
            raise ValueError("portfolio requires one anchor and two held-out cases")
        for case_id in anchor + held_out:
            workload.case(case_id)
        if anchor != ["headline_b32"] or held_out != ["b32_smoke", "public_b1"]:
            raise ValueError("portfolio frozen case roles differ")

        specialization = _object(
            study.document["specialization_policy"], "study.specialization_policy"
        )
        if specialization != {
            "kind": "exact_shape",
            "seed": "frozen",
            "retuning": "forbidden",
            "tail_policy": "reject",
        }:
            raise ValueError("portfolio specialization policy differs")
        dispatch = _object(study.document["dispatch_policy"], "study.dispatch_policy")
        if dispatch != {
            "key": ["B", "N", "K", "D"],
            "mapping_owner": "portfolio_manifest",
            "unsupported": "reject_before_launch",
            "fallback": "forbidden",
        }:
            raise ValueError("portfolio dispatch policy differs")
        evaluation = _object(
            study.document["evaluation_protocol"], "study.evaluation_protocol"
        )
        if evaluation != {
            "sequence": [
                "direct_preflight",
                "dispatcher_preflight",
                "kernel_timing",
                "dispatcher_timing",
                "postflight",
                "unsupported_probe",
            ],
            "cohorts_per_boundary": 5,
            "samples_per_cohort": 25,
            "maximum_cv": 0.05,
            "l2_flush_bytes": 268435456,
            "kernel_boundary": "cupti_target_kernel",
            "dispatcher_boundary": "host_validation_selection_launch_sync",
            "resampling": "forbidden",
        }:
            raise ValueError("portfolio Evaluation Protocol differs")
        execution = _object(study.document["execution"], "study.execution")
        if set(execution) != {
            "target",
            "executor_revision",
            "gpu",
            "sandbox",
        }:
            raise ValueError("portfolio execution fields differ")
        if execution.get("target") != "sm_100a" or execution.get("sandbox") != "workspace-write":
            raise ValueError("portfolio execution target or sandbox differs")
        _validate_executor_revision(self._root, execution, "study.execution")
        if _object(execution.get("gpu"), "study.execution.gpu") != {
            "name": "NVIDIA B200",
            "count": 1,
            "mode": "exclusive",
        }:
            raise ValueError("portfolio GPU admission differs")
        analysis = _object(study.document["analysis_plan"], "study.analysis_plan")
        if analysis != {
            "experimental_unit": "case_route",
            "estimand": "descriptive_exact_shape_correctness_and_measurement_quality",
            "correctness_scope": "three_frozen_cases_only",
            "measurement_quality": "per_case_per_boundary_cv",
            "cross_shape_contrast": "forbidden",
            "causal_claim": "forbidden",
            "missingness": "any_missing_declared_observation_makes_portfolio_endpoint_unavailable",
            "resampling": "forbidden",
        }:
            raise ValueError("portfolio Analysis Plan differs")
        evidence_policy = _object(study.document["evidence"], "study.evidence")
        if evidence_policy != {
            "schema_version": 2,
            "terminal_archive_required": True,
            "legacy_lineage": [
                "r43_harness_fault",
                "r44_host_contract_fault",
                "r45_unstable_timing",
            ],
            "legacy_claim_map_sha256": "7aae7a64a0638c0bf3ebfbffb88515ae6dce6329e6bb611d21c6052982ae2288",
        }:
            raise ValueError("portfolio Evidence policy differs")

        lock_document: dict[str, object] = {
            "schema_version": 1,
            "study": {
                "study_id": study.study_id,
                "kind": "portfolio",
                "claim_scope": study.document["claim_scope"],
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
                "kernel_seed": {
                    "seed_id": seed.seed_id,
                    "path": seed_relative,
                    "canonical_sha256": seed.canonical_sha256,
                },
                "case_roles": case_roles,
                "specialization_policy": specialization,
                "dispatch_policy": dispatch,
                "evidence_policy": evidence_policy,
            },
            "run_order": ["portfolio-1"],
            "evaluation_protocol": evaluation,
            "execution": execution,
            "analysis_plan": analysis,
            "analysis_plan_sha256": sha256(_canonical_json_bytes(analysis)).hexdigest(),
        }
        return CampaignLock.from_dict(lock_document)

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

        root = Path(evidence_root).absolute()
        if root.exists() or root.is_symlink():
            raise ValueError("Campaign evidence root must be new")
        if set(environments) != {"open_cake", "direct_cuda"}:
            raise ValueError("Campaign Authoring Environment set differs")
        if lock.study_kind != "matched_search":
            raise ValueError("Lab.execute matched-search path requires a matched Campaign Lock")
        _validate_executor_revision(
            self._root,
            _object(lock.document["execution"], "campaign_lock.execution"),
            "campaign_lock.execution",
        )
        resolved_inputs = _object(
            lock.document["resolved_inputs"], "campaign_lock.resolved_inputs"
        )
        budget = _object(resolved_inputs["budget"], "campaign_lock.resolved_inputs.budget")
        checkpoints = cast(list[int], budget["checkpoints"])
        maximum_turns = cast(int, budget["maximum_turns"])
        evaluation_protocol = _object(
            lock.document["evaluation_protocol"], "campaign_lock.evaluation_protocol"
        )
        expected_protocol_sha256 = sha256(
            _canonical_json_bytes(evaluation_protocol)
        ).hexdigest()
        if (
            getattr(evaluator, "protocol", None) != evaluation_protocol
            or getattr(evaluator, "protocol_sha256", None) != expected_protocol_sha256
        ):
            raise ValueError("Run Evaluator does not match the Campaign Lock")
        arms = _object(resolved_inputs["arm_environments"], "resolved_inputs.arm_environments")
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
        _, qualification_path = _project_path(
            self._root,
            qualification_ref["path"],
            "arm_environments.open_cake.provider_qualification.path",
        )
        qualification = ProviderQualificationReceipt.load(qualification_path)
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
        }
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
        for sequence, run_id in enumerate(lock.run_order, start=1):
            arm = run_id.rsplit("-", 1)[0]
            environment = environments[arm]
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
            try:
                for turn_number in range(1, maximum_turns + 1):
                    live_stage = "provider"
                    provider_turn = provider.turn(
                        TurnRequest(
                            run_id,
                            arm,
                            turn_number,
                            cumulative_tokens,
                            thread_id,
                            feedback,
                        )
                    )
                    if thread_id is not None and provider_turn.thread_id != thread_id:
                        raise ValueError("provider resume thread identity differs")
                    thread_id = provider_turn.thread_id
                    cumulative_tokens += provider_turn.provider_tokens
                    events_object = evidence.put(
                        provider_turn.raw_events,
                        media_type="application/x-ndjson",
                    )
                    candidate_object = evidence.put(
                        provider_turn.candidate,
                        media_type=environment.media_type,
                    )
                    if candidate_object.sha256 != provider_turn.candidate_sha256:
                        raise ValueError("provider candidate seal differs")
                    ledger.append(
                        "provider_turn_completed",
                        {
                            "turn": turn_number,
                            "thread_id": thread_id,
                            "turn_provider_tokens": provider_turn.provider_tokens,
                            "cumulative_provider_tokens": cumulative_tokens,
                            "normalization": provider_turn.normalization,
                            "objects": [
                                events_object.reference("provider_events"),
                                candidate_object.reference("candidate_submission"),
                            ],
                        },
                    )
                    submission = CandidateSubmission.seal(
                        environment.media_type, provider_turn.candidate
                    )
                    live_stage = "environment"
                    environment_result = environment.build(submission)
                    if environment_result.disposition == "rejected":
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
                        rejection_payload: dict[str, object] = {
                            "turn": turn_number,
                            "candidate_sha256": submission.sha256,
                            "feedback": dict(feedback),
                        }
                        if environment_result.artifact_payloads:
                            references = []
                            rejected_roles = []
                            for role, payload in sorted(
                                environment_result.artifact_payloads.items()
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
                        artifact_references = []
                        for role, payload in sorted(launchable.artifact_payloads.items()):
                            artifact = evidence.put(
                                payload,
                                media_type=_candidate_artifact_media_type(role),
                            )
                            artifact_references.append(artifact.reference(role))
                        ledger.append(
                            "launchable_candidate_sealed",
                            {
                                "turn": turn_number,
                                "candidate_sha256": launchable.candidate_sha256,
                                "candidate_record_sha256": launchable.canonical_sha256,
                                "objects": artifact_references,
                            },
                        )
                        live_stage = "evaluation"
                        search_attempt = evaluator.evaluate(
                            launchable,
                            case_id=case_id,
                            purpose="search",
                        )
                        search = search_attempt.final_receipt
                        search_attempt_references = _archive_logical_attempt(
                            evidence, search_attempt
                        )
                        ledger.append(
                            "evaluation_attempt_completed",
                            {
                                "turn": turn_number,
                                "purpose": "search",
                                "candidate_sha256": launchable.candidate_sha256,
                                "objects": search_attempt_references,
                            },
                        )
                        if search is None:
                            raise RuntimeError("search Evaluation has no final receipt")
                        _validate_receipt_authority(
                            search,
                            candidate=launchable,
                            workload_sha256=workload_sha256,
                            protocol_sha256=expected_protocol_sha256,
                            case_id=case_id,
                            purpose="search",
                        )
                        search_references = _archive_evaluation_receipt(evidence, search)
                        ledger.append(
                            "candidate_evaluated",
                            {
                                "turn": turn_number,
                                "purpose": "search",
                                "candidate_sha256": launchable.candidate_sha256,
                                "objects": search_references,
                            },
                        )
                        confirmed: EvaluationReceipt | None = None
                        if _receipt_qualifies(search):
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
                                },
                            )
                        qualified = confirmed is not None and _receipt_qualifies(confirmed)
                        latency = _receipt_latency_ms(confirmed) if qualified else None
                        observations.append(
                            TurnObservation(
                                turn_number,
                                cumulative_tokens,
                                launchable.candidate_sha256,
                                qualified,
                                latency,
                            )
                        )
                        feedback = MappingProxyType(
                            {
                                "kind": "evaluation",
                                "candidate_disposition": search.candidate_disposition,
                                "measurement_quality": search.measurement_quality,
                                "confirmed": qualified,
                                "search_latency_ms": _receipt_latency_ms(search),
                                "confirmed_latency_ms": latency,
                            }
                        )
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

            projected = project_checkpoints(
                turns=observations,
                checkpoints=checkpoints,
                terminal_provider_tokens=cumulative_tokens,
            )
            ledger.append(
                "checkpoints_projected",
                {
                    "checkpoints": [
                        {
                            "provider_tokens": item.provider_tokens,
                            "state": item.state,
                            "best_candidate_sha256": item.best_candidate_sha256,
                            "best_confirmed_latency_ms": item.best_confirmed_latency_ms,
                        }
                        for item in projected
                    ]
                },
            )
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

    def execute_portfolio(
        self,
        lock: CampaignLock,
        evidence_root: str | Path,
        *,
        assay: PortfolioAssay,
    ) -> CampaignRef:
        """Execute the sole frozen portfolio Run through Evaluation and Evidence."""

        if lock.study_kind != "portfolio" or lock.run_order != ("portfolio-1",):
            raise ValueError("portfolio execution requires a portfolio Campaign Lock")
        _validate_executor_revision(
            self._root,
            _object(lock.document["execution"], "campaign_lock.execution"),
            "campaign_lock.execution",
        )
        root = Path(evidence_root).absolute()
        if root.exists() or root.is_symlink():
            raise ValueError("Campaign evidence root must be new")
        evidence = EvidenceStore.create(root)
        ledger = evidence.start_run(
            "portfolio-1",
            authority_sha256=lock.canonical_sha256,
            authority=lock.document,
        )
        ledger.append("run_started", {"sequence": 1, "replacement_run": False})
        try:
            if hasattr(assay, "set_observer"):

                def observe(kind: str, payload: Mapping[str, object]) -> None:
                    observation = evidence.put(
                        _canonical_json_bytes(payload), media_type="application/json"
                    )
                    ledger.append(
                        "portfolio_observation",
                        {
                            "observation_kind": kind,
                            "objects": [observation.reference("portfolio_observation")],
                        },
                    )

                assay.set_observer(observe)
            resolved = _object(lock.document["resolved_inputs"], "resolved_inputs")
            seed = _object(resolved["kernel_seed"], "resolved_inputs.kernel_seed")
            workload = _object(lock.document["workload"], "campaign_lock.workload")
            protocol_sha256 = sha256(
                _canonical_json_bytes(lock.document["evaluation_protocol"])
            ).hexdigest()
            artifact = assay.prepare(lock)
            if (
                assay.protocol_sha256 != protocol_sha256
                or artifact.workload_sha256 != workload["canonical_sha256"]
                or artifact.seed_sha256 != seed["canonical_sha256"]
            ):
                raise ValueError("Portfolio Assay does not match the Campaign Lock")
            expected_cases = set(
                cast(list[str], _object(resolved["case_roles"], "case_roles")["anchor"])
                + cast(list[str], _object(resolved["case_roles"], "case_roles")["held_out"])
            )
            if {entry.case_id for entry in artifact.entries} != expected_cases:
                raise ValueError("PortfolioArtifact cases differ from the Campaign Lock")
            artifact_references: list[dict[str, object]] = []
            for entry in artifact.entries:
                candidate = entry.candidate
                if (
                    not _ARM_ARTIFACT_ROLES["open_cake"]
                    <= set(candidate.artifact_payloads)
                    or set(candidate.artifact_payloads) != set(candidate.artifact_roles)
                ):
                    raise ValueError("Portfolio candidate artifact custody is incomplete")
                for role, payload in sorted(candidate.artifact_payloads.items()):
                    item = evidence.put(
                        payload, media_type=_candidate_artifact_media_type(role)
                    )
                    artifact_references.append(
                        item.reference(f"{entry.case_id}_{role}")
                    )
            artifact_object = evidence.put(
                _canonical_json_bytes(artifact.document), media_type="application/json"
            )
            artifact_references.append(
                artifact_object.reference("portfolio_artifact_manifest")
            )
            ledger.append(
                "portfolio_sealed",
                {
                    "portfolio_sha256": artifact.canonical_sha256,
                    "objects": artifact_references,
                },
            )
            receipt = assay.evaluate(lock)
            if (
                receipt.portfolio_sha256 != artifact.canonical_sha256
                or receipt.evaluation_protocol_sha256 != protocol_sha256
            ):
                raise ValueError("Portfolio receipt does not match the Campaign Lock")
            receipt_object = evidence.put(
                _canonical_json_bytes(receipt.document),
                media_type="application/json",
            )
            endpoint = _portfolio_endpoint(receipt)
            ledger.append(
                "portfolio_evaluated",
                {"objects": [receipt_object.reference("portfolio_evaluation_receipt")]},
            )
            ledger.seal(
                protocol_adherence="adhered",
                endpoint_observation="observed",
                endpoint=endpoint,
            )
        except Exception as error:
            fault = (
                error.protocol_adherence
                if isinstance(error, RunProtocolFault)
                else "harness_fault"
            )
            fault_payload: dict[str, object] = {
                "fault": fault,
                "exception_type": type(error).__name__,
            }
            if isinstance(error, RunProtocolFault) and error.artifact_payloads:
                references = []
                rejected = []
                for role, payload in sorted(error.artifact_payloads.items()):
                    try:
                        references.append(
                            evidence.put(payload, media_type="application/octet-stream").reference(
                                role
                            )
                        )
                    except (OSError, ValueError):
                        rejected.append(role)
                if references:
                    fault_payload["objects"] = references
                if rejected:
                    fault_payload["artifact_rejections"] = rejected
            ledger.append("run_fault", fault_payload)
            ledger.seal(
                protocol_adherence=fault,
                endpoint_observation="missing",
            )
        return CampaignRef(lock=lock, evidence_root=evidence.root)

    def audit_portfolio(self, campaign: CampaignRef) -> PortfolioStudyReport:
        """Derive the bounded r45-style Claim View from one terminal archive."""

        if campaign.lock.study_kind != "portfolio":
            raise ValueError("portfolio audit requires a portfolio Campaign Lock")
        store = EvidenceStore.open(campaign.evidence_root)
        try:
            audit = store.audit_run("portfolio-1")
        except (OSError, ValueError, json.JSONDecodeError):
            audit = None
        complete = audit is not None and audit.authority_sha256 == campaign.lock.canonical_sha256
        integrity = bool(complete and audit is not None and audit.integrity)
        semantic_replay = False
        semantic_replay_error: str | None = None
        endpoint: Mapping[str, object] | None = None
        if integrity and audit is not None:
            try:
                events = store.replay_events("portfolio-1")

                def object_for(kind: str, role: str) -> Mapping[str, object]:
                    matching = [event for event in events if event.get("kind") == kind]
                    if len(matching) != 1:
                        raise ValueError(f"portfolio event {kind!r} coverage differs")
                    payload = _object(matching[0].get("payload"), f"event.{kind}.payload")
                    objects = payload.get("objects")
                    if not isinstance(objects, list):
                        raise ValueError(f"portfolio event {kind!r} objects differ")
                    selected = [item for item in objects if isinstance(item, Mapping) and item.get("role") == role]
                    if len(selected) != 1:
                        raise ValueError(f"portfolio object role {role!r} coverage differs")
                    return cast(Mapping[str, object], selected[0])

                artifact_document = json.loads(
                    store.read_object(
                        object_for("portfolio_sealed", "portfolio_artifact_manifest")
                    )
                )
                artifact = PortfolioArtifact.from_document(artifact_document)
                for entry in artifact.entries:
                    for role, expected_sha256 in entry.candidate.artifact_roles.items():
                        reference = object_for(
                            "portfolio_sealed", f"{entry.case_id}_{role}"
                        )
                        payload = store.read_object(reference)
                        if (
                            reference.get("sha256") != expected_sha256
                            or sha256(payload).hexdigest() != expected_sha256
                        ):
                            raise ValueError(
                                "replayed PortfolioArtifact object lineage differs"
                            )
                resolved = _object(
                    campaign.lock.document["resolved_inputs"], "resolved_inputs"
                )
                seed = _object(resolved["kernel_seed"], "resolved_inputs.kernel_seed")
                workload = _object(campaign.lock.document["workload"], "workload")
                if (
                    artifact.seed_sha256 != seed["canonical_sha256"]
                    or artifact.workload_sha256 != workload["canonical_sha256"]
                ):
                    raise ValueError("replayed PortfolioArtifact authority differs")
                protocol_sha256 = sha256(
                    _canonical_json_bytes(campaign.lock.document["evaluation_protocol"])
                ).hexdigest()
                receipt_document = json.loads(
                    store.read_object(
                        object_for(
                            "portfolio_evaluated", "portfolio_evaluation_receipt"
                        )
                    )
                )
                receipt = replay_portfolio_receipt(
                    receipt_document,
                    artifact,
                    expected_protocol_sha256=protocol_sha256,
                )
                endpoint = _portfolio_endpoint(receipt)
                if audit.endpoint != endpoint:
                    raise ValueError("portfolio terminal endpoint differs from replay")
                semantic_replay = True
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                semantic_replay = False
                semantic_replay_error = f"{type(error).__name__}: {error}"
        correctness = endpoint.get("correctness_by_case") if endpoint is not None else None
        kernel_quality = (
            endpoint.get("kernel_measurement_quality_by_case") if endpoint is not None else None
        )
        dispatcher_quality = (
            endpoint.get("dispatcher_measurement_quality_by_case") if endpoint is not None else None
        )
        all_correct = (
            isinstance(correctness, Mapping)
            and len(correctness) == 3
            and all(value is True for value in correctness.values())
        )
        all_kernel_stable = (
            isinstance(kernel_quality, Mapping)
            and len(kernel_quality) == 3
            and all(value == "stable" for value in kernel_quality.values())
        )
        resolved = _object(campaign.lock.document["resolved_inputs"], "resolved_inputs")
        case_roles = _object(resolved["case_roles"], "resolved_inputs.case_roles")
        held_out = cast(list[str], case_roles["held_out"])
        heldout_dispatch_stable = isinstance(dispatcher_quality, Mapping) and all(
            dispatcher_quality.get(case_id) == "stable" for case_id in held_out
        )
        adhered = audit is not None and audit.protocol_adherence == "adhered"
        supported = integrity and adhered and semantic_replay
        claim_view = ClaimView(
            heldout_correctness_supported=supported and all_correct,
            dispatcher_correctness_supported=supported and all_correct,
            stable_kernel_performance_supported=supported and all_correct and all_kernel_stable,
            stable_heldout_dispatcher_performance_supported=(
                supported and all_correct and heldout_dispatch_stable
            ),
            arbitrary_shape_generalization_supported=False,
            serving_supported=False,
            paper_result_reproduced=False,
        )
        return PortfolioStudyReport(
            study_id=campaign.lock.study_id,
            campaign_complete=complete,
            archive_integrity_passed=integrity,
            semantic_replay_passed=semantic_replay,
            semantic_replay_error=semantic_replay_error,
            protocol_adherence=audit.protocol_adherence if audit is not None else "missing",
            endpoint=endpoint,
            claim_view=claim_view,
            run_audit=audit,
        )

    def _replay_matched_run(
        self,
        evidence: EvidenceStore,
        audit: RunAudit,
        lock: CampaignLock,
    ) -> bool:
        events = evidence.replay_events(audit.run_id)
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
            return (
                fault_payload.get("fault") == audit.protocol_adherence
                and audit.endpoint_observation == "missing"
                and audit.endpoint is None
                and isinstance(checkpoints, list)
                and bool(checkpoints)
                and all(
                    isinstance(item, Mapping) and item.get("state") == "unreached"
                    for item in checkpoints
                )
            )
        threads: set[str] = set()
        cumulative_by_turn: dict[int, int] = {}
        provider_candidate_by_turn: dict[int, str] = {}
        prior_cumulative = 0
        for expected_turn, event in enumerate(provider_events, start=1):
            payload = _object(event.get("payload"), "provider_turn.payload")
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
            by_role = {
                str(item.get("role")): cast(Mapping[str, object], item)
                for item in objects
                if isinstance(item, Mapping) and isinstance(item.get("role"), str)
            }
            if (
                len(objects) != 2
                or len(by_role) != 2
                or set(by_role) != {"provider_events", "candidate_submission"}
            ):
                return False
            raw_events = evidence.read_object(by_role["provider_events"])
            candidate = evidence.read_object(by_role["candidate_submission"])
            if (
                sha256(raw_events).hexdigest()
                != by_role["provider_events"].get("sha256")
                or sha256(candidate).hexdigest()
                != by_role["candidate_submission"].get("sha256")
            ):
                return False
            expected_terminal = json.dumps(
                {
                    "arm": audit.run_id.rsplit("-", 1)[0],
                    "candidate_written": True,
                    "kind": "open_cake_ir_turn",
                    "tool_calls": 1,
                    "turn": expected_turn,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            parsed = parse_codex_turn_events(
                raw_events,
                expected_terminal_message=expected_terminal,
            )
            turn_tokens = payload.get("turn_provider_tokens")
            if (
                parsed.thread_id != thread_id
                or turn_tokens != parsed.provider_tokens
                or cumulative != prior_cumulative + turn_tokens
            ):
                return False
            expected_change = "add" if expected_turn == 1 else "update"
            expected_name = "candidate.json" if audit.run_id.startswith("open_cake-") else "candidate.cu"
            if (
                parsed.change_kind != expected_change
                or Path(parsed.candidate_path).name != expected_name
                or payload.get("normalization") != parsed.normalization
            ):
                return False
            threads.add(thread_id)
            cumulative_by_turn[expected_turn] = cumulative
            provider_candidate_by_turn[expected_turn] = sha256(candidate).hexdigest()
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
        candidate_by_turn: dict[int, str] = {}
        qualified_by_turn: dict[int, float] = {}
        attempt_events = [
            event for event in events if event.get("kind") == "evaluation_attempt_completed"
        ]
        attempt_payloads: dict[tuple[int, str], Mapping[str, object]] = {}
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
                or purpose not in {"search", "confirmatory"}
                or not isinstance(candidate_sha256, str)
                or _DIGEST.fullmatch(candidate_sha256) is None
                or not isinstance(payload.get("objects"), list)
            ):
                return False
            key = (turn, cast(str, purpose))
            if key in attempt_payloads:
                return False
            attempt_payloads[key] = payload
        replayed_attempts: set[tuple[int, str]] = set()
        launchable_events = [
            event for event in events if event.get("kind") == "launchable_candidate_sealed"
        ]
        arm = audit.run_id.rsplit("-", 1)[0]
        launchables_by_turn: dict[int, LaunchableCandidate] = {}
        for event in launchable_events:
            payload = _object(event.get("payload"), "launchable.payload")
            turn = payload.get("turn")
            candidate_sha256 = payload.get("candidate_sha256")
            if (
                not isinstance(turn, int)
                or isinstance(turn, bool)
                or turn <= 0
                or turn in launchables_by_turn
                or not isinstance(candidate_sha256, str)
                or _DIGEST.fullmatch(candidate_sha256) is None
            ):
                return False
            launchables_by_turn[turn] = _replay_launchable_candidate(
                evidence,
                launchable_events,
                turn=turn,
                candidate_sha256=candidate_sha256,
                arm=arm,
            )
        for event in events:
            kind = event.get("kind")
            payload = _object(event.get("payload"), f"event.{kind}.payload")
            if kind == "candidate_rejected":
                turn = int(payload["turn"])
                candidate_by_turn[turn] = str(payload["candidate_sha256"])
            elif kind == "candidate_evaluated":
                turn = int(payload["turn"])
                candidate_sha256 = str(payload["candidate_sha256"])
                candidate_by_turn[turn] = candidate_sha256
                objects = payload.get("objects")
                if not isinstance(objects, list):
                    return False
                receipt_refs = [
                    item
                    for item in objects
                    if isinstance(item, Mapping) and item.get("role") == "evaluation_receipt"
                ]
                required_roles = {
                    "correctness_output",
                    "launch_receipt",
                    "timing_samples",
                    "evaluation_receipt",
                }
                if len(receipt_refs) != 1 or not required_roles <= {
                    str(item.get("role")) for item in objects if isinstance(item, Mapping)
                }:
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
                launchable = launchables_by_turn.get(turn)
                if (
                    launchable is None
                    or launchable.candidate_sha256 != candidate_sha256
                ):
                    return False
                raw_payloads: dict[str, bytes] = {}
                for role in ("correctness_output", "launch_receipt", "timing_samples"):
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
                expected_raw = receipt.get("artifact_payload_sha256")
                if not isinstance(expected_raw, Mapping) or expected_raw != {
                    role: sha256(raw).hexdigest()
                    for role, raw in sorted(raw_payloads.items())
                }:
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
                if validated_receipt.canonical_sha256 != sha256(
                    _canonical_json_bytes(receipt)
                ).hexdigest():
                    return False
                attempt_key = (turn, str(payload.get("purpose")))
                attempt_payload = attempt_payloads.get(attempt_key)
                if (
                    attempt_payload is None
                    or attempt_payload.get("candidate_sha256") != candidate_sha256
                ):
                    return False
                replayed_attempts.add(attempt_key)
                _replay_evaluation_attempt_event(
                    evidence,
                    attempt_payload,
                    candidate=launchable,
                    protocol_sha256=protocol_sha256,
                    final_receipt=validated_receipt,
                )
                if payload.get("purpose") == "confirmatory":
                    timing = validated_receipt.timing
                    if (
                        validated_receipt.correctness_passed
                        and validated_receipt.kernel_calls == 1
                        and validated_receipt.fallback_calls == 0
                        and isinstance(timing, Mapping)
                        and timing.get("measurement_quality_passed") is True
                    ):
                        latency = timing.get("pooled_median_ms")
                        if not isinstance(latency, (int, float)) or float(latency) <= 0:
                            return False
                        qualified_by_turn[turn] = float(latency)
        unreplayed_attempts = set(attempt_payloads) - replayed_attempts
        if unreplayed_attempts:
            faults = [event for event in events if event.get("kind") == "run_fault"]
            if len(unreplayed_attempts) != 1 or len(faults) != 1:
                return False
            turn, purpose = next(iter(unreplayed_attempts))
            if (
                turn != max(provider_candidate_by_turn)
                or (purpose == "confirmatory" and (turn, "search") not in replayed_attempts)
            ):
                return False
            attempt_payload = attempt_payloads[(turn, purpose)]
            launchable = launchables_by_turn.get(turn)
            if (
                launchable is None
                or launchable.candidate_sha256
                != attempt_payload["candidate_sha256"]
            ):
                return False
            _replay_evaluation_attempt_event(
                evidence,
                attempt_payload,
                candidate=launchable,
                protocol_sha256=protocol_sha256,
                final_receipt=None,
            )
        observations = []
        unmatched_provider_turns = set(provider_candidate_by_turn) - set(candidate_by_turn)
        if (
            any(
                provider_candidate_by_turn.get(turn) != launchable.candidate_sha256
                for turn, launchable in launchables_by_turn.items()
            )
            or any(
                provider_candidate_by_turn.get(turn) != candidate_sha256
                for turn, candidate_sha256 in candidate_by_turn.items()
            )
            or unmatched_provider_turns
            and (
                unmatched_provider_turns != {max(provider_candidate_by_turn)}
                or len([event for event in events if event.get("kind") == "run_fault"])
                != 1
            )
        ):
            return False
        for turn, candidate_sha256 in sorted(candidate_by_turn.items()):
            if turn not in cumulative_by_turn:
                return False
            observations.append(
                TurnObservation(
                    turn,
                    cumulative_by_turn[turn],
                    candidate_sha256,
                    turn in qualified_by_turn,
                    qualified_by_turn.get(turn),
                )
            )
        if [item.turn for item in observations] != list(range(1, len(observations) + 1)):
            return False
        resolved = _object(lock.document["resolved_inputs"], "resolved_inputs")
        budget = _object(resolved["budget"], "resolved_inputs.budget")
        projected = project_checkpoints(
            turns=observations,
            checkpoints=cast(list[int], budget["checkpoints"]),
            terminal_provider_tokens=max(cumulative_by_turn.values()),
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
        if checkpoint_payload.get("checkpoints") != expected_projection:
            return False
        expected_observation, expected_endpoint = _matched_endpoint_from_checkpoint(
            projected[-1], audit.protocol_adherence
        )
        return (
            audit.endpoint_observation == expected_observation
            and audit.endpoint == expected_endpoint
        )

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
        archive_integrity_passed = campaign_complete and all(audit.integrity for audit in audits)
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
        missing_run_count = sum(
            audit.endpoint_observation == "missing" or audit.protocol_adherence != "adhered"
            for audit in audits
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
                        "archive_integrity": audit.integrity,
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
                semantic_replay_passed=semantic_replay_passed,
                estimand_available=False,
                missing_run_count=missing_run_count,
                estimate=None,
                uncertainty=None,
                descriptive=descriptive,
                run_inclusion=inclusions,
                run_audits=tuple(audits),
            )
        arm_runs: dict[str, list[RunAudit]] = {"open_cake": [], "direct_cuda": []}
        inclusions: list[AnalysisInclusion] = []
        for audit in audits:
            arm = audit.run_id.rsplit("-", 1)[0]
            if arm not in arm_runs:
                raise ValueError(f"Run {audit.run_id!r} has an unknown assigned arm")
            arm_runs[arm].append(audit)
            if not audit.integrity:
                inclusions.append(
                    AnalysisInclusion(audit.run_id, False, False, "archive_integrity")
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
        estimand_available = (
            archive_integrity_passed
            and semantic_replay_passed
            and missing_run_count == 0
            and campaign.lock.analysis_plan.get("availability")
            == "all_prescheduled_runs_qualified_at_final_checkpoint"
            and all(audit.endpoint_observation == "qualified" for audit in audits)
        )
        qualification_rate: dict[str, float] = {}
        medians: dict[str, float | None] = {}
        ranges: dict[str, list[float] | None] = {}
        latencies: dict[str, dict[str, float]] = {}
        for arm, arm_audits in arm_runs.items():
            qualified = [audit for audit in arm_audits if audit.endpoint_observation == "qualified"]
            qualification_rate[arm] = len(qualified) / len(arm_audits) if arm_audits else 0.0
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
        paired: list[dict[str, object]] = []
        for repetition in sorted(set(latencies["open_cake"]) | set(latencies["direct_cuda"])):
            open_latency = latencies["open_cake"].get(repetition)
            cuda_latency = latencies["direct_cuda"].get(repetition)
            paired.append(
                {
                    "repetition": int(repetition) if repetition.isdigit() else repetition,
                    "open_cake_latency_ms": open_latency,
                    "direct_cuda_latency_ms": cuda_latency,
                    "open_cake_speedup": (
                        cuda_latency / open_latency
                        if open_latency is not None and cuda_latency is not None
                        else None
                    ),
                }
            )
        descriptive: Mapping[str, object] = {
            "qualification_rate": qualification_rate,
            "median_confirmed_latency_ms": medians,
            "paired_runs": paired,
        }
        estimate: Mapping[str, object] | None = None
        uncertainty: Mapping[str, object] | None = None
        if estimand_available:
            estimate = {
                "qualification_rate": qualification_rate,
                "median_confirmed_latency_ms": medians,
                "ratio_of_arm_medians": cast(float, medians["direct_cuda"])
                / cast(float, medians["open_cake"]),
            }
            uncertainty = {"latency_range_ms": ranges}
        return StudyReport(
            study_id=campaign.lock.study_id,
            claim_scope=campaign.lock.claim_scope,
            system_qualification_passed=None,
            estimand=campaign.lock.estimand,
            campaign_complete=campaign_complete,
            archive_integrity_passed=archive_integrity_passed,
            semantic_replay_passed=semantic_replay_passed,
            estimand_available=estimand_available,
            missing_run_count=missing_run_count,
            estimate=estimate,
            uncertainty=uncertainty,
            descriptive=descriptive,
            run_inclusion=tuple(inclusions),
            run_audits=tuple(audits),
        )
