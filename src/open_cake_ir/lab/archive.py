"""Write retained artifacts and validate their shared receipt authority."""

from __future__ import annotations

from hashlib import sha256
from typing import Mapping

from open_cake_ir.evaluation import (
    EvaluationReceipt,
    LaunchableCandidate,
    LogicalEvaluationAttempt,
)
from open_cake_ir.evaluation.artifacts import executable_role, required_build_roles
from open_cake_ir.evaluation.core import _plain_json as _evaluation_plain_json
from open_cake_ir.evaluation.paired import validate_receipt_policy
from open_cake_ir.evidence import EvidenceStore
from open_cake_ir.evidence.store import RunLedger

from ._documents import _canonical_json_bytes
from .faults import RunProtocolFault
from .providers import CANDIDATE_SET_ENVELOPE_V1, ProviderTurn, _project_candidate_submission


def _arm_artifact_roles(arm: str, target: str) -> frozenset[str]:
    """Arm owns source provenance; backend owns its actual compiled products."""
    if executable_role(target) == "metal_binary_archive":
        if arm != "open_cake":
            raise ValueError("Authoring Environment and compiled target differ")
        return required_build_roles("metal") | {"lowered_source"}
    if arm == "open_cake":
        return required_build_roles("triton") | {"lowered_source"}
    if arm in {"direct_cuda", "native_triton", "native_cute_dsl"}:
        return required_build_roles("cuda" if arm == "direct_cuda" else "triton") | {"authored_source"}
    raise ValueError("Authoring Environment and compiled target differ")


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
    if role == "metal_binary_archive":
        return "application/octet-stream"
    if role in {"launch_manifest", "metal_build_report"}:
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
        if not {"broker_record", "evaluator_result", "evaluator_request", "stdout", "stderr"} <= set(
            broker_attempt.artifact_payloads
        ):
            raise ValueError("BrokerAttempt raw artifact custody is incomplete")
        for role, payload in sorted(broker_attempt.artifact_payloads.items()):
            media_type = "application/json" if role in {"broker_record", "evaluator_result", "evaluator_request"} else "text/plain"
            item = evidence.put(payload, media_type=media_type)
            references.append(item.reference(f"attempt_{index}_{role}"))
    document = evidence.put(
        _canonical_json_bytes(_logical_attempt_document(attempt)),
        media_type="application/json",
    )
    references.append(document.reference("broker_attempt_ledger"))
    return references

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

def _archive_provider_turn(
    *,
    arm: str,
    candidate_media_type: str,
    cumulative_tokens: int,
    evidence: EvidenceStore,
    ledger: RunLedger,
    maximum_candidates_per_turn: int,
    provider_document: Mapping[str, object],
    provider_turn: ProviderTurn,
    thread_id: str,
    turn_number: int,
) -> None:
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
        sealed = evidence.put(payload, media_type=candidate_media_type)
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
