"""Separate writers and independent readers for retained execution artifacts."""

from __future__ import annotations

import json
import re
from hashlib import sha256
from typing import Callable, Mapping, cast

from open_cake_ir.evaluation import EvaluationReceipt, LaunchableCandidate, LogicalEvaluationAttempt
from open_cake_ir.evaluation.core import _plain_json as _evaluation_plain_json
from open_cake_ir.evaluation.paired import validate_paired_broker, validate_receipt_policy
from open_cake_ir.evidence import EvidenceStore
from open_cake_ir.evidence.store import RunLedger

from ._documents import _DIGEST, _canonical_json_bytes, _object
from .faults import RunProtocolFault
from .providers import CANDIDATE_SET_ENVELOPE_V1, ProviderTurn, _project_candidate_submission


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


def _replay_evaluation_receipt(
    evidence: EvidenceStore,
    payload: Mapping[str, object],
    *,
    launchable: LaunchableCandidate | None,
    candidate_sha256: str,
    workload_sha256: str,
    protocol_sha256: str,
    case_id: str,
    purpose: str,
    evaluation_protocol: Mapping[str, object],
    fixed_baseline: Mapping[str, object] | None,
) -> EvaluationReceipt | None:
    """Reconstruct a receipt from independent archived raw-role objects."""
    objects = payload.get("objects")
    if not isinstance(objects, list):
        return None
    receipt_refs = [
        item
        for item in objects
        if isinstance(item, Mapping) and item.get("role") == "evaluation_receipt"
    ]
    if len(receipt_refs) != 1:
        return None
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
        return None
    if launchable is None:
        return None
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
        return None
    raw_payloads: dict[str, bytes] = {}
    for role in sorted(expected_raw):
        matching = [
            item
            for item in objects
            if isinstance(item, Mapping) and item.get("role") == role
        ]
        if len(matching) != 1:
            return None
        raw_payloads[role] = evidence.read_object(
            cast(Mapping[str, object], matching[0])
        )
    if expected_raw != {
        role: sha256(raw).hexdigest()
        for role, raw in sorted(raw_payloads.items())
    }:
        return None
    if {
        item.get("role")
        for item in objects
        if isinstance(item, Mapping)
    } != {"evaluation_receipt", *expected_raw}:
        return None
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
    validate_receipt_policy(validated_receipt, evaluation_protocol,
        fixed_baseline, launchable)
    if validated_receipt.canonical_sha256 != sha256(
        _canonical_json_bytes(receipt)
    ).hexdigest():
        return None
    return validated_receipt


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
