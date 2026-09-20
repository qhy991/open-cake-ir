"""Independently validate retained broker attempts and recovery accounting."""

from __future__ import annotations

import json, re
from hashlib import sha256
from typing import Mapping, cast

from open_cake_ir.evaluation import EvaluationReceipt, LaunchableCandidate
from open_cake_ir.evaluation.paired import validate_paired_broker
from open_cake_ir.evidence import EvidenceStore

from .._documents import _DIGEST, _object
from .refusals import refuse

from open_cake_ir.evaluation.attempts import valid_job_mode


def _replay_broker_attempt_ledger(
    evidence: EvidenceStore,
    references: list[object],
    document: Mapping[str, object],
    *,
    candidate: LaunchableCandidate,
    protocol_sha256: str,
    compiler_reference: Mapping[str, object],
    final_receipt: EvaluationReceipt | None,
    purpose: str,
    case_id: str,
    location: str = "evaluation_attempt_completed",
) -> None:
    """Rebuild every broker attempt from retained raw results and compare its ledger.

    `location` names the attempt event whose ledger and raw objects these are.
    """
    ledger = f"{location}.broker_attempt_ledger"

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
    if set(document) != root_fields:
        refuse(ledger, "broker attempt ledger authority differs", observed=set(document), expected=root_fields)
    if document.get("candidate_sha256") != candidate.candidate_sha256:
        refuse(f"{ledger}.candidate_sha256", "broker attempt ledger authority differs",
               observed=document.get("candidate_sha256"), expected=candidate.candidate_sha256)
    attempts = document.get("attempts")
    if not isinstance(attempts, list) or len(attempts) not in {1, 2}:
        refuse(f"{ledger}.attempts", "broker attempt ledger cardinality differs",
               observed=len(attempts) if isinstance(attempts, list) else type(attempts).__name__,
               expected="1 or 2")
    expected_final_sha256 = (
        final_receipt.canonical_sha256 if final_receipt is not None else None
    )
    if document.get("final_receipt_sha256") != expected_final_sha256:
        refuse(f"{ledger}.final_receipt_sha256", "broker attempt ledger final receipt differs",
               observed=document.get("final_receipt_sha256"), expected=expected_final_sha256)

    by_role: dict[str, Mapping[str, object]] = {}
    for index, value in enumerate(references):
        if not isinstance(value, Mapping):
            refuse(f"{location}.objects[{index}]", "broker attempt object reference differs",
                   observed=type(value).__name__)
        role = value.get("role")
        if not isinstance(role, str) or role in by_role:
            refuse(f"{location}.objects[{index}].role", "broker attempt object roles differ", observed=role)
        by_role[role] = cast(Mapping[str, object], value)

    expected_reference_roles = {"broker_attempt_ledger"}
    authorities: list[tuple[object, ...]] = []
    job_ids: set[object] = set()
    receipt_attempts: list[int] = []
    for index, value in enumerate(attempts, start=1):
        row = f"{ledger}.attempts[{index - 1}]"
        attempt = _object(value, f"broker_attempts[{index - 1}]")
        if set(attempt) != attempt_fields:
            refuse(row, "broker attempt ledger fields differ", observed=set(attempt), expected=attempt_fields)
        artifact_digests = attempt.get("artifact_payload_sha256")
        receipt_sha256 = attempt.get("receipt_sha256")
        expected_artifact_roles = {
            "broker_record",
            "evaluator_result",
            "evaluator_request",
            "stdout",
            "stderr",
        }
        if receipt_sha256 is not None:
            receipt_attempts.append(index)
        if (
            not isinstance(artifact_digests, Mapping)
            or set(artifact_digests) != expected_artifact_roles
        ):
            refuse(f"{row}.artifact_payload_sha256", "broker attempt raw artifact roles differ",
                   observed=set(artifact_digests) if isinstance(artifact_digests, Mapping) else artifact_digests,
                   expected=expected_artifact_roles)
        raw_payloads: dict[str, bytes] = {}
        for artifact_role in sorted(expected_artifact_roles):
            role = f"attempt_{index}_{artifact_role}"
            expected_reference_roles.add(role)
            reference = by_role.get(role)
            expected_digest = artifact_digests.get(artifact_role)
            if reference is None or expected_digest != reference.get("sha256"):
                refuse(f"{location}.objects.{role}", "broker attempt raw artifact reference differs",
                       observed=None if reference is None else reference.get("sha256"),
                       expected=expected_digest)
            raw_payloads[artifact_role] = evidence.read_object(reference)
            if sha256(raw_payloads[artifact_role]).hexdigest() != expected_digest:
                refuse(f"{location}.objects.{role}", "broker attempt raw artifact bytes differ")
        raw = f"{location}.objects.attempt_{index}"

        request = _object(json.loads(raw_payloads["evaluator_request"]), "evaluator_request")
        if request.get("compiler_revision") != dict(compiler_reference):
            refuse(f"{raw}_evaluator_request.compiler_revision",
                   "worker Compiler dependency differs from Campaign Lock",
                   observed=request.get("compiler_revision"), expected=dict(compiler_reference))
        for field, expected in (
            ("candidate_sha256", candidate.candidate_sha256),
            ("launch_spec_sha256", candidate.launch_spec_sha256),
            ("evaluation_protocol_sha256", protocol_sha256),
            ("attempt", index),
            ("purpose", purpose),
            ("case_id", case_id),
        ):
            if request.get(field) != expected or (field == "attempt" and type(request.get(field)) is not int):
                refuse(f"{raw}_evaluator_request.{field}", "worker request authority differs",
                       observed=request.get(field), expected=expected)
        authority_document = dict(request)
        authority_document.pop("attempt")
        from .._documents import _canonical_json_bytes
        if sha256(_canonical_json_bytes(authority_document)).hexdigest() != attempt["evaluator_arguments_sha256"]:
            refuse(f"{row}.evaluator_arguments_sha256", "worker request differs from broker argument identity")

        try:
            broker_result = json.loads(raw_payloads["broker_record"])
        except (UnicodeError, json.JSONDecodeError) as error:
            refuse(f"{raw}_broker_record", f"broker attempt raw result is not JSON: {error}")
        result = _object(broker_result, "broker_record")
        if set(result) != result_fields or result.get("schema_version") != 1:
            refuse(f"{raw}_broker_record", "broker attempt raw result fields differ",
                   observed=set(result), expected=result_fields)
        counters = _object(result.get("counters"), "broker_record.counters")
        if set(counters) != set(counter_fields) or any(
            not isinstance(counters.get(field), int)
            or isinstance(counters.get(field), bool)
            or cast(int, counters[field]) < 0
            for field in counter_fields
        ):
            refuse(f"{raw}_broker_record.counters", "broker attempt raw counters differ",
                   observed=dict(counters), expected=list(counter_fields))
        if (
            not isinstance(attempt.get("job_id"), str)
            or not valid_job_mode(cast(str, attempt["job_id"]), cast(str, attempt.get("mode")))
            or not isinstance(attempt.get("admitted"), bool)
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
        ):
            refuse(row, "broker attempt raw observation differs from ledger: the ledger row is not closed",
                   observed={field: attempt.get(field)
                             for field in ("job_id", "mode", "admitted", "error", *counter_fields)})
        for field in ("job_id", "mode", "admitted", "error", *counter_fields):
            observed = counters.get(field) if field in counter_fields else result.get(field)
            if observed != attempt.get(field) or (field == "admitted" and observed is not attempt.get(field)):
                refuse(f"{row}.{field}", "broker attempt raw observation differs from ledger",
                       observed=attempt.get(field), expected=observed)
        if result.get("failure_class") is not None and not isinstance(
            result.get("failure_class"), str
        ):
            refuse(f"{raw}_broker_record.failure_class", "broker attempt failure class differs",
                   observed=result.get("failure_class"))
        if result.get("error") is None and result.get("failure_class") is not None:
            refuse(f"{raw}_broker_record.failure_class", "broker attempt failure class contradicts success",
                   observed=result.get("failure_class"), expected=None)
        for field, expected in (
            ("candidate_sha256", candidate.candidate_sha256),
            ("manifest_sha256", candidate.launch_spec_sha256),
            ("policy_sha256", protocol_sha256),
        ):
            if attempt.get(field) != expected:
                refuse(f"{row}.{field}", "broker attempt execution authority differs",
                       observed=attempt.get(field), expected=expected)
        if (
            not isinstance(attempt.get("evaluator_arguments_sha256"), str)
            or _DIGEST.fullmatch(cast(str, attempt["evaluator_arguments_sha256"]))
            is None
        ):
            refuse(f"{row}.evaluator_arguments_sha256", "broker attempt execution authority differs",
                   observed=attempt.get("evaluator_arguments_sha256"))

        raw_receipt = result.get("receipt")
        if raw_receipt is None:
            if receipt_sha256 is not None:
                refuse(f"{row}.receipt_sha256", "broker attempt receipt absence differs",
                       observed=receipt_sha256, expected=None)
        else:
            if final_receipt is None or receipt_sha256 != final_receipt.canonical_sha256:
                refuse(f"{row}.receipt_sha256", "broker attempt receipt seal differs",
                       observed=receipt_sha256,
                       expected=None if final_receipt is None else final_receipt.canonical_sha256)
            receipt = _object(raw_receipt, "broker_record.receipt")
            artifacts = _object(receipt.get("artifacts"), "broker_record.receipt.artifacts")
            expected_receipt_artifacts = (
                {"correctness_output", "launch_receipt", "profile"}
                if final_receipt.purpose == "attribution"
                else {"correctness_output", "launch_receipt", "timing_samples"}
            )
            raw_receipt_location = f"{raw}_broker_record.receipt"
            if set(receipt) != receipt_fields:
                refuse(raw_receipt_location, "broker attempt raw receipt differs",
                       observed=set(receipt), expected=receipt_fields)
            if (
                set(artifacts) != expected_receipt_artifacts
                or any(not isinstance(path, str) or not path for path in artifacts.values())
                or len(set(artifacts.values())) != len(expected_receipt_artifacts)
            ):
                refuse(f"{raw_receipt_location}.artifacts", "broker attempt raw receipt differs",
                       observed=dict(artifacts), expected=expected_receipt_artifacts)
            if result.get("admitted") is not True or result.get("error") is not None:
                refuse(f"{raw}_broker_record", "broker attempt raw receipt differs: a receipt on an unadmitted or failed attempt",
                       observed={"admitted": result.get("admitted"), "error": result.get("error")})
            for field, expected in (
                ("correctness_passed", final_receipt.correctness_passed),
                ("correctness", final_receipt.correctness),
                ("kernel_calls", final_receipt.kernel_calls),
                ("fallback_calls", final_receipt.fallback_calls),
                ("timing", final_receipt.timing),
            ):
                if receipt.get(field) != expected or (
                    field == "correctness_passed" and receipt.get(field) is not expected
                ):
                    refuse(f"{raw_receipt_location}.{field}", "broker attempt raw receipt differs",
                           observed=receipt.get(field), expected=expected)
            validate_paired_broker(final_receipt, str(result["job_id"]), counters)
        try:
            evaluator_result = _object(
                json.loads(raw_payloads["evaluator_result"]),
                "evaluator_result",
            )
        except (UnicodeError, json.JSONDecodeError) as error:
            refuse(f"{raw}_evaluator_result", f"evaluator result is not JSON: {error}")
        if evaluator_result.get("job_id") not in {
            result.get("job_id"),
            "gpuq-000000000000" if str(result.get("job_id", "")).startswith("gpuq-") else result.get("job_id"),
        }:
            refuse(f"{raw}_evaluator_result.job_id", "worker and broker job identities differ",
                   observed=evaluator_result.get("job_id"), expected=result.get("job_id"))
        normalized_evaluator_result = dict(evaluator_result)
        normalized_evaluator_result["job_id"] = result["job_id"]
        if normalized_evaluator_result != broker_result:
            refuse(f"{raw}_evaluator_result", "broker and evaluator raw results differ")

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
        refuse(f"{location}.objects", "broker attempt archived object coverage differs",
               observed=set(by_role), expected=expected_reference_roles)
    if len(job_ids) != len(attempts):
        refuse(f"{ledger}.attempts", "broker attempt job identity is duplicated", observed=job_ids)
    if len(attempts) == 2:
        first = _object(attempts[0], "broker_attempts[0]")
        if (
            first.get("mode") != "exclusive"
            or first.get("admitted") is not False
            or first.get("error") != "gpu_admission_differs"
            or first.get("receipt_sha256") is not None
            or any(first.get(field) != 0 for field in counter_fields)
            or authorities[0] != authorities[1]
        ):
            refuse(f"{ledger}.attempts[0]", "broker admission recovery differs",
                   observed={field: first.get(field)
                             for field in ("mode", "admitted", "error", "receipt_sha256", *counter_fields)})
    expected_receipt_attempts = [len(attempts)] if final_receipt is not None else []
    if receipt_attempts != expected_receipt_attempts:
        refuse(f"{ledger}.attempts", "broker attempt receipt placement differs",
               observed=receipt_attempts, expected=expected_receipt_attempts)


def _replay_evaluation_attempt_event(
    evidence: EvidenceStore,
    payload: Mapping[str, object],
    *,
    candidate: LaunchableCandidate,
    protocol_sha256: str,
    compiler_reference: Mapping[str, object],
    final_receipt: EvaluationReceipt | None,
    case_id: str,
    used_job_ids: set,
    location: str = "evaluation_attempt_completed",
) -> None:
    """Resolve one attempt event to its sole ledger and retained raw artifacts."""

    if payload.get("candidate_sha256") != candidate.candidate_sha256:
        refuse(f"{location}.payload.candidate_sha256", "evaluation attempt candidate differs",
               observed=payload.get("candidate_sha256"), expected=candidate.candidate_sha256)
    references = payload.get("objects")
    if not isinstance(references, list):
        refuse(f"{location}.payload.objects", "evaluation attempt objects differ",
               observed=type(references).__name__)
    ledger_references = [
        value
        for value in references
        if isinstance(value, Mapping) and value.get("role") == "broker_attempt_ledger"
    ]
    if len(ledger_references) != 1:
        refuse(f"{location}.payload.objects", "evaluation attempt ledger coverage differs",
               observed=len(ledger_references), expected=1)
    document = _object(
        json.loads(
            evidence.read_object(cast(Mapping[str, object], ledger_references[0]))
        ),
        "broker_attempt_ledger",
    )
    jobs = [item.get('job_id') for item in document.get('attempts', []) if isinstance(item, Mapping)]
    if any(not isinstance(job,str) for job in jobs) or used_job_ids.intersection(jobs):
        refuse(location, 'broker job was reused across Evaluation invocations')
    _replay_broker_attempt_ledger(
        evidence,
        cast(list[object], references),
        document,
        candidate=candidate,
        protocol_sha256=protocol_sha256,
        compiler_reference=compiler_reference,
        final_receipt=final_receipt,
        purpose=payload['purpose'], case_id=case_id,
        location=location,
    )
    used_job_ids.update(jobs)
