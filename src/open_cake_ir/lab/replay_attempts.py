"""Independently validate retained broker attempts and recovery accounting."""

from __future__ import annotations

import json, re
from hashlib import sha256
from typing import Mapping, cast

from open_cake_ir.evaluation import EvaluationReceipt, LaunchableCandidate
from open_cake_ir.evaluation.paired import validate_paired_broker
from open_cake_ir.evidence import EvidenceStore

from ._documents import _DIGEST, _object


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
