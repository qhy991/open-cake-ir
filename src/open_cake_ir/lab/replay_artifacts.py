"""Validate retained launchable artifacts and evaluation receipts without writing evidence."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Callable, Mapping, cast

from open_cake_ir.evaluation import EvaluationReceipt, LaunchableCandidate
from open_cake_ir.evaluation.paired import validate_receipt_policy
from open_cake_ir.evidence import EvidenceStore

from ._documents import _canonical_json_bytes, _object
from .archive import _arm_artifact_roles


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
    if "launch_manifest" not in artifact_payloads:
        raise ValueError("launchable candidate lacks its launch manifest")
    manifest = manifest_parser(json.loads(artifact_payloads["launch_manifest"]))
    if (not _arm_artifact_roles(arm, manifest.target) <= set(artifact_roles)
            or set(artifact_payloads) != set(artifact_roles)):
        raise ValueError("launchable candidate arm artifact roles differ")
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
