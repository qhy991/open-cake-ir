"""Validate retained launchable artifacts and evaluation receipts without writing evidence."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Callable, Mapping, cast

from open_cake_ir.evaluation import EvaluationReceipt, LaunchableCandidate
from open_cake_ir.evaluation.paired import validate_receipt_policy
from open_cake_ir.evidence import EvidenceStore

from .._documents import _canonical_json_bytes, _object
from ..archive import _arm_artifact_roles
from .refusals import event_location, refuse


def _replay_launchable_candidate(
    evidence: EvidenceStore,
    launchable_events: list[Mapping[str, object]],
    *,
    turn: int,
    candidate_sha256: str,
    arm: str,
    manifest_parser: Callable,
    compiler_factory=None,
    authored_bytes=None,
) -> LaunchableCandidate:
    """Rebuild one sealed launchable and enforce its arm-owned artifact contract."""

    location = event_location("launchable_candidate_sealed", turn=turn, candidate=candidate_sha256)
    matching = []
    for event in launchable_events:
        payload = _object(event.get("payload"), "launchable.payload")
        if (
            payload.get("turn") == turn
            and payload.get("candidate_sha256") == candidate_sha256
        ):
            matching.append(payload)
    if len(matching) != 1:
        refuse(location, "launchable candidate event coverage differs", observed=len(matching), expected=1)
    payload = matching[0]
    expected_fields = {"turn", "candidate_sha256", "candidate_record_sha256", "objects"}
    if set(payload) != expected_fields:
        refuse(f"{location}.payload", "launchable candidate event authority differs",
               observed=set(payload), expected=expected_fields)
    references = payload.get("objects")
    if not isinstance(references, list):
        refuse(f"{location}.payload.objects", "launchable candidate object references differ",
               observed=type(references).__name__)
    artifact_payloads: dict[str, bytes] = {}
    artifact_roles: dict[str, str] = {}
    for index, reference in enumerate(references):
        if not isinstance(reference, Mapping):
            refuse(f"{location}.payload.objects[{index}]", "launchable candidate object reference differs",
                   observed=type(reference).__name__)
        role = reference.get("role")
        digest = reference.get("sha256")
        if (
            not isinstance(role, str)
            or role in artifact_roles
            or not isinstance(digest, str)
        ):
            refuse(f"{location}.payload.objects[{index}]", "launchable candidate object roles differ",
                   observed={"role": role, "sha256": digest})
        artifact_payloads[role] = evidence.read_object(reference)
        artifact_roles[role] = digest
        if sha256(artifact_payloads[role]).hexdigest() != digest:
            refuse(f"{location}.{role}", "launchable candidate artifact bytes differ")
    if "launch_manifest" not in artifact_payloads:
        refuse(f"{location}.payload.objects", "launchable candidate lacks its launch manifest",
               observed=set(artifact_roles))
    manifest = manifest_parser(json.loads(artifact_payloads["launch_manifest"]))
    if hasattr(manifest, 'check_complete_domain'):
        manifest.check_complete_domain()
    if not _arm_artifact_roles(arm, manifest.target, program="program_bundle" in artifact_roles) <= set(artifact_roles):
        refuse(f"{location}.payload.objects", "launchable candidate arm artifact roles differ",
               observed=set(artifact_roles), expected=_arm_artifact_roles(arm, manifest.target, program="program_bundle" in artifact_roles))
    if artifact_roles["launch_manifest"] != manifest.canonical_sha256:
        refuse(f"{location}.launch_manifest", "launchable candidate launch manifest seal differs")
    candidate = LaunchableCandidate(
        candidate_sha256=candidate_sha256,
        target=manifest.target,
        entry_point=manifest.kernel_name,
        artifact_roles=artifact_roles,
        launch_spec_sha256=manifest.canonical_sha256,
        artifact_payloads=artifact_payloads,
    )
    if candidate.is_program:
        if compiler_factory is None:
            raise ValueError('Program replay requires its exact Compiler')
        from open_cake_ir.compiler import Program
        if authored_bytes is None or Program.from_dict(json.loads(authored_bytes)).document != manifest.program.document:
            raise ValueError('Program manifest differs from the archived author candidate')
        lowered = compiler_factory().lower_program(manifest.program)
        if manifest.lowered_sources != {stage.name: lowering.source_sha256
                for stage, lowering in zip(lowered.program.stages, lowered.lowerings, strict=True)}:
            raise ValueError('Program stage source differs from its pinned Compiler lowering')
        from open_cake_ir.evaluation.program import program_components
        _, children, manifests = program_components(candidate)
        for stage, lowering in zip(lowered.program.stages, lowered.lowerings, strict=True):
            requirements = lowering.toolchain_requirements
            if (children[stage.name].entry_point != requirements['kernel_entry_point']
                or list(manifests[stage.name].grid) != list(requirements['grid'])):
                raise ValueError('Program stage launch differs from its pinned Compiler lowering')
    if payload.get("candidate_record_sha256") != candidate.canonical_sha256:
        refuse(f"{location}.payload.candidate_record_sha256", "launchable candidate record seal differs")
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
    location: str,
) -> EvaluationReceipt:
    """Reconstruct a receipt from independent archived raw-role objects.

    `location` names the `candidate_evaluated` event whose payload this is; every
    refusal is placed under it.
    """
    objects = payload.get("objects")
    if not isinstance(objects, list):
        refuse(f"{location}.objects", "object references are not a list",
               observed=type(objects).__name__)
    receipt_refs = [
        item
        for item in objects
        if isinstance(item, Mapping) and item.get("role") == "evaluation_receipt"
    ]
    if len(receipt_refs) != 1:
        refuse(f"{location}.objects", "evaluation_receipt reference count differs",
               observed=len(receipt_refs), expected=1)
    receipt = _object(
        json.loads(evidence.read_object(cast(Mapping[str, object], receipt_refs[0]))),
        "evaluation_receipt",
    )
    for field, expected in (
        ("candidate_sha256", candidate_sha256),
        ("workload_sha256", workload_sha256),
        ("evaluation_protocol_sha256", protocol_sha256),
        ("case_id", case_id),
        ("purpose", payload.get("purpose")),
    ):
        if receipt.get(field) != expected:
            refuse(f"{location}.evaluation_receipt.{field}", "receipt authority differs",
                   observed=receipt.get(field), expected=expected)
    if launchable is None:
        refuse(location, "no launchable_candidate_sealed event for this candidate and Turn")
    expected_raw = receipt.get("artifact_payload_sha256")
    expected_raw_roles = (
        {"correctness_output", "launch_receipt", "profile"}
        if purpose == "attribution"
        else {"correctness_output", "launch_receipt", "timing_samples"}
    )
    if not isinstance(expected_raw, Mapping) or set(expected_raw) != expected_raw_roles:
        refuse(f"{location}.evaluation_receipt.artifact_payload_sha256",
               "raw artifact roles differ",
               observed=set(expected_raw) if isinstance(expected_raw, Mapping) else expected_raw,
               expected=expected_raw_roles)
    raw_payloads: dict[str, bytes] = {}
    for role in sorted(expected_raw):
        matching = [
            item
            for item in objects
            if isinstance(item, Mapping) and item.get("role") == role
        ]
        if len(matching) != 1:
            refuse(f"{location}.objects", f"{role} reference count differs",
                   observed=len(matching), expected=1)
        raw_payloads[role] = evidence.read_object(
            cast(Mapping[str, object], matching[0])
        )
    for role, raw in sorted(raw_payloads.items()):
        if expected_raw[role] != sha256(raw).hexdigest():
            refuse(f"{location}.evaluation_receipt.artifact_payload_sha256.{role}",
                   "retained raw bytes differ from the receipt's digest")
    archived_roles = {
        item.get("role")
        for item in objects
        if isinstance(item, Mapping)
    }
    if archived_roles != {"evaluation_receipt", *expected_raw}:
        refuse(f"{location}.objects", "archived object roles differ",
               observed=archived_roles, expected={"evaluation_receipt", *expected_raw})
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
        refuse(f"{location}.evaluation_receipt",
               "retained receipt document differs from the receipt rebuilt from its raw objects")
    return validated_receipt
