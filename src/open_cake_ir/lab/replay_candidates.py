"""Reconstruct candidate assessment, build filtering and artifact outcomes."""

from __future__ import annotations

import math, re
from typing import Callable, Mapping, Sequence, cast

from open_cake_ir.evaluation import EvaluationReceipt, LaunchableCandidate
from open_cake_ir.evidence import EvidenceStore

from ._documents import _DIGEST, _object
from .replay_attempts import _replay_evaluation_attempt_event
from .replay_artifacts import _replay_evaluation_receipt, _replay_launchable_candidate
from .contracts import CampaignLock
from .pairing import comparison_arm, native_backend
from open_cake_ir.evaluation.paired import paired_protocol
from .routing import route_rejection


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


def _replay_candidates(
    *,
    arm: str,
    case_id: str,
    events: Sequence[Mapping[str, object]],
    evidence: EvidenceStore,
    fault_turn: int | None,
    faults: Sequence[Mapping[str, object]],
    lock: CampaignLock,
    manifest_parser: Callable,
    protocol_sha256: str,
    provider_candidates_by_turn: Mapping[int, tuple[str, ...]],
    workload_sha256: str,
) -> tuple[
    dict[tuple[int, str], LaunchableCandidate],
    dict[tuple[int, str, str], EvaluationReceipt],
    list[tuple[int, str, str]],
    dict[tuple[int, str], Mapping[str, object]],
] | None:
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
            return None
        key = (turn, cast(str, purpose), candidate_sha256)
        if key in attempt_payloads:
            return None
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
            return None
        key = (turn, candidate_sha256)
        if key in launchables:
            return None
        launchables[key] = _replay_launchable_candidate(
            evidence,
            launchable_events,
            turn=turn,
            candidate_sha256=candidate_sha256,
            arm=arm,
            manifest_parser=manifest_parser,
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
                return None
            decision = route_rejection(feedback)
            if (
                payload.get("routed_to") != decision.destination
                or payload.get("routing_reason") != decision.reason
            ):
                return None
            if (
                not isinstance(turn, int)
                or isinstance(turn, bool)
                or not isinstance(candidate_sha256, str)
                or _DIGEST.fullmatch(candidate_sha256) is None
                or (turn, candidate_sha256) in rejected
            ):
                return None
            rejected[(turn, candidate_sha256)] = payload
        elif kind == "candidate_evaluated":
            turn = payload.get("turn")
            purpose = payload.get("purpose")
            candidate_sha256 = payload.get("candidate_sha256")
            if (
                (set(payload) != ({
                    "turn", "purpose", "candidate_sha256", "objects",
                } | ({"elapsed_wall_seconds"} if purpose == "confirmatory" and
                     (native_backend(comparison_arm(lock.document["resolved_inputs"]["arm_environments"])) is not None or paired_protocol(lock.document["evaluation_protocol"]) is not None) else set())))
                or
                not isinstance(turn, int)
                or isinstance(turn, bool)
                or purpose not in {"search", "confirmatory", "attribution"}
                or not isinstance(candidate_sha256, str)
                or _DIGEST.fullmatch(candidate_sha256) is None
            ):
                return None
            if "elapsed_wall_seconds" in payload and (
                type(payload["elapsed_wall_seconds"]) not in {int, float}
                or not math.isfinite(payload["elapsed_wall_seconds"])
                or payload["elapsed_wall_seconds"] < 0
            ):
                return None
            launchable = launchables.get((turn, candidate_sha256))
            validated_receipt = _replay_evaluation_receipt(
                evidence, payload, launchable=launchable, candidate_sha256=candidate_sha256,
                workload_sha256=workload_sha256, protocol_sha256=protocol_sha256,
                case_id=case_id, purpose=purpose,
                evaluation_protocol=lock.document['evaluation_protocol'],
                fixed_baseline=lock.document['execution'].get('fixed_baseline', {}).get('candidate'),
            )
            if validated_receipt is None:
                return None
            receipt_key = (turn, cast(str, purpose), candidate_sha256)
            if receipt_key in receipts:
                return None
            receipts[receipt_key] = validated_receipt
            receipt_order.append(receipt_key)
            attempt_payload = attempt_payloads.get(receipt_key)
            if attempt_payload is None:
                return None
            replayed_attempts.add(receipt_key)
            _replay_evaluation_attempt_event(
                evidence,
                attempt_payload,
                candidate=launchable,
                protocol_sha256=protocol_sha256,
                compiler_reference=lock.document["compiler_revision"],
                final_receipt=validated_receipt,
            )
    unreplayed_attempts = set(attempt_payloads) - replayed_attempts
    if unreplayed_attempts:
        if len(unreplayed_attempts) != 1 or len(faults) != 1:
            return None
        turn, purpose, candidate_sha256 = next(iter(unreplayed_attempts))
        if turn != fault_turn:
            return None
        attempt_payload = attempt_payloads[(turn, purpose, candidate_sha256)]
        launchable = launchables.get((turn, candidate_sha256))
        if launchable is None:
            return None
        _replay_evaluation_attempt_event(
            evidence,
            attempt_payload,
            candidate=launchable,
            protocol_sha256=protocol_sha256,
            compiler_reference=lock.document["compiler_revision"],
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
        return None
    return launchables, receipts, receipt_order, rejected
