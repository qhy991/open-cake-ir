"""Reconstruct candidate assessment, build filtering and artifact outcomes."""

from __future__ import annotations

import math, re
from typing import Callable, Mapping, Sequence, cast

from open_cake_ir.evaluation import EvaluationReceipt, LaunchableCandidate
from open_cake_ir.evidence import EvidenceStore

from .._documents import _DIGEST, _object
from .attempts import _replay_evaluation_attempt_event
from .artifacts import _replay_evaluation_receipt, _replay_launchable_candidate
from .refusals import event_location, refuse
from ..contracts import CampaignLock
from ..pairing import comparison_arm, native_backend
from open_cake_ir.evaluation.paired import paired_protocol
from ..routing import route_rejection
from ..evaluation_lifecycle import evaluation_origin


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
    compiler_factory=None,
    provider_candidate_bytes=None,
) -> tuple[
    dict[tuple[int, str], LaunchableCandidate],
    dict[tuple[int, str, str], EvaluationReceipt],
    list[tuple[int, str, str]],
    dict[tuple[int, str], Mapping[str, object]],
]:
    attempt_events = [
        event for event in events if event.get("kind") == "evaluation_attempt_completed"
    ]
    attempt_payloads: dict[tuple[int, str, str], Mapping[str, object]] = {}
    for ordinal, event in enumerate(attempt_events):
        location = event_location("evaluation_attempt_completed", ordinal=ordinal)
        payload = _object(
            event.get("payload"), "evaluation_attempt_completed.payload"
        )
        turn = evaluation_origin(payload)
        purpose = payload.get("purpose")
        candidate_sha256 = payload.get("candidate_sha256")
        expected_fields = {"source_turn" if "source_turn" in payload else "turn", "purpose", "candidate_sha256", "objects"}
        if set(payload) != expected_fields:
            refuse(f"{location}.payload", "fields differ", observed=set(payload),
                   expected=expected_fields)
        if not isinstance(turn, int) or isinstance(turn, bool) or turn <= 0:
            refuse(f"{location}.payload.turn", "not a positive integer", observed=turn)
        if purpose not in {"search", "confirmatory", "attribution"}:
            refuse(f"{location}.payload.purpose", "not an evaluation purpose", observed=purpose,
                   expected={"search", "confirmatory", "attribution"})
        if not isinstance(candidate_sha256, str) or _DIGEST.fullmatch(candidate_sha256) is None:
            refuse(f"{location}.payload.candidate_sha256", "not a SHA256 digest",
                   observed=candidate_sha256)
        if not isinstance(payload.get("objects"), list):
            refuse(f"{location}.payload.objects", "object references are not a list",
                   observed=type(payload.get("objects")).__name__)
        key = (turn, cast(str, purpose), candidate_sha256)
        if key in attempt_payloads:
            refuse(location, "a second attempt event for one Turn, purpose and candidate",
                   observed={"turn": turn, "purpose": purpose, "candidate_sha256": candidate_sha256})
        attempt_payloads[key] = payload
    replayed_attempts: set[tuple[int, str, str]] = set()
    launchable_events = [
        event for event in events if event.get("kind") == "launchable_candidate_sealed"
    ]
    launchables: dict[tuple[int, str], LaunchableCandidate] = {}
    for ordinal, event in enumerate(launchable_events):
        location = event_location("launchable_candidate_sealed", ordinal=ordinal)
        payload = _object(event.get("payload"), "launchable.payload")
        turn = payload.get("turn")
        candidate_sha256 = payload.get("candidate_sha256")
        if not isinstance(turn, int) or isinstance(turn, bool) or turn <= 0:
            refuse(f"{location}.payload.turn", "not a positive integer", observed=turn)
        if not isinstance(candidate_sha256, str) or _DIGEST.fullmatch(candidate_sha256) is None:
            refuse(f"{location}.payload.candidate_sha256", "not a SHA256 digest",
                   observed=candidate_sha256)
        key = (turn, candidate_sha256)
        if key in launchables:
            refuse(location, "a second sealed launchable for one Turn and candidate",
                   observed={"turn": turn, "candidate_sha256": candidate_sha256})
        if candidate_sha256 not in provider_candidates_by_turn.get(turn, ()):
            refuse(location, 'launchable candidate was not submitted in this Turn')
        if provider_candidate_bytes is not None and key not in provider_candidate_bytes:
            refuse(location, 'archived author candidate bytes are missing')
        launchables[key] = _replay_launchable_candidate(
            evidence,
            launchable_events,
            turn=turn,
            candidate_sha256=candidate_sha256,
            arm=arm,
            manifest_parser=manifest_parser, compiler_factory=compiler_factory,
            authored_bytes=provider_candidate_bytes[(turn, candidate_sha256)] if provider_candidate_bytes is not None else None,
        )

    receipts: dict[tuple[int, str, str], EvaluationReceipt] = {}
    receipt_order: list[tuple[int, str, str]] = []
    rejected: dict[tuple[int, str], Mapping[str, object]] = {}
    ordinals = {"candidate_rejected": 0, "candidate_evaluated": 0}
    for event in events:
        kind = event.get("kind")
        payload = _object(event.get("payload"), f"event.{kind}.payload")
        if kind not in ordinals:
            continue
        location = event_location(cast(str, kind), ordinal=ordinals[kind])
        ordinals[kind] += 1
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
            ):
                refuse(f"{location}.payload", "fields differ", observed=set(payload),
                       expected=required_rejection_fields | {"objects", "artifact_rejections"})
            if not isinstance(feedback, Mapping):
                refuse(f"{location}.payload.feedback", "not an object",
                       observed=type(feedback).__name__)
            if not _artifact_outcomes_are_closed(payload):
                refuse(f"{location}.payload", "retained/rejected artifact roles are not a closed partition",
                       observed={"objects": payload.get("objects"),
                                 "artifact_rejections": payload.get("artifact_rejections")})
            decision = route_rejection(feedback, arm=arm)
            if (
                payload.get("routed_to") != decision.destination
                or payload.get("routing_reason") != decision.reason
            ):
                refuse(f"{location}.payload", "routing differs from the decision rederived from its feedback",
                       observed={"routed_to": payload.get("routed_to"),
                                 "routing_reason": payload.get("routing_reason")},
                       expected={"routed_to": decision.destination,
                                 "routing_reason": decision.reason})
            if not isinstance(turn, int) or isinstance(turn, bool):
                refuse(f"{location}.payload.turn", "not an integer", observed=turn)
            if not isinstance(candidate_sha256, str) or _DIGEST.fullmatch(candidate_sha256) is None:
                refuse(f"{location}.payload.candidate_sha256", "not a SHA256 digest",
                       observed=candidate_sha256)
            if (turn, candidate_sha256) in rejected:
                refuse(location, "a second rejection for one Turn and candidate",
                       observed={"turn": turn, "candidate_sha256": candidate_sha256})
            rejected[(turn, candidate_sha256)] = payload
        elif kind == "candidate_evaluated":
            turn = evaluation_origin(payload)
            purpose = payload.get("purpose")
            candidate_sha256 = payload.get("candidate_sha256")
            expected_fields = {"source_turn" if "source_turn" in payload else "turn", "purpose", "candidate_sha256", "objects"}
            if purpose == "confirmatory":
                expected_fields.add("elapsed_wall_seconds")
            if set(payload) != expected_fields:
                refuse(f"{location}.payload", "fields differ", observed=set(payload),
                       expected=expected_fields)
            if not isinstance(turn, int) or isinstance(turn, bool):
                refuse(f"{location}.payload.turn", "not an integer", observed=turn)
            if purpose not in {"search", "confirmatory", "attribution"}:
                refuse(f"{location}.payload.purpose", "not an evaluation purpose", observed=purpose,
                       expected={"search", "confirmatory", "attribution"})
            if not isinstance(candidate_sha256, str) or _DIGEST.fullmatch(candidate_sha256) is None:
                refuse(f"{location}.payload.candidate_sha256", "not a SHA256 digest",
                       observed=candidate_sha256)
            if "elapsed_wall_seconds" in payload and (
                type(payload["elapsed_wall_seconds"]) not in {int, float}
                or not math.isfinite(payload["elapsed_wall_seconds"])
                or payload["elapsed_wall_seconds"] < 0
            ):
                refuse(f"{location}.payload.elapsed_wall_seconds", "not a finite non-negative number",
                       observed=payload["elapsed_wall_seconds"])
            location = event_location("candidate_evaluated", turn=turn, purpose=purpose,
                                      candidate=candidate_sha256)
            launchable = launchables.get((turn, candidate_sha256))
            validated_receipt = _replay_evaluation_receipt(
                evidence, payload, launchable=launchable, candidate_sha256=candidate_sha256,
                workload_sha256=workload_sha256, protocol_sha256=protocol_sha256,
                case_id=case_id, purpose=purpose,
                evaluation_protocol=lock.document['evaluation_protocol'],
                fixed_baseline=lock.document['execution'].get('fixed_baseline', {}).get('candidate'),
                location=location,
            )
            receipt_key = (turn, cast(str, purpose), candidate_sha256)
            if receipt_key in receipts:
                refuse(location, "a second evaluation for one Turn, purpose and candidate")
            receipts[receipt_key] = validated_receipt
            receipt_order.append(receipt_key)
            attempt_payload = attempt_payloads.get(receipt_key)
            if attempt_payload is None:
                refuse(location, "no evaluation_attempt_completed event for this evaluation")
            replayed_attempts.add(receipt_key)
            _replay_evaluation_attempt_event(
                evidence,
                attempt_payload,
                candidate=launchable,
                protocol_sha256=protocol_sha256,
                compiler_reference=lock.document["compiler_revision"],
                final_receipt=validated_receipt,
                location=event_location("evaluation_attempt_completed", turn=turn, purpose=purpose,
                                        candidate=candidate_sha256),
            )
    unreplayed_attempts = set(attempt_payloads) - replayed_attempts
    if unreplayed_attempts:
        unevaluated = [
            {"turn": turn, "purpose": purpose, "candidate_sha256": candidate_sha256}
            for turn, purpose, candidate_sha256 in sorted(unreplayed_attempts)
        ]
        if len(unreplayed_attempts) != 1 or len(faults) != 1:
            refuse("evaluation_attempt_completed",
                   "attempts without a candidate_evaluated event: only one, and only under one fault",
                   observed={"unevaluated": unevaluated, "faults": len(faults)})
        turn, purpose, candidate_sha256 = next(iter(unreplayed_attempts))
        location = event_location("evaluation_attempt_completed", turn=turn, purpose=purpose,
                                  candidate=candidate_sha256)
        if turn != fault_turn:
            refuse(f"{location}.payload.turn", "an unevaluated attempt lies outside the fault Turn",
                   observed=turn, expected=fault_turn)
        attempt_payload = attempt_payloads[(turn, purpose, candidate_sha256)]
        launchable = launchables.get((turn, candidate_sha256))
        if launchable is None:
            refuse(location, "no launchable_candidate_sealed event for this candidate and Turn")
        _replay_evaluation_attempt_event(
            evidence,
            attempt_payload,
            candidate=launchable,
            protocol_sha256=protocol_sha256,
            compiler_reference=lock.document["compiler_revision"],
            final_receipt=None,
            location=location,
        )

    for kind, members in (
        ("launchable_candidate_sealed", launchables),
        ("candidate_rejected", rejected),
        ("candidate_evaluated", [(key[0], key[2]) for key in receipts]),
        ("evaluation_attempt_completed", [(key[0], key[2]) for key in attempt_payloads]),
    ):
        for turn, candidate_sha256 in members:
            if candidate_sha256 not in provider_candidates_by_turn.get(turn, ()):
                refuse(event_location(kind, turn=turn, candidate=candidate_sha256),
                       "the candidate is not one the provider submitted in that Turn",
                       expected=provider_candidates_by_turn.get(turn, ()))
    return launchables, receipts, receipt_order, rejected
