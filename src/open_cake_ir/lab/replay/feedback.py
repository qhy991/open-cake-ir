"""Reconstruct author-visible feedback from retained candidate and receipt facts.

Task-package replay authenticates a bundle's rendering. This module independently
binds its feedback to completed earlier Turns; the bundle cannot authenticate its
own selected candidate, measurement, diagnostics or profile.
"""

from __future__ import annotations

import json

from .._documents import _canonical_json_bytes, _object
from .._policies import _ATTRIBUTION_EVALUATION
from ..evaluation_lifecycle import evaluation_origin
from ..feedback import derive_turn_feedback
from .refusals import event_location, refuse


def _check_state_feedback(state, expected, location):
    state = _object(state, location)
    observed = state.get("previous_feedback")
    try:
        matches = _canonical_json_bytes(observed) == _canonical_json_bytes(expected)
    except (TypeError, ValueError):
        matches = False
    if not matches:
        refuse(location + ".previous_feedback",
               "differs from the feedback reconstructed from completed earlier candidate evidence")


def _check_bundle_feedback(raw, expected, location):
    try:
        bundle = json.loads(raw)
    except (UnicodeError, ValueError, TypeError):
        refuse(location, "retained provider reference bundle is not JSON")
    bundle = _object(bundle, location)
    _check_state_feedback(bundle.get("state_card"), expected, location + ".state_card")


def replay_feedback(*, events, evidence, specification,
                    provider_candidates_by_turn=None, receipts=None,
                    rejected=None, fault_turn=None, launchables=None, authored=None):
    """Verify every delivered and terminal projection without running an author.

    Candidate selection closes a completed Turn. An interrupted search Turn leaves
    the preceding feedback intact; confirmation never changes search feedback.
    The empty maps also handle zero-Turn budget stops and first-provider faults.
    Other replay owners validate the rows, receipts, action resolutions and terminal
    order before this consumer reconstructs their author-visible projection.
    """
    provider_candidates_by_turn = provider_candidates_by_turn or {}
    receipts = receipts or {}
    rejected = rejected or {}
    feedback = {"kind": "initial"}
    filters = {}
    actions = {}
    seen_searches = {}
    seen_attributions = {}
    seen_launchables = {}
    launchables = launchables or {}
    authored = authored or {}
    provider = specification.document["authoring"].get("provider", {})

    for ordinal, event in enumerate(events):
        kind = event.get("kind")
        payload = _object(event.get("payload"), f"events[{ordinal}].payload")
        turn = payload.get("turn")
        if kind == "provider_turn_completed":
            location = event_location(kind, turn=turn)
            references = [row for row in payload["objects"]
                          if row.get("role") == "provider_reference_bundle"]
            if len(references) != 1:
                refuse(location + ".provider_reference_bundle",
                       "exactly one provider reference bundle is required")
            _check_bundle_feedback(evidence.read_object(references[0]), feedback,
                                   location + ".provider_reference_bundle")
        elif kind == "author_actions_resolved":
            actions[turn] = [{key: value for key, value in row.items() if key != "objects"}
                             for row in payload["actions"]]
        elif kind == "candidate_set_filtered":
            filters[turn] = payload
        elif kind == 'launchable_candidate_sealed':
            key = (turn, payload['candidate_sha256'])
            if key in launchables:
                seen_launchables[key] = launchables[key]
        elif kind == "candidate_evaluated":
            origin = evaluation_origin(payload)
            purpose = payload["purpose"]
            identity = payload["candidate_sha256"]
            key = (origin, purpose, identity)
            if purpose == "search":
                seen_searches.setdefault(origin, {})[identity] = receipts[key]
            # A first-provider fault has no Evaluation dependency to resolve.
            elif (purpose == "attribution"
                  and specification.document["evaluation_protocol"].get("attribution_evaluation") == _ATTRIBUTION_EVALUATION
                  and "source_turn" not in payload):
                seen_attributions.setdefault(origin, {})[identity] = receipts[key]
        elif kind == "candidate_selected" and turn != fault_turn:
            location = event_location(kind, turn=turn)
            if turn not in filters or turn not in actions or turn not in provider_candidates_by_turn:
                refuse(location, "feedback selection lacks preceding candidate or action evidence")
            filtered = filters[turn]
            feedback = derive_turn_feedback(
                turn=turn,
                candidates=provider_candidates_by_turn[turn],
                filter_rows=filtered["order"],
                selected=payload["candidate_sha256"],
                receipts=seen_searches.get(turn, {}),
                attributions=seen_attributions.get(turn, {}),
                rejected_feedback={identity: row["feedback"]
                    for (origin, identity), row in rejected.items() if origin == turn},
                actions=actions[turn],
                arm=specification.environment_kind,
                specification=specification,
                selection_summary=filtered.get("candidate_selection"),
                launchables={identity: candidate for (origin, identity), candidate
                             in seen_launchables.items() if origin == turn
                             and identity in seen_searches.get(turn, {})},
                authored={identity: value for (origin, identity), value in authored.items()
                          if origin == turn},
            )
        elif kind == "search_completed":
            _check_state_feedback(payload.get("state"), feedback, "search_completed.state")
        elif kind == "checkpoints_projected":
            _check_state_feedback(payload.get("ralph"), feedback, "checkpoints_projected.ralph")
        elif (kind == "run_fault" and payload.get("stage") == "provider"
              and provider.get("event_contract") == "responses_messages_v1"):
            # The message provider retains its exact failed request in stdout.
            # Its native request validation already authenticates the rendering;
            # still bind that request's feedback to the same earlier evidence.
            references = [row for row in payload.get("objects", [])
                          if row.get("role") == "provider_stdout"]
            if references:
                from ..message_provider import _json
                location = "run_fault.payload.objects.provider_stdout.request"
                try:
                    record = _json(evidence.read_object(references[0]))
                    bundle = record["request"]["input"][-1]["content"]
                except (UnicodeError, ValueError, KeyError, IndexError, TypeError):
                    refuse(location, "failed message request lacks its task bundle")
                _check_bundle_feedback(bundle, feedback, location + ".reference_bundle")
