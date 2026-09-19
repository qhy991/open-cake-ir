"""Replay provider turns against retained raw bytes and fixed admission rules."""

from __future__ import annotations

import json, re
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Sequence, cast

from open_cake_ir.evidence import EvidenceStore, RunAudit

from ._documents import _object
from .faults import ReportedProviderUsage
from .provider_events import reported_provider_usage
from .providers import (
    CANDIDATE_SET_ENVELOPE_V1,
    _project_candidate_submission,
    parse_codex_turn_events,
)
from .replay_refusals import event_location, refuse
from .task_package import TASK_AGENTS_RALPH_V1, TaskPackage

from .claude import CLAUDE_EVENT_CONTRACTS, observed_claude_quota, parse_claude_turn_events


def _expected_terminal_message(arm: str, turn: int, event_contract: str) -> str:
    """The exact structured terminal each completed or faulted Turn must have declared."""
    terminal_document: dict[str, object] = {
        "arm": arm,
        "candidate_written": True,
        "kind": "open_cake_ir_turn",
        "turn": turn,
    }
    if event_contract == "closed_file_change_v1":
        terminal_document["tool_calls"] = 1
    return json.dumps(
        terminal_document,
        sort_keys=True,
        separators=(",", ":"),
    )


def _replay_provider_turns(
    *,
    arm: str,
    audit: RunAudit,
    event_contract: str,
    evidence: EvidenceStore,
    expected_task_package: TaskPackage,
    maximum_candidates_per_turn: int,
    provider_authority: Mapping[str, object],
    provider_events: Sequence[Mapping[str, object]],
) -> tuple[
    dict[int, int],
    dict[int, tuple[str, ...]],
    dict[tuple[int, str], bytes],
    set[int],
    int,
]:
    threads: set[str] = set()
    cumulative_by_turn: dict[int, int] = {}
    provider_candidates_by_turn: dict[int, tuple[str, ...]] = {}
    provider_candidate_bytes: dict[tuple[int, str], bytes] = {}
    candidate_set_turns: set[int] = set()
    prior_cumulative = 0
    for expected_turn, event in enumerate(provider_events, start=1):
        location = event_location("provider_turn_completed", turn=expected_turn)
        payload = _object(event.get("payload"), "provider_turn.payload")
        expected_provider_fields = {
            "turn",
            "thread_id",
            "turn_provider_tokens",
            "cumulative_provider_tokens",
            "normalization",
            "candidate_count",
            "objects",
        }
        if "event_contract" in provider_authority:
            expected_provider_fields.add("auxiliary_activity")
        if set(payload) != expected_provider_fields:
            refuse(f"{location}.payload", "fields differ", observed=set(payload),
                   expected=expected_provider_fields)
        if payload.get("turn") != expected_turn:
            refuse(f"{location}.payload.turn", "Turn differs from the event's position",
                   observed=payload.get("turn"), expected=expected_turn)
        thread_id = payload.get("thread_id")
        cumulative = payload.get("cumulative_provider_tokens")
        if not isinstance(thread_id, str):
            refuse(f"{location}.payload.thread_id", "not a string", observed=thread_id)
        if not isinstance(cumulative, int) or isinstance(cumulative, bool) or cumulative <= 0:
            refuse(f"{location}.payload.cumulative_provider_tokens", "not a positive integer",
                   observed=cumulative)
        objects = payload.get("objects")
        if not isinstance(objects, list):
            refuse(f"{location}.payload.objects", "object references are not a list",
                   observed=type(objects).__name__)
        event_references = [
            cast(Mapping[str, object], item)
            for item in objects
            if isinstance(item, Mapping) and item.get("role") == "provider_events"
        ]
        reference_bundle_references = [
            cast(Mapping[str, object], item)
            for item in objects
            if isinstance(item, Mapping)
            and item.get("role") == "provider_reference_bundle"
        ]
        submission_references = [
            cast(Mapping[str, object], item)
            for item in objects
            if isinstance(item, Mapping)
            and item.get("role") == "provider_submission_envelope"
        ]
        candidate_count = payload.get("candidate_count")
        candidate_set_turns.add(expected_turn)
        indexed_references: dict[int, Mapping[str, object]] = {}
        for item in objects:
            if not isinstance(item, Mapping):
                continue
            role = item.get("role")
            match = (
                re.fullmatch(r"candidate_submission_(\d{4})", role)
                if isinstance(role, str)
                else None
            )
            if match is not None:
                index = int(match.group(1))
                if index in indexed_references:
                    refuse(f"{location}.payload.objects", "a candidate submission index is duplicated",
                           observed=role)
                indexed_references[index] = item
        candidate_references = [
            indexed_references[index]
            for index in range(len(indexed_references))
        ]
        for role, count in (
            ("provider_events", len(event_references)),
            ("provider_submission_envelope", len(submission_references)),
            ("provider_reference_bundle", len(reference_bundle_references)),
        ):
            if count != 1:
                refuse(f"{location}.payload.objects", f"{role} reference count differs",
                       observed=count, expected=1)
        if (
            not isinstance(candidate_count, int)
            or isinstance(candidate_count, bool)
            or candidate_count <= 0
            or candidate_count > maximum_candidates_per_turn
        ):
            refuse(f"{location}.payload.candidate_count",
                   "not an integer within the Turn's candidate budget",
                   observed=candidate_count, expected=f"1..{maximum_candidates_per_turn}")
        if len(candidate_references) != candidate_count:
            refuse(f"{location}.payload.objects", "candidate submission references differ from candidate_count",
                   observed=len(candidate_references), expected=candidate_count)
        if len(objects) != candidate_count + 3:
            refuse(f"{location}.payload.objects", "object reference count differs",
                   observed=len(objects), expected=candidate_count + 3)
        raw_events = evidence.read_object(event_references[0])
        reference_bundle = (
            evidence.read_object(reference_bundle_references[0])
            if reference_bundle_references
            else None
        )
        candidates = tuple(
            evidence.read_object(reference) for reference in candidate_references
        )
        try:
            projected_candidates = _project_candidate_submission(
                evidence.read_object(submission_references[0]),
                submission_contract=CANDIDATE_SET_ENVELOPE_V1,
                arm=arm,
                maximum_candidates_per_turn=maximum_candidates_per_turn,
            )
        except (UnicodeError, ValueError) as error:
            refuse(f"{location}.provider_submission_envelope",
                   f"the retained submission envelope does not project: {error}")
        if projected_candidates != candidates:
            refuse(f"{location}.provider_submission_envelope",
                   "candidates projected from the envelope differ from the retained candidate submissions")
        candidate_digests = tuple(sha256(candidate).hexdigest() for candidate in candidates)
        if sha256(raw_events).hexdigest() != event_references[0].get("sha256"):
            refuse(f"{location}.provider_events", "retained raw bytes differ from the reference digest")
        if reference_bundle is not None and (
            not reference_bundle
            or sha256(reference_bundle).hexdigest() != reference_bundle_references[0].get("sha256")
        ):
            refuse(f"{location}.provider_reference_bundle",
                   "retained raw bytes are empty or differ from the reference digest")
        for index, (digest, reference) in enumerate(zip(candidate_digests, candidate_references)):
            if digest != reference.get("sha256"):
                refuse(f"{location}.candidate_submission_{index:04d}",
                       "retained raw bytes differ from the reference digest")
        if len(set(candidate_digests)) != len(candidate_digests):
            refuse(f"{location}.payload.objects", "two candidate submissions carry the same bytes",
                   observed=candidate_digests)
        if reference_bundle is not None:
            bundle_location = f"{location}.provider_reference_bundle"
            try:
                bundle = _object(
                    json.loads(reference_bundle.decode("utf-8")),
                    "provider Ralph task package",
                )
                state = _object(
                    bundle.get("state_card"),
                    "provider Ralph StateCard",
                )
            except (UnicodeError, json.JSONDecodeError, ValueError) as error:
                refuse(bundle_location, f"not a Ralph task package: {error}")
            expected_bundle_fields = {
                "schema_version",
                "kind",
                "run_id",
                "arm",
                "task_markdown",
                "agents_markdown",
                "state_card",
                "rubric",
            }
            if set(bundle) != expected_bundle_fields:
                refuse(bundle_location, "fields differ", observed=set(bundle),
                       expected=expected_bundle_fields)
            for field, expected in (
                ("schema_version", 1),
                ("kind", TASK_AGENTS_RALPH_V1),
                ("run_id", audit.run_id),
                ("arm", arm),
            ):
                if bundle.get(field) != expected:
                    refuse(f"{bundle_location}.{field}", "differs from the Run's authority",
                           observed=bundle.get(field), expected=expected)
            for field in ("task_markdown", "agents_markdown"):
                if bundle.get(field) != getattr(expected_task_package, field):
                    refuse(f"{bundle_location}.{field}",
                           "differs from the task package rederived from the Campaign Lock")
            for field, expected in (
                ("kind", "ralph_state_v1"),
                ("iteration", expected_turn),
                ("cumulative_provider_tokens", prior_cumulative),
                ("terminal_reason", None),
            ):
                if state.get(field) != expected:
                    refuse(f"{bundle_location}.state_card.{field}", "differs from the replayed state",
                           observed=state.get(field), expected=expected)
            if reference_bundle != expected_task_package.evidence_bundle(state):
                refuse(bundle_location,
                       "retained bytes differ from the bundle rederived from the task package and state")
        expected_terminal = _expected_terminal_message(arm, expected_turn, event_contract)
        expected_change = "add" if expected_turn == 1 else "update"
        expected_name = (
            "candidate-set.json"
        )
        if event_contract in CLAUDE_EVENT_CONTRACTS:
            parsed = parse_claude_turn_events(raw_events, expected_terminal_message=expected_terminal, event_contract=event_contract)
            if parsed.reported_models != (provider_authority["model"],):
                refuse(f"{location}.provider_events", "reported models differ from the arm's provider authority",
                       observed=parsed.reported_models, expected=(provider_authority["model"],))
            if expected_change == "add" and parsed.write_tools[0] != "Write":
                refuse(f"{location}.provider_events", "the first Turn's candidate write is not a Write",
                       observed=parsed.write_tools[0], expected="Write")
        else:
            parsed = parse_codex_turn_events(raw_events, expected_terminal_message=expected_terminal,
                                            event_contract=event_contract)
        turn_tokens = payload.get("turn_provider_tokens")
        if parsed.thread_id != thread_id:
            refuse(f"{location}.payload.thread_id", "differs from the retained provider events",
                   observed=thread_id, expected=parsed.thread_id)
        if turn_tokens != parsed.provider_tokens:
            refuse(f"{location}.payload.turn_provider_tokens", "differs from the retained provider events",
                   observed=turn_tokens, expected=parsed.provider_tokens)
        if cumulative != prior_cumulative + turn_tokens:
            refuse(f"{location}.payload.cumulative_provider_tokens", "differs from the replayed running total",
                   observed=cumulative, expected=prior_cumulative + turn_tokens)
        if parsed.candidate_path is not None:
            if event_contract not in CLAUDE_EVENT_CONTRACTS and parsed.change_kind != expected_change:
                refuse(f"{location}.provider_events", "candidate change kind differs",
                       observed=parsed.change_kind, expected=expected_change)
            if Path(parsed.candidate_path).name != expected_name:
                refuse(f"{location}.provider_events", "candidate file name differs",
                       observed=Path(parsed.candidate_path).name, expected=expected_name)
        elif event_contract != "tool_rich_candidate_v1":
            refuse(f"{location}.provider_events", "no candidate write under a contract that requires one",
                   observed=event_contract)
        if payload.get("normalization") != parsed.normalization:
            refuse(f"{location}.payload.normalization", "differs from the retained provider events",
                   observed=payload.get("normalization"), expected=parsed.normalization)
        if "event_contract" in provider_authority:
            expected_activity = [dict(activity.document) for activity in parsed.tool_activity]
            if payload.get("auxiliary_activity") != expected_activity:
                refuse(f"{location}.payload.auxiliary_activity", "differs from the retained provider events",
                       observed=payload.get("auxiliary_activity"), expected=expected_activity)
        threads.add(thread_id)
        cumulative_by_turn[expected_turn] = cumulative
        provider_candidates_by_turn[expected_turn] = candidate_digests
        provider_candidate_bytes.update(
            ((expected_turn, digest), payload) for digest, payload in zip(candidate_digests, candidates)
        )
        prior_cumulative = cumulative
    if len(threads) != 1:
        refuse("provider_turn_completed", "the Run's Turns do not share one thread", observed=threads)
    if list(cumulative_by_turn.values()) != sorted(cumulative_by_turn.values()):
        refuse("provider_turn_completed", "cumulative provider tokens do not increase by Turn",
               observed=cumulative_by_turn)
    return cumulative_by_turn, provider_candidates_by_turn, provider_candidate_bytes, candidate_set_turns, prior_cumulative


def replay_fault_usage(*, payload, evidence, provider, expected_thread_id=None) -> int:
    """Rederive failed-invocation usage from retained stdout, never its declaration.

    Every refusal is located under the `run_fault` event's payload.
    """
    location = "run_fault.payload"
    usage_fields = {"provider_usage", "terminal_provider_tokens_scope", "provider_usage_witness_mismatch"}
    if payload.get("stage") != "provider":
        if usage_fields & set(payload):
            refuse(location, "provider usage fields on a fault outside the provider stage",
                   observed=usage_fields & set(payload), expected=set())
        return 0
    references = payload.get("objects", [])
    stdout = [reference for reference in references if reference.get("role") == "provider_stdout"]
    if len(stdout) > 1:
        refuse(f"{location}.objects", "provider_stdout reference count differs",
               observed=len(stdout), expected="0 or 1")
    try:
        observed = reported_provider_usage(evidence.read_object(stdout[0]), provider=provider,
            expected_thread_id=expected_thread_id) if stdout else None
    except (OSError, ValueError, KeyError) as error:
        refuse(f"{location}.objects.provider_stdout",
               f"retained stdout does not yield provider usage: {error}")
    retained_usage = payload.get("provider_usage")
    if observed is not None:
        expected_usage_fields = {"status", "event_contract", "thread_id", "provider_tokens"}
        if (not isinstance(retained_usage, Mapping)
                or set(retained_usage) != expected_usage_fields
                or retained_usage.get("status") != "observed"):
            refuse(f"{location}.provider_usage", "not an observed-usage record",
                   observed=retained_usage, expected=expected_usage_fields)
        try:
            retained = ReportedProviderUsage(retained_usage["event_contract"],
                retained_usage["thread_id"], retained_usage["provider_tokens"])
        except (TypeError, ValueError) as error:
            refuse(f"{location}.provider_usage", f"not a provider usage record: {error}",
                   observed=retained_usage)
        if retained != observed:
            refuse(f"{location}.provider_usage", "differs from the usage rederived from retained stdout",
                   observed=retained_usage, expected=observed)
    elif retained_usage != {"status": "unavailable", "provider_tokens": None}:
        refuse(f"{location}.provider_usage", "retained stdout yields no usage",
               observed=retained_usage, expected={"status": "unavailable", "provider_tokens": None})
    if observed is None:
        if payload.get("terminal_provider_tokens_scope") != "known_subtotal":
            refuse(f"{location}.terminal_provider_tokens_scope", "an unobserved usage is a known subtotal",
                   observed=payload.get("terminal_provider_tokens_scope"), expected="known_subtotal")
    elif "terminal_provider_tokens_scope" in payload:
        refuse(f"{location}.terminal_provider_tokens_scope", "scoped although usage was observed",
               observed=payload.get("terminal_provider_tokens_scope"), expected=None)
    if "provider_usage_witness_mismatch" in payload:
        declared = payload["provider_usage_witness_mismatch"]
        if declared is not None:
            expected_witness_fields = {"event_contract", "thread_id", "provider_tokens"}
            if not isinstance(declared, Mapping) or set(declared) != expected_witness_fields:
                refuse(f"{location}.provider_usage_witness_mismatch", "not a provider usage record",
                       observed=declared, expected=expected_witness_fields)
            try:
                declared = ReportedProviderUsage(**declared)
            except (TypeError, ValueError) as error:
                refuse(f"{location}.provider_usage_witness_mismatch",
                       f"not a provider usage record: {error}", observed=declared)
        if declared == observed:
            refuse(f"{location}.provider_usage_witness_mismatch",
                   "the declared witness equals the usage rederived from retained stdout",
                   observed=declared)
    if stdout and provider.get("event_contract") in CLAUDE_EVENT_CONTRACTS:
        # Fault attribution is rederived from the retained stdout, never trusted
        # from its declaration; evidence sealed before the field replays unchanged.
        try:
            quota = observed_claude_quota(evidence.read_object(stdout[0]))
        except (OSError, ValueError, KeyError) as error:
            refuse(f"{location}.objects.provider_stdout",
                   f"retained stdout does not yield a quota observation: {error}")
        if quota is None:
            # F-2026-09-16-001: at a fault seam an absent notice is itself the
            # recorded observation; evidence sealed while the field stayed absent
            # (retained None) still replays unchanged below.
            quota = {"observed": "no_notice"}
        retained_quota = payload.get("observed_quota")
        if retained_quota is not None and retained_quota != quota:
            refuse(f"{location}.observed_quota", "differs from the quota rederived from retained stdout",
                   observed=retained_quota, expected=quota)
    return observed.provider_tokens if observed is not None else 0
