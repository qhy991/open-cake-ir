"""Replay provider turns against retained raw bytes and fixed admission rules."""

from __future__ import annotations

import json, re
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Sequence, cast

from open_cake_ir.evidence import EvidenceStore, RunAudit

from ._documents import _object
from .providers import (
    CANDIDATE_SET_ENVELOPE_V1,
    _project_candidate_submission,
    parse_codex_turn_events,
)
from .task_package import TASK_AGENTS_RALPH_V1, TaskPackage

from .claude import CLAUDE_EVENT_CONTRACT, parse_claude_turn_events


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
] | None:
    threads: set[str] = set()
    cumulative_by_turn: dict[int, int] = {}
    provider_candidates_by_turn: dict[int, tuple[str, ...]] = {}
    provider_candidate_bytes: dict[tuple[int, str], bytes] = {}
    candidate_set_turns: set[int] = set()
    prior_cumulative = 0
    for expected_turn, event in enumerate(provider_events, start=1):
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
            return None
        if payload.get("turn") != expected_turn:
            return None
        thread_id = payload.get("thread_id")
        cumulative = payload.get("cumulative_provider_tokens")
        if (
            not isinstance(thread_id, str)
            or not isinstance(cumulative, int)
            or isinstance(cumulative, bool)
            or cumulative <= 0
        ):
            return None
        objects = payload.get("objects")
        if not isinstance(objects, list):
            return None
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
                    return None
                indexed_references[index] = item
        candidate_references = [
            indexed_references[index]
            for index in range(len(indexed_references))
        ]
        if (
            len(event_references) != 1
            or len(submission_references) != 1
            or len(reference_bundle_references) > 1
            or (
                len(reference_bundle_references) != 1
            )
            or not isinstance(candidate_count, int)
            or isinstance(candidate_count, bool)
            or candidate_count <= 0
            or candidate_count > maximum_candidates_per_turn
            or len(candidate_references) != candidate_count
            or len(objects)
            != candidate_count + 2 + len(reference_bundle_references)
        ):
            return None
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
        except (UnicodeError, ValueError):
            return None
        if projected_candidates != candidates:
            return None
        candidate_digests = tuple(sha256(candidate).hexdigest() for candidate in candidates)
        if (
            sha256(raw_events).hexdigest() != event_references[0].get("sha256")
            or (
                reference_bundle is not None
                and (
                    not reference_bundle
                    or sha256(reference_bundle).hexdigest()
                    != reference_bundle_references[0].get("sha256")
                )
            )
            or any(
                digest != reference.get("sha256")
                for digest, reference in zip(candidate_digests, candidate_references)
            )
            or len(set(candidate_digests)) != len(candidate_digests)
        ):
            return None
        if reference_bundle is not None:
            try:
                decoded_reference = reference_bundle.decode("utf-8")
                bundle = _object(
                    json.loads(decoded_reference),
                    "provider Ralph task package",
                )
                state = _object(
                    bundle.get("state_card"),
                    "provider Ralph StateCard",
                )
                if (
                    set(bundle)
                    != {
                        "schema_version",
                        "kind",
                        "run_id",
                        "arm",
                        "task_markdown",
                        "agents_markdown",
                        "state_card",
                        "rubric",
                    }
                    or bundle.get("schema_version") != 1
                    or bundle.get("kind") != TASK_AGENTS_RALPH_V1
                    or bundle.get("run_id") != audit.run_id
                    or bundle.get("arm") != arm
                    or bundle.get("task_markdown")
                    != expected_task_package.task_markdown
                    or bundle.get("agents_markdown")
                    != expected_task_package.agents_markdown
                    or state.get("kind") != "ralph_state_v1"
                    or state.get("iteration") != expected_turn
                    or state.get("cumulative_provider_tokens")
                    != prior_cumulative
                    or state.get("terminal_reason") is not None
                    or reference_bundle != expected_task_package.evidence_bundle(state)
                ):
                    return None
            except (UnicodeError, json.JSONDecodeError, ValueError):
                return None
        terminal_document: dict[str, object] = {
            "arm": arm,
            "candidate_written": True,
            "kind": "open_cake_ir_turn",
            "turn": expected_turn,
        }
        if event_contract == "closed_file_change_v1":
            terminal_document["tool_calls"] = 1
        expected_terminal = json.dumps(
            terminal_document,
            sort_keys=True,
            separators=(",", ":"),
        )
        expected_change = "add" if expected_turn == 1 else "update"
        expected_name = (
            "candidate-set.json"
        )
        if event_contract == CLAUDE_EVENT_CONTRACT:
            parsed = parse_claude_turn_events(raw_events, expected_terminal_message=expected_terminal)
            if (parsed.reported_models != (provider_authority["model"],)
                    or expected_change == "add" and parsed.write_tools[0] != "Write"):
                return False
        else:
            parsed = parse_codex_turn_events(raw_events, expected_terminal_message=expected_terminal,
                                            event_contract=event_contract)
        turn_tokens = payload.get("turn_provider_tokens")
        if (
            parsed.thread_id != thread_id
            or turn_tokens != parsed.provider_tokens
            or cumulative != prior_cumulative + turn_tokens
        ):
            return None
        if parsed.candidate_path is not None and (
            (event_contract != CLAUDE_EVENT_CONTRACT and parsed.change_kind != expected_change)
            or Path(parsed.candidate_path).name != expected_name
        ):
            return None
        if parsed.candidate_path is None and event_contract != "tool_rich_candidate_v1":
            return None
        if payload.get("normalization") != parsed.normalization:
            return None
        if "event_contract" in provider_authority and payload.get(
            "auxiliary_activity"
        ) != [dict(activity.document) for activity in parsed.tool_activity]:
            return None
        threads.add(thread_id)
        cumulative_by_turn[expected_turn] = cumulative
        provider_candidates_by_turn[expected_turn] = candidate_digests
        provider_candidate_bytes.update(
            ((expected_turn, digest), payload) for digest, payload in zip(candidate_digests, candidates)
        )
        prior_cumulative = cumulative
    if len(threads) != 1 or list(cumulative_by_turn.values()) != sorted(
        cumulative_by_turn.values()
    ):
        return None
    return cumulative_by_turn, provider_candidates_by_turn, provider_candidate_bytes, candidate_set_turns, prior_cumulative
