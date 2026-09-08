"""Interpret retained Codex event streams without launching a provider."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Mapping, cast

from .faults import ReportedProviderUsage
from .provider_documents import (
    CANDIDATE_SET_ENVELOPE_V1,
    ParsedCodexTurnEvents,
    ProviderAuxiliaryActivity,
    ProviderTurn,
    _THREAD_ID,
    _canonical_json_bytes,
    _project_candidate_submission,
    _read_candidate_nofollow,
    _unique_json_object,
    _reject_json_constant,
)


def parse_codex_turn_events(
    raw_events: bytes,
    *,
    expected_terminal_message: str,
    event_contract: str = "closed_file_change_v1",
) -> ParsedCodexTurnEvents:
    """Parse the complete closed Codex JSONL Turn without reading its candidate."""

    if not expected_terminal_message or event_contract not in {
        "closed_file_change_v1",
        "tool_rich_candidate_v1",
    }:
        raise ValueError("provider terminal expectation differs")
    try:
        events = [json.loads(line) for line in raw_events.splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("provider events are not JSONL") from error
    if any(not isinstance(event, Mapping) for event in events):
        raise ValueError("provider event sequence is incomplete")
    typed_events = cast(list[Mapping[str, object]], events)
    types = [event.get("type") for event in typed_events]
    admitted_types = {
        "thread.started",
        "turn.started",
        "item.started",
        "item.updated",
        "item.completed",
        "turn.completed",
    }
    if (
        (
            event_contract == "closed_file_change_v1"
            and len(typed_events) not in {6, 7, 8, 9}
        )
        or (event_contract == "closed_file_change_v1" and "item.updated" in types)
        or len(typed_events) < 6
        or any(event_type not in admitted_types for event_type in types)
        or types[0] != "thread.started"
        or types[1] != "turn.started"
        or types[-1] != "turn.completed"
        or types.count("thread.started") != 1
        or types.count("turn.started") != 1
        or types.count("turn.completed") != 1
    ):
        if any(
            isinstance(event_type, str) and event_type.startswith("item.")
            for event_type in types
            if event_type not in {"item.started", "item.completed"}
        ):
            raise ValueError("provider item lifecycle differs")
        raise ValueError("provider Turn boundary differs")
    thread_id = typed_events[0].get("thread_id")
    if not isinstance(thread_id, str) or _THREAD_ID.fullmatch(thread_id) is None:
        raise ValueError("provider thread identity differs")

    file_events: list[tuple[int, Mapping[str, object], Mapping[str, object]]] = []
    messages: list[tuple[int, str]] = []
    auxiliary_events: dict[
        str, list[tuple[str, Mapping[str, object]]]
    ] = {}
    activity_indices: list[int] = []
    auxiliary_types = {
        "reasoning",
        "command_execution",
        "file_change",
        "mcp_tool_call",
        "collab_tool_call",
        "web_search",
        "todo_list",
        "error",
    }
    for index, event in enumerate(typed_events[2:-1], start=2):
        event_type = event.get("type")
        if event_type not in {"item.started", "item.updated", "item.completed"}:
            raise ValueError("provider item lifecycle differs")
        item = event.get("item")
        if not isinstance(item, Mapping):
            raise ValueError("provider item payload differs")
        item_type = item.get("type")
        changes = item.get("changes")
        candidate_file_change = (
            item_type == "file_change" and event_contract == "closed_file_change_v1"
        )
        if candidate_file_change:
            file_events.append((index, event, cast(Mapping[str, object], item)))
            activity_indices.append(index)
        elif item_type == "agent_message" and event_type == "item.completed":
            if set(item) != {"id", "type", "text"}:
                raise ValueError("provider terminal message fields differ")
            item_id = item.get("id")
            text = item.get("text")
            if not isinstance(item_id, str) or not item_id or not isinstance(text, str):
                raise ValueError("provider terminal message differs")
            messages.append((index, text))
        elif event_contract == "tool_rich_candidate_v1" and item_type in auxiliary_types:
            item_id = item.get("id")
            if not isinstance(item_id, str) or not item_id:
                raise ValueError("provider auxiliary item identity differs")
            auxiliary_events.setdefault(item_id, []).append(
                (cast(str, event_type), cast(Mapping[str, object], item))
            )
            # Passive CLI notices remain retained auxiliary evidence, but do not
            # move the terminal brackets around actual tool/file activity.
            if item_type not in {"error", "reasoning"}:
                activity_indices.append(index)
        else:
            raise ValueError("provider emitted an unadmitted item type")

    tool_activity = _auxiliary_activity(auxiliary_events)
    if event_contract == "tool_rich_candidate_v1" and not activity_indices:
        raise ValueError("provider Turn lacks functional tool or file activity")

    candidate_path: str | None = None
    change_kind: str | None = None
    start_index = min(activity_indices, default=2)
    stop_index = max(activity_indices, default=1)
    if file_events:

        if len(file_events) == 2:
            candidate_path, change_kind, _, _ = _complete_file_change(
                file_events[0], file_events[1]
            )
            if change_kind not in {"add", "update"}:
                raise ValueError("provider candidate path or change kind differs")
        elif len(file_events) == 4:
            removed_path, removed_kind, _, removed_stop = _complete_file_change(
                file_events[0], file_events[1]
            )
            added_path, added_kind, added_start, _ = _complete_file_change(
                file_events[2], file_events[3]
            )
            if (
                removed_path != added_path
                or removed_kind != "delete"
                or added_kind != "add"
                or removed_stop >= added_start
            ):
                raise ValueError("provider replacement lifecycle differs")
            candidate_path = added_path
            change_kind = "update"
        else:
            raise ValueError("provider must emit one candidate update")
    elif event_contract == "closed_file_change_v1":
        raise ValueError("provider must emit one complete file-change lifecycle")

    normalization = _normalize_terminal_messages(
        messages, expected_terminal_message, start_index, stop_index,
    )

    tokens = _codex_usage_tokens(typed_events[-1].get("usage"))
    return ParsedCodexTurnEvents(
        thread_id=thread_id,
        provider_tokens=tokens,
        candidate_path=candidate_path,
        change_kind=cast(str, change_kind),
        terminal_message_count=len(messages),
        normalization=normalization,
        tool_activity=tuple(tool_activity),
    )


def _codex_usage_tokens(usage, *, allow_zero=False) -> int:
    if not isinstance(usage, Mapping):
        raise ValueError("provider usage is missing")
    required_usage = {"input_tokens", "output_tokens"}
    optional_usage = {
        "cached_input_tokens",
        "cache_write_input_tokens",
        "reasoning_output_tokens",
    }
    if not required_usage <= set(usage) or not set(usage) <= required_usage | optional_usage:
        raise ValueError("provider usage fields differ")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in usage.values()
    ):
        raise ValueError("provider usage differs")
    input_tokens = cast(int, usage["input_tokens"])
    output_tokens = cast(int, usage["output_tokens"])
    cached_input_tokens = cast(int, usage.get("cached_input_tokens", 0))
    cache_write_input_tokens = cast(
        int, usage.get("cache_write_input_tokens", 0)
    )
    if (
        (not allow_zero and input_tokens + output_tokens <= 0)
        or cached_input_tokens > input_tokens
        or cache_write_input_tokens > input_tokens
        or cast(int, usage.get("reasoning_output_tokens", 0)) > output_tokens
    ):
        raise ValueError("provider usage differs")
    return input_tokens + output_tokens


def reported_codex_usage(raw_events: bytes, *, event_contract: str,
                         expected_thread_id: str | None = None) -> ReportedProviderUsage | None:
    """Read a complete native usage report without admitting a candidate or Turn."""
    if (not isinstance(raw_events, bytes) or not isinstance(event_contract, str)
            or event_contract not in {"closed_file_change_v1", "tool_rich_candidate_v1"}):
        return None
    try:
        events = [json.loads(line, object_pairs_hook=_unique_json_object,
                             parse_constant=_reject_json_constant) for line in raw_events.splitlines()]
        if len(events) < 3 or any(not isinstance(event, Mapping) for event in events):
            return None
        types = [event.get("type") for event in events]
        if (types[:2] != ["thread.started", "turn.started"] or types[-1] != "turn.completed"
                or any(types.count(kind) != 1 for kind in ("thread.started", "turn.started", "turn.completed"))):
            return None
        thread_id = events[0].get("thread_id")
        if expected_thread_id is not None and thread_id != expected_thread_id:
            return None
        if any("thread_id" in event and event["thread_id"] != thread_id for event in events):
            return None
        return ReportedProviderUsage(event_contract, thread_id,
            _codex_usage_tokens(events[-1].get("usage"), allow_zero=True))
    except (UnicodeError, ValueError, TypeError, OverflowError, RecursionError):
        return None


def reported_provider_usage(raw_events: bytes, *, provider: Mapping[str, object],
                            expected_thread_id: str | None = None) -> ReportedProviderUsage | None:
    """Dispatch reported invocation usage using its frozen native provider contract."""
    if not isinstance(raw_events, bytes) or not raw_events:
        return None
    contract = provider.get("event_contract", "closed_file_change_v1")
    if contract in {"closed_file_change_v1", "tool_rich_candidate_v1"}:
        return reported_codex_usage(raw_events, event_contract=contract,
                                    expected_thread_id=expected_thread_id)
    from .claude import CLAUDE_EVENT_CONTRACT, reported_claude_usage
    if contract == CLAUDE_EVENT_CONTRACT:
        return reported_claude_usage(raw_events, expected_model=provider.get("model"),
                                     expected_thread_id=expected_thread_id)
    return None


def normalize_codex_turn(
    raw_events: bytes,
    *,
    candidate_path: Path,
    expected_change: str,
    expected_terminal_message: str,
    event_contract: str = "closed_file_change_v1",
    submission_contract: str = CANDIDATE_SET_ENVELOPE_V1,
    arm: str | None = None,
    maximum_candidates_per_turn: int = 1,
) -> ProviderTurn:
    """Accept only the two terminal forms observed by the frozen r42 boundary."""

    if expected_change not in {"add", "update"} or not expected_terminal_message:
        raise ValueError("provider Turn expectation differs")
    parsed = parse_codex_turn_events(
        raw_events,
        expected_terminal_message=expected_terminal_message,
        event_contract=event_contract,
    )
    observed_change = parsed.change_kind
    if expected_change == "update" and observed_change == "add":
        # The Lab checked that the candidate existed before this resumed Turn.
        # Codex may nevertheless label its in-place replacement as an add.
        observed_change = "update"
    if (
        parsed.candidate_path is not None
        and (
            parsed.candidate_path != str(candidate_path.absolute())
            or observed_change != expected_change
        )
    ):
        raise ValueError("provider candidate path or change kind differs")
    submission = _read_candidate_nofollow(candidate_path)
    candidates = _project_candidate_submission(
        submission,
        submission_contract=submission_contract,
        arm=arm,
        maximum_candidates_per_turn=maximum_candidates_per_turn,
    )
    return ProviderTurn(
        thread_id=parsed.thread_id,
        provider_tokens=parsed.provider_tokens,
        candidates=candidates,
        candidate_sha256s=tuple(sha256(candidate).hexdigest() for candidate in candidates),
        raw_submission=submission,
        raw_events=raw_events,
        raw_events_sha256=sha256(raw_events).hexdigest(),
        terminal_message=expected_terminal_message,
        terminal_message_count=parsed.terminal_message_count,
        normalization=parsed.normalization,
        tool_activity=parsed.tool_activity,
    )


def _auxiliary_activity(auxiliary_events):
    """Validate each auxiliary item lifecycle and preserve first-seen order."""
    tool_activity: list[ProviderAuxiliaryActivity] = []
    for item_id, lifecycle in auxiliary_events.items():
        event_types = [event_type for event_type, _ in lifecycle]
        item_types = {item.get("type") for _, item in lifecycle}
        if (
            len(item_types) != 1
            or not isinstance(next(iter(item_types)), str)
            or event_types
            not in (
                ["item.completed"],
                ["item.started", "item.completed"],
            )
            and not (
                len(event_types) >= 3
                and event_types[0] == "item.started"
                and event_types[-1] == "item.completed"
                and set(event_types[1:-1]) == {"item.updated"}
            )
        ):
            raise ValueError("provider auxiliary item lifecycle differs")
        first = lifecycle[0][1]
        final = lifecycle[-1][1]
        if next(iter(item_types)) == "file_change":
            if (
                event_types != ["item.started", "item.completed"]
                or set(first) != {"id", "type", "changes", "status"}
                or set(final) != {"id", "type", "changes", "status"}
                or first.get("status") != "in_progress"
                or final.get("status") != "completed"
                or first.get("changes") != final.get("changes")
            ):
                raise ValueError("provider auxiliary file-change lifecycle differs")
            changes = first.get("changes")
            if (
                not isinstance(changes, list)
                or not changes
                or any(
                    not isinstance(change, Mapping)
                    or set(change) != {"path", "kind"}
                    or not isinstance(change.get("path"), str)
                    or not Path(cast(str, change["path"])).is_absolute()
                    or change.get("kind") not in {"add", "update", "delete"}
                    for change in changes
                )
            ):
                raise ValueError("provider auxiliary file-change payload differs")
        if any(
            item.get("server") != first.get("server")
            or item.get("tool") != first.get("tool")
            for _, item in lifecycle
        ):
            raise ValueError("provider auxiliary item authority changed")
        status = final.get("status", "completed")
        if not isinstance(status, str) or not status:
            raise ValueError("provider auxiliary item status differs")
        server = final.get("server")
        tool = final.get("tool")
        if server is not None and not isinstance(server, str):
            raise ValueError("provider auxiliary item server differs")
        if tool is not None and not isinstance(tool, str):
            raise ValueError("provider auxiliary item tool differs")
        tool_activity.append(
            ProviderAuxiliaryActivity(
                item_id=item_id,
                item_type=cast(str, next(iter(item_types))),
                status=status,
                server=cast(str | None, server),
                tool=cast(str | None, tool),
            )
        )
    return tool_activity


def _complete_file_change(
    start: tuple[int, Mapping[str, object], Mapping[str, object]],
    stop: tuple[int, Mapping[str, object], Mapping[str, object]],
) -> tuple[str, str, int, int]:
    start_event_index, start_event, start_item = start
    stop_event_index, stop_event, stop_item = stop
    if (
        set(start_item) != {"id", "type", "changes", "status"}
        or set(stop_item) != {"id", "type", "changes", "status"}
        or start_event.get("type") != "item.started"
        or stop_event.get("type") != "item.completed"
        or not isinstance(start_item.get("id"), str)
        or not start_item.get("id")
        or start_item.get("id") != stop_item.get("id")
        or start_item.get("status") != "in_progress"
        or stop_item.get("status") != "completed"
        or start_item.get("changes") != stop_item.get("changes")
    ):
        raise ValueError("provider file-change lifecycle differs")
    changes = start_item.get("changes")
    if not isinstance(changes, list) or len(changes) != 1:
        raise ValueError("provider candidate file-change differs")
    change = changes[0]
    if not isinstance(change, Mapping) or set(change) != {"path", "kind"}:
        raise ValueError("provider candidate file-change differs")
    path = change.get("path")
    kind = change.get("kind")
    if (
        not isinstance(path, str)
        or not path
        or not Path(path).is_absolute()
        or kind not in {"add", "update", "delete"}
    ):
        raise ValueError("provider candidate path or change kind differs")
    return path, cast(str, kind), start_event_index, stop_event_index


def _normalize_terminal_messages(messages, expected_terminal_message, start_index, stop_index):
    """Admit only the documented exact or bracketed semantic normalization."""
    message_texts = [text for _, text in messages]
    if message_texts == [expected_terminal_message] and messages[0][0] > stop_index:
        normalization = "single_exact"
    elif (
        len(message_texts) >= 2
        and all(text == expected_terminal_message for text in message_texts)
        and messages[0][0] < start_index
        and messages[-1][0] > stop_index
    ):
        normalization = "duplicate_exact_bracketed"
    elif (
        len(message_texts) >= 2
        and messages[0][0] < start_index
        and messages[-1][0] > stop_index
    ):
        try:
            semantic_messages = [
                _canonical_json_bytes(json.loads(text)).decode("utf-8")
                for text in message_texts
            ]
        except (UnicodeError, ValueError, json.JSONDecodeError):
            semantic_messages = []
        if semantic_messages != [expected_terminal_message] * len(message_texts):
            raise ValueError("provider terminal-event normalization differs")
        normalization = "duplicate_semantic_bracketed"
    else:
        raise ValueError("provider terminal-event normalization differs")
    return normalization
