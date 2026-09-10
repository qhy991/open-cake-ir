"""Claude-native provider pieces for the existing Lab Turn boundary.

This module does not admit a Campaign or create a qualification. TaskLab's
provider configuration, live qualification and Replay dispatch still need an
explicit Claude contract. --model binds the main conversation; native CLI helper
model usage is retained and charged under provider-default artifact-only scope,
without claiming a scientific arm comparison or treating helpers as main fallback.
Native transport retries remain inside one CLI invocation; observing them does not
restart a Lab Turn or Run. Only final reported modelUsage contributes tokens.
Flags follow Claude Code 2.1.241 local help and
https://code.claude.com/docs/en/headless. No Codex events are manufactured.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Mapping

from .faults import RunProtocolFault, ReportedProviderUsage
from .task_package import TaskPackage
from .process import (
    SupervisedProcessOutputLimit, SupervisedProcessTimeout,
    run_supervised, sanitized_environment,
)
from .providers import (
    CANDIDATE_SET_ENVELOPE_V1, ProviderAuxiliaryActivity, ProviderInvocation,
    ProviderTurn, ProviderQualificationReceipt, QualifiedRunProvider, _project_candidate_submission,
)

from .provider_documents import _THREAD_ID, _read_candidate_nofollow, _reject_json_constant, _unique_json_object
from ._documents import _canonical_json_bytes

CLAUDE_EVENT_CONTRACT = "claude_stream_candidate_v3"
CLAUDE_TERMINAL_TOOL = "StructuredOutput"
CLAUDE_AUTHORING_TOOLS = ("Read", "Write", "Edit", "Glob", "Grep")
# The largest auto-compact window this CLI admits. It has no value that turns compaction
# off, so the boundary is pushed past any Turn context a campaign is expected to reach.
CLAUDE_AUTOCOMPACT_WINDOW = "1M"
CLAUDE_USAGE_FIELDS = (
    "input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
)


def _json(payload: str | bytes):
    return json.loads(payload, object_pairs_hook=_unique_json_object, parse_constant=_reject_json_constant)


def terminal_schema() -> dict:
    """Stable initial/resume schema; expected arm/turn are checked after reception."""
    return {"type": "object", "properties": {
        "kind": {"type": "string", "const": "open_cake_ir_turn"},
        "arm": {"type": "string"}, "turn": {"type": "integer"},
        "candidate_written": {"type": "boolean", "const": True}},
        "required": ["kind", "arm", "turn", "candidate_written"], "additionalProperties": False}


def _terminal(value: object) -> bool:
    return (isinstance(value, Mapping) and set(value) == {"kind", "arm", "turn", "candidate_written"}
        and value["kind"] == "open_cake_ir_turn" and value["candidate_written"] is True
        and isinstance(value["arm"], str) and bool(value["arm"])
        and type(value["turn"]) is int and value["turn"] > 0)


# Quota statuses under which the CLI still serves the request. A warning reports how much
# of the window is gone; it does not withhold the turn, so refusing it would strand a
# campaign on an account that is merely over halfway through its quota.
_QUOTA_SERVED = ("allowed", "allowed_warning")
# Subtypes that mean the CLI rewrote the conversation the author was working in.
_CONTEXT_MUTATIONS = ("compact_boundary", "compacting")


def _metadata(event: Mapping) -> bool:
    """Admit the explicit native metadata shapes, not arbitrary system events."""
    kind = event.get("type")
    if kind == "rate_limit_event":
        if set(event) != {"type", "rate_limit_info", "uuid", "session_id"}:
            raise ValueError("Claude quota metadata fields differ")
        info = event["rate_limit_info"]
        required = {"status", "resetsAt", "rateLimitType"}
        # `utilization` is the fraction of the window the account has consumed. The CLI
        # began reporting it alongside `allowed_warning`, which is a heads-up about that
        # fraction and not a refusal: the request it accompanies is served normally.
        # Both are admitted; `rejected` and any status this does not name still fail
        # closed, as does any field the CLI adds after these.
        optional = {"overageStatus", "overageDisabledReason", "isUsingOverage", "utilization"}
        if (not isinstance(info, Mapping) or not required <= set(info) <= required | optional
                or info.get("status") not in _QUOTA_SERVED
                or "utilization" in info and (isinstance(info["utilization"], bool)
                    or not isinstance(info["utilization"], (int, float))
                    or not 0.0 <= info["utilization"] <= 1.0)
                or type(info.get("resetsAt")) is not int or info["resetsAt"] < 0
                or not isinstance(info.get("rateLimitType"), str) or not info["rateLimitType"]
                or "isUsingOverage" in info and type(info["isUsingOverage"]) is not bool
                or "overageStatus" in info and info["overageStatus"] not in ("allowed", "rejected")
                or "overageDisabledReason" in info and info["overageDisabledReason"] is not None
                    and not isinstance(info["overageDisabledReason"], str)
                or info.get("isUsingOverage") is True and info.get("overageStatus") != "allowed"):
            raise ValueError("Claude quota is rejected or metadata differs")
    elif kind == "system" and event.get("subtype") in _CONTEXT_MUTATIONS:
        # Identified by subtype alone and refused. The declared window above should keep
        # this unreachable; if it fires anyway the Turn ran on a context the Lab cannot
        # reconstruct, so it is reported as exactly that instead of as an unclassified
        # event. No field schema is asserted here because none has been observed -- a
        # guessed one would be a new campaign-fatal path of the kind this finding is about.
        raise ValueError(
            "Claude compacted the Turn context mid-run, so what the author saw is not "
            "reconstructible and the Turn is not comparable")
    elif kind == "system" and event.get("subtype") == "api_retry":
        if (set(event) != {"type", "subtype", "attempt", "max_retries", "retry_delay_ms", "error_status", "error", "uuid", "session_id"}
                or any(type(event[key]) is not int for key in ("attempt", "max_retries", "retry_delay_ms"))
                or not 1 <= event["attempt"] <= event["max_retries"] or event["retry_delay_ms"] < 0
                or event["error_status"] is not None and (type(event["error_status"]) is not int or not 100 <= event["error_status"] <= 599)
                or event["error"] not in ("authentication_failed", "oauth_org_not_allowed", "billing_error", "rate_limit",
                    "overloaded", "invalid_request", "model_not_found", "server_error", "max_output_tokens", "unknown")):
            raise ValueError("Claude API-retry observation differs")
    elif kind == "system" and event.get("subtype") == "post_turn_summary":
        # The CLI's own account of a finished Turn. The result event remains the only
        # authority on whether the Turn completed; this is retained, not judged, so an
        # unfamiliar category is preserved in Evidence instead of read as an outcome.
        if (set(event) != {"type", "subtype", "summarizes_uuid", "status_category",
                           "status_detail", "needs_action", "uuid", "session_id"}
                or not isinstance(event["summarizes_uuid"], str)
                or _THREAD_ID.fullmatch(event["summarizes_uuid"]) is None
                or not isinstance(event["status_category"], str) or not event["status_category"]
                or not isinstance(event["status_detail"], str)
                or not isinstance(event["needs_action"], str)):
            raise ValueError("Claude post-turn summary metadata differs")
    elif kind == "system" and event.get("subtype") == "thinking_tokens":
        if (set(event) != {"type", "subtype", "estimated_tokens", "estimated_tokens_delta", "uuid", "session_id"}
                or type(event["estimated_tokens"]) is not int or type(event["estimated_tokens_delta"]) is not int
                or not 0 <= event["estimated_tokens_delta"] <= event["estimated_tokens"]):
            raise ValueError("Claude thinking-token metadata differs")
    else:
        return False
    if not isinstance(event.get("uuid"), str) or _THREAD_ID.fullmatch(event["uuid"]) is None:
        raise ValueError("Claude metadata identity differs")
    return True


def claude_model_usage(terminal: Mapping, main_model: str, *, allow_zero: bool = False) -> tuple[int, tuple[ProviderAuxiliaryActivity, ...]]:
    """Charge each native modelUsage row once; top-level usage is a lower-bound check.

    Estimated thinking, output-token details, cache partitions, iterations and costs
    are never additive counters. Claude may use the requested model for internal work
    such as session titles, so its aggregate modelUsage row may exceed the main Turn's
    top-level usage. It may never understate that Turn. This projection alone does not
    admit a Turn.
    """
    usage = terminal.get("usage")
    if not isinstance(usage, Mapping) or any(type(usage.get(key)) is not int or usage[key] < 0 for key in CLAUDE_USAGE_FIELDS):
        raise ValueError("Claude complete usage is missing or malformed")
    if not allow_zero and sum(usage[key] for key in CLAUDE_USAGE_FIELDS) <= 0:
        raise ValueError("Claude main usage must be positive")
    rows = terminal.get("modelUsage")
    mapping = dict(zip(CLAUDE_USAGE_FIELDS, ("inputTokens", "outputTokens", "cacheCreationInputTokens", "cacheReadInputTokens")))
    optional = {"webSearchRequests", "costUSD", "contextWindow", "maxOutputTokens", "canonicalModel", "provider"}
    if (not isinstance(rows, Mapping) or main_model not in rows
            or any(not isinstance(model, str) or not model for model in rows)):
        raise ValueError("Claude modelUsage lacks the exact main model")
    activities, total = [], 0
    for model, row in sorted(rows.items()):
        if (not isinstance(model, str) or not model or not isinstance(row, Mapping)
                or not set(mapping.values()) <= set(row) <= set(mapping.values()) | optional
                or any(type(row.get(key)) is not int or row[key] < 0 for key in mapping.values())):
            raise ValueError("Claude modelUsage counters differ")
        for key in ("webSearchRequests", "contextWindow", "maxOutputTokens"):
            if key in row and (type(row[key]) is not int or row[key] < 0):
                raise ValueError("Claude modelUsage metadata differs")
        if (row.get("webSearchRequests", 0) != 0
                or "costUSD" in row and (type(row["costUSD"]) not in (int, float) or not math.isfinite(row["costUSD"]) or row["costUSD"] < 0)
                or any(key in row and (not isinstance(row[key], str) or not row[key]) for key in ("canonicalModel", "provider"))):
            raise ValueError("Claude modelUsage reports unsupported activity or malformed metadata")
        if model == main_model and any(row[other] < usage[key] for key, other in mapping.items()):
            raise ValueError("Claude main modelUsage understates terminal usage")
        tokens = sum(row[key] for key in mapping.values())
        total += tokens
        activities.append(ProviderAuxiliaryActivity(f"modelUsage[{model}]", "model_usage", "reported", model=model, provider_tokens=tokens))
    if not allow_zero and total <= 0:
        raise ValueError("Claude provider usage must be positive")
    return total, tuple(activities)


def reported_claude_usage(raw_events: bytes, *, expected_model: str,
                          expected_thread_id: str | None = None) -> ReportedProviderUsage | None:
    """Observe a complete native usage statement without accepting its candidate.

    Invalid/partial or unbound reporting remains unavailable, not zero. A reported
    zero requires complete matching modelUsage rows. No retry metadata is charged.
    """
    if not isinstance(raw_events, bytes):
        return None
    try:
        events = [_json(line) for line in raw_events.splitlines()]
        if len(events) < 2 or any(not isinstance(event, Mapping) for event in events):
            return None
        initial, terminal = events[0], events[-1]
        thread_id = initial.get("session_id")
        if (initial.get("type") != "system" or initial.get("subtype") != "init"
                or initial.get("model") != expected_model or not isinstance(expected_model, str) or not expected_model
                or not isinstance(thread_id, str) or _THREAD_ID.fullmatch(thread_id) is None
                or expected_thread_id is not None and thread_id != expected_thread_id
                or any(event.get("session_id") != thread_id for event in events)
                or terminal.get("type") != "result" or not isinstance(terminal.get("subtype"), str)
                or not terminal["subtype"] or type(terminal.get("is_error")) is not bool):
            return None
        for event in events[1:-1]:
            if event.get("type") == "result" or (event.get("type") == "system" and event.get("subtype") == "init"):
                return None
            if event.get("type") == "assistant" and event.get("parent_tool_use_id") is None:
                message = event.get("message")
                if not isinstance(message, Mapping) or message.get("model") != expected_model:
                    return None
        tokens, _ = claude_model_usage(terminal, expected_model, allow_zero=True)
        return ReportedProviderUsage(CLAUDE_EVENT_CONTRACT, thread_id, tokens)
    except (UnicodeError, ValueError, TypeError, OverflowError, RecursionError):
        return None


@dataclass(frozen=True)
class ParsedClaudeTurnEvents:
    thread_id: str
    provider_tokens: int
    candidate_path: str
    write_tools: tuple[str, ...]
    normalization: str
    tool_activity: tuple[ProviderAuxiliaryActivity, ...]
    reported_models: tuple[str, ...]
    """Identifiers actually emitted by the main conversation; aliases unresolved."""


def parse_claude_turn_events(raw_events: bytes, *, expected_terminal_message: str) -> ParsedClaudeTurnEvents:
    """Require one completed native stream, coherent session and successful writes."""
    try:
        events = [_json(line) for line in raw_events.splitlines()]
        expected = _json(expected_terminal_message)
    except (UnicodeError, ValueError) as error:
        raise ValueError("Claude events or terminal expectation are not JSON") from error
    if (not _terminal(expected) or len(events) < 4 or
            any(not isinstance(event, Mapping) for event in events)):
        raise ValueError("Claude Turn boundary differs")
    initial, terminal = events[0], events[-1]
    if (initial.get("type") != "system" or initial.get("subtype") != "init" or
            terminal.get("type") != "result" or terminal.get("subtype") != "success" or
            terminal.get("is_error") is not False or terminal.get("permission_denials", []) != []
            or terminal.get("api_error_status") is not None):
        raise ValueError("Claude Turn did not complete under the declared event contract")
    # The envelope check below is about containment, so a write has to be judged at the
    # path the CLI actually resolves, not at its spelling. The init event reports the cwd
    # it ran in; a turn that does not report one admits absolute paths only.
    working_directory = initial.get("cwd")
    if working_directory is not None and (not isinstance(working_directory, str)
                                          or not Path(working_directory).is_absolute()):
        raise ValueError("Claude reported working directory differs")
    thread_id = initial.get("session_id")
    if (not isinstance(thread_id, str) or _THREAD_ID.fullmatch(thread_id) is None or
            terminal.get("session_id") != thread_id or any(
                event.get("session_id") != thread_id for event in events)):
        raise ValueError("Claude session identity differs")
    result = terminal.get("structured_output")
    if not _terminal(result) or _canonical_json_bytes(result) != _canonical_json_bytes(expected):
        raise ValueError("Claude native structured terminal message differs")

    tools: dict[str, dict] = {}
    completed: set[str] = set()
    writes: list[tuple[str, str]] = []
    models: list[str] = []
    activity: list[ProviderAuxiliaryActivity] = []
    # Where each invocation's auxiliary record sits, so a later errored result can restate
    # that one entry rather than adding a second record for the same item.
    errors: dict[str, int] = {}
    if isinstance(initial.get("model"), str) and initial["model"]:
        models.append(initial["model"])
    for event in events[1:-1]:
        if _metadata(event):
            if event.get("subtype") == "api_retry":
                activity.append(ProviderAuxiliaryActivity(event["uuid"], "api_retry", "observed"))
            elif event.get("subtype") == "post_turn_summary":
                activity.append(ProviderAuxiliaryActivity(
                    event["uuid"], "post_turn_summary", event["status_category"]))
            continue
        if event.get("type") not in ("assistant", "user"):
            raise ValueError("Claude event is outside the declared native contract")
        if event.get("parent_tool_use_id") is not None:
            raise ValueError("Claude subagents are outside the declared authoring tools")
        message = event.get("message")
        if not isinstance(message, Mapping) or not isinstance(message.get("content"), list):
            raise ValueError("Claude message content differs")
        if event["type"] == "assistant":
            if not isinstance(message.get("model"), str) or not message["model"]:
                raise ValueError("Claude main conversation model identity differs")
            if message["model"] not in models:
                models.append(message["model"])
        for block in message["content"]:
            if not isinstance(block, Mapping):
                raise ValueError("Claude content block differs")
            kind = block.get("type")
            if event["type"] == "assistant" and kind == "tool_use":
                identity, name, arguments = block.get("id"), block.get("name"), block.get("input")
                if (not isinstance(identity, str) or not identity or identity in tools or
                        name not in (*CLAUDE_AUTHORING_TOOLS, CLAUDE_TERMINAL_TOOL) or not isinstance(arguments, Mapping)):
                    raise ValueError("Claude tool invocation differs")
                tools[identity] = dict(block)
                errors[identity] = len(activity)
                activity.append(ProviderAuxiliaryActivity(identity, "tool_use", "completed", tool=name))
                if name == CLAUDE_TERMINAL_TOOL:
                    if not _terminal(arguments) or _canonical_json_bytes(arguments) != _canonical_json_bytes(expected):
                        raise ValueError("Claude schema terminal tool differs")
                if name in {"Write", "Edit"}:
                    path = arguments.get("file_path")
                    if not isinstance(path, str) or not path:
                        raise ValueError("Claude write is outside the candidate envelope")
                    resolved = Path(path)
                    if not resolved.is_absolute() and working_directory is not None:
                        resolved = Path(working_directory) / resolved
                    # Containment and identity are checked on the resolved path; a `..`
                    # anywhere in the spelling is still refused outright rather than
                    # normalized away, so no write can climb out of the envelope.
                    if (not resolved.is_absolute() or ".." in resolved.parts
                            or resolved.name != "candidate-set.json"):
                        raise ValueError("Claude write is outside the candidate envelope")
                    writes.append((str(resolved), name))
            elif event["type"] == "user" and kind == "tool_result":
                identity = block.get("tool_use_id")
                errored = block.get("is_error", False)
                # Pairing is still zero-tolerance: an unregistered id, a repeated
                # completion or a non-boolean flag is a structural mismatch and fatal.
                # A tool that reported an error is not. The provider sees that result and
                # keeps working inside the same turn, and the lifecycle checks below still
                # require every invocation to complete and the candidate to be written, so
                # a turn that did not recover cannot pass. Refusing here instead spent two
                # campaigns and roughly 7M provider tokens on probes the author survived.
                if (not isinstance(identity, str) or identity not in tools
                        or identity in completed or not isinstance(errored, bool)
                        # The terminal tool is not an authoring probe: it is how the Turn
                        # declares its own completion, so an error there stays fatal.
                        or errored and tools[identity].get("name") == CLAUDE_TERMINAL_TOOL):
                    raise ValueError("Claude tool completion differs")
                completed.add(identity)
                if errored:
                    activity[errors[identity]] = replace(
                        activity[errors[identity]], status="error_recovered")
            elif event["type"] == "assistant" and kind in ("text", "thinking", "redacted_thinking"):
                continue
            elif event["type"] == "user" and kind == "text" and event.get("isSynthetic") is True:
                # The CLI writes a turn of its own when a response carried no visible
                # output, which providers that answer with thinking alone trigger often.
                # It is content the Lab did not author, so it is retained as observed
                # activity rather than passed over: a run where the provider had to be
                # prompted to speak is not the same run as one where it did not. Only the
                # CLI's own synthetic turn is admitted here; an unmarked user text block
                # would still be someone injecting into the conversation, and stays fatal.
                activity.append(ProviderAuxiliaryActivity(
                    event["uuid"], "synthetic_continuation", "observed"))
            else:
                raise ValueError("Claude content is outside the declared event contract")
    if set(tools) != completed or not writes or len({path for path, _ in writes}) != 1:
        raise ValueError("Claude candidate write lifecycle is incomplete")
    if len(models) != 1 or models[0] != initial.get("model"):
        raise ValueError("Claude main conversation model identity differs")
    tokens, model_activity = claude_model_usage(terminal, models[0])
    return ParsedClaudeTurnEvents(
        thread_id=thread_id, provider_tokens=tokens, candidate_path=writes[0][0],
        write_tools=tuple(name for _, name in writes),
        normalization="claude_native_structured_output_exact",
        tool_activity=tuple(activity) + model_activity,
        reported_models=tuple(models),
    )


def normalize_claude_turn(raw_events: bytes, *, candidate_path: Path, expected_change: str,
                          expected_terminal_message: str, expected_thread_id: str | None = None,
                          event_contract: str = CLAUDE_EVENT_CONTRACT,
                          submission_contract: str = CANDIDATE_SET_ENVELOPE_V1,
                          arm: str | None = None, maximum_candidates_per_turn: int = 1) -> ProviderTurn:
    """Seal the existing candidate envelope; Python remains source inside its member."""
    if event_contract != CLAUDE_EVENT_CONTRACT or expected_change not in {"add", "update"}:
        raise ValueError("Claude event or candidate lifecycle contract differs")
    parsed = parse_claude_turn_events(raw_events, expected_terminal_message=expected_terminal_message)
    if (parsed.candidate_path != str(candidate_path.absolute()) or
            expected_change == "add" and parsed.write_tools[0] != "Write"):
        raise ValueError("Claude candidate path or initial write differs")
    if expected_thread_id is not None and parsed.thread_id != expected_thread_id:
        raise ValueError("Claude resumed session identity differs")
    submission = _read_candidate_nofollow(candidate_path)
    candidates = _project_candidate_submission(submission, submission_contract=submission_contract,
        arm=arm, maximum_candidates_per_turn=maximum_candidates_per_turn)
    return ProviderTurn(
        thread_id=parsed.thread_id, provider_tokens=parsed.provider_tokens, candidates=candidates,
        candidate_sha256s=tuple(sha256(candidate).hexdigest() for candidate in candidates),
        raw_submission=submission, raw_events=raw_events, raw_events_sha256=sha256(raw_events).hexdigest(),
        terminal_message=expected_terminal_message, terminal_message_count=1,
        normalization=parsed.normalization, tool_activity=parsed.tool_activity,
    )


class ClaudeInvocationBuilder:
    """Exact model/effort and persistent cwd; no qualification or model fallback."""

    def __init__(self, *, executable: Path, provider_revision: str, model: str,
                 reasoning_effort: str, workspace: Path, removed_environment: tuple[str, ...]) -> None:
        if any(not isinstance(value, str) or not value or "\x00" in value
               for value in (provider_revision, model, reasoning_effort)):
            raise ValueError("Claude invocation identity and effort are required")
        if reasoning_effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError("Claude effort is not supported by the declared CLI")
        if not removed_environment or len(set(removed_environment)) != len(removed_environment):
            raise ValueError("Claude removed environment names differ")
        self.executable = executable.resolve(strict=True)
        self.workspace = workspace.resolve(strict=True)
        if not self.executable.is_file() or not os.access(self.executable, os.X_OK) or not self.workspace.is_dir():
            raise ValueError("Claude executable or workspace differs")
        self.provider_revision = provider_revision
        self._model, self._effort = model, reasoning_effort
        self._removed_environment = removed_environment

    @property
    def configuration(self) -> Mapping[str, object]:
        return {"harness": "claude-code", "model": self._model, "reasoning_effort": self._effort,
                "permission_mode": "acceptEdits", "sandbox": "none", "safe_mode": True, "tools": list(CLAUDE_AUTHORING_TOOLS),
                "cwd_policy": "independent_task_workspace", "reference_visibility": "workspace_task_files",
                "removed_environment": list(self._removed_environment), "event_contract": CLAUDE_EVENT_CONTRACT,
                "submission_contract": CANDIDATE_SET_ENVELOPE_V1, "terminal_schema": terminal_schema()}

    def build(self, prompt: str, *, thread_id: str | None) -> ProviderInvocation:
        if not isinstance(prompt, str) or not prompt or "\x00" in prompt:
            raise ValueError("Claude provider prompt is required")
        if thread_id is not None and (not isinstance(thread_id, str) or _THREAD_ID.fullmatch(thread_id) is None):
            raise ValueError("Claude resume session id differs")
        tools = ",".join(CLAUDE_AUTHORING_TOOLS)
        # F-2026-09-10-008: auto-compact silently drops provider context mid-turn, which
        # changes what the author saw and breaks comparability between arms. This CLI has
        # no off switch -- `--autocompact` takes only `auto` or a 100k..1M window -- so the
        # boundary is pinned at the maximum it accepts and a compaction that still happens
        # is refused below rather than absorbed.
        arguments = (str(self.executable), "-p", "--output-format", "stream-json", "--verbose", "--safe-mode",
                     "--autocompact", CLAUDE_AUTOCOMPACT_WINDOW,
                     "--json-schema", _canonical_json_bytes(terminal_schema()).decode(), "--model", self._model, "--effort", self._effort, "--permission-mode", "acceptEdits",
                     "--tools", tools, "--allowedTools", tools)
        if thread_id is not None:
            arguments += ("--resume", thread_id)
        # The existing process owner takes argv and DEVNULL stdin. Its transport
        # must evolve explicitly if stdin prompts are required in live integration.
        return ProviderInvocation(arguments + ("--", prompt), self.workspace, "none",
                                  self.provider_revision, self._removed_environment, thread_id)


class ClaudeProviderAdapter:
    """One already-authorized invocation through the shared Lab supervisor."""

    def __init__(self, *, timeout_seconds: int = 1800) -> None:
        if type(timeout_seconds) is not int or timeout_seconds <= 0:
            raise ValueError("Claude provider timeout must be positive")
        self._timeout_seconds = timeout_seconds

    def execute(self, invocation: ProviderInvocation, *, candidate_path: Path, expected_change: str,
                expected_terminal_message: str, event_contract: str = CLAUDE_EVENT_CONTRACT,
                submission_contract: str = CANDIDATE_SET_ENVELOPE_V1,
                arm: str | None = None, maximum_candidates_per_turn: int = 1) -> ProviderTurn:
        if (invocation.sandbox != "none" or event_contract != CLAUDE_EVENT_CONTRACT or
                submission_contract != CANDIDATE_SET_ENVELOPE_V1 or expected_change not in {"add", "update"} or
                candidate_path.absolute() != invocation.cwd.absolute() / "candidate-set.json"):
            raise ValueError("Claude invocation or candidate contract differs")
        try:
            if (invocation.argv.count("--json-schema") != 1
                    or _canonical_json_bytes(_json(invocation.argv[invocation.argv.index("--json-schema") + 1])) != _canonical_json_bytes(terminal_schema())):
                raise ValueError("native schema differs")
        except (IndexError, UnicodeError, ValueError) as error:
            raise ValueError("Claude invocation native terminal schema differs") from error
        try:
            if invocation.argv.count("--model") != 1:
                raise ValueError("exact model flag differs")
            requested_model = invocation.argv[invocation.argv.index("--model") + 1]
            if not requested_model:
                raise ValueError("exact model value differs")
        except (IndexError, ValueError) as error:
            raise ValueError("Claude invocation exact model differs") from error
        try:
            completed = run_supervised(invocation.argv, cwd=invocation.cwd,
                environment=sanitized_environment(invocation.removed_environment), timeout_seconds=self._timeout_seconds)
        except (SupervisedProcessTimeout, SupervisedProcessOutputLimit) as error:
            raise RunProtocolFault("provider_fault", str(error), artifact_payloads={
                "provider_stdout": error.stdout, "provider_stderr": error.stderr},
                reported_usage=reported_claude_usage(error.stdout, expected_model=requested_model,
                    expected_thread_id=invocation.thread_id)) from error
        except OSError as error:
            raise RunProtocolFault("provider_fault", str(error)) from error
        try:
            if completed.returncode != 0:
                raise ValueError(f"Claude process failed with exit code {completed.returncode}")
            parsed = parse_claude_turn_events(completed.stdout, expected_terminal_message=expected_terminal_message)
            if parsed.reported_models != (requested_model,):
                raise ValueError("Claude reported model differs from the exact requested model")
            return normalize_claude_turn(completed.stdout, candidate_path=candidate_path,
                expected_change=expected_change, expected_terminal_message=expected_terminal_message,
                expected_thread_id=invocation.thread_id, event_contract=event_contract,
                submission_contract=submission_contract, arm=arm,
                maximum_candidates_per_turn=maximum_candidates_per_turn)
        except (OSError, ValueError, TypeError, OverflowError, RecursionError) as error:
            raise RunProtocolFault("provider_fault", str(error), artifact_payloads={
                "provider_stdout": completed.stdout, "provider_stderr": completed.stderr},
                reported_usage=reported_claude_usage(completed.stdout, expected_model=requested_model,
                    expected_thread_id=invocation.thread_id)) from error


class ClaudeRunProvider(QualifiedRunProvider):
    """Use the shared Lab workspace/TaskPackage lifecycle with Claude-native events.

    A receipt supplied here is still subject to TaskLab's Campaign admission and
    independent qualification audit. This class does not create that authority.
    """

    def __init__(self, *, qualification: ProviderQualificationReceipt,
                 builders: Mapping[str, ClaudeInvocationBuilder], task_packages: Mapping[str, TaskPackage],
                 adapter: ClaudeProviderAdapter | None = None) -> None:
        if not (qualification.initial_and_resume_equivalent and qualification.file_lifecycle_observed
                and qualification.usage_observed):
            raise ValueError("Claude provider qualification lacks observed capabilities")
        if any(builder.configuration.get("harness") != "claude-code" or
               builder.configuration.get("event_contract") != CLAUDE_EVENT_CONTRACT for builder in builders.values()):
            raise ValueError("Claude Run builder event contract differs")
        super().__init__(qualification=qualification, builders=builders, task_packages=task_packages,
                         adapter=adapter or ClaudeProviderAdapter())
