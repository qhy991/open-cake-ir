"""Claude-native provider pieces for the existing Lab Turn boundary.

This module does not admit a Campaign or create a qualification. TaskLab's
provider configuration, live qualification and Replay dispatch still need an
explicit Claude contract. Flags follow Claude Code 2.1.241 local help and
https://code.claude.com/docs/en/headless. No Codex events are manufactured.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping

from .faults import RunProtocolFault
from .task_package import TaskPackage
from .process import (
    SupervisedProcessOutputLimit, SupervisedProcessTimeout,
    run_supervised, sanitized_environment,
)
from .providers import (
    CANDIDATE_SET_ENVELOPE_V1, ProviderAuxiliaryActivity, ProviderInvocation,
    ProviderTurn, ProviderQualificationReceipt, QualifiedRunProvider, _THREAD_ID, _canonical_json_bytes, _project_candidate_submission,
    _read_candidate_nofollow, _reject_json_constant, _unique_json_object,
)

CLAUDE_EVENT_CONTRACT = "claude_stream_candidate_v1"
CLAUDE_AUTHORING_TOOLS = ("Read", "Write", "Edit", "Glob", "Grep")
CLAUDE_USAGE_FIELDS = (
    "input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
)


def _json(payload: str | bytes):
    return json.loads(payload, object_pairs_hook=_unique_json_object, parse_constant=_reject_json_constant)


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
    if (not isinstance(expected, Mapping) or len(events) < 4 or
            any(not isinstance(event, Mapping) for event in events)):
        raise ValueError("Claude Turn boundary differs")
    initial, terminal = events[0], events[-1]
    if (initial.get("type") != "system" or initial.get("subtype") != "init" or
            terminal.get("type") != "result" or terminal.get("subtype") != "success" or
            terminal.get("is_error") is not False or
            any(event.get("type") not in ("assistant", "user") for event in events[1:-1])):
        raise ValueError("Claude Turn did not complete under the declared event contract")
    thread_id = initial.get("session_id")
    if (not isinstance(thread_id, str) or _THREAD_ID.fullmatch(thread_id) is None or
            terminal.get("session_id") != thread_id or any(
                "session_id" in event and event["session_id"] != thread_id for event in events)):
        raise ValueError("Claude session identity differs")
    result = terminal.get("result")
    try:
        if not isinstance(result, str) or _canonical_json_bytes(_json(result)) != _canonical_json_bytes(expected):
            raise ValueError("Claude terminal message differs")
    except (UnicodeError, ValueError) as error:
        raise ValueError("Claude terminal message differs") from error
    usage = terminal.get("usage")
    if not isinstance(usage, Mapping) or any(
            type(usage.get(field)) is not int or usage[field] < 0 for field in CLAUDE_USAGE_FIELDS):
        raise ValueError("Claude complete usage is missing or malformed")
    # Anthropic cache counters are separate inputs. Nested duration breakdowns
    # partition cache_creation_input_tokens and must never be added a second time.
    tokens = sum(usage[field] for field in CLAUDE_USAGE_FIELDS)
    if tokens <= 0:
        raise ValueError("Claude provider usage must be positive")

    tools: dict[str, dict] = {}
    completed: set[str] = set()
    writes: list[tuple[str, str]] = []
    models: list[str] = []
    if isinstance(initial.get("model"), str) and initial["model"]:
        models.append(initial["model"])
    for event in events[1:-1]:
        if event.get("parent_tool_use_id") is not None:
            raise ValueError("Claude subagents are outside the declared authoring tools")
        message = event.get("message")
        if not isinstance(message, Mapping) or not isinstance(message.get("content"), list):
            raise ValueError("Claude message content differs")
        if event["type"] == "assistant" and isinstance(message.get("model"), str) and message["model"]:
            if message["model"] not in models:
                models.append(message["model"])
        for block in message["content"]:
            if not isinstance(block, Mapping):
                raise ValueError("Claude content block differs")
            kind = block.get("type")
            if event["type"] == "assistant" and kind == "tool_use":
                identity, name, arguments = block.get("id"), block.get("name"), block.get("input")
                if (not isinstance(identity, str) or not identity or identity in tools or
                        name not in CLAUDE_AUTHORING_TOOLS or not isinstance(arguments, Mapping)):
                    raise ValueError("Claude tool invocation differs")
                tools[identity] = dict(block)
                if name in {"Write", "Edit"}:
                    path = arguments.get("file_path")
                    if (not isinstance(path, str) or not Path(path).is_absolute() or
                            ".." in Path(path).parts or Path(path).name != "candidate-set.json"):
                        raise ValueError("Claude write is outside the candidate envelope")
                    writes.append((path, name))
            elif event["type"] == "user" and kind == "tool_result":
                identity = block.get("tool_use_id")
                if (not isinstance(identity, str) or identity not in tools or identity in completed or
                        block.get("is_error", False) is not False):
                    raise ValueError("Claude tool completion differs")
                completed.add(identity)
            elif event["type"] == "assistant" and kind in ("text", "thinking", "redacted_thinking"):
                continue
            else:
                raise ValueError("Claude content is outside the declared event contract")
    if set(tools) != completed or not writes or len({path for path, _ in writes}) != 1:
        raise ValueError("Claude candidate write lifecycle is incomplete")
    return ParsedClaudeTurnEvents(
        thread_id=thread_id, provider_tokens=tokens, candidate_path=writes[0][0],
        write_tools=tuple(name for _, name in writes),
        normalization="claude_result_exact" if result == expected_terminal_message else "claude_result_semantic",
        tool_activity=tuple(ProviderAuxiliaryActivity(identity, "tool_use", "completed", tool=block["name"])
                            for identity, block in tools.items()),
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
                "submission_contract": CANDIDATE_SET_ENVELOPE_V1}

    def build(self, prompt: str, *, thread_id: str | None) -> ProviderInvocation:
        if not isinstance(prompt, str) or not prompt or "\x00" in prompt:
            raise ValueError("Claude provider prompt is required")
        if thread_id is not None and (not isinstance(thread_id, str) or _THREAD_ID.fullmatch(thread_id) is None):
            raise ValueError("Claude resume session id differs")
        tools = ",".join(CLAUDE_AUTHORING_TOOLS)
        arguments = (str(self.executable), "-p", "--output-format", "stream-json", "--verbose", "--safe-mode",
                     "--model", self._model, "--effort", self._effort, "--permission-mode", "acceptEdits",
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
            completed = run_supervised(invocation.argv, cwd=invocation.cwd,
                environment=sanitized_environment(invocation.removed_environment), timeout_seconds=self._timeout_seconds)
        except (SupervisedProcessTimeout, SupervisedProcessOutputLimit) as error:
            raise RunProtocolFault("provider_fault", str(error), artifact_payloads={
                "provider_stdout": error.stdout, "provider_stderr": error.stderr}) from error
        except OSError as error:
            raise RunProtocolFault("provider_fault", str(error)) from error
        try:
            if completed.returncode != 0:
                raise ValueError(f"Claude process failed with exit code {completed.returncode}")
            return normalize_claude_turn(completed.stdout, candidate_path=candidate_path,
                expected_change=expected_change, expected_terminal_message=expected_terminal_message,
                expected_thread_id=invocation.thread_id, event_contract=event_contract,
                submission_contract=submission_contract, arm=arm,
                maximum_candidates_per_turn=maximum_candidates_per_turn)
        except (OSError, ValueError) as error:
            raise RunProtocolFault("provider_fault", str(error), artifact_payloads={
                "provider_stdout": completed.stdout, "provider_stderr": completed.stderr}) from error


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
