"""Provider invocation construction shared by initial and resumed Turns."""

from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Protocol, cast

from .faults import RunProtocolFault
from .process import (
    SupervisedProcessOutputLimit,
    SupervisedProcessTimeout,
    run_supervised,
    sanitized_environment,
)
from .task_package import TaskPackage, verify_task_package

_THREAD_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_MAX_CANDIDATE_BYTES = 64 * 1024 * 1024
SINGLE_CANDIDATE_V1 = "single_candidate_v1"
CANDIDATE_SET_ENVELOPE_V1 = "candidate_set_envelope_v1"
CODEX_DISABLED_FEATURES = (
    "apps",
    "auth_elicitation",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "code_mode_host",
    "computer_use",
    "goals",
    "guardian_approval",
    "hooks",
    "image_generation",
    "in_app_browser",
    "multi_agent",
    "plugin_sharing",
    "plugins",
    "remote_plugin",
    "shell_tool",
    "skill_mcp_dependency_install",
    "tool_call_mcp_elicitation",
    "tool_suggest",
    "workspace_dependencies",
)


def required_live_provider_qualification_scope(claim_scope: str) -> str:
    """Return the one live provider capability authorized by a matched Claim Scope."""

    if claim_scope == "artifact_optimization_only":
        return "live_two_turn_tool_rich_provider"
    if claim_scope in {"scientific_matched_search", "system_qualification_only"}:
        return "live_two_turn_current_provider"
    raise ValueError("matched Claim Scope has no live provider qualification")


def _read_candidate_nofollow(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.geteuid()
            or metadata.st_size <= 0
            or metadata.st_size > _MAX_CANDIDATE_BYTES
        ):
            raise ValueError("provider candidate owner, type, or size differs")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError("provider candidate read ended early")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.fstat(descriptor).st_size != metadata.st_size:
            raise ValueError("provider candidate changed while sealing")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def read_frozen_reference_bundle(root: Path) -> tuple[str, str]:
    """Read and hash one immutable external reference directory."""

    resolved = root.resolve(strict=True)
    if root.is_symlink() or not resolved.is_dir() or resolved.stat().st_mode & 0o222:
        raise ValueError("provider reference root custody differs")
    documents: list[tuple[str, bytes]] = []
    for path in sorted(resolved.iterdir()):
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_uid != os.geteuid()
                or metadata.st_mode & 0o222
                or metadata.st_size <= 0
                or metadata.st_size > 16 * 1024 * 1024
            ):
                raise ValueError("provider reference file custody differs")
            payload = os.read(descriptor, metadata.st_size)
            if len(payload) != metadata.st_size or os.read(descriptor, 1):
                raise ValueError("provider reference file changed while reading")
        finally:
            os.close(descriptor)
        documents.append((path.name, payload))
    if not documents:
        raise ValueError("provider reference bundle is empty")
    digest = sha256(
        b"".join(name.encode() + b"\0" + payload + b"\0" for name, payload in documents)
    ).hexdigest()
    try:
        rendered = "\n\n".join(
            f"===== {name} =====\n{payload.decode('utf-8')}" for name, payload in documents
        )
    except UnicodeError as error:
        raise ValueError("provider reference bundle is not UTF-8") from error
    return digest, rendered


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("provider feedback object keys must be strings")
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("provider feedback contains a non-JSON value")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _project_candidate_submission(
    payload: bytes,
    *,
    submission_contract: str,
    arm: str | None,
    maximum_candidates_per_turn: int,
) -> tuple[bytes, ...]:
    """Project one sealed provider file into the ordered semantic Candidate set."""

    if (
        not isinstance(maximum_candidates_per_turn, int)
        or isinstance(maximum_candidates_per_turn, bool)
        or maximum_candidates_per_turn <= 0
    ):
        raise ValueError("provider maximum candidates per Turn differs")
    if submission_contract == SINGLE_CANDIDATE_V1:
        if maximum_candidates_per_turn != 1 or arm is not None:
            raise ValueError("legacy provider submission contract differs")
        return (payload,)
    if submission_contract != CANDIDATE_SET_ENVELOPE_V1 or arm not in {
        "open_cake",
        "direct_cuda",
    }:
        raise ValueError("provider submission contract differs")
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("provider candidate-set envelope is not JSON") from error
    if (
        not isinstance(document, Mapping)
        or set(document) != {"schema_version", "arm", "candidates"}
        or document.get("schema_version") != 1
        or document.get("arm") != arm
        or payload != _canonical_json_bytes(document) + b"\n"
    ):
        raise ValueError("provider candidate-set envelope fields or canonical bytes differ")
    candidates = document.get("candidates")
    if (
        not isinstance(candidates, list)
        or not candidates
        or len(candidates) > maximum_candidates_per_turn
    ):
        raise ValueError("provider candidate-set envelope count differs")
    if arm == "open_cake":
        if any(not isinstance(candidate, Mapping) for candidate in candidates):
            raise ValueError("Open Cake candidate-set member is not a Schedule object")
        projected = tuple(_canonical_json_bytes(candidate) for candidate in candidates)
    else:
        if any(not isinstance(candidate, str) or not candidate for candidate in candidates):
            raise ValueError("direct CUDA candidate-set member is not non-empty source")
        projected = tuple(candidate.encode("utf-8") for candidate in candidates)
    if len({sha256(candidate).hexdigest() for candidate in projected}) != len(projected):
        raise ValueError("provider candidate-set contains duplicate Candidate bytes")
    return projected


@dataclass(frozen=True)
class ProviderInvocation:
    """Complete immutable provider process request."""

    argv: tuple[str, ...]
    cwd: Path
    sandbox: str
    provider_revision: str
    removed_environment: tuple[str, ...]
    thread_id: str | None


@dataclass(frozen=True)
class ProviderTurn:
    """Strict projection of one provider JSONL Turn and its sealed candidate."""

    thread_id: str
    provider_tokens: int
    candidates: tuple[bytes, ...]
    """Every candidate this Turn wrote, in the order the provider wrote them.

    A tuple rather than one, because the paper's loop generates structurally distinct
    candidates and ranks them before spending GPU time. One candidate leaves the ranking
    stage with nothing to rank. How many a Turn may write is declared by the Study
    Contract and granted identically to both arms (`docs/adr/0006`).
    """

    candidate_sha256s: tuple[str, ...]
    raw_events: bytes
    raw_events_sha256: str
    terminal_message: str
    terminal_message_count: int
    normalization: str
    tool_activity: tuple["ProviderAuxiliaryActivity", ...] = ()
    reference_bundle: bytes | None = None
    """Exact rendered reference bytes embedded in this Turn, when one exists."""


@dataclass(frozen=True)
class ProviderAuxiliaryActivity:
    """One provider-emitted non-submission item retained from raw JSONL."""

    item_id: str
    item_type: str
    status: str
    server: str | None = None
    tool: str | None = None

    @property
    def document(self) -> Mapping[str, object]:
        """Return the deterministic replay projection of this auxiliary item."""

        return {
            "item_id": self.item_id,
            "item_type": self.item_type,
            "status": self.status,
            "server": self.server,
            "tool": self.tool,
        }


@dataclass(frozen=True)
class ParsedCodexTurnEvents:
    """Filesystem-independent meaning of one admitted Codex JSONL Turn."""

    thread_id: str
    provider_tokens: int
    candidate_path: str | None
    change_kind: str | None
    terminal_message_count: int
    normalization: str
    tool_activity: tuple[ProviderAuxiliaryActivity, ...] = ()


@dataclass(frozen=True)
class ProviderQualificationReceipt:
    """Non-secret capability gate for a provider revision."""

    provider_revision: str
    executable_sha256: str
    configuration_sha256: str
    initial_and_resume_equivalent: bool
    file_lifecycle_observed: bool
    usage_observed: bool
    qualified: bool
    scope: str

    def __post_init__(self) -> None:
        if (
            not self.provider_revision
            or len(self.executable_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.executable_sha256)
            or len(self.configuration_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.configuration_sha256
            )
            or self.scope
            not in {
                "zero_gpu_contract_fixture_only",
                "live_two_turn_current_provider",
                "live_two_turn_tool_rich_provider",
            }
        ):
            raise ValueError("provider qualification identity differs")

    @classmethod
    def load(cls, path: str | Path) -> "ProviderQualificationReceipt":
        """Load the closed non-secret provider capability receipt."""

        document = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(document, Mapping) or set(document) != {
            "schema_version",
            "provider_revision",
            "executable_sha256",
            "configuration_sha256",
            "initial_and_resume_equivalent",
            "file_lifecycle_observed",
            "usage_observed",
            "qualified",
            "scope",
        } or document.get("schema_version") != 1:
            raise ValueError("provider qualification fields differ")
        return cls(
            provider_revision=str(document["provider_revision"]),
            executable_sha256=str(document["executable_sha256"]),
            configuration_sha256=str(document["configuration_sha256"]),
            initial_and_resume_equivalent=document["initial_and_resume_equivalent"] is True,
            file_lifecycle_observed=document["file_lifecycle_observed"] is True,
            usage_observed=document["usage_observed"] is True,
            qualified=document["qualified"] is True,
            scope=str(document["scope"]),
        )

    @property
    def document(self) -> Mapping[str, object]:
        """Return the canonical non-secret receipt document."""

        return {
            "schema_version": 1,
            "provider_revision": self.provider_revision,
            "executable_sha256": self.executable_sha256,
            "configuration_sha256": self.configuration_sha256,
            "initial_and_resume_equivalent": self.initial_and_resume_equivalent,
            "file_lifecycle_observed": self.file_lifecycle_observed,
            "usage_observed": self.usage_observed,
            "qualified": self.qualified,
            "scope": self.scope,
        }

    @property
    def canonical_sha256(self) -> str:
        return sha256(
            json.dumps(
                self.document,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()


def parse_codex_turn_events(
    raw_events: bytes,
    *,
    expected_terminal_message: str,
    event_contract: str = "closed_file_change_v1",
    legacy_candidate_name: str | None = None,
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
        candidate_file_change = item_type == "file_change" and (
            event_contract == "closed_file_change_v1"
            or (
                legacy_candidate_name is not None
                and isinstance(changes, list)
                and len(changes) == 1
                and isinstance(changes[0], Mapping)
                and isinstance(changes[0].get("path"), str)
                and Path(cast(str, changes[0]["path"])).name
                == legacy_candidate_name
            )
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
            activity_indices.append(index)
        else:
            raise ValueError("provider emitted an unadmitted item type")

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

    candidate_path: str | None = None
    change_kind: str | None = None
    start_index = min(activity_indices, default=2)
    stop_index = max(activity_indices, default=1)
    if file_events:
        def complete_file_change(
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

        if len(file_events) == 2:
            candidate_path, change_kind, _, _ = complete_file_change(
                file_events[0], file_events[1]
            )
            if change_kind not in {"add", "update"}:
                raise ValueError("provider candidate path or change kind differs")
        elif len(file_events) == 4:
            removed_path, removed_kind, _, removed_stop = complete_file_change(
                file_events[0], file_events[1]
            )
            added_path, added_kind, added_start, _ = complete_file_change(
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

    usage = typed_events[-1].get("usage")
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
        input_tokens + output_tokens <= 0
        or cached_input_tokens > input_tokens
        or cache_write_input_tokens > input_tokens
        or cast(int, usage.get("reasoning_output_tokens", 0)) > output_tokens
    ):
        raise ValueError("provider usage differs")
    return ParsedCodexTurnEvents(
        thread_id=thread_id,
        provider_tokens=input_tokens + output_tokens,
        candidate_path=candidate_path,
        change_kind=cast(str, change_kind),
        terminal_message_count=len(messages),
        normalization=normalization,
        tool_activity=tuple(tool_activity),
    )


def normalize_codex_turn(
    raw_events: bytes,
    *,
    candidate_path: Path,
    expected_change: str,
    expected_terminal_message: str,
    event_contract: str = "closed_file_change_v1",
    submission_contract: str = SINGLE_CANDIDATE_V1,
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
        raw_events=raw_events,
        raw_events_sha256=sha256(raw_events).hexdigest(),
        terminal_message=expected_terminal_message,
        terminal_message_count=parsed.terminal_message_count,
        normalization=parsed.normalization,
        tool_activity=parsed.tool_activity,
    )


class CodexProviderAdapter:
    """Execute one already-authorized Codex invocation and normalize its JSONL."""

    def __init__(self, *, timeout_seconds: int = 1800) -> None:
        if timeout_seconds <= 0:
            raise ValueError("provider timeout must be positive")
        self._timeout_seconds = timeout_seconds

    def execute(
        self,
        invocation: ProviderInvocation,
        *,
        candidate_path: Path,
        expected_change: str,
        expected_terminal_message: str,
        event_contract: str = "closed_file_change_v1",
        submission_contract: str = SINGLE_CANDIDATE_V1,
        arm: str | None = None,
        maximum_candidates_per_turn: int = 1,
    ) -> ProviderTurn:
        """Run without shell expansion and remove every contract-declared environment name."""

        environment = sanitized_environment(invocation.removed_environment)
        try:
            completed = run_supervised(
                invocation.argv,
                cwd=invocation.cwd,
                environment=environment,
                timeout_seconds=self._timeout_seconds,
            )
        except (SupervisedProcessTimeout, SupervisedProcessOutputLimit) as error:
            raise RunProtocolFault(
                "provider_fault",
                str(error),
                artifact_payloads={
                    "provider_stdout": error.stdout,
                    "provider_stderr": error.stderr,
                },
            ) from error
        if completed.returncode != 0:
            raise RunProtocolFault(
                "provider_fault",
                f"provider process failed with exit code {completed.returncode}",
                artifact_payloads={
                    "provider_stdout": completed.stdout,
                    "provider_stderr": completed.stderr,
                },
            )
        try:
            return normalize_codex_turn(
                completed.stdout,
                candidate_path=candidate_path,
                expected_change=expected_change,
                expected_terminal_message=expected_terminal_message,
                event_contract=event_contract,
                submission_contract=submission_contract,
                arm=arm,
                maximum_candidates_per_turn=maximum_candidates_per_turn,
            )
        except (OSError, ValueError) as error:
            raise RunProtocolFault(
                "provider_fault",
                str(error),
                artifact_payloads={
                    "provider_stdout": completed.stdout,
                    "provider_stderr": completed.stderr,
                },
            ) from error


class TurnRequestLike(Protocol):
    """Structural Lab Turn request used without creating an import cycle."""

    run_id: str
    arm: str
    turn: int
    cumulative_provider_tokens: int
    thread_id: str | None
    feedback: Mapping[str, object]
    maximum_candidates_per_turn: int
    state_card: Mapping[str, object] | None


class CodexRunProvider:
    """Canonical Campaign-Lock-compatible provider from Turn to normalized evidence."""

    _SINGLE_CANDIDATE_NAMES = {
        "open_cake": "candidate.json",
        "direct_cuda": "candidate.cu",
    }

    def __init__(
        self,
        *,
        qualification: ProviderQualificationReceipt,
        builders: Mapping[str, CodexInvocationBuilder],
        reference_roots: Mapping[str, Path],
        prompt_templates: Mapping[str, Path],
        task_packages: Mapping[str, TaskPackage] | None = None,
        adapter: CodexProviderAdapter | None = None,
    ) -> None:
        task_packages = dict(task_packages or {})
        task_mode = bool(task_packages)
        if (
            not qualification.qualified
            or qualification.scope
            not in {
                "live_two_turn_current_provider",
                "live_two_turn_tool_rich_provider",
            }
            or not builders
            or (
                task_mode
                and (
                    set(task_packages) != set(builders)
                    or reference_roots
                    or prompt_templates
                )
            )
            or (
                not task_mode
                and (
                    set(reference_roots) != set(builders)
                    or set(prompt_templates) != set(self._SINGLE_CANDIDATE_NAMES)
                )
            )
        ):
            raise ValueError("live Codex Run Provider authority differs")
        revisions = {builder.provider_revision for builder in builders.values()}
        executables = {builder.executable.resolve(strict=True) for builder in builders.values()}
        if revisions != {qualification.provider_revision} or len(executables) != 1:
            raise ValueError("Codex Run builders differ from provider qualification")
        executable = next(iter(executables))
        if sha256(executable.read_bytes()).hexdigest() != qualification.executable_sha256:
            raise ValueError("Codex executable bytes differ from provider qualification")
        templates: dict[str, str] = {}
        if not task_mode:
            for arm, path in prompt_templates.items():
                source = path.resolve(strict=True)
                if source.is_symlink() or not source.is_file():
                    raise ValueError("Codex prompt template custody differs")
                templates[arm] = source.read_text(encoding="utf-8")
        workspaces = [builder.workspace.absolute() for builder in builders.values()]
        if len(set(workspaces)) != len(workspaces):
            raise ValueError("each Run requires an independent workspace")
        configurations = {
            json.dumps(builder.configuration, sort_keys=True, separators=(",", ":"))
            for builder in builders.values()
        }
        if len(configurations) != 1:
            raise ValueError("Codex Run builder configurations differ")
        self.provider_revision = qualification.provider_revision
        self.qualification_sha256 = qualification.canonical_sha256
        self.executable_sha256 = qualification.executable_sha256
        self.configuration = next(iter(builders.values())).configuration
        configuration_sha256 = sha256(
            json.dumps(
                self.configuration, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        if configuration_sha256 != qualification.configuration_sha256:
            raise ValueError("Codex configuration differs from provider qualification")
        self._event_contract = str(
            self.configuration.get("event_contract", "closed_file_change_v1")
        )
        self._submission_contract = str(
            self.configuration.get("submission_contract", SINGLE_CANDIDATE_V1)
        )
        if self._submission_contract not in {
            SINGLE_CANDIDATE_V1,
            CANDIDATE_SET_ENVELOPE_V1,
        }:
            raise ValueError("Codex submission contract differs")
        self._references = {}
        if not task_mode:
            self._references = {
                run_id: (path.resolve(strict=True), *read_frozen_reference_bundle(path))
                for run_id, path in reference_roots.items()
            }
            if any(
                builder.workspace in self._references[run_id][0].parents
                or self._references[run_id][0] in builder.workspace.parents
                for run_id, builder in builders.items()
            ):
                raise ValueError("provider references must be outside each writable workspace")
        else:
            for run_id, package in task_packages.items():
                if package.run_id != run_id:
                    raise ValueError("Ralph task package Run identity differs")
                verify_task_package(builders[run_id].workspace, package)
        self._builders = dict(builders)
        self._templates = templates
        self._task_packages = task_packages
        self._adapter = adapter or CodexProviderAdapter()

    def _render_prompt(self, request: TurnRequestLike, candidate_path: Path) -> str:
        if self._task_packages:
            if request.state_card is None:
                raise ValueError("Ralph Turn requires a controller StateCard")
            state = json.dumps(
                _plain_json(request.state_card),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            return (
                "Read TASK.md and AGENTS.md completely. Continue the same Ralph Run "
                "under those immutable rules. Write only candidate-set.json. The "
                f"external controller StateCard for this iteration is: {state}"
            )
        feedback = json.dumps(
            _plain_json(request.feedback),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        replacements = {
            "{{RUN_ID}}": request.run_id,
            "{{ARM}}": request.arm,
            "{{TURN}}": str(request.turn),
            "{{CANDIDATE_PATH}}": str(candidate_path),
            "{{CUMULATIVE_PROVIDER_TOKENS}}": str(request.cumulative_provider_tokens),
            "{{FEEDBACK_JSON}}": feedback,
            "{{REFERENCE_BUNDLE}}": self._references[request.run_id][2],
        }
        if self._submission_contract == CANDIDATE_SET_ENVELOPE_V1:
            replacements["{{MAXIMUM_CANDIDATES_PER_TURN}}"] = str(
                request.maximum_candidates_per_turn
            )
        prompt = self._templates[request.arm]
        if set(re.findall(r"\{\{[A-Z0-9_]+\}\}", prompt)) != set(replacements):
            raise ValueError("Codex prompt template marker set differs")
        for marker, value in replacements.items():
            if prompt.count(marker) != 1:
                raise ValueError(f"Codex prompt template marker {marker!r} differs")
            prompt = prompt.replace(marker, value)
        return prompt

    def turn(self, request: TurnRequestLike) -> ProviderTurn:
        """Execute initial/add or same-thread resume/update under one environment."""

        builder = self._builders.get(request.run_id)
        if (
            builder is None
            or request.arm not in self._SINGLE_CANDIDATE_NAMES
            or request.turn <= 0
            or not isinstance(request.maximum_candidates_per_turn, int)
            or isinstance(request.maximum_candidates_per_turn, bool)
            or request.maximum_candidates_per_turn <= 0
            or (
                self._submission_contract == SINGLE_CANDIDATE_V1
                and request.maximum_candidates_per_turn != 1
            )
        ):
            raise ValueError("Codex Run or arm is outside the Campaign Lock")
        workspace = builder.workspace.absolute()
        if self._task_packages:
            package = self._task_packages[request.run_id]
            verify_task_package(workspace, package)
            reference_root = None
            reference_sha256 = None
        else:
            reference_root, reference_sha256, _ = self._references[request.run_id]
            if read_frozen_reference_bundle(reference_root)[0] != reference_sha256:
                raise ValueError("provider references changed before Turn")
        if request.turn == 1:
            expected_initial_entries = (
                {workspace / "TASK.md", workspace / "AGENTS.md"}
                if self._task_packages
                else set()
            )
            if (
                request.thread_id is not None
                or not workspace.is_dir()
                or set(workspace.iterdir()) != expected_initial_entries
            ):
                raise ValueError("initial Codex Turn requires one empty workspace")
        elif request.thread_id is None:
            raise ValueError("resumed Codex Turn requires the existing thread")
        candidate_path = workspace / (
            "candidate-set.json"
            if self._submission_contract == CANDIDATE_SET_ENVELOPE_V1
            else self._SINGLE_CANDIDATE_NAMES[request.arm]
        )
        expected_change = "add" if request.turn == 1 else "update"
        if (expected_change == "add" and candidate_path.exists()) or (
            expected_change == "update" and not candidate_path.is_file()
        ):
            raise ValueError("Codex candidate lifecycle differs before invocation")
        prompt = self._render_prompt(request, candidate_path)
        terminal_document: dict[str, object] = {
            "arm": request.arm,
            "candidate_written": True,
            "kind": "open_cake_ir_turn",
            "turn": request.turn,
        }
        if self._event_contract == "closed_file_change_v1":
            terminal_document["tool_calls"] = 1
        terminal = json.dumps(
            terminal_document,
            sort_keys=True,
            separators=(",", ":"),
        )
        invocation = builder.build(prompt, thread_id=request.thread_id)
        result = self._adapter.execute(
            invocation,
            candidate_path=candidate_path,
            expected_change=expected_change,
            expected_terminal_message=terminal,
            event_contract=self._event_contract,
            submission_contract=self._submission_contract,
            arm=(
                request.arm
                if self._submission_contract == CANDIDATE_SET_ENVELOPE_V1
                else None
            ),
            maximum_candidates_per_turn=request.maximum_candidates_per_turn,
        )
        if self._submission_contract == CANDIDATE_SET_ENVELOPE_V1:
            entries = list(workspace.iterdir())
            expected_entries = {candidate_path}
            if self._task_packages:
                expected_entries.update({workspace / "TASK.md", workspace / "AGENTS.md"})
            if (
                set(entries) != expected_entries
                or candidate_path.is_symlink()
                or not candidate_path.is_file()
            ):
                raise RunProtocolFault(
                    "provider_fault",
                    "Codex candidate-set workspace custody differs",
                )
        if self._task_packages:
            try:
                verify_task_package(workspace, package)
            except ValueError as error:
                raise RunProtocolFault("contamination", str(error)) from error
            reference_bundle = package.evidence_bundle(
                cast(Mapping[str, object], request.state_card)
            )
        else:
            assert reference_root is not None and reference_sha256 is not None
            if read_frozen_reference_bundle(reference_root)[0] != reference_sha256:
                raise RunProtocolFault("contamination", "provider references changed during Turn")
            reference_bundle = self._references[request.run_id][2].encode("utf-8")
        return replace(
            result,
            reference_bundle=reference_bundle,
        )


class CodexInvocationBuilder:
    """Build Codex initial and resume invocations from one shared contract."""

    def __init__(
        self,
        *,
        executable: Path,
        provider_revision: str,
        model: str,
        reasoning_effort: str,
        service_tier: str,
        workspace: Path,
        output_schema: Path,
        removed_environment: tuple[str, ...],
        disabled_features: tuple[str, ...] = CODEX_DISABLED_FEATURES,
        event_contract: str = "closed_file_change_v1",
        submission_contract: str = SINGLE_CANDIDATE_V1,
        cwd_policy: str = "independent_empty_workspace",
        reference_visibility: str = "embedded_frozen_bundle",
    ) -> None:
        values = (provider_revision, model, reasoning_effort, service_tier)
        if any(not value for value in values) or not removed_environment:
            raise ValueError("Codex invocation authority fields are required")
        if len(set(removed_environment)) != len(removed_environment):
            raise ValueError("removed environment names must be unique")
        if (
            len(set(disabled_features)) != len(disabled_features)
            or any(not isinstance(feature, str) or not feature for feature in disabled_features)
        ):
            raise ValueError("disabled feature names must be unique and non-empty")
        if (disabled_features, event_contract) not in {
            (CODEX_DISABLED_FEATURES, "closed_file_change_v1"),
            ((), "tool_rich_candidate_v1"),
        }:
            raise ValueError("Codex feature and event contracts differ")
        if submission_contract not in {
            SINGLE_CANDIDATE_V1,
            CANDIDATE_SET_ENVELOPE_V1,
        }:
            raise ValueError("Codex submission contract differs")
        if (cwd_policy, reference_visibility) not in {
            ("independent_empty_workspace", "embedded_frozen_bundle"),
            ("independent_task_workspace", "workspace_task_files"),
        }:
            raise ValueError("Codex workspace/reference policy differs")
        self._executable = executable
        self._provider_revision = provider_revision
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._service_tier = service_tier
        self._workspace = workspace.resolve(strict=True)
        self._output_schema = output_schema.resolve(strict=True)
        self._removed_environment = removed_environment
        self._disabled_features = disabled_features
        self._event_contract = event_contract
        self._submission_contract = submission_contract
        self._cwd_policy = cwd_policy
        self._reference_visibility = reference_visibility

    @property
    def workspace(self) -> Path:
        return self._workspace

    @property
    def provider_revision(self) -> str:
        return self._provider_revision

    @property
    def executable(self) -> Path:
        return self._executable

    @property
    def configuration(self) -> Mapping[str, object]:
        configuration: dict[str, object] = {
            "model": self._model,
            "reasoning_effort": self._reasoning_effort,
            "service_tier": self._service_tier,
            "output_schema_sha256": sha256(self._output_schema.read_bytes()).hexdigest(),
            "removed_environment": list(self._removed_environment),
            "sandbox": "workspace-write",
            "cwd_policy": self._cwd_policy,
            "reference_visibility": self._reference_visibility,
            "disabled_features": list(self._disabled_features),
        }
        if self._event_contract != "closed_file_change_v1":
            configuration["event_contract"] = self._event_contract
        if self._submission_contract != SINGLE_CANDIDATE_V1:
            configuration["submission_contract"] = self._submission_contract
        return configuration

    @property
    def configuration_sha256(self) -> str:
        """Hash the complete non-secret invocation policy qualified for every Turn."""

        return sha256(
            json.dumps(
                self.configuration, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()

    def build(self, prompt: str, *, thread_id: str | None) -> ProviderInvocation:
        """Build one Turn while keeping initial/resume invariants identical."""

        if not isinstance(prompt, str) or not prompt:
            raise ValueError("provider prompt is required")
        if thread_id is not None and _THREAD_ID.fullmatch(thread_id) is None:
            raise ValueError("provider thread_id is invalid")
        common = (
            "--ignore-user-config",
            "--ignore-rules",
            "--json",
            "--strict-config",
            "--skip-git-repo-check",
            "--model",
            self._model,
            "--config",
            f'model_reasoning_effort="{self._reasoning_effort}"',
            "--config",
            f'service_tier="{self._service_tier}"',
            "--config",
            'approval_policy="never"',
            "--config",
            'sandbox_mode="workspace-write"',
            "--output-schema",
            str(self._output_schema),
        ) + tuple(
            value
            for feature in self._disabled_features
            for value in ("--disable", feature)
        )
        prefix = (
            (str(self._executable), "exec")
            if thread_id is None
            else (str(self._executable), "exec", "resume")
        )
        suffix = (prompt,) if thread_id is None else (thread_id, prompt)
        return ProviderInvocation(
            argv=prefix + common + suffix,
            cwd=self._workspace,
            sandbox="workspace-write",
            provider_revision=self._provider_revision,
            removed_environment=self._removed_environment,
            thread_id=thread_id,
        )
