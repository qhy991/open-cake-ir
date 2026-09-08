"""Supervised provider turns; public provider names remain direct aliases to their owners."""

from __future__ import annotations

from open_cake_ir.serialization import canonical_json_bytes

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Protocol

from .faults import RunProtocolFault
from .process import (
    SupervisedProcessOutputLimit,
    SupervisedProcessTimeout,
    run_supervised,
    sanitized_environment,
)
from .task_package import TaskPackage, verify_task_package, render_task_request
from .provider_documents import (
    CANDIDATE_SET_ENVELOPE_V1,
    ProviderInvocation,
    ProviderQualificationReceipt,
    ProviderTurn,
    CODEX_DISABLED_FEATURES,
    ProviderAuxiliaryActivity,
    ParsedCodexTurnEvents,
    required_live_provider_qualification_scope,
    _project_candidate_submission,
)
from .provider_events import normalize_codex_turn, parse_codex_turn_events, reported_codex_usage, reported_provider_usage
from .provider_invocation import CodexInvocationBuilder, resolve_codex_code_mode_host


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
        submission_contract: str = CANDIDATE_SET_ENVELOPE_V1,
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
                reported_usage=reported_codex_usage(error.stdout, event_contract=event_contract,
                                                     expected_thread_id=invocation.thread_id),
            ) from error
        if completed.returncode != 0:
            raise RunProtocolFault(
                "provider_fault",
                f"provider process failed with exit code {completed.returncode}",
                artifact_payloads={
                    "provider_stdout": completed.stdout,
                    "provider_stderr": completed.stderr,
                },
                reported_usage=reported_codex_usage(completed.stdout, event_contract=event_contract,
                                                     expected_thread_id=invocation.thread_id),
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
                reported_usage=reported_codex_usage(completed.stdout, event_contract=event_contract,
                                                     expected_thread_id=invocation.thread_id),
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


class InvocationBuilder(Protocol):
    workspace: Path
    executable: Path
    provider_revision: str

    @property
    def configuration(self) -> Mapping[str, object]: ...

    def build(self, prompt: str, *, thread_id: str | None) -> ProviderInvocation: ...


class ProviderAdapter(Protocol):
    def execute(
        self, invocation: ProviderInvocation, *, candidate_path: Path,
        expected_change: str, expected_terminal_message: str, event_contract: str,
        submission_contract: str, arm: str | None, maximum_candidates_per_turn: int,
    ) -> ProviderTurn: ...


class QualifiedRunProvider:
    """Shared qualified Run lifecycle; each harness supplies its own adapter."""


    def __init__(
        self,
        *,
        qualification: ProviderQualificationReceipt,
        builders: Mapping[str, InvocationBuilder],
        task_packages: Mapping[str, TaskPackage],
        adapter: ProviderAdapter,
    ) -> None:
        if (
            not qualification.qualified
            or qualification.scope not in {
                "live_two_turn_current_provider", "live_two_turn_tool_rich_provider",
            }
            or not builders
            or set(task_packages) != set(builders)
        ):
            raise ValueError("live provider Run Provider authority differs")
        revisions = {builder.provider_revision for builder in builders.values()}
        executables = {builder.executable.resolve(strict=True) for builder in builders.values()}
        if revisions != {qualification.provider_revision} or len(executables) != 1:
            raise ValueError("provider Run builders differ from provider qualification")
        executable = next(iter(executables))
        if sha256(executable.read_bytes()).hexdigest() != qualification.executable_sha256:
            raise ValueError("provider executable bytes differ from provider qualification")
        workspaces = [builder.workspace.absolute() for builder in builders.values()]
        if len(set(workspaces)) != len(workspaces):
            raise ValueError("each Run requires an independent workspace")
        configurations = {
            json.dumps(builder.configuration, sort_keys=True, separators=(",", ":"))
            for builder in builders.values()
        }
        if len(configurations) != 1:
            raise ValueError("provider Run builder configurations differ")
        self.provider_revision = qualification.provider_revision
        self.qualification_sha256 = qualification.canonical_sha256
        self.executable_sha256 = qualification.executable_sha256
        self.configuration = next(iter(builders.values())).configuration
        configuration_sha256 = sha256(
            canonical_json_bytes(self.configuration)
        ).hexdigest()
        if configuration_sha256 != qualification.configuration_sha256:
            raise ValueError("provider configuration differs from provider qualification")
        self._event_contract = str(
            self.configuration.get("event_contract", "closed_file_change_v1")
        )
        self._submission_contract = str(
            self.configuration.get("submission_contract")
        )
        if self._submission_contract != CANDIDATE_SET_ENVELOPE_V1:
            raise ValueError("provider submission contract differs")
        for run_id, package in task_packages.items():
            if package.run_id != run_id:
                raise ValueError("Ralph task package Run identity differs")
            verify_task_package(builders[run_id].workspace, package)
        self._builders = dict(builders)
        self._task_packages = task_packages
        self._adapter = adapter

    def turn(self, request: TurnRequestLike) -> ProviderTurn:
        """Execute initial/add or same-thread resume/update under one environment."""

        builder = self._builders.get(request.run_id)
        if (
            builder is None
            or request.arm not in {"open_cake", "direct_cuda", "native_triton", "native_cute_dsl"}
            or request.turn <= 0
            or not isinstance(request.maximum_candidates_per_turn, int)
            or isinstance(request.maximum_candidates_per_turn, bool)
            or request.maximum_candidates_per_turn <= 0
        ):
            raise ValueError("provider Run or arm is outside the Campaign Lock")
        workspace = builder.workspace.absolute()
        package = self._task_packages[request.run_id]
        verify_task_package(workspace, package)
        if package.arm != request.arm:
            raise ValueError('Ralph request arm differs from the task package')
        if request.turn == 1:
            expected_initial_entries = (
                {workspace / "TASK.md", workspace / "AGENTS.md"}
            )
            if (
                request.thread_id is not None
                or not workspace.is_dir()
                or set(workspace.iterdir()) != expected_initial_entries
            ):
                raise ValueError("initial provider Turn requires one empty workspace")
        elif request.thread_id is None:
            raise ValueError("resumed provider Turn requires the existing thread")
        candidate_path = workspace / "candidate-set.json"
        expected_change = "add" if request.turn == 1 else "update"
        if (expected_change == "add" and candidate_path.exists()) or (
            expected_change == "update" and not candidate_path.is_file()
        ):
            raise ValueError("provider candidate lifecycle differs before invocation")
        prompt, reference_bundle = render_task_request(package, request.state_card)
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
            arm=request.arm,
            maximum_candidates_per_turn=request.maximum_candidates_per_turn,
        )
        entries = list(workspace.iterdir())
        expected_entries = {candidate_path}
        expected_entries.update({workspace / "TASK.md", workspace / "AGENTS.md"})
        if (
            set(entries) != expected_entries
            or candidate_path.is_symlink()
            or not candidate_path.is_file()
        ):
            raise RunProtocolFault(
                "provider_fault",
                "provider candidate-set workspace custody differs",
                artifact_payloads={"provider_stdout": result.raw_events},
                reported_usage=reported_provider_usage(result.raw_events, provider=self.configuration,
                                                       expected_thread_id=request.thread_id),
            )
        try:
            verify_task_package(workspace, package)
        except ValueError as error:
            raise RunProtocolFault("contamination", str(error),
                artifact_payloads={"provider_stdout": result.raw_events},
                reported_usage=reported_provider_usage(result.raw_events, provider=self.configuration,
                                                       expected_thread_id=request.thread_id)) from error
        return replace(
            result,
            reference_bundle=reference_bundle,
        )


class CodexRunProvider(QualifiedRunProvider):
    """Existing Codex entrypoint with its native event adapter."""

    def __init__(
        self, *, qualification: ProviderQualificationReceipt,
        builders: Mapping[str, CodexInvocationBuilder], task_packages: Mapping[str, TaskPackage],
        adapter: CodexProviderAdapter | None = None,
    ) -> None:
        super().__init__(qualification=qualification, builders=builders, task_packages=task_packages,
                         adapter=adapter or CodexProviderAdapter())
