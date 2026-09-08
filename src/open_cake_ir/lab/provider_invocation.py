"""Resolve the pinned Codex host and build initial or resumed process requests."""

from __future__ import annotations

from open_cake_ir.serialization import canonical_json_bytes

import os, stat
from hashlib import sha256
from pathlib import Path
from typing import Mapping

from .process import sanitized_environment
from .provider_documents import (
    CANDIDATE_SET_ENVELOPE_V1,
    CODEX_DISABLED_FEATURES,
    ProviderInvocation,
    _THREAD_ID,
)


def resolve_codex_code_mode_host(
    executable: Path, *, expected: Mapping[str, object] | None = None,
    removed_environment: tuple[str, ...] = (),
) -> dict[str, str]:
    """Bind the native CLI's selected local helper, never a helper override.

    Lookup follows codex-rs/install-context at rust-v0.153.4: package resources,
    legacy standalone resources, then the package bin or executable sibling.
    The digest establishes the qualified runtime's byte identity, not correctness.
    """

    executable = executable.resolve(strict=True)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("Codex native executable is not an executable file")
    environment = sanitized_environment(removed_environment)
    if "CODEX_HOME" in environment:
        # The CLI resolves this in invocation.cwd, which differs from our cwd.
        # Only an existing absolute directory gives both processes one identity.
        codex_home = Path(environment["CODEX_HOME"])
        if not codex_home.is_absolute():
            raise ValueError("Codex CODEX_HOME must be an existing absolute directory")
        try:
            codex_home = codex_home.resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as error:
            raise ValueError("Codex CODEX_HOME must be an existing absolute directory") from error
        if not codex_home.is_dir():
            raise ValueError("Codex CODEX_HOME must be an existing absolute directory")
    else:
        codex_home = (Path.home() / ".codex").resolve()
    directory = executable.parent
    package_bin = None
    if directory.name in {"bin", "codex-resources"}:
        package_bin = directory.parent / "bin"
    elif (directory.name == "MacOS" and directory.parent.name == "Contents"
          and directory.parent.parent.name == "CodexCLI.app"):
        package_bin = directory.parent.parent.parent / "bin"
    if package_bin is not None and not (
        package_bin.is_dir() and (package_bin.parent / "codex-package.json").is_file()
    ):
        package_bin = None
    candidates = []
    if package_bin is not None:
        candidates.append(package_bin.parent / "codex-resources" / "codex-code-mode-host")
    release_dir = package_bin.parent if package_bin is not None else directory
    managed_override = any(name in environment for name in (
        "CODEX_MANAGED_BY_VITE_PLUS", "CODEX_MANAGED_BY_PNPM",
        "CODEX_MANAGED_BY_NPM", "CODEX_MANAGED_BY_BUN",
    ))
    if (not managed_override
        and release_dir.is_relative_to(codex_home / "packages/standalone/releases")):
        candidates.append(release_dir / "codex-resources" / "codex-code-mode-host")
    candidates.append((package_bin or directory) / "codex-code-mode-host")
    # Native lookup falls back to the executable sibling if package bin has no helper.
    candidates.append(directory / "codex-code-mode-host")
    for helper in dict.fromkeys(candidates):
        if helper.is_symlink():
            raise ValueError("Codex Code Mode host must not be a symlink")
        if not helper.is_file():
            continue
        if helper.resolve(strict=True) != helper:
            raise ValueError("Codex Code Mode host path is not canonical")
        descriptor = os.open(helper, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            before = os.fstat(descriptor)
            if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_uid not in {0, os.geteuid()} or before.st_size <= 0
                or before.st_mode & 0o022 or not os.access(helper, os.X_OK)):
                raise ValueError("Codex Code Mode host custody or executable mode differs")
            digest = sha256()
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
            identity_fields = lambda value: (value.st_dev, value.st_ino, value.st_mode,
                value.st_nlink, value.st_uid, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
            if (identity_fields(os.fstat(descriptor)) != identity_fields(before)
                or identity_fields(helper.stat()) != identity_fields(before)
                or helper.resolve(strict=True) != helper):
                raise ValueError("Codex Code Mode host changed while binding")
        finally:
            os.close(descriptor)
        identity = {"path": str(helper), "sha256": digest.hexdigest()}
        if expected is not None and identity != expected:
            raise ValueError("Codex Code Mode host differs from the bound runtime")
        return identity
    raise ValueError("Codex Code Mode host is missing beside the native CLI")


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
        submission_contract: str = CANDIDATE_SET_ENVELOPE_V1,
        cwd_policy: str = "independent_task_workspace",
        reference_visibility: str = "workspace_task_files",
        code_mode_host: Mapping[str, object] | None = None,
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
        if submission_contract != CANDIDATE_SET_ENVELOPE_V1:
            raise ValueError("Codex submission contract differs")
        if (cwd_policy, reference_visibility) not in {
            ("independent_task_workspace", "workspace_task_files"),
        }:
            raise ValueError("Codex workspace/reference policy differs")
        self._executable = executable.resolve(strict=True)
        self._code_mode_host = resolve_codex_code_mode_host(
            self._executable, expected=code_mode_host, removed_environment=removed_environment,
        )
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
            "code_mode_host": dict(self._code_mode_host),
        }
        if self._event_contract == "closed_file_change_v1":
            configuration["web_search"] = "disabled"
        if self._event_contract != "closed_file_change_v1":
            configuration["event_contract"] = self._event_contract
        configuration["submission_contract"] = self._submission_contract
        return configuration

    @property
    def configuration_sha256(self) -> str:
        """Hash the complete non-secret invocation policy qualified for every Turn."""

        return sha256(
            canonical_json_bytes(self.configuration)
        ).hexdigest()

    def build(self, prompt: str, *, thread_id: str | None) -> ProviderInvocation:
        """Build one Turn while keeping initial/resume invariants identical."""

        if not isinstance(prompt, str) or not prompt:
            raise ValueError("provider prompt is required")
        if thread_id is not None and _THREAD_ID.fullmatch(thread_id) is None:
            raise ValueError("provider thread_id is invalid")
        resolve_codex_code_mode_host(
            self._executable, expected=self._code_mode_host,
            removed_environment=self._removed_environment,
        )
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
            "--enable",
            "code_mode_host",
        ) + (("--config", 'web_search="disabled"')
             if self._event_contract == "closed_file_change_v1" else ()) + tuple(
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
