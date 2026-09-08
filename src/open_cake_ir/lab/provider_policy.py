"""One provider configuration projection for Lab preflight, execution and replay."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .providers import CANDIDATE_SET_ENVELOPE_V1, CODEX_DISABLED_FEATURES, _canonical_json_bytes
from .claude import CLAUDE_EVENT_CONTRACT, CLAUDE_AUTHORING_TOOLS, terminal_schema

_AUTHORITY = {"revision", "qualification", "qualification_anchor", "executable_sha256"}
_COMMON = {"model", "reasoning_effort", "removed_environment", "sandbox", "cwd_policy", "reference_visibility"}
_CODEX = _COMMON | {"service_tier", "output_schema", "disabled_features", "code_mode_host"}
_CLAUDE = _COMMON | {"harness", "permission_mode", "safe_mode", "tools", "event_contract", "terminal_schema"}


def provider_harness(provider: Mapping[str, object]) -> str:
    return str(provider.get("harness", "codex"))


def provider_configuration(provider: Mapping[str, object], claim_scope: str, *, arms) -> dict:
    """Validate declared treatment; qualification separately binds actual capability."""
    for name in ("model", "reasoning_effort"):
        value = provider.get(name)
        if not isinstance(value, str) or not value or value != value.strip() or "\x00" in value:
            raise ValueError(f"Study Contract provider configuration {name} differs")
    if (provider.get("cwd_policy") != "independent_task_workspace"
            or provider.get("reference_visibility") != "workspace_task_files"
            or provider.get("removed_environment") != ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]):
        raise ValueError("Study Contract provider configuration workspace/environment differs")
    harness = provider_harness(provider)
    if harness == "claude-code":
        if (set(provider) != _AUTHORITY | _CLAUDE or claim_scope != "artifact_optimization_only"
                or set(arms) != {"open_cake"}
                or provider.get("sandbox") != "none" or provider.get("permission_mode") != "acceptEdits"
                or provider.get("safe_mode") is not True or provider.get("tools") != list(CLAUDE_AUTHORING_TOOLS)
                or provider.get("event_contract") != CLAUDE_EVENT_CONTRACT
                or _canonical_json_bytes(provider.get("terminal_schema")) != _canonical_json_bytes(terminal_schema())
                or provider["reasoning_effort"] not in {"low", "medium", "high", "xhigh", "max"}):
            raise ValueError("Study Contract Claude provider configuration or authoring scope differs")
        return {**{name: provider[name] for name in _CLAUDE}, "submission_contract": CANDIDATE_SET_ENVELOPE_V1}
    if harness != "codex" or frozenset(provider) not in {
        frozenset(_AUTHORITY | _CODEX | {"web_search"}),
        frozenset(_AUTHORITY | _CODEX | {"event_contract"}),
    }:
        raise ValueError("Study Contract provider configuration fields differ")
    defaults = claim_scope == "artifact_optimization_only"
    expected_event = "tool_rich_candidate_v1" if defaults else "closed_file_change_v1"
    if (provider.get("service_tier") != "default" or provider.get("sandbox") != "workspace-write"
            or provider.get("disabled_features") != ([] if defaults else list(CODEX_DISABLED_FEATURES))
            or provider.get("event_contract", "closed_file_change_v1") != expected_event
            or (not defaults and provider.get("web_search") != "disabled")):
        raise ValueError("Study Contract provider configuration differs")
    host = provider.get("code_mode_host")
    if (not isinstance(host, Mapping) or set(host) != {"path", "sha256"}
            or not isinstance(host["path"], str) or not Path(host["path"]).is_absolute()
            or ".." in Path(host["path"]).parts
            or not isinstance(host["sha256"], str) or len(host["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in host["sha256"])):
        raise ValueError("Study Contract Code Mode host identity differs")
    schema = provider.get("output_schema")
    if not isinstance(schema, Mapping) or set(schema) != {"path", "sha256"}:
        raise ValueError("Study Contract provider output_schema reference differs")
    configuration = {name: provider[name] for name in _CODEX - {"output_schema"}}
    configuration["output_schema_sha256"] = schema["sha256"]
    configuration["submission_contract"] = CANDIDATE_SET_ENVELOPE_V1
    for field in ("web_search", "event_contract"):
        if field in provider:
            configuration[field] = provider[field]
    return configuration
