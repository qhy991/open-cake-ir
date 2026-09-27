"""Parse broker runtime configuration and derive its execution binding."""

from __future__ import annotations

from open_cake_ir.serialization import canonical_json_bytes

import json, shlex, shutil
from hashlib import sha256
from pathlib import Path
from typing import Mapping

from ._documents import _name
from .toolchains import toolchain_for


def _runtime_positive_int(value: object, context: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _runtime_command(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        value = shlex.split(value)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("runtime_config.broker.command must be a non-empty command")
    return tuple(_name(item, "runtime_config.broker.command[]") for item in value)


def load_runtime_config(path: str | Path, *, toolchain_kind: str, provider_kind: str = 'codex') -> dict[str, object]:
    """Parse the runtime document without resolving its paths.

    Builder-specific mount/admission policy and raw file identity stay with their
    consumers. In particular, Triton guest aliases must not be resolved here.
    """
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(value, Mapping)
        or set(value) != {"schema_version", "provider", "toolchain", "broker"}
        or type(value["schema_version"]) is not int or value["schema_version"] != 1):
        raise ValueError("runtime configuration fields differ")
    # The field set is the toolchain row's; `toolchain_kind` keeps accepting the spelling
    # the runtime documents under runtime/ and their callers already use.
    toolchain_row = toolchain_for(toolchain_kind)
    if provider_kind not in {'codex','claude-code','responses'}:
        raise ValueError('runtime provider kind is unsupported')
    toolchain_fields = set(toolchain_row.runtime_fields)
    sections = {}
    for name, expected in (
        ("provider", set() if provider_kind=='responses' else {"executable", "workspace_root"}),
        ("toolchain", toolchain_fields),
        ("broker", {"command", "cwd", "timeout_seconds", "service_user", "service_group"}),
    ):
        section = value[name]
        optional = (toolchain_row.optional_runtime_fields if name == 'toolchain' else
                    {'auth_source'} if name == 'provider' and provider_kind == 'codex' else
                    frozenset())
        if not isinstance(section, Mapping) or not expected <= set(section) <= expected | optional:
            raise ValueError(f"runtime_config.{name} fields differ")
        sections[name] = dict(section)
    provider, toolchain, broker = (sections[name] for name in ("provider", "toolchain", "broker"))
    for name, item in provider.items():
        _name(item, f"runtime_config.provider.{name}")
    for name, item in toolchain.items():
        context = f"runtime_config.toolchain.{name}"
        if name == 'pointer_alignment':
            _runtime_positive_int(item, context)
            if item & (item - 1):
                raise ValueError(f'{context} must be a power of two')
        elif name == "timeout_seconds":
            _runtime_positive_int(item, context)
        elif name == "runtime_roots":
            # Emptiness stays the builder's admission decision, not this parser's.
            if not isinstance(item, list):
                raise ValueError(f"{context} must be a list")
            for root in item:
                _name(root, f"{context}[]")
        elif name == "build_environment":
            # A CUDA host declares none, so empty is valid here and not in runtime_roots.
            if not isinstance(item, Mapping):
                raise ValueError(f"{context} must be an object")
            for variable, value in item.items():
                _name(variable, f"{context} name")
                _name(value, f"{context}.{variable}")
        else:
            _name(item, context)
    broker["command"] = _runtime_command(broker["command"])
    _runtime_positive_int(broker["timeout_seconds"], "runtime_config.broker.timeout_seconds")
    for name in ("cwd", "service_user", "service_group"):
        _name(broker[name], f"runtime_config.broker.{name}")
    return {"schema_version": 1, **sections}


def broker_execution_sha256(
    command: tuple[str, ...],
    *,
    cwd: Path,
    project_root: Path,
    timeout_seconds: int,
    service_user: str,
    service_group: str,
) -> str:
    command = _runtime_command(command)
    _runtime_positive_int(timeout_seconds, "runtime_config.broker.timeout_seconds")
    _name(service_user, "runtime_config.broker.service_user")
    _name(service_group, "runtime_config.broker.service_group")
    if cwd.resolve(strict=True) != project_root:
        raise ValueError("broker cwd policy or timeout differs")
    executable = shutil.which(command[0])
    if executable is None:
        raise ValueError(f"external command {command[0]!r} is unavailable")
    files: dict[str, str] = {
        str(Path(executable).resolve(strict=True)): sha256(
            Path(executable).resolve(strict=True).read_bytes()
        ).hexdigest()
    }
    for value in command[1:]:
        path = Path(value)
        if path.is_absolute() and path.is_file() and not path.is_symlink():
            files[str(path.resolve(strict=True))] = sha256(path.read_bytes()).hexdigest()
    return sha256(
        canonical_json_bytes({
                "argv": list(command),
                "files": files,
                "cwd_policy": "project_root",
                "timeout_seconds": timeout_seconds,
                "service_user": service_user,
                "service_group": service_group,
            })
    ).hexdigest()
