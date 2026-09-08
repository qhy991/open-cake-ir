"""Parse broker runtime configuration and derive its execution binding."""

from __future__ import annotations

import json, shlex, shutil
from hashlib import sha256
from pathlib import Path
from typing import Mapping

from ._documents import _canonical_json_bytes


def _runtime_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _runtime_positive_int(value: object, context: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _runtime_command(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        value = shlex.split(value)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("runtime_config.broker.command must be a non-empty command")
    return tuple(_runtime_string(item, "runtime_config.broker.command[]") for item in value)


def load_runtime_config(path: str | Path, *, toolchain_kind: str) -> dict[str, object]:
    """Parse the current matched runtime document without resolving its paths.

    Builder-specific mount/admission policy and raw file identity stay with their
    consumers. In particular, Triton guest aliases must not be resolved here.
    """
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(value, Mapping)
        or set(value) != {"schema_version", "provider", "toolchain", "broker"}
        or type(value["schema_version"]) is not int or value["schema_version"] != 1):
        raise ValueError("runtime configuration fields differ")
    if toolchain_kind == "triton":
        toolchain_fields = {"python", "bubblewrap", "runtime_roots", "triton_version", "timeout_seconds"}
    elif toolchain_kind == "cutlass_cute_dsl":
        toolchain_fields = {"python", "bubblewrap", "runtime_roots", "cuobjdump", "cutlass_version", "timeout_seconds"}
    elif toolchain_kind == "nvcc":
        toolchain_fields = {"nvcc", "cuobjdump"}
    else:
        raise ValueError("runtime toolchain kind is unsupported")
    sections = {}
    for name, expected in (
        ("provider", {"executable", "workspace_root"}),
        ("toolchain", toolchain_fields),
        ("broker", {"command", "cwd", "timeout_seconds", "service_user", "service_group"}),
    ):
        section = value[name]
        if not isinstance(section, Mapping) or set(section) != expected:
            raise ValueError(f"runtime_config.{name} fields differ")
        sections[name] = dict(section)
    provider, toolchain, broker = (sections[name] for name in ("provider", "toolchain", "broker"))
    for name, item in provider.items():
        _runtime_string(item, f"runtime_config.provider.{name}")
    for name, item in toolchain.items():
        context = f"runtime_config.toolchain.{name}"
        if name == "timeout_seconds":
            _runtime_positive_int(item, context)
        elif name == "runtime_roots":
            if not isinstance(item, list):
                raise ValueError(f"{context} must be a list")
            for root in item:
                _runtime_string(root, f"{context}[]")
        else:
            _runtime_string(item, context)
    broker["command"] = _runtime_command(broker["command"])
    _runtime_positive_int(broker["timeout_seconds"], "runtime_config.broker.timeout_seconds")
    for name in ("cwd", "service_user", "service_group"):
        _runtime_string(broker[name], f"runtime_config.broker.{name}")
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
    _runtime_string(service_user, "runtime_config.broker.service_user")
    _runtime_string(service_group, "runtime_config.broker.service_group")
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
        json.dumps(
            {
                "argv": list(command),
                "files": files,
                "cwd_policy": "project_root",
                "timeout_seconds": timeout_seconds,
                "service_user": service_user,
                "service_group": service_group,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
