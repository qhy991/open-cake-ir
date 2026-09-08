"""Native Metal host binding at the existing Executor admission boundary.

Inspection creates no library, archive, pipeline, command buffer or GPU dispatch.
The closure binds Python/Swift executables, selected SDK facts, native helper
executables and the exact device/OS identity used by compiled Metal archives.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import tempfile
from types import MappingProxyType
from typing import Mapping

from .executor import _digest, _file_record
from open_cake_ir.evaluation.artifacts import METAL_TARGETS

HOST_FIELDS = {"device_name", "device_registry_id", "operating_system", "target"}


def command_text(arguments: list[str]) -> str:
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=30)
    if result.returncode or not result.stdout.strip():
        raise RuntimeError(f"Metal host observation failed: {arguments!r}; {result.stderr.strip()}")
    return result.stdout.strip()


def observe_sdk() -> dict[str, str]:
    base = ["/usr/bin/xcrun", "--sdk", "macosx"]
    return {"path": str(Path(command_text(base + ["--show-sdk-path"])).resolve(strict=True)),
            "version": command_text(base + ["--show-sdk-version"]),
            "build_version": command_text(base + ["--show-sdk-build-version"])}


def validate_metal_host(host: Mapping[str, object]) -> None:
    if set(host) != {"kind", "python", "packages", "swift", "sdk", "host", "archive_executable", "observer_executable"} or host.get("kind") != "metal":
        raise ValueError("Metal Executor host fields differ")
    for name in ("python", "swift"):
        record = host[name]
        if (not isinstance(record, Mapping) or set(record) != {"invocation_path", "version", "resolved_sha256"}
                or not isinstance(record["invocation_path"], str) or not Path(record["invocation_path"]).is_absolute()
                or not isinstance(record["version"], str) or not record["version"]):
            raise ValueError(f"Metal Executor {name} identity differs")
        _digest(record["resolved_sha256"], f"Metal {name}")
    packages = host["packages"]
    if not isinstance(packages, Mapping) or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in packages.items()):
        raise ValueError("Metal Executor package fields differ")
    sdk = host["sdk"]
    if (not isinstance(sdk, Mapping) or set(sdk) != {"path", "version", "build_version"}
            or any(not isinstance(v, str) or not v for v in sdk.values()) or not Path(sdk["path"]).is_absolute()):
        raise ValueError("Metal Executor SDK fields differ")
    device = host["host"]
    if (not isinstance(device, Mapping) or set(device) != HOST_FIELDS
            or any(not isinstance(v, str) or not v for v in device.values())
            or device["target"] not in METAL_TARGETS or not device["device_registry_id"].isdigit()):
        raise ValueError("Metal Executor device/OS fields differ")
    for name in ("archive_executable", "observer_executable"):
        record = _file_record(host[name], f"Metal {name}")
        if not isinstance(record["path"], str) or not Path(record["path"]).is_absolute():
            raise ValueError(f"Metal {name} path must be absolute")


def admit_metal_executable(record: Mapping[str, object], context: str) -> Mapping[str, object]:
    _file_record(record, context)
    path = Path(str(record["path"]))
    if (not path.is_absolute() or path.is_symlink() or not path.is_file() or not os.access(path, os.X_OK)
            or any((parent / ".git").exists() for parent in path.parents)):
        raise ValueError(f"{context} executable custody differs")
    payload = path.read_bytes()
    if len(payload) != record["size_bytes"] or sha256(payload).hexdigest() != record["sha256"]:
        raise ValueError(f"{context} executable bytes differ")
    return MappingProxyType(dict(record))


def inspect_metal_host(executable: Path, *, target: str, expected_device_names: list[str],
                       directory: Path) -> dict[str, str]:
    """Retain an inspect-only helper request/result in a fresh external directory."""
    if target not in METAL_TARGETS or not expected_device_names:
        raise ValueError("Metal host inspection requires an exact target/device")
    from .metal_build import MetalArchiveHost
    report = MetalArchiveHost(executable, {}).invoke({"action": "inspect", "target": target,
        "expected_device_names": expected_device_names}, directory)
    device = report.get("host")
    profiling = report.get("profiling")
    if (set(report) != {"schema_version", "status", "dispatches", "host", "stage", "profiling"}
            or type(report.get("schema_version")) is not int or report["schema_version"] != 1
            or report.get("status") != "completed" or report.get("stage") != "host_admission"
            or not isinstance(profiling, dict) or set(profiling) != {"compute_stage_sampling", "gpu_timestamp_counter"}
            or any(value is not True for value in profiling.values())
            or type(report.get("dispatches")) is not int or report["dispatches"] != 0
            or not isinstance(device, dict) or set(device) != HOST_FIELDS
            or any(not isinstance(value, str) or not value for value in device.values())
            or device["target"] != target or device["device_name"] not in expected_device_names):
        raise ValueError("Metal host inspection did not produce exact dispatch-free admission")
    return device


def admit_metal_host(host: Mapping[str, object]) -> Mapping[str, object]:
    validate_metal_host(host)
    python = host["python"]
    invocation = Path(sys.executable).absolute()
    if (str(invocation) != python["invocation_path"] or sys.version.split()[0] != python["version"]
            or sha256(invocation.resolve(strict=True).read_bytes()).hexdigest() != python["resolved_sha256"]):
        raise ValueError("Metal Executor Python runtime differs")
    for name, version in host["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Metal Executor package {name!r} differs")
    swift = host["swift"]
    swiftc = Path(swift["invocation_path"])
    if (sha256(swiftc.resolve(strict=True).read_bytes()).hexdigest() != swift["resolved_sha256"]
            or command_text([str(swiftc), "--version"]) != swift["version"]):
        raise ValueError("Metal Executor Swift runtime differs")
    if observe_sdk() != dict(host["sdk"]):
        raise ValueError("Metal Executor selected SDK differs")
    archive = admit_metal_executable(host["archive_executable"], "Metal archive helper")
    observer = admit_metal_executable(host["observer_executable"], "Metal observer")
    expected = dict(host["host"])
    with tempfile.TemporaryDirectory(prefix="metal-host-admission-") as temporary:
        observed = inspect_metal_host(Path(archive["path"]), target=expected["target"],
            expected_device_names=[expected["device_name"]], directory=Path(temporary) / "inspect")
    if observed != expected:
        raise ValueError("Metal Executor exact device/OS admission differs")
    return MappingProxyType({"kind": "metal", "host": observed,
        "observer_executable": str(observer["path"]), "archive_executable": str(archive["path"]),
        "swiftc": str(swiftc), "sdk": dict(host["sdk"])})
