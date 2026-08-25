#!/usr/bin/env python3
"""Compile and execute the fixed Apple-family-9 BF16 SIMD-group MMA probe."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Mapping, cast


_PROBE_KIND = "open_cake_metal_toolchain_probe_v1"
_DISPATCH_KIND = "open_cake_metal_bfloat_simdgroup_mma_dispatch_v1"
_ASSET_ROOT = Path(__file__).resolve().parent / "metal_probe"
_SOURCE_NAME = "bfloat_simdgroup_mma.metal"
_RUNNER_NAME = "run_bfloat_simdgroup_mma.swift"
_ASSET_SHA256 = {
    _SOURCE_NAME: "67a31b8fc67fbae58ac623d7d521edf99293fe84f060415de9f8bd2a780b007c",
    _RUNNER_NAME: "b895d7a7fbe3c3908b4806a294d93c0a0af424a2caa0ca6bf012dfc6161aabd0",
}
_MAXIMUM_OUTPUT_BYTES = 1024 * 1024
_MAXIMUM_ARTIFACT_BYTES = 64 * 1024 * 1024
_DIAGNOSTIC_BYTES = 4096


class ProbeFailure(RuntimeError):
    """One fail-closed probe stage failure."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fixed_asset(name: str) -> tuple[Path, bytes]:
    unresolved = _ASSET_ROOT / name
    if unresolved.is_symlink():
        raise ProbeFailure(
            "asset_custody", f"probe asset {name!r} must not be a symlink"
        )
    try:
        path = unresolved.resolve(strict=True)
        root = _ASSET_ROOT.resolve(strict=True)
    except OSError as error:
        raise ProbeFailure(
            "asset_custody", f"probe asset {name!r} is unavailable"
        ) from error
    if path.parent != root or not path.is_file() or path.stat().st_size <= 0:
        raise ProbeFailure("asset_custody", f"probe asset {name!r} custody differs")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ProbeFailure(
            "asset_custody", f"probe asset {name!r} cannot be read"
        ) from error
    expected_sha256 = _ASSET_SHA256.get(name)
    if expected_sha256 is None or sha256(payload).hexdigest() != expected_sha256:
        raise ProbeFailure("asset_custody", f"probe asset {name!r} bytes differ")
    return path, payload


def _resolve_executable(value: str | os.PathLike[str]) -> Path:
    raw = os.fspath(value)
    selected = shutil.which(raw) if not Path(raw).is_absolute() else raw
    if selected is None:
        raise ProbeFailure("host_preflight", f"xcrun executable {raw!r} is unavailable")
    try:
        path = Path(selected).resolve(strict=True)
    except OSError as error:
        raise ProbeFailure(
            "host_preflight", f"xcrun executable {raw!r} is unavailable"
        ) from error
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ProbeFailure("host_preflight", "xcrun path is not an executable file")
    return path


def _diagnostic(payload: bytes) -> str:
    return payload[-_DIAGNOSTIC_BYTES:].decode("utf-8", errors="replace").strip()


def _run(
    arguments: list[str],
    *,
    stage: str,
    cwd: Path,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[bytes]:
    try:
        completed = subprocess.run(
            arguments,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise ProbeFailure(
            stage, f"command timed out after {timeout_seconds} seconds"
        ) from error
    except OSError as error:
        raise ProbeFailure(stage, f"command could not start: {error}") from error
    if (
        len(completed.stdout) > _MAXIMUM_OUTPUT_BYTES
        or len(completed.stderr) > _MAXIMUM_OUTPUT_BYTES
    ):
        raise ProbeFailure(stage, "command output exceeded the retained-output limit")
    if completed.returncode != 0:
        detail = _diagnostic(completed.stderr) or _diagnostic(completed.stdout)
        suffix = f": {detail}" if detail else ""
        raise ProbeFailure(stage, f"command exited {completed.returncode}{suffix}")
    return completed


def _one_line(payload: bytes, *, stage: str, field: str) -> str:
    try:
        text = payload.decode("utf-8")
    except UnicodeError as error:
        raise ProbeFailure(stage, f"{field} is not UTF-8") from error
    values = text.splitlines()
    if len(values) != 1 or not values[0]:
        raise ProbeFailure(stage, f"{field} must be one non-empty line")
    return values[0]


def _tool_record(path_text: str, *, stage: str, name: str) -> dict[str, object]:
    unresolved = Path(path_text)
    if not unresolved.is_absolute():
        raise ProbeFailure(stage, f"xcrun returned a non-absolute {name} path")
    try:
        resolved = unresolved.resolve(strict=True)
    except OSError as error:
        raise ProbeFailure(
            stage, f"xcrun returned an unavailable {name} path"
        ) from error
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ProbeFailure(stage, f"xcrun returned a non-executable {name} path")
    return {
        "path": str(unresolved),
        "resolved_path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _verify_tool_record(record: Mapping[str, object], *, name: str) -> None:
    resolved_path = record.get("resolved_path")
    expected_sha256 = record.get("sha256")
    expected_size = record.get("size_bytes")
    if (
        not isinstance(resolved_path, str)
        or not isinstance(expected_sha256, str)
        or not isinstance(expected_size, int)
    ):
        raise ProbeFailure("tool_custody", f"{name} tool record differs")
    try:
        path = Path(resolved_path).resolve(strict=True)
    except OSError as error:
        raise ProbeFailure("tool_custody", f"{name} tool became unavailable") from error
    if path.stat().st_size != expected_size or _sha256_file(path) != expected_sha256:
        raise ProbeFailure("tool_custody", f"{name} tool bytes changed during probe")


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ProbeFailure("dispatch_output", f"{context} is not an admitted integer")
    return value


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProbeFailure("dispatch_output", f"{context} is not an object")
    return cast(Mapping[str, object], value)


def _validate_dispatch(payload: bytes) -> Mapping[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ProbeFailure(
            "dispatch_output", "Swift runner output is not one JSON document"
        ) from error
    document = _mapping(value, "dispatch observation")
    expected_fields = {
        "schema_version",
        "kind",
        "status",
        "kernel_calls",
        "device",
        "pipeline",
        "dispatch",
    }
    if set(document) != expected_fields:
        raise ProbeFailure("dispatch_output", "Swift runner observation fields differ")
    if (
        document.get("schema_version") != 1
        or document.get("kind") != _DISPATCH_KIND
        or document.get("status") != "passed"
        or _integer(document.get("kernel_calls"), "kernel_calls") != 1
    ):
        raise ProbeFailure("dispatch_output", "Swift runner route contract differs")

    device = _mapping(document.get("device"), "device")
    if set(device) != {
        "name",
        "registry_id",
        "supports_apple9",
        "has_unified_memory",
        "max_threadgroup_memory_length",
    }:
        raise ProbeFailure("dispatch_output", "Swift runner device fields differ")
    if (
        not isinstance(device.get("name"), str)
        or not device.get("name")
        or device.get("supports_apple9") is not True
        or not isinstance(device.get("has_unified_memory"), bool)
        or _integer(device.get("registry_id"), "device.registry_id", minimum=1) <= 0
        or _integer(
            device.get("max_threadgroup_memory_length"),
            "device.max_threadgroup_memory_length",
            minimum=1,
        )
        <= 0
    ):
        raise ProbeFailure("dispatch_output", "Swift runner device observation differs")

    pipeline = _mapping(document.get("pipeline"), "pipeline")
    if set(pipeline) != {
        "entry_point",
        "thread_execution_width",
        "max_total_threads_per_threadgroup",
        "static_threadgroup_memory_length",
    }:
        raise ProbeFailure("dispatch_output", "Swift runner pipeline fields differ")
    if (
        pipeline.get("entry_point") != "open_cake_bfloat_simdgroup_mma_probe"
        or _integer(
            pipeline.get("thread_execution_width"),
            "pipeline.thread_execution_width",
            minimum=1,
        )
        != 32
        or _integer(
            pipeline.get("max_total_threads_per_threadgroup"),
            "pipeline.max_total_threads_per_threadgroup",
            minimum=32,
        )
        < 32
        or _integer(
            pipeline.get("static_threadgroup_memory_length"),
            "pipeline.static_threadgroup_memory_length",
        )
        < 0
    ):
        raise ProbeFailure(
            "dispatch_output", "Swift runner pipeline observation differs"
        )

    dispatch = _mapping(document.get("dispatch"), "dispatch")
    if set(dispatch) != {
        "mma_shape",
        "input_dtype",
        "accumulator_dtype",
        "input_pattern",
        "expected_outputs",
        "observed_outputs",
        "output_element_count",
        "all_outputs_match",
        "threads_dispatched",
        "threadgroups_dispatched",
        "command_buffer_status",
    }:
        raise ProbeFailure("dispatch_output", "Swift runner dispatch fields differ")
    shape = _mapping(dispatch.get("mma_shape"), "dispatch.mma_shape")
    expected_outputs = dispatch.get("expected_outputs")
    observed_outputs = dispatch.get("observed_outputs")
    if (
        not isinstance(expected_outputs, list)
        or not isinstance(observed_outputs, list)
        or len(expected_outputs) != 64
        or len(observed_outputs) != 64
        or any(
            not isinstance(item, (int, float)) or isinstance(item, bool)
            for item in (*expected_outputs, *observed_outputs)
        )
    ):
        raise ProbeFailure("dispatch_output", "Swift runner output vectors differ")
    expected_values = tuple(float(item) for item in expected_outputs)
    observed_values = tuple(float(item) for item in observed_outputs)
    if (
        dict(shape) != {"m": 8, "n": 8, "k": 8}
        or dispatch.get("input_dtype") != "bfloat16"
        or dispatch.get("accumulator_dtype") != "float32"
        or dispatch.get("input_pattern") != "lhs_identity_rhs_row_major_1_to_64"
        or expected_values != tuple(float(index) for index in range(1, 65))
        or observed_values != expected_values
        or _integer(
            dispatch.get("output_element_count"), "dispatch.output_element_count"
        )
        != 64
        or dispatch.get("all_outputs_match") is not True
        or _integer(dispatch.get("threads_dispatched"), "dispatch.threads_dispatched")
        != 32
        or _integer(
            dispatch.get("threadgroups_dispatched"),
            "dispatch.threadgroups_dispatched",
        )
        != 1
        or dispatch.get("command_buffer_status") != "completed"
    ):
        raise ProbeFailure("dispatch_output", "Swift runner result contract differs")
    return document


def probe(
    *,
    xcrun: str | os.PathLike[str],
    timeout_seconds: int = 120,
    _host_system: str | None = None,
) -> dict[str, object]:
    """Run the fixed compile/link/load/dispatch probe once."""

    host_system = platform.system() if _host_system is None else _host_system
    if host_system != "Darwin":
        raise ProbeFailure("host_preflight", "Metal probe requires macOS")
    if (
        not isinstance(timeout_seconds, int)
        or isinstance(timeout_seconds, bool)
        or not 1 <= timeout_seconds <= 600
    ):
        raise ProbeFailure("host_preflight", "timeout_seconds must be in [1, 600]")
    xcrun_path = _resolve_executable(xcrun)
    xcrun_record = _tool_record(str(xcrun_path), stage="host_preflight", name="xcrun")
    source, source_payload = _fixed_asset(_SOURCE_NAME)
    runner, runner_payload = _fixed_asset(_RUNNER_NAME)

    selected_tools: dict[str, dict[str, object]] = {}
    for name in ("metal", "metallib", "swift"):
        observation = _run(
            [str(xcrun_path), "--find", name],
            stage="host_preflight",
            cwd=_ASSET_ROOT,
            timeout_seconds=timeout_seconds,
        )
        selected_tools[name] = _tool_record(
            _one_line(
                observation.stdout,
                stage="host_preflight",
                field=f"{name} path",
            ),
            stage="host_preflight",
            name=name,
        )
    sdk_observation = _run(
        [str(xcrun_path), "--sdk", "macosx", "--show-sdk-path"],
        stage="host_preflight",
        cwd=_ASSET_ROOT,
        timeout_seconds=timeout_seconds,
    )
    sdk_text = _one_line(
        sdk_observation.stdout,
        stage="host_preflight",
        field="macOS SDK path",
    )
    sdk_path = Path(sdk_text)
    if not sdk_path.is_absolute() or not sdk_path.resolve(strict=True).is_dir():
        raise ProbeFailure("host_preflight", "xcrun returned an invalid macOS SDK path")

    result: dict[str, object]
    temporary_path: Path | None = None
    with tempfile.TemporaryDirectory(prefix="open-cake-metal-probe-") as directory:
        temporary_path = Path(directory)
        source_snapshot = temporary_path / _SOURCE_NAME
        runner_snapshot = temporary_path / _RUNNER_NAME
        air_path = temporary_path / "probe.air"
        metallib_path = temporary_path / "probe.metallib"
        source_snapshot.write_bytes(source_payload)
        runner_snapshot.write_bytes(runner_payload)
        _run(
            [
                str(xcrun_path),
                "--sdk",
                "macosx",
                "metal",
                "-std=metal3.2",
                "-Werror",
                "-c",
                str(source_snapshot),
                "-o",
                str(air_path),
            ],
            stage="metal_compile",
            cwd=_ASSET_ROOT,
            timeout_seconds=timeout_seconds,
        )
        if (
            not air_path.is_file()
            or air_path.stat().st_size <= 0
            or air_path.stat().st_size > _MAXIMUM_ARTIFACT_BYTES
        ):
            raise ProbeFailure("metal_compile", "metal produced an invalid AIR file")
        _run(
            [
                str(xcrun_path),
                "--sdk",
                "macosx",
                "metallib",
                str(air_path),
                "-o",
                str(metallib_path),
            ],
            stage="metallib_link",
            cwd=_ASSET_ROOT,
            timeout_seconds=timeout_seconds,
        )
        if (
            not metallib_path.is_file()
            or metallib_path.stat().st_size <= 4
            or metallib_path.stat().st_size > _MAXIMUM_ARTIFACT_BYTES
        ):
            raise ProbeFailure("metallib_link", "metallib produced an invalid library")
        with metallib_path.open("rb") as stream:
            if stream.read(4) != b"MTLB":
                raise ProbeFailure(
                    "metallib_link", "linked library is missing the MTLB header"
                )

        dispatch_process = _run(
            [
                str(xcrun_path),
                "--sdk",
                "macosx",
                "swift",
                str(runner_snapshot),
                str(metallib_path),
            ],
            stage="metal_dispatch",
            cwd=_ASSET_ROOT,
            timeout_seconds=timeout_seconds,
        )
        dispatch = _validate_dispatch(dispatch_process.stdout)
        _verify_tool_record(xcrun_record, name="xcrun")
        for name, record in selected_tools.items():
            _verify_tool_record(record, name=name)
        result = {
            "schema_version": 1,
            "kind": _PROBE_KIND,
            "status": "passed",
            "scientific_claim_authorized": False,
            "performance_measured": False,
            "host": {
                "system": host_system,
                "machine": platform.machine(),
                "macos_version": platform.mac_ver()[0],
            },
            "inputs": {
                "metal_source": {
                    "path": str(source.relative_to(source.parents[2])),
                    "sha256": sha256(source_payload).hexdigest(),
                    "size_bytes": len(source_payload),
                },
                "swift_runner": {
                    "path": str(runner.relative_to(runner.parents[2])),
                    "sha256": sha256(runner_payload).hexdigest(),
                    "size_bytes": len(runner_payload),
                },
            },
            "toolchain": {
                "metal_standard": "metal3.2",
                "sdk": "macosx",
                "sdk_path": str(sdk_path.resolve(strict=True)),
                "xcrun": xcrun_record,
                "selected_tools": selected_tools,
            },
            "artifacts": {
                "air": {
                    "sha256": _sha256_file(air_path),
                    "size_bytes": air_path.stat().st_size,
                },
                "metallib": {
                    "sha256": _sha256_file(metallib_path),
                    "size_bytes": metallib_path.stat().st_size,
                    "magic": "MTLB",
                },
            },
            "dispatch_observation": json.loads(_canonical_json_bytes(dispatch)),
        }
    if temporary_path is None or temporary_path.exists():
        raise ProbeFailure("cleanup", "temporary Metal artifacts were not removed")
    result["temporary_directory_cleaned"] = True
    return result


def _failure_document(error: ProbeFailure) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": _PROBE_KIND,
        "status": "failed",
        "stage": error.stage,
        "error_type": type(error).__name__,
        "message": str(error),
        "scientific_claim_authorized": False,
        "performance_measured": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xcrun",
        default="xcrun",
        help="xcrun executable selected for the fixed Metal/Swift toolchain",
    )
    parser.add_argument("--timeout-seconds", type=int, default=120)
    arguments = parser.parse_args()
    try:
        result = probe(
            xcrun=arguments.xcrun,
            timeout_seconds=arguments.timeout_seconds,
        )
    except ProbeFailure as error:
        sys.stderr.buffer.write(_canonical_json_bytes(_failure_document(error)) + b"\n")
        return 1
    except Exception as error:  # Defensive CLI boundary; never report a partial pass.
        failure = ProbeFailure("internal", f"{type(error).__name__}: {error}")
        sys.stderr.buffer.write(
            _canonical_json_bytes(_failure_document(failure)) + b"\n"
        )
        return 1
    sys.stdout.buffer.write(_canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
