#!/usr/bin/env python3
"""Lower, compile, and correctness-smoke two finite Metal operator graphs.

This is an engineering probe, not an Evaluation assay.  It takes no timings and grants
no scientific claim authority.  Launch geometry and buffer order come from each live
Compiler lowering; the Swift runner receives that materialized launch contract rather
than reproducing it from constants in this tool.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler.core import Compiler, Lowering  # noqa: E402


_PROBE_KIND = "open_cake_metal_operator_probe_v1"
_DISPATCH_KIND = "open_cake_metal_operator_dispatch_v1"
_RUNNER = ROOT / "tools/metal_operator_probe/run_metal_operators.swift"
_OPERATORS = (
    (
        "indexed_gather",
        "corpus/schedules/indexed-gather-b8-metal-family9.json",
        ("expert_rows", "expert_ids", "row_ids", "gathered_rows"),
        8 * 8 * 16,
    ),
    (
        "weighted_combine",
        "corpus/schedules/kda-weighted-combine-b8-metal-family9.json",
        ("expert_rows", "expert_ids", "row_ids", "route_weights", "output"),
        8 * 16,
    ),
)
_MAXIMUM_OUTPUT_BYTES = 4 * 1024 * 1024
_MAXIMUM_ARTIFACT_BYTES = 64 * 1024 * 1024
_DIAGNOSTIC_BYTES = 4096


class ProbeFailure(RuntimeError):
    """One fail-closed operator-probe stage failure."""

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


def _mapping(
    value: object,
    context: str,
    *,
    stage: str = "dispatch_output",
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProbeFailure(stage, f"{context} must be an object")
    return cast(Mapping[str, object], value)


def _strings(value: object, context: str, *, stage: str) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ProbeFailure(stage, f"{context} must be a list of non-empty strings")
    return tuple(cast(Sequence[str], value))


def _integer(
    value: object,
    context: str,
    *,
    minimum: int = 0,
    stage: str = "dispatch_output",
) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ProbeFailure(stage, f"{context} is not an admitted integer")
    return value


def _dimensions(value: object, context: str, *, stage: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ProbeFailure(stage, f"{context} must contain three dimensions")
    return cast(
        tuple[int, int, int],
        tuple(
            _integer(item, f"{context}[{index}]", minimum=1, stage=stage)
            for index, item in enumerate(value)
        ),
    )


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


def _artifact(path: Path, *, stage: str, kind: str, magic: bytes | None = None) -> None:
    if (
        not path.is_file()
        or path.stat().st_size <= 0
        or path.stat().st_size > _MAXIMUM_ARTIFACT_BYTES
    ):
        raise ProbeFailure(stage, f"toolchain produced an invalid {kind} artifact")
    if magic is not None:
        with path.open("rb") as stream:
            if stream.read(len(magic)) != magic:
                raise ProbeFailure(stage, f"{kind} artifact has an invalid header")


def _lowerings(revision: str) -> tuple[tuple[str, str, Lowering], ...]:
    try:
        compiler = Compiler.load(ROOT, revision)
    except Exception as error:
        raise ProbeFailure(
            "compiler", f"Compiler Revision could not be loaded: {error}"
        ) from error
    results: list[tuple[str, str, Lowering]] = []
    revision_identity: tuple[str, str] | None = None
    for kind, relative_path, expected_buffers, _ in _OPERATORS:
        path = ROOT / relative_path
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            assessment = compiler.assess(
                _mapping(document, relative_path, stage="compiler")
            )
        except Exception as error:
            raise ProbeFailure(
                "compiler", f"{relative_path} assessment failed: {error}"
            ) from error
        if not assessment.accepted or not assessment.lowering_eligible:
            codes = ", ".join(finding.code for finding in assessment.findings) or "none"
            raise ProbeFailure(
                "compiler", f"{relative_path} is not accepted and lowerable: {codes}"
            )
        try:
            lowering = compiler.lower(assessment)
        except Exception as error:
            raise ProbeFailure(
                "compiler", f"{relative_path} lowering failed: {error}"
            ) from error
        identity = (lowering.compiler_revision_id, lowering.compiler_revision_sha256)
        if revision_identity is None:
            revision_identity = identity
        elif identity != revision_identity:
            raise ProbeFailure(
                "compiler", "operator lowerings came from different Revisions"
            )
        requirements = _mapping(
            lowering.toolchain_requirements,
            f"{kind} toolchain requirements",
            stage="compiler",
        )
        if (
            not lowering.generated
            or lowering.route.backend.value != "metal"
            or requirements.get("source_language") != "metal"
            or requirements.get("compiler") != "metal"
        ):
            raise ProbeFailure(
                "compiler", f"{kind} did not produce a generated Metal lowering"
            )
        buffers = _strings(
            requirements.get("buffer_order"),
            f"{kind}.buffer_order",
            stage="compiler",
        )
        if buffers != expected_buffers:
            raise ProbeFailure(
                "compiler", f"{kind} lowering ABI differs from the smoke runner"
            )
        _dimensions(
            requirements.get("threadgroups_per_grid"),
            f"{kind}.threadgroups_per_grid",
            stage="compiler",
        )
        _dimensions(
            requirements.get("threads_per_threadgroup"),
            f"{kind}.threads_per_threadgroup",
            stage="compiler",
        )
        _integer(
            requirements.get("threadgroup_memory_bytes"),
            f"{kind}.threadgroup_memory_bytes",
            stage="compiler",
        )
        _strings(
            requirements.get("compiler_flags"),
            f"{kind}.compiler_flags",
            stage="compiler",
        )
        results.append((kind, relative_path, lowering))
    return tuple(results)


def _launch_document(
    lowerings: tuple[tuple[str, str, Lowering], ...],
) -> dict[str, object]:
    operators: list[dict[str, object]] = []
    for kind, _, lowering in lowerings:
        requirements = lowering.toolchain_requirements
        operators.append(
            {
                "kind": kind,
                "entry_point": lowering.route.entry_point,
                "buffer_order": list(cast(Sequence[str], requirements["buffer_order"])),
                "threadgroups_per_grid": list(
                    cast(Sequence[int], requirements["threadgroups_per_grid"])
                ),
                "threads_per_threadgroup": list(
                    cast(Sequence[int], requirements["threads_per_threadgroup"])
                ),
                "threadgroup_memory_bytes": cast(
                    int, requirements["threadgroup_memory_bytes"]
                ),
            }
        )
    return {"schema_version": 1, "operators": operators}


def _bits(value: object, count: int, context: str) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != count:
        raise ProbeFailure("dispatch_output", f"{context} has the wrong element count")
    return tuple(
        _integer(item, f"{context}[{index}]") for index, item in enumerate(value)
    )


def _validate_dispatch(
    payload: bytes,
    launches: Mapping[str, object],
) -> Mapping[str, object]:
    try:
        document = _mapping(json.loads(payload), "dispatch observation")
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ProbeFailure(
            "dispatch_output", "Swift runner output is not one JSON document"
        ) from error
    if (
        document.get("schema_version") != 1
        or document.get("kind") != _DISPATCH_KIND
        or document.get("status") != "passed"
        or _integer(document.get("kernel_calls"), "kernel_calls") != 2
    ):
        raise ProbeFailure("dispatch_output", "Swift runner route contract differs")
    device = _mapping(document.get("device"), "device")
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
    coverage = _mapping(document.get("input_coverage"), "input_coverage")
    expected_coverage = {
        "valid_pairs": 32,
        "negative_expert_ids": 8,
        "upper_bound_expert_ids": 8,
        "negative_row_ids": 8,
        "upper_bound_row_ids": 8,
    }
    if dict(coverage) != expected_coverage:
        raise ProbeFailure(
            "dispatch_output", "Swift runner edge-input coverage differs"
        )

    requested = launches.get("operators")
    observed = document.get("operators")
    if (
        not isinstance(requested, list)
        or not isinstance(observed, list)
        or len(observed) != 2
    ):
        raise ProbeFailure(
            "dispatch_output", "Swift runner operator observations differ"
        )
    for index, ((kind, _, _, count), requested_value, observed_value) in enumerate(
        zip(_OPERATORS, requested, observed, strict=True)
    ):
        request = _mapping(requested_value, f"launch.operators[{index}]")
        operator = _mapping(observed_value, f"operators[{index}]")
        if (
            operator.get("kind") != kind
            or operator.get("entry_point") != request.get("entry_point")
            or _integer(operator.get("kernel_calls"), f"{kind}.kernel_calls") != 1
        ):
            raise ProbeFailure("dispatch_output", f"{kind} dispatch identity differs")
        pipeline = _mapping(operator.get("pipeline"), f"{kind}.pipeline")
        threads = _dimensions(
            pipeline.get("threads_per_threadgroup"),
            f"{kind}.pipeline.threads_per_threadgroup",
            stage="dispatch_output",
        )
        groups = _dimensions(
            pipeline.get("threadgroups_per_grid"),
            f"{kind}.pipeline.threadgroups_per_grid",
            stage="dispatch_output",
        )
        width = _integer(
            pipeline.get("thread_execution_width"),
            f"{kind}.pipeline.thread_execution_width",
            minimum=1,
        )
        maximum = _integer(
            pipeline.get("max_total_threads_per_threadgroup"),
            f"{kind}.pipeline.max_total_threads_per_threadgroup",
            minimum=1,
        )
        if (
            pipeline.get("entry_point") != request.get("entry_point")
            or list(threads) != request.get("threads_per_threadgroup")
            or list(groups) != request.get("threadgroups_per_grid")
            or pipeline.get("threadgroup_memory_bytes")
            != request.get("threadgroup_memory_bytes")
            or width != 32
            or threads[0] * threads[1] * threads[2] > maximum
            or (threads[0] * threads[1] * threads[2]) % width != 0
        ):
            raise ProbeFailure(
                "dispatch_output", f"{kind} pipeline/launch observation differs"
            )
        result = _mapping(operator.get("result"), f"{kind}.result")
        expected_bits = _bits(
            result.get("expected_bf16_bits"), count, f"{kind}.expected"
        )
        observed_bits = _bits(
            result.get("observed_bf16_bits"), count, f"{kind}.observed"
        )
        if (
            any(value > 0xFFFF for value in (*expected_bits, *observed_bits))
            or expected_bits != observed_bits
            or _integer(
                result.get("output_element_count"), f"{kind}.output_element_count"
            )
            != count
            or _integer(result.get("mismatch_count"), f"{kind}.mismatch_count") != 0
            or result.get("all_bits_match") is not True
            or result.get("command_buffer_status") != "completed"
        ):
            raise ProbeFailure(
                "dispatch_output", f"{kind} correctness observation differs"
            )
        if kind == "weighted_combine" and (
            result.get("cpu_oracle_accumulation_order") != "route_0_through_7_fp32"
            or result.get("output_conversion")
            != "bfloat16_round_to_nearest_ties_to_even"
            or result.get("bf16_exact_halfway_cases") != 2
            or result.get("bf16_exact_halfway_even_lsb_cases") != 1
            or result.get("bf16_exact_halfway_odd_lsb_cases") != 1
        ):
            raise ProbeFailure(
                "dispatch_output", "weighted_combine oracle contract differs"
            )
    return document


def probe(
    *,
    xcrun: str | os.PathLike[str] = "xcrun",
    revision: str = "compiler/revision.json",
    timeout_seconds: int = 120,
    _host_system: str | None = None,
) -> dict[str, object]:
    """Lower, compile, link, and dispatch both operator graphs exactly once."""

    host_system = platform.system() if _host_system is None else _host_system
    if host_system != "Darwin":
        raise ProbeFailure("host_preflight", "Metal operator probe requires macOS")
    if (
        not isinstance(timeout_seconds, int)
        or isinstance(timeout_seconds, bool)
        or not 1 <= timeout_seconds <= 600
    ):
        raise ProbeFailure("host_preflight", "timeout_seconds must be in [1, 600]")
    xcrun_path = _resolve_executable(xcrun)
    if not _RUNNER.is_file() or _RUNNER.is_symlink():
        raise ProbeFailure("host_preflight", "Swift operator runner is unavailable")
    lowerings = _lowerings(revision)
    launch_document = _launch_document(lowerings)
    standards = {
        lowering.toolchain_requirements.get("language_standard")
        for _, _, lowering in lowerings
    }
    if len(standards) != 1 or not all(
        isinstance(item, str) and item for item in standards
    ):
        raise ProbeFailure("compiler", "Metal lowerings disagree on language standard")
    language_standard = cast(str, next(iter(standards)))

    sdk_process = _run(
        [str(xcrun_path), "--sdk", "macosx", "--show-sdk-path"],
        stage="host_preflight",
        cwd=ROOT,
        timeout_seconds=timeout_seconds,
    )
    sdk_text = _one_line(
        sdk_process.stdout, stage="host_preflight", field="macOS SDK path"
    )
    sdk_path = Path(sdk_text)
    if not sdk_path.is_absolute() or not sdk_path.resolve(strict=True).is_dir():
        raise ProbeFailure("host_preflight", "xcrun returned an invalid macOS SDK path")

    result: dict[str, object]
    temporary_path: Path | None = None
    with tempfile.TemporaryDirectory(prefix="open-cake-metal-operators-") as directory:
        temporary_path = Path(directory)
        air_paths: list[Path] = []
        for index, (kind, _, lowering) in enumerate(lowerings):
            source_path = temporary_path / f"{index}-{kind}.metal"
            air_path = temporary_path / f"{index}-{kind}.air"
            source_path.write_text(lowering.source, encoding="utf-8")
            flags = list(
                _strings(
                    lowering.toolchain_requirements.get("compiler_flags"),
                    f"{kind}.compiler_flags",
                    stage="compiler",
                )
            )
            _run(
                [
                    str(xcrun_path),
                    "--sdk",
                    "macosx",
                    cast(str, lowering.toolchain_requirements["compiler"]),
                    *flags,
                    "-Werror",
                    "-c",
                    str(source_path),
                    "-o",
                    str(air_path),
                ],
                stage=f"metal_compile_{kind}",
                cwd=ROOT,
                timeout_seconds=timeout_seconds,
            )
            _artifact(air_path, stage=f"metal_compile_{kind}", kind="AIR")
            air_paths.append(air_path)

        metallib_path = temporary_path / "operators.metallib"
        _run(
            [
                str(xcrun_path),
                "--sdk",
                "macosx",
                "metallib",
                *(str(path) for path in air_paths),
                "-o",
                str(metallib_path),
            ],
            stage="metallib_link",
            cwd=ROOT,
            timeout_seconds=timeout_seconds,
        )
        _artifact(metallib_path, stage="metallib_link", kind="metallib", magic=b"MTLB")
        launch_path = temporary_path / "lowering-launch.json"
        launch_path.write_bytes(_canonical_json_bytes(launch_document) + b"\n")
        runner_path = temporary_path / _RUNNER.name
        runner_path.write_bytes(_RUNNER.read_bytes())
        dispatch_process = _run(
            [
                str(xcrun_path),
                "--sdk",
                "macosx",
                "swift",
                str(runner_path),
                str(metallib_path),
                str(launch_path),
            ],
            stage="metal_dispatch",
            cwd=ROOT,
            timeout_seconds=timeout_seconds,
        )
        dispatch = _validate_dispatch(dispatch_process.stdout, launch_document)
        first_lowering = lowerings[0][2]
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
            "compiler_revision": {
                "path": revision,
                "revision_id": first_lowering.compiler_revision_id,
            },
            "lowerings": [
                {
                    "kind": kind,
                    "schedule_path": relative_path,
                    "schedule_id": lowering.schedule_id,
                    "target": lowering.target,
                    "backend": lowering.route.backend.value,
                    "entry_point": lowering.route.entry_point,
                    "generated": lowering.generated,
                    "launch": launch,
                }
                for (kind, relative_path, lowering), launch in zip(
                    lowerings,
                    cast(list[dict[str, object]], launch_document["operators"]),
                    strict=True,
                )
            ],
            "toolchain": {
                "sdk": "macosx",
                "sdk_path": str(sdk_path.resolve(strict=True)),
                "language_standard": language_standard,
                "compiled_air_count": len(air_paths),
                "linked_metallib": True,
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
    parser.add_argument("--xcrun", default="xcrun")
    parser.add_argument("--revision", default="compiler/revision.json")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    arguments = parser.parse_args()
    try:
        result = probe(
            xcrun=arguments.xcrun,
            revision=arguments.revision,
            timeout_seconds=arguments.timeout_seconds,
        )
    except ProbeFailure as error:
        sys.stderr.buffer.write(_canonical_json_bytes(_failure_document(error)) + b"\n")
        return 1
    except Exception as error:  # Defensive CLI boundary: never report a partial pass.
        failure = ProbeFailure("internal", f"{type(error).__name__}: {error}")
        sys.stderr.buffer.write(
            _canonical_json_bytes(_failure_document(failure)) + b"\n"
        )
        return 1
    sys.stdout.buffer.write(_canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
