#!/usr/bin/env python3
"""Compile and run a no-timing Apple M4 hierarchical-reduction semantics probe.

This is a local engineering probe, not an Evaluation assay.  It verifies one
finite MSL choreography used by the Stage-3 hardware design: FP32 ``simd_sum``,
lane-zero partial publication through threadgroup memory, a uniform
``threadgroup_barrier(mem_threadgroup)``, one final-reducer owner, scalar
broadcast, and barrier-governed scratch reuse.  It grants no scientific or
performance claim authority and retains no device identifier.  Its inputs are
FP32; BF16 conversion and the Layer-style all-SIMDgroup owner are out of scope.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast


ROOT = Path(__file__).resolve().parents[1]

_PROBE_KIND = "open_cake_metal_hierarchical_reduction_probe_v1"
_DISPATCH_KIND = "open_cake_metal_hierarchical_reduction_dispatch_v1"
_SOURCE = (
    ROOT
    / "tools/metal_hierarchical_reduction_probe/hierarchical_reduction.metal"
)
_RUNNER = (
    ROOT
    / "tools/metal_hierarchical_reduction_probe/run_hierarchical_reduction.swift"
)
_ENTRY_POINT = "hierarchical_reduction_w32_probe"
_GROUP_COUNTS = (1, 2, 4, 8, 16, 32)
_RUNTIME_WIDTH = 32
_EPOCHS = 2
_DYNAMIC_THREADGROUP_MEMORY_BYTES = 144
_INPUT_PATTERNS = (
    "((thread_mod_13)-6)*0.125+0.5",
    "((thread_mod_11)-5)*0.25-0.75",
)
_DIRTY_SENTINELS = (-12345.25, 9876.5)
_TOLERANCE_RULE = "exact_fp32_bits_for_dyadic_fixture"
_NON_CLAIMS = (
    "no_timing_or_performance_claim",
    "no_layer_norm_all_simdgroup_final_reducer_claim",
    "no_bfloat_input_conversion_claim",
    "no_compiler_lowering_or_ir_admission_claim",
)
_COMPILER_FLAGS = (
    "-std=metal3.2",
    "-fmetal-math-mode=safe",
    "-ffp-contract=off",
)
_MAXIMUM_SOURCE_BYTES = 1024 * 1024
_MAXIMUM_ARTIFACT_BYTES = 64 * 1024 * 1024
_MAXIMUM_OUTPUT_BYTES = 4 * 1024 * 1024
_DIAGNOSTIC_BYTES = 4096
_GIT_REVISION = re.compile(r"[0-9a-f]{40}")
_SOURCE_PATHS = (
    "tools/probe_metal_hierarchical_reduction.py",
    "tools/metal_hierarchical_reduction_probe/hierarchical_reduction.metal",
    "tools/metal_hierarchical_reduction_probe/run_hierarchical_reduction.swift",
)


class ProbeFailure(RuntimeError):
    """One fail-closed hierarchical-reduction probe stage failure."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


class _DuplicateKeyError(ValueError):
    pass


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def _reject_non_json_number(value: str) -> object:
    raise ValueError(f"non-JSON number {value}")


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProbeFailure("dispatch_output", f"{context} must be an object")
    return cast(Mapping[str, object], value)


def _exact_mapping(
    value: object,
    context: str,
    keys: frozenset[str],
) -> Mapping[str, object]:
    result = _mapping(value, context)
    missing = sorted(keys - set(result))
    unknown = sorted(set(result) - keys)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        raise ProbeFailure("dispatch_output", f"{context} has {'; '.join(details)}")
    return result


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ProbeFailure(
            "dispatch_output", f"{context} must be an integer >= {minimum}"
        )
    return value


def _number(value: object, context: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ProbeFailure("dispatch_output", f"{context} must be a finite number")
    return float(value)


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProbeFailure("dispatch_output", f"{context} must be a non-empty string")
    return value


def _boolean(value: object, context: str, expected: bool) -> None:
    if value is not expected:
        raise ProbeFailure("dispatch_output", f"{context} must be {expected!r}")


def _sequence(value: object, context: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProbeFailure("dispatch_output", f"{context} must be an array")
    return cast(Sequence[object], value)


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


def _regular_source(path: Path, kind: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ProbeFailure("host_preflight", f"{kind} source is unavailable")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ProbeFailure("host_preflight", f"{kind} source could not be read") from error
    if not payload or len(payload) > _MAXIMUM_SOURCE_BYTES:
        raise ProbeFailure("host_preflight", f"{kind} source has an invalid size")
    return payload


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
    lines = text.splitlines()
    if len(lines) != 1 or not lines[0]:
        raise ProbeFailure(stage, f"{field} must be one non-empty line")
    return lines[0]


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


def _repository_binding(
    *,
    timeout_seconds: int,
    override: Mapping[str, object] | None,
) -> dict[str, object]:
    if override is not None:
        revision = override.get("repository_revision")
        clean = override.get("worktree_clean")
        paths = override.get("source_paths")
        if (
            not isinstance(revision, str)
            or _GIT_REVISION.fullmatch(revision) is None
            or clean is not True
            or paths != list(_SOURCE_PATHS)
        ):
            raise ProbeFailure("source_binding", "test source-binding override is invalid")
        return {
            "repository_revision": revision,
            "worktree_clean": True,
            "source_paths": list(_SOURCE_PATHS),
        }

    revision_process = _run(
        ["git", "-C", str(ROOT), "rev-parse", "--verify", "HEAD"],
        stage="source_binding",
        cwd=ROOT,
        timeout_seconds=timeout_seconds,
    )
    revision = _one_line(
        revision_process.stdout,
        stage="source_binding",
        field="repository revision",
    )
    if _GIT_REVISION.fullmatch(revision) is None:
        raise ProbeFailure("source_binding", "repository revision is not a full Git object ID")
    status_process = _run(
        [
            "git",
            "-C",
            str(ROOT),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        stage="source_binding",
        cwd=ROOT,
        timeout_seconds=timeout_seconds,
    )
    if status_process.stdout:
        raise ProbeFailure(
            "source_binding",
            "repository worktree must be clean before a retained probe run",
        )
    tree_process = _run(
        [
            "git",
            "-C",
            str(ROOT),
            "ls-tree",
            "-r",
            "--name-only",
            "HEAD",
            "--",
            *_SOURCE_PATHS,
        ],
        stage="source_binding",
        cwd=ROOT,
        timeout_seconds=timeout_seconds,
    )
    try:
        tracked_paths = tuple(tree_process.stdout.decode("utf-8").splitlines())
    except UnicodeError as error:
        raise ProbeFailure("source_binding", "Git source closure is not UTF-8") from error
    if tracked_paths != tuple(sorted(_SOURCE_PATHS)):
        raise ProbeFailure(
            "source_binding", "probe source closure is not tracked by the bound revision"
        )
    return {
        "repository_revision": revision,
        "worktree_clean": True,
        "source_paths": list(_SOURCE_PATHS),
    }


def _parse_dispatch(payload: bytes) -> Mapping[str, object]:
    line = _one_line(payload, stage="dispatch_output", field="runner output")
    try:
        value = json.loads(
            line,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_json_number,
        )
    except (
        json.JSONDecodeError,
        _DuplicateKeyError,
        UnicodeError,
        ValueError,
    ) as error:
        raise ProbeFailure("dispatch_output", f"runner output is invalid JSON: {error}") from error
    return _mapping(value, "runner output")


def _validate_epoch(
    value: object,
    *,
    case_index: int,
    epoch_index: int,
    thread_count: int,
) -> tuple[str, float, float]:
    context = f"cases[{case_index}].epochs[{epoch_index}]"
    epoch = _exact_mapping(
        value,
        context,
        frozenset(
            {
                "epoch",
                "input_pattern",
                "dirty_sentinel",
                "observed_dirty_partial_slot",
                "observed_dirty_scalar_slot",
                "dirty_snapshot_max_abs_error",
                "dirty_snapshot_mismatch_count",
                "dirty_sentinel_preserved",
                "expected_sum",
                "observed_owner_sum",
                "owner_abs_error",
                "owner_mismatch_count",
                "broadcast_thread_count",
                "broadcast_mismatch_count",
                "broadcast_max_abs_error",
                "mismatch_count",
                "passed",
            }
        ),
    )
    if _integer(epoch.get("epoch"), f"{context}.epoch") != epoch_index:
        raise ProbeFailure("dispatch_output", f"{context}.epoch is out of order")
    input_pattern = _string(epoch.get("input_pattern"), f"{context}.input_pattern")
    if input_pattern != _INPUT_PATTERNS[epoch_index]:
        raise ProbeFailure("dispatch_output", f"{context} input pattern drifted")
    sentinel = _number(epoch.get("dirty_sentinel"), f"{context}.dirty_sentinel")
    if sentinel != _DIRTY_SENTINELS[epoch_index]:
        raise ProbeFailure("dispatch_output", f"{context} dirty sentinel drifted")
    partial = _number(
        epoch.get("observed_dirty_partial_slot"),
        f"{context}.observed_dirty_partial_slot",
    )
    scalar = _number(
        epoch.get("observed_dirty_scalar_slot"),
        f"{context}.observed_dirty_scalar_slot",
    )
    dirty_error = _number(
        epoch.get("dirty_snapshot_max_abs_error"),
        f"{context}.dirty_snapshot_max_abs_error",
    )
    expected = _number(epoch.get("expected_sum"), f"{context}.expected_sum")
    if epoch_index == 0:
        independently_expected = sum(
            ((index % 13) - 6) * 0.125 + 0.5 for index in range(thread_count)
        )
    else:
        independently_expected = sum(
            ((index % 11) - 5) * 0.25 - 0.75 for index in range(thread_count)
        )
    if expected != independently_expected:
        raise ProbeFailure(
            "dispatch_output",
            f"{context} expected sum differs from the independent probe oracle",
        )
    observed = _number(
        epoch.get("observed_owner_sum"), f"{context}.observed_owner_sum"
    )
    owner_error = _number(epoch.get("owner_abs_error"), f"{context}.owner_abs_error")
    broadcast_error = _number(
        epoch.get("broadcast_max_abs_error"),
        f"{context}.broadcast_max_abs_error",
    )
    for field in (
        "dirty_snapshot_mismatch_count",
        "owner_mismatch_count",
        "broadcast_mismatch_count",
        "mismatch_count",
    ):
        if _integer(epoch.get(field), f"{context}.{field}") != 0:
            raise ProbeFailure(
                "dispatch_output", f"{context} reported a correctness mismatch"
            )
    if _integer(
        epoch.get("broadcast_thread_count"),
        f"{context}.broadcast_thread_count",
        minimum=1,
    ) != thread_count:
        raise ProbeFailure(
            "dispatch_output", f"{context} did not validate every broadcast consumer"
        )
    _boolean(
        epoch.get("dirty_sentinel_preserved"),
        f"{context}.dirty_sentinel_preserved",
        True,
    )
    _boolean(epoch.get("passed"), f"{context}.passed", True)
    if any(error != 0.0 for error in (dirty_error, owner_error, broadcast_error)):
        raise ProbeFailure("dispatch_output", f"{context} reported non-zero error")
    if partial != sentinel or scalar != sentinel:
        raise ProbeFailure(
            "dispatch_output", f"{context} dirty-sentinel snapshot was not preserved"
        )
    if observed != expected:
        raise ProbeFailure(
            "dispatch_output", f"{context} owner result differs from its CPU oracle"
        )
    return input_pattern, sentinel, expected


def _validate_dispatch(payload: bytes) -> Mapping[str, object]:
    dispatch = _exact_mapping(
        _parse_dispatch(payload),
        "runner output",
        frozenset(
            {
                "schema_version",
                "kind",
                "status",
                "kernel_calls",
                "performance_measured",
                "scientific_claim_authorized",
                "probe_scope",
                "non_claims",
                "device",
                "pipeline",
                "coverage",
                "oracle",
                "cases",
            }
        ),
    )
    if dispatch.get("schema_version") != 1 or dispatch.get("kind") != _DISPATCH_KIND:
        raise ProbeFailure("dispatch_output", "runner output has the wrong contract identity")
    if dispatch.get("status") != "passed":
        raise ProbeFailure("dispatch_output", "runner did not report passed status")
    if _integer(dispatch.get("kernel_calls"), "runner output.kernel_calls") != len(
        _GROUP_COUNTS
    ):
        raise ProbeFailure("dispatch_output", "runner kernel-call count drifted")
    _boolean(
        dispatch.get("performance_measured"),
        "runner output.performance_measured",
        False,
    )
    _boolean(
        dispatch.get("scientific_claim_authorized"),
        "runner output.scientific_claim_authorized",
        False,
    )
    if (
        dispatch.get("probe_scope")
        != "shared_choreography_and_rms_single_owner_broadcast_only"
    ):
        raise ProbeFailure("dispatch_output", "runner output widened the probe claim scope")
    non_claims = tuple(
        _string(item, f"runner output.non_claims[{index}]")
        for index, item in enumerate(
            _sequence(dispatch.get("non_claims"), "runner output.non_claims")
        )
    )
    if non_claims != _NON_CLAIMS:
        raise ProbeFailure("dispatch_output", "runner non-claim boundary drifted")

    device = _exact_mapping(
        dispatch.get("device"),
        "runner output.device",
        frozenset(
            {
                "device_class",
                "supports_apple9_or_newer",
                "max_threadgroup_memory_bytes",
            }
        ),
    )
    if device.get("device_class") != "Apple M4":
        raise ProbeFailure("dispatch_output", "probe requires the local Apple M4 device class")
    _boolean(
        device.get("supports_apple9_or_newer"),
        "runner output.device.supports_apple9_or_newer",
        True,
    )
    if _integer(
        device.get("max_threadgroup_memory_bytes"),
        "runner output.device.max_threadgroup_memory_bytes",
        minimum=1,
    ) != 32768:
        raise ProbeFailure(
            "dispatch_output", "local M4 threadgroup-memory observation drifted"
        )

    pipeline = _exact_mapping(
        dispatch.get("pipeline"),
        "runner output.pipeline",
        frozenset(
            {
                "entry_point",
                "thread_execution_width",
                "max_total_threads_per_threadgroup",
                "static_threadgroup_memory_bytes",
                "dynamic_threadgroup_memory_bytes",
            }
        ),
    )
    if pipeline.get("entry_point") != _ENTRY_POINT:
        raise ProbeFailure("dispatch_output", "runner selected the wrong entry point")
    if _integer(
        pipeline.get("thread_execution_width"),
        "runner output.pipeline.thread_execution_width",
        minimum=1,
    ) != _RUNTIME_WIDTH:
        raise ProbeFailure("dispatch_output", "pipeline width is not the guarded width 32")
    if _integer(
        pipeline.get("max_total_threads_per_threadgroup"),
        "runner output.pipeline.max_total_threads_per_threadgroup",
        minimum=1,
    ) < _GROUP_COUNTS[-1] * _RUNTIME_WIDTH:
        raise ProbeFailure("dispatch_output", "pipeline cannot run the 32-SIMDgroup case")
    if _integer(
        pipeline.get("static_threadgroup_memory_bytes"),
        "runner output.pipeline.static_threadgroup_memory_bytes",
    ) != 0:
        raise ProbeFailure("dispatch_output", "unexpected static threadgroup-memory use")
    if _integer(
        pipeline.get("dynamic_threadgroup_memory_bytes"),
        "runner output.pipeline.dynamic_threadgroup_memory_bytes",
        minimum=1,
    ) != _DYNAMIC_THREADGROUP_MEMORY_BYTES:
        raise ProbeFailure("dispatch_output", "dynamic threadgroup-memory layout drifted")

    coverage = _exact_mapping(
        dispatch.get("coverage"),
        "runner output.coverage",
        frozenset(
            {
                "simdgroup_counts",
                "epochs_per_dispatch",
                "single_owner_final_scalar_broadcast",
                "scratch_reused_within_dispatch",
                "uniform_reuse_barrier",
                "layer_all_simdgroups_tested",
            }
        ),
    )
    group_counts = tuple(
        _integer(item, f"runner output.coverage.simdgroup_counts[{index}]", minimum=1)
        for index, item in enumerate(
            _sequence(
                coverage.get("simdgroup_counts"),
                "runner output.coverage.simdgroup_counts",
            )
        )
    )
    if group_counts != _GROUP_COUNTS:
        raise ProbeFailure("dispatch_output", "SIMDgroup coverage matrix drifted")
    if _integer(
        coverage.get("epochs_per_dispatch"),
        "runner output.coverage.epochs_per_dispatch",
        minimum=1,
    ) != _EPOCHS:
        raise ProbeFailure("dispatch_output", "scratch-reuse epoch count drifted")
    for field in (
        "single_owner_final_scalar_broadcast",
        "scratch_reused_within_dispatch",
    ):
        _boolean(coverage.get(field), f"runner output.coverage.{field}", True)
    if coverage.get("uniform_reuse_barrier") != "threadgroup_barrier(mem_threadgroup)":
        raise ProbeFailure("dispatch_output", "scratch reuse barrier contract drifted")
    _boolean(
        coverage.get("layer_all_simdgroups_tested"),
        "runner output.coverage.layer_all_simdgroups_tested",
        False,
    )

    oracle = _exact_mapping(
        dispatch.get("oracle"),
        "runner output.oracle",
        frozenset({"dtype", "tolerance_rule"}),
    )
    if oracle.get("dtype") != "float32":
        raise ProbeFailure("dispatch_output", "runner oracle dtype must be float32")
    if oracle.get("tolerance_rule") != _TOLERANCE_RULE:
        raise ProbeFailure("dispatch_output", "runner oracle tolerance drifted")

    cases = _sequence(dispatch.get("cases"), "runner output.cases")
    if len(cases) != len(_GROUP_COUNTS):
        raise ProbeFailure("dispatch_output", "runner returned the wrong case count")
    for case_index, (value, group_count) in enumerate(zip(cases, _GROUP_COUNTS, strict=True)):
        context = f"runner output.cases[{case_index}]"
        case = _exact_mapping(
            value,
            context,
            frozenset(
                {
                    "group_count",
                    "threads_per_threadgroup",
                    "threadgroup_memory_bytes",
                    "command_buffer_status",
                    "epochs",
                    "passed",
                }
            ),
        )
        thread_count = group_count * _RUNTIME_WIDTH
        if _integer(case.get("group_count"), f"{context}.group_count") != group_count:
            raise ProbeFailure("dispatch_output", f"{context} group count drifted")
        if _integer(
            case.get("threads_per_threadgroup"),
            f"{context}.threads_per_threadgroup",
            minimum=1,
        ) != thread_count:
            raise ProbeFailure("dispatch_output", f"{context} launch width drifted")
        if _integer(
            case.get("threadgroup_memory_bytes"),
            f"{context}.threadgroup_memory_bytes",
            minimum=1,
        ) != _DYNAMIC_THREADGROUP_MEMORY_BYTES:
            raise ProbeFailure("dispatch_output", f"{context} scratch allocation drifted")
        if case.get("command_buffer_status") != "completed":
            raise ProbeFailure("dispatch_output", f"{context} command buffer did not complete")
        _boolean(case.get("passed"), f"{context}.passed", True)
        epochs = _sequence(case.get("epochs"), f"{context}.epochs")
        if len(epochs) != _EPOCHS:
            raise ProbeFailure("dispatch_output", f"{context} epoch count drifted")
        epoch_results = [
            _validate_epoch(
                epoch,
                case_index=case_index,
                epoch_index=epoch_index,
                thread_count=thread_count,
            )
            for epoch_index, epoch in enumerate(epochs)
        ]
        if epoch_results[0][0] == epoch_results[1][0]:
            raise ProbeFailure("dispatch_output", f"{context} reused one input pattern")
        if epoch_results[0][1] == epoch_results[1][1]:
            raise ProbeFailure("dispatch_output", f"{context} reused one dirty sentinel")
        if epoch_results[0][2] == epoch_results[1][2]:
            raise ProbeFailure("dispatch_output", f"{context} epochs have one oracle sum")

    # Canonical re-encoding both rejects non-JSON values and returns a detached,
    # sanitized value rather than the parser's mutable mappings.
    return cast(Mapping[str, object], json.loads(_canonical_json_bytes(dispatch)))


def probe(
    *,
    xcrun: str | os.PathLike[str] = "xcrun",
    timeout_seconds: int = 120,
    _host_system: str | None = None,
    _source_binding_override: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Compile, link, and dispatch the finite semantics matrix exactly once."""

    host_system = platform.system() if _host_system is None else _host_system
    if host_system != "Darwin":
        raise ProbeFailure("host_preflight", "Metal reduction probe requires macOS")
    if (
        not isinstance(timeout_seconds, int)
        or isinstance(timeout_seconds, bool)
        or not 1 <= timeout_seconds <= 600
    ):
        raise ProbeFailure("host_preflight", "timeout_seconds must be in [1, 600]")
    xcrun_path = _resolve_executable(xcrun)
    source_bytes = _regular_source(_SOURCE, "MSL probe")
    runner_bytes = _regular_source(_RUNNER, "Swift runner")
    source_binding = _repository_binding(
        timeout_seconds=timeout_seconds,
        override=_source_binding_override,
    )

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
    try:
        sdk_is_directory = sdk_path.is_absolute() and sdk_path.resolve(strict=True).is_dir()
    except OSError:
        sdk_is_directory = False
    if not sdk_is_directory:
        raise ProbeFailure("host_preflight", "xcrun returned an invalid macOS SDK path")

    result: dict[str, object]
    temporary_path: Path | None = None
    with tempfile.TemporaryDirectory(
        prefix="open-cake-metal-hierarchical-reduction-"
    ) as directory:
        temporary_path = Path(directory)
        source_path = temporary_path / _SOURCE.name
        runner_path = temporary_path / _RUNNER.name
        air_path = temporary_path / "hierarchical_reduction.air"
        metallib_path = temporary_path / "hierarchical_reduction.metallib"
        source_path.write_bytes(source_bytes)
        runner_path.write_bytes(runner_bytes)
        _run(
            [
                str(xcrun_path),
                "--sdk",
                "macosx",
                "metal",
                *_COMPILER_FLAGS,
                "-Werror",
                "-c",
                str(source_path),
                "-o",
                str(air_path),
            ],
            stage="metal_compile",
            cwd=ROOT,
            timeout_seconds=timeout_seconds,
        )
        _artifact(air_path, stage="metal_compile", kind="AIR")
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
            cwd=ROOT,
            timeout_seconds=timeout_seconds,
        )
        _artifact(
            metallib_path,
            stage="metallib_link",
            kind="metallib",
            magic=b"MTLB",
        )
        dispatch_process = _run(
            [
                str(xcrun_path),
                "--sdk",
                "macosx",
                "swift",
                "-warnings-as-errors",
                str(runner_path),
                str(metallib_path),
            ],
            stage="metal_dispatch",
            cwd=ROOT,
            timeout_seconds=timeout_seconds,
        )
        dispatch = _validate_dispatch(dispatch_process.stdout)
        result = {
            "schema_version": 1,
            "kind": _PROBE_KIND,
            "status": "passed",
            "claim_scope": "local_engineering_observation_only",
            "evaluation_evidence": False,
            "external_workload_oracle_used": False,
            "scientific_claim_authorized": False,
            "performance_measured": False,
            "compiler_change_authorized": False,
            "human_hardware_review_cleared": False,
            "promotion_authorized": False,
            "stable_device_identifiers_retained": False,
            "host": {"system": host_system, "machine": platform.machine()},
            "source_binding": source_binding,
            "probe_contract": {
                "source_path": _SOURCE.relative_to(ROOT).as_posix(),
                "runner_path": _RUNNER.relative_to(ROOT).as_posix(),
                "simdgroup_counts": list(_GROUP_COUNTS),
                "epochs_per_dispatch": _EPOCHS,
            },
            "toolchain": {
                "sdk": "macosx",
                "language_standard": "metal3.2",
                "compiler_flags": list(_COMPILER_FLAGS),
                "swift_flags": ["-warnings-as-errors"],
                "compiled_air_count": 1,
                "linked_metallib": True,
            },
            "dispatch_observation": dispatch,
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
        "claim_scope": "local_engineering_observation_only",
        "evaluation_evidence": False,
        "external_workload_oracle_used": False,
        "scientific_claim_authorized": False,
        "performance_measured": False,
        "compiler_change_authorized": False,
        "human_hardware_review_cleared": False,
        "promotion_authorized": False,
        "stable_device_identifiers_retained": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xcrun", default="xcrun")
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
