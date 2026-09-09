"""Host boundary for Compiler-generated Metal, independent of operator names."""
from __future__ import annotations

import json
import math
from pathlib import Path
import struct
import subprocess
from typing import Mapping

from open_cake_ir.compiler import Assessment, Lowering
from open_cake_ir.compiler.ir import Schedule


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _size(value: object, label: str) -> list[int]:
    _require(isinstance(value, (list, tuple)) and len(value) == 3,
             f"{label} must contain three dimensions")
    _require(all(type(v) is int and 0 < v <= 2**32 - 1 for v in value),
             f"{label} dimensions must be positive uint32 values")
    return list(value)


def manifest(assessment: Assessment, lowering: Lowering, inputs: Mapping[str, bytes],
             directory: Path, *, device_names: list[str]) -> dict[str, object]:
    """Project a canonical assessed Schedule into the Swift buffer/launch boundary.

    No files are created until all inputs and launch commitments have been checked.
    The caller owns a fresh external case directory and the revision-bound device names.
    """
    _require(assessment.accepted and assessment.lowering_eligible,
             "a refused assessment cannot be executed")
    for field in ("compiler_revision_id", "compiler_revision_sha256", "schedule_id",
                  "schedule_sha256", "target", "route"):
        _require(getattr(assessment, field) == getattr(lowering, field),
                 f"lowering {field} differs from the assessment")
    schedule = Schedule.from_dict(json.loads(assessment.schedule_bytes))
    _require(schedule.target == lowering.target and
             schedule.target in {"apple_gpu_family7", "apple_gpu_family8"} and
             schedule.lowering.backend.value == "metal", "exact Metal target required")
    _require(lowering.generated, "Metal requires generated source")
    result = _project(schedule, lowering.source, dict(lowering.toolchain_requirements), inputs,
                      directory, device_names=device_names)
    (directory / "origin.json").write_text(json.dumps({"kind": "compiler_generated",
        "compiler_revision_id": lowering.compiler_revision_id, "schedule_id": lowering.schedule_id}) + "\n")
    return result


def reference_manifest(schedule: Schedule, source: str, toolchain: dict, inputs: Mapping[str, bytes],
                       directory: Path, *, device_names: list[str], provenance: dict) -> dict:
    """Explicit reference boundary; never fabricates a Compiler Assessment or Lowering."""
    _require(provenance.get("kind") == "handwritten_reference" and provenance.get("source_path"),
             "reference source requires explicit handwritten provenance")
    result = _project(schedule, source, toolchain, inputs, directory, device_names=device_names)
    (directory / "origin.json").write_text(json.dumps(provenance) + "\n")
    return result


def _project(schedule: Schedule, source: str, tc: dict, inputs: Mapping[str, bytes],
             directory: Path, *, device_names: list[str]) -> dict:
    _require(isinstance(device_names, list) and device_names and
             all(isinstance(n, str) and n for n in device_names),
             "revision-bound exact device names are required")
    _require((schedule.target, tuple(device_names)) in {
        ("apple_gpu_family7", ("Apple M1 Pro",)),
        ("apple_gpu_family8", ("Apple M2",)),
    }, "exact Metal target/device names differ")
    expected = {"target": schedule.target, "source_language": "metal",
                "compiler": "MTLDevice.makeLibrary", "language_standard": "metal2.3",
                "fast_math_enabled": False}
    _require(set(tc) == set(expected) | {"buffer_order", "threads_per_threadgroup", "threadgroup_memory_bytes",
                                       "threadgroups_per_grid", "execution_model", "active_threads_per_threadgroup"},
             "unsupported or missing Metal toolchain commitment")
    threads = tc["threads_per_threadgroup"]
    _require(isinstance(threads, list) and len(threads) == 3 and threads[1:] == [1, 1]
             and type(threads[0]) is int and threads[0] % 32 == 0 and 32 <= threads[0] <= 1024
             and type(tc["threadgroup_memory_bytes"]) is int
             and 0 <= tc["threadgroup_memory_bytes"] <= 32768
             and (threads[0] == 32) == (tc["threadgroup_memory_bytes"] == 0),
             "unsupported Metal threadgroup shape or storage")
    for name, value in expected.items():
        _require(type(tc[name]) is type(value) and tc[name] == value,
                 f"unsupported Metal {name}")
    _require(type(tc["active_threads_per_threadgroup"]) is int and
             (tc["execution_model"], tc["active_threads_per_threadgroup"]) in
             {("serial_program_tile", 1), ("simd_program_tile", 32)},
             "unsupported Metal execution model/active-lane commitment")
    grid = _size(tc["threadgroups_per_grid"], "threadgroups_per_grid")
    threads = _size(tc["threads_per_threadgroup"], "threads_per_threadgroup")
    _require(threads == [32, 1, 1] and len(schedule.roles) == 1 and
             schedule.roles[0].warps == (0,), "unsupported Metal role/launch mapping")
    if schedule.grid is not None:
        expected_grid = list(schedule.grid)
    else:
        _require(schedule.program_map is not None, "Schedule has no launch grid")
        expected_grid = [1, 1, 1]
        for axis in schedule.program_map.axes:
            buffer = schedule.buffer(axis.buffer)
            _require(buffer is not None and 0 <= axis.axis < 3 and
                     0 <= axis.dimension < len(buffer.shape) and axis.tile == 1,
                     "unsupported Schedule program axis")
            expected_grid[axis.axis] = axis.tile_count(buffer.shape[axis.dimension])
    _require(grid == expected_grid, "emitted grid differs from the Schedule")
    buffers = [b for b in schedule.buffers if b.space.value == "global"]
    _require(tc["buffer_order"] == [b.name for b in buffers],
             "buffer_order differs from canonical global declarations")
    _require(0 < len(buffers) <= 31, "unsupported Metal buffer argument count")
    _require(set(inputs) == {b.name for b in buffers if b.mode.value == "input"},
             "input buffers differ from the Schedule")
    records = []
    for index, buffer in enumerate(buffers):
        _require(buffer.dtype.value == "fp32" and buffer.byte_offset == 0 and
                 buffer.stages == 1 and buffer.allocation is None and
                 buffer.swizzle is None and buffer.scale_of is None and
                 buffer.valid_extent is None and buffer.mode.value in {"input", "output"},
                 f"unsupported buffer declaration: {buffer.name}")
        _require(buffer.size_bytes <= 2**31 - 1, f"buffer too large: {buffer.name}")
        data = inputs.get(buffer.name)
        if data is not None:
            _require(type(data) is bytes and len(data) == buffer.size_bytes,
                     f"FP32 byte length differs: {buffer.name}")
            _require(all(math.isfinite(x[0]) for x in struct.iter_unpack("<f", data)),
                     f"finite FP32 input required: {buffer.name}")
        records.append({"name": buffer.name, "dtype": buffer.dtype.value,
                        "shape": list(buffer.shape), "size_bytes": buffer.size_bytes,
                        "mode": buffer.mode.value,
                        "input_path": str(directory / f"buffer-{index}.input.bin") if data is not None else None,
                        "output_path": str(directory / f"buffer-{index}.result.bin")})
    _require(directory.is_absolute() and directory.is_dir() and not any(directory.iterdir()),
             "case directory must be absolute, existing, and empty")
    result = {**tc, "entry_point": schedule.lowering.entry_point,
              "device_names": device_names, "source_path": str(directory / "kernel.metal"),
              "buffers": records}
    (directory / "kernel.metal").write_text(source, encoding="utf-8")
    for buffer, record in zip(buffers, records):
        if record["input_path"] is not None:
            Path(record["input_path"]).write_bytes(inputs[buffer.name])
    (directory / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def compile_runner(directory: Path) -> Path:
    """Build only the Swift host executable; this does not dispatch GPU work."""
    binary = directory / "metal-runner"
    _require(directory.is_absolute() and directory.is_dir() and not binary.exists(),
             "runner requires a fresh absolute output location")
    command = ["xcrun", "swiftc", "-O", "-target", "arm64-apple-macosx15.0",
               str(Path(__file__).with_name("runner.swift")), "-o", str(binary)]
    (directory / "swift-build-command.json").write_text(json.dumps(command, indent=2) + "\n")
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (directory / "swift-build.log").write_text(completed.stdout + completed.stderr)
    _require(completed.returncode == 0, "Swift host compilation failed; see swift-build.log")
    return binary


def invoke(binary: Path, directory: Path, *, compile_only: bool = False) -> dict[str, object]:
    """Run the generic adapter and retain its completion or refusal verbatim."""
    command = [str(binary), "--compile-only" if compile_only else "--run",
               str(directory / "manifest.json")]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired as error:
        (directory / "runtime-error.txt").write_text("Metal runtime timed out; completion unknown.\n")
        raise RuntimeError("Metal runtime timed out; completion unknown") from error
    (directory / "runtime.stdout.json").write_text(result.stdout)
    (directory / "runtime.stderr.txt").write_text(result.stderr)
    _require(result.returncode == 0, "Metal runtime refused or failed; see runtime.stderr.txt")
    receipt = json.loads(result.stdout)
    _require(receipt.get("status") == ("compiled" if compile_only else "completed"),
             "Metal runtime did not report the expected completion")
    _require(compile_only or receipt.get("command_status") == "completed",
             "Metal command completion is missing")
    return receipt
