"""In-process MLX host for the bounded exact-target Apple Metal correctness harness.

Reads the manifest and buffer files `tools/metal/adapter.py` writes and runs the same
emitted source, but the library is JIT compiled by `mlx.core.fast.metal_kernel` instead
of the Swift runner's `MTLDevice.makeLibrary`. MLX owns the generated kernel signature,
so the source is bridged rather than handed over: its file-scope preamble becomes the
MLX header, its body is kept verbatim, and one alias statement per position attribute
restores the emitter's own parameter names.

This is a second host for one Compiler output. It is not a lowering route, not the
sealed binary-archive Executor path, and it seals no LaunchableCandidate. MLX accepts
one of the three MTLCompileOptions the Swift host sets and exposes no
MTLComputePipelineState, so every receipt names what this host could not set or check.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import struct
import time
from typing import Mapping, Sequence

from tools.metal.adapter import EXACT_DEVICE_NAMES

# runner.swift sets languageVersion, mathMode and mathFloatingPointFunctions on
# MTLCompileOptions. mx.fast.metal_kernel accepts math_mode alone, so the other two
# stay at MLX's own defaults. The emitted body already spells `precise::` on every
# transcendental it uses, which is an argument for the residual risk being small --
# not a measurement of it, which is why both names are reported rather than dropped.
COMPILE_OPTIONS = {"math_mode": "safe"}
UNSET_COMPILE_OPTIONS = ("language_standard", "math_floating_point_functions")
# runner.swift refuses a pipeline whose execution width, static threadgroup memory or
# maximum thread count differ from the manifest. MLX returns arrays, never the pipeline.
UNCHECKED_PIPELINE_COMMITMENTS = ("thread_execution_width", "static_threadgroup_memory_length",
                                  "max_total_threads_per_threadgroup")
# The Swift host fills every output with the NaN payload so a missing store cannot
# compare equal to zero. MLX expresses the same intent through init_value.
OUTPUT_FILL = float("nan")

_ENTRY = re.compile(r"^kernel void (\w+)\($")
_BUFFER = re.compile(r"^ {4}device (const )?float\* (\w+) \[\[buffer\((\d+)\)\]\],$")
_ATTRIBUTE = re.compile(r"^ {4}(uint3|uint) (\w+) \[\[(\w+)\]\](,|\) \{)$")
_BODY_END = "    // CAKE_KERNEL_END"


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Bridge:
    """One emitted Metal kernel expressed as the arguments mx.fast.metal_kernel takes."""

    header: str
    body: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    grid: tuple[int, int, int]
    threadgroup: tuple[int, int, int]
    buffer_indices_match_emission: bool


def bridge(source: str, *, entry_point: str, threadgroups_per_grid: Sequence[int],
           threads_per_threadgroup: Sequence[int]) -> Bridge:
    """Split emitted source into the header, body and launch MLX generates a kernel from.

    Nothing in the body is rewritten. MLX adds a position-attribute parameter when the
    attribute's own name appears in the source, so an alias statement per attribute both
    requests the parameter and restores the name the emitter used below it.
    """
    lines = source.splitlines()
    for index, line in enumerate(lines):
        entry = _ENTRY.match(line)
        if entry:
            break
    else:
        raise ValueError("emitted Metal source declares no kernel to bridge")
    _require(entry.group(1) == entry_point, "emitted entry point differs from the manifest")
    # MLX concatenates the header directly ahead of the signature it generates, and the
    # emitted preamble ends in `//` comment lines. Without this newline the first
    # generated line is commented out and the kernel declaration disappears.
    header, index = "\n".join(lines[:index]) + "\n", index + 1
    buffers: list[tuple[str, str]] = []
    aliases: list[str] = []
    while True:
        _require(index < len(lines), "emitted kernel signature is unterminated")
        line, index = lines[index], index + 1
        parameter = _BUFFER.match(line)
        if parameter:
            const, name, slot = parameter.groups()
            _require(int(slot) == len(buffers), "emitted buffer ids are not declaration ordered")
            buffers.append((name, "input" if const else "output"))
            continue
        parameter = _ATTRIBUTE.match(line)
        _require(parameter, f"unsupported emitted kernel parameter: {line.strip()}")
        kind, name, attribute, terminator = parameter.groups()
        if name != attribute:
            aliases.append(f"    {kind} {name} = {attribute};")
        if terminator != ",":
            break
    body = lines[index:]
    while body and not body[-1].strip():
        body.pop()
    _require(body and body[-1] == "}" and _BODY_END in body, "emitted kernel body is truncated")
    body.pop()
    modes = [mode for _, mode in buffers]
    inputs = tuple(name for name, mode in buffers if mode == "input")
    outputs = tuple(name for name, mode in buffers if mode == "output")
    _require(inputs and outputs, "a bridged kernel needs at least one input and one output")
    threads, grid = tuple(threads_per_threadgroup), tuple(threadgroups_per_grid)
    # The Swift host calls dispatchThreadgroups: its grid counts threadgroups. MLX calls
    # dispatchThreads: its grid counts threads and is divided by the threadgroup size.
    # Multiplying the first axis keeps threadgroup_position_in_grid equal to `program`.
    _require(threads == (32, 1, 1), "the MLX host bridges one SIMD group per threadgroup only")
    return Bridge(header=header, body="\n".join(aliases + body), input_names=inputs,
                  output_names=outputs, grid=(grid[0] * threads[0], grid[1], grid[2]),
                  threadgroup=threads,
                  buffer_indices_match_emission=modes == ["input"] * len(inputs) + ["output"] * len(outputs))


def compile_runner(directory: Path) -> dict[str, object]:
    """Record the in-process host instead of building one; MLX compiles per kernel."""
    import mlx.core as mx

    _require(directory.is_absolute() and directory.is_dir(),
             "the MLX host requires an absolute existing receipt directory")
    info = mx.device_info()
    host = {"host": "mlx", "mlx_version": mx.__version__, "compiler": "mx.fast.metal_kernel",
            "device_name": info["device_name"], "architecture": info["architecture"],
            "compile_options": dict(COMPILE_OPTIONS),
            "unset_compile_options": list(UNSET_COMPILE_OPTIONS),
            "unchecked_pipeline_commitments": list(UNCHECKED_PIPELINE_COMMITMENTS)}
    (directory / "mlx-host.json").write_text(json.dumps(host, indent=2) + "\n")
    return host


def invoke(host: Mapping[str, object], directory: Path, *, compile_only: bool = False) -> dict[str, object]:
    """Run one prepared case through MLX and retain the bridged source and receipt."""
    import mlx.core as mx

    # MLX compiles at the first dispatch of a kernel, so there is no compiled-but-not-run
    # state for this host to report. Refusing is the truthful answer.
    _require(not compile_only, "the MLX host cannot compile without dispatching")
    manifest = json.loads((directory / "manifest.json").read_text())
    _require(manifest["source_language"] == "metal" and manifest["compiler"] == "MTLDevice.makeLibrary",
             "unsupported exact Metal route")
    _require(manifest["fast_math_enabled"] is False, "the MLX host requires fast math disabled")
    _require(EXACT_DEVICE_NAMES.get(manifest["target"]) == tuple(manifest["device_names"]),
             "exact Metal target/device names differ")
    _require(host["device_name"] in manifest["device_names"],
             f"live MLX device {host['device_name']!r} is not an exact {manifest['target']} device")
    _require(manifest["threadgroup_memory_bytes"] == 0,
             "the MLX host bridges kernels that declare no threadgroup storage")
    bridged = bridge(Path(manifest["source_path"]).read_text(), entry_point=manifest["entry_point"],
                     threadgroups_per_grid=manifest["threadgroups_per_grid"],
                     threads_per_threadgroup=manifest["threads_per_threadgroup"])
    records = {f"v{index}": record for index, record in enumerate(manifest["buffers"])}
    _require(tuple(n for n, r in records.items() if r["mode"] == "input") == bridged.input_names and
             tuple(n for n, r in records.items() if r["mode"] == "output") == bridged.output_names,
             "emitted buffer address spaces differ from the manifest modes")
    inputs = []
    for name in bridged.input_names:
        record = records[name]
        payload = Path(record["input_path"]).read_bytes()
        _require(len(payload) == record["size_bytes"], f"FP32 byte length differs: {record['name']}")
        values = struct.unpack(f"<{record['size_bytes'] // 4}f", payload)
        inputs.append(mx.reshape(mx.array(values, dtype=mx.float32), tuple(record["shape"])))
    kernel = mx.fast.metal_kernel(name=manifest["entry_point"], input_names=list(bridged.input_names),
                                  output_names=list(bridged.output_names), source=bridged.body,
                                  header=bridged.header, compile_options=dict(COMPILE_OPTIONS))
    (directory / "mlx-bridge.metal").write_text(bridged.header + "// MLX generates the signature here "
                                                "from input_names/output_names.\n" + bridged.body + "\n")
    # MLX keys its compiled-kernel cache on the source as well as the name, so a second
    # shape of the same entry point recompiles rather than reusing the first. That makes
    # this interval a JIT plus dispatch on a first source and a dispatch on a repeat.
    began = time.monotonic()
    outputs = kernel(inputs=inputs, grid=bridged.grid, threadgroup=bridged.threadgroup,
                     output_shapes=[tuple(records[name]["shape"]) for name in bridged.output_names],
                     output_dtypes=[mx.float32] * len(bridged.output_names), init_value=OUTPUT_FILL)
    mx.eval(outputs)
    elapsed = time.monotonic() - began
    for name, array in zip(bridged.output_names + bridged.input_names, list(outputs) + inputs):
        record = records[name]
        flat = mx.reshape(array, (record["size_bytes"] // 4,)).tolist()
        Path(record["output_path"]).write_bytes(struct.pack(f"<{len(flat)}f", *flat))
    receipt = {"status": "completed", "command_status": "completed", "host": "mlx",
               "device": host["device_name"], "architecture": host["architecture"],
               "target": manifest["target"], "mlx_version": host["mlx_version"],
               "compiler": "mx.fast.metal_kernel", "math_mode": COMPILE_OPTIONS["math_mode"],
               "execution_model": manifest["execution_model"],
               "active_threads_per_threadgroup": manifest["active_threads_per_threadgroup"],
               "dispatch": "dispatchThreads", "grid": list(bridged.grid),
               "threadgroup": list(bridged.threadgroup),
               "buffer_indices_match_emission": bridged.buffer_indices_match_emission,
               "jit_and_dispatch_seconds": elapsed,
               "output_fill": "nan",
               "unset_compile_options": list(UNSET_COMPILE_OPTIONS),
               "unchecked_pipeline_commitments": list(UNCHECKED_PIPELINE_COMMITMENTS),
               "input_write_back": "read back from the MLX input array, which MLX allocates and owns",
               "scope": "MLX in-process JIT of one emitted kernel; no binary archive and no sealed Candidate"}
    (directory / "runtime.stdout.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
