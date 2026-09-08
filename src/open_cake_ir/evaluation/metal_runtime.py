"""Real archived-pipeline observation for the existing common Evaluation worker.

This module runs one declared launch plan. The task worker owns its oracle, assay
policy and receipt; Lab/Ralph owns optimization. No source compilation occurs while
observing, and no observer is built as a fallback for a missing admitted executable.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import struct
import subprocess

from .core import compare_tile_outputs
from .metal_manifest import MetalTensorLaunchManifest
from .metal_observations import command_buffer_ms, validate_host

OBSERVER_SOURCE = Path(__file__).with_name("metal") / "observer.swift"


def _external_directory(directory: Path) -> Path:
    directory = Path(directory)
    if not directory.is_absolute():
        raise ValueError('Metal output directory must be absolute')
    directory = directory.resolve()
    if any((parent / '.git').exists() for parent in (directory, *directory.parents)):
        raise ValueError('Metal output directory must be outside every enclosing checkout')
    return directory


def compile_observer(directory: Path, *, swiftc: str = "swiftc") -> Path:
    """Explicit preparation-only compilation, never called by Evaluation."""
    directory = _external_directory(directory)
    directory.mkdir(parents=True, exist_ok=False)
    executable = directory / "metal-observer"
    command = [swiftc, "-O", str(OBSERVER_SOURCE), "-o", str(executable)]
    (directory / "command.json").write_text(json.dumps(command))
    result = subprocess.run(command, capture_output=True, timeout=120)
    (directory / "stdout.log").write_bytes(result.stdout)
    (directory / "stderr.log").write_bytes(result.stderr)
    if result.returncode or not executable.is_file():
        raise ValueError(f"observer compilation failed; see {directory}")
    return executable


def _pack(values, width):
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError("Metal input/oracle values must be finite numbers")
    payload = struct.pack(f"<{len(values)}{width}", *values)
    if width == "f" and list(struct.unpack(f"<{len(values)}f", payload)) != list(values):
        raise ValueError("Metal inputs must already be rounded to FP32")
    return payload


def _snapshot(directory, name, elements):
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("Metal snapshot path differs")
    path = directory / name
    if path.is_symlink() or not path.is_file() or path.stat().st_size != elements * 4:
        raise ValueError("Metal snapshot file or byte length differs")
    return list(struct.unpack(f"<{elements}f", path.read_bytes()))


def observe(*, workload, candidates: dict, manifests: dict, input_cases: dict, launch_plan: list,
            observer_executable: Path, expected_host: dict, directory: Path, timeout_seconds: int = 300) -> dict:
    """Observe fresh output buffers and independently check every returned launch.

    ``input_cases`` maps Workload case IDs to trusted ``inputs`` and ``expected``
    flat vectors. The native preflight guard avoids timed work after bad outputs;
    Python independently derives the guard again from every actual buffer snapshot.
    """
    validate_host(expected_host)
    executable = Path(observer_executable)
    if not executable.is_absolute() or not executable.is_file():
        raise ValueError("admitted Metal observer executable is unavailable")
    if not candidates or set(candidates) != set(manifests):
        raise ValueError("Metal observer participant/manifest set differs")
    primary = next(iter(manifests.values()))
    for role, candidate in candidates.items():
        manifest = manifests[role]
        if not isinstance(manifest, MetalTensorLaunchManifest):
            raise ValueError("Metal observer requires the admitted tensor manifest")
        manifest.check_workload(workload, primary.case_id)
        if (candidate.target != manifest.target or candidate.entry_point != manifest.kernel_name
                or candidate.launch_spec_sha256 != manifest.canonical_sha256
                or expected_host["target"] != candidate.target):
            raise ValueError("Metal observer sealed participant differs")
        report = json.loads(candidate.artifact_payloads["metal_build_report"])
        if (report["build"]["host"] != expected_host or report["archive_only_reload"]["host"] != expected_host
                or report["archive_only_reload"]["source_library_rebuilt"] is not False):
            raise ValueError("Metal archive build host differs from the admitted Executor")
    for case_id, case in input_cases.items():
        actual = tuple((arg.name, arg.shape, arg.dtype, arg.mode) for arg in workload.tensor_abi(case_id))
        if actual != primary.tensor_abi:
            raise ValueError("validation input case changes the compiled tensor ABI")
        if set(case) != {"inputs", "expected"}:
            raise ValueError("Metal input case payload differs")
    for index, launch in enumerate(launch_plan):
        if (launch.get("index") != index or launch.get("role") not in candidates
                or launch.get("input_case_id") not in input_cases
                or launch.get("phase") not in {"preflight", "cohort", "postflight", "profile"}
                or type(launch.get("timed")) is not bool or type(launch.get("profile")) is not bool
                or launch["timed"] and launch["profile"]):
            raise ValueError("Metal declared launch plan differs")
    directory = _external_directory(directory)
    directory.mkdir(parents=True, exist_ok=False)
    participant_rows = []
    for index, (role, candidate) in enumerate(candidates.items()):
        archive = directory / f"participant-{index}.binary.metallib"
        archive.write_bytes(candidate.artifact_payloads["metal_binary_archive"])
        participant_rows.append({"role": role, "archive_path": str(archive), "manifest": manifests[role].as_dict()})
    case_rows = []
    validation = workload.document["validation"]
    for index, (case_id, case) in enumerate(input_cases.items()):
        paths, oracles = {}, {}
        for tensor_index, (name, shape, _, mode) in enumerate(primary.tensor_abi):
            count = math.prod(shape)
            if mode == "input":
                values = case["inputs"][name]
                if len(values) != count:
                    raise ValueError("Metal input element count differs")
                path = directory / f"input-{index}-{tensor_index}.bin"
                path.write_bytes(_pack(values, "f")); paths[name] = str(path)
            else:
                values = case["expected"][name]
                if len(values) != count:
                    raise ValueError("Metal oracle element count differs")
                expected = directory / f"expected-{index}-{tensor_index}.bin"
                tolerance = directory / f"tolerance-{index}-{tensor_index}.bin"
                expected.write_bytes(_pack(values, "d"))
                tolerance.write_bytes(_pack([validation["atol"] + validation["rtol"] * abs(value) for value in values], "d"))
                oracles[name] = {"expected": str(expected), "tolerance": str(tolerance)}
        case_rows.append({"id": case_id, "inputs": paths, "oracles": oracles})
    request = {"expected_host": expected_host, "participants": participant_rows, "cases": case_rows,
               "launches": launch_plan, "output_directory": str(directory)}
    request_path = directory / "request.json"
    request_path.write_text(json.dumps(request, sort_keys=True))
    completed = subprocess.run([str(executable), str(request_path)], capture_output=True, timeout=timeout_seconds)
    (directory / "observer.stdout.json").write_bytes(completed.stdout)
    (directory / "observer.stderr.log").write_bytes(completed.stderr)
    report = json.loads(completed.stdout)
    if (completed.returncode or report.get("status") != "completed" or report.get("host") != expected_host
            or report.get("source_library_rebuilt") is not False
            or report.get("archive_miss_policy") != "failOnBinaryArchiveMiss"
            or report.get("module_loads") != len(candidates) or not isinstance(report.get("launches"), list)):
        raise ValueError(f"Metal observer failed or changed its admission: {report.get('error')}")
    checked, preflight_passed = [], True
    for record in report["launches"]:
        index = record.get("index")
        if type(index) is not int or not 0 <= index < len(launch_plan):
            raise ValueError("Metal observed launch index differs")
        planned = launch_plan[index]
        if any(record.get(key) != planned[key] for key in ("role", "phase", "input_case_id")):
            raise ValueError("Metal observed launch differs from the declared plan")
        if ("profile" in record) is not planned["profile"]:
            raise ValueError("Metal observer instrumentation differs from declared assay")
        command_buffer_ms(record["command_buffer"])
        if (record["command_buffer"]["launch_index"] != index
                or record["command_buffer"]["timed"] is not planned["timed"]):
            raise ValueError("Metal timestamp observation belongs to another launch")
        case = input_cases[planned["input_case_id"]]
        paths = record.get("buffer_paths")
        if not isinstance(paths, dict) or set(paths) != {row[0] for row in primary.tensor_abi}:
            raise ValueError("Metal raw buffer snapshot coverage differs")
        observed, after = {}, {}
        for name, shape, _, mode in primary.tensor_abi:
            values = _snapshot(directory, paths[name], math.prod(shape))
            (after if mode == "input" else observed)[name] = values
        passed, metrics = compare_tile_outputs(workload, case["inputs"], case["expected"], observed, after)
        if record.get("preflight_guard_passed") is not passed:
            raise ValueError("Metal native preflight guard differs from the independent CPU oracle")
        if planned["phase"] == "preflight":
            preflight_passed &= passed
        checked.append({**planned, "passed": passed, "metrics": metrics, "command_buffer": record["command_buffer"],
                        **({"profile_raw": record["profile"]} if "profile" in record else {})})
    expected_indices = [row["index"] for row in launch_plan if preflight_passed or row["phase"] == "preflight"]
    if [row["index"] for row in checked] != expected_indices:
        raise ValueError("Metal observed plan is missing, reordered or duplicated")
    return {"host": expected_host, "module_loads": len(candidates), "launches": checked}
