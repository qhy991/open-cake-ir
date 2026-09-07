#!/usr/bin/env python3
"""Capture and admit the current Executor host into a new external JSON file."""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

from open_cake_ir.lab.executor import (
    ExecutorRevision, HIP_PACKAGES, HIP_BUILD_TOOLS, HIP_PROFILERS, HIP_RUNTIME_LIBRARIES,
    _external_file, admit_host_environment, admit_profiler_environment,
)


def _file_record(path: Path, recorded_path: str) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": recorded_path,
        "sha256": sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _distribution_files(distribution: importlib.metadata.Distribution) -> list[str]:
    # Distribution.files can silently omit missing paths on newer Python versions.
    # Read RECORD itself so missing or unsafe bound files fail during capture.
    record = distribution.read_text("RECORD")
    if not record:
        raise ValueError(f"distribution {distribution.metadata['Name']!r} has no file manifest")
    rows = list(csv.reader(record.splitlines()))
    if any(len(row) != 3 or not row[0] for row in rows):
        raise ValueError("distribution RECORD contains a malformed file entry")
    return sorted(row[0] for row in rows)


def _capture_cupti(name: str) -> dict[str, object]:
    distribution = importlib.metadata.distribution(name)
    site = Path(distribution.locate_file("")).resolve(strict=True)
    files = _distribution_files(distribution)
    selected = [
        value for value in files
        if (
            Path(value).parts[:1] == ("cupti",)
            and "__pycache__" not in Path(value).parts
            and Path(value).suffix != ".pyc"
        ) or (
            len(Path(value).parts) == 2
            and Path(value).parts[0].endswith(".dist-info")
            and Path(value).name in {"METADATA", "RECORD"}
        )
    ]
    if "cupti/__init__.py" not in selected or any(
        sum(Path(value).name == required for value in selected) != 1
        for required in ("METADATA", "RECORD")
    ):
        raise ValueError("CUPTI distribution lacks its package or METADATA/RECORD")
    return {
        "site_packages_path": str(site),
        "distribution": name,
        "version": distribution.version,
        "files": [
            _file_record(_external_file(site, value, "CUPTI capture"), value)
            for value in selected
        ],
    }


def _capture_flashinfer(name: str) -> dict[str, object]:
    distribution = importlib.metadata.distribution(name)
    relative = "flashinfer/testing/utils.py"
    if relative not in _distribution_files(distribution):
        raise ValueError("FlashInfer distribution does not own its testing helper")
    site = Path(distribution.locate_file("")).resolve(strict=True)
    path = _external_file(site, relative, "FlashInfer capture")
    return {
        **_file_record(path, str(path)),
        "distribution": name,
        "version": distribution.version,
    }


def _capture_profiler(path: Path) -> dict[str, object]:
    if (
        not path.is_absolute() or path.is_symlink() or not path.is_file()
        or not os.access(path, os.X_OK)
    ):
        raise ValueError("Nsight Compute requires an explicit absolute executable file")
    completed = subprocess.run(
        [str(path), "--version"], check=False, capture_output=True, text=True, timeout=30,
    )
    if completed.returncode:
        raise RuntimeError(
            f"Nsight Compute version command exited {completed.returncode}: "
            f"stdout={completed.stdout!r}; stderr={completed.stderr!r}"
        )
    match = re.search(r"^Version (\S+)", completed.stdout, re.MULTILINE)
    if match is None:
        raise ValueError(f"Nsight Compute version output differs: {completed.stdout!r}")
    return {**_file_record(path, str(path)), "version": match.group(1)}


def _capture_hip_tool(kind: str, path: Path) -> dict[str, object]:
    if (
        kind not in HIP_BUILD_TOOLS | HIP_PROFILERS | {"amd-smi"}
        or not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK)
    ):
        raise ValueError("HIP tool requires a known kind and explicit absolute executable")
    # The schema preserves invocation paths (e.g. /bin/sh); the record pins the
    # resolved executable bytes. The shell's interface label has no --version API.
    arguments = ["-c", "printf 'POSIX-sh\\n'"] if kind == "sh" else ["--version"]
    record = _file_record(path.resolve(strict=True), str(path))
    completed = subprocess.run(
        [str(path), *arguments], check=False, capture_output=True, text=True, timeout=30,
    )
    if completed.returncode:
        raise RuntimeError(
            f"HIP {kind} version command exited {completed.returncode}: "
            f"stdout={completed.stdout!r}; stderr={completed.stderr!r}"
        )
    lines = (completed.stdout or completed.stderr).splitlines()
    version = next((line.strip() for line in lines if line.strip()), "")
    if not version:
        raise ValueError(f"HIP {kind} version output is empty")
    return {**record, "kind": kind, "version": version}


def _capture_hip_library(soname: str, path: Path) -> dict[str, object]:
    if soname not in HIP_RUNTIME_LIBRARIES or not path.is_absolute() or not path.is_file():
        raise ValueError("HIP library requires a known soname and explicit absolute file")
    return {**_file_record(path.resolve(strict=True), str(path)), "soname": soname}


def _capture_host(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.runtime_kind == "hip" and set(arguments.package) != HIP_PACKAGES:
        raise ValueError("Executor HIP package set differs")
    python = Path(sys.executable).absolute()
    common = {
        "python": {
            "invocation_path": str(python),
            "version": sys.version.split()[0],
            "resolved_sha256": sha256(python.resolve(strict=True).read_bytes()).hexdigest(),
        },
        "packages": {name: importlib.metadata.version(name) for name in sorted(set(arguments.package))},
    }
    if arguments.runtime_kind == "hip":
        torch = importlib.import_module("torch")
        version = getattr(torch, "version", None)
        hip = getattr(version, "hip", None)
        if not isinstance(hip, str) or not hip or getattr(version, "cuda", None) is not None:
            raise ValueError("Executor HIP runtime differs")
        return {
            **common,
            "runtime_kind": "hip",
            "platform": {
                "system": platform.system(), "machine": platform.machine(),
                "kernel_release": platform.release(),
            },
            # This is the required topology. Exact device observation is owned by
            # admit_exact_hip and requires the Compiler's lowering requirements.
            "runtime": {"backend": "hip", "torch_hip_version": hip, "visible_device_count": 1},
            "tools": {
                "device_monitor": _capture_hip_tool("amd-smi", arguments.amd_smi),
                "build_tools": [
                    _capture_hip_tool(kind, Path(path))
                    for kind, path in arguments.hip_build_tool
                ],
                "profilers": [
                    _capture_hip_tool(kind, Path(path))
                    for kind, path in arguments.hip_profiler
                ],
            },
            "runtime_libraries": [
                _capture_hip_library(soname, Path(path))
                for soname, path in arguments.hip_runtime_library
            ],
        }
    return {
        **common,
        "cupti_python": _capture_cupti(arguments.cupti_distribution),
        "flashinfer_helper": _capture_flashinfer(arguments.flashinfer_distribution),
        "nsight_compute": _capture_profiler(arguments.ncu),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", action="append", required=True,
                        help="installed distribution to bind; repeat for each runtime dependency")
    parser.add_argument("--runtime-kind", choices=("cuda", "hip"), default="cuda")
    parser.add_argument("--cupti-distribution")
    parser.add_argument("--flashinfer-distribution")
    parser.add_argument("--ncu", type=Path)
    parser.add_argument("--amd-smi", type=Path)
    parser.add_argument("--hip-build-tool", nargs=2, action="append", default=[],
                        metavar=("KIND", "PATH"))
    parser.add_argument("--hip-profiler", nargs=2, action="append", default=[],
                        metavar=("KIND", "PATH"))
    parser.add_argument("--hip-runtime-library", nargs=2, action="append", default=[],
                        metavar=("SONAME", "PATH"))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    cuda_arguments = (arguments.cupti_distribution, arguments.flashinfer_distribution, arguments.ncu)
    hip_arguments = (arguments.amd_smi, arguments.hip_build_tool, arguments.hip_runtime_library)
    if arguments.runtime_kind == "cuda":
        if not all(cuda_arguments) or any(hip_arguments) or arguments.hip_profiler:
            raise ValueError("CUDA capture requires its CUPTI, FlashInfer and NCU inputs only")
    elif not all(hip_arguments) or any(cuda_arguments):
        raise ValueError("HIP capture requires its monitor, build tools and libraries only")
    if not arguments.output.is_absolute():
        raise ValueError("host capture output must be an absolute external path")
    output = arguments.output.parent.resolve(strict=True) / arguments.output.name
    if output.exists() or output.is_symlink():
        raise FileExistsError("refusing to overwrite an Executor host capture")
    project_root = Path(__file__).resolve().parents[1]
    if project_root in output.parents or any(
        (parent / ".git").exists() for parent in output.parents
    ):
        raise ValueError("host capture output must be outside project checkouts")

    host = _capture_host(arguments)
    ExecutorRevision._validate_host_document(
        host, schema_version=2 if arguments.runtime_kind == "hip" else 1,
    )
    admit_host_environment(host)
    if arguments.runtime_kind == "cuda":
        admit_profiler_environment(host)
    payload = json.dumps(host, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    profiler_admitted = arguments.runtime_kind == "cuda" or bool(arguments.hip_profiler)
    print(json.dumps({"output": str(output), "host_admitted": True, "profiler_admitted": profiler_admitted}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
