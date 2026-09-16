#!/usr/bin/env python3
"""Capture and admit the current Executor host into a new external JSON file."""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from hashlib import sha256
from pathlib import Path

from open_cake_ir.lab.executor import (
    ExecutorRevision, HIP_PACKAGES, HIP_BUILD_TOOLS, HIP_DEVICE_MONITORS,
    HIP_PROFILERS, HIP_RUNTIME_LIBRARIES,
    _external_file, admit_host_environment, admit_profiler_environment,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def _inspection_root(arguments: argparse.Namespace):
    """Metal inspection artifacts stay outside every checkout, and are kept only if asked."""

    if arguments.inspection_directory is None:
        with tempfile.TemporaryDirectory(prefix="metal-host-capture-") as directory:
            yield Path(directory)
        return
    directory = arguments.inspection_directory
    if (not directory.is_absolute() or PROJECT_ROOT in directory.parents
            or any((parent / ".git").exists() for parent in (directory, *directory.parents))):
        raise ValueError("--inspection-directory must be absolute and outside every checkout")
    directory.mkdir(parents=True, exist_ok=True)
    yield directory


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


# Some tools expose no version interface at all, so asking for one and refusing on a
# non-zero exit refuses the host rather than the tool. Measured on a Hygon DTK host:
# rocprof exits 1 for --version, -v and --help alike, printing its run banner every time.
# A tool listed here is identified by the digest this record already pins, which is the
# stronger identity anyway, and its version field says plainly that there is none rather
# than carrying a line scraped out of a usage message.
_NO_VERSION_INTERFACE = "no version interface"
_HIP_TOOLS_WITHOUT_VERSION = frozenset({"rocprof"})


def _capture_hip_tool(kind: str, path: Path) -> dict[str, object]:
    if (
        kind not in HIP_BUILD_TOOLS | HIP_PROFILERS | HIP_DEVICE_MONITORS
        or not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK)
    ):
        raise ValueError("HIP tool requires a known kind and explicit absolute executable")
    if kind in _HIP_TOOLS_WITHOUT_VERSION:
        return {**_file_record(path.resolve(strict=True), str(path)),
                "kind": kind, "version": _NO_VERSION_INTERFACE}
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


def _capture_build_environment(entries: list[list[str]]) -> dict[str, str]:
    """Admit the environment this host's toolchain needs inside the isolated build jail.

    Every absolute path a value names must exist here, now, so a missing directory is a
    capture-time error rather than a jail that starts and then cannot find the runtime it
    was pointed at. Values are kept verbatim: a search path's order is the loader's.
    """
    admitted: dict[str, str] = {}
    for name, value in entries:
        if name in admitted:
            raise ValueError(f"build environment declares {name!r} twice")
        for part in value.split(":"):
            if part.startswith("/") and not Path(part).is_dir():
                raise ValueError(f"build environment {name} names a missing directory: {part}")
        admitted[name] = value
    return admitted


def _capture_host(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.host_kind == "hip" and set(arguments.package) != HIP_PACKAGES:
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
    if arguments.host_kind == "hip":
        torch = importlib.import_module("torch")
        version = getattr(torch, "version", None)
        hip = getattr(version, "hip", None)
        if not isinstance(hip, str) or not hip or getattr(version, "cuda", None) is not None:
            raise ValueError("Executor HIP runtime differs")
        return {
            **common,
            "kind": "hip",
            "platform": {
                "system": platform.system(), "machine": platform.machine(),
                "kernel_release": platform.release(),
            },
            # This is the required topology. Exact device observation is owned by
            # admit_exact_hip and requires the Compiler's lowering requirements.
            # `build_environment` is what this host's toolchain needs inside the
            # isolated build jail, which runs --clearenv. On the Hygon DTK host that is
            # LD_LIBRARY_PATH, without which libgalaxyhip.so.5 is mounted and unfindable
            # because ldconfig does not know /opt/dtk, and ROCM_PATH, without which
            # clang-18 reports "cannot find ROCm device library". Both live only in
            # /opt/dtk/env.sh, which the jail correctly discards. A CUDA host declares
            # none and keeps the empty environment it has always had.
            "runtime": {"backend": "hip", "torch_hip_version": hip, "visible_device_count": 1,
                        "build_environment": _capture_build_environment(
                            arguments.hip_build_environment)},
            "tools": {
                "device_monitor": _capture_hip_tool(
                    arguments.device_monitor[0], Path(arguments.device_monitor[1])
                ),
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


def _capture_metal_host(arguments: argparse.Namespace) -> dict[str, object]:
    from open_cake_ir.compiler.target import Target
    from open_cake_ir.lab.metal_host import command_text, inspect_metal_host, observe_sdk
    if (arguments.target is None or arguments.swiftc is None or arguments.archive_executable is None
            or arguments.observer_executable is None):
        raise ValueError("Metal capture requires target, swiftc, archive executable and observer executable")
    if any(value is not None for value in (arguments.cupti_distribution, arguments.flashinfer_distribution, arguments.ncu)):
        raise ValueError("Metal capture must not inherit CUDA host fields")
    for name in ("swiftc", "archive_executable", "observer_executable"):
        path = getattr(arguments, name)
        if (not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK)
                or name != "swiftc" and path.is_symlink()):
            raise ValueError(f"Metal {name} requires an explicit executable path")
    swiftc = arguments.swiftc.absolute()
    target = Target.load(arguments.project_root / "compiler/targets" / f"{arguments.target}.json")
    python = Path(sys.executable).absolute()
    host = {"kind": "metal", "python": {
        "invocation_path": str(python), "version": sys.version.split()[0],
        "resolved_sha256": sha256(python.resolve(strict=True).read_bytes()).hexdigest()},
        "packages": {name: importlib.metadata.version(name) for name in arguments.package},
        "swift": {"invocation_path": str(swiftc), "version": command_text([str(swiftc), "--version"]),
                  "resolved_sha256": sha256(swiftc.resolve(strict=True).read_bytes()).hexdigest()},
        "sdk": observe_sdk(),
        "archive_executable": _file_record(arguments.archive_executable, str(arguments.archive_executable)),
        "observer_executable": _file_record(arguments.observer_executable, str(arguments.observer_executable))}
    with _inspection_root(arguments) as inspection:
        host["host"] = inspect_metal_host(arguments.archive_executable, target=arguments.target,
            expected_device_names=list(target.device_names), directory=inspection / "inspect")
    return host


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Not required at the parser: a Metal host's package map is legitimately empty, and
    # v110 and v112 both carry one. CUDA and HIP require their own sets below.
    parser.add_argument("--package", action="append", default=[],
                        help="installed distribution to bind; repeat for each runtime dependency")
    parser.add_argument("--kind", "--runtime-kind", dest="host_kind",
                        choices=("cuda", "hip", "amd", "metal"), default="cuda",
                        help="host runtime kind; amd is the public name for the HIP/DCU host")
    parser.add_argument("--cupti-distribution")
    parser.add_argument("--flashinfer-distribution")
    parser.add_argument("--ncu", type=Path)
    parser.add_argument("--device-monitor", "--amd-smi", dest="device_monitor",
                        nargs=2, metavar=("KIND", "PATH"),
                        help="installed device monitor: one of "
                             + ", ".join(sorted(HIP_DEVICE_MONITORS)))
    parser.add_argument("--hip-build-tool", nargs=2, action="append", default=[],
                        metavar=("KIND", "PATH"))
    parser.add_argument("--hip-profiler", nargs=2, action="append", default=[],
                        metavar=("KIND", "PATH"))
    parser.add_argument("--hip-runtime-library", nargs=2, action="append", default=[],
                        metavar=("SONAME", "PATH"))
    parser.add_argument("--hip-build-environment", nargs=2, action="append", default=[],
                        metavar=("NAME", "VALUE"),
                        help="environment variable the isolated build jail must be given; "
                             "repeat per variable. The jail clears the environment, so a "
                             "value only an env.sh knows about is declared here or lost.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT,
                        help="checkout whose runtime/hosts/<target>.json this capture writes")
    parser.add_argument("--target", required=True,
                        help="exact target this host runs; the capture is committed as "
                             "runtime/hosts/<target>.json")
    parser.add_argument("--swiftc", type=Path)
    parser.add_argument("--archive-executable", type=Path)
    parser.add_argument("--observer-executable", type=Path)
    parser.add_argument("--replace", action="store_true",
                        help="recapture a target whose host capture already exists")
    parser.add_argument("--inspection-directory", type=Path,
                        help="absolute external directory keeping the Metal inspection request and result")
    arguments = parser.parse_args(argv)
    if arguments.host_kind == "amd":
        arguments.host_kind = "hip"
    cuda_arguments = (arguments.cupti_distribution, arguments.flashinfer_distribution, arguments.ncu)
    hip_arguments = (arguments.device_monitor, arguments.hip_build_tool,
                     arguments.hip_runtime_library)
    metal_arguments = (arguments.swiftc, arguments.archive_executable,
                       arguments.observer_executable)
    if arguments.host_kind == "cuda":
        if not arguments.package or not all(cuda_arguments) or any(hip_arguments) \
                or arguments.hip_profiler \
                or any(value is not None for value in metal_arguments):
            raise ValueError(
                "CUDA capture requires packages and its CUPTI, FlashInfer and NCU inputs only")
    elif arguments.host_kind == "metal":
        # _capture_metal_host owns the refusal of inherited CUDA fields and names them;
        # only the HIP inputs it has never heard of are refused here.
        if any(hip_arguments) or arguments.hip_profiler:
            raise ValueError("Metal capture must not receive HIP host fields")
    elif not all(hip_arguments) or any(cuda_arguments) \
            or any(value is not None for value in metal_arguments):
        raise ValueError("HIP capture requires its monitor, build tools and libraries only")
    # The capture belongs to the checkout: it is committed, and the commit plus this
    # document is the Executor identity (ADR 0065).
    project_root = arguments.project_root.resolve(strict=True)
    if not (project_root / "compiler/targets" / f"{arguments.target}.json").is_file():
        raise ValueError(
            f"{arguments.target!r} is not a Target this checkout declares; "
            "capture a host only for a declared exact target"
        )
    output = project_root / "runtime/hosts" / f"{arguments.target}.json"
    if output.exists() and not arguments.replace:
        raise FileExistsError(
            f"{output} already describes this target; pass --replace to recapture this host"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    host = (_capture_metal_host(arguments) if arguments.host_kind == "metal"
            else _capture_host(arguments))
    ExecutorRevision._validate_host_document(host)
    admit_host_environment(host)
    if arguments.host_kind == "cuda":
        admit_profiler_environment(host)
    document = {"schema_version": 1, "target": arguments.target, "host_environment": host}
    payload = json.dumps(document, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False) + "\n"
    output.write_text(payload, encoding="utf-8")
    # Metal's observer is admitted inside admit_host_environment, not as a separate
    # profiler step, so its capture reports the same admitted observation the CUDA
    # NCU step does.
    profiler_admitted = (arguments.host_kind in ("cuda", "metal")
                         or bool(arguments.hip_profiler))
    print(json.dumps({"output": str(output), "target": arguments.target,
                      "host_admitted": True, "profiler_admitted": profiler_admitted}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
