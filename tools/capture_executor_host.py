#!/usr/bin/env python3
"""Capture and admit the current Executor host into a new external JSON file."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

from open_cake_ir.lab.executor import (
    ExecutorRevision, _external_file, admit_host_environment, admit_profiler_environment,
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


def _capture_host(arguments: argparse.Namespace) -> dict[str, object]:
    python = Path(sys.executable).absolute()
    return {
        "python": {
            "invocation_path": str(python),
            "version": sys.version.split()[0],
            "resolved_sha256": sha256(python.resolve(strict=True).read_bytes()).hexdigest(),
        },
        "packages": {
            name: importlib.metadata.version(name) for name in arguments.package
        },
        "cupti_python": _capture_cupti(arguments.cupti_distribution),
        "flashinfer_helper": _capture_flashinfer(arguments.flashinfer_distribution),
        "nsight_compute": _capture_profiler(arguments.ncu),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", action="append", required=True,
                        help="installed distribution to bind; repeat for each runtime dependency")
    parser.add_argument("--cupti-distribution", required=True)
    parser.add_argument("--flashinfer-distribution", required=True)
    parser.add_argument("--ncu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
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
    ExecutorRevision._validate_host_document(host)
    admit_host_environment(host)
    admit_profiler_environment(host)
    payload = json.dumps(host, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    print(json.dumps({"output": str(output), "host_admitted": True, "profiler_admitted": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
