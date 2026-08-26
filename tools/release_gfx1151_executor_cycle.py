#!/usr/bin/env python3
"""Release an exact gfx1151 Executor on the admitted ROCm host.

The identity is derived from frozen witnesses. A replacement is built and admitted at a
temporary in-repository path before an unwitnessed working descriptor can be replaced;
failure leaves every existing descriptor byte untouched.
"""

from __future__ import annotations

import importlib.metadata
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


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.evaluation.triton_hip import admit_exact_hip  # noqa: E402
from open_cake_ir.lab import ExecutorRevision  # noqa: E402
from tools.executor_revision_witnesses import (  # noqa: E402
    parse_executor_revision_id,
    plan_executor_revision_cycle,
    verify_executor_revision_commit,
)


_PROFILERS = ("rocprofv3", "rocprof", "omniperf")
_BUILD_TOOLS = (
    ("git", "git", ["--version"]),
    ("cxx", "c++", ["--version"]),
    ("ninja", "ninja", ["--version"]),
    ("sh", "/bin/sh", ["-c", "printf POSIX-sh"]),
)
_ROCM_BUILD_TOOLS = (
    ("hipcc", ["--version"]),
    ("hipconfig", ["--version"]),
    ("rocminfo", []),
)
_HIP_PACKAGES = (
    "packaging",
    "pybind11",
    "psutil",
    "setuptools",
    "torch",
    "triton",
)
_AITER_LIBXML2_ENV = "OPEN_CAKE_AITER_LIBXML2"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _tool_record(kind: str, executable: str | Path, version_args: list[str]) -> dict[str, object]:
    unresolved = Path(executable).absolute()
    path = unresolved.resolve(strict=True)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"{kind} executable custody differs")
    completed = subprocess.run(
        [str(unresolved), *version_args],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15,
    )
    version_lines = [
        line.strip()
        for line in (completed.stdout + "\n" + completed.stderr).splitlines()
        if line.strip()
    ]
    if not version_lines:
        raise ValueError(f"{kind} version output differs")
    payload = path.read_bytes()
    return {
        "kind": kind,
        "path": str(unresolved),
        "version": version_lines[0],
        "sha256": sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _library_record(soname: str, value: str | Path) -> dict[str, object]:
    unresolved = Path(value).absolute()
    if unresolved.name != soname or not unresolved.is_file():
        raise ValueError(f"{soname} library custody differs")
    payload = unresolved.resolve(strict=True).read_bytes()
    return {
        "soname": soname,
        "path": str(unresolved),
        "sha256": sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _rocm_build_tool_paths() -> dict[str, Path]:
    discovered = shutil.which("hipcc")
    if discovered is None:
        raise RuntimeError("gfx1151 Executor requires hipcc")
    discovered_target = Path(discovered).resolve(strict=True)
    candidates = []
    for value in (os.environ.get("ROCM_HOME"), os.environ.get("ROCM_PATH")):
        if value:
            candidates.append(Path(value).absolute())
    derived = discovered_target.parent.parent
    if derived.name == "hip":
        derived = derived.parent
    candidates.extend((derived, Path("/opt/rocm")))
    for root in dict.fromkeys(candidates):
        paths = {kind: root / f"bin/{kind}" for kind, _ in _ROCM_BUILD_TOOLS}
        if (
            all(path.is_file() and os.access(path, os.X_OK) for path in paths.values())
            and paths["hipcc"].resolve(strict=True) == discovered_target
        ):
            return paths
    raise RuntimeError("gfx1151 Executor requires one canonical ROCm tool root")


def collect_gfx1151_host_environment() -> dict[str, object]:
    """Capture only live host facts consumed by the formal AMD runner."""

    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("gfx1151 Executor release requires Linux x86_64")
    requirements = {
        "triton_target": {"backend": "hip", "arch": "gfx1151", "warp_size": 32}
    }
    torch, _, _ = admit_exact_hip(requirements)
    hip_version = getattr(torch.version, "hip", None)
    if not isinstance(hip_version, str) or not hip_version:
        raise RuntimeError("gfx1151 Executor requires an exact PyTorch HIP version")

    monitor = shutil.which("amd-smi")
    if monitor is None:
        raise RuntimeError("gfx1151 Executor requires amd-smi")
    profilers = []
    for kind in _PROFILERS:
        executable = shutil.which(kind)
        if executable is not None:
            profilers.append(_tool_record(kind, executable, ["--version"]))
    rocm_tools = _rocm_build_tool_paths()
    build_tools = [
        _tool_record(kind, rocm_tools[kind], version_args)
        for kind, version_args in _ROCM_BUILD_TOOLS
    ]
    for kind, command, version_args in _BUILD_TOOLS:
        executable = shutil.which(command)
        if executable is None:
            raise RuntimeError(f"gfx1151 Executor requires {kind}")
        build_tools.append(_tool_record(kind, executable, version_args))
    libxml2 = os.environ.get(_AITER_LIBXML2_ENV)
    if not libxml2:
        raise RuntimeError(
            f"gfx1151 Executor requires {_AITER_LIBXML2_ENV}"
        )

    python = Path(sys.executable).absolute()
    return {
        "runtime_kind": "hip",
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "kernel_release": platform.release(),
        },
        "python": {
            "invocation_path": str(python),
            "version": platform.python_version(),
            "resolved_sha256": sha256(
                python.resolve(strict=True).read_bytes()
            ).hexdigest(),
        },
        "packages": {
            name: importlib.metadata.version(name) for name in _HIP_PACKAGES
        },
        "runtime": {
            "backend": "hip",
            "torch_hip_version": hip_version,
            "visible_device_count": 1,
        },
        "runtime_libraries": [
            _library_record("libxml2.so.2", libxml2)
        ],
        "tools": {
            "build_tools": build_tools,
            "device_monitor": _tool_record("amd-smi", monitor, ["version"]),
            "profilers": profilers,
        },
    }


def _current_gfx1151_id(root: Path) -> str:
    identities = []
    for path in (root / "runtime/executors").glob("open-cake-ir-gfx1151-v*.json"):
        if not path.is_file() or path.is_symlink():
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        identity = parse_executor_revision_id(
            document.get("executor_id") if isinstance(document, Mapping) else None
        )
        if identity is None or identity.family != "gfx1151":
            raise ValueError(f"gfx1151 Executor descriptor {path.name} differs")
        if path.name != f"{identity.revision_id}.json":
            raise ValueError(f"gfx1151 Executor descriptor {path.name} identity differs")
        identities.append(identity)
    return (
        max(identities, key=lambda value: value.ordinal).revision_id
        if identities
        else "open-cake-ir-gfx1151-v1"
    )


def release_gfx1151_executor(root: Path = ROOT) -> dict[str, object]:
    root = root.resolve(strict=True)
    current_id = _current_gfx1151_id(root)
    plan = plan_executor_revision_cycle(root, current_id)
    if plan.family != "gfx1151":
        raise RuntimeError("gfx1151 Executor release plan differs")
    executor_id = plan.next_revision_id
    final = root / f"runtime/executors/{executor_id}.json"
    relative_final = final.relative_to(root).as_posix()
    if final.exists() and relative_final not in plan.reclaimable_descriptors:
        raise FileExistsError("refusing to replace a witnessed gfx1151 Executor")
    initial_final_sha256 = sha256(final.read_bytes()).hexdigest() if final.exists() else None

    host = collect_gfx1151_host_environment()
    proposal = {
        "schema_version": 2,
        "executor_id": executor_id,
        "state": "draft",
        "sources": [],
        "host_environment": host,
    }
    candidate = root / f"runtime/executors/.{executor_id}.candidate.{os.getpid()}.json"
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError("gfx1151 Executor candidate path already exists")

    with tempfile.TemporaryDirectory(prefix="open-cake-gfx1151-executor-") as directory:
        proposal_path = Path(directory) / "proposal.json"
        proposal_path.write_bytes(_canonical_json_bytes(proposal) + b"\n")
        command = [
            sys.executable,
            str(root / "tools/release_executor.py"),
            "--project-root",
            str(root),
            "--proposal",
            str(proposal_path),
            "--output",
            str(candidate),
        ]
        if final.exists():
            command.extend(("--replace-unwitnessed", str(final)))
        try:
            subprocess.run(
                command,
                cwd=root,
                check=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            released = ExecutorRevision.load(root, candidate)
            if released.executor_id != executor_id:
                raise RuntimeError("gfx1151 Executor candidate identity differs")
            admission = released.admit_hip_host()
            if admission.executor_id != executor_id:
                raise RuntimeError("gfx1151 Executor live admission differs")
            verify_executor_revision_commit(
                root, executor_id, final, initial_final_sha256
            )
            committed = ExecutorRevision.load(root, candidate)
            if (
                committed.executor_id != executor_id
                or committed.canonical_sha256 != released.canonical_sha256
            ):
                raise RuntimeError("gfx1151 Executor candidate changed during release")
            os.replace(candidate, final)
        finally:
            if candidate.exists() or candidate.is_symlink():
                candidate.unlink()

    for relative in plan.reclaimable_descriptors:
        stale = root / relative
        if stale != final and stale.exists():
            stale.unlink()
    raw = final.read_bytes()
    document = cast(dict[str, object], json.loads(raw))
    result = {
        "schema_version": 1,
        "executor_id": executor_id,
        "path": relative_final,
        "canonical_sha256": sha256(_canonical_json_bytes(document)).hexdigest(),
        "descriptor_raw_sha256": sha256(raw).hexdigest(),
        "source_count": len(cast(list[object], document["sources"])),
        "profiler_kinds": [
            value["kind"]
            for value in cast(
                list[dict[str, object]],
                cast(dict[str, object], host["tools"])["profilers"],
            )
        ],
    }
    return result


def main() -> int:
    if len(sys.argv) != 1:
        raise SystemExit("usage: release_gfx1151_executor_cycle.py")
    print(json.dumps(release_gfx1151_executor(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
