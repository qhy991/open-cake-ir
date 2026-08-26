#!/usr/bin/env python3
"""Prepare or run the pinned AITER RMSNorm correctness baseline on gfx1151."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import importlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from hashlib import sha256
from pathlib import Path
from typing import Iterator, Mapping, cast


ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.evaluation import (  # noqa: E402
    WorkloadContract,
    generate_rmsnorm_case,
    rmsnorm_metrics,
    rmsnorm_oracle,
)
from open_cake_ir.evaluation.triton_hip import (  # noqa: E402
    canonical_json_bytes,
    require_object,
    write_new_json,
)
from open_cake_ir.lab import ExecutorRevision  # noqa: E402


WORKLOAD = "contracts/workloads/llama-rmsnorm-mul-fp32-v2.json"
WORKLOAD_SHA256 = "62a132326208fdd7c7023d9f7ee2e9775433d1c4a0a17b04f873fa0a1f716306"
RUNNER_SOURCE = "examples/gpu/aiter_rmsnorm_amd_baseline.py"
RESULT_KIND = "open_cake_external_aiter_rmsnorm_gfx1151_correctness_v1"
AITER_REPOSITORY = "https://github.com/ROCm/aiter.git"
AITER_TAG = "v0.1.20"
AITER_REVISION = "fc2e5d57fb5b8ad8e7e23f7103071dde798ea618"
ENTRY_POINT = "aiter.ops.rmsnorm.rms_norm_opus"
_EXECUTOR_ID = re.compile(r"open-cake-ir-gfx1151-v[1-9][0-9]*")
_CODE_OBJECT = re.compile(rb"amdhsa--(gfx[0-9a-z]+)")
_OFFLOAD_ARCH = re.compile(r"--offload-arch=(gfx[0-9a-z]+)")
_EXPECTED_CASE_IDS = ("seeded_random", "reduction_rsqrt_stress")
_EXPECTED_SHAPE = (8, 512, 128)
_SOURCE_MARKERS = {
    "aiter/ops/rmsnorm.py": (
        '@compile_ops("module_rmsnorm", fc_name="rms_norm_opus", ffi_type="ctypes")',
        "def rms_norm_opus(",
        "_rms_norm_opus_raw(",
    ),
    "aiter/jit/core.py": ('"gfx1151",',),
    "aiter/jit/optCompilerConfig.json": (
        '"module_rmsnorm": {',
        "rmsnorm_opus_norm_fp32.cu",
    ),
    "csrc/kernels/rmsnorm/rmsnorm_opus_norm_fp32.cu": (
        "OPUS_NORM_DEFINE(opus_norm_fp32, fp32_t)",
    ),
    "csrc/kernels/rmsnorm/rmsnorm_opus_norm_entry.cu": (
        "OPUS_EXPORT void rms_norm_opus",
    ),
    "csrc/include/rmsnorm.h": ("rmsnorm_opus_kernel<",),
    "csrc/include/opus/rmsnorm_opus_kernel.hpp": (
        "__global__ void rmsnorm_opus_kernel",
    ),
}
_JIT_ENVIRONMENT = {
    "AITER_AOT_IMPORT": "1",
    "AITER_ENABLE_EXPERIMENTAL": "0",
    "AITER_LOG_MORE": "0",
    "AITER_LOG_TUNED_CONFIG": "0",
    "AITER_REBUILD": "0",
    "AITER_SYMBOL_VISIBLE": "0",
    "AITER_TRITON_ONLY": "0",
    "AITER_DISABLE_KERNARG_PRELOAD": "0",
    "AITER_ASM_DEBUG": "0",
    "AITER_FP4x2": "0",
    "ENABLE_CK": "0",
    "ENABLE_ROPE_POSITIONS_INT32": "0",
    "GPU_ARCHS": "gfx1151",
    "PYTORCH_ROCM_ARCH": "gfx1151",
}


def _object(value: object, context: str) -> Mapping[str, object]:
    return require_object(value, context)


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(root: Path, *arguments: str, executable: str = "git") -> str:
    return subprocess.run(
        [executable, *arguments],
        cwd=root,
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _project_git_state(
    project_root: Path, git_executable: str = "git"
) -> dict[str, object]:
    revision = _git(project_root, "rev-parse", "HEAD", executable=git_executable)
    status = _git(
        project_root,
        "status",
        "--porcelain",
        "--untracked-files=all",
        executable=git_executable,
    )
    return {"revision": revision, "tree_clean": not bool(status)}


def _remote_is_aiter(value: str) -> bool:
    normalized = value.strip().lower()
    prefixes = (
        "https://github.com/",
        "http://github.com/",
        "ssh://git@github.com/",
        "git@github.com:",
    )
    for prefix in prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    return normalized.removesuffix(".git").rstrip("/") == "rocm/aiter"


def _validate_aiter_source_chain(
    root: Path, git_executable: str = "git"
) -> None:
    relative_paths = tuple(_SOURCE_MARKERS)
    _git(
        root,
        "ls-files",
        "--error-unmatch",
        "--",
        *relative_paths,
        executable=git_executable,
    )
    for relative, markers in _SOURCE_MARKERS.items():
        unresolved = root / relative
        if unresolved.is_symlink():
            raise ValueError(f"AITER source {relative!r} custody differs")
        path = unresolved.resolve(strict=True)
        if root not in path.parents or not path.is_file():
            raise ValueError(f"AITER source {relative!r} custody differs")
        source = path.read_text(encoding="utf-8")
        if any(marker not in source for marker in markers):
            raise ValueError(f"AITER source chain differs at {relative}")


def _admit_aiter_checkout(
    value: Path, git_executable: str = "git"
) -> dict[str, object]:
    if value.is_symlink():
        raise ValueError("AITER checkout must not be a symlink")
    root = value.resolve(strict=True)
    if (
        _git(
            root,
            "rev-parse",
            "--show-toplevel",
            executable=git_executable,
        )
        != str(root)
    ):
        raise ValueError("AITER checkout root differs")
    revision = _git(root, "rev-parse", "HEAD", executable=git_executable)
    if revision != AITER_REVISION:
        raise ValueError("AITER checkout revision differs")
    tag_revision = _git(
        root,
        "rev-parse",
        f"refs/tags/{AITER_TAG}^{{commit}}",
        executable=git_executable,
    )
    if tag_revision != AITER_REVISION:
        raise ValueError("AITER release tag differs")
    remotes = tuple(
        filter(None, _git(root, "remote", executable=git_executable).splitlines())
    )
    remote_urls = tuple(
        _git(root, "remote", "get-url", name, executable=git_executable)
        for name in remotes
    )
    if not any(_remote_is_aiter(url) for url in remote_urls):
        raise ValueError("AITER checkout repository differs")
    if _git(
        root,
        "status",
        "--porcelain",
        "--untracked-files=all",
        executable=git_executable,
    ):
        raise ValueError("AITER checkout tree must be clean")
    ignored = tuple(
        filter(
            None,
            _git(
                root,
                "ls-files",
                "--others",
                "--ignored",
                "--exclude-standard",
                "--",
                "aiter",
                executable=git_executable,
            ).splitlines(),
        )
    )
    import_hazards = tuple(
        path
        for path in ignored
        if Path(path).suffix in {".pyc", ".pyd", ".so"}
        or "__pycache__" in Path(path).parts
    )
    if import_hazards:
        raise ValueError("AITER checkout contains ignored import artifacts")
    submodules = tuple(
        line
        for line in _git(
            root,
            "submodule",
            "status",
            "--recursive",
            executable=git_executable,
        ).splitlines()
        if line
    )
    if any(line[0] in "+U" for line in submodules):
        raise ValueError("AITER submodule revision differs")
    _validate_aiter_source_chain(root, git_executable)
    return {
        "repository": AITER_REPOSITORY,
        "tag": AITER_TAG,
        "revision": revision,
        "tree_clean": True,
        "submodule_count": len(submodules),
        "submodules_initialized": sum(line[0] != "-" for line in submodules),
        "submodules_used": [],
        "ignored_import_artifacts": 0,
        "entry_point": ENTRY_POINT,
        "jit_module": "module_rmsnorm",
        "c_symbol": "rms_norm_opus",
    }


def _load_workload(project_root: Path) -> WorkloadContract:
    workload = WorkloadContract.load(project_root / WORKLOAD)
    document = workload.document
    if (
        workload.canonical_sha256 != WORKLOAD_SHA256
        or workload.workload_id != "llama-rmsnorm-mul-fp32-independent-v2"
        or workload.case_ids != _EXPECTED_CASE_IDS
        or document.get("operator") != "rmsnorm_mul_fp32"
    ):
        raise ValueError("AITER baseline Workload authority differs")
    semantics = _object(document.get("semantics"), "workload.semantics")
    tensors = _object(document.get("tensors"), "workload.tensors")
    if (
        semantics.get("definition")
        != "y = x * gamma * rsqrt(mean_D(x * x) + epsilon)"
        or semantics.get("epsilon") != 1.0e-6
        or semantics.get("reduction_axis") != "last"
        or semantics.get("input_dtype") != "fp32"
        or semantics.get("output_dtype") != "fp32"
        or any(
            _object(tensors.get(name), f"workload.tensors.{name}").get("dtype")
            != "fp32"
            for name in ("x", "gamma", "y")
        )
    ):
        raise ValueError("AITER baseline Workload semantics differ")
    for case_id in workload.case_ids:
        shape = _object(workload.case(case_id).get("shape"), "workload.case.shape")
        if tuple(int(shape[name]) for name in ("B", "N", "D")) != _EXPECTED_SHAPE:
            raise ValueError("AITER baseline Workload shape differs")
    return workload


def _admit_project_root(project_root: Path) -> None:
    root = project_root.resolve(strict=True)
    runner = root / RUNNER_SOURCE
    if (
        root != ROOT.resolve(strict=True)
        or runner.is_symlink()
        or runner.resolve(strict=True) != Path(__file__).resolve(strict=True)
    ):
        raise ValueError("Open Cake project root differs from the running checkout")


def _prepare(
    project_root: Path,
    aiter_root: Path,
) -> tuple[WorkloadContract, dict[str, object]]:
    _admit_project_root(project_root)
    project_source = _project_git_state(project_root)
    if not bool(project_source["tree_clean"]):
        raise ValueError("Open Cake checkout tree must be clean")
    aiter_source = _admit_aiter_checkout(aiter_root)
    workload = _load_workload(project_root)
    summary: dict[str, object] = {
        "schema_version": 1,
        "kind": RESULT_KIND,
        "status": "prepared",
        "scope": {
            "operator": "rmsnorm_mul_fp32",
            "role": "external_baseline_candidate",
            "correctness_only": True,
            "performance_measured": False,
            "speedup_claim_authorized": False,
            "promotion_authorized": False,
            "llama_cpp_e2e_evaluated": False,
        },
        "open_cake_source": project_source,
        "aiter_source": aiter_source,
        "baseline": {
            "library": "AITER",
            "entry_point": ENTRY_POINT,
            "dispatch": "direct_module_rmsnorm_opus_fp32",
            "model_sensitive": 0,
            "gemma_norm": False,
            "target": "gfx1151",
            "prebuilt_wheel_used": False,
            "silent_fallback_permitted": False,
        },
        "workload": {
            "path": WORKLOAD,
            "workload_id": workload.workload_id,
            "canonical_sha256": workload.canonical_sha256,
            "case_ids": list(workload.case_ids),
            "shape": list(_EXPECTED_SHAPE),
        },
        "jit_plan": {
            **_JIT_ENVIRONMENT,
            "AITER_META_DIR": "<exact-aiter-checkout>",
            "AITER_JIT_DIR": "<new-external-directory>",
            "CK_DIR": "<disabled-nonexistent-directory>",
            "CXX": "<executor-admitted-cxx>",
            "HIP_KITTENS_DIR": "<disabled-nonexistent-directory>",
            "ROCM_HOME": "<executor-admitted-hipcc-root>",
            "LD_LIBRARY_PATH": "<executor-admitted-libxml2-directory>",
            "executor_build_tools": [
                "cxx",
                "git",
                "hipcc",
                "hipconfig",
                "ninja",
                "rocminfo",
                "sh",
            ],
        },
        "evaluation": {
            "torch_imported": False,
            "aiter_imported": False,
            "jit_started": False,
            "gpu_submitted": False,
            "operator_calls": 0,
            "kernel_launch_count_observed": False,
            "expected_fallback_calls": 0,
            "fallback_calls_observed": False,
            "correctness_passed": False,
            "performance_measured": False,
        },
    }
    return workload, summary


def _admit_executor_contract(executor: object) -> None:
    document = executor.document
    host = _object(document.get("host_environment"), "Executor host")
    tools = _object(host.get("tools"), "Executor tools")
    build_tools = tools.get("build_tools")
    runtime_libraries = host.get("runtime_libraries")
    if (
        document.get("schema_version") != 2
        or _EXECUTOR_ID.fullmatch(executor.executor_id) is None
        or host.get("runtime_kind") != "hip"
        or not isinstance(build_tools, (list, tuple))
        or not isinstance(runtime_libraries, (list, tuple))
    ):
        raise ValueError("AITER baseline requires an exact released gfx1151 Executor")
    tool_kinds = {
        str(_object(value, "Executor build tool").get("kind"))
        for value in build_tools
    }
    if tool_kinds != {
        "cxx",
        "git",
        "hipcc",
        "hipconfig",
        "ninja",
        "rocminfo",
        "sh",
    }:
        raise ValueError("gfx1151 Executor does not own the AITER JIT toolchain")
    sonames = {
        str(_object(value, "Executor runtime library").get("soname"))
        for value in runtime_libraries
    }
    if sonames != {"libxml2.so.2"}:
        raise ValueError("gfx1151 Executor does not own the AITER runtime libraries")
    source_paths = {
        str(_object(value, "Executor source").get("path"))
        for value in cast(tuple[object, ...], document["sources"])
    }
    if RUNNER_SOURCE not in source_paths:
        raise ValueError("gfx1151 Executor does not own the AITER runner")


def _new_external_directory(value: Path, roots: tuple[Path, ...]) -> Path:
    candidate = value.absolute()
    path = candidate.parent.resolve(strict=True) / candidate.name
    if path.exists() or path.is_symlink():
        raise ValueError("attempt directories must be new paths")
    if any(path == root or root in path.parents for root in roots):
        raise ValueError("attempt directories must stay outside both source checkouts")
    return path


def _resolve_attempt_directories(
    project_root: Path,
    aiter_root: Path,
    evidence_value: Path,
    jit_value: Path,
) -> tuple[Path, Path]:
    roots = (project_root.resolve(strict=True), aiter_root.resolve(strict=True))
    evidence = _new_external_directory(evidence_value, roots)
    jit = _new_external_directory(jit_value, roots)
    if evidence == jit or evidence in jit.parents or jit in evidence.parents:
        raise ValueError("evidence and JIT directories must not overlap")
    return evidence, jit


def _write_manifest(root: Path) -> None:
    files = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.name == "manifest.json":
            continue
        payload = path.read_bytes()
        files.append(
            {
                "path": path.name,
                "sha256": sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    write_new_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "kind": "open_cake_external_aiter_baseline_manifest_v1",
            "files": files,
        },
    )


def _failure_class(stage: str) -> str:
    if stage in {
        "executor_admission",
        "source_custody",
        "jit_environment",
        "artifact_validation",
        "import_artifact_validation",
        "post_artifact_validation",
        "post_source_custody",
    }:
        return "AUTHORITY_BLOCKED"
    if stage in {
        "executor_host_admission",
        "device_admission",
        "aiter_import",
        "jit_build",
    }:
        return "ENVIRONMENT_BLOCKED"
    if stage == "operator_execution":
        return "RUNTIME_FAULT"
    return "HARNESS_FAULT"


def _admit_device() -> tuple[object, object]:
    torch = importlib.import_module("torch")
    if not getattr(torch.version, "hip", None):
        raise RuntimeError("a ROCm PyTorch build is required")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the AITER baseline requires exactly one visible HIP GPU")
    properties = torch.cuda.get_device_properties(0)
    if (
        getattr(properties, "gcnArchName", None) != "gfx1151"
        or int(properties.warp_size) != 32
    ):
        raise RuntimeError("the HIP device is not exact gfx1151 wave32")
    return torch, properties


def _tool_path(
    build_tools: Mapping[str, Mapping[str, object]], kind: str
) -> Path:
    record = build_tools.get(kind)
    if record is None:
        raise ValueError(f"gfx1151 Executor is missing build tool {kind!r}")
    path = Path(str(record["path"]))
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"gfx1151 Executor build tool {kind!r} custody differs")
    return path


def _library_path(
    runtime_libraries: Mapping[str, Mapping[str, object]], soname: str
) -> Path:
    record = runtime_libraries.get(soname)
    if record is None:
        raise ValueError(f"gfx1151 Executor is missing runtime library {soname!r}")
    path = Path(str(record["path"]))
    if path.name != soname or not path.is_absolute() or not path.is_file():
        raise ValueError(f"gfx1151 Executor runtime library {soname!r} differs")
    return path


def _rocm_home(build_tools: Mapping[str, Mapping[str, object]]) -> Path:
    hipcc = _tool_path(build_tools, "hipcc")
    home = hipcc.parent.parent
    if home / "bin/hipcc" != hipcc:
        raise ValueError("gfx1151 Executor hipcc is not in one exact ROCm root")
    for kind in ("hipconfig", "rocminfo"):
        expected = _tool_path(build_tools, kind)
        candidate = home / f"bin/{kind}"
        if candidate != expected:
            raise ValueError(f"gfx1151 Executor {kind} differs from the ROCm root")
    return home


def _admit_build_tool_resolution(
    build_tools: Mapping[str, Mapping[str, object]], path_value: str
) -> None:
    for command, kind in (
        ("git", "git"),
        ("hipcc", "hipcc"),
        ("hipconfig", "hipconfig"),
        ("ninja", "ninja"),
        ("rocminfo", "rocminfo"),
    ):
        observed = shutil.which(command, path=path_value)
        if observed is None or Path(observed).resolve(strict=True) != _tool_path(
            build_tools, kind
        ).resolve(strict=True):
            raise ValueError(f"AITER would resolve an unadmitted {command}")
    if Path("/bin/sh").resolve(strict=True) != _tool_path(
        build_tools, "sh"
    ).resolve(strict=True):
        raise ValueError("AITER subprocess shell differs from the Executor")


@contextlib.contextmanager
def _aiter_environment(
    aiter_root: Path,
    jit_dir: Path,
    build_tools: Mapping[str, Mapping[str, object]],
    runtime_libraries: Mapping[str, Mapping[str, object]],
) -> Iterator[dict[str, object]]:
    if any(name == "aiter" or name.startswith("aiter.") for name in sys.modules):
        raise RuntimeError("AITER was imported before exact source admission")
    rocm_home = _rocm_home(build_tools)
    tool_directories = tuple(
        dict.fromkeys(
            str(_tool_path(build_tools, kind).parent)
            for kind in ("git", "hipcc", "hipconfig", "rocminfo", "cxx", "ninja")
        )
    )
    original_path = os.environ.get("PATH", "")
    path_value = os.pathsep.join((*tool_directories, original_path))
    libxml2 = _library_path(runtime_libraries, "libxml2.so.2")
    original_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    library_path = os.pathsep.join(
        value for value in (str(libxml2.parent), original_library_path) if value
    )
    values = {
        **_JIT_ENVIRONMENT,
        "AITER_META_DIR": str(aiter_root),
        "AITER_JIT_DIR": str(jit_dir),
        "CK_DIR": str(jit_dir / "disabled-composable-kernel"),
        "CXX": str(_tool_path(build_tools, "cxx")),
        "HIP_KITTENS_DIR": str(jit_dir / "disabled-hip-kittens"),
        "LD_LIBRARY_PATH": library_path,
        "PATH": path_value,
        "ROCM_HOME": str(rocm_home),
        "ROCM_PATH": str(rocm_home),
    }
    unset = (
        "CC",
        "HIP_CLANG_PATH",
        "PREBUILD_THREAD_NUM",
        "TORCH_DONT_CHECK_COMPILER_ABI",
    )
    aiter_written = (
        "AITER_ASM_DIR",
        "AITER_GPU_ARCHS",
        "FLYDSL_RUNTIME_CACHE_DIR",
        "MAX_JOBS",
    )
    previous = {
        name: os.environ.get(name) for name in (*values, *unset, *aiter_written)
    }
    old_path = list(sys.path)
    try:
        os.environ.update(values)
        for name in unset:
            os.environ.pop(name, None)
        _admit_build_tool_resolution(build_tools, path_value)
        sys.path.insert(0, str(aiter_root))
        yield {
            **_JIT_ENVIRONMENT,
            "AITER_META_DIR": str(aiter_root),
            "AITER_JIT_DIR": str(jit_dir),
            "CK_DIR": values["CK_DIR"],
            "CXX": values["CXX"],
            "HIP_KITTENS_DIR": values["HIP_KITTENS_DIR"],
            "LD_LIBRARY_PATH": str(libxml2.parent),
            "ROCM_HOME": str(rocm_home),
            "build_tools": {
                kind: str(_tool_path(build_tools, kind))
                for kind in sorted(build_tools)
            },
            "path_precedence": list(tool_directories),
            "runtime_libraries": {"libxml2.so.2": str(libxml2)},
        }
    finally:
        sys.path[:] = old_path
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _inside(root: Path, path: Path, context: str) -> None:
    if path != root and root not in path.parents:
        raise RuntimeError(f"{context} did not load from the exact AITER checkout")


def _reject_preexisting_aiter_operator(torch: object) -> None:
    try:
        getattr(torch.ops.aiter, "_rms_norm_opus_raw")
    except AttributeError:
        return
    raise RuntimeError("AITER Torch operator existed before pinned source import")


def _validate_loaded_aiter_modules(
    aiter_root: Path, git_executable: str
) -> list[str]:
    relative_paths = []
    for name, value in sorted(sys.modules.items()):
        if name != "aiter" and not name.startswith("aiter."):
            continue
        module_file = getattr(value, "__file__", None)
        if module_file is None:
            continue
        path = Path(str(module_file)).resolve(strict=True)
        _inside(aiter_root, path, f"AITER module {name}")
        if path.suffix != ".py":
            raise RuntimeError(f"AITER module {name} did not load tracked Python source")
        relative_paths.append(path.relative_to(aiter_root).as_posix())
    if not relative_paths:
        raise RuntimeError("AITER source import loaded no tracked modules")
    _git(
        aiter_root,
        "ls-files",
        "--error-unmatch",
        "--",
        *relative_paths,
        executable=git_executable,
    )
    return relative_paths


def _import_aiter_operator(
    aiter_root: Path, git_executable: str
) -> tuple[object, object, list[str]]:
    module = importlib.import_module("aiter.ops.rmsnorm")
    package = importlib.import_module("aiter")
    package_path = Path(str(package.__file__)).resolve(strict=True)
    module_path = Path(str(module.__file__)).resolve(strict=True)
    _inside(aiter_root, package_path, "AITER package")
    _inside(aiter_root, module_path, "AITER RMSNorm module")
    entry_point = getattr(module, "rms_norm_opus", None)
    if not callable(entry_point):
        raise RuntimeError("pinned AITER RMSNorm entry point is missing")
    loaded_sources = _validate_loaded_aiter_modules(aiter_root, git_executable)
    return module, entry_point, loaded_sources


def _admit_aiter_runtime_architecture() -> str:
    chip_info = importlib.import_module("aiter.jit.utils.chip_info")
    observed = str(chip_info.get_gfx_runtime())
    if observed != "gfx1151":
        raise RuntimeError("AITER runtime architecture differs from gfx1151")
    return observed


def _build_aiter_rmsnorm_module(jit_dir: Path) -> None:
    module_path = jit_dir / "module_rmsnorm.so"
    core_path = jit_dir / "module_aiter_core.so"
    if (
        module_path.exists()
        or module_path.is_symlink()
        or set(jit_dir.glob("*.so")) != {core_path}
    ):
        raise RuntimeError("AITER JIT import dependency set differs before RMSNorm build")
    core = importlib.import_module("aiter.jit.core")
    arguments = core.get_args_of_build("module_rmsnorm")
    arguments["torch_exclude"] = True
    core.build_module(
        "module_rmsnorm",
        arguments["srcs"],
        arguments["flags_extra_cc"],
        arguments["flags_extra_hip"],
        arguments["blob_gen_cmd"],
        arguments["extra_include"],
        arguments["extra_ldflags"],
        arguments["verbose"],
        arguments["is_python_module"],
        arguments["is_standalone"],
        arguments["torch_exclude"],
        arguments.get("third_party", []),
        arguments.get("hipify", False),
        flags_extra_hip_per_source=arguments.get(
            "flags_extra_hip_per_source", {}
        ),
    )
    if set(jit_dir.glob("*.so")) != {core_path, module_path}:
        raise RuntimeError("AITER JIT built an unexpected shared module")


def _admit_built_module_runtime_path() -> None:
    core = importlib.import_module("aiter.jit.core")
    if bool(core._needs_arch_rebuild("module_rmsnorm")):
        raise RuntimeError("AITER would rebuild module_rmsnorm before operator use")


def _evaluate_cases(
    torch: object,
    entry_point: object,
    workload: WorkloadContract,
) -> list[dict[str, object]]:
    epsilon = float(_object(workload.document["semantics"], "semantics")["epsilon"])
    cases: list[dict[str, object]] = []
    for case_id in workload.case_ids:
        x, gamma = generate_rmsnorm_case(workload, case_id, device="cuda")
        x_before = x.clone()
        gamma_before = gamma.clone()
        x_pointer = int(x.data_ptr())
        gamma_pointer = int(gamma.data_ptr())
        output = torch.empty_like(x)
        entry_point(output, x, gamma, epsilon, 0, False)
        torch.cuda.synchronize()
        input_unchanged = (
            int(x.data_ptr()) == x_pointer
            and int(gamma.data_ptr()) == gamma_pointer
            and bool(torch.equal(x, x_before))
            and bool(torch.equal(gamma, gamma_before))
        )
        output_contract = (
            tuple(output.shape) == _EXPECTED_SHAPE
            and output.dtype == torch.float32
            and bool(output.is_contiguous())
            and int(output.data_ptr()) not in {x_pointer, gamma_pointer}
        )
        reference = rmsnorm_oracle(workload, x_before, gamma_before)
        metrics = rmsnorm_metrics(workload, output, reference)
        passed = bool(metrics["passed"]) and input_unchanged and output_contract
        cases.append(
            {
                "case_id": case_id,
                **metrics,
                "input_unchanged": input_unchanged,
                "output_contract_passed": output_contract,
                "passed": passed,
            }
        )
    return cases


def _validate_elf_artifact(
    path: Path, exported_symbol: str
) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"AITER {path.name} artifact is missing")
    payload = path.read_bytes()
    if not payload.startswith(b"\x7fELF"):
        raise RuntimeError(f"AITER {path.name} is not an ELF shared object")
    code_objects = sorted(
        {match.group(1).decode() for match in _CODE_OBJECT.finditer(payload)}
    )
    if code_objects != ["gfx1151"]:
        raise RuntimeError(f"AITER {path.name} code object differs from gfx1151")
    library = ctypes.CDLL(str(path))
    if getattr(library, exported_symbol, None) is None:
        raise RuntimeError(f"AITER {path.name} does not export {exported_symbol}")
    return {
        "path": str(path),
        "sha256": sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "elf": True,
        "code_objects": code_objects,
        "exported_symbol": exported_symbol,
    }


def _validate_module_artifact(path: Path) -> dict[str, object]:
    return _validate_elf_artifact(path, "rms_norm_opus")


def _validate_core_artifact(path: Path) -> dict[str, object]:
    return _validate_elf_artifact(path, "PyInit_module_aiter_core")


def _validate_build_plan(
    path: Path,
    build_tools: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("AITER module_rmsnorm build plan is missing")
    payload = path.read_bytes()
    source = payload.decode("utf-8")
    architectures = sorted(set(_OFFLOAD_ARCH.findall(source)))
    if architectures != ["gfx1151"]:
        raise RuntimeError("AITER module_rmsnorm build target differs")
    expected_cxx = str(_tool_path(build_tools, "cxx"))
    expected_hipcc = str(_rocm_home(build_tools) / "bin/hipcc")
    if (
        f"cxx = {expected_cxx}" not in source
        or f"nvcc = {expected_hipcc}" not in source
    ):
        raise RuntimeError("AITER module_rmsnorm build tools differ")
    return {
        "path": str(path),
        "sha256": sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "offload_architectures": architectures,
        "cxx": expected_cxx,
        "hipcc": expected_hipcc,
    }


def _copy_new(source: Path, destination: Path) -> None:
    with destination.open("xb") as stream:
        stream.write(source.read_bytes())


def _run_live_impl(
    project_root: Path,
    aiter_root: Path,
    executor: ExecutorRevision,
    workload: WorkloadContract,
    summary: dict[str, object],
    evidence_root: Path,
    jit_dir: Path,
    attempt: dict[str, str],
) -> int:
    attempt["stage"] = "executor_admission"
    _admit_executor_contract(executor)
    attempt["stage"] = "executor_host_admission"
    host = executor.admit_hip_host()
    build_tools = host.build_tools
    runtime_libraries = host.runtime_libraries
    git_executable = str(_tool_path(build_tools, "git"))
    attempt["stage"] = "source_custody"
    project_source = _project_git_state(project_root, git_executable)
    if not bool(project_source["tree_clean"]):
        raise ValueError("Open Cake checkout tree must be clean")
    aiter_source = _admit_aiter_checkout(aiter_root, git_executable)
    attempt["stage"] = "device_admission"
    torch, properties = _admit_device()
    _reject_preexisting_aiter_operator(torch)
    attempt["stage"] = "jit_environment"
    if jit_dir.exists() or jit_dir.is_symlink():
        raise ValueError("AITER JIT directory must still be a new path")

    attempt["stage"] = "aiter_import"
    with _aiter_environment(
        aiter_root, jit_dir, build_tools, runtime_libraries
    ) as jit_environment:
        _, entry_point, loaded_sources = _import_aiter_operator(
            aiter_root, git_executable
        )
        aiter_runtime_arch = _admit_aiter_runtime_architecture()
        attempt["stage"] = "import_artifact_validation"
        core_path = jit_dir / "module_aiter_core.so"
        core_record = _validate_core_artifact(core_path)
        core_plan_path = jit_dir / "build/module_aiter_core/build/build.ninja"
        core_plan_record = _validate_build_plan(core_plan_path, build_tools)
        retained_core = evidence_root / "module_aiter_core.so"
        retained_core_plan = evidence_root / "module_aiter_core.build.ninja"
        _copy_new(core_path, retained_core)
        _copy_new(core_plan_path, retained_core_plan)
        attempt["stage"] = "jit_build"
        _build_aiter_rmsnorm_module(jit_dir)
        attempt["stage"] = "artifact_validation"
        module_path = jit_dir / "module_rmsnorm.so"
        forbidden_module = jit_dir / "module_rmsnorm_quant.so"
        if forbidden_module.exists() or forbidden_module.is_symlink():
            raise RuntimeError(
                "AITER dynamic RMSNorm dispatch compiled a forbidden module"
            )
        module_record = _validate_module_artifact(module_path)
        build_plan_path = jit_dir / "build/module_rmsnorm/build/build.ninja"
        build_plan_record = _validate_build_plan(build_plan_path, build_tools)
        _admit_built_module_runtime_path()
        retained_module = evidence_root / "module_rmsnorm.so"
        retained_build_plan = evidence_root / "module_rmsnorm.build.ninja"
        _copy_new(module_path, retained_module)
        _copy_new(build_plan_path, retained_build_plan)
        attempt["stage"] = "operator_execution"
        cases = _evaluate_cases(torch, entry_point, workload)
        attempt["stage"] = "post_artifact_validation"
        final_module_record = _validate_module_artifact(module_path)
        final_build_plan_record = _validate_build_plan(build_plan_path, build_tools)
        final_core_record = _validate_core_artifact(core_path)
        final_core_plan_record = _validate_build_plan(core_plan_path, build_tools)
        if (
            module_record != final_module_record
            or build_plan_record != final_build_plan_record
            or core_record != final_core_record
            or core_plan_record != final_core_plan_record
            or forbidden_module.exists()
            or forbidden_module.is_symlink()
        ):
            raise RuntimeError("AITER build artifacts changed during correctness")

    attempt["stage"] = "post_source_custody"
    final_project_source = _project_git_state(project_root, git_executable)
    final_aiter_source = _admit_aiter_checkout(aiter_root, git_executable)
    if project_source != final_project_source or aiter_source != final_aiter_source:
        raise RuntimeError("source checkout changed during AITER correctness")

    attempt["stage"] = "result_retention"
    passed = all(bool(case["passed"]) for case in cases)
    summary["status"] = "passed" if passed else "STOP_CORRECTNESS_REJECTED"
    summary["open_cake_source"] = final_project_source
    summary["aiter_source"] = final_aiter_source
    summary["executor"] = {
        **dict(executor.reference),
        "host_admitted": True,
        "torch_hip_version": host.torch_hip_version,
    }
    summary["runtime"] = {
        "python": sys.version.split()[0],
        "torch": importlib.metadata.version("torch"),
        "torch_hip": str(torch.version.hip),
        "device_name": str(properties.name),
        "gcn_arch_name": str(properties.gcnArchName),
        "warp_size": int(properties.warp_size),
        "visible_device_count": 1,
    }
    summary["build"] = {
        "jit_environment": jit_environment,
        "loaded_aiter_sources": loaded_sources,
        "aiter_runtime_architecture": aiter_runtime_arch,
        "import_dependency": {
            "module": {
                **core_record,
                "retained_path": retained_core.name,
            },
            "build_plan": {
                **core_plan_record,
                "retained_path": retained_core_plan.name,
            },
        },
        "module": {
            **module_record,
            "retained_path": retained_module.name,
        },
        "build_plan": {
            **build_plan_record,
            "retained_path": retained_build_plan.name,
        },
        "forbidden_module_present": False,
    }
    summary["evaluation"] = {
        "torch_imported": True,
        "aiter_imported": True,
        "jit_started": True,
        "gpu_submitted": True,
        "operator_calls": len(cases),
        "expected_kernel_launches_per_call": 1,
        "kernel_launch_count_observed": False,
        "expected_fallback_calls": 0,
        "fallback_calls_observed": False,
        "cases": cases,
        "correctness_passed": passed,
        "performance_measured": False,
        "speedup_claim_authorized": False,
        "promotion_authorized": False,
    }
    summary["artifact_directory"] = str(evidence_root)
    write_new_json(evidence_root / "result.json", summary)
    return 0 if passed else 2


def _run_live(
    project_root: Path,
    aiter_root: Path,
    executor: ExecutorRevision,
    workload: WorkloadContract,
    summary: dict[str, object],
    evidence_root: Path,
    jit_dir: Path,
) -> int:
    evidence_root.mkdir(mode=0o700)
    authority = {
        "schema_version": 1,
        "kind": RESULT_KIND,
        "workload": summary["workload"],
        "aiter_source": summary["aiter_source"],
        "baseline": summary.get("baseline"),
        "executor": dict(executor.reference),
        "jit_directory": str(jit_dir),
        "performance_authorized": False,
    }
    write_new_json(evidence_root / "attempt-authority.json", authority)
    attempt = {"stage": "executor_admission"}
    try:
        result = _run_live_impl(
            project_root,
            aiter_root,
            executor,
            workload,
            summary,
            evidence_root,
            jit_dir,
            attempt,
        )
        _write_manifest(evidence_root)
        return result
    except BaseException as error:
        failure_class = _failure_class(attempt["stage"])
        failure = {
            "schema_version": 1,
            "kind": "open_cake_external_aiter_baseline_failure_v1",
            "status": failure_class,
            "failure_class": failure_class,
            "failed_stage": attempt["stage"],
            "authority": authority,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "gpu_result_authorized": False,
            "performance_conclusion_authorized": False,
        }
        if not (evidence_root / "failure.json").exists():
            write_new_json(evidence_root / "failure.json", failure)
        if not (evidence_root / "manifest.json").exists():
            _write_manifest(evidence_root)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aiter-checkout", type=Path, required=True)
    parser.add_argument("--executor", type=Path)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--jit-dir", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    arguments = parser.parse_args()

    project_root = ROOT.resolve(strict=True)
    aiter_argument = arguments.aiter_checkout
    if aiter_argument.is_symlink():
        parser.error("--aiter-checkout must not be a symlink")
    aiter_root = aiter_argument.resolve(strict=True)
    workload, summary = _prepare(project_root, aiter_root)

    if arguments.prepare_only:
        if any(
            value is not None
            for value in (arguments.executor, arguments.evidence_root, arguments.jit_dir)
        ):
            parser.error("prepare-only does not accept live execution paths")
        sys.stdout.buffer.write(canonical_json_bytes(summary) + b"\n")
        return 0

    if arguments.executor is None:
        parser.error("--executor is required for live correctness")
    if arguments.evidence_root is None or arguments.jit_dir is None:
        parser.error("--evidence-root and --jit-dir are required for live correctness")
    executor = ExecutorRevision.load(
        project_root, arguments.executor.resolve(strict=True)
    )
    evidence_root, jit_dir = _resolve_attempt_directories(
        project_root,
        aiter_root,
        arguments.evidence_root,
        arguments.jit_dir,
    )
    summary["executor"] = {**dict(executor.reference), "host_admitted": False}
    exit_code = _run_live(
        project_root,
        aiter_root,
        executor,
        workload,
        summary,
        evidence_root,
        jit_dir,
    )
    sys.stdout.buffer.write(canonical_json_bytes(summary) + b"\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
