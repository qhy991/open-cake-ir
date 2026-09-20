"""MACA software-host admission; device admission remains in Evaluation."""

from __future__ import annotations

from collections.abc import Mapping
import importlib
import platform


PACKAGES = frozenset({"torch", "flagtree", "packaging", "pybind11", "psutil", "setuptools"})
BUILD_TOOLS = frozenset({"cxx", "mxcc", "bwrap", "sh"})
LIBRARIES = frozenset({"libmcruntime.so"})


def validate_host(host: Mapping) -> None:
    from .executor import (_python_authority, _package_authority, _build_environment,
                           _executable_record, _shared_library_record)

    if set(host) != {"kind", "platform", "python", "packages", "runtime", "tools", "runtime_libraries"} or host["kind"] != "maca":
        raise ValueError("MACA host fields differ")
    observed_platform = host["platform"]
    if (not isinstance(observed_platform, Mapping)
            or set(observed_platform) != {"system", "machine", "kernel_release"}
            or observed_platform["system"] != "Linux" or observed_platform["machine"] != "x86_64"
            or not isinstance(observed_platform["kernel_release"], str) or not observed_platform["kernel_release"]):
        raise ValueError("MACA host platform differs")
    _python_authority(host["python"], "MACA Python")
    if set(_package_authority(host["packages"], "MACA packages")) != PACKAGES:
        raise ValueError("MACA host package set differs")
    runtime = host["runtime"]
    if (not isinstance(runtime, Mapping)
            or set(runtime) != {"backend", "torch_maca_version", "triton_version", "build_environment"}
            or runtime["backend"] != "maca"
            or any(not isinstance(runtime[k], str) or not runtime[k]
                   for k in ("torch_maca_version", "triton_version"))
            or not _build_environment(runtime["build_environment"])):
        raise ValueError("MACA runtime authority differs")
    if (not isinstance(host["tools"], Mapping) or set(host["tools"]) != {"build_tools"}
            or not isinstance(host["tools"]["build_tools"], (list, tuple))
            or not isinstance(host["runtime_libraries"], (list, tuple))):
        raise ValueError("MACA tool or library records differ")
    tools = [_executable_record(row, "MACA build tool") for row in host["tools"]["build_tools"]]
    libraries = [_shared_library_record(row, "MACA runtime library") for row in host["runtime_libraries"]]
    if len(tools) != len(BUILD_TOOLS) or {row["kind"] for row in tools} != BUILD_TOOLS:
        raise ValueError("MACA build-tool set differs")
    if len(libraries) != len(LIBRARIES) or {row["soname"] for row in libraries} != LIBRARIES:
        raise ValueError("MACA runtime-library set differs")


def admit_host(host: Mapping, *, executor_id: str = "") -> Mapping:
    from .executor import _admit_python_and_packages, _admit_executable, _admit_shared_library

    validate_host(host)
    _admit_python_and_packages(host)
    current = {"system": platform.system(), "machine": platform.machine(), "kernel_release": platform.release()}
    if current != dict(host["platform"]):
        raise ValueError("MACA host platform changed after capture")
    torch = importlib.import_module("torch")
    triton = importlib.import_module("triton")
    backends = importlib.import_module("triton.backends").backends
    if (getattr(torch.version, "maca", None) != host["runtime"]["torch_maca_version"]
            or triton.__version__ != host["runtime"]["triton_version"] or "metax" not in backends):
        raise ValueError("MACA PyTorch or Triton provider differs from capture")
    for row in host["tools"]["build_tools"]:
        _admit_executable(row, "MACA build tool")
    for row in host["runtime_libraries"]:
        _admit_shared_library(row, "MACA runtime library")
    return {"executor_id": executor_id, "kind": "maca", "triton_version": triton.__version__}
