"""Content-bound Executor Revision shared by every Study variant."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping, cast


_GFX1151_EXECUTOR_ID = re.compile(r"open-cake-ir-gfx1151-v[1-9][0-9]*")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _digest(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} digest differs")
    return value


def _relative_file(root: Path, value: object, context: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} path differs")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "\\" in value:
        raise ValueError(f"{context} path is unsafe")
    unresolved = root
    for part in relative.parts:
        unresolved /= part
        if unresolved.is_symlink():
            raise ValueError(f"{context} file custody differs")
    path = unresolved.resolve(strict=True)
    if root not in path.parents or not path.is_file():
        raise ValueError(f"{context} file custody differs")
    return value, path


def _external_file(root: Path, value: object, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} path differs")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "\\" in value:
        raise ValueError(f"{context} path is unsafe")
    unresolved = root
    for part in relative.parts:
        unresolved /= part
        if unresolved.is_symlink():
            raise ValueError(f"{context} file custody differs")
    path = unresolved.resolve(strict=True)
    if root not in path.parents or not path.is_file():
        raise ValueError(f"{context} file custody differs")
    return path


def _file_record(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "size_bytes"}:
        raise ValueError(f"{context} fields differ")
    _digest(value["sha256"], f"{context}.sha256")
    if (
        not isinstance(value["size_bytes"], int)
        or isinstance(value["size_bytes"], bool)
        or value["size_bytes"] <= 0
    ):
        raise ValueError(f"{context}.size_bytes differs")
    return cast(Mapping[str, object], value)


def _python_authority(value: object, context: str) -> Mapping[str, object]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"invocation_path", "version", "resolved_sha256"}
        or not isinstance(value["invocation_path"], str)
        or not Path(value["invocation_path"]).is_absolute()
        or not isinstance(value["version"], str)
        or not value["version"]
    ):
        raise ValueError(f"{context} authority differs")
    _digest(value["resolved_sha256"], f"{context}.resolved_sha256")
    return cast(Mapping[str, object], value)


def _package_authority(value: object, context: str) -> Mapping[str, object]:
    if (
        not isinstance(value, Mapping)
        or not value
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
            for name, version in value.items()
        )
    ):
        raise ValueError(f"{context} requirements differ")
    return cast(Mapping[str, object], value)


def _executable_record(value: object, context: str) -> Mapping[str, object]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"kind", "path", "version", "sha256", "size_bytes"}
        or not isinstance(value["kind"], str)
        or not value["kind"]
        or not isinstance(value["path"], str)
        or not Path(value["path"]).is_absolute()
        or not isinstance(value["version"], str)
        or not value["version"]
        or not isinstance(value["size_bytes"], int)
        or isinstance(value["size_bytes"], bool)
        or value["size_bytes"] <= 0
    ):
        raise ValueError(f"{context} authority differs")
    _digest(value["sha256"], f"{context}.sha256")
    return cast(Mapping[str, object], value)


def _admit_executable(
    value: object, context: str
) -> Mapping[str, object]:
    record = _executable_record(value, context)
    unresolved = Path(str(record["path"]))
    if not unresolved.is_file() or not os.access(unresolved, os.X_OK):
        raise ValueError(f"{context} custody differs")
    path = unresolved.resolve(strict=True)
    payload = path.read_bytes()
    if (
        sha256(payload).hexdigest() != record["sha256"]
        or len(payload) != record["size_bytes"]
    ):
        raise ValueError(f"{context} bytes differ")
    return MappingProxyType(dict(record))


@dataclass(frozen=True)
class HipHostAdmission:
    """Exact static ROCm host facts admitted before device-specific execution."""

    executor_id: str
    torch_hip_version: str
    visible_device_count: int
    device_monitor: Mapping[str, object]
    profilers: tuple[Mapping[str, object], ...]
    build_tools: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True)
class ExecutorRevision:
    """One source and host-runtime closure for all experiment side effects."""

    executor_id: str
    canonical_sha256: str
    document: Mapping[str, object]
    project_root: Path
    relative_path: str

    @classmethod
    def load(cls, project_root: str | Path, path: str | Path) -> "ExecutorRevision":
        """Load and verify every repository-owned byte in a released Executor."""

        root = Path(project_root).resolve(strict=True)
        unresolved_source = Path(path)
        if unresolved_source.is_symlink():
            raise ValueError("Executor Revision custody differs")
        source = unresolved_source.resolve(strict=True)
        try:
            relative_source = source.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError("Executor Revision escapes the project root") from error
        document = json.loads(source.read_text(encoding="utf-8"))
        if (
            not isinstance(document, Mapping)
            or set(document)
            != {
                "schema_version",
                "executor_id",
                "state",
                "sources",
                "host_environment",
            }
            or document.get("schema_version") not in {1, 2}
            or document.get("state") != "released"
            or not isinstance(document.get("executor_id"), str)
            or not document["executor_id"]
        ):
            raise ValueError("Executor Revision fields, schema, or state differ")
        if (
            document["schema_version"] == 2
            and _GFX1151_EXECUTOR_ID.fullmatch(str(document["executor_id"])) is None
        ):
            raise ValueError("Executor schema v2 gfx1151 identity differs")
        sources = document["sources"]
        host = document["host_environment"]
        if not isinstance(sources, list) or not sources or not isinstance(host, Mapping):
            raise ValueError("Executor Revision closure differs")
        seen: set[str] = set()
        for index, value in enumerate(sources):
            record = _file_record(value, f"executor.sources[{index}]")
            relative, file_path = _relative_file(
                root, record["path"], f"executor.sources[{index}]"
            )
            if relative in seen:
                raise ValueError(f"Executor Revision path {relative!r} is duplicated")
            seen.add(relative)
            payload = file_path.read_bytes()
            if (
                record["sha256"] != sha256(payload).hexdigest()
                or record["size_bytes"] != len(payload)
            ):
                raise ValueError(f"Executor Revision file {relative!r} differs")
        cls._validate_host_document(
            cast(int, document["schema_version"]),
            cast(Mapping[str, object], host),
        )
        detached = cast(
            Mapping[str, object],
            _freeze_json(json.loads(_canonical_json_bytes(document))),
        )
        return cls(
            executor_id=str(document["executor_id"]),
            canonical_sha256=sha256(_canonical_json_bytes(document)).hexdigest(),
            document=detached,
            project_root=root,
            relative_path=relative_source,
        )

    @staticmethod
    def _validate_host_document(
        schema_version: int, host: Mapping[str, object]
    ) -> None:
        if schema_version == 2:
            ExecutorRevision._validate_hip_host_document(host)
            return
        if schema_version != 1:
            raise ValueError("Executor host environment schema differs")
        legacy_fields = {
            "python",
            "packages",
            "cupti_python",
            "flashinfer_helper",
        }
        if frozenset(host) not in {
            frozenset(legacy_fields),
            frozenset(legacy_fields | {"nsight_compute"}),
        }:
            raise ValueError("Executor host environment fields differ")
        python = host["python"]
        packages = host["packages"]
        cupti = host["cupti_python"]
        helper = host["flashinfer_helper"]
        _python_authority(python, "Executor Python")
        _package_authority(packages, "Executor package")
        if (
            not isinstance(cupti, Mapping)
            or set(cupti)
            != {"site_packages_path", "distribution", "version", "files"}
            or not isinstance(cupti["site_packages_path"], str)
            or not Path(cupti["site_packages_path"]).is_absolute()
            or not isinstance(cupti["distribution"], str)
            or not isinstance(cupti["version"], str)
            or not isinstance(cupti["files"], list)
            or not cupti["files"]
        ):
            raise ValueError("Executor CUPTI Python authority differs")
        for index, value in enumerate(cast(list[object], cupti["files"])):
            _file_record(value, f"executor.cupti_python.files[{index}]")
        if (
            not isinstance(helper, Mapping)
            or set(helper) != {"path", "distribution", "version", "sha256", "size_bytes"}
            or not isinstance(helper["path"], str)
            or not Path(helper["path"]).is_absolute()
            or not isinstance(helper["distribution"], str)
            or not isinstance(helper["version"], str)
        ):
            raise ValueError("Executor FlashInfer helper authority differs")
        _digest(helper["sha256"], "executor.flashinfer_helper.sha256")
        if (
            not isinstance(helper["size_bytes"], int)
            or isinstance(helper["size_bytes"], bool)
            or helper["size_bytes"] <= 0
        ):
            raise ValueError("Executor FlashInfer helper size differs")
        if "nsight_compute" in host:
            profiler = host["nsight_compute"]
            if (
                not isinstance(profiler, Mapping)
                or set(profiler) != {"path", "version", "sha256", "size_bytes"}
                or not isinstance(profiler["path"], str)
                or not Path(profiler["path"]).is_absolute()
                or not isinstance(profiler["version"], str)
                or not profiler["version"]
                or not isinstance(profiler["size_bytes"], int)
                or isinstance(profiler["size_bytes"], bool)
                or profiler["size_bytes"] <= 0
            ):
                raise ValueError("Executor Nsight Compute authority differs")
            _digest(profiler["sha256"], "executor.nsight_compute.sha256")

    @staticmethod
    def _validate_hip_host_document(host: Mapping[str, object]) -> None:
        if set(host) != {
            "runtime_kind",
            "platform",
            "python",
            "packages",
            "runtime",
            "tools",
        } or host.get("runtime_kind") != "hip":
            raise ValueError("Executor HIP host environment fields differ")
        platform_value = host["platform"]
        if (
            not isinstance(platform_value, Mapping)
            or set(platform_value) != {"system", "machine", "kernel_release"}
            or any(
                not isinstance(platform_value[field], str)
                or not platform_value[field]
                for field in platform_value
            )
        ):
            raise ValueError("Executor HIP platform authority differs")
        _python_authority(host["python"], "Executor HIP Python")
        packages = _package_authority(host["packages"], "Executor HIP package")
        if set(packages) != {
            "packaging",
            "pybind11",
            "psutil",
            "setuptools",
            "torch",
            "triton",
        }:
            raise ValueError("Executor HIP package set differs")
        runtime = host["runtime"]
        if (
            not isinstance(runtime, Mapping)
            or set(runtime) != {
                "backend",
                "torch_hip_version",
                "visible_device_count",
            }
            or runtime.get("backend") != "hip"
            or not isinstance(runtime.get("torch_hip_version"), str)
            or not runtime["torch_hip_version"]
            or runtime.get("visible_device_count") != 1
        ):
            raise ValueError("Executor HIP runtime authority differs")
        tools = host["tools"]
        if not isinstance(tools, Mapping) or set(tools) != {
            "build_tools",
            "device_monitor",
            "profilers",
        }:
            raise ValueError("Executor HIP tool authority differs")
        monitor = _executable_record(
            tools["device_monitor"], "Executor HIP device monitor"
        )
        if monitor["kind"] != "amd-smi":
            raise ValueError("Executor HIP device monitor kind differs")
        profilers = tools["profilers"]
        if not isinstance(profilers, list):
            raise ValueError("Executor HIP profiler authority differs")
        seen: set[str] = set()
        for index, value in enumerate(profilers):
            profiler = _executable_record(
                value, f"Executor HIP profilers[{index}]"
            )
            kind = cast(str, profiler["kind"])
            if kind not in {"rocprofv3", "rocprof", "omniperf"} or kind in seen:
                raise ValueError("Executor HIP profiler kind differs")
            seen.add(kind)
        build_tools = tools["build_tools"]
        if not isinstance(build_tools, list):
            raise ValueError("Executor HIP build-tool authority differs")
        seen_build_tools: set[str] = set()
        for index, value in enumerate(build_tools):
            tool = _executable_record(
                value, f"Executor HIP build_tools[{index}]"
            )
            kind = cast(str, tool["kind"])
            if kind in seen_build_tools:
                raise ValueError("Executor HIP build-tool kind differs")
            seen_build_tools.add(kind)
        if seen_build_tools != {
            "cxx",
            "git",
            "hipcc",
            "hipconfig",
            "ninja",
            "rocminfo",
            "sh",
        }:
            raise ValueError("Executor HIP build-tool set differs")

    @property
    def reference(self) -> Mapping[str, str]:
        """Return the exact Study/Campaign reference for this revision."""

        return MappingProxyType(
            {
                "path": self.relative_path,
                "canonical_sha256": self.canonical_sha256,
                "executor_id": self.executor_id,
            }
        )

    @staticmethod
    def _admit_python_and_packages(host: Mapping[str, object]) -> None:
        python = _python_authority(host["python"], "Executor Python")
        expected_invocation = Path(str(python["invocation_path"])).absolute()
        observed_invocation = Path(sys.executable).absolute()
        if (
            observed_invocation != expected_invocation
            or sys.version.split()[0] != python["version"]
            or sha256(observed_invocation.resolve(strict=True).read_bytes()).hexdigest()
            != python["resolved_sha256"]
        ):
            raise ValueError("Executor Python runtime differs")
        packages = _package_authority(host["packages"], "Executor package")
        for distribution, expected in packages.items():
            if importlib.metadata.version(distribution) != expected:
                raise ValueError(f"Executor package {distribution!r} differs")

    def admit_host(self) -> object:
        """Verify the pinned B200 host and return its admitted CUPTI helper."""

        if self.document["schema_version"] != 1:
            raise ValueError("B200 host admission requires Executor schema v1")
        host = cast(Mapping[str, object], self.document["host_environment"])
        self._admit_python_and_packages(host)

        cupti = cast(Mapping[str, object], host["cupti_python"])
        site = Path(str(cupti["site_packages_path"])).resolve(strict=True)
        if not site.is_dir() or site.is_symlink():
            raise ValueError("Executor CUPTI site-packages custody differs")
        for index, value in enumerate(cast(list[object], cupti["files"])):
            record = _file_record(value, f"executor.cupti_python.files[{index}]")
            path = _external_file(site, record["path"], f"executor.cupti.files[{index}]")
            payload = path.read_bytes()
            if (
                sha256(payload).hexdigest() != record["sha256"]
                or len(payload) != record["size_bytes"]
            ):
                raise ValueError("Executor CUPTI runtime file differs")
        if str(site) not in sys.path:
            sys.path.append(str(site))
        if importlib.metadata.version(str(cupti["distribution"])) != cupti["version"]:
            raise ValueError("Executor CUPTI distribution differs")
        from cupti import cupti as cupti_extension

        if cupti_extension is None:
            raise ValueError("Executor CUPTI extension is unavailable")

        helper = cast(Mapping[str, object], host["flashinfer_helper"])
        helper_path = Path(str(helper["path"]))
        if helper_path.is_symlink() or not helper_path.is_file():
            raise ValueError("Executor FlashInfer helper custody differs")
        helper_payload = helper_path.read_bytes()
        if (
            sha256(helper_payload).hexdigest() != helper["sha256"]
            or len(helper_payload) != helper["size_bytes"]
            or importlib.metadata.version(str(helper["distribution"])) != helper["version"]
        ):
            raise ValueError("Executor FlashInfer helper differs")
        module_spec = importlib.util.spec_from_file_location(
            "open_cake_ir_pinned_flashinfer_testing_utils", helper_path
        )
        if module_spec is None or module_spec.loader is None:
            raise ValueError("Executor FlashInfer helper specification failed")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        required = (
            "bench_gpu_time_with_cupti",
            "bench_gpu_time_with_cuda_event",
            "bench_gpu_time_with_cudagraph",
        )
        if any(not callable(getattr(module, name, None)) for name in required):
            raise ValueError("Executor FlashInfer helper surface differs")
        return module

    def admit_hip_host(self) -> HipHostAdmission:
        """Admit one exact schema-v2 HIP host before device-specific execution."""

        if self.document["schema_version"] != 2:
            raise ValueError("HIP host admission requires Executor schema v2")
        host = cast(Mapping[str, object], self.document["host_environment"])
        if host.get("runtime_kind") != "hip":
            raise ValueError("Executor runtime kind is not HIP")
        self._admit_python_and_packages(host)

        expected_platform = cast(Mapping[str, object], host["platform"])
        observed_platform = {
            "system": platform.system(),
            "machine": platform.machine(),
            "kernel_release": platform.release(),
        }
        if observed_platform != dict(expected_platform):
            raise ValueError("Executor HIP platform differs")

        runtime = cast(Mapping[str, object], host["runtime"])
        torch = importlib.import_module("torch")
        torch_version = getattr(torch, "version", None)
        observed_hip = getattr(torch_version, "hip", None)
        if (
            not isinstance(observed_hip, str)
            or observed_hip != runtime["torch_hip_version"]
        ):
            raise ValueError("Executor HIP runtime differs")

        tools = cast(Mapping[str, object], host["tools"])
        monitor = _admit_executable(
            tools["device_monitor"], "Executor HIP device monitor"
        )
        profilers = tuple(
            _admit_executable(value, f"Executor HIP profilers[{index}]")
            for index, value in enumerate(cast(tuple[object, ...], tools["profilers"]))
        )
        admitted_build_tools = tuple(
            _admit_executable(value, f"Executor HIP build_tools[{index}]")
            for index, value in enumerate(
                cast(tuple[object, ...], tools["build_tools"])
            )
        )
        build_tools = MappingProxyType(
            {cast(str, value["kind"]): value for value in admitted_build_tools}
        )
        return HipHostAdmission(
            executor_id=self.executor_id,
            torch_hip_version=cast(str, runtime["torch_hip_version"]),
            visible_device_count=cast(int, runtime["visible_device_count"]),
            device_monitor=monitor,
            profilers=profilers,
            build_tools=build_tools,
        )

    def admit_profiler(self) -> Mapping[str, object]:
        """Verify and return the optional exact NCU executable for attribution."""

        host = cast(Mapping[str, object], self.document["host_environment"])
        if "nsight_compute" not in host:
            raise ValueError("Executor Revision does not pin Nsight Compute")
        profiler = cast(Mapping[str, object], host["nsight_compute"])
        path = Path(str(profiler["path"]))
        if path.is_symlink() or not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError("Executor Nsight Compute custody differs")
        payload = path.read_bytes()
        if (
            sha256(payload).hexdigest() != profiler["sha256"]
            or len(payload) != profiler["size_bytes"]
        ):
            raise ValueError("Executor Nsight Compute bytes differ")
        return MappingProxyType(dict(profiler))
