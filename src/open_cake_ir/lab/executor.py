"""Executor identity: one clean source commit and the captured host that runs it."""

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
from typing import Callable, Mapping, cast

from open_cake_ir.serialization import canonical_json_bytes as _canonical_json_bytes
from open_cake_ir.source_identity import SourceIdentityError, checkout_commit, untracked_paths

from ._documents import differs


HIP_PACKAGES = frozenset({"packaging", "pybind11", "psutil", "setuptools", "torch", "triton"})
HIP_BUILD_TOOLS = frozenset({"cxx", "git", "hipcc", "hipconfig", "ninja", "rocminfo", "sh"})
HIP_PROFILERS = frozenset({"rocprofv3", "rocprof", "omniperf"})
# The device monitor is a declared kind for the same reason the build tools and profilers
# above are: a ROCm host ships amd-smi, a Hygon DTK host ships rocm-smi and hy-smi, and a
# record that pinned one name would either refuse the other host or, worse, carry its
# binary under a label naming a tool that is not installed.
HIP_DEVICE_MONITORS = frozenset({"amd-smi", "rocm-smi", "hy-smi"})


HIP_RUNTIME_LIBRARIES = frozenset({"libxml2.so.2"})


_ENVIRONMENT_NAME = re.compile(r"[A-Z][A-Z0-9_]*")


def _build_environment(value: object) -> dict[str, str]:
    """Admit the environment a host's toolchain needs inside the isolated build jail.

    The jail runs --clearenv, which is correct: it must not inherit whatever the invoking
    shell exported. That costs a CUDA host nothing, because torch finds its libraries
    through RPATH and nvcc needs no variable. It is fatal on the Hygon DTK host, where
    two separate facts live only in /opt/dtk/env.sh -- LD_LIBRARY_PATH, without which
    libgalaxyhip.so.5 is present and unfindable, and ROCM_PATH, without which clang-18
    reports "cannot find ROCm device library" and the hcu backend's own path_to_rocm()
    falls back to a directory that is not there.

    Both are the same kind of fact, so they are one declaration rather than one field
    each. A value's absolute-path components are checked against the jail's mounts by the
    compiler that consumes this, so a declared path can never name something the jail
    cannot see.
    """
    if not isinstance(value, Mapping) or not value:
        return {}
    admitted: dict[str, str] = {}
    for name, item in value.items():
        if (not isinstance(name, str) or _ENVIRONMENT_NAME.fullmatch(name) is None
                or not isinstance(item, str) or not item or item != item.strip()):
            return {}
        if any(part.endswith("/") or ".." in Path(part).parts
               for part in item.split(":") if part.startswith("/")):
            return {}
        admitted[name] = item
    return admitted


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
        raise differs(f"{context} digest", expected="<64 lowercase hex characters>", observed=value)
    return value


def _relative_file(root: Path, value: object, context: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        raise differs(f"{context} path", expected="<non-empty root-relative path>", observed=value)
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "\\" in value:
        raise ValueError(f"{context} path is unsafe")
    unresolved = root
    for part in relative.parts:
        unresolved /= part
        if unresolved.is_symlink():
            raise differs(f"{context} file custody", expected="<no symlink on the path>", observed=str(unresolved))
    path = unresolved.resolve(strict=True)
    if root not in path.parents or not path.is_file():
        raise differs(f"{context} file custody", expected=f"<regular file under {root}>", observed=str(path))
    return value, path


def _external_file(root: Path, value: object, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise differs(f"{context} path", expected="<non-empty root-relative path>", observed=value)
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "\\" in value:
        raise ValueError(f"{context} path is unsafe")
    unresolved = root
    for part in relative.parts:
        unresolved /= part
        if unresolved.is_symlink():
            raise differs(f"{context} file custody", expected="<no symlink on the path>", observed=str(unresolved))
    path = unresolved.resolve(strict=True)
    if root not in path.parents or not path.is_file():
        raise differs(f"{context} file custody", expected=f"<regular file under {root}>", observed=str(path))
    return path


def _file_record(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "size_bytes"}:
        raise differs(
            f"{context} fields differ", expected=["path", "sha256", "size_bytes"],
            observed=sorted(value) if isinstance(value, Mapping) else value,
        )
    _digest(value["sha256"], f"{context}.sha256")
    if (
        not isinstance(value["size_bytes"], int)
        or isinstance(value["size_bytes"], bool)
        or value["size_bytes"] <= 0
    ):
        raise differs(f"{context}.size_bytes", expected="<positive int>", observed=value["size_bytes"])
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
        raise differs(
            f"{context} authority",
            expected={"invocation_path": "<absolute path>", "version": "<non-empty>",
                      "resolved_sha256": "<digest>"},
            observed=value,
        )
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
        raise differs(
            f"{context} requirements differ", expected="<non-empty {distribution: version} of strings>",
            observed=value,
        )
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
        raise differs(
            f"{context} authority",
            expected={"kind": "<non-empty>", "path": "<absolute path>", "version": "<non-empty>",
                      "sha256": "<digest>", "size_bytes": "<positive int>"},
            observed=value,
        )
    _digest(value["sha256"], f"{context}.sha256")
    return cast(Mapping[str, object], value)


def _admit_executable(
    value: object, context: str
) -> Mapping[str, object]:
    record = _executable_record(value, context)
    unresolved = Path(str(record["path"]))
    if not unresolved.is_file() or not os.access(unresolved, os.X_OK):
        raise differs(f"{context} custody", expected="<executable regular file>", observed=str(unresolved))
    path = unresolved.resolve(strict=True)
    payload = path.read_bytes()
    if (
        sha256(payload).hexdigest() != record["sha256"]
        or len(payload) != record["size_bytes"]
    ):
        raise differs(
            f"{context} bytes differ",
            expected={"sha256": record["sha256"], "size_bytes": record["size_bytes"]},
            observed={"sha256": sha256(payload).hexdigest(), "size_bytes": len(payload)},
        )
    return MappingProxyType(dict(record))


def _shared_library_record(value: object, context: str) -> Mapping[str, object]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"soname", "path", "sha256", "size_bytes"}
        or not isinstance(value["soname"], str)
        or not value["soname"]
        or not isinstance(value["path"], str)
        or not Path(value["path"]).is_absolute()
        or not isinstance(value["size_bytes"], int)
        or isinstance(value["size_bytes"], bool)
        or value["size_bytes"] <= 0
    ):
        raise differs(
            f"{context} authority",
            expected={"soname": "<non-empty>", "path": "<absolute path>", "sha256": "<digest>",
                      "size_bytes": "<positive int>"},
            observed=value,
        )
    _digest(value["sha256"], f"{context}.sha256")
    return cast(Mapping[str, object], value)


def _admit_shared_library(value: object, context: str) -> Mapping[str, object]:
    record = _shared_library_record(value, context)
    unresolved = Path(str(record["path"]))
    if not unresolved.is_file():
        raise differs(f"{context} custody", expected="<regular file>", observed=str(unresolved))
    payload = unresolved.resolve(strict=True).read_bytes()
    if (
        sha256(payload).hexdigest() != record["sha256"]
        or len(payload) != record["size_bytes"]
    ):
        raise differs(
            f"{context} bytes differ",
            expected={"sha256": record["sha256"], "size_bytes": record["size_bytes"]},
            observed={"sha256": sha256(payload).hexdigest(), "size_bytes": len(payload)},
        )
    return MappingProxyType(dict(record))


@dataclass(frozen=True)
class HipHostAdmission:
    """Exact static ROCm host facts admitted before device-specific execution."""

    executor_id: str
    torch_hip_version: str
    visible_device_count: int
    build_environment: Mapping[str, str]
    device_monitor: Mapping[str, object]
    profilers: tuple[Mapping[str, object], ...]
    build_tools: Mapping[str, Mapping[str, object]]
    runtime_libraries: Mapping[str, Mapping[str, object]]


HOSTS_DIRECTORY = "runtime/hosts"


@dataclass(frozen=True)
class ExecutorRevision:
    """One clean source commit and the captured host that runs it (ADR 0065).

    The commit binds every tracked source byte, so there is no per-file closure to
    re-release when a shared file changes, and a change never retires another host. The
    host capture is committed under `runtime/hosts/<target>.json` and changes only when
    the host does. `executor_id` is `<target>@<commit>`, and the capture is a tracked
    file at that commit, so the id is the whole identity.
    """

    executor_id: str
    document: Mapping[str, object]
    project_root: Path
    relative_path: str

    @classmethod
    def load_reference(
        cls, project_root: str | Path, reference: object, context: str
    ) -> "ExecutorRevision":
        """Verify one exact Executor reference against this checkout and host capture.

        Host admission is a separate live-execution boundary. Every producer,
        worker and replay consumer calls this validator independently.
        """
        if not isinstance(reference, Mapping) or set(reference) != {"path", "executor_id"}:
            raise differs(
                f"{context} fields differ", expected=["executor_id", "path"],
                observed=sorted(reference) if isinstance(reference, Mapping) else reference,
            )
        root = Path(project_root).resolve(strict=True)
        _, path = _relative_file(root, reference["path"], context)
        revision = cls.load(root, path)
        if revision.executor_id != reference["executor_id"]:
            raise ValueError(
                f"{context} Executor Revision differs: the reference pins "
                f"{reference['executor_id']!r} and this checkout provides "
                f"{revision.executor_id!r}"
            )
        return revision

    @classmethod
    def for_target(cls, project_root: str | Path, target: object) -> "ExecutorRevision":
        """Return the Executor for one exact target from its committed host capture."""

        if not isinstance(target, str) or not target:
            raise ValueError("current Executor resolution requires an exact target")
        root = Path(project_root).resolve(strict=True)
        path = root / HOSTS_DIRECTORY / f"{target}.json"
        if not path.is_file():
            raise ValueError(
                f"no host capture is published for exact target {target!r}; capture one "
                "with tools/capture_executor_host.py and commit it"
            )
        return cls.load(root, path)

    @classmethod
    def load(cls, project_root: str | Path, path: str | Path) -> "ExecutorRevision":
        """Load one committed host capture and bind it to the checkout's clean commit."""

        root = Path(project_root).resolve(strict=True)
        unresolved_source = Path(path)
        if unresolved_source.is_symlink():
            raise differs("Executor host capture custody", expected="<no symlink>", observed=str(unresolved_source))
        source = unresolved_source.resolve(strict=True)
        try:
            relative_source = source.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError("Executor host capture escapes the project root") from error
        if (PurePosixPath(relative_source).parent != PurePosixPath(HOSTS_DIRECTORY)
                or source.suffix != ".json"):
            raise ValueError(
                f"an Executor host capture lives at {HOSTS_DIRECTORY}/<target>.json"
            )
        target = source.stem
        document = json.loads(source.read_text(encoding="utf-8"))
        if (
            not isinstance(document, Mapping)
            or set(document) != {"schema_version", "target", "host_environment"}
            or type(document.get("schema_version")) is not int
            or document["schema_version"] != 1
            or document.get("target") != target
        ):
            raise differs(
                "Executor host capture fields, schema or target differ",
                expected={"fields": ["host_environment", "schema_version", "target"],
                          "schema_version": 1, "target": target},
                observed=({"fields": sorted(document), "schema_version": document.get("schema_version"),
                           "target": document.get("target")}
                          if isinstance(document, Mapping) else document),
            )
        host = cast(Mapping[str, object], document["host_environment"])
        cls._validate_host_document(host)
        captured = _host_kind(host).captured_target(host)
        if captured is not None and captured != target:
            raise ValueError(f"{host['kind']} host capture describes another target")
        try:
            commit = checkout_commit(root)
        except SourceIdentityError as error:
            raise ValueError(
                f"the Executor for {target!r} requires a clean committed checkout: {error}"
            ) from error
        if untracked_paths(root, (relative_source,)):
            raise ValueError(
                f"the Executor for {target!r} reads {relative_source}, which commit "
                f"{commit} does not track"
            )
        executor_id = f"{target}@{commit}"
        detached = cast(
            Mapping[str, object],
            _freeze_json(json.loads(_canonical_json_bytes({
                "schema_version": 1,
                "executor_id": executor_id,
                "commit": commit,
                "target": target,
                "host_environment": host,
            }))),
        )
        return cls(
            executor_id=executor_id,
            document=detached,
            project_root=root,
            relative_path=relative_source,
        )

    @staticmethod
    def _validate_host_document(host: Mapping[str, object]) -> None:
        """Validate one captured host by the kind it declares.

        A host declaring no kind is the explicitly named pre-kind CUDA form: the None
        row of `_HOST_VALIDATORS`, never a fall-through for a kind this module does not
        know.
        """
        if not isinstance(host, Mapping):
            raise differs("Executor host environment fields differ", expected="<object>", observed=host)
        _host_kind(host).validate(host)

    @staticmethod
    def _validate_cuda_host_document(host: Mapping[str, object]) -> None:
        """The pre-kind CUDA form; runtime/hosts/sm_103a.json still carries it (D10)."""
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
            raise differs(
                "Executor host environment fields differ",
                expected=[sorted(legacy_fields), sorted(legacy_fields | {"nsight_compute"})],
                observed=sorted(host),
            )
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
            or not isinstance(cupti["files"], (list, tuple))
            or not cupti["files"]
        ):
            raise differs(
                "Executor CUPTI Python authority",
                expected={"site_packages_path": "<absolute path>", "distribution": "<str>",
                          "version": "<str>", "files": "<non-empty list>"},
                observed=cupti,
            )
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
            raise differs(
                "Executor FlashInfer helper authority",
                expected={"path": "<absolute path>", "distribution": "<str>", "version": "<str>",
                          "sha256": "<digest>", "size_bytes": "<positive int>"},
                observed=helper,
            )
        _digest(helper["sha256"], "executor.flashinfer_helper.sha256")
        if (
            not isinstance(helper["size_bytes"], int)
            or isinstance(helper["size_bytes"], bool)
            or helper["size_bytes"] <= 0
        ):
            raise differs("Executor FlashInfer helper size", expected="<positive int>", observed=helper["size_bytes"])
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
                raise differs(
                    "Executor Nsight Compute authority",
                    expected={"path": "<absolute path>", "version": "<non-empty>", "sha256": "<digest>",
                              "size_bytes": "<positive int>"},
                    observed=profiler,
                )
            _digest(profiler["sha256"], "executor.nsight_compute.sha256")

    @staticmethod
    def _validate_hip_host_document(host: Mapping[str, object]) -> None:
        hip_fields = {"kind", "platform", "python", "packages", "runtime", "runtime_libraries", "tools"}
        if not isinstance(host, Mapping) or set(host) != hip_fields or host.get("kind") != "hip":
            raise differs(
                "Executor HIP host environment fields differ",
                expected={"kind": "hip", "fields": sorted(hip_fields)},
                observed=({"kind": host.get("kind"), "fields": sorted(host)}
                          if isinstance(host, Mapping) else host),
            )
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
            raise differs(
                "Executor HIP platform authority",
                expected={"system": "<non-empty>", "machine": "<non-empty>", "kernel_release": "<non-empty>"},
                observed=platform_value,
            )
        _python_authority(host["python"], "Executor HIP Python")
        packages = _package_authority(host["packages"], "Executor HIP package")
        if set(packages) != HIP_PACKAGES:
            raise differs("Executor HIP package set", expected=sorted(HIP_PACKAGES), observed=sorted(packages))
        runtime = host["runtime"]
        if (
            not isinstance(runtime, Mapping)
            or set(runtime) != {
                "backend",
                "build_environment",
                "torch_hip_version",
                "visible_device_count",
            }
            or runtime.get("backend") != "hip"
            or not isinstance(runtime.get("torch_hip_version"), str)
            or not runtime["torch_hip_version"]
            or type(runtime.get("visible_device_count")) is not int
            or runtime["visible_device_count"] != 1
            or not _build_environment(runtime.get("build_environment"))
        ):
            raise differs(
                "Executor HIP runtime authority",
                expected={"backend": "hip", "torch_hip_version": "<non-empty>", "visible_device_count": 1,
                          "build_environment": "<non-empty {NAME: value}>"},
                observed=runtime,
            )
        tools = host["tools"]
        if not isinstance(tools, Mapping) or set(tools) != {
            "build_tools",
            "device_monitor",
            "profilers",
        }:
            raise differs(
                "Executor HIP tool authority", expected=["build_tools", "device_monitor", "profilers"],
                observed=sorted(tools) if isinstance(tools, Mapping) else tools,
            )
        monitor = _executable_record(
            tools["device_monitor"], "Executor HIP device monitor"
        )
        if monitor["kind"] not in HIP_DEVICE_MONITORS:
            raise differs(
                "Executor HIP device monitor kind", expected=sorted(HIP_DEVICE_MONITORS), observed=monitor["kind"],
            )
        profilers = tools["profilers"]
        if not isinstance(profilers, (list, tuple)):
            raise differs("Executor HIP profiler authority", expected="<list of records>", observed=profilers)
        seen: set[str] = set()
        for index, value in enumerate(profilers):
            profiler = _executable_record(
                value, f"Executor HIP profilers[{index}]"
            )
            kind = cast(str, profiler["kind"])
            if kind not in HIP_PROFILERS or kind in seen:
                raise differs(
                    "Executor HIP profiler kind",
                    expected=f"one of {sorted(HIP_PROFILERS)}, not yet declared in {sorted(seen)}",
                    observed=kind,
                )
            seen.add(kind)
        build_tools = tools["build_tools"]
        if not isinstance(build_tools, (list, tuple)):
            raise differs("Executor HIP build-tool authority", expected="<list of records>", observed=build_tools)
        seen_build_tools: set[str] = set()
        for index, value in enumerate(build_tools):
            tool = _executable_record(
                value, f"Executor HIP build_tools[{index}]"
            )
            kind = cast(str, tool["kind"])
            if kind in seen_build_tools:
                raise differs(
                    "Executor HIP build-tool kind", expected=f"not yet declared in {sorted(seen_build_tools)}",
                    observed=kind,
                )
            seen_build_tools.add(kind)
        if seen_build_tools != HIP_BUILD_TOOLS:
            raise differs(
                "Executor HIP build-tool set", expected=sorted(HIP_BUILD_TOOLS), observed=sorted(seen_build_tools),
            )
        runtime_libraries = host["runtime_libraries"]
        if not isinstance(runtime_libraries, (list, tuple)):
            raise differs(
                "Executor HIP runtime-library authority", expected="<list of records>", observed=runtime_libraries,
            )
        seen_libraries: set[str] = set()
        for index, value in enumerate(runtime_libraries):
            library = _shared_library_record(
                value, f"Executor HIP runtime_libraries[{index}]"
            )
            soname = cast(str, library["soname"])
            if soname in seen_libraries:
                raise differs(
                    "Executor HIP runtime-library soname", expected=f"not yet declared in {sorted(seen_libraries)}",
                    observed=soname,
                )
            seen_libraries.add(soname)
        if seen_libraries != HIP_RUNTIME_LIBRARIES:
            raise differs(
                "Executor HIP runtime-library set", expected=sorted(HIP_RUNTIME_LIBRARIES),
                observed=sorted(seen_libraries),
            )

    @property
    def reference(self) -> Mapping[str, str]:
        """Return the exact Study/Campaign reference for this revision."""

        return MappingProxyType(
            {"path": self.relative_path, "executor_id": self.executor_id}
        )

    def admit_host(self) -> object:
        """Verify the captured host and return its admission, for the kinds that admit here."""

        host = cast(Mapping[str, object], self.document["host_environment"])
        row = _host_kind(host)
        if row.admits_through != "admit_host":
            raise ValueError(f"a {host['kind']} host is admitted through {row.admits_through}")
        return admit_host_environment(host)

    def admit_hip_host(self) -> HipHostAdmission:
        """Admit the captured software host; exact device admission follows lowering."""

        host = cast(Mapping[str, object], self.document["host_environment"])
        if _host_kind(host).admits_through != "admit_hip_host":
            raise ValueError("HIP host admission requires a HIP host capture")
        return cast(HipHostAdmission, admit_host_environment(host, executor_id=self.executor_id))

    def admit_profiler(self) -> Mapping[str, object]:
        """Verify and return the optional exact NCU executable for attribution."""

        return admit_profiler_environment(
            cast(Mapping[str, object], self.document["host_environment"])
        )


def _admit_python_and_packages(host: Mapping[str, object]) -> None:
    python = _python_authority(host["python"], "Executor Python")
    expected_invocation = Path(str(python["invocation_path"])).absolute()
    observed_invocation = Path(sys.executable).absolute()
    observed_python = {
        "invocation_path": str(observed_invocation), "version": sys.version.split()[0],
        "resolved_sha256": sha256(observed_invocation.resolve(strict=True).read_bytes()).hexdigest(),
    }
    if (
        observed_invocation != expected_invocation
        or observed_python["version"] != python["version"]
        or observed_python["resolved_sha256"] != python["resolved_sha256"]
    ):
        raise differs("Executor Python runtime", expected=dict(python), observed=observed_python)
    packages = _package_authority(host["packages"], "Executor package")
    for distribution, expected in packages.items():
        observed_version = importlib.metadata.version(distribution)
        if observed_version != expected:
            raise differs(f"Executor package {distribution!r}", expected=expected, observed=observed_version)


def admit_host_environment(
    host: Mapping[str, object], *, executor_id: str = "",
) -> object:
    """Validate and admit one captured software host through the boundary, by its kind."""

    if not isinstance(host, Mapping):
        raise differs("Executor host environment fields differ", expected="<object>", observed=host)
    ExecutorRevision._validate_host_document(host)
    return _host_kind(host).admit(host, executor_id=executor_id)


def _admit_cuda_environment(host: Mapping[str, object], *, executor_id: str = "") -> object:
    """Admit the pre-kind CUDA host: interpreter, packages, CUPTI and the timing helper."""

    _admit_python_and_packages(host)
    cupti = cast(Mapping[str, object], host["cupti_python"])
    site = Path(str(cupti["site_packages_path"])).resolve(strict=True)
    if not site.is_dir() or site.is_symlink():
        raise differs("Executor CUPTI site-packages custody", expected="<directory, no symlink>", observed=str(site))
    for index, value in enumerate(cast(list[object], cupti["files"])):
        record = _file_record(value, f"executor.cupti_python.files[{index}]")
        path = _external_file(site, record["path"], f"executor.cupti.files[{index}]")
        payload = path.read_bytes()
        if (
            sha256(payload).hexdigest() != record["sha256"]
            or len(payload) != record["size_bytes"]
        ):
            raise differs(
                f"Executor CUPTI runtime file {record['path']!r}",
                expected={"sha256": record["sha256"], "size_bytes": record["size_bytes"]},
                observed={"sha256": sha256(payload).hexdigest(), "size_bytes": len(payload)},
            )
    if str(site) not in sys.path:
        sys.path.append(str(site))
    observed_cupti_version = importlib.metadata.version(str(cupti["distribution"]))
    if observed_cupti_version != cupti["version"]:
        raise differs(
            f"Executor CUPTI distribution {cupti['distribution']!r} version",
            expected=cupti["version"], observed=observed_cupti_version,
        )
    from cupti import cupti as cupti_extension

    if cupti_extension is None:
        raise ValueError("Executor CUPTI extension is unavailable")

    helper = cast(Mapping[str, object], host["flashinfer_helper"])
    helper_path = Path(str(helper["path"]))
    if helper_path.is_symlink() or not helper_path.is_file():
        raise differs("Executor FlashInfer helper custody", expected="<regular file, no symlink>", observed=str(helper_path))
    helper_payload = helper_path.read_bytes()
    observed_helper = {
        "sha256": sha256(helper_payload).hexdigest(), "size_bytes": len(helper_payload),
        "version": importlib.metadata.version(str(helper["distribution"])),
    }
    expected_helper = {key: helper[key] for key in ("sha256", "size_bytes", "version")}
    if observed_helper != expected_helper:
        raise differs("Executor FlashInfer helper", expected=expected_helper, observed=observed_helper)
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
        raise differs(
            "Executor FlashInfer helper surface", expected=list(required),
            observed=[name for name in required if callable(getattr(module, name, None))],
        )
    return module


def admit_profiler_environment(host: Mapping[str, object]) -> Mapping[str, object]:
    """Verify the declared native observation executable for attribution.

    Each host kind names its own observer: Metal's native observer executable, CUDA's
    Nsight Compute. Falling through to the NCU branch reports a Metal host as one that
    "does not pin Nsight Compute", which is a refusal in a vendor the caller never named.
    A kind whose row names no separately admitted profiler takes attribution inside
    evaluation (its platform row says so) and is refused as that kind.
    """

    row = _host_kind(host)
    if row.admit_profiler is None:
        raise ValueError(
            f"Executor host kind {host.get('kind')!r} takes attribution inside evaluation "
            "and names no separately admitted profiler; the Executor Revision does not "
            "pin Nsight Compute"
        )
    return row.admit_profiler(host)


def _admit_cuda_profiler(host: Mapping[str, object]) -> Mapping[str, object]:
    if "nsight_compute" not in host:
        raise ValueError("Executor Revision does not pin Nsight Compute")
    profiler = cast(Mapping[str, object], host["nsight_compute"])
    path = Path(str(profiler["path"]))
    if path.is_symlink() or not path.is_file() or not os.access(path, os.X_OK):
        raise differs("Executor Nsight Compute custody", expected="<executable regular file, no symlink>", observed=str(path))
    payload = path.read_bytes()
    if (
        sha256(payload).hexdigest() != profiler["sha256"]
        or len(payload) != profiler["size_bytes"]
    ):
        raise differs(
            "Executor Nsight Compute bytes differ",
            expected={"sha256": profiler["sha256"], "size_bytes": profiler["size_bytes"]},
            observed={"sha256": sha256(payload).hexdigest(), "size_bytes": len(payload)},
        )
    return MappingProxyType(dict(profiler))


def _admit_hip_environment(
    host: Mapping[str, object], *, executor_id: str,
) -> HipHostAdmission:
    """Verify HIP software facts without querying or initializing a device."""

    _admit_python_and_packages(host)
    expected_platform = cast(Mapping[str, object], host["platform"])
    observed_platform = {
        "system": platform.system(),
        "machine": platform.machine(),
        "kernel_release": platform.release(),
    }
    if observed_platform["system"] != "Linux" or observed_platform != dict(expected_platform):
        raise differs("Executor HIP platform", expected=dict(expected_platform), observed=observed_platform)

    runtime = cast(Mapping[str, object], host["runtime"])
    torch = importlib.import_module("torch")
    torch_version = getattr(torch, "version", None)
    observed_hip = getattr(torch_version, "hip", None)
    if (
        not isinstance(observed_hip, str)
        or observed_hip != runtime["torch_hip_version"]
        or getattr(torch_version, "cuda", None) is not None
    ):
        raise differs(
            "Executor HIP runtime",
            expected={"torch.version.hip": runtime["torch_hip_version"], "torch.version.cuda": None},
            observed={"torch.version.hip": observed_hip, "torch.version.cuda": getattr(torch_version, "cuda", None)},
        )

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
    admitted_libraries = tuple(
        _admit_shared_library(
            value, f"Executor HIP runtime_libraries[{index}]"
        )
        for index, value in enumerate(
            cast(tuple[object, ...], host["runtime_libraries"])
        )
    )
    runtime_libraries = MappingProxyType(
        {cast(str, value["soname"]): value for value in admitted_libraries}
    )
    return HipHostAdmission(
        executor_id=executor_id,
        torch_hip_version=cast(str, runtime["torch_hip_version"]),
        visible_device_count=cast(int, runtime["visible_device_count"]),
        build_environment=MappingProxyType(_build_environment(runtime["build_environment"])),
        device_monitor=monitor,
        profilers=profilers,
        build_tools=build_tools,
        runtime_libraries=runtime_libraries,
    )


# metal_host imports this module's record helpers, so its callables are reached lazily
# here rather than named in the table at import time.
def _validate_metal_host_document(host: Mapping[str, object]) -> None:
    from .metal_host import validate_metal_host
    validate_metal_host(host)


def _admit_metal_environment(host: Mapping[str, object], *, executor_id: str = "") -> object:
    # A Metal host owns its whole admission, including a package map that is
    # legitimately empty -- v110 and v112 both carry `"packages": {}`. The shared
    # Python-and-packages check is the CUDA and HIP one and requires a non-empty map,
    # so routing Metal through it refuses a released Executor.
    from .metal_host import admit_metal_host
    return admit_metal_host(host)


def _admit_metal_profiler(host: Mapping[str, object]) -> Mapping[str, object]:
    from .metal_host import admit_metal_executable, validate_metal_host
    validate_metal_host(host)
    return admit_metal_executable(host["observer_executable"], "Metal observer")


def _metal_captured_target(host: Mapping[str, object]) -> str:
    return cast(str, cast(Mapping[str, object], host["host"])["target"])


def _no_captured_target(host: Mapping[str, object]) -> None:
    return None


def _validate_metax_host(host: Mapping[str, object]) -> None:
    from .metax_host import validate_host
    validate_host(host)


def _admit_metax_host(host: Mapping[str, object], *, executor_id: str = "") -> object:
    from .metax_host import admit_host
    return admit_host(host, executor_id=executor_id)


def triton_version(host: Mapping[str, object]) -> str:
    """Module version for a declared provider, or the original triton distribution.

    FlagTree's distribution version is not the Triton API version it supplies. Both
    are captured and admitted by that host; CUDA/HIP captures retain their old form.
    """
    runtime = host.get("runtime", {})
    if isinstance(runtime, Mapping) and "triton_version" in runtime:
        return str(runtime["triton_version"])
    return str(host["packages"]["triton"])


@dataclass(frozen=True)
class HostKind:
    """What one declared host kind owns at the Executor boundary.

    The `kind` a capture declares selects this row; `evaluation.platforms` declares the
    same kinds per code object, and the two sets are held equal by
    `declared_host_kinds()` rather than by either restating the other.
    """

    validate: Callable[[Mapping[str, object]], None]
    # (host, *, executor_id) -> the admission the kind returns.
    admit: Callable[..., object]
    # None: the platform takes attribution inside evaluation (its platform row's
    # `attribution`), so there is no separately admitted profiler to return.
    admit_profiler: Callable[[Mapping[str, object]], Mapping[str, object]] | None
    # The ExecutorRevision method that returns this kind's admission. HIP's carries the
    # executor id into a HipHostAdmission, so it has its own entry.
    admits_through: str
    # The exact target a capture of this kind names inside itself, when it does.
    captured_target: Callable[[Mapping[str, object]], str | None]


_HOST_VALIDATORS: Mapping[str | None, HostKind] = MappingProxyType({
    # The pre-kind CUDA form. runtime/hosts/sm_103a.json carries it and is not recaptured
    # until a B300 session exists (D10); it is a row, not a fall-through.
    None: HostKind(
        validate=ExecutorRevision._validate_cuda_host_document,
        admit=_admit_cuda_environment,
        admit_profiler=_admit_cuda_profiler,
        admits_through="admit_host",
        captured_target=_no_captured_target,
    ),
    "hip": HostKind(
        validate=ExecutorRevision._validate_hip_host_document,
        admit=_admit_hip_environment,
        admit_profiler=None,
        admits_through="admit_hip_host",
        captured_target=_no_captured_target,
    ),
    "metal": HostKind(
        validate=_validate_metal_host_document,
        admit=_admit_metal_environment,
        admit_profiler=_admit_metal_profiler,
        admits_through="admit_host",
        captured_target=_metal_captured_target,
    ),
    "maca": HostKind(
        validate=_validate_metax_host,
        admit=_admit_metax_host,
        admit_profiler=None,
        admits_through="admit_host",
        captured_target=_no_captured_target,
    ),
})


def _host_kind(host: Mapping[str, object]) -> HostKind:
    """The row for the kind a host declares; a kind with no row is refused by name."""
    kind = host.get("kind")
    if not isinstance(kind, (str, type(None))) or kind not in _HOST_VALIDATORS:
        raise ValueError(f"Executor host kind {kind!r} is not admitted")
    return _HOST_VALIDATORS[kind]


def declared_host_kinds() -> frozenset[str | None]:
    """Every host kind this boundary validates; equal to the platform rows' host kinds."""
    return frozenset(_HOST_VALIDATORS)
