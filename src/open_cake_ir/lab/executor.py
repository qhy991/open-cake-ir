"""Content-bound Executor Revision shared by every Study variant."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping, cast


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
            or document.get("schema_version") != 1
            or document.get("state") != "released"
            or not isinstance(document.get("executor_id"), str)
            or not document["executor_id"]
        ):
            raise ValueError("Executor Revision fields, schema, or state differ")
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
        cls._validate_host_document(cast(Mapping[str, object], host))
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
    def _validate_host_document(host: Mapping[str, object]) -> None:
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
        if (
            not isinstance(python, Mapping)
            or set(python) != {"invocation_path", "version", "resolved_sha256"}
            or not isinstance(python["invocation_path"], str)
            or not Path(python["invocation_path"]).is_absolute()
            or not isinstance(python["version"], str)
        ):
            raise ValueError("Executor Python authority differs")
        _digest(python["resolved_sha256"], "executor.python.resolved_sha256")
        if (
            not isinstance(packages, Mapping)
            or not packages
            or any(
                not isinstance(name, str)
                or not name
                or not isinstance(version, str)
                or not version
                for name, version in packages.items()
            )
        ):
            raise ValueError("Executor package requirements differ")
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

    def admit_host(self) -> object:
        """Verify the pinned host and return its admitted CUPTI helper."""

        return admit_host_environment(
            cast(Mapping[str, object], self.document["host_environment"])
        )

    def admit_profiler(self) -> Mapping[str, object]:
        """Verify and return the optional exact NCU executable for attribution."""

        return admit_profiler_environment(
            cast(Mapping[str, object], self.document["host_environment"])
        )


def admit_host_environment(host: Mapping[str, object]) -> object:
    """Admit a schema-validated host environment and return its CUPTI helper."""

    python = cast(Mapping[str, object], host["python"])
    expected_invocation = Path(str(python["invocation_path"])).absolute()
    observed_invocation = Path(sys.executable).absolute()
    if (
        observed_invocation != expected_invocation
        or sys.version.split()[0] != python["version"]
        or sha256(observed_invocation.resolve(strict=True).read_bytes()).hexdigest()
        != python["resolved_sha256"]
    ):
        raise ValueError("Executor Python runtime differs")
    packages = cast(Mapping[str, object], host["packages"])
    for distribution, expected in packages.items():
        if importlib.metadata.version(distribution) != expected:
            raise ValueError(f"Executor package {distribution!r} differs")

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


def admit_profiler_environment(host: Mapping[str, object]) -> Mapping[str, object]:
    """Verify and return the optional exact NCU executable for attribution."""

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
