"""Closed direct-CUDA source manifest for host-owned launch."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Mapping, cast

from open_cake_ir.compiler.target import cuda_target

MANIFEST_PREFIX = "// CAKE_REPRO_LAUNCH_V1 "
MAX_DYNAMIC_SHARED_MEMORY_BYTES = 232_448
_BASE_FIELDS = {
    "schema_version",
    "abi",
    "target",
    "kernel_name",
    "grid",
    "block",
    "dynamic_shared_memory_bytes",
}
_FIELDS_WITH_HIDDEN = _BASE_FIELDS | {"hidden_null_pointer_parameters"}
_KERNEL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return cast(Mapping[str, object], value)


def _integer(value: object, context: str, minimum: int, maximum: int) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not minimum <= value <= maximum
    ):
        raise ValueError(
            f"{context} must be an integer in [{minimum}, {maximum}]; got {value!r}"
        )
    return value


def _dimensions(
    value: object,
    context: str,
    limits: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{context} must contain exactly three dimensions")
    parsed = tuple(
        _integer(item, f"{context}[{index}]", *limits[index])
        for index, item in enumerate(value)
    )
    return cast(tuple[int, int, int], parsed)


@dataclass(frozen=True)
class CudaKernelSpec:
    """Structural launch limits shared by the distinct public tensor ABIs."""

    target: str
    kernel_name: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    dynamic_shared_memory_bytes: int
    hidden_null_pointer_parameters: int = 0

    @classmethod
    def from_dict(cls, value: object) -> "CudaKernelSpec":
        document = _object(value, "CUDA kernel specification")
        fields = _BASE_FIELDS - {"schema_version", "abi"}
        if set(document) not in (fields, fields | {"hidden_null_pointer_parameters"}):
            raise ValueError("CUDA kernel specification fields differ")
        target = cuda_target(document.get("target"))
        limits = target.resource_limits
        spec = cls(
            target=target.target_id,
            kernel_name=str(document.get("kernel_name")),
            grid=_dimensions(document.get("grid"), "manifest.grid",
                             tuple((1, maximum) for maximum in limits.maximum_grid)),
            block=_dimensions(document.get("block"), "manifest.block",
                              ((1, 1024), (1, 1024), (1, 64))),
            dynamic_shared_memory_bytes=_integer(document.get("dynamic_shared_memory_bytes"),
                "manifest.dynamic_shared_memory_bytes", 0, limits.maximum_shared_memory_bytes),
            hidden_null_pointer_parameters=_integer(document.get("hidden_null_pointer_parameters", 0),
                "manifest.hidden_null_pointer_parameters", 0, 2),
        )
        if _KERNEL_NAME.fullmatch(spec.kernel_name) is None or spec.block_threads > limits.maximum_threads_per_cta:
            raise ValueError("CUDA kernel name or block differs")
        return spec

    @property
    def block_threads(self) -> int:
        return self.block[0] * self.block[1] * self.block[2]


@dataclass(frozen=True)
class CudaLaunchManifest:
    """Exact host-owned launch contract contributed by a direct CUDA candidate."""

    schema_version: int
    abi: str
    target: str
    kernel_name: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    dynamic_shared_memory_bytes: int
    hidden_null_pointer_parameters: int = 0

    @classmethod
    def from_dict(cls, value: object) -> "CudaLaunchManifest":
        document = _object(value, "CUDA launch manifest")
        fields = set(document)
        if fields != _BASE_FIELDS and fields != _FIELDS_WITH_HIDDEN:
            raise ValueError("CUDA launch manifest fields differ")
        spec = CudaKernelSpec.from_dict({key: item for key, item in document.items()
                                         if key not in {"schema_version", "abi"}})
        manifest = cls(
            schema_version=_integer(document.get("schema_version"), "manifest.schema_version", 1, 1),
            abi=str(document.get("abi")),
            **asdict(spec),
        )
        if (
            manifest.abi != "flash_kmeans_assign_v1"
            or manifest.target != "sm_100a"
            or _KERNEL_NAME.fullmatch(manifest.kernel_name) is None
            or manifest.block_threads > 1_024
        ):
            raise ValueError("CUDA launch manifest ABI, target, kernel or block differs")
        return manifest

    @property
    def block_threads(self) -> int:
        return self.block[0] * self.block[1] * self.block[2]

    def as_dict(self) -> dict[str, object]:
        document: dict[str, object] = {
            "schema_version": self.schema_version,
            "abi": self.abi,
            "target": self.target,
            "kernel_name": self.kernel_name,
            "grid": list(self.grid),
            "block": list(self.block),
            "dynamic_shared_memory_bytes": self.dynamic_shared_memory_bytes,
        }
        if self.hidden_null_pointer_parameters:
            document["hidden_null_pointer_parameters"] = self.hidden_null_pointer_parameters
        return document

    @property
    def canonical_sha256(self) -> str:
        return sha256(_canonical_json_bytes(self.as_dict())).hexdigest()


def parse_cuda_launch_manifest(source: bytes) -> CudaLaunchManifest:
    """Parse one canonical first-line launch manifest from UTF-8 CUDA source."""

    try:
        text = source.decode("utf-8")
    except UnicodeError as error:
        raise ValueError("CUDA candidate source is not UTF-8") from error
    if not text or "\x00" in text or text.count(MANIFEST_PREFIX) != 1:
        raise ValueError("CUDA candidate source or launch manifest occurrence differs")
    first, separator, _ = text.partition("\n")
    if not separator or not first.startswith(MANIFEST_PREFIX):
        raise ValueError("CUDA launch manifest must be the first source line")
    encoded = first[len(MANIFEST_PREFIX) :]
    try:
        value = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ValueError("CUDA launch manifest JSON is invalid") from error
    manifest = CudaLaunchManifest.from_dict(value)
    if encoded.encode("utf-8") != _canonical_json_bytes(manifest.as_dict()):
        raise ValueError("CUDA launch manifest JSON is not canonical")
    return manifest
