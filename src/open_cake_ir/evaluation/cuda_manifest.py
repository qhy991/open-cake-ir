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
