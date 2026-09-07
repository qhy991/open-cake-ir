from __future__ import annotations
import json,re
from dataclasses import dataclass,asdict
from hashlib import sha256
from typing import Mapping
from open_cake_ir.evaluation.cuda_manifest import CudaKernelSpec,_canonical_json_bytes,_object,_integer,_BASE_FIELDS,_FIELDS_WITH_HIDDEN,_KERNEL_NAME,MANIFEST_PREFIX

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
