"""Exact Metal tensor ABI for sealed binary-archive pipeline launches."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import re
from typing import Mapping

from .artifacts import METAL_TARGETS
from .workload import WorkloadContract

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_COMPILE_OPTIONS = {"language_standard": "metal2.3", "math_mode": "safe",
                    "floating_point_functions": "precise", "fast_math_enabled": False}


def compile_options() -> dict:
    return dict(_COMPILE_OPTIONS)


# Apple family 7 and 8 both admit 32 KiB of threadgroup memory; the Target is the
# authority a builder checks against, this is the manifest's own upper bound.
MAXIMUM_THREADGROUP_BYTES = 32768


def _triple(value, name):
    if (not isinstance(value, list) or len(value) != 3 or
            any(type(item) is not int or not 0 < item <= 4294967295 for item in value)):
        raise ValueError(f"Metal manifest {name} must contain three positive uint32 extents")
    return tuple(value)


@dataclass(frozen=True)
class MetalTensorLaunchManifest:
    workload_sha256: str
    case_id: str
    tensor_abi: tuple[tuple[str, tuple[int, ...], str, str], ...]
    target: str
    kernel_name: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    # A single SIMD group needs no threadgroup storage and declares none, so every
    # manifest written before consecutive groups existed reads back unchanged.
    threadgroup_memory_bytes: int = 0

    @classmethod
    def from_dict(cls, document: object) -> "MetalTensorLaunchManifest":
        fields = {"schema_version", "abi", "workload_sha256", "case_id", "tensor_abi", "target",
                  "kernel_name", "grid", "block", "threadgroup_memory_bytes", "execution_model",
                  "active_threads_per_threadgroup", "compile_options", "archive_miss_policy"}
        if (not isinstance(document, Mapping) or set(document) != fields or
                type(document.get("schema_version")) is not int or document.get("schema_version") != 1 or document.get("abi") != "metal_workload_tensors_v1"):
            raise ValueError("Metal tensor launch manifest fields differ")
        if (not isinstance(document["target"], str) or document["target"] not in METAL_TARGETS or not isinstance(document["workload_sha256"], str)
                or not _DIGEST.fullmatch(document["workload_sha256"])
                or not isinstance(document["case_id"], str) or not document["case_id"]
                or not isinstance(document["kernel_name"], str) or not document["kernel_name"].isascii()
                or not document["kernel_name"].isidentifier()):
            raise ValueError("Metal manifest target, Workload identity or entry point differs")
        grid, block = _triple(document["grid"], "grid"), _triple(document["block"], "block")
        threads = block[0]
        shared = document["threadgroup_memory_bytes"]
        if (block[1:] != (1, 1) or threads % 32 or not 32 <= threads <= 1024
                or type(shared) is not int or not 0 <= shared <= MAXIMUM_THREADGROUP_BYTES
                or (threads == 32) != (shared == 0)
                or document["execution_model"] != "simd_program_tile"
                or type(document["active_threads_per_threadgroup"]) is not int
                or document["active_threads_per_threadgroup"] != threads
                or document["compile_options"] != _COMPILE_OPTIONS
                or type(document["compile_options"].get("fast_math_enabled")) is not bool
                or document["archive_miss_policy"] != "failOnBinaryArchiveMiss"):
            raise ValueError("Metal manifest pipeline or launch commitments differ")
        rows = document["tensor_abi"]
        if not isinstance(rows, list) or not 2 <= len(rows) <= 31:
            raise ValueError("Metal manifest tensor ABI count differs")
        abi = []
        for row in rows:
            if (not isinstance(row, Mapping) or set(row) != {"name", "shape", "dtype", "mode"}
                    or not isinstance(row["name"], str) or not row["name"].isascii() or not row["name"].isidentifier()
                    or row["dtype"] != "fp32" or row["mode"] not in {"input", "output"}
                    or not isinstance(row["shape"], list) or not row["shape"]
                    or any(type(v) is not int or v <= 0 for v in row["shape"])):
                raise ValueError("Metal manifest tensor ABI differs")
            abi.append((row["name"], tuple(row["shape"]), row["dtype"], row["mode"]))
        modes = [row[3] for row in abi]
        if (len({row[0] for row in abi}) != len(abi) or "input" not in modes or "output" not in modes
                or modes != sorted(modes)):
            raise ValueError("Metal manifest tensor ABI order differs")
        return cls(document["workload_sha256"], document["case_id"], tuple(abi),
                   document["target"], document["kernel_name"], grid, block, shared)

    @classmethod
    def for_workload(cls, workload: WorkloadContract, case_id: str, *, target: str,
                     kernel_name: str, grid: list[int], block: list[int],
                     threadgroup_memory_bytes: int = 0) -> "MetalTensorLaunchManifest":
        manifest = cls.from_dict({"schema_version": 1, "abi": "metal_workload_tensors_v1",
            "workload_sha256": workload.canonical_sha256, "case_id": case_id,
            "tensor_abi": [{**asdict(arg), "shape": list(arg.shape)} for arg in workload.tensor_abi(case_id)],
            "target": target, "kernel_name": kernel_name, "grid": grid, "block": block,
            "threadgroup_memory_bytes": threadgroup_memory_bytes, "execution_model": "simd_program_tile",
            "active_threads_per_threadgroup": block[0], "compile_options": compile_options(),
            "archive_miss_policy": "failOnBinaryArchiveMiss"})
        manifest.check_workload(workload, case_id)
        return manifest

    def check_workload(self, workload: WorkloadContract, case_id: str) -> None:
        expected = tuple((arg.name, arg.shape, arg.dtype, arg.mode) for arg in workload.tensor_abi(case_id))
        if (self.workload_sha256 != workload.canonical_sha256 or self.case_id != case_id
                or self.target != workload.target or self.tensor_abi != expected):
            raise ValueError("Metal launch ABI differs from selected Workload")

    def as_dict(self) -> dict:
        return {"schema_version": 1, "abi": "metal_workload_tensors_v1",
            "workload_sha256": self.workload_sha256, "case_id": self.case_id,
            "tensor_abi": [dict(name=n, shape=list(s), dtype=d, mode=m) for n, s, d, m in self.tensor_abi],
            "target": self.target, "kernel_name": self.kernel_name, "grid": list(self.grid), "block": list(self.block),
            "threadgroup_memory_bytes": self.threadgroup_memory_bytes,
            "execution_model": "simd_program_tile", "active_threads_per_threadgroup": self.block[0],
            "compile_options": compile_options(), "archive_miss_policy": "failOnBinaryArchiveMiss"}

    @property
    def canonical_sha256(self) -> str:
        return sha256(json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()

    @property
    def block_threads(self) -> int:
        return self.block[0]
