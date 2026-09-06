"""Portable compiled allocation facts, inspected without loading a GPU module.

The CUBIN inspector owns physical allocation. This module validates its projection and
binds it to the source and launch that an author is examining. Stack and local bytes
are static storage facts; neither is a count of dynamic spill traffic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import json
import re
from typing import Mapping


@dataclass(frozen=True)
class CompiledResources:
    source_sha256: str
    cubin_sha256: str
    target: str
    entry_point: str
    threads_per_cta: int
    registers_per_thread: int
    static_shared_bytes: int
    dynamic_shared_bytes: int
    local_bytes: int
    stack_bytes: int
    compiler_version: str
    inspector_version: str

    def __post_init__(self) -> None:
        for identity in (self.source_sha256, self.cubin_sha256):
            if not isinstance(identity, str) or re.fullmatch(r"[0-9a-f]{64}", identity) is None:
                raise ValueError("compiled resource artifact identity differs")
        if any(not isinstance(value, str) or not value.strip() for value in (
            self.target, self.entry_point, self.compiler_version, self.inspector_version
        )):
            raise ValueError("compiled resource target or toolchain identity differs")
        for name in ("threads_per_cta", "registers_per_thread", "static_shared_bytes",
                     "dynamic_shared_bytes", "local_bytes", "stack_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"compiled resource {name} must be a nonnegative integer")
        if not self.threads_per_cta or not self.registers_per_thread:
            raise ValueError("compiled kernel requires nonzero thread and register allocation")

    @property
    def shared_bytes(self) -> int:
        return self.static_shared_bytes + self.dynamic_shared_bytes

    def as_dict(self) -> dict[str, object]:
        return {"schema_version": 1, **asdict(self)}

    @classmethod
    def from_dict(cls, value: object) -> "CompiledResources":
        if not isinstance(value, Mapping):
            raise ValueError("compiled resources must be an object")
        if set(value) != {"schema_version", *cls.__dataclass_fields__} or (
            type(value.get("schema_version")) is not int or value["schema_version"] != 1
        ):
            raise ValueError("compiled resource fields differ")
        return cls(**{name: value[name] for name in cls.__dataclass_fields__})

    def require_context(self, *, source: str, target: str, entry_point: str,
                        threads_per_cta: int) -> None:
        # Exact byte identity is needed here: a portable allocation observation cannot
        # describe a different emitted program merely because its Schedule name matches.
        if (
            sha256(source.encode("utf-8")).hexdigest() != self.source_sha256
            or target != self.target
            or entry_point != self.entry_point
            or threads_per_cta != self.threads_per_cta
        ):
            raise ValueError("compiled resources belong to a different source, target, entry or launch")


def parse_cuobjdump_resources(report: str, entry_point: str) -> dict[str, int]:
    """Read one unambiguous per-function block from cuobjdump --dump-resource-usage.

    https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html documents REG as a
    count and STACK/SHARED/LOCAL as bytes. Unknown fields are not interpreted.
    """
    blocks = re.findall(
        r"^\s*Function\s+([^\s:]+)\s*:\s*\n(.*?)(?=^\s*Function\s+|\Z)",
        report, flags=re.MULTILINE | re.DOTALL,
    )
    matches = [body for name, body in blocks if name == entry_point]
    if len(matches) != 1:
        raise ValueError("CUBIN resource report does not name exactly one requested kernel")
    result: dict[str, int] = {}
    for label, name in (("REG", "registers_per_thread"), ("STACK", "stack_bytes"),
                        ("SHARED", "static_shared_bytes"), ("LOCAL", "local_bytes")):
        values = re.findall(rf"(?<!\S){label}:(\d+)(?=\s|$)", matches[0])
        if len(values) != 1:
            raise ValueError(f"CUBIN resource report has missing or ambiguous {label}")
        result[name] = int(values[0])
    return result


def load_compiled_resources(path: Path) -> dict[str, CompiledResources]:
    """Match retained source and binary to a portable allocation observation."""
    root = path.resolve(strict=True).parent
    document = json.loads(path.read_text())
    if not isinstance(document, dict) or document.get("schema_version") != 1 or not isinstance(document.get("rows"), list):
        raise ValueError("compiled report structure differs")
    observations = {}
    for index, row in enumerate(document["rows"]):
        if not isinstance(row, dict) or not isinstance(row.get("profile"), dict):
            raise ValueError("compiled report row structure differs")
        value = row["profile"].get("compiled_resources")
        if value is None:
            continue
        resource = CompiledResources.from_dict(value)
        binary_path = root / f"{index:04d}" / "kernel.cubin"
        source_path = root / f"{index:04d}" / "lowered.py"
        if binary_path.parent.is_symlink() or any(
            item.is_symlink() or root not in item.resolve(strict=True).parents
            for item in (binary_path, source_path)
        ):
            raise ValueError("compiled report artifact escapes its output directory")
        binary = binary_path.read_bytes()
        if not binary.startswith(b"\x7fELF") or sha256(binary).hexdigest() != resource.cubin_sha256:
            raise ValueError("compiled report CUBIN differs from its observation")
        if sha256(source_path.read_bytes()).hexdigest() != resource.source_sha256:
            raise ValueError("compiled report source differs from its observation")
        if resource.source_sha256 in observations:
            raise ValueError("compiled report has ambiguous observations for one source")
        observations[resource.source_sha256] = resource
    return observations
