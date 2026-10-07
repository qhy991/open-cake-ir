"""Portable compiled allocation facts, inspected without loading a GPU module.

Code-object inspectors own physical allocation. This module validates its projection and
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


@dataclass(frozen=True, init=False)
class CompiledResources:
    source_sha256: str
    binary_sha256: str
    target: str
    entry_point: str
    threads_per_cta: int
    registers_per_thread: int
    static_shared_bytes: int
    dynamic_shared_bytes: int
    local_bytes: int
    stack_bytes: int | None
    compiler_version: str
    inspector_version: str
    code_object: str = "cubin"

    def __init__(self, source_sha256: str, binary_sha256: str | None = None,
                 target: str | None = None, entry_point: str | None = None,
                 threads_per_cta: int | None = None, registers_per_thread: int | None = None,
                 static_shared_bytes: int | None = None, dynamic_shared_bytes: int | None = None,
                 local_bytes: int | None = None, stack_bytes: int | None = None,
                 compiler_version: str | None = None, inspector_version: str | None = None,
                 code_object: str = "cubin", *, cubin_sha256: str | None = None) -> None:
        # Preserve the exported v1 Python constructor at its input boundary.
        # There is still one stored binary identity; two supplied names are ambiguous.
        if cubin_sha256 is not None:
            if binary_sha256 is not None or code_object != "cubin":
                raise ValueError("CUBIN identity cannot alias another binary or code object")
            binary_sha256 = cubin_sha256
        values = locals()
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, values[name])
        self.__post_init__()

    def __post_init__(self) -> None:
        for identity in (self.source_sha256, self.binary_sha256):
            if not isinstance(identity, str) or re.fullmatch(r"[0-9a-f]{64}", identity) is None:
                raise ValueError("compiled resource artifact identity differs")
        if any(not isinstance(value, str) or not value.strip() for value in (
            self.target, self.entry_point, self.compiler_version, self.inspector_version
        )):
            raise ValueError("compiled resource target or toolchain identity differs")
        for name in ("threads_per_cta", "registers_per_thread", "static_shared_bytes",
                     "dynamic_shared_bytes", "local_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"compiled resource {name} must be a nonnegative integer")
        if self.code_object not in {"cubin", "hsaco"}:
            raise ValueError("compiled resource code object is not supported")
        if self.code_object == "hsaco":
            if self.stack_bytes is not None:
                raise ValueError("HSACO reports combined private storage, not separate stack bytes")
        elif type(self.stack_bytes) is not int or self.stack_bytes < 0:
            raise ValueError("compiled resource stack_bytes must be a nonnegative integer")
        if not self.threads_per_cta or not self.registers_per_thread:
            raise ValueError("compiled kernel requires nonzero thread and register allocation")

    @property
    def shared_bytes(self) -> int:
        return self.static_shared_bytes + self.dynamic_shared_bytes

    @property
    def cubin_sha256(self) -> str:
        """Historical CUDA consumer boundary; never label HSACO as CUBIN."""
        if self.code_object != "cubin":
            raise ValueError("HSACO has no CUBIN identity")
        return self.binary_sha256

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        if self.code_object == "cubin":
            # Existing CUDA reports retain their v1 wire contract.
            value.pop("code_object")
            value["cubin_sha256"] = value.pop("binary_sha256")
            return {"schema_version": 1, **value}
        return {"schema_version": 2, **value}

    @classmethod
    def from_dict(cls, value: object) -> "CompiledResources":
        if not isinstance(value, Mapping) or type(value.get("schema_version")) is not int:
            raise ValueError("compiled resource fields differ")
        fields = set(cls.__dataclass_fields__)
        document = dict(value)
        version = document.pop("schema_version")
        if version == 1:
            legacy = fields - {"binary_sha256", "code_object"} | {"cubin_sha256"}
            if set(document) != legacy:
                raise ValueError("compiled resource fields differ")
            document["binary_sha256"] = document.pop("cubin_sha256")
            document["code_object"] = "cubin"
        elif version != 2 or set(document) != fields or document["code_object"] != "hsaco":
            raise ValueError("compiled resource fields differ")
        return cls(**document)

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
        binary_path = root / f"{index:04d}" / f"kernel.{resource.code_object}"
        source_path = root / f"{index:04d}" / "lowered.py"
        if binary_path.parent.is_symlink() or any(
            item.is_symlink() or root not in item.resolve(strict=True).parents
            for item in (binary_path, source_path)
        ):
            raise ValueError("compiled report artifact escapes its output directory")
        binary = binary_path.read_bytes()
        if not binary.startswith(b"\x7fELF") or sha256(binary).hexdigest() != resource.binary_sha256:
            raise ValueError("compiled report binary differs from its observation")
        if sha256(source_path.read_bytes()).hexdigest() != resource.source_sha256:
            raise ValueError("compiled report source differs from its observation")
        if resource.source_sha256 in observations:
            raise ValueError("compiled report has ambiguous observations for one source")
        observations[resource.source_sha256] = resource
    return observations
