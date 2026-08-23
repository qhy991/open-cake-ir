"""Typed Target definition.

A Target is the exact hardware contract a Schedule is verified against: which memory
spaces and operations exist, what the CTA resource budget is, and which instruction and
synchronization contracts are admitted. It is the only source of hardware facts the
verifier may use, so an unsupported capability is reported rather than silently assumed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .ir import MemorySpace, OperationKind, ScheduleParseError, _enum, _string


class TargetParseError(ValueError):
    """One Target document is not admissible."""


def _int_field(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TargetParseError(f"{context} must be a positive integer")
    return value


@dataclass(frozen=True)
class ResourceLimits:
    maximum_threads_per_cta: int
    maximum_warps_per_cta: int
    maximum_shared_memory_bytes: int
    maximum_tensor_memory_bytes: int
    maximum_grid: tuple[int, int, int]

    def capacity(self, space: MemorySpace) -> int | None:
        """Byte budget for one CTA in `space`, or None when the space is unbudgeted.

        `global` is not CTA-scoped and `register` has no declared budget on any
        current Target -- see `Verifier` for why the latter is reported rather than
        assumed unlimited.
        """

        if space is MemorySpace.SHARED:
            return self.maximum_shared_memory_bytes
        if space is MemorySpace.TENSOR:
            return self.maximum_tensor_memory_bytes
        return None

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "ResourceLimits":
        if not isinstance(value, Mapping):
            raise TargetParseError(f"{context} must be an object")
        grid = value.get("grid")
        if not isinstance(grid, Mapping) or set(grid) != {"x", "y", "z"}:
            raise TargetParseError(f"{context}.grid must declare x, y and z")
        return cls(
            _int_field(value.get("maximum_threads_per_cta"), f"{context}.maximum_threads_per_cta"),
            _int_field(value.get("maximum_warps_per_cta"), f"{context}.maximum_warps_per_cta"),
            _int_field(
                value.get("maximum_shared_memory_bytes"),
                f"{context}.maximum_shared_memory_bytes",
            ),
            _int_field(
                value.get("maximum_tensor_memory_bytes"),
                f"{context}.maximum_tensor_memory_bytes",
            ),
            (
                _int_field(grid["x"], f"{context}.grid.x"),
                _int_field(grid["y"], f"{context}.grid.y"),
                _int_field(grid["z"], f"{context}.grid.z"),
            ),
        )


@dataclass(frozen=True)
class Occupancy:
    """Per-multiprocessor facts, read from the device rather than asserted.

    A CTA budget bounds one CTA; these bound how many CTAs an SM can hold at once,
    which is what makes a resource the binding one.
    """

    multiprocessor_count: int
    registers_per_multiprocessor: int
    shared_memory_per_multiprocessor_bytes: int
    maximum_threads_per_multiprocessor: int

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "Occupancy":
        if not isinstance(value, Mapping):
            raise TargetParseError(f"{context} must be an object")
        fields = (
            "multiprocessor_count",
            "registers_per_multiprocessor",
            "shared_memory_per_multiprocessor_bytes",
            "maximum_threads_per_multiprocessor",
        )
        if set(value) != set(fields):
            raise TargetParseError(f"{context} fields differ")
        return cls(*(_int_field(value[name], f"{context}.{name}") for name in fields))


@dataclass(frozen=True)
class Target:
    target_id: str
    architecture: str
    device_names: tuple[str, ...]
    compute_capability: tuple[int, int]
    memory_spaces: frozenset[MemorySpace]
    operation_kinds: frozenset[OperationKind]
    resource_limits: ResourceLimits
    instruction_contracts: frozenset[str]
    synchronization_contracts: frozenset[str]
    occupancy: Occupancy | None

    @property
    def warp_size(self) -> int:
        return 32

    @classmethod
    def load(cls, path: str | Path) -> "Target":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, value: Any) -> "Target":
        if not isinstance(value, Mapping):
            raise TargetParseError("target must be an object")
        if value.get("schema_version") != 1:
            raise TargetParseError("target.schema_version must be 1")

        def string_tuple(field: str) -> tuple[str, ...]:
            items = value.get(field)
            if not isinstance(items, list) or not items:
                raise TargetParseError(f"target.{field} must be a non-empty list")
            return tuple(_string(item, f"target.{field}[]") for item in items)

        capability = value.get("compute_capability")
        if not isinstance(capability, list) or len(capability) != 2:
            raise TargetParseError("target.compute_capability must be a pair")

        try:
            spaces = frozenset(
                _enum(MemorySpace, item, "target.memory_spaces[]")
                for item in string_tuple("memory_spaces")
            )
            kinds = frozenset(
                _enum(OperationKind, item, "target.operation_kinds[]")
                for item in string_tuple("operation_kinds")
            )
        except ScheduleParseError as error:
            raise TargetParseError(str(error)) from error

        return cls(
            target_id=_string(value.get("target_id"), "target.target_id"),
            architecture=_string(value.get("architecture"), "target.architecture"),
            device_names=string_tuple("device_names"),
            compute_capability=(int(capability[0]), int(capability[1])),
            memory_spaces=spaces,
            operation_kinds=kinds,
            resource_limits=ResourceLimits.from_dict(
                value.get("resource_limits"), "target.resource_limits"
            ),
            instruction_contracts=frozenset(string_tuple("instruction_contracts")),
            synchronization_contracts=frozenset(string_tuple("synchronization_contracts")),
            occupancy=(
                Occupancy.from_dict(value["occupancy"], "target.occupancy")
                if "occupancy" in value
                else None
            ),
        )
