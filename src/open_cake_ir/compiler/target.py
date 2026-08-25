"""Typed Target definition.

A Target is the exact hardware contract a Schedule is verified against.  Schema v1 is
the retained CUDA-shaped document.  Schema v2 names execution groups and workgroup
resources without requiring another architecture to invent a CUDA compute capability or
a tensor-memory budget it does not have.
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
    maximum_threads_per_workgroup: int
    maximum_execution_groups_per_workgroup: int
    maximum_threadgroup_memory_bytes: int
    maximum_tensor_memory_bytes: int | None
    maximum_grid: tuple[int, int, int]

    @property
    def maximum_threads_per_cta(self) -> int:
        """Schema-v1 compatibility spelling."""

        return self.maximum_threads_per_workgroup

    @property
    def maximum_warps_per_cta(self) -> int:
        """Schema-v1 compatibility spelling."""

        return self.maximum_execution_groups_per_workgroup

    @property
    def maximum_shared_memory_bytes(self) -> int:
        """Schema-v1 compatibility spelling."""

        return self.maximum_threadgroup_memory_bytes

    def capacity(self, space: MemorySpace) -> int | None:
        """Byte budget for one CTA in `space`, or None when the space is unbudgeted.

        `global` is not CTA-scoped and `register` has no declared budget on any
        current Target -- see `Verifier` for why the latter is reported rather than
        assumed unlimited.
        """

        if space is MemorySpace.SHARED:
            return self.maximum_threadgroup_memory_bytes
        if space is MemorySpace.TENSOR:
            return self.maximum_tensor_memory_bytes
        return None

    @classmethod
    def from_v1(cls, value: Any, context: str) -> "ResourceLimits":
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

    @classmethod
    def from_v2(
        cls,
        value: Any,
        context: str,
        *,
        execution_group_width: int,
        has_tensor_memory: bool,
    ) -> "ResourceLimits":
        required = {
            "maximum_threads_per_workgroup",
            "maximum_threadgroup_memory_bytes",
            "maximum_grid",
        }
        optional = {"maximum_tensor_memory_bytes"}
        if not isinstance(value, Mapping) or not required <= set(value) <= required | optional:
            raise TargetParseError(f"{context} fields differ")
        tensor_limit_declared = "maximum_tensor_memory_bytes" in value
        if has_tensor_memory and not tensor_limit_declared:
            raise TargetParseError(
                f"{context} requires a tensor memory limit when target.memory_spaces "
                "contains tensor"
            )
        if tensor_limit_declared and not has_tensor_memory:
            raise TargetParseError(
                f"{context} declares a tensor memory limit without tensor memory"
            )
        maximum_threads = _int_field(
            value.get("maximum_threads_per_workgroup"),
            f"{context}.maximum_threads_per_workgroup",
        )
        if maximum_threads % execution_group_width:
            raise TargetParseError(
                f"{context}.maximum_threads_per_workgroup must contain whole "
                "execution groups"
            )
        grid = value.get("maximum_grid")
        if not isinstance(grid, Mapping) or set(grid) != {"x", "y", "z"}:
            raise TargetParseError(f"{context}.maximum_grid must declare x, y and z")
        return cls(
            maximum_threads,
            maximum_threads // execution_group_width,
            _int_field(
                value.get("maximum_threadgroup_memory_bytes"),
                f"{context}.maximum_threadgroup_memory_bytes",
            ),
            (
                _int_field(
                    value.get("maximum_tensor_memory_bytes"),
                    f"{context}.maximum_tensor_memory_bytes",
                )
                if tensor_limit_declared
                else None
            ),
            (
                _int_field(grid["x"], f"{context}.maximum_grid.x"),
                _int_field(grid["y"], f"{context}.maximum_grid.y"),
                _int_field(grid["z"], f"{context}.maximum_grid.z"),
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
    schema_version: int
    target_id: str
    architecture: str
    execution_group_width: int
    register_budget_group_width: int | None
    device_names: tuple[str, ...]
    compute_capability: tuple[int, int] | None
    memory_spaces: frozenset[MemorySpace]
    operation_kinds: frozenset[OperationKind]
    resource_limits: ResourceLimits
    instruction_contracts: frozenset[str]
    synchronization_contracts: frozenset[str]
    occupancy: Occupancy | None

    @property
    def warp_size(self) -> int:
        """Compatibility spelling for the Target-owned execution-group width."""

        return self.execution_group_width

    @property
    def warps_per_warpgroup(self) -> int | None:
        """Execution groups that issue one register-budget instruction together.

        Schema v1 retains Blackwell's four-warp ``setmaxnreg`` scope.  Schema v2 leaves
        this absent unless the Target has an independently declared equivalent; another
        architecture must not inherit the CUDA issue width.
        """

        return self.register_budget_group_width

    @classmethod
    def load(cls, path: str | Path) -> "Target":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, value: Any) -> "Target":
        if not isinstance(value, Mapping):
            raise TargetParseError("target must be an object")
        schema_version = value.get("schema_version")
        if schema_version not in (1, 2):
            raise TargetParseError("target.schema_version must be 1 or 2")

        if schema_version == 2:
            if "execution_group_width" not in value:
                raise TargetParseError(
                    "target.execution_group_width must be a positive integer"
                )
            required = {
                "schema_version",
                "target_id",
                "architecture",
                "execution_group_width",
                "memory_spaces",
                "operation_kinds",
                "resource_limits",
                "instruction_contracts",
                "synchronization_contracts",
                "citations",
            }
            optional = {
                "device_names",
                "compute_capability",
                "register_budget_group_width",
            }
            if not required <= set(value) <= required | optional:
                raise TargetParseError("target fields differ")

        def string_tuple(field: str, *, allow_empty: bool = False) -> tuple[str, ...]:
            items = value.get(field)
            if not isinstance(items, list) or (not items and not allow_empty):
                qualifier = "a list" if allow_empty else "a non-empty list"
                raise TargetParseError(f"target.{field} must be {qualifier}")
            return tuple(_string(item, f"target.{field}[]") for item in items)

        capability = value.get("compute_capability")
        if capability is not None and (
            not isinstance(capability, list)
            or len(capability) != 2
            or any(
                not isinstance(item, int) or isinstance(item, bool) or item < 0
                for item in capability
            )
        ):
            raise TargetParseError(
                "target.compute_capability must be a non-negative pair"
            )
        if schema_version == 1 and capability is None:
            raise TargetParseError("target.compute_capability must be a pair")

        execution_group_width = (
            32
            if schema_version == 1
            else _int_field(
                value.get("execution_group_width"),
                "target.execution_group_width",
            )
        )
        register_budget_group_width = (
            4
            if schema_version == 1
            else (
                None
                if value.get("register_budget_group_width") is None
                else _int_field(
                    value.get("register_budget_group_width"),
                    "target.register_budget_group_width",
                )
            )
        )

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

        device_names = (
            string_tuple("device_names")
            if schema_version == 1 or "device_names" in value
            else ()
        )
        limits = (
            ResourceLimits.from_v1(value.get("resource_limits"), "target.resource_limits")
            if schema_version == 1
            else ResourceLimits.from_v2(
                value.get("resource_limits"),
                "target.resource_limits",
                execution_group_width=execution_group_width,
                has_tensor_memory=MemorySpace.TENSOR in spaces,
            )
        )

        return cls(
            schema_version=schema_version,
            target_id=_string(value.get("target_id"), "target.target_id"),
            architecture=_string(value.get("architecture"), "target.architecture"),
            execution_group_width=execution_group_width,
            register_budget_group_width=register_budget_group_width,
            device_names=device_names,
            compute_capability=(
                None
                if capability is None
                else (capability[0], capability[1])
            ),
            memory_spaces=spaces,
            operation_kinds=kinds,
            resource_limits=limits,
            instruction_contracts=frozenset(
                string_tuple(
                    "instruction_contracts",
                    allow_empty=schema_version == 2,
                )
            ),
            synchronization_contracts=frozenset(
                string_tuple(
                    "synchronization_contracts",
                    allow_empty=schema_version == 2,
                )
            ),
            occupancy=(
                Occupancy.from_dict(value["occupancy"], "target.occupancy")
                if "occupancy" in value
                else None
            ),
        )
