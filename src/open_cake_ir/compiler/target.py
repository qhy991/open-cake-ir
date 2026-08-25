"""Typed Target definition.

A Target is the exact architecture contract a Schedule is verified against. Schema v1
is the retained CUDA-shaped document; schema v2 names execution groups and workgroup
resources without forcing another architecture to invent CUDA identities. Both parse to
one semantic object, so verifiers, analyses, and lowering Adapters consume the same
hardware facts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .ir import (
    DType,
    MemorySpace,
    OperationKind,
    ScheduleParseError,
    _enum,
    _string as _ir_string,
)


class TargetParseError(ValueError):
    """One Target document is not admissible."""


def _string(value: object, context: str) -> str:
    """Keep Schedule parser errors from leaking across the Target interface."""

    try:
        return _ir_string(value, context)
    except ScheduleParseError as error:
        raise TargetParseError(str(error)) from error


def _positive_int(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TargetParseError(f"{context} must be a positive integer")
    return value


def _string_tuple(
    value: Mapping[str, object], field: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    items = value.get(field)
    if not isinstance(items, list) or (not items and not allow_empty):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise TargetParseError(f"target.{field} must be {qualifier}")
    return tuple(_string(item, f"target.{field}[]") for item in items)


def _grid(value: object, context: str) -> tuple[int, int, int]:
    if not isinstance(value, Mapping) or set(value) != {"x", "y", "z"}:
        raise TargetParseError(f"{context} must declare x, y and z")
    return (
        _positive_int(value["x"], f"{context}.x"),
        _positive_int(value["y"], f"{context}.y"),
        _positive_int(value["z"], f"{context}.z"),
    )


class InstructionPlacement(str, Enum):
    """Which MMA atom details a Schedule states instead of leaving to an Adapter."""

    BACKEND = "backend"
    EXPLICIT = "explicit"


class InstructionShape(str, Enum):
    """Whether the atom shape is a Schedule commitment or an Adapter decision."""

    BACKEND = "backend"
    EXPLICIT = "explicit"


@dataclass(frozen=True)
class InstructionContract:
    """One instruction admitted by a Target.

    Schema-v2 contracts can carry the MMA facts a lowering Adapter needs. Schema v1
    contracts are names only: the current verifier uses Target membership and owns its
    existing per-contract legality tables. Representing those names directly avoids a
    second, inevitably stale whitelist in the Target parser.
    """

    name: str
    operand_dtypes: frozenset[DType] | None = None
    accumulator_dtype: DType | None = None
    atom_shapes: tuple[tuple[int, int, int], ...] = ()
    placement: InstructionPlacement | None = None
    shape: InstructionShape | None = None

    @classmethod
    def named(cls, name: str) -> "InstructionContract":
        """Construct a schema-v1 name-only admission contract."""

        return cls(name=name)

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "InstructionContract":
        if not isinstance(value, Mapping) or set(value) != {
            "name",
            "operand_dtypes",
            "accumulator_dtype",
            "atom_shapes",
            "placement",
            "shape",
        }:
            raise TargetParseError(f"{context} fields differ")
        operands = value.get("operand_dtypes")
        if not isinstance(operands, list) or not operands:
            raise TargetParseError(f"{context}.operand_dtypes must be a non-empty list")
        raw_shapes = value.get("atom_shapes")
        if not isinstance(raw_shapes, list) or not raw_shapes:
            raise TargetParseError(f"{context}.atom_shapes must be a non-empty list")
        try:
            operand_dtypes = frozenset(
                _enum(DType, item, f"{context}.operand_dtypes[]") for item in operands
            )
            accumulator = _enum(
                DType, value.get("accumulator_dtype"), f"{context}.accumulator_dtype"
            )
            placement = _enum(
                InstructionPlacement, value.get("placement"), f"{context}.placement"
            )
            shape = _enum(InstructionShape, value.get("shape"), f"{context}.shape")
        except ScheduleParseError as error:
            raise TargetParseError(str(error)) from error

        shapes: list[tuple[int, int, int]] = []
        for index, raw_shape in enumerate(raw_shapes):
            if not isinstance(raw_shape, list) or len(raw_shape) != 3:
                raise TargetParseError(
                    f"{context}.atom_shapes[{index}] must declare exactly M, N and K"
                )
            shapes.append(
                tuple(
                    _positive_int(
                        extent, f"{context}.atom_shapes[{index}][{axis}]"
                    )
                    for axis, extent in enumerate(raw_shape)
                )
            )

        return cls(
            name=_string(value.get("name"), f"{context}.name"),
            operand_dtypes=operand_dtypes,
            accumulator_dtype=accumulator,
            atom_shapes=tuple(shapes),
            placement=placement,
            shape=shape,
        )


@dataclass(frozen=True)
class ResourceLimits:
    maximum_threads_per_workgroup: int
    maximum_execution_groups_per_workgroup: int
    maximum_threadgroup_memory_bytes: int
    maximum_tensor_memory_bytes: int | None
    maximum_grid: tuple[int, int, int] | None

    @property
    def maximum_threads_per_cta(self) -> int:
        """Schema-v1 compatibility spelling used by the current verifier."""

        return self.maximum_threads_per_workgroup

    @property
    def maximum_warps_per_cta(self) -> int:
        """Schema-v1 compatibility spelling used by the current verifier."""

        return self.maximum_execution_groups_per_workgroup

    @property
    def maximum_shared_memory_bytes(self) -> int:
        """Schema-v1 compatibility spelling used by current analyses."""

        return self.maximum_threadgroup_memory_bytes

    def capacity(self, space: MemorySpace) -> int | None:
        """Workgroup byte budget for ``space``, when the Target declares one."""

        if space is MemorySpace.SHARED:
            return self.maximum_threadgroup_memory_bytes
        if space is MemorySpace.TENSOR:
            return self.maximum_tensor_memory_bytes
        return None

    @classmethod
    def from_v1(cls, value: Any, execution_group_width: int) -> "ResourceLimits":
        context = "target.resource_limits"
        if not isinstance(value, Mapping) or set(value) != {
            "maximum_threads_per_cta",
            "maximum_warps_per_cta",
            "maximum_shared_memory_bytes",
            "maximum_tensor_memory_bytes",
            "grid",
        }:
            raise TargetParseError(f"{context} fields differ")
        threads = _positive_int(
            value.get("maximum_threads_per_cta"),
            f"{context}.maximum_threads_per_cta",
        )
        groups = _positive_int(
            value.get("maximum_warps_per_cta"),
            f"{context}.maximum_warps_per_cta",
        )
        if groups * execution_group_width > threads:
            raise TargetParseError(
                f"{context} execution-group capacity exceeds its thread capacity"
            )
        return cls(
            maximum_threads_per_workgroup=threads,
            maximum_execution_groups_per_workgroup=groups,
            maximum_threadgroup_memory_bytes=_positive_int(
                value.get("maximum_shared_memory_bytes"),
                f"{context}.maximum_shared_memory_bytes",
            ),
            maximum_tensor_memory_bytes=_positive_int(
                value.get("maximum_tensor_memory_bytes"),
                f"{context}.maximum_tensor_memory_bytes",
            ),
            maximum_grid=_grid(value.get("grid"), f"{context}.grid"),
        )

    @classmethod
    def from_v2(cls, value: Any, execution_group_width: int) -> "ResourceLimits":
        context = "target.resource_limits"
        required = {
            "maximum_threads_per_workgroup",
            "maximum_threadgroup_memory_bytes",
        }
        optional = {"maximum_tensor_memory_bytes", "maximum_grid"}
        if (
            not isinstance(value, Mapping)
            or not required <= set(value)
            or not set(value) <= required | optional
        ):
            raise TargetParseError(f"{context} fields differ")
        threads = _positive_int(
            value.get("maximum_threads_per_workgroup"),
            f"{context}.maximum_threads_per_workgroup",
        )
        if threads % execution_group_width:
            raise TargetParseError(
                f"{context}.maximum_threads_per_workgroup must contain whole "
                "execution groups"
            )
        return cls(
            maximum_threads_per_workgroup=threads,
            maximum_execution_groups_per_workgroup=threads // execution_group_width,
            maximum_threadgroup_memory_bytes=_positive_int(
                value.get("maximum_threadgroup_memory_bytes"),
                f"{context}.maximum_threadgroup_memory_bytes",
            ),
            maximum_tensor_memory_bytes=(
                _positive_int(
                    value.get("maximum_tensor_memory_bytes"),
                    f"{context}.maximum_tensor_memory_bytes",
                )
                if "maximum_tensor_memory_bytes" in value
                else None
            ),
            maximum_grid=(
                _grid(value.get("maximum_grid"), f"{context}.maximum_grid")
                if "maximum_grid" in value
                else None
            ),
        )


@dataclass(frozen=True)
class Occupancy:
    """Optional legacy per-multiprocessor facts used by static residency analysis."""

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
        return cls(*(_positive_int(value[name], f"{context}.{name}") for name in fields))


@dataclass(frozen=True)
class Target:
    schema_version: int
    target_id: str
    architecture: str
    execution_group_width: int
    register_budget_group_width: int | None
    memory_spaces: frozenset[MemorySpace]
    operation_kinds: frozenset[OperationKind]
    resource_limits: ResourceLimits
    instructions: Mapping[str, InstructionContract]
    synchronization_contracts: frozenset[str]
    citations: tuple[Mapping[str, object], ...]
    occupancy: Occupancy | None
    device_names: tuple[str, ...] = ()
    compute_capability: tuple[int, int] | None = None

    @property
    def warp_size(self) -> int:
        """Schedule-v1 compatibility spelling for execution-group width."""

        return self.execution_group_width

    @property
    def warps_per_warpgroup(self) -> int | None:
        """Schedule-v1 compatibility spelling for register-budget issue scope."""

        return self.register_budget_group_width

    @property
    def instruction_contracts(self) -> frozenset[str]:
        """Instruction names admitted by this exact Target."""

        return frozenset(self.instructions)

    def instruction(self, name: str) -> InstructionContract | None:
        """Return the Target-owned instruction contract, when admitted."""

        return self.instructions.get(name)

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

        common = {
            "schema_version",
            "target_id",
            "architecture",
            "memory_spaces",
            "operation_kinds",
            "resource_limits",
            "instruction_contracts",
            "synchronization_contracts",
            "citations",
        }
        if schema_version == 1:
            required = common | {"device_names", "compute_capability"}
            if not required <= set(value) or not set(value) <= required | {"occupancy"}:
                raise TargetParseError("target fields differ")
            execution_group_width = 32
        else:
            required = common | {"execution_group_width"}
            if set(value) != required:
                raise TargetParseError("target fields differ")
            execution_group_width = _positive_int(
                value.get("execution_group_width"), "target.execution_group_width"
            )

        try:
            spaces = frozenset(
                _enum(MemorySpace, item, "target.memory_spaces[]")
                for item in _string_tuple(value, "memory_spaces")
            )
            kinds = frozenset(
                _enum(OperationKind, item, "target.operation_kinds[]")
                for item in _string_tuple(value, "operation_kinds")
            )
        except ScheduleParseError as error:
            raise TargetParseError(str(error)) from error

        raw_citations = value.get("citations")
        if not isinstance(raw_citations, list) or not raw_citations:
            raise TargetParseError("target.citations must be a non-empty list")
        citations: list[Mapping[str, object]] = []
        for index, item in enumerate(raw_citations):
            if not isinstance(item, Mapping) or not item:
                raise TargetParseError(
                    f"target.citations[{index}] must be a non-empty object"
                )
            citations.append(MappingProxyType(dict(item)))

        if schema_version == 1:
            device_names = _string_tuple(value, "device_names")
            raw_capability = value.get("compute_capability")
            if (
                not isinstance(raw_capability, list)
                or len(raw_capability) != 2
                or any(
                    not isinstance(item, int) or isinstance(item, bool) or item < 0
                    for item in raw_capability
                )
            ):
                raise TargetParseError(
                    "target.compute_capability must be a non-negative pair"
                )
            contract_names = _string_tuple(value, "instruction_contracts")
            contracts = {
                name: InstructionContract.named(name) for name in contract_names
            }
            if len(contracts) != len(contract_names):
                raise TargetParseError(
                    "target.instruction_contracts names must be unique"
                )
            limits = ResourceLimits.from_v1(
                value.get("resource_limits"), execution_group_width
            )
            occupancy = (
                Occupancy.from_dict(value["occupancy"], "target.occupancy")
                if "occupancy" in value
                else None
            )
            register_budget_group_width = 4
            compute_capability = (raw_capability[0], raw_capability[1])
        else:
            device_names = ()
            compute_capability = None
            raw_contracts = value.get("instruction_contracts")
            if not isinstance(raw_contracts, list):
                raise TargetParseError(
                    "target.instruction_contracts must be a list"
                )
            parsed_contracts = [
                InstructionContract.from_dict(
                    item, f"target.instruction_contracts[{index}]"
                )
                for index, item in enumerate(raw_contracts)
            ]
            contracts = {item.name: item for item in parsed_contracts}
            if len(contracts) != len(parsed_contracts):
                raise TargetParseError(
                    "target.instruction_contracts names must be unique"
                )
            limits = ResourceLimits.from_v2(
                value.get("resource_limits"), execution_group_width
            )
            occupancy = None
            register_budget_group_width = None

        return cls(
            schema_version=schema_version,
            target_id=_string(value.get("target_id"), "target.target_id"),
            architecture=_string(value.get("architecture"), "target.architecture"),
            execution_group_width=execution_group_width,
            register_budget_group_width=register_budget_group_width,
            memory_spaces=spaces,
            operation_kinds=kinds,
            resource_limits=limits,
            instructions=MappingProxyType(contracts),
            synchronization_contracts=frozenset(
                _string_tuple(
                    value,
                    "synchronization_contracts",
                    allow_empty=schema_version == 2,
                )
            ),
            citations=tuple(citations),
            occupancy=occupancy,
            device_names=device_names,
            compute_capability=compute_capability,
        )
