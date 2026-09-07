"""Typed Target definition.

A Target is the exact hardware contract a Schedule is verified against: which memory
spaces and operations exist, what the CTA resource budget is, and which instruction and
synchronization contracts are admitted. It is the only source of hardware facts the
verifier may use, so an unsupported capability is reported rather than silently assumed.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .ir import MemorySpace, OperationKind, ScheduleParseError, _enum, _string


class TargetParseError(ValueError):
    """One Target document is not admissible."""


def cuda_architecture(target_id: str) -> int:
    """Decode the two admitted exact CUDA code-generation targets."""
    if not isinstance(target_id, str) or re.fullmatch(r"sm_(100|103)a", target_id) is None:
        raise TargetParseError("unsupported exact CUDA target")
    return int(target_id[3:-1])


def cuda_target(target_id: str) -> "Target":
    """Read the canonical hardware facts bound by the Compiler and Executor.

    Runtime consumers pin these same files in their Executor closure. Offline
    compilation needs only cuda_architecture and never opens this data in its jail.
    """
    architecture = cuda_architecture(target_id)
    path = Path(__file__).resolve().parents[3] / "compiler" / "targets" / f"{target_id}.json"
    target = Target.load(path)
    if target.target_id != target_id or target.compute_capability != divmod(architecture, 10):
        raise TargetParseError("CUDA target identity and compute capability differ")
    return target


def _int_field(value: Any, context: str, *, allow_zero: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < (0 if allow_zero else 1):
        raise TargetParseError(f"{context} must be a {'nonnegative' if allow_zero else 'positive'} integer")
    return value


def _rate_field(value: Any, context: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise TargetParseError(f"{context} must be a positive finite rate")
    return float(value)


class PeakSource(str, Enum):
    """Where a declared peak rate came from, which decides what a ratio against it means.

    A `device_specification` rate is the architecture's ceiling, so a utilisation against
    it is the number everyone else reports and a value above one refutes something. A
    `microbenchmark` rate is the best this repository has measured, so a utilisation
    against it says how far a Schedule is from the best known kernel and can legitimately
    exceed one when a better kernel turns up. They are not interchangeable and a peak
    that did not say which it is would be read as whichever the reader assumed.
    """

    DEVICE_SPECIFICATION = "device_specification"
    MICROBENCHMARK = "microbenchmark"


@dataclass(frozen=True)
class PeakRate:
    """One peak rate and the provenance that makes it quotable.

    `observed_at` is required even for a specification rate, because a Target is content
    bound to a Compiler Revision and a rate that changed silently would change every
    utilisation ever derived from that Revision without changing its identity.
    """

    value: float
    source: PeakSource
    observed_at: str

    @classmethod
    def from_dict(cls, value: Any, field: str, context: str) -> "PeakRate":
        if not isinstance(value, Mapping):
            raise TargetParseError(f"{context} must be an object")
        if set(value) != {field, "source", "observed_at"}:
            raise TargetParseError(
                f"{context} must declare exactly {field}, source and observed_at"
            )
        try:
            source = PeakSource(value["source"])
        except ValueError as error:
            raise TargetParseError(
                f"{context}.source must be one of "
                + ", ".join(sorted(item.value for item in PeakSource))
            ) from error
        return cls(
            _rate_field(value[field], f"{context}.{field}"),
            source,
            _string(value["observed_at"], f"{context}.observed_at"),
        )


@dataclass(frozen=True)
class Peak:
    """The rates a measured time can be divided by, and nothing else.

    These are the only hardware facts on a Target that are not structural: everything
    else here decides whether a Schedule is admissible, and these decide nothing. They
    exist so that a *measured* time can be turned into a utilisation, which is why the
    verifier never reads them and the Target stays admissible without them.

    Arithmetic is keyed by instruction contract rather than by dtype. The contract is
    already the closed vocabulary a Schedule names and the verifier gates on, and it
    settles by construction what a dtype leaves open -- dense against sparse, which
    accumulator, which of two admitted spellings of one primitive. A key that is not a
    declared contract is refused rather than carried, so a peak cannot outlive the
    instruction it was measured for.
    """

    memory_bandwidth: PeakRate | None
    arithmetic: Mapping[str, PeakRate]

    def for_contract(self, contract: str) -> PeakRate | None:
        return self.arithmetic.get(contract)

    @classmethod
    def from_dict(cls, value: Any, contracts: frozenset[str], context: str) -> "Peak":
        if not isinstance(value, Mapping):
            raise TargetParseError(f"{context} must be an object")
        if not set(value) <= {"memory_bandwidth", "arithmetic"}:
            raise TargetParseError(
                f"{context} may declare memory_bandwidth and arithmetic only"
            )
        # Presence, not value, and the same reason `AccessIndex` reads its optional
        # fields that way: `get` would read an explicit null as an absent key, so
        # `{"memory_bandwidth": null}` would be a second spelling of omitting it.
        bandwidth = (
            PeakRate.from_dict(
                value["memory_bandwidth"],
                "bytes_per_second",
                f"{context}.memory_bandwidth",
            )
            if "memory_bandwidth" in value
            else None
        )
        raw = value["arithmetic"] if "arithmetic" in value else {}
        if not isinstance(raw, Mapping):
            raise TargetParseError(f"{context}.arithmetic must be an object")
        arithmetic: dict[str, PeakRate] = {}
        for contract, entry in raw.items():
            if contract not in contracts:
                raise TargetParseError(
                    f"{context}.arithmetic names {contract!r}, which is not a declared "
                    "instruction contract"
                )
            arithmetic[contract] = PeakRate.from_dict(
                entry, "flops_per_second", f"{context}.arithmetic.{contract}"
            )
        if bandwidth is None and not arithmetic:
            # A block declaring no rate is a second spelling of an absent block, and two
            # spellings of one fact are two Target byte strings for one Target.
            raise TargetParseError(f"{context} must declare at least one rate")
        return cls(bandwidth, MappingProxyType(arithmetic))


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
        if set(value) != {"maximum_threads_per_cta", "maximum_warps_per_cta", "maximum_shared_memory_bytes", "maximum_tensor_memory_bytes", "grid"}:
            raise TargetParseError(f"{context} fields differ")
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
                allow_zero=True,
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
    peak: Peak | None

    @property
    def triton_target(self) -> Mapping[str, object] | None:
        """Exact Triton device projection shared by lowering and Executor admission."""
        if self.target_id == "gfx1151":
            if self.architecture != "gfx1151" or self.execution_group_width != 32 or self.compute_capability is not None:
                return None
            return MappingProxyType({"backend": "hip", "arch": self.architecture, "warp_size": self.execution_group_width})
        if self.compute_capability is not None:
            major, minor = self.compute_capability
            return MappingProxyType({"backend": "cuda", "arch": major * 10 + minor, "warp_size": self.execution_group_width})
        return None

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
        if type(schema_version) is not int or schema_version not in (1, 2):
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
        if schema_version == 1 and capability is None and value.get("architecture") != "apple8":
            raise TargetParseError("target.compute_capability must be a pair")

        if value.get("architecture") == "apple8" and "compute_capability" in value:
            raise TargetParseError("Apple GPU targets have no CUDA compute capability")
        if value.get("architecture") == "apple8" and ("occupancy" in value or "peak" in value):
            raise TargetParseError("Apple8 has no admitted occupancy or peak calibration")

        execution_group_width = (
            32
            if schema_version == 1
            else _int_field(
                value.get("execution_group_width"),
                "target.execution_group_width",
            )
        )
        register_budget_group_width = (
            (4 if capability is not None else None)
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

        instruction_contracts = frozenset(string_tuple("instruction_contracts", allow_empty=True))
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
                    allow_empty=True,
                )
            ),
            synchronization_contracts=frozenset(
                string_tuple(
                    "synchronization_contracts",
                    allow_empty=True,
                )
            ),
            occupancy=(
                Occupancy.from_dict(value["occupancy"], "target.occupancy")
                if "occupancy" in value
                else None
            ),
            peak=(Peak.from_dict(value["peak"], instruction_contracts, "target.peak") if "peak" in value else None),
        )
