"""Typed Target definition.

A Target is the exact hardware contract a Schedule is verified against: which memory
spaces and operations exist, what the CTA resource budget is, and which instruction and
synchronization contracts are admitted. It is the only source of hardware facts the
verifier may use, so an unsupported capability is reported rather than silently assumed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .ir import MemorySpace, OperationKind, ScheduleParseError, _enum, _string


class TargetParseError(ValueError):
    """One Target document is not admissible."""


def _int_field(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TargetParseError(f"{context} must be a positive integer")
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
    peak: Peak | None

    @property
    def warp_size(self) -> int:
        return 32

    @property
    def warps_per_warpgroup(self) -> int:
        """Warps that issue a warpgroup-wide instruction together.

        A constant of the ISA rather than a device observation, like `warp_size`, so it
        lives here instead of in a Target document. `setmaxnreg` is warpgroup-wide, which
        is what makes this a legality rule on a role's warp range rather than a
        preference.
        """

        return 4

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

        instruction_contracts = frozenset(string_tuple("instruction_contracts"))
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
            instruction_contracts=instruction_contracts,
            synchronization_contracts=frozenset(string_tuple("synchronization_contracts")),
            occupancy=(
                Occupancy.from_dict(value["occupancy"], "target.occupancy")
                if "occupancy" in value
                else None
            ),
            peak=(
                Peak.from_dict(value["peak"], instruction_contracts, "target.peak")
                if "peak" in value
                else None
            ),
        )
