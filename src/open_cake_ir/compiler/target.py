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


class Vendor(str, Enum):
    """Which vendor's hardware a Target describes, declared rather than inferred.

    Every rule keyed on vendor reads this field. The alternative, and what this replaces,
    was reading the presence of a CUDA field: a well-formed AMD document was refused with
    `target.compute_capability is required for CUDA targets`, and the same document with a
    fabricated capability pair parsed and then carried NVIDIA's warpgroup rule against a
    wavefront. No vendor lives in an `else` here, which is why this is a closed set that
    refuses an unknown name rather than a string anyone may extend from a Target document.
    """

    NVIDIA = "nvidia"
    APPLE = "apple"
    AMD = "amd"
    # Hygon is its own vendor, not an AMD spelling. Its DCU reports vendor C-3000 and
    # device type HCU rather than AMD and GPU, Triton ships a separate `hcu` backend
    # beside `amd`, its FP16 dot lowers to v_mmac rather than v_mfma, its toolchain is
    # the dcc fork whose llc names gfx926/928/936/938, and it installs hy-smi where a
    # ROCm host installs amd-smi. The emitted code object still carries the
    # `amdgcn-amd-amdhsa--` triple, which is the ABI it conforms to and not a claim about
    # who built the hardware; that distinction is why vendor and code-object family are
    # two fields here and not one.
    HYGON = "hygon"
    METAX = "metax"


class CodeObject(str, Enum):
    """The object a target's toolchain produces, declared rather than enumerated.

    The third of the three axes F-2026-09-15-004 names: `vendor` is who built the
    hardware, a Schedule's `lowering.backend` is which mechanism emits source, and this
    is what the compiled artifact is. gfx938 and gfx1151 share this and differ by vendor;
    sm_100a and gfx938 share a lowering route and neither of the other two. Declaring it
    is what lets the Evaluation layer stop partitioning target ids by hand, so an eighth
    target is a document rather than a document plus six edits in shared code.
    """

    CUBIN = "cubin"
    METAL_BINARY_ARCHIVE = "metal_binary_archive"
    HSACO = "hsaco"
    MCFATBIN = "mcfatbin"


def declared_target(target_id: str) -> "Target":
    """Read the Target document this checkout declares for one exact id.

    The Evaluation layer resolves declared hardware facts through here. Offline
    compilation does not and must not: it never opens a Target document inside its jail;
    the route facts it needs ride the toolchain requirements the Compiler composes.
    """
    if not isinstance(target_id, str) or not target_id:
        raise TargetParseError("an exact target id is required")
    path = Path(__file__).resolve().parents[3] / "compiler" / "targets" / f"{target_id}.json"
    if not path.is_file():
        raise TargetParseError(f"no Target document declares {target_id!r}")
    target = Target.load(path)
    if target.target_id != target_id:
        raise TargetParseError("Target identity differs from the document naming it")
    return target


def _int_field(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
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
    it can refute a value above one only when the counted work and measurement scope
    match that ceiling; logical bytes served from cache do not meet a DRAM premise. A
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


_LIMIT_FIELDS = frozenset({"maximum_threads_per_cta", "maximum_shared_memory_bytes", "grid"})
_OPTIONAL_LIMIT_FIELDS = frozenset({"maximum_registers_per_thread"})

# The document's own shape has one owner, this parser. `citations` is required here and
# read by the Revision loader, which is where non-emptiness is checked; the typed Target
# carries no citation.
_TARGET_FIELDS = frozenset({
    "schema_version",
    "target_id",
    "architecture",
    "device_names",
    "vendor",
    "code_object",
    "memory_spaces",
    "operation_kinds",
    "warp_size",
    "resource_limits",
    "instruction_contracts",
    "synchronization_contracts",
    "citations",
})
# `compute_capability` and `warps_per_warpgroup` are admitted by code object below.
_OPTIONAL_TARGET_FIELDS = frozenset({
    "occupancy", "peak", "compute_capability", "warps_per_warpgroup", "triton_arch",
})


@dataclass(frozen=True)
class ResourceLimits:
    maximum_threads_per_cta: int
    # The Target's role-slot width, carried here so the slot budget below is derived
    # from the thread budget instead of being declared a second time beside it.
    warp_size: int
    maximum_shared_memory_bytes: int
    # None when the Target declares no tensor memory space: a limit it never modeled,
    # reported as such rather than written down as zero.
    maximum_tensor_memory_bytes: int | None
    maximum_grid: tuple[int, int, int]
    maximum_registers_per_thread: int | None = None

    @property
    def maximum_warps_per_cta(self) -> int:
        return self.maximum_threads_per_cta // self.warp_size

    def capacity(self, space: MemorySpace) -> int | None:
        """Byte budget for one CTA in `space`, or None when the space is unbudgeted.

        `global` is not CTA-scoped, `register` has no declared budget on any current
        Target -- see `Verifier` for why the latter is reported rather than assumed
        unlimited -- and `tensor` is unmodeled on a Target without that space.
        """

        if space is MemorySpace.SHARED:
            return self.maximum_shared_memory_bytes
        if space is MemorySpace.TENSOR:
            return self.maximum_tensor_memory_bytes
        return None

    @classmethod
    def from_dict(
        cls, value: Any, context: str, *, warp_size: int, tensor_memory: bool
    ) -> "ResourceLimits":
        if not isinstance(value, Mapping):
            raise TargetParseError(f"{context} must be an object")
        # The tensor limit is admitted exactly when the Target declares the space it
        # bounds; a zero written for a space that does not exist is a substituted fact.
        if not tensor_memory and "maximum_tensor_memory_bytes" in value:
            raise TargetParseError(
                "the target declares no tensor memory space; "
                f"{context}.maximum_tensor_memory_bytes must be omitted"
            )
        admitted = _LIMIT_FIELDS | _OPTIONAL_LIMIT_FIELDS | {"maximum_tensor_memory_bytes"}
        if not set(value) <= admitted:
            raise TargetParseError(f"{context} fields differ")
        grid = value.get("grid")
        if not isinstance(grid, Mapping) or set(grid) != {"x", "y", "z"}:
            raise TargetParseError(f"{context}.grid must declare x, y and z")
        return cls(
            _int_field(value.get("maximum_threads_per_cta"), f"{context}.maximum_threads_per_cta"),
            warp_size,
            _int_field(
                value.get("maximum_shared_memory_bytes"), f"{context}.maximum_shared_memory_bytes"
            ),
            _int_field(
                value.get("maximum_tensor_memory_bytes"), f"{context}.maximum_tensor_memory_bytes"
            ) if tensor_memory else None,
            (
                _int_field(grid["x"], f"{context}.grid.x"),
                _int_field(grid["y"], f"{context}.grid.y"),
                _int_field(grid["z"], f"{context}.grid.z"),
            ),
            None if value.get("maximum_registers_per_thread") is None else _int_field(
                value["maximum_registers_per_thread"], f"{context}.maximum_registers_per_thread"),
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
    vendor: Vendor
    # What this target's toolchain produces. Read by the Evaluation layer in place of its
    # own target-id partition; see CodeObject.
    code_object: CodeObject
    compute_capability: tuple[int, int] | None
    memory_spaces: frozenset[MemorySpace]
    operation_kinds: frozenset[OperationKind]
    # Width of a role slot -- an NVIDIA warp, an Apple SIMD group, another vendor's own
    # name for the same thing. The Target that owns the fact declares it, beside the
    # citation that evidences it; a shared constant here would be one value with two
    # owners, and would read 32 against any width that is not 32.
    warp_size: int
    resource_limits: ResourceLimits
    instruction_contracts: frozenset[str]
    synchronization_contracts: frozenset[str]
    occupancy: Occupancy | None
    peak: Peak | None
    # Role slots that issue a warpgroup-wide instruction together, on an ISA that has the
    # concept. `setmaxnreg` is warpgroup-wide, which is what makes this a legality rule on
    # a role's slot range rather than a preference. Declared beside its citation like
    # `warp_size`, and absent -- not zero, not a borrowed four -- on an ISA without it.
    warps_per_warpgroup: int | None = None
    # MACA's Triton compatibility architecture is an API fact, distinct from both
    # the physical xcore target and its native code-generation family. It is not a
    # CUDA compute capability and may not confer CUDA contracts or instructions.
    triton_arch: int | None = None

    @classmethod
    def load(cls, path: str | Path) -> "Target":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, value: Any) -> "Target":
        if not isinstance(value, Mapping):
            raise TargetParseError("target must be an object")
        # An unknown key is refused here; a missing one is refused by name below, by
        # the parser of that field.
        if not set(value) <= _TARGET_FIELDS | _OPTIONAL_TARGET_FIELDS:
            raise TargetParseError("target fields differ")
        if value.get("schema_version") != 1:
            raise TargetParseError("target.schema_version must be 1")
        if "citations" not in value:
            raise TargetParseError("target.citations is required")

        def string_tuple(field: str, *, allow_empty: bool = False) -> tuple[str, ...]:
            items = value.get(field)
            if not isinstance(items, list) or (not items and not allow_empty):
                raise TargetParseError(f"target.{field} must be a non-empty list")
            return tuple(_string(item, f"target.{field}[]") for item in items)

        try:
            vendor = Vendor(value.get("vendor"))
        except ValueError as error:
            raise TargetParseError(
                "target.vendor must be one of "
                + ", ".join(sorted(item.value for item in Vendor))
            ) from error
        try:
            code_object = CodeObject(value.get("code_object"))
        except ValueError as error:
            raise TargetParseError(
                "target.code_object must be one of "
                + ", ".join(sorted(item.value for item in CodeObject))
            ) from error
        # The CUDA-only fields are keyed on the code object, not the vendor: a compute
        # capability and a warpgroup are facts about what a cubin encodes, and a document
        # that produces another object is refused for them by that object's name.
        cubin = code_object is CodeObject.CUBIN
        capability = value.get("compute_capability")
        if cubin and capability is None:
            raise TargetParseError("target.compute_capability is required for cubin targets")
        if not cubin and "compute_capability" in value:
            raise TargetParseError(
                f"{code_object.value} targets have no CUDA compute capability"
            )
        if capability is not None and (
            not isinstance(capability, list) or len(capability) != 2
            or any(type(item) is not int or item < 0 for item in capability)
        ):
            raise TargetParseError("target.compute_capability must be a nonnegative integer pair")
        if not cubin and "warps_per_warpgroup" in value:
            raise TargetParseError(f"{code_object.value} targets have no warps_per_warpgroup")
        if code_object is CodeObject.MCFATBIN:
            triton_arch = _int_field(value.get("triton_arch"), "target.triton_arch")
        else:
            if "triton_arch" in value:
                raise TargetParseError(f"{code_object.value} derives its Triton architecture from existing Target facts")
            triton_arch = None

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

        instruction_contracts = frozenset(string_tuple("instruction_contracts", allow_empty=True))
        peak = (
            Peak.from_dict(value["peak"], instruction_contracts, "target.peak")
            if "peak" in value else None
        )
        # An undeclared width is refused, never substituted: reading 32 for a target that
        # never said 32 is exactly the silent answer this field exists to stop.
        warp_size = _int_field(value.get("warp_size"), "target.warp_size")
        warpgroup = value.get("warps_per_warpgroup")
        if warpgroup is not None:
            warpgroup = _int_field(warpgroup, "target.warps_per_warpgroup")
        limits = ResourceLimits.from_dict(
            value.get("resource_limits"), "target.resource_limits",
            warp_size=warp_size, tensor_memory=MemorySpace.TENSOR in spaces,
        )
        return cls(
            target_id=_string(value.get("target_id"), "target.target_id"),
            architecture=_string(value.get("architecture"), "target.architecture"),
            device_names=string_tuple("device_names"),
            vendor=vendor,
            code_object=code_object,
            compute_capability=None if capability is None else (capability[0], capability[1]),
            memory_spaces=spaces,
            operation_kinds=kinds,
            warp_size=warp_size,
            resource_limits=limits,
            instruction_contracts=instruction_contracts,
            synchronization_contracts=frozenset(string_tuple("synchronization_contracts", allow_empty=True)),
            occupancy=(
                Occupancy.from_dict(value["occupancy"], "target.occupancy")
                if "occupancy" in value
                else None
            ),
            peak=peak,
            warps_per_warpgroup=warpgroup,
            triton_arch=triton_arch,
        )
