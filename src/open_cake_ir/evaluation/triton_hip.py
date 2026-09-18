"""Shared exact-HIP runtime custody for generated Open Cake Triton kernels."""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from typing import Mapping, cast


_AMDHSA_KERNEL = re.compile(r"^\s*\.amdhsa_kernel\s+(\S+)\s*$", re.MULTILINE)
_AMDHSA_FIELD = re.compile(
    r"^\s*\.amdhsa_(?P<name>[a-z0-9_]+)\s+(?P<value>[0-9]+)\s*$",
    re.MULTILINE,
)


def require_object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return cast(Mapping[str, object], value)


_DEVICE_ARCH_FEATURES = re.compile(r"(?::[a-z0-9]+[+-])*")


def device_arch_matches(observed: object, declared: str) -> bool:
    """Whether a device's reported ISA name is the declared one, features aside.

    A device names its ISA with the features it was built with, and the Target declares
    the bare ISA: a BW1101 reports `gfx938:sramecc+:xnack-` for a Target that declares
    `gfx938`, and comparing the two as equal strings refuses the device it describes.
    Only well-formed feature suffixes are admitted after the declared name, which is what
    keeps `gfx938` from matching a `gfx9380`. This is the same distinction the emitted
    object's `amdhsa.target` line already needs, where the toolchain writes a third
    spelling again -- `gfx938:xnack-`.
    """
    if not isinstance(observed, str) or not declared:
        return False
    if not observed.startswith(declared):
        return False
    return _DEVICE_ARCH_FEATURES.fullmatch(observed[len(declared):]) is not None


def admit_exact_hip(
    requirements: Mapping[str, object],
) -> tuple[object, object, object]:
    """Admit one visible ROCm GPU matching the exact lowering Target."""

    requirements = require_object(requirements, "lowering requirements")
    target = require_object(requirements.get("triton_target"), "triton_target")
    if (
        set(target) != {"backend", "arch", "warp_size"}
        or target.get("backend") != "hip"
        or not isinstance(target.get("arch"), str) or not target["arch"]
        or requirements.get("target") != target["arch"]
        or type(target.get("warp_size")) is not int or target["warp_size"] <= 0
        or requirements.get("binary_role") != "hsaco"
        or requirements.get("assembly_role") != "amdgcn"
    ):
        raise ValueError("HIP lowering Target or artifact requirements differ")
    expected = (target["backend"], target["arch"], target["warp_size"])
    torch = importlib.import_module("torch")
    triton = importlib.import_module("triton")
    version = getattr(torch, "version", None)
    if (
        not isinstance(getattr(version, "hip", None), str) or not version.hip
        or getattr(version, "cuda", None) is not None
    ):
        raise RuntimeError("a ROCm PyTorch build is required")
    if not torch.cuda.is_available():
        raise RuntimeError("the runner requires exactly one visible HIP GPU")
    count = torch.cuda.device_count()
    if type(count) is not int or count != 1:
        raise RuntimeError("the runner requires exactly one visible HIP GPU")
    properties = torch.cuda.get_device_properties(0)
    driver = importlib.import_module("triton.runtime.driver").driver
    runtime_target = driver.active.get_current_target()
    observed = (
        getattr(runtime_target, "backend", None),
        getattr(runtime_target, "arch", None),
        getattr(runtime_target, "warp_size", None),
    )
    if type(observed[2]) is not int or observed != expected:
        raise RuntimeError(
            f"Triton runtime Target {observed!r} differs from lowering {expected!r}"
        )
    if (
        not device_arch_matches(getattr(properties, "gcnArchName", None), expected[1])
        or type(getattr(properties, "warp_size", None)) is not int
        or properties.warp_size != expected[2]
    ):
        raise RuntimeError("HIP device properties differ from the exact Target")
    return torch, triton, properties


@dataclass(frozen=True)
class HipDeviceAdmission:
    """What a HIP evaluation was admitted against, recorded before anything launches.

    The peer of `CudaDeviceAdmission`, and shaped by the same two questions its consumers
    ask: which broker job owns this process, and which physical device answered. It is not
    a copy of the CUDA one -- there is no CUPTI handle here and no exclusive cluster
    lease, because a DCU is reached through the local broker that serializes one machine's
    single device, the same allocation an Apple GPU uses.
    """

    broker_job_id: str
    target: str
    device_arch: str
    device_name: str
    warp_size: int
    gpu_uuid: str


def hip_admission_requirements(target_id: str) -> dict[str, object]:
    """State the exact device contract for one AMDGCN target, from its own route.

    `admit_exact_hip` takes the lowering requirements a build was compiled under. An
    evaluation worker is handed a sealed candidate and not those requirements, and the
    answer is not to add a field carrying them: the Target document declares which code
    object, ISA and lane width this target compiles and runs under, and the route facts
    are read from it here exactly as the emitter wrote them into the build's contract.
    This layer is outside the compile jail and may open the document.
    """
    from open_cake_ir.compiler.backends.triton import target_route_facts
    from open_cake_ir.compiler.target import declared_target
    from open_cake_ir.compiler.toolchain import triton_route

    route = triton_route({"target": target_id, **target_route_facts(declared_target(target_id))})
    if route.gpu_backend != "hip":
        raise ValueError(f"{target_id!r} does not lower through the HIP backend")
    return {
        "target": target_id,
        "binary_role": route.binary_role,
        "assembly_role": route.text_role,
        "triton_target": {"backend": route.gpu_backend, "arch": str(route.architecture),
                          "warp_size": route.warp_size},
    }


def observe_local_hip(target_id: str) -> HipDeviceAdmission:
    """Admit this process's local-broker job and the one visible HIP device.

    `admit_exact_hip` owns the device half and is reused verbatim; what is added here is
    the allocation half, which the CUDA path gets from its cluster broker and this one
    from the local serializer. A job id issued for another device family is refused: a
    DCU evaluation recorded under a `metal-` job would misattribute the run exactly the
    way a DCU latency recorded as CUPTI would misattribute the measurement.
    """
    from .local_broker import observe_local_job

    requirements = hip_admission_requirements(target_id)
    job = observe_local_job("hip")
    torch, _triton, properties = admit_exact_hip(requirements)
    target = require_object(requirements["triton_target"], "triton_target")
    uuid = getattr(properties, "uuid", None)
    return HipDeviceAdmission(
        broker_job_id=job,
        target=str(target["arch"]),
        device_arch=str(getattr(properties, "gcnArchName", "")),
        device_name=str(getattr(properties, "name", "")),
        warp_size=int(properties.warp_size),
        # A DTK device reports no UUID; the record says so rather than inventing one or
        # leaving the reader to guess what an empty string meant.
        gpu_uuid=str(uuid) if uuid else "not_reported_by_this_runtime",
    )


_AMDGPU_METADATA = re.compile(
    r"^\s*\.amdgpu_metadata\s*$(.*?)^\s*\.end_amdgpu_metadata\s*$",
    re.MULTILINE | re.DOTALL,
)
_KERNARG_SIZE = re.compile(r"^\s*\.kernarg_segment_size:\s*(\d+)\s*$", re.MULTILINE)
_ARGUMENT_KIND = re.compile(r"^\s*\.value_kind:\s*(?P<kind>\S+)\s*$")
_METADATA_KEY = re.compile(r"^(?P<indent>\s*)(?P<marker>-\s+)?\.(?P<name>[a-z_]+):")


def _key_depth(match: "re.Match[str]") -> int:
    """Where a key sits in the document, counting a list marker as indentation.

    `  - .args:` and `    .kernarg_segment_size:` are siblings: the first key of a list
    item is written after the dash, and the dash occupies the columns the item's later
    keys are indented to. Reading the marker as zero-width made `.args` look shallower
    than its own siblings, so the scan ran past the end of the list and into
    `.unfolded_args`, counting every argument twice.
    """
    return len(match.group("indent")) + len(match.group("marker") or "")


def _argument_kinds(block: str) -> list[str]:
    """The value kinds under `.args`, and nothing else.

    This emitter writes the same entries twice -- once under `.args` and again under
    `.unfolded_args` -- so a value-kind count over the whole block counts every argument
    twice, which is what it did. The list ends at the next key written at the same
    indentation as `.args` itself, which is how the document nests.
    """
    lines = block.splitlines()
    kinds: list[str] = []
    depth: int | None = None
    for line in lines:
        key = _METADATA_KEY.match(line)
        if depth is None:
            if key is not None and key.group("name") == "args":
                depth = _key_depth(key)
            continue
        if key is not None and _key_depth(key) <= depth and key.group("name") != "args":
            break
        kind = _ARGUMENT_KIND.match(line)
        if kind is not None:
            kinds.append(kind.group("kind"))
    return kinds


def amdgcn_kernarg_pointers(payload: bytes) -> int:
    """Count the pointer arguments the emitted kernel actually declares.

    The launch passes one address per declared argument, and the count is the kernel's
    own fact: its `.amdgpu_metadata` lists every argument under `.args` with a value kind,
    and a Triton AMDGCN kernel's are all `global_buffer`. Deriving it instead from the
    route's scratch fields was wrong and the device said so -- an rmsnorm taking three
    tensors declares five, the launcher passed four, and the kernel read its fifth pointer
    out of uninitialized kernarg memory. One launch survived that because this kernel
    never dereferences its scratch; a second did not.

    Two of the kernel's own statements have to agree: the number of entries under `.args`,
    and `.kernarg_segment_size` at eight bytes per pointer. A kernel that takes scalars
    breaks that relation, and is refused here rather than counted as if it did not.
    """
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("AMDGCN artifact must contain assembly bytes")
    try:
        source = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("AMDGCN artifact is not UTF-8 assembly") from error
    blocks = _AMDGPU_METADATA.findall(source)
    if len(blocks) != 1:
        raise ValueError("AMDGCN artifact must carry exactly one .amdgpu_metadata block")
    kinds = _argument_kinds(blocks[0])
    if not kinds:
        raise ValueError("AMDGCN kernel declares no arguments")
    if any(kind != "global_buffer" for kind in kinds):
        raise ValueError(
            "AMDGCN kernel declares "
            + ", ".join(sorted(set(kinds)))
            + "; this launch packs pointer arguments alone"
        )
    segment = _KERNARG_SIZE.search(blocks[0])
    if segment is None:
        raise ValueError("AMDGCN kernel declares no kernarg segment size")
    if int(segment.group(1)) != 8 * len(kinds):
        raise ValueError(
            f"AMDGCN kernarg segment is {segment.group(1)} bytes for {len(kinds)} pointer "
            "arguments; this kernel does not take pointers alone"
        )
    return len(kinds)


def amdgcn_resource_record(payload: bytes) -> dict[str, object]:
    """Extract exact AMDHSA resource declarations from one emitted assembly artifact."""

    if not isinstance(payload, bytes) or not payload:
        raise ValueError("AMDGCN artifact must contain assembly bytes")
    try:
        source = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("AMDGCN artifact is not UTF-8 assembly") from error
    kernels = _AMDHSA_KERNEL.findall(source)
    if len(kernels) != 1:
        raise ValueError("AMDGCN artifact must declare exactly one kernel")
    blocks = re.findall(
        r"^\s*\.amdhsa_kernel\s+\S+\s*$(.*?)^\s*\.end_amdhsa_kernel\s*$",
        source, re.MULTILINE | re.DOTALL,
    )
    if len(blocks) != 1 or len(re.findall(r"^\s*\.end_amdhsa_kernel\b", source, re.MULTILINE)) != 1:
        raise ValueError("AMDGCN kernel resource block differs")
    fields: dict[str, int] = {}
    for line in blocks[0].splitlines():
        if not line.strip().startswith(".amdhsa_"):
            continue
        match = _AMDHSA_FIELD.fullmatch(line)
        if match is None or match.group("name") in fields:
            raise ValueError("AMDGCN resource declarations are malformed or duplicated")
        fields[match.group("name")] = int(match.group("value"))
    # Every AMDGCN kernel declares these, whatever generation it targets.
    required = {
        "group_segment_fixed_size",
        "private_segment_fixed_size",
        "kernarg_size",
        "uses_dynamic_stack",
        "next_free_vgpr",
        "next_free_sgpr",
    }
    # These three are RDNA's. `wavefront_size32` selects between the two wave modes gfx10+
    # has, `workgroup_processor_mode` selects WGP against CU mode, and `shared_vgpr_count`
    # describes a register file gfx9 does not have. A CDNA-class kernel emits none of
    # them -- measured on gfx938, whose 43 directives include all six above and not one of
    # these -- so requiring them described one generation and refused the other. Their
    # absence is reported as absence, not defaulted to a value the kernel never declared.
    rdna_only = {"wavefront_size32", "workgroup_processor_mode", "shared_vgpr_count"}
    if not required.issubset(fields):
        raise ValueError("AMDGCN resource declarations differ")
    if fields["uses_dynamic_stack"] not in (0, 1):
        raise ValueError("AMDGCN uses_dynamic_stack declaration differs")
    for name in rdna_only & set(fields):
        if name != "shared_vgpr_count" and fields[name] not in (0, 1):
            raise ValueError(f"AMDGCN {name} declaration differs")
    # The wave mode is what the kernel declares, not what this reader assumes. gfx10+
    # says so with wavefront_size32; a kernel that never mentions it is wave64.
    wave_size = 32 if fields.get("wavefront_size32") == 1 else 64
    return {
        "kernel_name": kernels[0],
        "wave_size": wave_size,
        "vgpr_count": fields["next_free_vgpr"],
        "sgpr_count": fields["next_free_sgpr"],
        "shared_vgpr_count": fields.get("shared_vgpr_count"),
        "lds_bytes_per_workgroup": fields["group_segment_fixed_size"],
        "scratch_bytes_per_workitem": fields["private_segment_fixed_size"],
        "kernarg_bytes": fields["kernarg_size"],
        "uses_dynamic_stack": bool(fields["uses_dynamic_stack"]),
        "workgroup_processor_mode": (
            None if "workgroup_processor_mode" not in fields
            else bool(fields["workgroup_processor_mode"])
        ),
        "occupancy_derived": False,
        # Named for the target it is missing for, rather than for one target this reader
        # was first written against.
        "occupancy_limit": "target_facts_unavailable",
    }
