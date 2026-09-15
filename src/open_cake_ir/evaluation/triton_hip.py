"""Shared exact-HIP runtime custody for generated Open Cake Triton kernels."""

from __future__ import annotations

import importlib
import importlib.util
import json
import re
import subprocess
import tempfile
from types import MappingProxyType
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Mapping, cast


ARTIFACT_ROLES = ("source", "ttir", "ttgir", "llir", "amdgcn", "hsaco")
_AMDHSA_KERNEL = re.compile(r"^\s*\.amdhsa_kernel\s+(\S+)\s*$", re.MULTILINE)
_AMDHSA_FIELD = re.compile(
    r"^\s*\.amdhsa_(?P<name>[a-z0-9_]+)\s+(?P<value>[0-9]+)\s*$",
    re.MULTILINE,
)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def require_object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return cast(Mapping[str, object], value)


def artifact_bytes(value: object, role: str) -> bytes:
    if role not in ARTIFACT_ROLES:
        raise ValueError(f"Triton artifact role {role!r} differs")
    if isinstance(value, bytes) and value:
        return value
    if role != "hsaco" and isinstance(value, str) and value:
        return value.encode()
    raise ValueError(f"Triton artifact {role!r} has unsupported bytes")


def git_state(project_root: Path) -> dict[str, object]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_root,
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout
    return {"revision": revision, "tree_clean": not bool(status)}


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
    answer is not to add a field carrying them: the route is already the owner of which
    backend, ISA and lane width this target compiles and runs under, and reading it here
    is reading the same fact the build read, not a second copy of it.
    """
    from open_cake_ir.compiler.toolchain import triton_route

    route = triton_route(target_id)
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


def load_generated_module(
    lowering: object,
) -> tuple[object, tempfile.TemporaryDirectory[str]]:
    """Load one generated module without placing source or bytecode in the checkout."""

    directory = tempfile.TemporaryDirectory(prefix="open-cake-amdgcn-")
    source_path = Path(directory.name) / "generated.py"
    source_path.write_text(lowering.source, encoding="utf-8")
    specification = importlib.util.spec_from_file_location(
        f"open_cake_amdgcn_{lowering.source_sha256[:16]}", source_path
    )
    if specification is None or specification.loader is None:
        directory.cleanup()
        raise RuntimeError("generated AMD module specification failed")
    module = importlib.util.module_from_spec(specification)
    try:
        specification.loader.exec_module(module)
    except BaseException:
        directory.cleanup()
        raise
    return module, directory


def extract_artifacts(compiled: object) -> dict[str, bytes]:
    assembly = require_object(getattr(compiled, "asm", None), "Triton artifacts")
    if not set(ARTIFACT_ROLES).issubset(assembly) or {"cubin", "ptx"}.intersection(assembly):
        raise ValueError("Triton HIP artifact roles differ")
    payloads = {role: artifact_bytes(assembly[role], role) for role in ARTIFACT_ROLES}
    if not payloads["hsaco"].startswith(b"\x7fELF"):
        raise RuntimeError("Triton did not produce an ELF HSACO")
    return payloads


def artifact_records(payloads: Mapping[str, bytes]) -> dict[str, dict[str, object]]:
    if not isinstance(payloads, Mapping) or set(payloads) != set(ARTIFACT_ROLES):
        raise ValueError("Triton HIP artifact record roles differ")
    for role, payload in payloads.items():
        if not isinstance(payload, bytes) or not payload:
            raise ValueError(f"Triton artifact {role!r} has unsupported bytes")
    if not payloads["hsaco"].startswith(b"\x7fELF"):
        raise ValueError("Triton HIP artifact record is not ELF HSACO")
    return {
        role: {"sha256": sha256(payload).hexdigest(), "size_bytes": len(payload)}
        for role, payload in sorted(payloads.items())
    }


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


def resolve_new_external_directory(project_root: Path, value: Path) -> Path:
    """Resolve, but do not create, a new evidence directory outside the checkout."""

    candidate = value.absolute()
    path = candidate.parent.resolve(strict=True) / candidate.name
    if path.exists() or path.is_symlink():
        raise ValueError("evidence directory must be a new path")
    if path == project_root or project_root in path.parents:
        raise ValueError("evidence directory must be outside the checkout")
    return path


def write_new_json(path: Path, value: object) -> None:
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(value) + b"\n")


@dataclass(frozen=True)
class LoadedHipCandidate:
    """One sealed AMDGCN candidate, admitted and loaded, before common Evaluation.

    This is the seam the CUDA path reaches through `LoadedCudaCandidate`: an arm's output
    becomes a thing that can be launched, with its artifacts and its declared resource
    allocation recorded beside it. The two are not built alike and should not be. CUDA
    loads a CUBIN through the driver API and launches the function itself; here Triton
    owns the module and its own launch, so what this adds is the admission, the exact
    artifact set, the AMDGCN resource record and one entry point -- not a second launcher.

    It holds no device handle and no GPU state of its own. `close()` removes the
    generated source that backs the loaded module, and nothing else.
    """

    target: str
    entry_point: str
    source_sha256: str
    artifacts: Mapping[str, dict[str, object]]
    resources: Mapping[str, object]
    device_arch: str
    warp_size: int
    entry: object
    _payloads: Mapping[str, bytes] = field(repr=False, default_factory=dict)
    _directory: object = field(repr=False, default=None)

    def payload(self, role: str) -> bytes:
        """The exact bytes of one admitted artifact role."""
        if role not in self._payloads:
            raise ValueError(f"AMDGCN candidate has no artifact role {role!r}")
        return self._payloads[role]

    def close(self) -> None:
        if self._directory is not None:
            self._directory.cleanup()


def load_hip_candidate(
    lowering: object, requirements: Mapping[str, object],
) -> LoadedHipCandidate:
    """Admit the exact device, compile the lowering, and return its launchable entry.

    Every refusal here belongs to this route and names it. The device is admitted before
    anything is compiled, the artifact roles are the AMDGCN ones and the absence of a
    `ptx` or `cubin` role is part of that check, and the entry point named by the lowering
    has to exist in the module the generated source defines.

    No timing happens here and none can: this returns something launchable, and what a
    launch is worth is a measurement this route does not yet have a source for.
    """
    requirements = require_object(requirements, "lowering requirements")
    _, triton, properties = admit_exact_hip(requirements)
    entry_point = requirements.get("host_entry_point") or getattr(lowering, "entry_point", None)
    if not isinstance(entry_point, str) or not entry_point:
        raise ValueError("AMDGCN candidate requires its host entry point")

    module, directory = load_generated_module(lowering)
    try:
        entry = getattr(module, entry_point, None)
        if entry is None or not callable(entry):
            raise ValueError(
                f"generated AMDGCN module defines no callable entry {entry_point!r}"
            )
        compiled = _compile_through_the_module(module, requirements, triton)
        payloads = extract_artifacts(compiled)
        records = artifact_records(payloads)
        resources = amdgcn_resource_record(payloads["amdgcn"])
    except BaseException:
        directory.cleanup()
        raise
    return LoadedHipCandidate(
        target=str(requirements["target"]),
        entry_point=entry_point,
        source_sha256=str(lowering.source_sha256),
        artifacts=MappingProxyType(records),
        resources=MappingProxyType(dict(resources)),
        device_arch=str(getattr(properties, "gcnArchName", "")),
        warp_size=int(getattr(properties, "warp_size")),
        entry=entry,
        _payloads=MappingProxyType(dict(payloads)),
        _directory=directory,
    )


def _compile_through_the_module(
    module: object, requirements: Mapping[str, object], triton: object,
) -> object:
    """Compile the kernel the generated module defines, for its declared Target."""

    kernel_name = requirements.get("kernel_entry_point")
    signature = requirements.get("signature")
    constants = requirements.get("compile_constants")
    options = requirements.get("compile_options")
    if (not isinstance(kernel_name, str) or not kernel_name
            or any(not isinstance(item, Mapping) for item in (signature, constants, options))):
        raise ValueError("AMDGCN compile contract differs")
    kernel = getattr(module, kernel_name, None)
    if kernel is None:
        raise ValueError(f"generated AMDGCN module defines no kernel {kernel_name!r}")
    target = require_object(requirements["triton_target"], "triton_target")
    gpu_target = importlib.import_module("triton.backends.compiler").GPUTarget
    compiler = importlib.import_module("triton.compiler")
    return compiler.compile(
        compiler.ASTSource(kernel, dict(signature), dict(constants)),
        target=gpu_target(target["backend"], target["arch"], target["warp_size"]),
        options=dict(options),
    )
