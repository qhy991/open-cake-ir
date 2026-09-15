"""Shared exact-HIP runtime custody for generated Open Cake Triton kernels."""

from __future__ import annotations

import importlib
import importlib.util
import json
import re
import subprocess
import tempfile
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
        getattr(properties, "gcnArchName", None) != expected[1]
        or type(getattr(properties, "warp_size", None)) is not int
        or properties.warp_size != expected[2]
    ):
        raise RuntimeError("HIP device properties differ from the exact Target")
    return torch, triton, properties


def load_generated_module(
    lowering: object,
) -> tuple[object, tempfile.TemporaryDirectory[str]]:
    """Load one generated module without placing source or bytecode in the checkout."""

    directory = tempfile.TemporaryDirectory(prefix="open-cake-gfx1151-")
    source_path = Path(directory.name) / "generated.py"
    source_path.write_text(lowering.source, encoding="utf-8")
    specification = importlib.util.spec_from_file_location(
        f"open_cake_gfx1151_{lowering.source_sha256[:16]}", source_path
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
    required = {
        "group_segment_fixed_size",
        "private_segment_fixed_size",
        "kernarg_size",
        "wavefront_size32",
        "uses_dynamic_stack",
        "next_free_vgpr",
        "next_free_sgpr",
        "shared_vgpr_count",
        "workgroup_processor_mode",
    }
    if (
        not required.issubset(fields) or fields["wavefront_size32"] != 1
        or any(fields[name] not in (0, 1) for name in ("uses_dynamic_stack", "workgroup_processor_mode"))
    ):
        raise ValueError("AMDGCN resource declarations differ")
    return {
        "kernel_name": kernels[0],
        "wave_size": 32,
        "vgpr_count": fields["next_free_vgpr"],
        "sgpr_count": fields["next_free_sgpr"],
        "shared_vgpr_count": fields["shared_vgpr_count"],
        "lds_bytes_per_workgroup": fields["group_segment_fixed_size"],
        "scratch_bytes_per_workitem": fields["private_segment_fixed_size"],
        "kernarg_bytes": fields["kernarg_size"],
        "uses_dynamic_stack": bool(fields["uses_dynamic_stack"]),
        "workgroup_processor_mode": bool(fields["workgroup_processor_mode"]),
        "occupancy_derived": False,
        "occupancy_limit": "gfx1151_target_facts_unavailable",
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
