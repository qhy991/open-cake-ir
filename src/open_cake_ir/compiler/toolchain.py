"""Explicit-target Triton compilation and CUBIN inspection without GPU handles.

Both the Lab builder and the offline profile tool use this path. Importing this module
needs neither Triton nor CUDA; those are required only for an explicit compilation.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import importlib.metadata
from pathlib import Path
import re
import subprocess
import tempfile
from types import MappingProxyType
from typing import Mapping

from .compiled_resources import CompiledResources, parse_cuobjdump_resources


@dataclass(frozen=True)
class TritonCompilation:
    source: bytes
    target: str
    entry_point: str
    artifacts: Mapping[str, bytes]
    threads_per_cta: int
    dynamic_shared_bytes: int
    compiler_version: str


def compile_triton(source: bytes, requirements: Mapping[str, object]) -> TritonCompilation:
    """Compile a complete emitted source; never launch it or initialize GPU handles."""
    if (
        requirements.get("compiler") != "triton"
        or requirements.get("source_language") != "python"
        or requirements.get("target") != "sm_100a"
        or not source
    ):
        raise ValueError("offline Triton compilation requires the explicit sm_100a contract")
    name = requirements.get("kernel_entry_point")
    signature = requirements.get("signature")
    constants = requirements.get("compile_constants")
    options = requirements.get("compile_options")
    if not isinstance(name, str) or not name or any(
        not isinstance(item, Mapping) for item in (signature, constants, options)
    ):
        raise ValueError("Triton compile contract differs")
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, compile as triton_compile

    with tempfile.TemporaryDirectory(prefix="open-cake-triton-") as directory:
        path = Path(directory) / "lowered.py"
        path.write_bytes(source)
        spec = importlib.util.spec_from_file_location("open_cake_offline_lowering", path)
        if spec is None or spec.loader is None:
            raise ValueError("Triton lowering module specification failed")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        kernel = getattr(module, name, None)
        if kernel is None:
            raise ValueError("Triton lowering kernel entry point is missing")
        compiled = triton_compile(
            ASTSource(kernel, dict(signature), dict(constants)),
            target=GPUTarget("cuda", 100, 32), options=dict(options),
        )
        artifacts = {}
        for role in ("source", "ttir", "ttgir", "llir", "ptx", "cubin"):
            payload = compiled.asm[role]
            if isinstance(payload, str):
                payload = payload.encode("utf-8")
            if not isinstance(payload, bytes) or not payload:
                raise ValueError(f"Triton artifact {role!r} differs")
            artifacts[role] = payload
        # .asm and .metadata are compile-time objects. .run, _init_handles(), n_regs
        # and n_spills require the GPU runtime and must not be used in this path.
        metadata = compiled.metadata
        if metadata.name != name or metadata.num_ctas != 1:
            raise ValueError("offline resource model requires the requested single-CTA kernel")
        if metadata.global_scratch_size or metadata.profile_scratch_size:
            raise ValueError("offline resource model does not cover auxiliary global scratch")
        if not artifacts["cubin"].startswith(b"\x7fELF") or re.search(
            rb"^\.target\s+sm_100a(?:\s|,|$)", artifacts["ptx"], re.MULTILINE
        ) is None:
            raise ValueError("Triton output does not match the exact sm_100a CUBIN target")
        return TritonCompilation(
            source, "sm_100a", name, MappingProxyType(artifacts),
            int(metadata.num_warps) * 32, int(metadata.shared), importlib.metadata.version("triton"),
        )


def inspect_triton_resources(compilation: TritonCompilation, cuobjdump: str | Path) -> CompiledResources:
    """Inspect compiler output using NVIDIA's binary utility, with no GPU runtime."""
    inspector = str(Path(cuobjdump).resolve(strict=True))
    version = subprocess.check_output([inspector, "--version"], text=True, timeout=20).strip()
    with tempfile.TemporaryDirectory(prefix="open-cake-cubin-") as directory:
        path = Path(directory) / "kernel.cubin"
        path.write_bytes(compilation.artifacts["cubin"])
        report = subprocess.check_output(
            [inspector, "--dump-resource-usage", str(path)], text=True, timeout=20,
        )
    return CompiledResources(
        source_sha256=sha256(compilation.source).hexdigest(),
        cubin_sha256=sha256(compilation.artifacts["cubin"]).hexdigest(),
        target=compilation.target, entry_point=compilation.entry_point,
        threads_per_cta=compilation.threads_per_cta,
        dynamic_shared_bytes=compilation.dynamic_shared_bytes,
        compiler_version=compilation.compiler_version, inspector_version=version,
        **parse_cuobjdump_resources(report, compilation.entry_point),
    )
