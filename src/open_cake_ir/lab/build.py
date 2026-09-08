"""Typed source-to-artifact handoff and the isolated Triton toolchain builder."""

from __future__ import annotations

from open_cake_ir.serialization import canonical_json_bytes

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping, Protocol

from open_cake_ir.compiler import Finding, FindingCategory, FindingSeverity
from open_cake_ir.compiler.toolchain import project_triton_kernel, validate_triton_kernel
from open_cake_ir.evaluation import LaunchableCandidate
from open_cake_ir.evaluation.core import TensorLaunchManifest

from .faults import RunProtocolFault


@dataclass(frozen=True)
class BuildRequest:
    """Arm-owned source handed to one pinned toolchain implementation."""

    candidate_sha256: str
    source: bytes
    source_role: str
    source_sha256: str
    target: str
    entry_point: str
    toolchain_requirements: Mapping[str, object]

    def __post_init__(self) -> None:
        digests = (self.candidate_sha256, self.source_sha256)
        if (
            any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in digests
            )
            or sha256(self.source).hexdigest() != self.source_sha256
            or self.source_role not in {"lowered_source", "authored_source"}
            or not self.target
            or not self.entry_point
        ):
            raise ValueError("toolchain BuildRequest authority differs")


class ToolchainBuilder(Protocol):
    """Build source into a sealed LaunchableCandidate or raise."""

    def build(self, request: BuildRequest) -> LaunchableCandidate:
        """Build with the Campaign-pinned toolchain and retain artifact roles."""


class TritonToolchainBuilder:
    """Compile the canonical parametric Triton lowering to its exact CUDA CUBIN."""

    def __init__(self, *, workload, case_id, isolated_compiler=None):
        self._workload = workload
        self._case_id = case_id
        self._isolated = isolated_compiler
        if workload is None or case_id is None:
            raise ValueError("Triton builder Workload and case must be bound together")

    def build(self, request: BuildRequest) -> LaunchableCandidate:
        requirements = request.toolchain_requirements
        if (
            requirements.get("compiler") != "triton"
            or requirements.get("source_language") != "python"
            or requirements.get("target") != request.target
        ):
            raise ValueError("Triton toolchain requirements differ")
        grid = requirements.get("grid")
        if not isinstance(grid, list) or len(grid) != 3:
            raise ValueError("Triton launch grid differs")
        if request.target != self._workload.document['semantics'].get('target'):
            raise ValueError("Triton build target differs from the Workload")
        if self._isolated is None:
            raise RunProtocolFault("harness_fault", "paired Triton requires filesystem-isolated compilation")
        kernel_source = (project_triton_kernel(request.source, requirements)
                         if request.source_role == "lowered_source" else request.source)
        validate_triton_kernel(kernel_source, requirements)
        compilation = self._isolated.compile(kernel_source, requirements)
        if (compilation.target != request.target
            or compilation.entry_point != requirements.get('kernel_entry_point')):
            raise ValueError("Triton compilation target or entry point differs from its request")
        stages = compilation.artifacts
        kernel_name = compilation.entry_point
        launch = {
            "target": request.target, "kernel_name": kernel_name, "grid": grid,
            "block": [compilation.threads_per_cta, 1, 1],
            "dynamic_shared_memory_bytes": compilation.dynamic_shared_bytes,
            "hidden_null_pointer_parameters": 2,
        }
        manifest = (TensorLaunchManifest.for_workload(self._workload, self._case_id, **launch))
        manifest_bytes = canonical_json_bytes(manifest.as_dict())
        payloads = {
            request.source_role: request.source,
            "compiler_expanded_source": stages["source"],
            "ttir": stages["ttir"],
            "ttgir": stages["ttgir"],
            "llir": stages["llir"],
            "ptx": stages["ptx"],
            "cubin": stages["cubin"],
            "launch_manifest": manifest_bytes,
        }
        return LaunchableCandidate(
            candidate_sha256=request.candidate_sha256,
            target=request.target,
            entry_point=kernel_name,
            artifact_roles={
                role: sha256(payload).hexdigest() for role, payload in payloads.items()
            },
            launch_spec_sha256=sha256(manifest_bytes).hexdigest(),
            artifact_payloads=payloads,
        )


def _ptxas_finding_rows(
    launchable: LaunchableCandidate,
) -> list[dict[str, object]]:
    """Project what ptxas measured, in the shape the Open Cake arm's findings use.

    The arms are matched on one static channel each: the Compiler's verifier for a
    Schedule, the CUDA toolchain's own assembler for authored source. Both report what
    bounds the artifact before it runs, so an advantage measured between them is the
    representation rather than one author having been told its register count.

    Reported verbatim rather than parsed into fields. ptxas owns this text, and re-deriving
    numbers from it here would make this a second, staler authority on the same fact.
    """

    report = launchable.artifact_payloads.get("toolchain_resource_report", b"")
    lines = [
        line.strip()
        for line in report.decode("utf-8", errors="replace").splitlines()
        if "ptxas info" in line or "bytes spill" in line or "bytes stack frame" in line
    ]
    if not lines:
        return []
    return [
        Finding(
            "TOOLCHAIN_RESOURCE_REPORT", launchable.entry_point, " ".join(lines),
            FindingCategory.HARDWARE_CONFORMANCE, FindingSeverity.REPORT,
        ).to_dict()
    ]
