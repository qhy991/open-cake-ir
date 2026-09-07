from __future__ import annotations
import json, tempfile
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Mapping,cast
from open_cake_ir.compiler.toolchain import compile_triton
from open_cake_ir.evaluation import LaunchableCandidate
from open_cake_ir.evaluation.cuda_manifest import CudaKernelSpec
from open_cake_ir.lab.environments import BuildRequest,ToolchainBuilder,EnvironmentResult,_ptxas_finding_rows
from open_cake_ir.lab.faults import CandidateCompileRejected,RunProtocolFault
from open_cake_ir.lab.process import run_supervised,sanitized_environment,SupervisedProcessTimeout,SupervisedProcessOutputLimit
from .cuda_manifest import CudaLaunchManifest,parse_cuda_launch_manifest

class FlashTritonToolchainBuilder:
    """Compile the canonical parametric Triton lowering to its exact CUDA CUBIN."""


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
        if request.source_role != "lowered_source":
            raise ValueError("historical Triton builder accepts Compiler lowering only")
        compilation = compile_triton(request.source, requirements)
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
        manifest = (CudaLaunchManifest.from_dict({
                        "schema_version": 1, "abi": "flash_kmeans_assign_v1", **launch}))
        manifest_bytes = json.dumps(
            manifest.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
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


class NvccToolchainBuilder:
    """Compile direct CUDA source to PTX/CUBIN/SASS without shell expansion."""

    def __init__(
        self,
        *,
        nvcc: str | Path,
        cuobjdump: str | Path,
        timeout_seconds: int = 600,
    ) -> None:
        self._nvcc = Path(nvcc).resolve(strict=True)
        self._cuobjdump = Path(cuobjdump).resolve(strict=True)
        if timeout_seconds <= 0:
            raise ValueError("CUDA toolchain timeout must be positive")
        self._timeout_seconds = timeout_seconds

    @property
    def canonical_sha256(self) -> str:
        document = {
            "nvcc_sha256": sha256(self._nvcc.read_bytes()).hexdigest(),
            "cuobjdump_sha256": sha256(self._cuobjdump.read_bytes()).hexdigest(),
            "target": "sm_100a",
            "nvcc_arguments": ["-std=c++17", "-O3", "-arch=sm_100a"],
            "nvcc_cubin_arguments": ["-Xptxas=-v"],
            "cuobjdump_arguments": ["--dump-sass"],
            "timeout_seconds": self._timeout_seconds,
        }
        return sha256(
            json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _run(self, arguments: list[str]) -> tuple[bytes, bytes]:
        try:
            completed = run_supervised(
                arguments,
                cwd=Path.cwd(),
                environment=sanitized_environment(),
                timeout_seconds=self._timeout_seconds,
            )
        except (SupervisedProcessTimeout, SupervisedProcessOutputLimit) as error:
            raise RunProtocolFault(
                "harness_fault",
                str(error),
                artifact_payloads={
                    "toolchain_stdout": error.stdout,
                    "toolchain_stderr": error.stderr,
                },
            ) from error
        if completed.returncode != 0:
            diagnostic = completed.stderr.decode("utf-8", errors="replace")[-4096:]
            raise CandidateCompileRejected(
                f"nvcc exit {completed.returncode}: {diagnostic}",
                artifact_payloads={
                    "toolchain_stdout": completed.stdout,
                    "toolchain_stderr": completed.stderr,
                },
            )
        return completed.stdout, completed.stderr

    def build(self, request: BuildRequest) -> LaunchableCandidate:
        if (
            request.toolchain_requirements.get("compiler") != "nvcc"
            or request.target != "sm_100a"
        ):
            raise ValueError("NVCC toolchain requirements differ")
        manifest = parse_cuda_launch_manifest(request.source)
        with tempfile.TemporaryDirectory(prefix="open-cake-nvcc-") as directory:
            root = Path(directory)
            source = root / "candidate.cu"
            ptx_path = root / "candidate.ptx"
            cubin_path = root / "candidate.cubin"
            source.write_bytes(request.source)
            common = [str(self._nvcc), "-std=c++17", "-O3", "-arch=sm_100a"]
            self._run(common + ["--ptx", str(source), "-o", str(ptx_path)])
            # ptxas reports registers, spills and shared memory for free on the assembly
            # pass, and writes them to stderr. Discarding them left this arm's author
            # blind to the resource facts its own toolchain had already measured.
            _, assembler_output = self._run(
                common + ["-Xptxas=-v", "--cubin", str(source), "-o", str(cubin_path)]
            )
            # ptxas also prints its own wall clock, which is a fact about this machine at
            # this moment rather than about the candidate. Every other artifact role here
            # is a function of the source alone, and this one must be too or the same
            # candidate would seal under a different digest on every build.
            resource_report = b"".join(
                line + b"\n"
                for line in assembler_output.splitlines()
                if b"Compile time" not in line
            )
            ptx = ptx_path.read_bytes()
            cubin = cubin_path.read_bytes()
            sass, _ = self._run([str(self._cuobjdump), "--dump-sass", str(cubin_path)])
        if not cubin.startswith(b"\x7fELF"):
            raise ValueError("NVCC did not produce an ELF CUBIN")
        manifest_bytes = json.dumps(
            manifest.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        payloads = {
            "authored_source": request.source,
            "ptx": ptx,
            "cubin": cubin,
            "sass": sass,
            "launch_manifest": manifest_bytes,
        }
        # An artifact payload must carry bytes, so a silent assembler contributes no role
        # rather than an empty one that would fail custody.
        if resource_report:
            payloads["toolchain_resource_report"] = resource_report
        return LaunchableCandidate(
            candidate_sha256=request.candidate_sha256,
            target=manifest.target,
            entry_point=manifest.kernel_name,
            artifact_roles={
                role: sha256(payload).hexdigest() for role, payload in payloads.items()
            },
            launch_spec_sha256=sha256(manifest_bytes).hexdigest(),
            artifact_payloads=payloads,
        )


class DirectCudaEnvironment:
    """Direct CUDA/PTX source plus the matched pinned CUDA toolchain."""

    media_type = "text/x-cuda"

    def __init__(
        self,
        toolchain: ToolchainBuilder,
        *,
        toolchain_requirements: Mapping[str, object],
        authority_document: Mapping[str, object],
    ) -> None:
        self._toolchain = toolchain
        self._requirements = MappingProxyType(dict(toolchain_requirements))
        self.authority_document = json.loads(
            json.dumps(authority_document, sort_keys=True, separators=(",", ":"))
        )
        self.canonical_sha256 = sha256(
            json.dumps(
                self.authority_document, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()

    def build(self, submission: CandidateSubmission) -> EnvironmentResult:
        if submission.media_type != self.media_type:
            raise ValueError("direct CUDA candidate media type differs")
        try:
            manifest = parse_cuda_launch_manifest(submission.payload)
        except (UnicodeError, ValueError) as error:
            return EnvironmentResult(
                "rejected",
                submission.sha256,
                None,
                MappingProxyType({"stage": "manifest", "error": str(error)}),
            )
        try:
            launchable = self._toolchain.build(
                BuildRequest(
                    candidate_sha256=submission.sha256,
                    source=submission.payload,
                    source_role="authored_source",
                    source_sha256=submission.sha256,
                    target=manifest.target,
                    entry_point=manifest.kernel_name,
                    toolchain_requirements=self._requirements,
                )
            )
        except CandidateCompileRejected as error:
            return EnvironmentResult(
                "rejected",
                submission_sha256=submission.sha256,
                launchable=None,
                feedback={"stage": "compile", "diagnostic": error.diagnostic},
                artifact_payloads=error.artifact_payloads,
            )
        if launchable.artifact_roles.get("authored_source") != submission.sha256:
            raise ValueError("direct CUDA toolchain lost authored-source custody")
        return EnvironmentResult(
            "launchable",
            submission.sha256,
            launchable,
            MappingProxyType(
                {"stage": "built", "findings": _ptxas_finding_rows(launchable)}
            ),
        )
