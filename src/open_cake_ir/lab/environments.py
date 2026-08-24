"""The two complete Authoring Environment treatments used by the Lab."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol, cast

from open_cake_ir.compiler import Assessment, Compiler, CompilerError
from open_cake_ir.compiler.ranking import Cost
from open_cake_ir.evaluation import (
    CudaLaunchManifest,
    LaunchableCandidate,
    WorkloadContract,
    parse_cuda_launch_manifest,
)

from .faults import CandidateCompileRejected, RunProtocolFault
from .process import (
    SupervisedProcessOutputLimit,
    SupervisedProcessTimeout,
    run_supervised,
    sanitized_environment,
)


@dataclass(frozen=True)
class CandidateSubmission:
    """Provider-authored bytes sealed before assessment or build."""

    media_type: str
    payload: bytes
    sha256: str

    @classmethod
    def seal(cls, media_type: str, payload: bytes) -> "CandidateSubmission":
        if media_type not in {"application/vnd.open-cake.schedule+json", "text/x-cuda"} or not payload:
            raise ValueError("candidate submission media type or bytes differ")
        return cls(media_type, payload, sha256(payload).hexdigest())


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


def _artifact_bytes(value: object, role: str) -> bytes:
    if isinstance(value, str):
        return value.encode()
    if isinstance(value, bytes):
        return value
    raise ValueError(f"toolchain artifact {role!r} has unsupported bytes")


class TritonToolchainBuilder:
    """Compile the canonical parametric Triton lowering to an exact sm_100a CUBIN."""

    def build(self, request: BuildRequest) -> LaunchableCandidate:
        requirements = request.toolchain_requirements
        if (
            requirements.get("compiler") != "triton"
            or requirements.get("source_language") != "python"
            or requirements.get("target") != request.target
        ):
            raise ValueError("Triton toolchain requirements differ")
        kernel_name = requirements.get("kernel_entry_point")
        signature = requirements.get("signature")
        constants = requirements.get("compile_constants")
        options = requirements.get("compile_options")
        grid = requirements.get("grid")
        if (
            not isinstance(kernel_name, str)
            or not isinstance(signature, Mapping)
            or not isinstance(constants, Mapping)
            or not isinstance(options, Mapping)
            or not isinstance(grid, list)
            or len(grid) != 3
        ):
            raise ValueError("Triton compile contract differs")
        with tempfile.TemporaryDirectory(prefix="open-cake-triton-") as directory:
            source_path = Path(directory) / "lowered.py"
            source_path.write_bytes(request.source)
            module_spec = importlib.util.spec_from_file_location(
                f"open_cake_lowering_{request.source_sha256[:16]}", source_path
            )
            if module_spec is None or module_spec.loader is None:
                raise ValueError("Triton lowering module specification failed")
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            kernel = getattr(module, kernel_name, None)
            if kernel is None:
                raise ValueError("Triton lowering kernel entry point is missing")
            from triton.backends.compiler import GPUTarget
            from triton.compiler import ASTSource
            from triton.compiler import compile as triton_compile

            compiled = triton_compile(
                ASTSource(kernel, dict(signature), dict(constants)),
                target=GPUTarget("cuda", 100, 32),
                options=dict(options),
            )
            stages = {
                role: _artifact_bytes(compiled.asm[role], role)
                for role in ("source", "ttir", "ttgir", "llir", "ptx", "cubin")
            }
        if not stages["cubin"].startswith(b"\x7fELF"):
            raise ValueError("Triton did not produce an ELF CUBIN")
        manifest = CudaLaunchManifest.from_dict(
            {
                "schema_version": 1,
                "abi": "flash_kmeans_assign_v1",
                "target": request.target,
                "kernel_name": kernel_name,
                "grid": grid,
                "block": [int(options["num_warps"]) * 32, 1, 1],
                "dynamic_shared_memory_bytes": int(compiled.metadata.shared),
                "hidden_null_pointer_parameters": 2,
            }
        )
        manifest_bytes = json.dumps(
            manifest.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        payloads = {
            "lowered_source": request.source,
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
        {
            "code": "TOOLCHAIN_RESOURCE_REPORT",
            "path": launchable.entry_point,
            "message": " ".join(lines),
            "blocks_acceptance": False,
            "blocks_lowering": False,
        }
    ]


@dataclass(frozen=True)
class EnvironmentResult:
    """One treatment response; Lab, not the environment, decides the next Turn."""

    disposition: str
    submission_sha256: str
    launchable: LaunchableCandidate | None
    feedback: Mapping[str, object]
    artifact_payloads: Mapping[str, bytes] = field(default_factory=dict)
    cost: Cost | None = None
    """What the pre-GPU filter can say about this candidate's order, if anything.

    Supplied by the environment because the environment owns the Compiler; the Lab only
    sorts by it. The Lab applies costs only when every launchable member of the set has
    one. If the environment has no cost model, or the model declines any member, the
    whole launchable set retains the order it was written -- unknown is not slower.
    """

    semantic_sha256: str | None = None
    """This candidate's identity as a program rather than as bytes, if known.

    The paper's first stage asks for *structurally distinct* candidates. Two Schedules
    that differ only in a name or in the order of independent declarations are one kernel
    with two spellings, and searching both spends a second measurement to learn what the
    first already said. Supplied by the environment for the same reason `cost` is: the
    environment owns the Compiler, and this is the Compiler's own semantic digest.
    """

    def __post_init__(self) -> None:
        if self.disposition not in {"launchable", "rejected"}:
            raise ValueError("Authoring Environment disposition differs")
        if (self.disposition == "launchable") != (self.launchable is not None):
            raise ValueError("Authoring Environment launchable boundary differs")
        if self.launchable is not None and self.launchable.candidate_sha256 != self.submission_sha256:
            raise ValueError("Authoring Environment replaced the sealed submission")
        if self.semantic_sha256 is not None and (
            len(self.semantic_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.semantic_sha256)
        ):
            raise ValueError("Authoring Environment semantic identity differs")


class AuthoringEnvironment(Protocol):
    """Complete assigned treatment from candidate bytes to launchable seam."""

    media_type: str
    canonical_sha256: str
    authority_document: Mapping[str, object]

    def build(self, submission: CandidateSubmission) -> EnvironmentResult:
        """Assess/build once and return bounded same-Run feedback."""


class OpenCakeEnvironment:
    """Structured Schedule, frozen Compiler Revision, and its toolchain."""

    media_type = "application/vnd.open-cake.schedule+json"

    def __init__(
        self,
        compiler: Compiler,
        toolchain: ToolchainBuilder,
        *,
        authority_document: Mapping[str, object],
        workload: WorkloadContract,
        case_id: str,
    ) -> None:
        if compiler.state != "released":
            raise ValueError("Open Cake Environment requires a released Compiler Revision")
        self._compiler = compiler
        self._toolchain = toolchain
        case = workload.case(case_id)
        shape = case.get("shape")
        if not isinstance(shape, Mapping):
            raise ValueError("Open Cake Workload case shape differs")
        self._workload_sha256 = workload.canonical_sha256
        self._shape = {name: int(shape[name]) for name in ("B", "N", "K", "D")}
        self._profile = str(authority_document.get("schedule_profile"))
        if self._profile != "flash_kmeans_b32_smoke":
            raise ValueError("Open Cake Authoring Environment profile differs")
        self.authority_document = json.loads(
            json.dumps(authority_document, sort_keys=True, separators=(",", ":"))
        )
        self.canonical_sha256 = sha256(
            json.dumps(
                self.authority_document, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()

    @staticmethod
    def _finding_rows(assessment: Assessment) -> list[dict[str, object]]:
        """Project findings for the agent.

        One shape for both dispositions. A rejection carries the blocking findings that
        caused it; an acceptance carries the reports that survived it, which is where the
        analysis attribution reaches the agent. Dropping them on acceptance would leave a
        working candidate with no stated reason for the performance it got.

        What a finding blocks is two facts, not one. A Schedule can be accepted and still
        not lowerable -- its kinds are well-formed and this backend has no body for one --
        and compressing that into a single `blocking` flag told an author the finding that
        stopped their candidate was not blocking anything.
        """

        return [
            {
                "code": item.code,
                "path": item.path,
                "message": item.message,
                "blocks_acceptance": item.blocks_acceptance,
                "blocks_lowering": item.blocks_lowering,
            }
            for item in assessment.findings
        ]

    def build(self, submission: CandidateSubmission) -> EnvironmentResult:
        if submission.media_type != self.media_type:
            raise ValueError("Open Cake candidate media type differs")
        try:
            parsed = json.loads(submission.payload)
            if not isinstance(parsed, Mapping):
                raise CompilerError("Schedule root must be an object")
            metadata = parsed.get("metadata")
            buffers = parsed.get("buffers")
            if not isinstance(metadata, Mapping) or metadata.get("profile") != self._profile:
                raise ValueError("Schedule profile is outside the admitted Authoring Environment")
            if (
                metadata.get("workload_contract_sha256") != self._workload_sha256
                or not isinstance(buffers, list)
            ):
                raise ValueError("Schedule Workload binding differs")
            by_name = {
                item.get("name"): item
                for item in buffers
                if isinstance(item, Mapping) and isinstance(item.get("name"), str)
            }
            expected = {
                "tokens": ("global", "bf16", [self._shape["B"], self._shape["N"], self._shape["D"]], "input"),
                "centroids": ("global", "bf16", [self._shape["B"], self._shape["K"], self._shape["D"]], "input"),
                "centroid_sq": ("global", "fp32", [self._shape["B"], self._shape["K"]], "input"),
                "assignments": ("global", "int32", [self._shape["B"], self._shape["N"]], "output"),
            }
            if any(
                name not in by_name
                or (
                    by_name[name].get("space"),
                    by_name[name].get("dtype"),
                    by_name[name].get("shape"),
                    by_name[name].get("mode"),
                )
                != contract
                for name, contract in expected.items()
            ):
                raise ValueError("Schedule external tensor contract differs from the Workload")
            assessment = self._compiler.assess(cast(Mapping[str, object], parsed))
        except (UnicodeError, json.JSONDecodeError, CompilerError, ValueError) as error:
            return EnvironmentResult(
                "rejected",
                submission.sha256,
                None,
                MappingProxyType({"stage": "assessment", "error": str(error)}),
            )
        if not assessment.lowering_eligible:
            return EnvironmentResult(
                "rejected",
                submission.sha256,
                None,
                MappingProxyType(
                    {
                        "stage": "assessment",
                        "findings": self._finding_rows(assessment),
                        "calibration_available": assessment.calibration_available,
                    }
                ),
            )
        lowering = self._compiler.lower(assessment)
        try:
            launchable = self._toolchain.build(
                BuildRequest(
                    candidate_sha256=submission.sha256,
                    source=lowering.source.encode(),
                    source_role="lowered_source",
                    source_sha256=lowering.source_sha256,
                    target=lowering.target,
                    entry_point=lowering.entry_point,
                    toolchain_requirements=lowering.toolchain_requirements,
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
        if launchable.artifact_roles.get("lowered_source") != lowering.source_sha256:
            raise ValueError("Open Cake toolchain lost lowered-source custody")
        return EnvironmentResult(
            "launchable",
            submission.sha256,
            launchable,
            MappingProxyType(
                {"stage": "built", "findings": self._finding_rows(assessment)}
            ),
            cost=next(iter(self._compiler.rank([assessment])[0]), None),
            semantic_sha256=(
                digest
                if isinstance(digest := assessment.analysis.get("semantic_sha256"), str)
                else None
            ),
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
