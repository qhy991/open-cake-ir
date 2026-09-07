"""The two complete Authoring Environment treatments used by the Lab."""

from __future__ import annotations

import ast
import json
import math
import tempfile
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol, cast

from open_cake_ir.compiler import (
    Assessment, Compiler, CompilerError, Finding, FindingCategory, FindingSeverity,
)
from open_cake_ir.compiler.empirical_cost import EmpiricalCostModel
from open_cake_ir.compiler.ranking import Cost
from open_cake_ir.compiler.toolchain import compile_triton, project_triton_kernel, validate_triton_kernel
from open_cake_ir.compiler.frontend import parse as parse_python_schedule, FrontendError
from open_cake_ir.evaluation.core import TensorLaunchManifest
from open_cake_ir.evaluation import (
    CudaLaunchManifest,
    LaunchableCandidate,
    WorkloadContract,
    parse_cuda_launch_manifest,
)

from .faults import CandidateCompileRejected, RunProtocolFault
from .executor import ExecutorRevision
from .process import (
    SupervisedProcessOutputLimit,
    SupervisedProcessTimeout,
    run_supervised,
    sanitized_environment,
)


_EMPIRICAL_SELECTION = "external_empirical_advisory_v1"


def _empirical_context(
    executor: ExecutorRevision, *, workload_sha256: str, case_id: str
) -> dict[str, object]:
    """Reference the actual Flash assay and its admitted, frozen runtime owner.

    The Executor content identity binds the helper, assay source and host closure;
    this projection does not infer equivalence between suppliers' free-text contexts.
    """
    packages = executor.document["host_environment"]["packages"]
    return {
        "timer": "flashinfer.testing.utils.bench_gpu_time_with_cupti;use_cuda_graph=false",
        "cache_protocol": "cold_l2_cache=true",
        "runtime": {
            "compiler_version": packages.get("triton"),
            "executor_revision": executor.canonical_sha256,
        },
        "input_scope": json.dumps(
            {"workload_contract_sha256": workload_sha256, "case_id": case_id},
            sort_keys=True, separators=(",", ":"),
        ),
    }


class _EmpiricalSelection:
    """One Study-bound advisory model, shared by execution and CPU replay."""

    def __init__(
        self, binding: Mapping[str, object], *, context: Mapping[str, object],
        compiler_revision_id: str, compiler_revision_sha256: str, target: str,
    ) -> None:
        if set(binding) != {"kind", "model"} or binding.get("kind") != _EMPIRICAL_SELECTION:
            raise ValueError("empirical selection binding differs")
        self.model = EmpiricalCostModel(binding["model"])
        self._context_differences = tuple(
            key for key in ("timer", "cache_protocol", "runtime", "input_scope")
            if binding["model"]["context"].get(key) != context.get(key)
        ) + (("fields",) if set(binding["model"]["context"]) != set(context) else ())
        self._compiler_revision_id = compiler_revision_id
        self._compiler_revision_sha256 = compiler_revision_sha256
        self._target = target

    def estimate(self, schedule: dict) -> dict[str, object]:
        result = self.model.estimate(
            schedule, compiler_revision_id=self._compiler_revision_id,
            compiler_revision_sha256=self._compiler_revision_sha256,
            target=self._target,
        )
        reason = None
        if self._context_differences:
            reason = "model context differs in " + ", ".join(self._context_differences) + "; exact Workload/case/assay/Executor binding required"
        elif result["covered"] and (
            not math.isfinite(result["predicted_kernel_us"])
            or result["predicted_kernel_us"] <= 0
            or any(not math.isfinite(value) for value in result["empirical_range_us"])
        ):
            reason = "model prediction or empirical range is not finite and positive"
        if reason is not None:
            result.update(covered=False, predicted_kernel_us=None, empirical_range_us=None, reason=reason)
        # The full supplier document is already frozen in the CampaignLock. Arbitrary
        # context/provenance maps (including raw observations) are not agent feedback.
        return {key: result[key] for key in (
            "kind", "model_id", "model_compiler_revision_id",
            "model_compiler_revision_sha256", "target", "covered",
            "predicted_kernel_us", "empirical_range_us", "reason",
        )}


@dataclass(frozen=True)
class CandidateSubmission:
    """Provider-authored bytes sealed before assessment or build."""

    media_type: str
    payload: bytes
    sha256: str

    @classmethod
    def seal(cls, media_type: str, payload: bytes) -> "CandidateSubmission":
        if media_type not in {"application/vnd.open-cake.schedule+json", "text/x-cuda", "application/vnd.open-cake.triton+json"} or not payload:
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


class TritonToolchainBuilder:
    """Compile the canonical parametric Triton lowering to an exact sm_100a CUBIN."""

    def __init__(self, *, workload=None, case_id=None, isolated_compiler=None):
        self._workload = workload
        self._case_id = case_id
        self._isolated = isolated_compiler
        if (workload is None) != (case_id is None):
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
        if self._workload is not None:
            if self._isolated is None:
                raise RunProtocolFault("harness_fault", "paired Triton requires filesystem-isolated compilation")
            kernel_source = (project_triton_kernel(request.source, requirements)
                             if request.source_role == "lowered_source" else request.source)
            validate_triton_kernel(kernel_source, requirements)
            compilation = self._isolated.compile(kernel_source, requirements)
        else:
            if request.source_role != "lowered_source":
                raise ValueError("historical Triton builder accepts Compiler lowering only")
            compilation = compile_triton(request.source, requirements)
        stages = compilation.artifacts
        kernel_name = compilation.entry_point
        launch = {
            "target": request.target, "kernel_name": kernel_name, "grid": grid,
            "block": [compilation.threads_per_cta, 1, 1],
            "dynamic_shared_memory_bytes": compilation.dynamic_shared_bytes,
            "hidden_null_pointer_parameters": 2,
        }
        manifest = (TensorLaunchManifest.for_workload(self._workload, self._case_id, **launch)
                    if self._workload is not None else CudaLaunchManifest.from_dict({
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

    empirical_cost: Mapping[str, object] | None = None
    """Conditional point estimate, separate from calibrated Compiler ranking."""

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
        executor: ExecutorRevision | None = None,
    ) -> None:
        if compiler.state != "released":
            raise ValueError("Open Cake Environment requires a released Compiler Revision")
        self._compiler = compiler
        self._toolchain = toolchain
        self._workload = workload
        self._case_id = case_id
        self._workload_sha256 = workload.canonical_sha256
        self._python_enabled = authority_document.get("input_format") == "schedule_or_python_v1"
        self._explicit_abi = isinstance(workload.document["semantics"].get("candidate_abi"), Mapping)
        if self._explicit_abi:
            self._expected = {arg.name: ("global", arg.dtype, list(arg.shape), arg.mode)
                              for arg in workload.tensor_abi(case_id)}
        else:
            # Closed historical input boundary. New contracts never infer this ABI.
            shape = workload.case(case_id).get("shape")
            if workload.document.get("operator") != "flash_kmeans_assign" or not isinstance(shape, Mapping):
                raise ValueError("historical Open Cake Workload ABI is unsupported")
            self._expected = {
                "tokens": ("global", "bf16", [shape["B"], shape["N"], shape["D"]], "input"),
                "centroids": ("global", "bf16", [shape["B"], shape["K"], shape["D"]], "input"),
                "centroid_sq": ("global", "fp32", [shape["B"], shape["K"]], "input"),
                "assignments": ("global", "int32", [shape["B"], shape["N"]], "output"),
            }
        route = authority_document.get("lowering_route")
        if (not isinstance(route, Mapping) or set(route) != {"backend", "entry_point"}
            or route["backend"] != "triton" or not isinstance(route["entry_point"], str)
            or not route["entry_point"].isidentifier()
            or not self._explicit_abi and route["entry_point"] != "cake_flash_kmeans_assign"):
            raise ValueError("Open Cake Authoring Environment lowering route differs")
        self._route = dict(route)
        self.authority_document = json.loads(
            json.dumps(authority_document, sort_keys=True, separators=(",", ":"))
        )
        self.canonical_sha256 = sha256(
            json.dumps(
                self.authority_document, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        self._empirical_selection = None
        selection = self.authority_document.get("candidate_selection")
        if selection is not None:
            if self._python_enabled or self._explicit_abi:
                raise ValueError("empirical selection requires the Flash/direct-CUDA assay")
            if executor is None:
                raise ValueError("empirical selection requires the bound Executor")
            compiler_ref = self.authority_document["compiler_revision"]
            self._empirical_selection = _EmpiricalSelection(
                selection,
                context=_empirical_context(executor, workload_sha256=workload.canonical_sha256, case_id=case_id),
                compiler_revision_id=compiler_ref["revision_id"],
                compiler_revision_sha256=compiler_ref["canonical_sha256"],
                target="sm_100a",
            )

    @staticmethod
    def _finding_rows(assessment: Assessment, source=None) -> list[dict[str, object]]:
        """Project findings for the agent.

        A rejection carries the blocking findings that caused it; an acceptance retains
        reports and hints with their original category and severity. These diagnostics
        describe modeled limits or missing commitments, not measured performance.

        What a finding blocks is two facts, not one. A Schedule can be accepted and still
        not lowerable -- its kinds are well-formed and this backend has no body for one --
        and compressing that into a single `blocking` flag told an author the finding that
        stopped their candidate was not blocking anything.
        """

        return [
            {
                **item.to_dict(),
                **({"source_location": asdict(location)} if source is not None
                   and (location := source.location_for(item.path)) is not None else {}),
            }
            for item in assessment.findings + assessment.guidance
        ]

    def build(self, submission: CandidateSubmission) -> EnvironmentResult:
        if submission.media_type != self.media_type:
            raise ValueError("Open Cake candidate media type differs")
        source = None
        try:
            parsed = json.loads(submission.payload)
            if isinstance(parsed, Mapping) and set(parsed) == {"python_source"}:
                if not self._python_enabled or not isinstance(parsed["python_source"], str):
                    raise ValueError("Python IR submission is outside the admitted Authoring Environment")
                source = parse_python_schedule(parsed["python_source"], filename="candidate.ir.py")
                parsed = source.document
            if not isinstance(parsed, Mapping):
                raise CompilerError("Schedule root must be an object")
            metadata = parsed.get("metadata")
            buffers = parsed.get("buffers")
            route = parsed.get("lowering")
            if route != self._route:
                raise ValueError(
                    "Schedule lowering route is outside the admitted Authoring Environment"
                )
            if (
                not isinstance(metadata, Mapping)
                or metadata.get("workload_contract_sha256") != self._workload_sha256
                or not isinstance(buffers, list)
            ):
                raise ValueError("Schedule Workload binding differs")
            by_name = {
                item.get("name"): item
                for item in buffers
                if isinstance(item, Mapping) and isinstance(item.get("name"), str)
            }
            expected = self._expected
            if self._explicit_abi and [item.get("name") for item in buffers
                    if isinstance(item, Mapping) and item.get("space") == "global"] != list(expected):
                raise ValueError("Schedule external tensor order differs from the Workload ABI")
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
            if self._empirical_selection is not None and (
                assessment.compiler_revision_id != self._empirical_selection._compiler_revision_id
                or assessment.compiler_revision_sha256 != self._empirical_selection._compiler_revision_sha256
                or assessment.target != self._empirical_selection._target
            ):
                raise ValueError("assessment Compiler Revision or target differs from the bound Environment")
        except (UnicodeError, json.JSONDecodeError, CompilerError, ValueError) as error:
            return EnvironmentResult(
                "rejected",
                submission.sha256,
                None,
                MappingProxyType({"stage": "assessment", "error": str(error),
                    **({"code": error.code, "source_location": asdict(error.location)}
                       if isinstance(error, FrontendError) else {})}),
            )
        if not assessment.lowering_eligible:
            return EnvironmentResult(
                "rejected",
                submission.sha256,
                None,
                MappingProxyType(
                    {
                        "stage": "assessment",
                        "findings": self._finding_rows(assessment, source),
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
                    entry_point=lowering.route.entry_point,
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
                {"stage": "built", "findings": self._finding_rows(assessment, source)}
            ),
            cost=next(iter(self._compiler.rank([assessment])[0]), None),
            semantic_sha256=(
                digest
                if isinstance(digest := assessment.analysis.get("semantic_sha256"), str)
                else None
            ),
            empirical_cost=(
                self._empirical_selection.estimate(json.loads(assessment.schedule_bytes))
                if self._empirical_selection is not None else None
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


class NativeTritonEnvironment:
    """Kernel-only native source, the same Workload ABI, and the same pinned builder."""

    media_type = "application/vnd.open-cake.triton+json"

    def __init__(self, toolchain: ToolchainBuilder, *, toolchain_requirements: Mapping[str, object],
                 authority_document: Mapping[str, object], workload: WorkloadContract, case_id: str):
        self._toolchain = toolchain
        self._requirements = json.loads(json.dumps(dict(toolchain_requirements)))
        self._abi = workload.tensor_abi(case_id)
        # The frozen Compiler owns backend spelling (for example int32 -> *i32).
        # Consume its existing table instead of assuming Workload names are Triton ABI names.
        from open_cake_ir.compiler.emit_triton import _TritonEmitter
        from open_cake_ir.compiler.ir import DType
        expected_signature = {arg.name: _TritonEmitter._POINTER[DType(arg.dtype)] for arg in self._abi}
        signature = self._requirements.get('signature')
        if (self._requirements.get('compiler') != 'triton' or self._requirements.get('target') != 'sm_100a'
            or signature != expected_signature):
            raise ValueError('native Triton signature differs from the Workload ABI')
        self._requirements['signature'] = expected_signature
        self.authority_document = json.loads(json.dumps(authority_document, sort_keys=True))
        self.canonical_sha256 = sha256(json.dumps(self.authority_document, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def build(self, submission: CandidateSubmission) -> EnvironmentResult:
        if submission.media_type != self.media_type:
            raise ValueError('native Triton candidate media type differs')
        try:
            document = json.loads(submission.payload)
            if not isinstance(document, Mapping) or set(document) != {'kernel_source', 'compile_constants', 'compile_options', 'grid'}:
                raise ValueError('native Triton candidate requires kernel_source and explicit compile/launch metadata')
            if not isinstance(document['kernel_source'], str) or not document['kernel_source']:
                raise ValueError('native Triton kernel_source must be nonempty text')
            requirements = dict(self._requirements)
            constants = document['compile_constants']
            options = document['compile_options']
            grid = document['grid']
            if (not isinstance(constants, Mapping) or set(constants) != set(requirements['compile_constants'])
                or any(type(value) not in {int, float, bool}
                       or type(value) is int and not -(2**63) <= value < 2**63
                       or type(value) is float and not math.isfinite(value) for value in constants.values())
                or not isinstance(options, Mapping) or set(options) != set(requirements['compile_options'])
                or any(type(value) is not int or value <= 0 for value in options.values())
                or not isinstance(grid, list) or len(grid) != 3
                or any(type(value) is not int or value <= 0 for value in grid)):
                raise ValueError('native Triton compile constants/options/grid differ from the declared interface')
            # Use the existing structural launch checker before any target compilation.
            CudaLaunchManifest.from_dict({'schema_version': 1, 'abi': 'flash_kmeans_assign_v1',
                'target': 'sm_100a', 'kernel_name': requirements['kernel_entry_point'], 'grid': grid,
                'block': [options.get('num_warps', 4) * 32, 1, 1], 'dynamic_shared_memory_bytes': 0})
            requirements.update(compile_constants=dict(constants), compile_options=dict(options), grid=grid)
            source = document['kernel_source'].encode()
            validate_triton_kernel(source, requirements)
            kernel = ast.parse(source).body[-1]
            if [arg.arg for arg in kernel.args.args[:len(self._abi)]] != [arg.name for arg in self._abi]:
                raise ValueError('native Triton tensor argument order differs from the Workload ABI')
        except (UnicodeError, json.JSONDecodeError, ValueError) as error:
            return EnvironmentResult('rejected', submission.sha256, None, {'stage': 'source_admission', 'error': str(error)})
        digest = sha256(source).hexdigest()
        try:
            launchable = self._toolchain.build(BuildRequest(
                submission.sha256, source, 'authored_source', digest, 'sm_100a',
                str(requirements['kernel_entry_point']), requirements))
        except CandidateCompileRejected as error:
            return EnvironmentResult('rejected', submission.sha256, None,
                {'stage': 'compile', 'diagnostic': error.diagnostic}, error.artifact_payloads)
        if launchable.artifact_roles.get('authored_source') != digest:
            raise ValueError('native Triton builder lost source custody')
        return EnvironmentResult('launchable', submission.sha256, launchable,
            {'stage': 'built', 'source_contract': 'triton_kernel_only_v1'})
