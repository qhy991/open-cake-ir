"""The two complete Authoring Environment treatments used by the Lab."""

from __future__ import annotations

from open_cake_ir.serialization import canonical_json_bytes

import ast, json, math
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from types import MappingProxyType
from typing import Mapping, Protocol, cast

from open_cake_ir.compiler import Assessment, Compiler, CompilerError
from open_cake_ir.compiler.frontend import FrontendError, parse as parse_python_schedule
from open_cake_ir.compiler.performance.ranking import Cost
from open_cake_ir.compiler.toolchain import validate_triton_kernel
from open_cake_ir.evaluation import LaunchableCandidate, WorkloadContract
from open_cake_ir.evaluation.cuda_manifest import CudaKernelSpec

from .pairing import backend_policy
from . import selection
from .executor import ExecutorRevision
from .faults import CandidateCompileRejected
from .build import BuildRequest, ToolchainBuilder, TritonToolchainBuilder, _ptxas_finding_rows


@dataclass(frozen=True)
class CandidateSubmission:
    """Provider-authored bytes sealed before assessment or build."""

    media_type: str
    payload: bytes
    sha256: str

    @classmethod
    def seal(cls, media_type: str, payload: bytes) -> "CandidateSubmission":
        if media_type not in {"application/vnd.open-cake.schedule+json", "text/x-cuda", "application/vnd.open-cake.triton+json", "application/vnd.open-cake.cute+json"} or not payload:
            raise ValueError("candidate submission media type or bytes differ")
        return cls(media_type, payload, sha256(payload).hexdigest())

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
        self._target = workload.target
        self._explicit_abi = isinstance(workload.document["semantics"].get("candidate_abi"), Mapping)
        self._expected = {arg.name: ("global", arg.dtype, list(arg.shape), arg.mode)
                          for arg in workload.tensor_abi(case_id)}
        route = authority_document.get("lowering_route")
        if (not isinstance(route, Mapping) or set(route) != {"backend", "entry_point"}
            or not isinstance(route["entry_point"], str)
            or not route["entry_point"].isidentifier()):
            raise ValueError("Open Cake Authoring Environment lowering route differs")
        if route["backend"] != "metal":
            backend_policy(route["backend"])
        self._route = dict(route)
        self.authority_document = json.loads(
            json.dumps(authority_document, sort_keys=True, separators=(",", ":"))
        )
        self.canonical_sha256 = sha256(
            canonical_json_bytes(self.authority_document)
        ).hexdigest()
        self._empirical_selection = None
        selection_binding = self.authority_document.get("candidate_selection")
        if selection_binding is not None:
            if self._python_enabled or self._explicit_abi:
                raise ValueError("empirical selection requires the complete-Schedule/direct-CUDA assay")
            if executor is None:
                raise ValueError("empirical selection requires the bound Executor")
            compiler_ref = self.authority_document["compiler_revision"]
            self._empirical_selection = selection._EmpiricalSelection(
                selection_binding,
                context=selection._empirical_context(
                    executor, workload_sha256=workload.canonical_sha256, case_id=case_id,
                ),
                compiler_revision_id=compiler_ref["revision_id"],
                compiler_revision_sha256=compiler_ref["canonical_sha256"],
                target=self._target,
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
                or parsed.get('target') != self._target
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

class NativeTritonEnvironment:
    """Kernel-only native source, the same Workload ABI, and the same pinned builder."""

    media_type = "application/vnd.open-cake.triton+json"

    def __init__(self, toolchain: ToolchainBuilder, *, toolchain_requirements: Mapping[str, object],
                 authority_document: Mapping[str, object], workload: WorkloadContract, case_id: str):
        self._toolchain = toolchain
        self._requirements = json.loads(json.dumps(dict(toolchain_requirements)))
        self._abi = workload.tensor_abi(case_id)
        self._target = workload.document['semantics']['target']
        # The frozen Compiler owns backend spelling (for example int32 -> *i32).
        # Consume its public spelling function instead of assuming Workload names are Triton ABI names.
        from open_cake_ir.compiler.backends.triton import pointer_type
        from open_cake_ir.compiler.ir import DType
        expected_signature = {arg.name: pointer_type(DType(arg.dtype)) for arg in self._abi}
        signature = self._requirements.get('signature')
        if (self._requirements.get('compiler') != 'triton' or self._requirements.get('target') != self._target
            or signature != expected_signature):
            raise ValueError('native Triton signature differs from the Workload ABI')
        self._requirements['signature'] = expected_signature
        self.authority_document = json.loads(json.dumps(authority_document, sort_keys=True))
        self.canonical_sha256 = sha256(canonical_json_bytes(self.authority_document)).hexdigest()

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
            CudaKernelSpec.from_dict({
                'target': self._target, 'kernel_name': requirements['kernel_entry_point'], 'grid': grid,
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
                submission.sha256, source, 'authored_source', digest, self._target,
                str(requirements['kernel_entry_point']), requirements))
        except CandidateCompileRejected as error:
            return EnvironmentResult('rejected', submission.sha256, None,
                {'stage': 'compile', 'diagnostic': error.diagnostic}, error.artifact_payloads)
        if launchable.artifact_roles.get('authored_source') != digest:
            raise ValueError('native Triton builder lost source custody')
        return EnvironmentResult('launchable', submission.sha256, launchable,
            {'stage': 'built', 'source_contract': 'triton_kernel_only_v1'})


class NativeCuTeEnvironment:
    """Kernel-only CuTe with ordered Workload pointers and the common sealed builder."""

    media_type = backend_policy("cutlass_cute_dsl").media_type

    def __init__(self, toolchain: ToolchainBuilder, *, toolchain_requirements: Mapping[str, object],
                 authority_document: Mapping[str, object], workload: WorkloadContract, case_id: str):
        from open_cake_ir.compiler.cute_toolchain import validate_cute_requirements
        self._toolchain = toolchain
        self._requirements = json.loads(json.dumps(dict(toolchain_requirements)))
        self._target = workload.target
        expected = [{'name': arg.name, 'dtype': arg.dtype} for arg in workload.tensor_abi(case_id)]
        validate_cute_requirements(self._requirements)
        if self._requirements['target'] != self._target or self._requirements['signature'] != expected:
            raise ValueError('native CuTe signature differs from the Workload ABI')
        self.authority_document = json.loads(json.dumps(authority_document, sort_keys=True))
        self.canonical_sha256 = sha256(canonical_json_bytes(self.authority_document)).hexdigest()

    def build(self, submission: CandidateSubmission) -> EnvironmentResult:
        from open_cake_ir.compiler.cute_toolchain import validate_cute_kernel
        if submission.media_type != self.media_type:
            raise ValueError('native CuTe candidate media type differs')
        try:
            document = json.loads(submission.payload)
            fields = {'kernel_source', 'grid', 'block', 'dynamic_shared_memory_bytes'}
            if not isinstance(document, Mapping) or set(document) != fields:
                raise ValueError('native CuTe candidate requires kernel_source and explicit launch metadata')
            if not isinstance(document['kernel_source'], str) or not document['kernel_source']:
                raise ValueError('native CuTe kernel_source must be nonempty text')
            requirements = {**self._requirements, **{name: document[name] for name in fields - {'kernel_source'}}}
            source = document['kernel_source'].encode()
            validate_cute_kernel(source, requirements)
            CudaKernelSpec.from_dict({'target': self._target,
                'kernel_name': requirements['kernel_entry_point'],
                **{name: requirements[name] for name in ('grid', 'block', 'dynamic_shared_memory_bytes')}})
        except (UnicodeError, json.JSONDecodeError, ValueError) as error:
            return EnvironmentResult('rejected', submission.sha256, None, {'stage': 'source_admission', 'error': str(error)})
        digest = sha256(source).hexdigest()
        try:
            launchable = self._toolchain.build(BuildRequest(submission.sha256, source, 'authored_source',
                digest, self._target, str(requirements['kernel_entry_point']), requirements))
        except CandidateCompileRejected as error:
            return EnvironmentResult('rejected', submission.sha256, None,
                {'stage': 'compile', 'diagnostic': error.diagnostic}, error.artifact_payloads)
        if launchable.artifact_roles.get('authored_source') != digest:
            raise ValueError('native CuTe builder lost source custody')
        return EnvironmentResult('launchable', submission.sha256, launchable,
            {'stage': 'built', 'source_contract': 'cute_kernel_only_v1'})
