"""Closed CLI composition for a live matched-search Campaign."""

from __future__ import annotations
from open_cake_ir.tasks.flash_kmeans.environment import FlashTritonToolchainBuilder
from open_cake_ir.tasks.workloads import load_workload

import json
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Mapping, cast

from open_cake_ir.compiler import Compiler
from open_cake_ir.tasks.flash_kmeans.cuda_manifest import CudaLaunchManifest
from open_cake_ir.tasks.flash_kmeans.portfolio_runtime import CuptiPortfolioAssay
from open_cake_ir.evaluation import LoadedCudaCandidate, WorkloadContract, observe_exclusive_b200
from open_cake_ir.tasks.flash_kmeans.portfolio import PortfolioArtifact, PortfolioEvaluationReceipt
from open_cake_ir.evaluation.benchmark import StrictCuptiBenchmark

from open_cake_ir.lab.core import CampaignLock, CampaignRef
from open_cake_ir.tasks.runtime import TaskLab
from open_cake_ir.lab.environments import BuildRequest, NativeTritonEnvironment, TritonToolchainBuilder
from open_cake_ir.tasks.flash_kmeans.environment import DirectCudaEnvironment, NvccToolchainBuilder
from open_cake_ir.tasks.environments import TaskOpenCakeEnvironment as OpenCakeEnvironment
from open_cake_ir.lab.executor import ExecutorRevision
from open_cake_ir.lab.faults import RunProtocolFault
from open_cake_ir.tasks.flash_kmeans.seed import KernelSeed, lower_specialists
from open_cake_ir.lab.providers import CANDIDATE_SET_ENVELOPE_V1, CodexInvocationBuilder, CodexProviderAdapter, CodexRunProvider, ProviderQualificationReceipt, required_live_provider_qualification_scope
from open_cake_ir.lab.pairing import comparison_arm, bind_baseline
from open_cake_ir.lab.claude import ClaudeInvocationBuilder, ClaudeProviderAdapter, ClaudeRunProvider
from open_cake_ir.lab.provider_policy import provider_harness
from open_cake_ir.lab.metal_build import MetalArchiveHost, MetalToolchainBuilder
from open_cake_ir.lab.triton_build import IsolatedTritonCompiler
from open_cake_ir.lab.runtime import BoundedBrokerEvaluator, CommandBrokerSubmitter, broker_execution_sha256, load_runtime_config
from open_cake_ir.lab.task_package import materialize_task_package
from open_cake_ir.lab.bindings import qualification_path, load_baseline_bundle, external_file
from open_cake_ir.evaluation.paired import candidate_identity, paired_protocol


def _object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return cast(Mapping[str, object], value)


def _raw_reference_path(
    root: Path, value: object, context: str
) -> Path:
    reference = _object(value, context)
    if set(reference) != {"path", "sha256"}:
        raise ValueError(f"{context} fields differ")
    raw_path = reference["path"]
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{context}.path differs")
    relative = PurePosixPath(raw_path)
    if relative.is_absolute() or ".." in relative.parts or "\\" in raw_path:
        raise ValueError(f"{context}.path is unsafe")
    unresolved = root / raw_path
    if unresolved.is_symlink():
        raise ValueError(f"{context} custody differs")
    path = unresolved.resolve(strict=True)
    if root not in path.parents or not path.is_file():
        raise ValueError(f"{context} custody differs")
    if sha256(path.read_bytes()).hexdigest() != reference["sha256"]:
        raise ValueError(f"{context} bytes differ")
    return path


def _admit_executor(root: Path, lock: CampaignLock) -> tuple[ExecutorRevision, object]:
    execution = _object(lock.document["execution"], "campaign_lock.execution")
    revision = ExecutorRevision.load_reference(
        root, execution["executor_revision"], "execution.executor_revision"
    )
    return revision, revision.admit_host()


class _LivePortfolioAssay:
    """Delay compile/load until Lab has opened the Portfolio Terminal Archive."""

    def __init__(
        self,
        *,
        compiler: Compiler,
        seed: KernelSeed,
        workload: WorkloadContract,
        case_ids: list[str],
        evaluation_protocol: Mapping[str, object],
        device: str,
        synchronize: object,
        torch_module: object,
        cupti_helper: object,
    ) -> None:
        self._compiler = compiler
        self._seed = seed
        self._workload = workload
        self._case_ids = case_ids
        self._protocol = dict(evaluation_protocol)
        self.protocol_sha256 = sha256(
            json.dumps(self._protocol, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self._device = device
        self._synchronize = synchronize
        self._torch = torch_module
        self._cupti_helper = cupti_helper
        self._observer = None
        self._delegate: CuptiPortfolioAssay | None = None

    def set_observer(self, observer: object) -> None:
        self._observer = observer

    def prepare(self, _campaign_lock: CampaignLock) -> PortfolioArtifact:
        retained: dict[str, bytes] = {}
        loaded: dict[str, LoadedCudaCandidate] = {}
        try:
            admission = observe_exclusive_b200()
            if self._observer is not None:
                self._observer(
                    "gpu_admitted",
                    {
                        "device_name": admission.device_name,
                        "compute_capability": list(admission.compute_capability),
                        "gpu_uuid": admission.gpu_uuid,
                        "broker_job_id": admission.broker_job_id,
                        "mode": admission.mode,
                    },
                )
            stream = self._torch.cuda.current_stream().cuda_stream
            cases = {
                case_id: self._workload.case(case_id)["shape"]
                for case_id in self._case_ids
            }
            lowered = lower_specialists(self._compiler, self._seed, cases)
            toolchain = FlashTritonToolchainBuilder()
            candidates = {}
            for item in lowered:
                request = BuildRequest(
                    candidate_sha256=item.lowering.schedule_sha256,
                    source=item.lowering.source.encode(),
                    source_role="lowered_source",
                    source_sha256=item.lowering.source_sha256,
                    target=item.lowering.target,
                    entry_point=item.lowering.route.entry_point,
                    toolchain_requirements=item.lowering.toolchain_requirements,
                )
                candidate = toolchain.build(request)
                candidates[item.case_id] = candidate
                retained.update(
                    {
                        f"{item.case_id}_{role}": payload
                        for role, payload in candidate.artifact_payloads.items()
                    }
                )
                if self._observer is not None:
                    self._observer(
                        "specialist_compiled",
                        {
                            "case_id": item.case_id,
                            "candidate_record_sha256": candidate.canonical_sha256,
                            "artifact_roles": dict(candidate.artifact_roles),
                        },
                    )
            artifact = PortfolioArtifact.build(
                self._workload, self._seed.canonical_sha256, candidates
            )
            for entry in artifact.entries:
                loaded[entry.case_id] = LoadedCudaCandidate.load(
                    entry.candidate,
                    entry.candidate.artifact_payloads["cubin"],
                    CudaLaunchManifest.from_dict(
                        json.loads(entry.candidate.artifact_payloads["launch_manifest"])
                    ),
                    admission,
                )
                if self._observer is not None:
                    self._observer(
                        "specialist_loaded",
                        {
                            "case_id": entry.case_id,
                            "candidate_record_sha256": entry.candidate.canonical_sha256,
                            "cubin_sha256": entry.candidate.artifact_roles["cubin"],
                        },
                    )
            flush_buffer = self._torch.empty(
                int(self._protocol["l2_flush_bytes"]),
                dtype=self._torch.uint8,
                device=self._device,
            )

            def flush_l2() -> None:
                flush_buffer.add_(1)

            self._delegate = CuptiPortfolioAssay(
                artifact=artifact,
                workload=self._workload,
                evaluation_protocol=self._protocol,
                loaded_candidates=loaded,
                cupti_benchmark=StrictCuptiBenchmark(self._cupti_helper),
                synchronize=self._synchronize,
                flush_l2=flush_l2,
                stream=stream,
                device=self._device,
            )
            if self._observer is not None:
                self._delegate.set_observer(self._observer)
            return artifact
        except Exception as error:
            for candidate in loaded.values():
                if not candidate.closed:
                    try:
                        candidate.close(synchronize=self._synchronize)
                    except BaseException:
                        pass
            raise RunProtocolFault(
                "harness_fault",
                "portfolio preparation failed",
                artifact_payloads=retained,
            ) from error

    def evaluate(self, campaign_lock: CampaignLock) -> PortfolioEvaluationReceipt:
        if self._delegate is None:
            raise ValueError("Portfolio Assay was not prepared")
        return self._delegate.evaluate(campaign_lock)


def _load_workload_binding(root: Path, lock: CampaignLock):
    reference = _object(lock.document["workload"], "workload")
    path = (external_file(root, reference["path"], "Workload")
            if Path(str(reference["path"])).is_absolute() else root / str(reference["path"]))
    workload = load_workload(path)
    if workload.canonical_sha256 != reference["canonical_sha256"]:
        raise ValueError("runtime Workload differs from the Campaign Lock")
    return reference, path, workload


def execute_matched_from_config(
    project_root: str | Path,
    lock: CampaignLock,
    runtime_config_path: str | Path,
    evidence_root: str | Path,
) -> CampaignRef:
    """Compose the current concrete adapters from one closed host configuration."""

    if lock.study_kind != "matched_search":
        raise ValueError("live matched composition requires a matched Campaign Lock")
    root = Path(project_root).resolve(strict=True)
    executor, _ = _admit_executor(root, lock)
    resolved = _object(lock.document["resolved_inputs"], "campaign_lock.resolved_inputs")
    arms = _object(resolved["arm_environments"], "arm_environments")
    comparison = comparison_arm(arms)
    paired_triton = comparison == "native_triton"
    metal = comparison is None and arms["open_cake"]["lowering_route"]["backend"] == "metal"
    if comparison is None and not metal:
        raise ValueError("single-environment live composition requires the declared Metal backend")
    config = load_runtime_config(runtime_config_path,
                                 toolchain_kind="metal" if metal else "triton" if paired_triton else "nvcc")
    provider_config, toolchain_config, broker_config = (config[name] for name in ("provider", "toolchain", "broker"))
    open_arm = _object(arms["open_cake"], "arm_environments.open_cake")
    direct_arm = _object(arms[comparison], f"arm_environments.{comparison}") if comparison is not None else {}
    provider_authority = _object(open_arm["provider"], "arm_environments.provider")
    budget = _object(resolved["budget"], "campaign_lock.resolved_inputs.budget")
    qualification_ref = _object(
        provider_authority["qualification"], "arm_environments.provider.qualification"
    )
    _, receipt_path = qualification_path(root, qualification_ref['path'], 'provider qualification')
    qualification = ProviderQualificationReceipt.load(receipt_path)
    if (
        qualification.scope
        != required_live_provider_qualification_scope(lock.claim_scope)
        or qualification.canonical_sha256 != qualification_ref["canonical_sha256"]
    ):
        raise ValueError("runtime execution requires the Claim Scope's live provider qualification")
    harness = provider_harness(provider_authority)
    output_schema = (_object(provider_authority["output_schema"], "arm_environments.provider.output_schema")
                     if harness == "codex" else None)
    executable = Path(provider_config["executable"]).resolve(strict=True)
    if sha256(executable.read_bytes()).hexdigest() != provider_authority["executable_sha256"]:
        raise ValueError("runtime provider executable differs from the Campaign Lock")
    if metal:
        workload, workload_path, workload_contract = _load_workload_binding(root, lock)
        protocol = _object(lock.document["evaluation_protocol"], "evaluation_protocol")
        toolchain = MetalToolchainBuilder(workload=workload_contract, case_id=str(protocol["case_id"]),
            output_root=Path(toolchain_config["output_root"]),
            host=MetalArchiveHost.from_executor(executor), project_root=root)
    else:
        toolchain = (IsolatedTritonCompiler(**toolchain_config) if paired_triton else
                     NvccToolchainBuilder(nvcc=toolchain_config["nvcc"], cuobjdump=toolchain_config["cuobjdump"]))
    if paired_triton:
        toolchain.check_executor(executor, author_workspace=provider_config["workspace_root"])
    toolchain_authority = open_arm if metal else direct_arm
    if toolchain.canonical_sha256 != toolchain_authority["toolchain_sha256"] or (
        paired_triton and open_arm["toolchain_sha256"] != direct_arm["toolchain_sha256"]):
        raise ValueError("runtime toolchain differs from the Campaign Lock")
    command = broker_config["command"]
    execution = _object(lock.document["execution"], "campaign_lock.execution")
    broker_cwd = Path(broker_config["cwd"]).resolve(strict=True)
    broker_timeout = broker_config["timeout_seconds"]
    broker_user = broker_config["service_user"]
    broker_group = broker_config["service_group"]
    if (
        broker_execution_sha256(
            command,
            cwd=broker_cwd,
            project_root=root,
            timeout_seconds=broker_timeout,
            service_user=broker_user,
            service_group=broker_group,
        )
        != execution["broker_execution_sha256"]
    ):
        raise ValueError("runtime broker execution differs from the Campaign Lock")

    output_schema_path = (_raw_reference_path(root, output_schema, "arm_environments.provider.output_schema")
                          if output_schema is not None else None)
    _raw_reference_path(root, open_arm["scaffold"], "arm_environments.scaffold")
    if comparison == "direct_cuda":
        _raw_reference_path(
            root, direct_arm["launch_contract"], "arm_environments.direct_cuda.launch_contract"
        )
        _raw_reference_path(
            root,
            direct_arm["candidate_skeleton"],
            "arm_environments.direct_cuda.candidate_skeleton",
        )
    compiler_ref = _object(lock.document["compiler_revision"], "compiler_revision")
    compiler = Compiler.load(root, root / str(compiler_ref["path"]))
    compiler_gate = compiler.check_corpus()
    if (
        compiler.state != "released"
        or not compiler_gate.passed
        or compiler_gate.compiler_revision_sha256 != compiler_ref["canonical_sha256"]
    ):
        raise ValueError("runtime Compiler Revision differs from the Campaign Lock")
    if not metal:
        workload, workload_path, workload_contract = _load_workload_binding(root, lock)
        protocol = _object(lock.document["evaluation_protocol"], "evaluation_protocol")
    if (paired_triton or metal) and paired_protocol(protocol) is None:
        raise ValueError('live tensor Campaign requires explicit fixed-baseline paired policy')
    if metal:
        builder = toolchain
        direct_environment = None
    elif paired_triton:
        skeleton = _object(open_arm["schedule_skeleton"], "schedule_skeleton")
        baseline = bind_baseline(json.loads((root / str(skeleton["path"])).read_text()), workload_contract, str(protocol["case_id"]))
        lowering = compiler.lower(compiler.assess(baseline))
        builder = TritonToolchainBuilder(workload=workload_contract, case_id=str(protocol["case_id"]), isolated_compiler=toolchain)
        direct_environment = NativeTritonEnvironment(builder, toolchain_requirements=lowering.toolchain_requirements,
            authority_document=direct_arm, workload=workload_contract, case_id=str(protocol["case_id"]))
    else:
        builder = FlashTritonToolchainBuilder()
        direct_environment = DirectCudaEnvironment(toolchain,
            toolchain_requirements={"compiler": "nvcc", "target": "sm_100a"}, authority_document=direct_arm)
    environments = {
        "open_cake": OpenCakeEnvironment(
            compiler,
            builder,
            authority_document=open_arm,
            workload=workload_contract,
            case_id=str(protocol["case_id"]),
            executor=executor,
        ),
    }
    if comparison is not None:
        environments[comparison] = direct_environment
    fixed_baseline = None
    if paired_protocol(protocol) is not None:
        anchor_ref = provider_authority['qualification_anchor']
        _, anchor_path = qualification_path(root, anchor_ref['path'], 'provider qualification anchor')
        anchor = json.loads(anchor_path.read_bytes())
        if (sha256(json.dumps(anchor, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            != anchor_ref['canonical_sha256'] or anchor.get('qualification_receipt_sha256') != qualification.canonical_sha256):
            raise ValueError('runtime provider qualification anchor differs from Campaign Lock')
        fixed = execution['fixed_baseline']
        fixed_baseline = load_baseline_bundle(root, fixed['bundle_path'])
        if candidate_identity(fixed_baseline) != fixed['candidate']:
            raise ValueError('runtime fixed baseline differs from Campaign Lock')
        runtime_ref = execution['runtime_config']
        runtime_path = external_file(root, runtime_ref['path'], 'runtime configuration')
        if (Path(runtime_config_path).resolve(strict=True) != runtime_path
            or sha256(runtime_path.read_bytes()).hexdigest() != runtime_ref['sha256']):
            raise ValueError('runtime configuration differs from Campaign Lock')
    protocol_sha256 = sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    submitter = CommandBrokerSubmitter(
        command=command,
        workload_path=workload_path,
        workload_sha256=str(workload["canonical_sha256"]),
        protocol_sha256=protocol_sha256,
        cwd=broker_cwd,
        timeout_seconds=broker_timeout,
        executor=executor,
        service_user=broker_user,
        service_group=broker_group,
        evaluation_protocol=protocol,
        baseline=fixed_baseline,
        workload_loader=load_workload,
    )
    evaluator = BoundedBrokerEvaluator(protocol, submitter)

    workspace_root = Path(str(provider_config["workspace_root"])).absolute()
    workspace_root.mkdir(mode=0o750, parents=False, exist_ok=False)
    builders = {}
    task_packages = {}
    for run_id in lock.run_order:
        workspace = workspace_root / run_id
        workspace.mkdir(mode=0o750)
        package = TaskLab(root).task_package(lock, run_id)
        materialize_task_package(workspace, package)
        task_packages[run_id] = package
        common_provider = dict(executable=executable, provider_revision=qualification.provider_revision,
            model=str(provider_authority["model"]), reasoning_effort=str(provider_authority["reasoning_effort"]),
            workspace=workspace, removed_environment=tuple(provider_authority["removed_environment"]))
        if harness == "claude-code":
            builders[run_id] = ClaudeInvocationBuilder(**common_provider)
        else:
            builders[run_id] = CodexInvocationBuilder(**common_provider,
                code_mode_host=_object(provider_authority["code_mode_host"], "provider.code_mode_host"),
                service_tier=str(provider_authority["service_tier"]), output_schema=output_schema_path,
                disabled_features=tuple(provider_authority["disabled_features"]),
                event_contract=str(provider_authority.get("event_contract", "closed_file_change_v1")),
                submission_contract=CANDIDATE_SET_ENVELOPE_V1,
                cwd_policy=str(provider_authority["cwd_policy"]),
                reference_visibility=str(provider_authority["reference_visibility"]))
    provider_type = ClaudeRunProvider if harness == "claude-code" else CodexRunProvider
    adapter_type = ClaudeProviderAdapter if harness == "claude-code" else CodexProviderAdapter
    provider = provider_type(qualification=qualification, builders=builders,
                             task_packages=task_packages, adapter=adapter_type())
    return TaskLab(root).execute(
        lock,
        evidence_root,
        provider=provider,
        environments=environments,
        evaluator=evaluator,
    )


def execute_portfolio_from_config(
    project_root: str | Path,
    lock: CampaignLock,
    runtime_config_path: str | Path,
    evidence_root: str | Path,
) -> CampaignRef:
    """Compile, persistently load and evaluate the frozen Portfolio on one B200."""

    if lock.study_kind != "portfolio":
        raise ValueError("portfolio composition requires a Portfolio Campaign Lock")
    root = Path(project_root).resolve(strict=True)
    _, cupti_helper = _admit_executor(root, lock)
    config = _object(
        json.loads(Path(runtime_config_path).read_text(encoding="utf-8")),
        "runtime_config",
    )
    if set(config) != {"schema_version", "portfolio"} or config.get("schema_version") != 1:
        raise ValueError("portfolio runtime configuration fields differ")
    portfolio_config = _object(config["portfolio"], "runtime_config.portfolio")
    if set(portfolio_config) != {"device"}:
        raise ValueError("portfolio runtime configuration section differs")
    torch = __import__("torch")
    compiler_ref = _object(lock.document["compiler_revision"], "compiler_revision")
    compiler = Compiler.load(root, root / str(compiler_ref["path"]))
    resolved = _object(lock.document["resolved_inputs"], "resolved_inputs")
    seed_ref = _object(resolved["kernel_seed"], "resolved_inputs.kernel_seed")
    seed = KernelSeed.load(root, root / str(seed_ref["path"]))
    workload_ref = _object(lock.document["workload"], "workload")
    workload = load_workload(root / str(workload_ref["path"]))
    case_roles = _object(resolved["case_roles"], "resolved_inputs.case_roles")
    case_ids = cast(list[str], case_roles["anchor"]) + cast(
        list[str], case_roles["held_out"]
    )
    assay = _LivePortfolioAssay(
        compiler=compiler,
        seed=seed,
        workload=workload,
        case_ids=case_ids,
        evaluation_protocol=_object(
            lock.document["evaluation_protocol"], "evaluation_protocol"
        ),
        synchronize=torch.cuda.synchronize,
        device=str(portfolio_config["device"]),
        torch_module=torch,
        cupti_helper=cupti_helper,
    )
    return TaskLab(root).execute_portfolio(
        lock,
        evidence_root,
        assay=assay,
    )
