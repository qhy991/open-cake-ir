"""Closed CLI composition for a live matched-search Campaign."""

from __future__ import annotations

import json
import shlex
import shutil
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Mapping, cast

from open_cake_ir.compiler import Compiler
from open_cake_ir.evaluation import (
    CudaLaunchManifest,
    CuptiPortfolioAssay,
    LoadedCudaCandidate,
    PortfolioArtifact,
    PortfolioEvaluationReceipt,
    StrictCuptiBenchmark,
    WorkloadContract,
    observe_exclusive_b200,
)

from .core import CampaignLock, CampaignRef, Lab
from .environments import (
    BuildRequest,
    DirectCudaEnvironment,
    NvccToolchainBuilder,
    OpenCakeEnvironment,
    TritonToolchainBuilder,
)
from .executor import ExecutorRevision
from .faults import RunProtocolFault
from .portfolio import KernelSeed, lower_specialists
from .providers import (
    CodexInvocationBuilder,
    CodexProviderAdapter,
    CodexRunProvider,
    ProviderQualificationReceipt,
)
from .runtime import BoundedBrokerEvaluator, CommandBrokerSubmitter


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
    reference = _object(execution["executor_revision"], "execution.executor_revision")
    revision = ExecutorRevision.load(root, root / str(reference["path"]))
    if (
        revision.canonical_sha256 != reference["canonical_sha256"]
        or revision.executor_id != reference["executor_id"]
    ):
        raise ValueError("live Executor Revision differs")
    return revision, revision.admit_host()


def broker_execution_sha256(
    command: tuple[str, ...],
    *,
    cwd: Path,
    project_root: Path,
    timeout_seconds: int,
    service_user: str,
    service_group: str,
) -> str:
    if not command:
        raise ValueError("external command is empty")
    if (
        cwd.resolve(strict=True) != project_root
        or timeout_seconds <= 0
        or not service_user
        or not service_group
    ):
        raise ValueError("broker cwd policy or timeout differs")
    executable = shutil.which(command[0])
    if executable is None:
        raise ValueError(f"external command {command[0]!r} is unavailable")
    files: dict[str, str] = {
        str(Path(executable).resolve(strict=True)): sha256(
            Path(executable).resolve(strict=True).read_bytes()
        ).hexdigest()
    }
    for value in command[1:]:
        path = Path(value)
        if path.is_absolute() and path.is_file() and not path.is_symlink():
            files[str(path.resolve(strict=True))] = sha256(path.read_bytes()).hexdigest()
    return sha256(
        json.dumps(
            {
                "argv": list(command),
                "files": files,
                "cwd_policy": "project_root",
                "timeout_seconds": timeout_seconds,
                "service_user": service_user,
                "service_group": service_group,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _write_reference(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
    path.chmod(0o444)


def _materialize_run_references(
    root: Path,
    references: Path,
    lock: CampaignLock,
    arm: Mapping[str, object],
) -> None:
    references.mkdir(mode=0o755)
    workload = _object(lock.document["workload"], "campaign_lock.workload")
    compiler = _object(lock.document["compiler_revision"], "campaign_lock.compiler")
    scaffold = _object(arm["scaffold"], "arm.scaffold")
    revision_document = _object(
        json.loads((root / str(compiler["path"])).read_text(encoding="utf-8")),
        "compiler_revision",
    )
    targets = _object(revision_document["target_definitions"], "target_definitions")
    target = _object(targets["sm_100a"], "target_definitions.sm_100a")
    resolved = _object(lock.document["resolved_inputs"], "resolved_inputs")
    run_authority = {
        "schema_version": 1,
        "study": lock.document["study"],
        "workload": lock.document["workload"],
        "authoring_environment": arm,
        "budget": resolved["budget"],
        "run_protocol": resolved["run_protocol"],
        "evaluation_protocol": lock.document["evaluation_protocol"],
        "execution": lock.document["execution"],
    }
    _write_reference(
        references / "run-authority.json",
        json.dumps(run_authority, sort_keys=True, separators=(",", ":")).encode(),
    )
    _write_reference(
        references / "workload.json",
        (root / str(workload["path"])).read_bytes(),
    )
    _write_reference(
        references / "target.json",
        (root / str(target["path"])).read_bytes(),
    )
    _write_reference(
        references / "scaffold.md",
        (root / str(scaffold["path"])).read_bytes(),
    )
    if arm.get("environment_kind") == "open_cake":
        skeleton_ref = _object(arm["schedule_skeleton"], "arm.schedule_skeleton")
        skeleton = cast(
            dict[str, object],
            json.loads((root / str(skeleton_ref["path"])).read_text(encoding="utf-8")),
        )
        case_id = str(_object(lock.document["evaluation_protocol"], "protocol")["case_id"])
        workload_contract = WorkloadContract.load(root / str(workload["path"]))
        shape = _object(workload_contract.case(case_id)["shape"], "workload.case.shape")
        buffers = {
            str(item["name"]): item
            for item in cast(list[dict[str, object]], skeleton["buffers"])
        }
        buffers["tokens"]["shape"] = [shape["B"], shape["N"], shape["D"]]
        buffers["centroids"]["shape"] = [shape["B"], shape["K"], shape["D"]]
        buffers["centroid_sq"]["shape"] = [shape["B"], shape["K"]]
        buffers["assignments"]["shape"] = [shape["B"], shape["N"]]
        skeleton["schedule_id"] = "open-cake-ir-matched-authoring-skeleton-v1"
        cast(dict[str, object], skeleton["metadata"])[
            "workload_contract_sha256"
        ] = workload_contract.canonical_sha256
        _write_reference(
            references / "schedule.schema.json",
            (root / "compiler/schedule.schema.json").read_bytes(),
        )
        _write_reference(
            references / "schedule-authoring.md",
            (root / "compiler/AUTHORING_CONTRACT.md").read_bytes(),
        )
        _write_reference(
            references / "schedule-skeleton.json",
            json.dumps(skeleton, sort_keys=True, separators=(",", ":")).encode(),
        )
    elif arm.get("environment_kind") == "direct_cuda":
        launch_contract = _object(arm["launch_contract"], "arm.launch_contract")
        candidate_skeleton = _object(
            arm["candidate_skeleton"], "arm.candidate_skeleton"
        )
        _write_reference(
            references / "cuda-launch-abi.json",
            (root / str(launch_contract["path"])).read_bytes(),
        )
        _write_reference(
            references / "candidate-skeleton.cu",
            (root / str(candidate_skeleton["path"])).read_bytes(),
        )
    else:
        raise ValueError("Authoring Environment kind differs during reference materialization")
    references.chmod(0o555)


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
            toolchain = TritonToolchainBuilder()
            candidates = {}
            for item in lowered:
                request = BuildRequest(
                    candidate_sha256=item.lowering.schedule_sha256,
                    source=item.lowering.source.encode(),
                    source_role="lowered_source",
                    source_sha256=item.lowering.source_sha256,
                    target=item.lowering.target,
                    entry_point=item.lowering.entry_point,
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
    config = _object(
        json.loads(Path(runtime_config_path).read_text(encoding="utf-8")),
        "runtime_config",
    )
    if set(config) != {"schema_version", "provider", "toolchain", "broker"} or config.get(
        "schema_version"
    ) != 1:
        raise ValueError("runtime configuration fields differ")
    provider_config = _object(config["provider"], "runtime_config.provider")
    toolchain_config = _object(config["toolchain"], "runtime_config.toolchain")
    broker_config = _object(config["broker"], "runtime_config.broker")
    if set(provider_config) != {"executable", "workspace_root"} or set(
        toolchain_config
    ) != {"nvcc", "cuobjdump"} or set(broker_config) != {
        "command",
        "cwd",
        "timeout_seconds",
        "service_user",
        "service_group",
    }:
        raise ValueError("runtime configuration section fields differ")

    resolved = _object(lock.document["resolved_inputs"], "campaign_lock.resolved_inputs")
    arms = _object(resolved["arm_environments"], "arm_environments")
    open_arm = _object(arms["open_cake"], "arm_environments.open_cake")
    direct_arm = _object(arms["direct_cuda"], "arm_environments.direct_cuda")
    provider_authority = _object(open_arm["provider"], "arm_environments.provider")
    qualification_ref = _object(
        provider_authority["qualification"], "arm_environments.provider.qualification"
    )
    qualification = ProviderQualificationReceipt.load(root / str(qualification_ref["path"]))
    if (
        qualification.scope != "live_two_turn_current_provider"
        or qualification.canonical_sha256 != qualification_ref["canonical_sha256"]
    ):
        raise ValueError("runtime execution requires a current live provider qualification")
    output_schema = _object(
        provider_authority["output_schema"], "arm_environments.provider.output_schema"
    )
    executable = Path(str(provider_config["executable"])).resolve(strict=True)
    if sha256(executable.read_bytes()).hexdigest() != provider_authority["executable_sha256"]:
        raise ValueError("runtime provider executable differs from the Campaign Lock")
    nvcc_builder = NvccToolchainBuilder(
        nvcc=str(toolchain_config["nvcc"]),
        cuobjdump=str(toolchain_config["cuobjdump"]),
    )
    if nvcc_builder.canonical_sha256 != direct_arm["toolchain_sha256"]:
        raise ValueError("runtime CUDA toolchain differs from the Campaign Lock")
    command_value = broker_config["command"]
    command = (
        tuple(str(value) for value in command_value)
        if isinstance(command_value, list)
        else tuple(shlex.split(str(command_value)))
    )
    execution = _object(lock.document["execution"], "campaign_lock.execution")
    broker_cwd = Path(str(broker_config["cwd"])).resolve(strict=True)
    broker_timeout = int(broker_config["timeout_seconds"])
    broker_user = str(broker_config["service_user"])
    broker_group = str(broker_config["service_group"])
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

    output_schema_path = _raw_reference_path(
        root, output_schema, "arm_environments.provider.output_schema"
    )
    _raw_reference_path(root, open_arm["scaffold"], "arm_environments.scaffold")
    _raw_reference_path(
        root, direct_arm["launch_contract"], "arm_environments.direct_cuda.launch_contract"
    )
    _raw_reference_path(
        root,
        direct_arm["candidate_skeleton"],
        "arm_environments.direct_cuda.candidate_skeleton",
    )
    prompt_templates = {
        arm: _raw_reference_path(
            root,
            document["prompt_template"],
            f"arm_environments.{arm}.prompt_template",
        )
        for arm, document in (("open_cake", open_arm), ("direct_cuda", direct_arm))
    }
    compiler_ref = _object(lock.document["compiler_revision"], "compiler_revision")
    compiler = Compiler.load(root, root / str(compiler_ref["path"]))
    compiler_gate = compiler.check_corpus()
    if (
        compiler.state != "released"
        or not compiler_gate.passed
        or compiler_gate.compiler_revision_sha256 != compiler_ref["canonical_sha256"]
    ):
        raise ValueError("runtime Compiler Revision differs from the Campaign Lock")
    protocol = _object(lock.document["evaluation_protocol"], "evaluation_protocol")
    workload = _object(lock.document["workload"], "workload")
    workload_contract = WorkloadContract.load(root / str(workload["path"]))
    if workload_contract.canonical_sha256 != workload["canonical_sha256"]:
        raise ValueError("runtime Workload differs from the Campaign Lock")
    environments = {
        "open_cake": OpenCakeEnvironment(
            compiler,
            TritonToolchainBuilder(),
            authority_document=open_arm,
            workload=workload_contract,
            case_id=str(protocol["case_id"]),
        ),
        "direct_cuda": DirectCudaEnvironment(
            nvcc_builder,
            toolchain_requirements={"compiler": "nvcc", "target": "sm_100a"},
            authority_document=direct_arm,
        ),
    }
    protocol_sha256 = sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    submitter = CommandBrokerSubmitter(
        command=command,
        workload_path=root / str(workload["path"]),
        workload_sha256=str(workload["canonical_sha256"]),
        protocol_sha256=protocol_sha256,
        cwd=broker_cwd,
        timeout_seconds=broker_timeout,
        executor=executor,
        service_user=broker_user,
        service_group=broker_group,
    )
    evaluator = BoundedBrokerEvaluator(protocol, submitter)

    workspace_root = Path(str(provider_config["workspace_root"])).absolute()
    workspace_root.mkdir(mode=0o750, parents=False, exist_ok=False)
    references_root = workspace_root / "_references"
    references_root.mkdir(mode=0o755)
    builders = {}
    reference_roots = {}
    for run_id in lock.run_order:
        workspace = workspace_root / run_id
        workspace.mkdir(mode=0o750)
        arm_name = run_id.rsplit("-", 1)[0]
        reference_root = references_root / run_id
        _materialize_run_references(
            root,
            reference_root,
            lock,
            _object(arms[arm_name], f"arm_environments.{arm_name}"),
        )
        reference_roots[run_id] = reference_root
        builders[run_id] = CodexInvocationBuilder(
            executable=executable,
            provider_revision=qualification.provider_revision,
            model=str(provider_authority["model"]),
            reasoning_effort=str(provider_authority["reasoning_effort"]),
            service_tier=str(provider_authority["service_tier"]),
            workspace=workspace,
            output_schema=output_schema_path,
            removed_environment=tuple(
                str(value) for value in cast(list[object], provider_authority["removed_environment"])
            ),
        )
    references_root.chmod(0o555)
    provider = CodexRunProvider(
        qualification=qualification,
        builders=builders,
        reference_roots=reference_roots,
        prompt_templates=prompt_templates,
        adapter=CodexProviderAdapter(),
    )
    return Lab(root).execute(
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
    workload = WorkloadContract.load(root / str(workload_ref["path"]))
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
    return Lab(root).execute_portfolio(
        lock,
        evidence_root,
        assay=assay,
    )
