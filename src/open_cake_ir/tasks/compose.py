"""Closed CLI composition for a live matched-search Campaign."""

from __future__ import annotations

from open_cake_ir.serialization import canonical_json_bytes
from open_cake_ir.tasks.flash_kmeans.environment import FlashTritonToolchainBuilder
from open_cake_ir.tasks.workloads import load_workload

import json
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Mapping, cast

from open_cake_ir.compiler import Compiler

from open_cake_ir.lab.contracts import CampaignLock, CampaignRef
from open_cake_ir.tasks.runtime import TaskLab
from open_cake_ir.tasks.flash_kmeans.environment import DirectCudaEnvironment, NvccToolchainBuilder
from open_cake_ir.tasks.environments import TaskOpenCakeEnvironment as OpenCakeEnvironment
from open_cake_ir.lab.executor import ExecutorRevision
from open_cake_ir.lab.providers import CANDIDATE_SET_ENVELOPE_V1, CodexInvocationBuilder, CodexProviderAdapter, CodexRunProvider, ProviderQualificationReceipt, required_live_provider_qualification_scope
from open_cake_ir.lab.pairing import comparison_arm, bind_baseline, native_backend
from open_cake_ir.lab.toolchains import single_environment_toolchain, toolchain_for_arm
from open_cake_ir.compiler.ir.vocabulary import LoweringBackend
from open_cake_ir.lab.claude import ClaudeInvocationBuilder, advertised_options, ClaudeProviderAdapter, ClaudeRunProvider
from open_cake_ir.lab.provider_policy import provider_harness
# MetalArchiveHost is bound through the Lab toolchain table; it stays named here because
# the composition tests patch `compose.MetalArchiveHost.from_executor`.
from open_cake_ir.lab.metal_build import MetalArchiveHost, MetalToolchainBuilder  # noqa: F401
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
    """Admit the Executor's host through the admission that host's kind declares.

    `admit_host` covers CUDA and Metal and refuses a HIP capture by name, pointing at
    `admit_hip_host` -- which carries the executor id into the admission and returns the
    ROCm facts the build jail needs. This dispatched to neither: it called `admit_host`
    for every capture, so a HIP Campaign died at composition on a refusal that was telling
    it which door to use. The kind is read from the capture that declares it rather than
    inferred from anything else.
    """

    execution = _object(lock.document["execution"], "campaign_lock.execution")
    revision = ExecutorRevision.load_reference(
        root, execution["executor_revision"], "execution.executor_revision"
    )
    host = _object(revision.document["host_environment"], "executor.host_environment")
    if host.get("kind") == "hip":
        return revision, revision.admit_hip_host()
    return revision, revision.admit_host()




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
    policy = native_backend(comparison)
    # One toolchain row: the comparison arm's, or the single arm's own route. The row
    # says which runtime fields to parse, which native policy applies and how the
    # identity-bearing toolchain is bound; a route with no row is refused by name.
    row = (single_environment_toolchain(arms["open_cake"]["lowering_route"]["backend"])
           if comparison is None else toolchain_for_arm(comparison))
    metal = row.backend is LoweringBackend.METAL
    tensor_policy = row.native
    config = load_runtime_config(runtime_config_path, toolchain_kind=row.runtime_kind)
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
            host=row.bind(toolchain_config, executor, author_workspace=provider_config["workspace_root"]),
            project_root=root, compiler_reference=lock.document["compiler_revision"])
    elif row.bind_toolchain is not None:
        toolchain = row.bind(toolchain_config, executor, author_workspace=provider_config["workspace_root"])
    else:
        # D12: the direct_cuda arm binds its own nvcc builder under tasks/flash_kmeans;
        # the Lab table names the row and leaves this binding to the layer that owns it.
        toolchain = NvccToolchainBuilder(nvcc=toolchain_config["nvcc"], cuobjdump=toolchain_config["cuobjdump"])
    toolchain_authority = open_arm if comparison is None else direct_arm
    if toolchain.canonical_sha256 != toolchain_authority["toolchain_sha256"] or (
        policy is not None and open_arm["toolchain_sha256"] != direct_arm["toolchain_sha256"]):
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
        compiler.commit is None
        or not compiler_gate.passed
        or compiler_gate.compiler_revision_id != compiler_ref["revision_id"]
    ):
        raise ValueError(
            "runtime Compiler Revision differs from the Campaign Lock: the lock pins "
            f"{compiler_ref.get('revision_id')!r}, this checkout provides "
            f"{compiler_gate.compiler_revision_id!r} (commit {compiler.commit!r}, "
            f"Corpus Gate passed={compiler_gate.passed})"
        )
    if not metal:
        workload, workload_path, workload_contract = _load_workload_binding(root, lock)
        protocol = _object(lock.document["evaluation_protocol"], "evaluation_protocol")
    if (tensor_policy is not None or metal) and paired_protocol(protocol) is None:
        raise ValueError('live tensor Campaign requires explicit fixed-baseline paired policy')
    if metal:
        builder = toolchain
        direct_environment = None
    elif tensor_policy is not None:
        builder = tensor_policy.builder(workload=workload_contract, case_id=str(protocol["case_id"]), isolated_compiler=toolchain)
        direct_environment = None
        if comparison is not None:
            skeleton = _object(open_arm["schedule_skeleton"], "schedule_skeleton")
            baseline = bind_baseline(json.loads((root / str(skeleton["path"])).read_text()), workload_contract, str(protocol["case_id"]))
            lowering = compiler.lower(compiler.assess(baseline))
            direct_environment = policy.environment(builder, toolchain_requirements=lowering.toolchain_requirements,
                authority_document=direct_arm, workload=workload_contract, case_id=str(protocol["case_id"]))
    else:
        builder = FlashTritonToolchainBuilder()
        # The historical flash_kmeans builder refuses any target but the one it was
        # written for; the contract still names the Campaign's own Workload target
        # rather than restating that builder's literal here.
        direct_environment = DirectCudaEnvironment(toolchain,
            toolchain_requirements={"compiler": "nvcc", "target": workload_contract.target}, authority_document=direct_arm)
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
        if (sha256(canonical_json_bytes(anchor)).hexdigest()
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
        canonical_json_bytes(protocol)
    ).hexdigest()
    submitter = CommandBrokerSubmitter(
        command=command,
        workload_path=workload_path,
        workload_sha256=str(workload["canonical_sha256"]),
        protocol_sha256=protocol_sha256,
        cwd=broker_cwd,
        timeout_seconds=broker_timeout,
        executor=executor,
        compiler_reference=compiler_ref,
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
    _claude_options: list = []

    def claude_cli_options():
        """Ask this one executable once; every Run in this Campaign uses the same binary."""
        if not _claude_options:
            _claude_options.append(advertised_options(executable))
        return _claude_options[0]

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
            builders[run_id] = ClaudeInvocationBuilder(
                **common_provider, cli_options=claude_cli_options(),
                event_contract=provider_authority["event_contract"])
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


