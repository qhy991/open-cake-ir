"""Production runtime assembly from one frozen independent Run."""

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
from open_cake_ir.lab.providers import CANDIDATE_SET_ENVELOPE_V1, CodexInvocationBuilder, CodexProviderAdapter, CodexRunProvider, ProviderQualificationReceipt
from open_cake_ir.lab.pairing import bind_baseline, native_backend
from open_cake_ir.lab.toolchains import toolchain_for, toolchain_for_arm
from open_cake_ir.compiler.ir.vocabulary import LoweringBackend
from open_cake_ir.lab.claude import ClaudeInvocationBuilder, advertised_options, ClaudeProviderAdapter, ClaudeRunProvider
from open_cake_ir.lab.provider_policy import provider_harness
from open_cake_ir.lab.message_provider import MessageQualification, ResponsesRunProvider
from open_cake_ir.lab.python_reference import read_skeleton_reference
from open_cake_ir.lab.reference_access import require_qualified_clean_start_execution
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
    root: Path, value: object, context: str, *, allow_external: bool = False
) -> Path:
    reference = _object(value, context)
    if set(reference) != {"path", "sha256"}:
        raise ValueError(f"{context} fields differ")
    raw_path = reference["path"]
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{context}.path differs")
    if allow_external and Path(raw_path).is_absolute():
        # Scaffolds already admit canonical external publications at Study admission
        # and TaskPackage rendering. Retain that same custody rule at execution.
        path = external_file(root, raw_path, context)
        if sha256(path.read_bytes()).hexdigest() != reference["sha256"]:
            raise ValueError(f"{context} bytes differ")
        return path
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


def _admit_executor(root: Path, lock) -> tuple[ExecutorRevision, object]:
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




def _load_workload_binding(root: Path, lock):
    reference = _object(lock.document["workload"], "workload")
    path = (external_file(root, reference["path"], "Workload")
            if Path(str(reference["path"])).is_absolute() else root / str(reference["path"]))
    workload = load_workload(path)
    if workload.canonical_sha256 != reference["canonical_sha256"]:
        raise ValueError("runtime Workload differs from the Run")
    return reference, path, workload


def run_runtime_factory(project_root, runtime_config_path):
    """Bind each Run to fresh adapters; no Study policy or shared author history.

    Runtime paths are execution bindings, never author inputs. A messages-only
    provider has no executable or workspace configuration and gets only TaskPackage
    messages. CLI providers retain one create-only workspace per Run.
    """
    root = Path(project_root).resolve(strict=True)
    runtime_path = Path(runtime_config_path).resolve(strict=True)
    lab = TaskLab(root)

    def build(specification, directory):
        specification = lab.preflight_run(specification)
        document = specification.document
        require_qualified_clean_start_execution((document['authoring'],))
        authoring, execution, protocol = (document[name] for name in ('authoring','execution','evaluation_protocol'))
        kind = specification.environment_kind
        declared_provider = authoring['provider']
        harness = provider_harness(declared_provider)
        reference, workload_path, workload = _load_workload_binding(root,specification)
        explicit = isinstance(workload.document['semantics'].get('candidate_abi'),Mapping)
        flash_cake = kind=='open_cake' and not explicit
        # The external Flash runtime file contains the nvcc comparator section.
        # The Cake Run itself uses its Compiler/Executor-bound Triton builder.
        row = (toolchain_for(authoring['lowering_route']['backend'] if explicit else 'native_cuda')
               if kind=='open_cake' else toolchain_for_arm(kind))
        config = load_runtime_config(runtime_path,toolchain_kind=row.runtime_kind,provider_kind=harness)
        provider_config, toolchain_config, broker_config = (config[name] for name in ('provider','toolchain','broker'))
        runtime_ref = execution.get('runtime_config')
        if runtime_ref is not None:
            retained = external_file(root,runtime_ref['path'],'runtime configuration')
            if runtime_path != retained or sha256(retained.read_bytes()).hexdigest() != runtime_ref['sha256']:
                raise ValueError('runtime configuration differs from the Run')
        elif paired_protocol(protocol) is not None:
            raise ValueError('paired Run requires its frozen runtime configuration')

        _, receipt_path = qualification_path(root,declared_provider['qualification']['path'],'provider qualification')
        if harness=='responses':
            qualification = MessageQualification.load(receipt_path)
            expected_scope = 'live_two_turn_message_provider'
        else:
            qualification = ProviderQualificationReceipt.load(receipt_path)
            expected_scope = ('live_two_turn_current_provider'
                if declared_provider.get('event_contract','closed_file_change_v1')=='closed_file_change_v1'
                else 'live_two_turn_tool_rich_provider')
        if qualification.scope != expected_scope or qualification.canonical_sha256 != declared_provider['qualification']['canonical_sha256']:
            raise ValueError('production Run requires its exact live provider qualification')

        executor,_ = _admit_executor(root,specification)
        compiler_ref = document['compiler_revision']
        compiler = Compiler.load(root,root/compiler_ref['path'])
        gate = compiler.check_corpus()
        if compiler.commit is None or not gate.passed or gate.compiler_revision_id != compiler_ref['revision_id']:
            raise ValueError('runtime Compiler differs from the Run or its Corpus Gate did not pass')
        author_workspace = (Path(directory).absolute() if harness=='responses'
                            else Path(provider_config['workspace_root']).absolute()/specification.run_id)
        if harness != 'responses':
            from open_cake_ir.lab.custody import admit_new_campaign_path
            if author_workspace.parent.exists():
                author_workspace = admit_new_campaign_path(root,author_workspace,role='Run author workspace')
            else:
                parent = admit_new_campaign_path(root,author_workspace.parent,role='Run author workspace root')
                author_workspace = parent/specification.run_id
        if flash_cake:
            toolchain = None
        elif row.backend is LoweringBackend.METAL:
            toolchain = MetalToolchainBuilder(workload=workload,case_id=protocol['case_id'],
                output_root=Path(toolchain_config['output_root']),
                host=row.bind(toolchain_config,executor,author_workspace=author_workspace),
                project_root=root,compiler_reference=compiler_ref)
        elif row.bind_toolchain is not None:
            toolchain = row.bind(toolchain_config,executor,author_workspace=author_workspace)
        else:
            toolchain = NvccToolchainBuilder(nvcc=toolchain_config['nvcc'],cuobjdump=toolchain_config['cuobjdump'])
        if not flash_cake and toolchain.canonical_sha256 != authoring['toolchain_sha256']:
            raise ValueError('runtime toolchain differs from the Run')

        if flash_cake:
            builder = FlashTritonToolchainBuilder()
        elif row.backend is LoweringBackend.METAL:
            builder = toolchain
        elif row.native is not None:
            builder = row.native.builder(workload=workload,case_id=protocol['case_id'],isolated_compiler=toolchain)
        else:
            builder = None  # Direct CUDA consumes its own nvcc toolchain below.
        if kind=='open_cake':
            environment = OpenCakeEnvironment(compiler,builder,authority_document=authoring,
                workload=workload,case_id=protocol['case_id'],executor=executor)
        elif native_backend(kind) is not None:
            _, baseline = read_skeleton_reference(root,document['reference_inputs']['baseline_schedule'],'Run baseline Schedule')
            lowering = compiler.lower(compiler.assess(bind_baseline(baseline,workload,protocol['case_id'])))
            environment = native_backend(kind).environment(builder,toolchain_requirements=lowering.toolchain_requirements,
                authority_document=authoring,workload=workload,case_id=protocol['case_id'])
        else:
            environment = DirectCudaEnvironment(toolchain,toolchain_requirements={'compiler':'nvcc','target':workload.target},
                                                authority_document=authoring)
        if explicit and paired_protocol(protocol) is None:
            raise ValueError('production tensor optimization requires an explicit fixed-baseline paired policy')
        baseline = None
        if paired_protocol(protocol) is not None:
            fixed = execution['fixed_baseline']
            baseline = load_baseline_bundle(root,fixed['bundle_path'])
            if candidate_identity(baseline) != fixed['candidate']:
                raise ValueError('runtime fixed baseline differs from the Run')
        broker_cwd = Path(broker_config['cwd']).resolve(strict=True)
        command,timeout,user,group = (broker_config[key] for key in ('command','timeout_seconds','service_user','service_group'))
        if broker_execution_sha256(command,cwd=broker_cwd,project_root=root,timeout_seconds=timeout,
                service_user=user,service_group=group) != execution['broker_execution_sha256']:
            raise ValueError('runtime broker execution differs from the Run')
        submitter = CommandBrokerSubmitter(command=command,workload_path=workload_path,
            workload_sha256=reference['canonical_sha256'],protocol_sha256=sha256(canonical_json_bytes(protocol)).hexdigest(),
            cwd=broker_cwd,timeout_seconds=timeout,executor=executor,compiler_reference=compiler_ref,
            service_user=user,service_group=group,evaluation_protocol=protocol,baseline=baseline,workload_loader=load_workload)
        evaluator = BoundedBrokerEvaluator(protocol,submitter)
        package = lab.task_package(specification,specification.run_id)
        packages = {specification.run_id:package}
        if harness=='responses':
            provider = ResponsesRunProvider(qualification=qualification,task_packages=packages)
        else:
            executable = Path(provider_config['executable']).resolve(strict=True)
            if sha256(executable.read_bytes()).hexdigest() != declared_provider['executable_sha256']:
                raise ValueError('runtime provider executable differs from the Run')
            # Do not recreate or reuse an actor directory from an attempted Run.
            author_workspace.parent.mkdir(mode=0o750,parents=False,exist_ok=True)
            author_workspace.mkdir(mode=0o750,exist_ok=False)
            materialize_task_package(author_workspace,package)
            common = dict(executable=executable,provider_revision=qualification.provider_revision,
                model=declared_provider['model'],reasoning_effort=declared_provider['reasoning_effort'],
                workspace=author_workspace,removed_environment=tuple(declared_provider['removed_environment']))
            if harness=='claude-code':
                invocation = ClaudeInvocationBuilder(**common,cli_options=advertised_options(executable),
                    event_contract=declared_provider['event_contract'],
                    submission_contract=declared_provider.get('submission_contract', 'candidate_set_envelope_v1'),
                    response_aliases=declared_provider.get('response_model_aliases', ()))
                provider = ClaudeRunProvider(qualification=qualification,builders={specification.run_id:invocation},
                    task_packages=packages,adapter=ClaudeProviderAdapter(response_aliases=invocation.response_aliases))
            else:
                schema = _raw_reference_path(root,declared_provider['output_schema'],'provider.output_schema')
                author_home_policy = declared_provider.get('author_home_policy')
                codex_home = None
                if author_home_policy is not None:
                    from open_cake_ir.lab.author_home import (
                        ISOLATED_AUTH_ONLY_V1, provision_codex_home,
                    )
                    if (author_home_policy != ISOLATED_AUTH_ONLY_V1
                        or 'auth_source' not in provider_config):
                        raise ValueError('Run isolated Codex home policy or credential source differs')
                    home_root = author_workspace.parent/'.codex-homes'
                    home_root.mkdir(mode=0o700, exist_ok=True)
                    auth_source = external_file(root, provider_config['auth_source'],
                                                'Run Codex credential source')
                    codex_home = provision_codex_home(auth_source,
                                                       home_root/specification.run_id)
                invocation = CodexInvocationBuilder(**common,code_mode_host=declared_provider['code_mode_host'],
                    service_tier=declared_provider['service_tier'],output_schema=schema,
                    disabled_features=tuple(declared_provider['disabled_features']),
                    event_contract=declared_provider.get('event_contract','closed_file_change_v1'),
                    submission_contract=declared_provider.get('submission_contract', CANDIDATE_SET_ENVELOPE_V1),cwd_policy=declared_provider['cwd_policy'],
                    reference_visibility=declared_provider['reference_visibility'],
                    author_home_policy=author_home_policy, codex_home=codex_home,
                    qualified_system_skills_sha256=qualification.system_skills_sha256)
                provider = CodexRunProvider(qualification=qualification,builders={specification.run_id:invocation},
                    task_packages=packages,adapter=CodexProviderAdapter())
        return {'provider':provider,'environment':environment,'evaluator':evaluator}
    return build


def execute_run_from_config(project_root,specification,runtime_config_path,evidence_root):
    from open_cake_ir.lab.custody import admit_new_campaign_path
    root = Path(project_root).resolve(strict=True)
    output = admit_new_campaign_path(root,evidence_root,role='Run Evidence root')
    require_qualified_clean_start_execution((specification.document['authoring'],))
    components = run_runtime_factory(root,runtime_config_path)(specification,output.parent/(output.name+'-runtime'))
    return TaskLab(root).execute_run(specification,output,**components)


def execute_matched_from_config(project_root,lock: CampaignLock,runtime_config_path,evidence_root) -> CampaignRef:
    """External Campaign input/report adapter over the same Run runtime and engine."""
    return TaskLab(project_root).execute_campaign_with_factory(lock,evidence_root,
        runtime_factory=run_runtime_factory(project_root,runtime_config_path))
