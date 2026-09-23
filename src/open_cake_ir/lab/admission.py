"""Study admission checks grouped by their external authority boundaries."""

from __future__ import annotations

import ast
import json
from hashlib import sha256
from typing import Mapping, cast

from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.toolchain import triton_route
from open_cake_ir.evaluation.paired import (
    METAL_KINDS, candidate_identity, paired_protocol, validate_pair_candidates, validation_case_ids,
)

from ._documents import _canonical_json_bytes, _digest, _name, _object, differs
from ._policies import _ATTRIBUTION_EVALUATION
from .bindings import load_baseline_bundle, qualification_path as _qualification_path, source_reference_path
from .build import _hidden_pointers
from .incumbents import admit_baseline_selection
from .providers import (
    ProviderQualificationReceipt,
    required_live_provider_qualification_scope,
)
from .pairing import native_source, native_block, backend_policy

from .provider_policy import provider_configuration, provider_harness


def validate_provider(*, open_cake, policy, project_root, study):
    """Matched-Study input policy; runtime qualification has an independent owner."""
    claim_scope = str(study.document['claim_scope'])
    for name, arm in study.document['arms'].items():
        provider = _object(arm.get('provider'), f'study.arms.{name}.provider')
        configuration = provider_configuration(provider, claim_scope, arms=study.document['arms'])
        validate_provider_binding(provider=provider, project_root=project_root,
            expected_provider_configuration=configuration,
            admitted_scopes={'zero_gpu_contract_fixture_only', required_live_provider_qualification_scope(claim_scope)},
            require_native_pair=policy is not None, evaluation_protocol=study.evaluation_protocol)
    return claim_scope


def validate_provider_binding(*, provider, project_root, expected_provider_configuration,
                              admitted_scopes, require_native_pair=False, evaluation_protocol=None):
    """Check provider receipt, retained qualification evidence and delivered schema."""
    provider_revision = _name(provider.get('revision'), 'provider.revision')
    if provider_harness(provider) == 'responses':
        from .message_provider import MessageQualification
        reference = _object(provider.get('qualification'), 'message provider qualification')
        if set(reference) != {'path', 'canonical_sha256'}:
            raise ValueError('message qualification reference fields differ')
        _, path = _qualification_path(project_root, reference['path'], 'message qualification')
        qualification = MessageQualification.load(path)
        if (qualification.canonical_sha256 != reference['canonical_sha256']
            or qualification.provider_revision != provider_revision
            or qualification.document['configuration'] != expected_provider_configuration
            or qualification.scope not in admitted_scopes):
            raise ValueError('message provider qualification differs from its frozen binding')
        return qualification
    executable_sha256 = _digest(
        provider.get("executable_sha256"), "study.arms.provider.executable_sha256"
    )
    qualification_ref = _object(
        provider.get("qualification"),
        "study.arms.provider.qualification",
    )
    if set(qualification_ref) != {"path", "canonical_sha256"}:
        raise differs(
            "provider qualification reference fields differ",
            expected=["canonical_sha256", "path"], observed=sorted(qualification_ref),
        )
    _, qualification_path = _qualification_path(
        project_root,
        qualification_ref.get("path"),
        "study.arms.provider.qualification.path",
    )
    qualification = ProviderQualificationReceipt.load(qualification_path)
    expected_configuration_sha256 = sha256(
        _canonical_json_bytes(expected_provider_configuration)).hexdigest()
    expected_qualification = {
        "provider_revision": provider_revision, "executable_sha256": executable_sha256,
        "configuration_sha256": expected_configuration_sha256,
        "initial_and_resume_equivalent": True, "file_lifecycle_observed": True,
        "usage_observed": True, "qualified": True, "scope": sorted(admitted_scopes),
        "canonical_sha256": qualification_ref.get("canonical_sha256"),
    }
    observed_qualification = {
        "provider_revision": qualification.provider_revision,
        "executable_sha256": qualification.executable_sha256,
        "configuration_sha256": qualification.configuration_sha256,
        "initial_and_resume_equivalent": qualification.initial_and_resume_equivalent,
        "file_lifecycle_observed": qualification.file_lifecycle_observed,
        "usage_observed": qualification.usage_observed, "qualified": qualification.qualified,
        "scope": qualification.scope, "canonical_sha256": qualification.canonical_sha256,
    }
    if (
        qualification.provider_revision != provider_revision
        or qualification.executable_sha256 != executable_sha256
        or qualification.configuration_sha256 != expected_configuration_sha256
        or not qualification.initial_and_resume_equivalent
        or not qualification.file_lifecycle_observed
        or not qualification.usage_observed
        or not qualification.qualified
        or qualification.scope not in admitted_scopes
        or qualification_ref.get("canonical_sha256") != qualification.canonical_sha256
    ):
        raise differs(
            "provider qualification bytes or capability",
            expected=expected_qualification, observed=observed_qualification,
        )
    if (require_native_pair and qualification.scope != 'zero_gpu_contract_fixture_only'
        and paired_protocol(evaluation_protocol) is None):
        raise ValueError('new live native Campaign requires explicit fixed-baseline paired policy')
    qualification_anchor = provider.get("qualification_anchor")
    if qualification.scope == "zero_gpu_contract_fixture_only":
        if qualification_anchor is not None:
            raise ValueError("fixture provider qualification anchor must be null")
    else:
        anchor_reference = _object(
            qualification_anchor,
            "study.arms.provider.qualification_anchor",
        )
        if set(anchor_reference) != {"path", "canonical_sha256"}:
            raise differs(
                "provider qualification anchor reference",
                expected=["canonical_sha256", "path"], observed=sorted(anchor_reference),
            )
        _, anchor_path = _qualification_path(
            project_root,
            anchor_reference.get("path"),
            "study.arms.provider.qualification_anchor.path",
        )
        anchor = _object(
            json.loads(anchor_path.read_text(encoding="utf-8")),
            "study.arms.provider.qualification_anchor.document",
        )
        anchor_fields = {
            "schema_version", "kind", "run_id", "evidence_root", "authority_sha256",
            "qualification_receipt_sha256", "immediate_audit_integrity", "terminal_seal_sha256",
        }
        if set(anchor) != anchor_fields:
            raise differs(
                "provider qualification anchor fields differ",
                expected=sorted(anchor_fields), observed=sorted(anchor),
            )
        if (
            anchor.get("schema_version") != 1
            or anchor.get("kind")
            != ("provider_qualification_evidence_anchor" if provider_harness(provider) == "claude-code" else "codex_provider_qualification_evidence_anchor")
            or not isinstance(anchor.get("run_id"), str)
            or not anchor["run_id"]
            or not isinstance(anchor.get("evidence_root"), str)
            or not anchor["evidence_root"]
            or _digest(
                anchor.get("authority_sha256"),
                "provider qualification anchor authority",
            )
            != anchor.get("authority_sha256")
            or anchor.get("qualification_receipt_sha256")
            != qualification.canonical_sha256
            or anchor.get("immediate_audit_integrity") is not True
            or _digest(
                anchor.get("terminal_seal_sha256"),
                "provider qualification anchor terminal seal",
            )
            != anchor.get("terminal_seal_sha256")
            or _digest(
                anchor_reference.get("canonical_sha256"),
                "study.arms.provider.qualification_anchor.canonical_sha256",
            )
            != sha256(_canonical_json_bytes(anchor)).hexdigest()
        ):
            raise ValueError("provider qualification anchor evidence differs")
    for field in (("output_schema",) if provider_harness(provider) == "codex" else ()):
        reference = _object(provider.get(field), f"study.arms.provider.{field}")
        if set(reference) != {"path", "sha256"}:
            raise differs(
                f"Study Contract provider {field} reference",
                expected=["path", "sha256"], observed=sorted(reference),
            )
        _, path = source_reference_path(
            project_root, reference.get("path"), f"study.arms.provider.{field}.path"
        )
        expected_sha256 = _digest(reference.get("sha256"), f"study.arms.provider.{field}.sha256")
        observed_sha256 = sha256(path.read_bytes()).hexdigest()
        if expected_sha256 != observed_sha256:
            raise differs(
                f"Study Contract provider {field} bytes differ",
                expected=expected_sha256, observed=observed_sha256,
            )
    return qualification


def validate_paired_baseline(*,project_root,workload,evaluation,execution,route,baseline_lowering,manifest_parser):
    """One owner for the selected baseline's source, launch and incumbent relation."""
    fixed = _object(execution['fixed_baseline'], 'execution.fixed_baseline')
    sealed_baseline = load_baseline_bundle(project_root, fixed['bundle_path'])
    validate_pair_candidates(sealed_baseline, sealed_baseline, workload, str(evaluation['case_id']))
    selection = fixed.get('selection')
    incumbent_baseline = False
    if selection is not None:
        incumbent_baseline = admit_baseline_selection(
            selection, candidate=fixed['candidate'], workload=workload,
            case_id=str(evaluation['case_id']), backend=str(route['backend']),
            evaluation_protocol=evaluation,
        )
    if fixed['candidate'] != candidate_identity(sealed_baseline):
        raise ValueError('fixed baseline identity differs from its sealed artifact')
    if incumbent_baseline:
        # Its complete executable contract was audited before promotion and its
        # exact current registry identity and Workload ABI were checked above.
        # It may be a Program or native kernel, independent of the starter source.
        return
    requirements = baseline_lowering.toolchain_requirements
    source = sealed_baseline.artifact_payloads.get('lowered_source')
    if source is None:
        raise ValueError('fixed baseline requires retained Compiler lowering source')
    manifest = manifest_parser(json.loads(sealed_baseline.artifact_payloads['launch_manifest']))
    if route["backend"] == "metal":
        source_matches = source == baseline_lowering.source.encode()
        grid, block = requirements["threadgroups_per_grid"], tuple(requirements["threads_per_threadgroup"])
    else:
        expected_source = native_source(baseline_lowering.source.encode(), requirements)
        observed_source = native_source(source, requirements)
        source_matches = ast.dump(ast.parse(observed_source)) == ast.dump(ast.parse(expected_source))
        # The width comes from the Target that declares it. Reading a shared 32 here
        # refused every wave64 baseline, and said the Compiler kernel differed.
        _, target_path = source_reference_path(
            project_root, f"compiler/targets/{workload.target}.json",
            'paired baseline target')
        grid = requirements['grid']
        block = tuple(native_block(
            requirements, warp_size=Target.load(target_path).warp_size))
        # How many pointers the kernel takes beyond its tensors is the kernel's own
        # fact, not a per-backend constant. For AMDGCN it is in the sealed assembly's
        # `.amdgpu_metadata`; the table's 2 was right for every Triton target there
        # was when it was written, and is still the CUDA route's. The route is read
        # from the frozen Compiler kernel's own compile contract, which carries the
        # code object, architecture and lane width of the Target it was lowered for.
        if route["backend"] == "triton":
            expected_hidden = _hidden_pointers(
                triton_route(requirements), sealed_baseline.artifact_payloads,
                len(workload.tensor_abi(str(evaluation['case_id']))))
        else:
            expected_hidden = backend_policy(route["backend"]).hidden_null_pointer_parameters
        if manifest.hidden_null_pointer_parameters != expected_hidden:
            raise differs(
                'fixed baseline hidden pointer commitments differ',
                expected=expected_hidden, observed=manifest.hidden_null_pointer_parameters,
            )
    reference_differs = (not source_matches or list(manifest.grid) != list(grid)
                         or manifest.block != block)
    if reference_differs:
        raise differs(
            'fixed baseline differs from the frozen Compiler kernel or launch commitments',
            expected={'candidate': fixed['candidate'], 'source_matches': True,
                      'grid': list(grid), 'block': block},
            observed={'candidate': candidate_identity(sealed_baseline), 'source_matches': source_matches,
                      'grid': list(manifest.grid), 'block': manifest.block},
        )


def validate_backend_assay(*,route,evaluation,workload,attribution_evaluation):
    """Check task timing, validation coverage and attribution against its backend."""
    if route["backend"] == "metal":
        if (evaluation.get("paired_timing", {}).get("kind") not in METAL_KINDS
                or validation_case_ids(evaluation) != tuple(workload.case_ids)
                or attribution_evaluation != _ATTRIBUTION_EVALUATION):
            raise ValueError("Metal optimization must bind its paired assay, all Workload cases and attribution")
    elif evaluation.get("paired_timing", {}).get("kind") in METAL_KINDS:
        raise ValueError("Metal paired assay cannot evaluate a different backend")


def validate_evaluation(
    *,
    attribution_evaluation,
    baseline_lowering,
    manifest_parser,
    policy,
    project_root,
    skeleton_document,
    study,
    workload,
):
    """Validate the numerical assay against the Workload and its execution admission.

    What the assay says by its own bytes (searches, materiality, Ralph limits) is
    `StudyContract.load`'s; this is the half that needs the Workload, the sealed
    baseline and the checkout.
    """
    evaluation = study.evaluation_protocol
    workload.case(_name(evaluation.get("case_id"), "study.evaluation_protocol.case_id"))
    assay = paired_protocol(evaluation)
    single_environment = study.comparison is None
    route = study.arms["open_cake"]["lowering_route"]
    if assay is not None and policy is None and not single_environment:
        raise ValueError('fixed-baseline assay requires the same-backend native Study')
    if single_environment and assay is None:
        # A single-arm optimization campaign selects on getting faster, so with no timed
        # assay there is nothing to select on. Say which of the two reasons applies: a
        # Study that declares a measurement-coverage limitation is not misconfigured, it
        # is running against a target no timer has been named for, and reporting that as
        # a missing assay sends the reader to fix a configuration that is already correct.
        coverage = evaluation.get("measurement_coverage")
        if isinstance(coverage, Mapping) and coverage.get("timed_assay") == "unavailable":
            raise ValueError(
                "single-environment optimization requires a timed assay, and this Study "
                f"reports none is available: {coverage.get('reason')}. Correctness "
                "evaluation of a sealed candidate for this target does not need one; a "
                "campaign that selects on latency does."
            )
        raise ValueError("single-environment optimization requires an explicit fixed-baseline paired assay")
    if route['backend']=='metal' and not single_environment:
        raise ValueError('Metal task optimization requires a single authoring environment')
    validate_backend_assay(route=route,evaluation=evaluation,workload=workload,
                           attribution_evaluation=attribution_evaluation)
    # Keep the external Study's existing case-projection policy here. Independent
    # Run admission separately enforces its Workload's all-cases requirement.
    if route['backend'] != 'metal' and ("validation_case_ids" in evaluation
            or single_environment and workload.document['validation'].get('all_cases_required') is True):
        if (validation_case_ids(evaluation) != tuple(workload.case_ids)
                or workload.document['validation'].get('all_cases_required') is not True):
            raise differs('CUDA validation case projection differs from Workload validation',
                expected={'validation_case_ids':tuple(workload.case_ids),'all_cases_required':True},
                observed={'validation_case_ids':validation_case_ids(evaluation),
                          'all_cases_required':workload.document['validation'].get('all_cases_required')})
    execution = study.execution
    expected_execution_fields = {'target', 'executor_revision', 'broker_execution_sha256', 'gpu', 'sandbox'}
    if assay is not None:
        expected_execution_fields.update({'fixed_baseline', 'runtime_config'})
    if set(execution) != expected_execution_fields:
        raise differs(
            "Study Contract execution fields differ",
            expected=sorted(expected_execution_fields), observed=sorted(execution),
        )
    if assay is not None:
        validate_paired_baseline(project_root=project_root,workload=workload,evaluation=evaluation,
            execution=execution,route=route,baseline_lowering=baseline_lowering,manifest_parser=manifest_parser)
    provider_sandbox = study.arms["open_cake"]["provider"].get("sandbox")
    if (
        execution.get("target") != workload.target
        or execution.get("target") != skeleton_document.get("target")
        or execution.get("sandbox") != provider_sandbox
    ):
        raise differs(
            "Study Contract execution authority",
            expected={"target": workload.target, "skeleton_target": workload.target,
                      "sandbox": provider_sandbox},
            observed={"target": execution.get("target"),
                      "skeleton_target": skeleton_document.get("target"),
                      "sandbox": execution.get("sandbox")},
        )
    _digest(
        execution.get("broker_execution_sha256"),
        "study.execution.broker_execution_sha256",
    )
    return evaluation, execution


def admit_run_inputs(specification, *, project_root, workload_loader):
    """The complete Run dependency boundary, before provider/Evidence side effects."""
    from .executor import ExecutorRevision
    from .provider_policy import execution_configuration
    from .bindings import load_baseline_bundle
    from open_cake_ir.evaluation.paired import candidate_identity, validate_pair_candidates, validation_case_ids, paired_protocol

    from .bindings import _resolve_compiler_reference
    from .reference_access import validate_reference_handoff

    document = specification.document
    _resolve_compiler_reference(project_root, document['compiler_revision'], 'run.compiler_revision', template=False)
    ExecutorRevision.load_reference(project_root, document['execution']['executor_revision'], 'run.executor')
    _, workload_path = source_reference_path(project_root, document['workload']['path'], 'run.workload.path')
    workload = workload_loader(workload_path)
    if workload.canonical_sha256 != document['workload']['canonical_sha256']:
        raise ValueError('Run Workload bytes differ')
    protocol, execution, authoring = (document[field] for field in ('evaluation_protocol', 'execution', 'authoring'))
    workload.case(protocol['case_id'])
    if execution['target'] != workload.target or execution.get('sandbox') != authoring['provider'].get('sandbox'):
        raise ValueError('Run target or author sandbox differs from its Workload and provider')
    target = Target.load(project_root / f"compiler/targets/{execution['target']}.json")
    gpu = execution.get('gpu')
    if (not isinstance(gpu, dict) or set(gpu) != {'name', 'count', 'mode'}
        or gpu['name'] not in target.device_names or type(gpu['count']) is not int or gpu['count'] != 1
        or gpu['mode'] not in {'local_serialized', 'exclusive'}):
        raise ValueError('Run device admission differs from its exact Target')
    _digest(execution.get('broker_execution_sha256'), 'run.execution.broker_execution_sha256')
    if workload.document['validation'].get('all_cases_required') is True:
        if validation_case_ids(protocol) != tuple(workload.case_ids):
            raise ValueError('Run evaluation omits Workload validation cases')
    if paired_protocol(protocol) is not None:
        fixed = _object(execution.get('fixed_baseline'), 'run.execution.fixed_baseline')
        baseline = load_baseline_bundle(project_root, fixed.get('bundle_path'))
        if candidate_identity(baseline) != fixed.get('candidate'):
            raise ValueError('Run baseline artifact differs from its frozen selection')
        validate_pair_candidates(baseline, baseline, workload, protocol['case_id'])
    validate_reference_handoff(project_root, {'author': authoring}, workload=workload, case_id=protocol['case_id'])
    from .python_reference import read_skeleton_reference
    for name in ('scaffold', * (('launch_contract', 'candidate_skeleton') if specification.environment_kind == 'direct_cuda' else ())):
        reference = _object(authoring.get(name), f'run.authoring.{name}')
        if set(reference) != {'path', 'sha256'}:
            raise ValueError(f'Run {name} reference fields differ')
        _, path = source_reference_path(project_root, reference['path'], f'run.authoring.{name}')
        if sha256(path.read_bytes()).hexdigest() != reference['sha256']:
            raise ValueError(f'Run {name} bytes differ')
    if specification.environment_kind == 'open_cake':
        python_clean_start = (authoring.get('reference_access') == 'clean_start'
                              and authoring.get('input_format') == 'python_source_v1')
        if python_clean_start:
            starter_reference = _object(authoring.get('python_starter'),
                'run.authoring.python_starter')
            if set(starter_reference) != {'path'}:
                raise ValueError('Python clean-start Run reference fields differ')
            _, starter_path = source_reference_path(project_root,
                starter_reference['path'], 'run.authoring.python_starter')
            if starter_path.suffix != '.py':
                raise ValueError('Python clean-start Run requires a .py starter')
        else:
            if authoring.get('input_format') == 'python_source_v1':
                starter_reference = _object(authoring.get('schedule_skeleton'),
                    'run.authoring.schedule_skeleton')
                _, starter_path = source_reference_path(project_root,
                    starter_reference.get('path'), 'run.authoring.schedule_skeleton')
                if starter_path.suffix != '.py':
                    raise ValueError('Python-only Run requires a .py Schedule starter')
            _, skeleton = read_skeleton_reference(project_root, authoring.get('schedule_skeleton'))
            if skeleton.get('target') != execution['target'] or skeleton.get('lowering') != authoring.get('lowering_route'):
                raise ValueError('Run Schedule skeleton target or lowering route differs')
    for name, reference in document['reference_inputs'].items():
        if name == 'baseline_programs':
            continue
        _, skeleton = read_skeleton_reference(project_root, reference, f'Run {name}')
        from .pairing import native_backend
        if (skeleton.get('target') != execution['target']
            or skeleton.get('lowering', {}).get('backend') != native_backend(specification.environment_kind).backend):
            raise ValueError('Run baseline Schedule target or backend differs')
    provider = authoring['provider']
    if 'output_schema' in provider:
        _, schema_path = source_reference_path(project_root, provider['output_schema']['path'], 'run.provider.output_schema')
        schema = json.loads(schema_path.read_bytes())
        arm_schema = schema.get('properties', {}).get('arm', {})
        if (arm_schema.get('type') != 'string'
            or 'enum' in arm_schema and specification.condition_id not in arm_schema['enum']
            or 'const' in arm_schema and specification.condition_id != arm_schema['const']):
            raise ValueError('provider output schema excludes the assigned condition id')
    configuration = execution_configuration(provider)
    scope = ('live_two_turn_message_provider' if provider_harness(provider) == 'responses' else
             'live_two_turn_current_provider' if configuration.get('event_contract', 'closed_file_change_v1') == 'closed_file_change_v1'
             else 'live_two_turn_tool_rich_provider')
    qualification = validate_provider_binding(provider=provider, project_root=project_root,
        expected_provider_configuration=configuration,
        admitted_scopes={'zero_gpu_contract_fixture_only', scope})
    return workload, qualification
