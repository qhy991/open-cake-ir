"""Study admission checks grouped by their external authority boundaries."""

from __future__ import annotations

import ast
import json
from hashlib import sha256
from pathlib import Path
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


def admit_native_skill_authoring(*, authoring, project_root):
    """Admit native authoring before credentials, factories, evidence or early returns.

    Qualifications cover environment kinds, not arbitrary Run/condition identifiers.
    The full receipt/configuration/schema and anchored native inputs keep their
    existing admission owners; no caller-supplied validation flag is accepted.
    """
    from .author_home import ISOLATED_SKILL_PACKAGE_V1
    from .provider_policy import execution_configuration
    provider = authoring.get('provider', {})
    if provider.get('author_home_policy') != ISOLATED_SKILL_PACKAGE_V1:
        return None
    if (not isinstance(provider.get('qualification'), Mapping)
        or not isinstance(provider.get('qualification_anchor'), Mapping)):
        raise ValueError('native skill discovery and actual initial/resume delivery are not qualified; receipt and anchor are required')
    kind = authoring.get('environment_kind')
    if not isinstance(kind, str) or not kind:
        raise ValueError('native authoring lacks its environment kind')
    configuration = execution_configuration(provider)
    scope = ('live_two_turn_current_provider'
             if configuration.get('event_contract', 'closed_file_change_v1') == 'closed_file_change_v1'
             else 'live_two_turn_tool_rich_provider')
    return validate_provider_binding(provider=provider, project_root=project_root,
        expected_provider_configuration=configuration, admitted_scopes={scope},
        required_environment_kinds=(kind,))


def validate_provider(*, open_cake, policy, project_root, study):
    """Matched-Study input policy; runtime qualification has an independent owner."""
    claim_scope = str(study.document['claim_scope'])
    for name, arm in study.document['arms'].items():
        provider = _object(arm.get('provider'), f'study.arms.{name}.provider')
        configuration = provider_configuration(provider, claim_scope, arms=study.document['arms'])
        validate_provider_binding(provider=provider, project_root=project_root,
            expected_provider_configuration=configuration,
            admitted_scopes={'zero_gpu_contract_fixture_only', required_live_provider_qualification_scope(claim_scope)},
            require_native_pair=policy is not None, evaluation_protocol=study.evaluation_protocol,
            required_environment_kinds=(arm['environment_kind'],), authoring=arm)
    return claim_scope


def validate_editable_starter_observation(*, authority, payload, read_object, provider, qualification):
    """Reconstruct the existing qualification plan, exact sources and native Edit events."""
    from .claude import parse_claude_turn_events
    from .provider_documents import PYTHON_CANDIDATE_BUNDLE_V1
    from .task_package import TaskPackage, render_task_request
    if (authority.get('harness') != 'claude-code' or authority.get('arms') != ['open_cake']
            or authority.get('submission_contract') != PYTHON_CANDIDATE_BUNDLE_V1
            or authority.get('turns') != ['initial_seed_update', 'same_thread_resume_update']
            or authority.get('gpu_execution_authorized') is not False
            or authority.get('provider_revision') != qualification.provider_revision
            or any(authority.get(key) != provider.get(key) for key in ('model', 'reasoning_effort', 'event_contract'))):
        raise ValueError('editable starter qualification has no declared initial-Edit/resume-Edit lifecycle')
    references = payload.get('objects', [])
    def raw(role):
        matches = [item for item in references if item.get('role') == role]
        if len(matches) != 1:
            raise ValueError('editable starter qualification is missing unique ' + role)
        return read_object(matches[0])
    if json.loads(raw('qualification_receipt')) != dict(qualification.document):
        raise ValueError('editable starter qualification receipt differs from its observed record')
    documents = json.loads(raw('qualification_reference'))
    if set(documents) != {'open_cake'} or set(documents['open_cake']) != {'task_markdown', 'agents_markdown'}:
        raise ValueError('editable starter qualification reference material differs')
    task, agents = (documents['open_cake'][key] for key in ('task_markdown', 'agents_markdown'))
    plans = [line.removeprefix('QUALIFICATION_PLAN_JSON=') for line in task.splitlines()
             if line.startswith('QUALIFICATION_PLAN_JSON=')]
    if len(plans) != 1:
        raise ValueError('editable starter qualification has no unique frozen plan')
    plan = json.loads(plans[0])
    turns = plan.get('turns', [])
    if (len(turns) != 2 or [(item.get('turn'), item.get('change')) for item in turns] != [(1, 'update'), (2, 'update')]
            or not all(isinstance(item.get('submission'), str) for item in turns)
            or turns[0]['submission'] == turns[1]['submission']):
        raise ValueError('editable starter qualification plan must update distinct complete sources twice')
    thread, initial_invocation = None, None
    for number, phase in ((1, 'initial'), (2, 'resumed')):
        prefix = 'open_cake_' + phase + '_'
        item = turns[number - 1]
        if raw(prefix + 'source_file') != item['submission'].encode():
            raise ValueError('editable starter qualification submitted source differs from its plan')
        parsed = parse_claude_turn_events(raw(prefix + 'provider_events'),
            expected_terminal_message=_canonical_json_bytes(item['terminal_message']).decode(),
            event_contract=provider['event_contract'], response_aliases=provider.get('response_model_aliases', ()),
            candidate_filename='candidate-set.py')
        if (parsed.write_tools != ('Edit',) or parsed.candidate_path != plan['candidate_path']
                or parsed.reported_models[0] != provider['model'] or parsed.provider_tokens <= 0
                or thread is not None and parsed.thread_id != thread):
            raise ValueError('editable starter qualification lacks successful same-thread native Edits')
        thread = parsed.thread_id
        projection = json.loads(raw(prefix + 'task_projection'))
        package = TaskPackage(projection['run_id'], 'open_cake', task, agents)
        prompt, expected_projection = render_task_request(package, {'turn': number})
        invocation = json.loads(raw(prefix + 'invocation'))
        argv = invocation.get('argv', [])
        if (raw(prefix + 'task_projection') != expected_projection or len(argv) < 2 or argv[-2:] != ['--', prompt]
                or str(Path(invocation['cwd']) / 'candidate-set.py') != plan['candidate_path']
                or invocation.get('provider_revision') != qualification.provider_revision
                or invocation.get('sandbox') != 'none'):
            raise ValueError('editable starter qualification invocation differs from its frozen task')
        if number == 1:
            if invocation.get('thread_id') is not None or '--resume' in argv:
                raise ValueError('editable starter qualification initial invocation was already resumed')
            initial_invocation = invocation
        elif (invocation.get('thread_id') != thread or argv[-4:-2] != ['--resume', thread]
                or initial_invocation['argv'][:-2] != argv[:-4]
                or any(initial_invocation.get(key) != invocation.get(key) for key in
                       ('cwd', 'sandbox', 'provider_revision', 'removed_environment'))):
            raise ValueError('editable starter qualification resumed invocation differs')


def require_editable_starter_qualification(qualification, anchor, provider):
    """Require anchored live observations; old receipt booleans cannot grant this path."""
    from open_cake_ir.evidence import EvidenceStore
    if (qualification.scope != 'live_two_turn_tool_rich_provider'
            or anchor.get('kind') != 'provider_qualification_evidence_anchor'
            or anchor.get('qualification_receipt_sha256') != qualification.canonical_sha256):
        raise ValueError('editable starter requires its live anchored qualification')
    evidence = EvidenceStore.open(anchor['evidence_root'])
    audit = evidence.audit_run(anchor['run_id'])
    endpoint = audit.endpoint
    if (not audit.archive_integrity or not audit.filesystem_custody_verified
            or audit.authority_sha256 != anchor.get('authority_sha256')
            or audit.terminal_seal_sha256 != anchor.get('terminal_seal_sha256')
            or audit.protocol_adherence != 'adhered' or audit.endpoint_observation != 'qualified'
            or not isinstance(endpoint, dict) or endpoint.get('add_observed') is not False
            or endpoint.get('qualification_receipt_sha256') != qualification.canonical_sha256
            or any(endpoint.get(key) is not True for key in ('update_observed', 'thread_continuity_observed',
                'candidate_changed', 'usage_observed', 'reference_visibility_observed'))):
        raise ValueError('editable starter initial-Edit/resume-Edit qualification is unverified')
    events = evidence.replay_events(audit.run_id)
    observations = [event['payload'] for event in events if event['kind'] == 'provider_qualification_observed']
    if len(observations) != 1 or any(event['kind'] == 'provider_qualification_failed' for event in events):
        raise ValueError('editable starter qualification must retain one successful observation')
    validate_editable_starter_observation(authority=evidence.replay_authority(audit.run_id),
        payload=observations[0], read_object=evidence.read_object, provider=provider, qualification=qualification)


def validate_provider_binding(*, provider, project_root, expected_provider_configuration,
                              admitted_scopes, require_native_pair=False, evaluation_protocol=None,
                              required_environment_kinds=(), authoring=None):
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
    from .author_home import CODEX_HOME_POLICIES, ISOLATED_SKILL_PACKAGE_V1
    if (provider.get('author_home_policy') in CODEX_HOME_POLICIES
        and (qualification.system_skills_sha256 is None
             or provider.get('system_skills_sha256')
             != qualification.system_skills_sha256)):
        raise ValueError('Provider system skills differ from the qualified author home')
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
    if (provider.get('author_home_policy') == ISOLATED_SKILL_PACKAGE_V1
        and qualification.scope != 'zero_gpu_contract_fixture_only'):
        from .native_skill_qualification import verify_qualification_evidence
        verify_qualification_evidence(qualification=qualification, anchor=anchor,
            required_environment_kinds=required_environment_kinds)
    if provider.get('isolation_policy') is not None:
        if qualification.scope == 'zero_gpu_contract_fixture_only':
            raise ValueError('isolated Claude scientific authoring requires live isolation evidence')
        from open_cake_ir.evidence import EvidenceStore
        retained = EvidenceStore.open(anchor['evidence_root'])
        audit = retained.audit_run(anchor['run_id'])
        if (not audit.archive_integrity or not audit.filesystem_custody_verified
            or audit.terminal_seal_sha256 != anchor['terminal_seal_sha256']):
            raise ValueError('Claude isolation qualification evidence is unverified')
        probes = [e['payload']['observation'] for e in retained.replay_events(anchor['run_id'])
                  if e['kind'] == 'provider_read_isolation_observed']
        from .claude_isolation import validate_probe_observation
        if not probes:
            raise ValueError('Claude isolation qualification lacks its actual OS probe')
        for observation in probes: validate_probe_observation(observation)
    from .task_package import TaskPackage
    if authoring is not None and TaskPackage.permits_editable_starter(authoring):
        if qualification.scope == 'zero_gpu_contract_fixture_only':
            raise ValueError('editable starter requires live initial-Edit and resume-Edit qualification')
        require_editable_starter_qualification(qualification, anchor, provider)
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


def admit_paired_baseline_artifact(*, project_root, workload, evaluation, execution, route):
    """Admit one sealed opponent and its existing selection policy.

    Return whether the selected opponent may be independent of current starter
    emission. CUBIN keeps its existing source/launch/ABI checks with the caller;
    author route identity is not evidence of an old binary's argument contract.
    This software check establishes no successor device or measurement readiness.
    """
    fixed = _object(execution['fixed_baseline'], 'execution.fixed_baseline')
    sealed_baseline = load_baseline_bundle(project_root, fixed['bundle_path'])
    manifests = validate_pair_candidates(sealed_baseline, sealed_baseline, workload, str(evaluation['case_id']))
    if fixed['candidate'] != candidate_identity(sealed_baseline):
        raise ValueError('fixed baseline identity differs from its sealed artifact')
    selection = fixed.get('selection')
    incumbent_baseline = False
    if selection is not None:
        incumbent_baseline = admit_baseline_selection(
            selection, candidate=fixed['candidate'], workload=workload,
            case_id=str(evaluation['case_id']), backend=str(route['backend']),
            evaluation_protocol=evaluation,
        )
    explicit = selection is not None and selection['policy'] == 'explicit_fixed_bundle'
    independent_explicit = False
    if explicit:
        from open_cake_ir.compiler.target import CodeObject
        _, target_path = source_reference_path(project_root,
            f'compiler/targets/{workload.target}.json', 'paired baseline target')
        target = Target.load(target_path)
        independent_explicit = target.code_object in {
            CodeObject.MCFATBIN, CodeObject.HSACO, CodeObject.METAL_BINARY_ARCHIVE}
        # Retain the native argument/family checks for the two inspected binary
        # formats. Their facts come from this Target and this sealed artifact, not
        # from the successor Compiler's new launch shape or source.
        if not sealed_baseline.is_program and target.code_object in {CodeObject.MCFATBIN, CodeObject.HSACO}:
            from open_cake_ir.compiler.backends.triton import target_route_facts
            facts = {'target': target.target_id, **target_route_facts(target)}
            manifest = manifests['baseline']
            native_route = triton_route(facts)
            if not {native_route.text_role, native_route.binary_role} <= set(sealed_baseline.artifact_payloads):
                raise ValueError('fixed baseline lacks native argument inspection artifacts')
            expected_hidden = _hidden_pointers(native_route, sealed_baseline.artifact_payloads,
                len(manifest.tensor_abi), codegen_arch=facts.get('codegen_arch'),
                kernel_name=manifest.kernel_name)
            if manifest.hidden_null_pointer_parameters != expected_hidden:
                raise differs('fixed baseline hidden pointer commitments differ',
                    expected=expected_hidden, observed=manifest.hidden_null_pointer_parameters)
    return sealed_baseline, bool(incumbent_baseline or independent_explicit)


def validate_paired_baseline(*,project_root,workload,evaluation,execution,route,baseline_lowering,manifest_parser):
    """One owner for the selected baseline's source, launch and incumbent relation."""
    fixed = _object(execution['fixed_baseline'], 'execution.fixed_baseline')
    sealed_baseline, independent = admit_paired_baseline_artifact(
        project_root=project_root, workload=workload, evaluation=evaluation,
        execution=execution, route=route)
    if independent:
        # The current incumbent or an explicitly selected fixed bundle can differ
        # from the current starter. Neither selection grants author reference access.
        return
    if baseline_lowering is None:
        raise ValueError('starter baseline requires the frozen Compiler lowering')
    from open_cake_ir.compiler.program import LoweredProgram
    if isinstance(baseline_lowering, LoweredProgram):
        from open_cake_ir.evaluation.program import program_components, single_kernel_lowering
        baseline_lowering.validate_binding()
        single = single_kernel_lowering(baseline_lowering)
        if single is not None and not sealed_baseline.is_program:
            _validate_starter_kernel(project_root, sealed_baseline, single, route,
                                     manifest_parser(json.loads(sealed_baseline.artifact_payloads['launch_manifest'])))
            return
        manifest, children, manifests = program_components(sealed_baseline)
        if manifest.program.document != baseline_lowering.program.document:
            raise ValueError('fixed baseline Program differs from the complete frozen Compiler Program')
        for stage, lowering in zip(baseline_lowering.program.stages, baseline_lowering.lowerings, strict=True):
            if stage.schedule.lowering.backend.value != route['backend']:
                raise ValueError(f'fixed baseline Program stage {stage.name!r} backend differs')
            if manifests[stage.name].kernel_name != lowering.toolchain_requirements['kernel_entry_point']:
                raise ValueError(f'fixed baseline Program stage {stage.name!r} entry point differs')
            try:
                _validate_starter_kernel(project_root, children[stage.name], lowering, route, manifests[stage.name])
            except ValueError as error:
                raise ValueError(f'fixed baseline Program stage {stage.name!r}: {error}') from error
        return
    if sealed_baseline.is_program:
        raise ValueError('fixed baseline Program requires complete frozen Program lowering')
    _validate_starter_kernel(project_root, sealed_baseline, baseline_lowering, route,
                             manifest_parser(json.loads(sealed_baseline.artifact_payloads['launch_manifest'])))


def _validate_starter_kernel(project_root, sealed_baseline, baseline_lowering, route, manifest):
    """Compare one real kernel's retained source and launch against its lowering."""
    requirements = baseline_lowering.toolchain_requirements
    source = sealed_baseline.artifact_payloads.get('lowered_source')
    if source is None:
        raise ValueError('fixed baseline requires retained Compiler lowering source')
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
            project_root, f"compiler/targets/{baseline_lowering.target}.json",
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
                len(manifest.tensor_abi),
                codegen_arch=requirements.get('codegen_arch'),
                kernel_name=requirements['kernel_entry_point'])
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
            expected={'candidate': candidate_identity(sealed_baseline), 'source_matches': True,
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
    """The complete Run dependency boundary, before provider/Evidence side effects.

    Sealed opponent and selection admission is common to both entry paths. Source
    equality, when independent admission is not granted, remains with task
    preparation and legacy Study preflight, which own the current baseline
    Schedule. This boundary retains its existing sealed-bundle/ABI scope; it does
    not independently prove a CUBIN's source or hidden-pointer contract.
    """
    from .executor import ExecutorRevision
    from .provider_policy import execution_configuration
    from open_cake_ir.evaluation.paired import validation_case_ids, paired_protocol

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
        baseline_route = authoring.get('lowering_route')
        if baseline_route is None:
            from .toolchains import toolchain_for_arm
            baseline_route = {'backend': toolchain_for_arm(specification.environment_kind).backend.value}
        admit_paired_baseline_artifact(project_root=project_root, workload=workload,
            evaluation=protocol, execution=execution, route=baseline_route)
    validate_reference_handoff(project_root, {'author': authoring}, workload=workload, case_id=protocol['case_id'])
    from .python_reference import read_skeleton_reference, skeleton_route
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
            if skeleton.get('target') != execution['target'] or skeleton_route(skeleton) != authoring.get('lowering_route'):
                raise ValueError('Run Schedule skeleton target or lowering route differs')
            if 'program_id' in skeleton:
                from .pairing import bind_baseline
                from .provider_documents import PYTHON_CANDIDATE_BUNDLE_V1
                if authoring['provider'].get('submission_contract') != PYTHON_CANDIDATE_BUNDLE_V1:
                    raise ValueError('Program starter requires the Python candidate-bundle submission contract')
                bind_baseline(skeleton, workload, str(protocol['case_id']),
                              backend=authoring['lowering_route']['backend'])
    for name, reference in document['reference_inputs'].items():
        if name == 'baseline_programs':
            continue
        _, skeleton = read_skeleton_reference(project_root, reference, f'Run {name}')
        from .pairing import native_backend
        if 'program_id' in skeleton:
            raise ValueError('native comparison does not support a Program starter')
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
        admitted_scopes={'zero_gpu_contract_fixture_only', scope},
        required_environment_kinds=(specification.environment_kind,), authoring=authoring)
    return workload, qualification
