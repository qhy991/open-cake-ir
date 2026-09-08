"""Study binding and authoring preparation without execution side effects."""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Callable, Mapping, cast

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.performance.empirical_cost import EmpiricalCostModel
from open_cake_ir.compiler.target import cuda_target
from open_cake_ir.evaluation.paired import paired_protocol

from ._documents import _canonical_json_bytes, _digest, _name, _object, _project_path
from ._policies import (
    _ARTIFACT_OPTIMIZATION_ANALYSIS_PLAN,
    _ATTRIBUTION_EVALUATION,
    _LEGACY_ATTRIBUTION_EVALUATION,
    _MATCHED_RALPH_EVENT_VOCABULARY_V1,
    _ONE_RUN_PER_ARM_SCOPES,
    _SYSTEM_QUALIFICATION_ANALYSIS_PLAN,
    _matched_evidence_policy_version,
    _scientific_analysis_plan_version,
)
from .bindings import (
    _resolve_compiler_reference,
    load_baseline_bundle,
    qualification_path as _qualification_path,
    resolve_execution_bindings,
    resolve_executor,
)
from .contracts import CampaignLock, StudyContract
from .pairing import bind_baseline, comparison_arm, native_baseline
from .providers import (
    CANDIDATE_SET_ENVELOPE_V1,
    CODEX_DISABLED_FEATURES,
    ProviderQualificationReceipt,
    required_live_provider_qualification_scope,
)
from .ralph import RalphBudget
from .selection import _EMPIRICAL_SELECTION
from .task_package import TaskPackage, render_task_package


def task_package(
    lock: CampaignLock,
    run_id: str,
    *,
    project_root: Path,
    workload_loader: Callable,
    prepare_schedule: Callable,
) -> TaskPackage:
    workload = workload_loader(project_root / str(lock.document["workload"]["path"]))
    return render_task_package(
        project_root,
        lock,
        run_id,
        workload_contract=workload,
        prepare_schedule=prepare_schedule,
    )


def preflight(
    study_path: str | Path,
    *,
    project_root: Path,
    workload_loader: Callable,
    validate_authoring: Callable,
    manifest_parser: Callable,
    empirical_cost_model_path: str | Path | None = None,
    execution_bindings_path: str | Path | None = None,
) -> CampaignLock:
    """Resolve one Study Contract without provider, GPU or evidence side effects."""

    study = StudyContract.load(study_path)
    if study.document["kind"] == "portfolio":
        if empirical_cost_model_path is not None or execution_bindings_path is not None:
            raise ValueError("empirical selection requires artifact_optimization_only matched search")
        raise ValueError("portfolio preflight belongs to its task entrypoint")
    resolved_document, bound_executor = resolve_execution_bindings(
        project_root, study, execution_bindings_path
    )
    study = replace(study, document=resolved_document)
    workload_ref = _object(study.document.get("workload"), "study.workload")
    if set(workload_ref) != {"path", "canonical_sha256"}:
        raise ValueError("study workload reference fields differ")
    workload_relative, workload_path = _project_path(
        project_root, workload_ref.get("path"), "study.workload.path"
    )
    workload = workload_loader(workload_path)
    if workload.canonical_sha256 != _digest(
        workload_ref.get("canonical_sha256"), "study.workload.canonical_sha256"
    ):
        raise ValueError("Study Contract workload bytes differ")

    arms = _object(study.document.get("arms"), "study.arms")
    comparison = comparison_arm(arms)
    paired_triton = comparison == "native_triton"
    open_cake = _object(arms.get("open_cake"), "study.arms.open_cake")
    direct_cuda = _object(arms.get(comparison), f"study.arms.{comparison}")
    validate_authoring(workload, arms, empirical_cost_model_path=empirical_cost_model_path)
    empirical_policy = open_cake.get("candidate_selection")
    has_empirical_policy = "candidate_selection" in open_cake
    if has_empirical_policy or empirical_cost_model_path is not None:
        if (
            study.document["claim_scope"] != "artifact_optimization_only"
            or empirical_policy != {"kind": _EMPIRICAL_SELECTION}
            or empirical_cost_model_path is None
            or "maximum_candidates_per_turn" not in study.document["budget"]
        ):
            raise ValueError("empirical selection requires artifact_optimization_only candidate-set policy and an explicit model")
    open_cake_fields = {
        "environment_kind",
        "provider",
        "scaffold",
        "compiler_revision",
        "lowering_route",
        "schedule_skeleton",
        "tool_surface",
        "feedback",
    }
    if has_empirical_policy:
        open_cake_fields.add("candidate_selection")
    direct_cuda_fields = {
        "environment_kind",
        "provider",
        "scaffold",
        "launch_contract",
        "candidate_skeleton",
        "toolchain_sha256",
        "tool_surface",
        "feedback",
    }
    if paired_triton:
        open_cake_fields.update({"input_format", "toolchain_sha256"})
        direct_cuda_fields -= {"launch_contract", "candidate_skeleton"}
        direct_cuda_fields.add("baseline")
        if (open_cake.get("input_format") != "schedule_or_python_v1"
            or direct_cuda.get("baseline") != {"binding": "open_cake_lowering"}
            or open_cake.get("toolchain_sha256") != direct_cuda.get("toolchain_sha256")):
            raise ValueError("paired Triton input, baseline or common toolchain binding differs")
    if set(open_cake) != open_cake_fields or set(direct_cuda) != direct_cuda_fields:
        raise ValueError("Study Contract Authoring Environment fields differ")
    if open_cake.get("environment_kind") != "open_cake" or direct_cuda.get(
        "environment_kind"
    ) != comparison:
        raise ValueError("Study Contract Authoring Environment kinds differ")
    route = open_cake.get("lowering_route")
    if (not isinstance(route, Mapping) or set(route) != {"backend", "entry_point"}
        or route.get("backend") != "triton" or not isinstance(route.get("entry_point"), str)
        or not route["entry_point"].isidentifier()):
        raise ValueError("Study Contract Open Cake lowering route differs")
    schedule_skeleton = _object(
        open_cake.get("schedule_skeleton"), "study.arms.open_cake.schedule_skeleton"
    )
    if set(schedule_skeleton) != {"path", "canonical_sha256"}:
        raise ValueError("Study Contract Schedule skeleton reference differs")
    _, schedule_skeleton_path = _project_path(
        project_root,
        schedule_skeleton.get("path"),
        "study.arms.open_cake.schedule_skeleton.path",
    )
    skeleton_document = _object(
        json.loads(schedule_skeleton_path.read_text(encoding="utf-8")),
        "study.arms.open_cake.schedule_skeleton",
    )
    if (
        skeleton_document.get("lowering") != open_cake.get("lowering_route")
        or _digest(
            schedule_skeleton.get("canonical_sha256"),
            "study.arms.open_cake.schedule_skeleton.canonical_sha256",
        )
        != sha256(_canonical_json_bytes(skeleton_document)).hexdigest()
    ):
        raise ValueError("Study Contract Schedule skeleton bytes or lowering route differ")
    if open_cake.get("provider") != direct_cuda.get(
        "provider"
    ) or open_cake.get("scaffold") != direct_cuda.get("scaffold"):
        raise ValueError("matched Authoring Environments differ in provider or scaffold")
    _digest(direct_cuda.get("toolchain_sha256"), "study.arms.direct_cuda.toolchain_sha256")
    provider = _object(open_cake.get("provider"), "study.arms.provider")
    provider_fields = {
        "revision",
        "qualification",
        "qualification_anchor",
        "executable_sha256",
        "model",
        "reasoning_effort",
        "service_tier",
        "output_schema",
        "removed_environment",
        "sandbox",
        "cwd_policy",
        "reference_visibility",
        "disabled_features",
        "code_mode_host",
    }
    if frozenset(provider) not in {
        frozenset(provider_fields | {"web_search"}),
        frozenset(provider_fields | {"event_contract"}),
    }:
        raise ValueError("Study Contract provider configuration fields differ")
    provider_revision = _name(provider.get("revision"), "study.arms.provider.revision")
    claim_scope = cast(str, study.document["claim_scope"])
    expected_disabled_features = (
        []
        if claim_scope == "artifact_optimization_only"
        else list(CODEX_DISABLED_FEATURES)
    )
    expected_event_contract = (
        "tool_rich_candidate_v1"
        if claim_scope == "artifact_optimization_only"
        else "closed_file_change_v1"
    )
    code_mode_host = _object(provider.get("code_mode_host"), "study.arms.provider.code_mode_host")
    if (set(code_mode_host) != {"path", "sha256"}
        or not isinstance(code_mode_host.get("path"), str)
        or not Path(code_mode_host["path"]).is_absolute()
        or ".." in Path(code_mode_host["path"]).parts):
        raise ValueError("Study Contract Code Mode host identity differs")
    _digest(code_mode_host.get("sha256"), "study.arms.provider.code_mode_host.sha256")
    _name(
        provider.get("reasoning_effort"),
        "study.arms.provider.reasoning_effort",
    )
    if (
        provider.get("model") != "gpt-5.6-sol"
        or provider.get("service_tier") != "default"
        or provider.get("sandbox") != "workspace-write"
        or provider.get("cwd_policy")
        != "independent_task_workspace"
        or provider.get("reference_visibility")
        != "workspace_task_files"
        or provider.get("disabled_features") != expected_disabled_features
        or (expected_event_contract == "closed_file_change_v1"
            and provider.get("web_search") != "disabled")
        or provider.get("event_contract", "closed_file_change_v1")
        != expected_event_contract
        or provider.get("removed_environment")
        != ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    ):
        raise ValueError("Study Contract provider configuration differs")
    executable_sha256 = _digest(
        provider.get("executable_sha256"), "study.arms.provider.executable_sha256"
    )
    qualification_ref = _object(
        provider.get("qualification"),
        "study.arms.provider.qualification",
    )
    if set(qualification_ref) != {"path", "canonical_sha256"}:
        raise ValueError("provider qualification reference fields differ")
    _, qualification_path = _qualification_path(
        project_root,
        qualification_ref.get("path"),
        "study.arms.provider.qualification.path",
    )
    qualification = ProviderQualificationReceipt.load(qualification_path)
    if (
        qualification.provider_revision != provider_revision
        or qualification.executable_sha256 != executable_sha256
        or qualification.configuration_sha256
        != sha256(
            _canonical_json_bytes(
                {
                    "model": provider["model"],
                    "reasoning_effort": provider["reasoning_effort"],
                    "service_tier": provider["service_tier"],
                    "output_schema_sha256": _object(
                        provider["output_schema"], "study.arms.provider.output_schema"
                    )["sha256"],
                    "removed_environment": provider["removed_environment"],
                    "sandbox": provider["sandbox"],
                    "cwd_policy": provider["cwd_policy"],
                    "reference_visibility": provider["reference_visibility"],
                    "disabled_features": provider["disabled_features"],
                    "code_mode_host": provider["code_mode_host"],
                    **({"web_search": provider["web_search"]} if "web_search" in provider else {}),
                    **(
                        {"event_contract": provider["event_contract"]}
                        if "event_contract" in provider
                        else {}
                    ),
                    **(
                        {
                            "submission_contract": CANDIDATE_SET_ENVELOPE_V1
                        }
                    ),
                }
            )
        ).hexdigest()
        or not qualification.initial_and_resume_equivalent
        or not qualification.file_lifecycle_observed
        or not qualification.usage_observed
        or not qualification.qualified
        or qualification.scope
        not in {
            "zero_gpu_contract_fixture_only",
            required_live_provider_qualification_scope(claim_scope),
        }
        or qualification_ref.get("canonical_sha256") != qualification.canonical_sha256
    ):
        raise ValueError("provider qualification bytes or capability differs")
    if (paired_triton and qualification.scope != 'zero_gpu_contract_fixture_only'
        and paired_protocol(study.document['evaluation_protocol']) is None):
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
            raise ValueError("provider qualification anchor reference differs")
        _, anchor_path = _qualification_path(
            project_root,
            anchor_reference.get("path"),
            "study.arms.provider.qualification_anchor.path",
        )
        anchor = _object(
            json.loads(anchor_path.read_text(encoding="utf-8")),
            "study.arms.provider.qualification_anchor.document",
        )
        if set(anchor) != {
            "schema_version",
            "kind",
            "run_id",
            "evidence_root",
            "authority_sha256",
            "qualification_receipt_sha256",
            "immediate_audit_integrity",
            "terminal_seal_sha256",
        }:
            raise ValueError("provider qualification anchor fields differ")
        if (
            anchor.get("schema_version") != 1
            or anchor.get("kind")
            != "codex_provider_qualification_evidence_anchor"
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
    for field in ("output_schema",):
        reference = _object(provider.get(field), f"study.arms.provider.{field}")
        if set(reference) != {"path", "sha256"}:
            raise ValueError(f"Study Contract provider {field} reference differs")
        _, path = _project_path(
            project_root, reference.get("path"), f"study.arms.provider.{field}.path"
        )
        if _digest(reference.get("sha256"), f"study.arms.provider.{field}.sha256") != sha256(
            path.read_bytes()
        ).hexdigest():
            raise ValueError(f"Study Contract provider {field} bytes differ")
    scaffold = _object(open_cake.get("scaffold"), "study.arms.scaffold")
    if set(scaffold) != {"path", "sha256"}:
        raise ValueError("Study Contract scaffold reference differs")
    _, scaffold_path = _project_path(
        project_root, scaffold.get("path"), "study.arms.scaffold.path"
    )
    if _digest(scaffold.get("sha256"), "study.arms.scaffold.sha256") != sha256(
        scaffold_path.read_bytes()
    ).hexdigest():
        raise ValueError("Study Contract scaffold bytes differ")
    if not paired_triton:
        launch_contract = _object(
            direct_cuda.get("launch_contract"), "study.arms.direct_cuda.launch_contract"
        )
        if set(launch_contract) != {"path", "sha256"}:
            raise ValueError("Study Contract direct launch contract reference differs")
        _, launch_contract_path = _project_path(
            project_root,
            launch_contract.get("path"),
            "study.arms.direct_cuda.launch_contract.path",
        )
        if _digest(
            launch_contract.get("sha256"),
            "study.arms.direct_cuda.launch_contract.sha256",
        ) != sha256(launch_contract_path.read_bytes()).hexdigest():
            raise ValueError("Study Contract direct launch contract bytes differ")
        candidate_skeleton = _object(
            direct_cuda.get("candidate_skeleton"),
            "study.arms.direct_cuda.candidate_skeleton",
        )
        if set(candidate_skeleton) != {"path", "sha256"}:
            raise ValueError("Study Contract direct candidate skeleton reference differs")
        _, candidate_skeleton_path = _project_path(
            project_root,
            candidate_skeleton.get("path"),
            "study.arms.direct_cuda.candidate_skeleton.path",
        )
        if _digest(
            candidate_skeleton.get("sha256"),
            "study.arms.direct_cuda.candidate_skeleton.sha256",
        ) != sha256(candidate_skeleton_path.read_bytes()).hexdigest():
            raise ValueError("Study Contract direct candidate skeleton bytes differ")
    if open_cake.get("tool_surface") != (["submit_schedule_or_python"] if paired_triton else ["submit_schedule"]) or direct_cuda.get(
        "tool_surface"
    ) != (["submit_triton_kernel"] if paired_triton else ["submit_cuda"]):
        raise ValueError("Study Contract Authoring Environment tool surfaces differ")
    attribution_evaluation = _object(
        study.document.get("evaluation_protocol"),
        "study.evaluation_protocol",
    ).get("attribution_evaluation")
    if attribution_evaluation not in {
        None,
        _LEGACY_ATTRIBUTION_EVALUATION,
        _ATTRIBUTION_EVALUATION,
    }:
        raise ValueError("Study Contract attribution Evaluation differs")
    profile_feedback = ["profile"] if attribution_evaluation is not None else []
    if open_cake.get("feedback") != [
        "findings",
        "correctness",
        "qualified_timing",
        *profile_feedback,
    ] or direct_cuda.get("feedback") != [
        "compile",
        "correctness",
        "qualified_timing",
        *profile_feedback,
    ]:
        raise ValueError("Study Contract Authoring Environment feedback differs")
    gate, compiler_relative, compiler_reference = (
        _resolve_compiler_reference(
            project_root,
            open_cake.get("compiler_revision"),
            "study.arms.open_cake.compiler_revision",
            template=study.state == "template",
        )
    )

    if paired_triton:
        case_id = str(_object(study.document["evaluation_protocol"], "evaluation_protocol")["case_id"])
        baseline = bind_baseline(skeleton_document, workload, case_id)
        baseline_compiler = Compiler.load(project_root, project_root / compiler_relative)
        assessment = baseline_compiler.assess(baseline)
        if not assessment.lowering_eligible:
            raise ValueError("paired optimization baseline is not lowerable")
        baseline_lowering = baseline_compiler.lower(assessment)
        native_baseline(baseline_lowering)

    allocation = _object(study.document.get("allocation"), "study.allocation")
    order = allocation.get("order")
    if allocation.get("method") != "predeclared_balanced_blocks" or not isinstance(order, list):
        raise ValueError("Study Contract allocation differs")
    run_order = tuple(_name(value, "study.allocation.order[]") for value in order)
    expected_arms = (
        sorted([comparison, "open_cake"])
        if claim_scope in _ONE_RUN_PER_ARM_SCOPES
        else sorted([comparison] * 3 + ["open_cake"] * 3)
    )
    if len(run_order) != len(set(run_order)) or sorted(
        name.rsplit("-", 1)[0] for name in run_order
    ) != expected_arms:
        required = "one" if claim_scope in _ONE_RUN_PER_ARM_SCOPES else "three"
        raise ValueError(
            f"Study Contract must predeclare {required} independent Run(s) per arm"
        )

    budget = _object(study.document.get("budget"), "study.budget")
    checkpoints = budget.get("checkpoints")
    limit = budget.get("limit")
    maximum_turns = budget.get("maximum_turns")
    maximum_candidates_per_turn = budget.get("maximum_candidates_per_turn", 1)
    budget_fields = {"unit", "limit", "checkpoints", "maximum_turns"}
    budget_fields.add("maximum_candidates_per_turn")
    budget_fields.update(
        {
            "wall_time_seconds",
            "active_authoring_time_seconds",
            "evaluation_limits",
        }
    )
    if (
        set(budget) != budget_fields
        or
        budget.get("unit") != "provider_tokens"
        or not isinstance(limit, int)
        or isinstance(limit, bool)
        or limit <= 0
        or not isinstance(checkpoints, list)
        or not checkpoints
        or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in checkpoints)
        or checkpoints != sorted(set(checkpoints))
        or checkpoints[-1] != limit
        or not isinstance(maximum_turns, int)
        or isinstance(maximum_turns, bool)
        or maximum_turns <= 0
        or not isinstance(maximum_candidates_per_turn, int)
        or isinstance(maximum_candidates_per_turn, bool)
        or maximum_candidates_per_turn <= 0
    ):
        raise ValueError("Study Contract budget grid differs")
    RalphBudget.from_mapping(budget)
    run_protocol = _object(study.document.get("run_protocol"), "study.run_protocol")
    expected_workspace = (
        run_protocol.get("workspace_seed") == "task_agents_only"
    )
    if (
        run_protocol.get("independent_thread") is not True
        or not expected_workspace
        or run_protocol.get("automatic_retries") != 0
        or run_protocol.get("replacement_runs") != 0
    ):
        raise ValueError("Study Contract Run Protocol differs")
    evaluation = _object(
        study.document.get("evaluation_protocol"), "study.evaluation_protocol"
    )
    workload.case(_name(evaluation.get("case_id"), "study.evaluation_protocol.case_id"))
    assay = paired_protocol(evaluation)
    if assay is not None and not paired_triton:
        raise ValueError('fixed-baseline assay requires the paired Triton Study')
    # How many candidates a Turn search-evaluates. Checked here because a Study that
    # asks for none, or for a word, would otherwise fault partway through a run --
    # and a run that faults has already spent the GPU time this Lab exists to gate.
    searches = evaluation.get("searches_per_turn", 1)
    if not isinstance(searches, int) or isinstance(searches, bool) or searches < 1:
        raise ValueError("Study Contract searches_per_turn differs")
    if searches > maximum_candidates_per_turn:
        raise ValueError(
            "Study Contract searches_per_turn exceeds maximum_candidates_per_turn"
        )
    ralph_limits = _object(
        budget.get("evaluation_limits"), "study.budget.evaluation_limits"
    )
    required_attribution = searches if attribution_evaluation == _ATTRIBUTION_EVALUATION else 0
    if (
        int(ralph_limits.get("search", 0)) < searches
        or int(ralph_limits.get("confirmatory", 0)) < 1
        or int(ralph_limits.get("attribution", 0)) < required_attribution
    ):
        raise ValueError("Ralph budget cannot admit one complete Turn")
    # How much faster the measurement has to be before the order counts as wrong.
    # A Study that searches more than one candidate has to say, because without it
    # every inversion inside the noise would be routed to the cost model as a defect
    # -- and the loss surface is a plateau, so most inversions are inside the noise
    # (`docs/ANALYSIS_CALIBRATION.md`).
    materiality = evaluation.get("search_materiality_ratio")
    if searches > 1:
        if (
            not isinstance(materiality, float)
            or not 1.0 < materiality < 100.0
        ):
            raise ValueError(
                "a Study searching more than one candidate declares "
                "search_materiality_ratio"
            )
    elif materiality is not None:
        # No second candidate to compare against, so a ratio here would state a
        # threshold nothing can cross.
        raise ValueError("search_materiality_ratio without searches_per_turn above one")
    execution = _object(study.document.get("execution"), "study.execution")
    expected_execution_fields = {'target', 'executor_revision', 'broker_execution_sha256', 'gpu', 'sandbox'}
    if assay is not None:
        expected_execution_fields.update({'fixed_baseline', 'runtime_config'})
    if set(execution) != expected_execution_fields:
        raise ValueError("Study Contract execution fields differ")
    if assay is not None:
        from open_cake_ir.evaluation.paired import candidate_identity, validate_pair_candidates
        fixed = _object(execution['fixed_baseline'], 'execution.fixed_baseline')
        sealed_baseline = load_baseline_bundle(project_root, fixed['bundle_path'])
        validate_pair_candidates(sealed_baseline, sealed_baseline, workload, str(evaluation['case_id']))
        import ast
        from open_cake_ir.compiler.toolchain import project_triton_kernel
        requirements = baseline_lowering.toolchain_requirements
        source = sealed_baseline.artifact_payloads.get('lowered_source')
        expected_source = project_triton_kernel(baseline_lowering.source.encode(), requirements)
        if source is None:
            raise ValueError('fixed baseline requires retained Compiler lowering source')
        observed_source = project_triton_kernel(source, requirements)
        manifest = manifest_parser(json.loads(sealed_baseline.artifact_payloads['launch_manifest']))
        if (fixed['candidate'] != candidate_identity(sealed_baseline)
            or ast.dump(ast.parse(observed_source)) != ast.dump(ast.parse(expected_source))
            or list(manifest.grid) != requirements['grid']
            or manifest.block != (requirements['compile_options']['num_warps'] * 32, 1, 1)):
            raise ValueError('fixed baseline differs from the frozen Compiler kernel or launch commitments')
    if (
        execution.get("target") != workload.target
        or execution.get("target") != skeleton_document.get("target")
        or execution.get("sandbox") != "workspace-write"
    ):
        raise ValueError("Study Contract execution authority differs")
    _digest(
        execution.get("broker_execution_sha256"),
        "study.execution.broker_execution_sha256",
    )
    # External binding has already applied the original template grammar and
    # verified this Executor. Preserve that resolution and the Study identity.
    executor = (
        bound_executor
        if bound_executor is not None
        else resolve_executor(
            project_root,
            execution.get("executor_revision"),
            "study.execution",
            template=study.state == "template",
        )
    )
    executor_reference = dict(executor.reference)
    gpu = _object(execution.get("gpu"), "study.execution.gpu")
    target = cuda_target(execution['target'])
    if (set(gpu) != {'name', 'count', 'mode'} or gpu.get('name') not in target.device_names
        or type(gpu.get('count')) is not int or gpu['count'] != 1 or gpu.get('mode') != 'exclusive'):
        raise ValueError("Study Contract GPU admission differs")
    analysis = _object(study.document.get("analysis_plan"), "study.analysis_plan")
    if claim_scope == "system_qualification_only":
        if analysis != _SYSTEM_QUALIFICATION_ANALYSIS_PLAN:
            raise ValueError("system qualification Analysis Plan differs")
        estimand = None
    elif claim_scope == "artifact_optimization_only":
        if analysis != _ARTIFACT_OPTIMIZATION_ANALYSIS_PLAN:
            raise ValueError("artifact optimization Analysis Plan differs")
        estimand = None
    else:
        version = _scientific_analysis_plan_version(analysis, "study.analysis_plan")
        if paired_triton != (version == "triton_optimization_v1"):
            raise ValueError("scientific treatment and analysis arm assignment differ")
        estimand = _name(analysis.get("estimand"), "study.analysis_plan.estimand")
    evidence_policy = _object(study.document.get("evidence"), "study.evidence")
    evidence_version = _matched_evidence_policy_version(
        evidence_policy, "study.evidence"
    )
    if evidence_version != _MATCHED_RALPH_EVENT_VOCABULARY_V1:
        raise ValueError("Study agent interface and Evidence policy differ")

    resolved_arms = cast(
        dict[str, object], json.loads(_canonical_json_bytes(arms))
    )
    _object(
        resolved_arms["open_cake"], "resolved open_cake arm"
    )["compiler_revision"] = compiler_reference
    if has_empirical_policy:
        model_path = Path(empirical_cost_model_path).resolve(strict=True)
        if project_root in model_path.parents:
            raise ValueError("external empirical model must stay outside the project checkout")
        model_document = json.loads(model_path.read_text(encoding="utf-8"))
        EmpiricalCostModel(model_document)
        resolved_arms["open_cake"]["candidate_selection"] = {
            "kind": _EMPIRICAL_SELECTION, "model": model_document,
        }
    resolved_execution = cast(
        dict[str, object], json.loads(_canonical_json_bytes(execution))
    )
    resolved_execution["executor_revision"] = executor_reference
    arm_digests = {
        name: sha256(_canonical_json_bytes(value)).hexdigest()
        for name, value in resolved_arms.items()
    }
    lock_document: dict[str, object] = {
        "schema_version": 1,
        "study": {
            "study_id": study.study_id,
            "kind": study.document["kind"],
            "claim_scope": claim_scope,
            "canonical_sha256": study.canonical_sha256,
        },
        "workload": {
            "workload_id": workload.workload_id,
            "path": workload_relative,
            "canonical_sha256": workload.canonical_sha256,
        },
        "compiler_revision": {
            "revision_id": gate.compiler_revision_id,
            "path": compiler_relative,
            "canonical_sha256": gate.compiler_revision_sha256,
        },
        "resolved_inputs": {
            "arm_environments": resolved_arms,
            "arm_environment_sha256": arm_digests,
            "budget": budget,
            "run_protocol": run_protocol,
            "evidence_policy": evidence_policy,
            **(
                {"agent_interface": study.document["agent_interface"]}
            ),
        },
        "run_order": list(run_order),
        "evaluation_protocol": evaluation,
        "execution": resolved_execution,
        "analysis_plan": analysis,
        "analysis_plan_sha256": sha256(_canonical_json_bytes(analysis)).hexdigest(),
    }
    lock = CampaignLock.from_dict(lock_document)
    if (
        lock.study_id != study.study_id
        or lock.workload_id != workload.workload_id
        or lock.compiler_revision_id != gate.compiler_revision_id
        or lock.claim_scope != claim_scope
        or lock.estimand != estimand
    ):
        raise ValueError("resolved Campaign Lock projection differs")
    return lock
