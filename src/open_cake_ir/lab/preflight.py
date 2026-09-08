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

from ._documents import _canonical_json_bytes, _digest, _name, _object, _project_path
from ._policies import (
    _ARTIFACT_OPTIMIZATION_ANALYSIS_PLAN,
    _ATTRIBUTION_EVALUATION,
    _LEGACY_ATTRIBUTION_EVALUATION,
    _MATCHED_RALPH_EVENT_VOCABULARY_V1,
    _SYSTEM_QUALIFICATION_ANALYSIS_PLAN,
    _matched_evidence_policy_version,
    _scientific_analysis_plan_version,
)
from .admission import validate_provider, validate_run_plan, validate_evaluation
from .bindings import _resolve_compiler_reference, resolve_execution_bindings, resolve_executor
from .contracts import CampaignLock, StudyContract
from .pairing import native_backend, bind_baseline, comparison_arm, native_baseline
from .selection import _EMPIRICAL_SELECTION
from .task_package import TaskPackage, render_task_package

from .bindings import source_reference_path
from .python_reference import read_skeleton
from .pairing import matched_run_arms


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
    prepare_schedule: Callable,
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
    workload_relative, workload_path = source_reference_path(
        project_root, workload_ref.get("path"), "study.workload.path"
    )
    workload = workload_loader(workload_path)
    if workload.canonical_sha256 != _digest(
        workload_ref.get("canonical_sha256"), "study.workload.canonical_sha256"
    ):
        raise ValueError("Study Contract workload bytes differ")

    arms = _object(study.document.get("arms"), "study.arms")
    comparison = comparison_arm(arms)
    policy = native_backend(comparison)
    single_environment = comparison is None
    matched_run_arms(arms, study.document["claim_scope"])
    open_cake = _object(arms.get("open_cake"), "study.arms.open_cake")
    direct_cuda = _object(arms[comparison], f"study.arms.{comparison}") if comparison is not None else {}
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
    if single_environment:
        direct_cuda_fields = set()
        open_cake_fields.update({"input_format", "toolchain_sha256"})
        if open_cake.get("input_format") != "schedule_or_python_v1":
            raise ValueError("single-environment optimization requires the Python-enabled authoring contract")
        _digest(open_cake.get("toolchain_sha256"), "study.arms.open_cake.toolchain_sha256")
    if policy is not None:
        open_cake_fields.update({"input_format", "toolchain_sha256"})
        direct_cuda_fields -= {"launch_contract", "candidate_skeleton"}
        direct_cuda_fields.add("baseline")
        if (open_cake.get("input_format") != "schedule_or_python_v1"
            or direct_cuda.get("baseline") != {"binding": "open_cake_lowering"}
            or open_cake.get("toolchain_sha256") != direct_cuda.get("toolchain_sha256")):
            raise ValueError("same-backend native input, baseline or common toolchain binding differs")
    if set(open_cake) != open_cake_fields or set(direct_cuda) != direct_cuda_fields:
        raise ValueError("Study Contract Authoring Environment fields differ")
    if open_cake.get("environment_kind") != "open_cake" or direct_cuda.get(
        "environment_kind"
    ) != comparison:
        raise ValueError("Study Contract Authoring Environment kinds differ")
    route = open_cake.get("lowering_route")
    if (not isinstance(route, Mapping) or set(route) != {"backend", "entry_point"}
        or route.get("backend") not in ({"metal", "triton"} if single_environment else {policy.backend if policy is not None else "triton"}) or not isinstance(route.get("entry_point"), str)
        or not route["entry_point"].isidentifier()):
        raise ValueError("Study Contract Open Cake lowering route differs")
    schedule_skeleton = _object(
        open_cake.get("schedule_skeleton"), "study.arms.open_cake.schedule_skeleton"
    )
    if set(schedule_skeleton) != {"path", "canonical_sha256"}:
        raise ValueError("Study Contract Schedule skeleton reference differs")
    _, schedule_skeleton_path = source_reference_path(
        project_root,
        schedule_skeleton.get("path"),
        "study.arms.open_cake.schedule_skeleton.path",
    )
    skeleton_document = _object(
        read_skeleton(schedule_skeleton_path),
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
    if comparison is not None:
        if (open_cake.get("provider") != direct_cuda.get("provider")
                or open_cake.get("scaffold") != direct_cuda.get("scaffold")):
            raise ValueError("matched Authoring Environments differ in provider or scaffold")
        _digest(direct_cuda.get("toolchain_sha256"), "study.arms.direct_cuda.toolchain_sha256")
    claim_scope = validate_provider(
        open_cake=open_cake,
        policy=policy,
        project_root=project_root,
        study=study,
    )
    scaffold = _object(open_cake.get("scaffold"), "study.arms.scaffold")
    if set(scaffold) != {"path", "sha256"}:
        raise ValueError("Study Contract scaffold reference differs")
    _, scaffold_path = source_reference_path(
        project_root, scaffold.get("path"), "study.arms.scaffold.path"
    )
    if _digest(scaffold.get("sha256"), "study.arms.scaffold.sha256") != sha256(
        scaffold_path.read_bytes()
    ).hexdigest():
        raise ValueError("Study Contract scaffold bytes differ")
    if comparison == "direct_cuda":
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
    if (open_cake.get("tool_surface") != (["submit_schedule_or_python"] if policy is not None or single_environment else ["submit_schedule"])
            or (comparison is not None and direct_cuda.get("tool_surface") != ([policy.submit_tool] if policy is not None else ["submit_cuda"]))):
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
    ] or (comparison is not None and direct_cuda.get("feedback") != [
        "compile", "correctness", "qualified_timing", *profile_feedback,
    ]):
        raise ValueError("Study Contract Authoring Environment feedback differs")
    gate, compiler_relative, compiler_reference = (
        _resolve_compiler_reference(
            project_root,
            open_cake.get("compiler_revision"),
            "study.arms.open_cake.compiler_revision",
            template=study.state == "template",
        )
    )

    if policy is not None or single_environment:
        case_id = str(_object(study.document["evaluation_protocol"], "evaluation_protocol")["case_id"])
        baseline = prepare_schedule(skeleton_document, workload, case_id, open_cake)
        baseline_compiler = Compiler.load(project_root, project_root / compiler_relative)
        assessment = baseline_compiler.assess(baseline)
        if not assessment.lowering_eligible:
            raise ValueError("paired optimization baseline is not lowerable")
        baseline_lowering = baseline_compiler.lower(assessment)
        if policy is not None:
            native_baseline(baseline_lowering)

    budget, maximum_candidates_per_turn, run_order, run_protocol = validate_run_plan(
        claim_scope=claim_scope,
        comparison=comparison,
        study=study,
    )
    evaluation, execution = validate_evaluation(
        attribution_evaluation=attribution_evaluation,
        baseline_lowering=baseline_lowering if policy is not None or single_environment else None,
        budget=budget,
        manifest_parser=manifest_parser,
        maximum_candidates_per_turn=maximum_candidates_per_turn,
        policy=policy,
        project_root=project_root,
        skeleton_document=skeleton_document,
        study=study,
        workload=workload,
    )
    # External binding already validated the template and this Executor.
    # Preserve that resolution and the Study identity.
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
    from open_cake_ir.compiler.target import Target
    revision_document = json.loads((project_root / compiler_relative).read_bytes())
    target_reference = revision_document["target_definitions"].get(execution["target"])
    if target_reference is None:
        raise ValueError("Study target is not bound by the Compiler Revision")
    _, target_path = _project_path(project_root, target_reference["path"], "compiler.target")
    target = Target.load(target_path)
    if (set(gpu) != {'name', 'count', 'mode'} or gpu.get('name') not in target.device_names
        or type(gpu.get('count')) is not int or gpu['count'] != 1 or gpu.get('mode') != ('local_serialized' if route['backend'] == 'metal' else 'exclusive')):
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
        if version != (policy.analysis_version if policy is not None else "two_part_v2"):
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
