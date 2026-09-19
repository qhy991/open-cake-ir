"""Study binding and authoring preparation without execution side effects."""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Callable, cast

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.performance.empirical_cost import EmpiricalCostModel
from open_cake_ir.compiler.target import Target

from ._documents import _canonical_json_bytes, _digest, _object, _project_path, differs
from .admission import validate_provider, validate_evaluation
from .bindings import _resolve_compiler_reference, resolve_execution_bindings, resolve_executor, source_reference_path
from .contracts import CampaignLock, StudyContract
from .pairing import native_backend, native_baseline
from .selection import _EMPIRICAL_SELECTION
from .task_package import TaskPackage, render_task_package

from .python_reference import read_skeleton
from .reference_access import validate_reference_handoff


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
    if study.kind == "portfolio":
        if empirical_cost_model_path is not None or execution_bindings_path is not None:
            raise ValueError("empirical selection requires artifact_optimization_only matched search")
        raise ValueError("portfolio preflight belongs to its task entrypoint")
    resolved_document, bound_executor = resolve_execution_bindings(
        project_root, study, execution_bindings_path
    )
    # The shape was decided by `StudyContract.load`; binding filled marker leaves only.
    study = replace(study, document=resolved_document)
    workload_ref = _object(study.document.get("workload"), "study.workload")
    if set(workload_ref) != {"path", "canonical_sha256"}:
        raise differs(
            "study workload reference fields differ",
            expected=["canonical_sha256", "path"], observed=sorted(workload_ref),
        )
    workload_relative, workload_path = source_reference_path(
        project_root, workload_ref.get("path"), "study.workload.path"
    )
    workload = workload_loader(workload_path)
    expected_workload_sha256 = _digest(
        workload_ref.get("canonical_sha256"), "study.workload.canonical_sha256"
    )
    if workload.canonical_sha256 != expected_workload_sha256:
        raise differs(
            "Study Contract workload bytes differ",
            expected=expected_workload_sha256, observed=workload.canonical_sha256,
        )

    arms = study.arms
    comparison = study.comparison
    policy = native_backend(comparison)
    single_environment = comparison is None
    open_cake = arms["open_cake"]
    direct_cuda = arms[comparison] if comparison is not None else {}
    validate_reference_handoff(project_root, arms)
    validate_authoring(workload, arms, empirical_cost_model_path=empirical_cost_model_path)
    has_empirical_policy = "candidate_selection" in open_cake
    # The document's half of this rule is `StudyContract.load`'s; the model path is an
    # input of this call, and the two must be declared together.
    if has_empirical_policy != (empirical_cost_model_path is not None):
        raise ValueError("empirical selection requires artifact_optimization_only candidate-set policy and an explicit model")
    if single_environment:
        _digest(open_cake.get("toolchain_sha256"), "study.arms.open_cake.toolchain_sha256")
    schedule_skeleton = _object(
        open_cake.get("schedule_skeleton"), "study.arms.open_cake.schedule_skeleton"
    )
    _, schedule_skeleton_path = source_reference_path(
        project_root,
        schedule_skeleton.get("path"),
        "study.arms.open_cake.schedule_skeleton.path",
    )
    skeleton_document = _object(
        read_skeleton(schedule_skeleton_path),
        "study.arms.open_cake.schedule_skeleton",
    )
    expected_skeleton_sha256 = _digest(
        schedule_skeleton.get("canonical_sha256"),
        "study.arms.open_cake.schedule_skeleton.canonical_sha256",
    )
    observed_skeleton_sha256 = sha256(_canonical_json_bytes(skeleton_document)).hexdigest()
    if (
        skeleton_document.get("lowering") != open_cake.get("lowering_route")
        or expected_skeleton_sha256 != observed_skeleton_sha256
    ):
        raise differs(
            "Study Contract Schedule skeleton bytes or lowering route differ",
            expected={"canonical_sha256": expected_skeleton_sha256, "lowering": open_cake.get("lowering_route")},
            observed={"canonical_sha256": observed_skeleton_sha256, "lowering": skeleton_document.get("lowering")},
        )
    if comparison is not None:
        _digest(direct_cuda.get("toolchain_sha256"), "study.arms.direct_cuda.toolchain_sha256")
    claim_scope = validate_provider(
        open_cake=open_cake,
        policy=policy,
        project_root=project_root,
        study=study,
    )
    for owner, field, noun, path_reader in (
        ("open_cake", "scaffold", "scaffold", source_reference_path),
        *(
            (("direct_cuda", "launch_contract", "direct launch contract", _project_path),
             ("direct_cuda", "candidate_skeleton", "direct candidate skeleton", source_reference_path))
            if comparison == "direct_cuda" else ()
        ),
    ):
        reference = _object(arms[owner].get(field), f"study.arms.{owner}.{field}")
        _, path = path_reader(project_root, reference.get("path"), f"study.arms.{owner}.{field}.path")
        expected_sha256 = _digest(reference.get("sha256"), f"study.arms.{owner}.{field}.sha256")
        observed_sha256 = sha256(path.read_bytes()).hexdigest()
        if expected_sha256 != observed_sha256:
            raise differs(
                f"Study Contract {noun} bytes differ", expected=expected_sha256, observed=observed_sha256,
            )
    evaluation_protocol = study.evaluation_protocol
    attribution_evaluation = evaluation_protocol.get("attribution_evaluation")
    gate, compiler_relative, compiler_reference = (
        _resolve_compiler_reference(
            project_root,
            open_cake.get("compiler_revision"),
            "study.arms.open_cake.compiler_revision",
            template=study.state == "template",
        )
    )

    if policy is not None or single_environment:
        case_id = str(evaluation_protocol["case_id"])
        baseline = prepare_schedule(skeleton_document, workload, case_id, open_cake)
        baseline_compiler = Compiler.load(project_root, project_root / compiler_relative)
        assessment = baseline_compiler.assess(baseline)
        if not assessment.lowering_eligible:
            raise ValueError("paired optimization baseline is not lowerable")
        baseline_lowering = baseline_compiler.lower(assessment)
        if policy is not None:
            native_baseline(baseline_lowering)

    evaluation, execution = validate_evaluation(
        attribution_evaluation=attribution_evaluation,
        baseline_lowering=baseline_lowering if policy is not None or single_environment else None,
        manifest_parser=manifest_parser,
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
            target=execution.get("target"),
        )
    )
    executor_reference = dict(executor.reference)
    gpu = _object(execution.get("gpu"), "study.execution.gpu")
    target_relative = f"compiler/targets/{execution['target']}.json"
    if not (project_root / target_relative).is_file():
        raise ValueError("Study target is not declared by the Compiler")
    _, target_path = _project_path(project_root, target_relative, "compiler.target")
    target = Target.load(target_path)
    # The mode is checked against its closed vocabulary here and against the device that
    # owns it in the task layer. It used to be read as `local_serialized if metal else
    # exclusive`, which infers how a run reaches its device from the route it lowers
    # through -- the two axes `tasks.devices` keeps as separate columns precisely because
    # "a DCU lowers through Triton like a B200 and is reached like an Apple device", and
    # inferring one from the other is what refused every DCU launch once already. `lab`
    # cannot read that registry (tests/contracts/test_task_boundaries.py), so
    # `TaskLab.preflight` checks which of the two is right for this target.
    if (set(gpu) != {'name', 'count', 'mode'} or gpu.get('name') not in target.device_names
        or type(gpu.get('count')) is not int or gpu['count'] != 1
        or gpu.get('mode') not in {'local_serialized', 'exclusive'}):
        raise differs(
            "Study Contract GPU admission",
            expected={"name": sorted(target.device_names), "count": 1,
                      "mode": ["exclusive", "local_serialized"]},
            observed=dict(gpu),
        )
    analysis = study.analysis_plan
    evidence_policy = study.evidence_policy

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
        },
        "resolved_inputs": {
            "arm_environments": resolved_arms,
            "arm_environment_sha256": arm_digests,
            "budget": study.budget,
            "run_protocol": study.run_protocol,
            "evidence_policy": evidence_policy,
            **(
                {"agent_interface": study.document["agent_interface"]}
            ),
        },
        "run_order": list(study.run_order),
        "evaluation_protocol": evaluation,
        "execution": resolved_execution,
        "analysis_plan": analysis,
        "analysis_plan_sha256": sha256(_canonical_json_bytes(analysis)).hexdigest(),
    }
    # `CampaignLock.from_dict` is the lock's one validator; the projection it returns is
    # read from the document written above, so re-comparing it here only restated it.
    return CampaignLock.from_dict(lock_document)
