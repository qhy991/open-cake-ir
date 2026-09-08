"""Study admission checks grouped by their external authority boundaries."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import cast

from open_cake_ir.evaluation.paired import paired_protocol

from ._documents import _canonical_json_bytes, _digest, _name, _object, _project_path
from ._policies import _ATTRIBUTION_EVALUATION, _ONE_RUN_PER_ARM_SCOPES
from .bindings import load_baseline_bundle, qualification_path as _qualification_path
from .providers import (
    CANDIDATE_SET_ENVELOPE_V1,
    CODEX_DISABLED_FEATURES,
    ProviderQualificationReceipt,
    required_live_provider_qualification_scope,
)
from .ralph import RalphBudget


def validate_provider(*, open_cake, paired_triton, project_root, study):
    """Validate provider qualification and declared authoring capabilities."""
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
    return claim_scope


def validate_run_plan(*, claim_scope, comparison, study):
    """Validate allocation order, finite budgets and run protocol."""
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
    return budget, maximum_candidates_per_turn, run_order, run_protocol


def validate_evaluation(
    *,
    attribution_evaluation,
    baseline_lowering,
    budget,
    manifest_parser,
    maximum_candidates_per_turn,
    paired_triton,
    project_root,
    skeleton_document,
    study,
    workload,
):
    """Validate the numerical assay and its execution admission contract."""
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
    return evaluation, execution
