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
from .pairing import native_source, native_block
from .ralph import RalphBudget

from .provider_policy import provider_configuration, provider_harness
from .bindings import source_reference_path
from .pairing import matched_run_arms
from open_cake_ir.evaluation.paired import PAIRED_METAL_KIND, validation_case_ids


def validate_provider(*, open_cake, policy, project_root, study):
    """Validate provider qualification and declared authoring capabilities."""
    provider = _object(open_cake.get("provider"), "study.arms.provider")
    provider_revision = _name(provider.get("revision"), "study.arms.provider.revision")
    claim_scope = cast(str, study.document["claim_scope"])
    expected_provider_configuration = provider_configuration(provider, claim_scope, arms=study.document["arms"])
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
        or qualification.configuration_sha256 != sha256(
            _canonical_json_bytes(expected_provider_configuration)).hexdigest()
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
    if (policy is not None and qualification.scope != 'zero_gpu_contract_fixture_only'
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
            raise ValueError(f"Study Contract provider {field} reference differs")
        _, path = source_reference_path(
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
    expected_arms = matched_run_arms(study.document["arms"], claim_scope)
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
    policy,
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
    single_environment = set(study.document["arms"]) == {"open_cake"}
    route = study.document["arms"]["open_cake"]["lowering_route"]
    if assay is not None and policy is None and not single_environment:
        raise ValueError('fixed-baseline assay requires the same-backend native Study')
    if single_environment and assay is None:
        raise ValueError("single-environment optimization requires an explicit fixed-baseline paired assay")
    if route["backend"] == "metal":
        if (not single_environment or evaluation.get("paired_timing", {}).get("kind") != PAIRED_METAL_KIND
                or validation_case_ids(evaluation) != tuple(workload.case_ids)
                or attribution_evaluation != _ATTRIBUTION_EVALUATION):
            raise ValueError("Metal optimization must bind its paired assay, all Workload cases and attribution")
    elif evaluation.get("paired_timing", {}).get("kind") == PAIRED_METAL_KIND:
        raise ValueError("Metal paired assay cannot evaluate a different backend")
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
            grid, block = requirements['grid'], tuple(native_block(requirements))
            if manifest.hidden_null_pointer_parameters != policy.hidden_null_pointer_parameters:
                raise ValueError('fixed baseline hidden pointer commitments differ')
        if (fixed['candidate'] != candidate_identity(sealed_baseline) or not source_matches
                or list(manifest.grid) != list(grid) or manifest.block != block):
            raise ValueError('fixed baseline differs from the frozen Compiler kernel or launch commitments')
    if (
        execution.get("target") != workload.target
        or execution.get("target") != skeleton_document.get("target")
        or execution.get("sandbox") != study.document["arms"]["open_cake"]["provider"].get("sandbox")
    ):
        raise ValueError("Study Contract execution authority differs")
    _digest(
        execution.get("broker_execution_sha256"),
        "study.execution.broker_execution_sha256",
    )
    return evaluation, execution
