"""Bounded QSA diagnostics projected into the existing provider Turn seam.

Node-owned logs and receipts remain evidence authority. These projections copy only
facts that can change an author's next action; an unavailable observer or infrastructure
failure is never turned into a candidate diagnosis.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, cast

from open_cake_ir.compiler import Assessment

from .core import TurnRequest
from .routing import CANDIDATE, VERIFIER, route_rejection

_RUN_SCHEMA = "kernelinfra.run-result.v1"
_ARMS = {"open_cake", "direct_cuda"}


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _freeze(document: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(cast(dict[str, object], _plain(document)))


def _bounded_empirical_cost(cost: object) -> Mapping[str, object]:
    """Keep the decision and identity; supplier payloads stay in the source report."""

    if not isinstance(cost, Mapping):
        raise ValueError("QSA empirical cost projection differs")
    return {
        key: cost[key]
        for key in (
            "kind", "model_id", "model_compiler_revision_id",
            "model_compiler_revision_sha256", "target", "covered",
            "predicted_kernel_us", "empirical_range_us", "reason",
        )
        if key in cost
    }


def qsa_compiler_feedback(
    assessment: Assessment,
    *,
    static_profile: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Project one Compiler Assessment without collapsing localized findings."""

    findings = [
        {
            "code": finding.code,
            "path": finding.path,
            "message": finding.message,
            "blocks_acceptance": finding.blocks_acceptance,
            "blocks_lowering": finding.blocks_lowering,
        }
        for finding in assessment.findings
    ]
    profile = None if static_profile is None else dict(static_profile)
    if profile is not None and "empirical_cost" in profile:
        profile["empirical_cost"] = _bounded_empirical_cost(profile["empirical_cost"])
    feedback: dict[str, object] = {
        "schema_version": 1,
        "kind": "compiler",
        "stage": "assessment",
        "accepted": assessment.accepted,
        "lowering_eligible": assessment.lowering_eligible,
        "findings": findings,
        "static_profile": profile,
        "actionable": not assessment.lowering_eligible,
    }
    if not assessment.lowering_eligible:
        route = route_rejection({"stage": "assessment", "findings": findings})
        feedback["routed_to"] = route.destination
        feedback["routing_reason"] = route.reason
    return _freeze(feedback)


def _stage_rows(document: Mapping[str, object]) -> list[Mapping[str, object]]:
    stages = document.get("stages")
    if not isinstance(stages, list) or any(not isinstance(item, Mapping) for item in stages):
        raise ValueError("QSA run stage summaries differ")
    return cast(list[Mapping[str, object]], stages)


def _failed_stage(stages: list[Mapping[str, object]]) -> Mapping[str, object] | None:
    return next((stage for stage in stages if stage.get("status") == "failed"), None)


def _bounded_workload(document: Mapping[str, object]) -> Mapping[str, object] | None:
    workloads = document.get("workloads")
    if not isinstance(workloads, list) or len(workloads) > 1:
        raise ValueError("QSA run workload rows differ")
    if not workloads:
        return None
    row = workloads[0]
    if not isinstance(row, Mapping) or row.get("id") != "qsa-prefill-t32768":
        raise ValueError("QSA run workload identity differs")
    return {
        key: row[key]
        for key in (
            "id",
            "correct",
            "candidate_ms",
            "baseline_ms",
            "speedup",
            "stable",
            "notes",
        )
        if key in row
    }


def _bounded_profile(profile: Mapping[str, object]) -> Mapping[str, object]:
    work = profile.get("work")
    residency = profile.get("residency")
    lowering = profile.get("lowering")
    metrics = profile.get("ncu_metrics")
    if (
        work is not None
        and not isinstance(work, Mapping)
        or residency is not None
        and not isinstance(residency, Mapping)
        or not isinstance(lowering, Mapping)
        or not isinstance(metrics, list)
    ):
        raise ValueError("QSA static profile projection differs")
    retained_metrics = []
    abstained_metrics = []
    for metric in metrics:
        if not isinstance(metric, Mapping) or not isinstance(metric.get("metric"), str):
            raise ValueError("QSA static profile metric differs")
        if metric.get("estimate_kind") == "unknown":
            abstained_metrics.append(dict(metric))
        else:
            retained_metrics.append(dict(metric))
    return {
        "work": None if work is None else dict(work),
        "compiled_resources": profile.get("compiled_resources"),
        "residency": None if residency is None else {
            key: residency.get(key)
            for key in (
                "ctas_per_sm_upper_bound",
                "binding_resource",
                "coverage",
                "logical_register_pressure_per_thread",
                "bounds",
            )
        },
        "lowering": {
            key: lowering.get(key)
            for key in (
                "generated_source_bytes",
                "top_k",
                "explicit_barrier_count",
                "tile_loop_count",
                "runtime_indexed_buffers",
            )
        },
        "ncu_estimates": retained_metrics,
        "ncu_abstentions": abstained_metrics,
        "abstentions": profile.get("abstentions", []),
        **({"findings": profile["findings"]} if "findings" in profile else {}),
        **({"empirical_cost": _bounded_empirical_cost(profile["empirical_cost"])}
           if "empirical_cost" in profile else {}),
    }


def _bounded_compiler_feedback(compiler: Mapping[str, object]) -> Mapping[str, object]:
    if compiler.get("kind") != "open_cake_program_static_profile":
        feedback = dict(compiler)
        profile = compiler.get("static_profile")
        if isinstance(profile, Mapping) and "empirical_cost" in profile:
            feedback["static_profile"] = {
                **profile, "empirical_cost": _bounded_empirical_cost(profile["empirical_cost"]),
            }
        return feedback
    nodes = compiler.get("nodes")
    if not isinstance(nodes, Mapping):
        raise ValueError("QSA Program static profile nodes differ")
    if any(not isinstance(profile, Mapping) for profile in nodes.values()):
        raise ValueError("QSA Program static profile node differs")
    return {
        "kind": compiler["kind"],
        "nodes": {
            str(name): _bounded_profile(profile)
            for name, profile in nodes.items()
        },
    }


def qsa_evaluation_feedback(
    result: Mapping[str, object], *, arm: str
) -> Mapping[str, object]:
    """Turn one terminal GPU Infra result into bounded same-thread feedback."""

    if arm not in _ARMS:
        raise ValueError("QSA feedback arm differs")
    if (
        result.get("schema") != _RUN_SCHEMA
        or not str(result.get("task_id", "")).startswith("open-cake-qsa-prefill-t32768-")
        or result.get("outcome") not in {"completed", "rejected", "infra_error"}
        or result.get("validity") not in {"valid", "invalid", "unknown"}
    ):
        raise ValueError("QSA GPU Infra result fields differ")
    stages = _stage_rows(result)
    failed = _failed_stage(stages)
    stage = str(failed.get("id")) if failed is not None else "evaluation"
    summary = str(failed.get("summary", "")) if failed is not None else ""
    validity = str(result["validity"])
    outcome = str(result["outcome"])

    if validity == "unknown" or outcome == "infra_error":
        return _freeze(
            {
                "schema_version": 1,
                "kind": "infrastructure_fault",
                "stage": stage,
                "arm": arm,
                "actionable": False,
                "terminal_reason": result.get("terminal_reason"),
                "summary": summary,
                "instruction": "retain the candidate unchanged; repair or re-observe infrastructure",
            }
        )

    metrics = result.get("metrics", {})
    if not isinstance(metrics, Mapping):
        raise ValueError("QSA result metrics differ")
    correctness = metrics.get("correctness")
    compiler = metrics.get("compile")
    profile = metrics.get("profile")
    if correctness is not None and not isinstance(correctness, Mapping):
        raise ValueError("QSA correctness feedback differs")
    if profile is not None and not isinstance(profile, Mapping):
        raise ValueError("QSA profile feedback differs")
    if compiler is not None and not isinstance(compiler, Mapping):
        raise ValueError("QSA compiler feedback differs")
    workload = _bounded_workload(result)

    feedback: dict[str, object] = {
        "schema_version": 1,
        "kind": "evaluation",
        "stage": stage,
        "arm": arm,
        "actionable": True,
        "outcome": outcome,
        "validity": validity,
        "compiler": (
            _bounded_compiler_feedback(compiler) if compiler is not None else None
        ),
        "correctness": dict(correctness) if correctness is not None else None,
        "timing": dict(workload) if workload is not None else None,
        "profile": dict(profile) if profile is not None else None,
    }
    if outcome == "rejected":
        declared_route = compiler.get("routed_to") if isinstance(compiler, Mapping) else None
        destination = (
            str(declared_route)
            if declared_route in {CANDIDATE, VERIFIER, "ir_vocabulary"}
            else (VERIFIER if stage == "compile" else CANDIDATE)
        )
        feedback["routed_to"] = destination
        feedback["routing_reason"] = (
            "Compiler gates admitted the program but the toolchain rejected it"
            if destination == VERIFIER
            else f"the candidate failed the common {stage} gate"
        )
        feedback["summary"] = summary
    return _freeze(feedback)


def qsa_next_turn_request(
    *,
    run_id: str,
    arm: str,
    turn: int,
    cumulative_provider_tokens: int,
    thread_id: str | None,
    maximum_candidates_per_turn: int,
    result: Mapping[str, object],
) -> TurnRequest:
    """Connect a terminal QSA result to the canonical same-thread provider request."""

    if turn <= 1:
        raise ValueError("QSA feedback can resume only a later provider Turn")
    return TurnRequest(
        run_id,
        arm,
        turn,
        cumulative_provider_tokens,
        thread_id,
        qsa_evaluation_feedback(result, arm=arm),
        maximum_candidates_per_turn,
    )
