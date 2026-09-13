"""Stable artifact-optimization policy; execution identities bind in CampaignLock."""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from open_cake_ir.compiler import frontend
from open_cake_ir.evaluation.paired import PAIRED_KIND, PAIRED_METAL_BATCHED_KIND, paired_protocol
from open_cake_ir.lab.bindings import CAMPAIGN_BINDING, CURRENT_RELEASE_BINDING
from open_cake_ir.lab.claude import CLAUDE_AUTHORING_TOOLS, CLAUDE_EVENT_CONTRACT, terminal_schema
from open_cake_ir.lab._policies import _ARTIFACT_OPTIMIZATION_ANALYSIS_PLAN
from open_cake_ir.lab.endpoints import NORMAL_BUDGET_TERMINAL
from open_cake_ir.lab.efficiency_policy import TASK_EFFICIENCY_V1
from open_cake_ir.lab.ralph import RalphBudget
# The portable registry, so a Study can name an NVIDIA device as readily as an
# Apple one; open_cake_ir.tasks.apple covers only the latter.
from open_cake_ir.tasks.devices import BACKENDS, backend_for_target, device_name

OUTPUT_SCHEMA = "contracts/providers/open-cake-optimization-output-schema-v1.json"
SCAFFOLD = "contracts/scaffolds/python-artifact-optimization-v2.md"


def canonical(document) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def evaluation_policy(workload, *, searches_per_turn: int = 2, dispatches_per_sample: int | None = None,
                      maximum_cv: float | None = 0.05, required_pair_wins: int | None = 6) -> dict:
    if type(searches_per_turn) is not int or searches_per_turn <= 0:
        raise ValueError("searches per Turn must be a positive integer")
    backend = backend_for_target(workload.target)
    if backend is None:
        raise ValueError("evaluation policy requires a supported exact target")
    metal = BACKENDS[backend]["route"] == "metal"
    if maximum_cv is None:
        maximum_cv = 0.05 if metal else 0.15
    if required_pair_wins is None:
        required_pair_wins = 6 if metal else 9
    if not metal and dispatches_per_sample is not None:
        raise ValueError("dispatches_per_sample is a Metal command-buffer control")
    dispatches = 64 if dispatches_per_sample is None else dispatches_per_sample
    if type(dispatches) is not int or not 1 <= dispatches <= 4096:
        raise ValueError("dispatches per timed command buffer must be 1..4096")
    timer = "metal" if metal else "cupti"
    policy = {
        "case_id": workload.document["validation"]["primary_case"],
        "validation_case_ids": list(workload.case_ids),
        "searches_per_turn": searches_per_turn,
        "search_evaluation": f"correctness_then_paired_{timer}",
        "confirmatory_evaluation": f"fresh_fixed_candidate_correctness_then_paired_{timer}",
        "attribution_evaluation": "correctness_then_profile_each_search_survivor",
        "paired_timing": {
            "kind": PAIRED_METAL_BATCHED_KIND if metal else PAIRED_KIND,
            "arms": ["candidate", "baseline"],
            "pair_order": [["candidate", "baseline"], ["baseline", "candidate"]] * 5,
            "samples_per_cohort": 25,
            "route_calls_per_cohort": _ROUTE_CALLS_PER_COHORT if metal else 6 + 11 + 25,
            "maximum_cv": maximum_cv, "materiality_ratio": 1.05, "required_pair_wins": required_pair_wins,
        },
    }
    if metal:
        policy["paired_timing"].update({
            # Fixed encode/submit/complete cost is paid once per command buffer. A
            # kernel shorter than that cost is otherwise measured mostly through it.
            "dispatches_per_sample": dispatches,
            # Same strictness as maximum_cv, on a statistic an isolated disturbed
            # sample cannot veto. This assay does not exclude other GPU clients.
            "maximum_relative_iqr": 0.05,
        })
    if searches_per_turn > 1:
        policy["search_materiality_ratio"] = 1.05
    paired_protocol(policy)
    return policy


# How many launches the native observer holds in one snapshot cohort. Named because the
# launcher checks this same number against the observer's payload bound before a campaign
# starts, and the two must not drift (F-2026-09-10-002).
_ROUTE_CALLS_PER_COHORT = 28


def study_template(root: Path, workload, workload_path: Path, starter_path: Path, *,
                   harness: str, model: str, effort: str, turns: int = 4,
                   token_budget: int = 150000, maximum_candidates: int = 3,
                   searches_per_turn: int = 2, wall_seconds: int = 14400,
                   dispatches_per_sample: int | None = None,
                   maximum_cv: float | None = 0.05, required_pair_wins: int | None = 6) -> dict:
    """Bind mathematical inputs and treatment while leaving runtime facts unresolved.

    The policy is operator-agnostic: every task validates through its own
    registered contract and shares this matched-search treatment.
    """
    from open_cake_ir.tasks.workloads import validate_workload_document
    validate_workload_document(workload.document)
    if harness not in {"codex", "claude-code"} or any(not isinstance(v, str) or not v.strip() or v != v.strip() for v in (model, effort)):
        raise ValueError("exact harness, model and effort are required")
    if type(searches_per_turn) is not int or type(maximum_candidates) is not int or not 1 <= searches_per_turn <= maximum_candidates:
        raise ValueError("searches per Turn must fit the candidate budget")
    budget = {"unit": "provider_tokens", "limit": token_budget, "checkpoints": [token_budget],
              "maximum_turns": turns, "maximum_candidates_per_turn": maximum_candidates,
              "wall_time_seconds": wall_seconds, "active_authoring_time_seconds": wall_seconds / 2,
              "evaluation_limits": {"search": turns * searches_per_turn, "confirmatory": turns,
                                    "attribution": turns * searches_per_turn}}
    RalphBudget.from_mapping(budget)
    source = frontend.read_schedule(starter_path)
    provider = {"model": model, "reasoning_effort": effort,
                "removed_environment": ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
                "cwd_policy": "independent_task_workspace", "reference_visibility": "workspace_task_files",
                **{name: dict(CAMPAIGN_BINDING) for name in ("revision", "executable_sha256", "qualification", "qualification_anchor")}}
    if harness == "claude-code":
        provider.update(harness=harness, permission_mode="acceptEdits", sandbox="none", safe_mode=True,
                        tools=list(CLAUDE_AUTHORING_TOOLS), event_contract=CLAUDE_EVENT_CONTRACT, terminal_schema=terminal_schema())
    else:
        provider.update(sandbox="workspace-write", service_tier="default", disabled_features=[],
                        event_contract="tool_rich_candidate_v1", code_mode_host=dict(CAMPAIGN_BINDING),
                        output_schema={"path": OUTPUT_SCHEMA, "sha256": sha256((root / OUTPUT_SCHEMA).read_bytes()).hexdigest()})
    return {
        "schema_version": 2, "state": "template", "kind": "matched_search",
        "study_id": f"{workload.workload_id}-{harness}-artifact-optimization",
        "claim_scope": "artifact_optimization_only",
        "agent_interface": {"schema_version": 1, "kind": "task_agents_ralph_v1"},
        "workload": {"path": str(workload_path), "canonical_sha256": workload.canonical_sha256},
        "arms": {"open_cake": {
            "environment_kind": "open_cake", "reference_access": "known_kernel_reproduction", "provider": provider,
            "scaffold": {"path": SCAFFOLD, "sha256": sha256((root / SCAFFOLD).read_bytes()).hexdigest()},
            "compiler_revision": dict(CURRENT_RELEASE_BINDING),
            "lowering_route": source.document["lowering"],
            "schedule_skeleton": {"path": str(starter_path), "canonical_sha256": sha256(canonical(source.document)).hexdigest()},
            "input_format": "schedule_or_python_v1", "tool_surface": ["submit_schedule_or_python"],
            "feedback": ["findings", "correctness", "qualified_timing", "profile"],
            "toolchain_sha256": dict(CAMPAIGN_BINDING),
        }},
        "allocation": {"method": "predeclared_balanced_blocks", "order": ["open_cake-1"]},
        "budget": budget,
        "run_protocol": {"automatic_retries": 0, "independent_thread": True, "replacement_runs": 0,
                         "resume_invariants": ["authority", "cwd", "sandbox", "provider", "scaffold", "arm_environment", "task_package"],
                         "workspace_seed": "task_agents_only"},
        "evaluation_protocol": evaluation_policy(workload, searches_per_turn=searches_per_turn,
                                                 dispatches_per_sample=dispatches_per_sample,
                                                 maximum_cv=maximum_cv, required_pair_wins=required_pair_wins),
        "execution": {"target": workload.target, "executor_revision": dict(CURRENT_RELEASE_BINDING),
                      "broker_execution_sha256": dict(CAMPAIGN_BINDING), "fixed_baseline": dict(CAMPAIGN_BINDING),
                      "gpu": {"name": device_name(workload.target), "count": 1,
                              "mode": "local_serialized" if source.document["lowering"]["backend"] == "metal" else "exclusive"},
                      "sandbox": provider["sandbox"]},
        "analysis_plan": {**json.loads(canonical(_ARTIFACT_OPTIMIZATION_ANALYSIS_PLAN)),
                          "endpoint_policy": NORMAL_BUDGET_TERMINAL,
                          "performance_reporting": TASK_EFFICIENCY_V1},
        "evidence": {"schema_version": 3, "terminal_archive_required_for_every_run": True,
                     "event_vocabulary": "matched_ralph_v1"},
    }
