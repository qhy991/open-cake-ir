"""Canonical Study, evidence, event and claim policy values."""

from __future__ import annotations

import json
from typing import Mapping, cast

from ._documents import _canonical_json_bytes
from .pairing import triton_optimization_analysis_plan


_RALPH_STUDY_FIELDS = {
    "schema_version",
    "study_id",
    "state",
    "kind",
    "claim_scope",
    "workload",
    "arms",
    "agent_interface",
    "allocation",
    "budget",
    "run_protocol",
    "evaluation_protocol",
    "execution",
    "analysis_plan",
    "evidence",
}


_PORTFOLIO_STUDY_FIELDS = {
    "schema_version",
    "study_id",
    "state",
    "kind",
    "claim_scope",
    "workload",
    "compiler_revision",
    "kernel_seed",
    "case_roles",
    "specialization_policy",
    "dispatch_policy",
    "evaluation_protocol",
    "execution",
    "analysis_plan",
    "evidence",
}


_SYSTEM_QUALIFICATION_ANALYSIS_PLAN = {
    "experimental_unit": "run",
    "estimand": None,
    "qualification_criterion": (
        "all_prescheduled_runs_complete_integrity_semantic_replay_"
        "adhered_and_evaluated"
    ),
    "comparative_statistics": "forbidden",
    "pooling": "forbidden",
}


_ARTIFACT_OPTIMIZATION_ANALYSIS_PLAN = {
    "experimental_unit": "run",
    "estimand": None,
    "artifact_promotion": {
        "scope": "per_run",
        "eligibility": "adhered_and_confirmatory_receipt_qualifies",
        "rank": "lowest_confirmed_latency_ms",
        "tie_break": "earliest_turn",
    },
    "comparative_statistics": "forbidden",
    "pooling": "forbidden",
    "scientific_inclusion": "forbidden",
}


_MATCHED_RALPH_EVENT_VOCABULARY_V1 = "matched_ralph_v1"


_MATCHED_EVENT_KINDS_V1 = frozenset(
    {
        "run_started",
        "provider_turn_completed",
        "candidate_set_filtered",
        "candidate_rejected",
        "launchable_candidate_sealed",
        "evaluation_attempt_completed",
        "candidate_evaluated",
        "diagnosis_routed",
        "candidate_selected",
        "run_fault",
        "checkpoints_projected",
        "run_terminal",
    }
)


_MATCHED_RALPH_EVIDENCE_POLICY_V1 = {
    "schema_version": 3,
    "terminal_archive_required_for_every_run": True,
    "event_vocabulary": _MATCHED_RALPH_EVENT_VOCABULARY_V1,
}


_SCIENTIFIC_MATCHED_ANALYSIS_PLAN_V2 = {
    "experimental_unit": "run",
    "target_population": "prescheduled_runs_under_exact_campaign_lock",
    "primary_endpoint": [
        "qualified_by_budget",
        "best_confirmed_latency_ms_if_qualified",
    ],
    "contrast": "two_part_open_cake_vs_direct_cuda",
    "estimand": (
        "terminal-budget qualification-rate difference and conditional confirmed "
        "performance"
    ),
    "missingness": {
        "candidate_failure": "observed_outcome",
        "external_fault": "missing",
        "replacement": "forbidden",
    },
    "pooling": "forbidden_without_successor_analysis_plan",
    "availability": "all_prescheduled_runs_observed_and_each_arm_has_qualified_run",
    "summary_statistics": {
        "qualification": "arm_rate",
        "qualification_contrast": "open_cake_rate_minus_direct_cuda_rate",
        "conditional_latency": "arm_median_ms",
        "contrast": "direct_cuda_median_divided_by_open_cake_median",
        "uncertainty": "per_arm_observed_range_ms",
    },
    "direction": "lower_latency_is_better",
}


_MATCHED_CLAIM_SCOPES = {
    "system_qualification_only",
    "artifact_optimization_only",
    "scientific_matched_search",
}


_ONE_RUN_PER_ARM_SCOPES = {
    "system_qualification_only",
    "artifact_optimization_only",
}


_LEGACY_ATTRIBUTION_EVALUATION = "correctness_then_profile"


_ATTRIBUTION_EVALUATION = "correctness_then_profile_each_search_survivor"


def scientific_matched_analysis_plan_v2() -> Mapping[str, object]:
    """Return the sole current two-part scientific Analysis Plan projection."""

    return cast(
        Mapping[str, object],
        json.loads(_canonical_json_bytes(_SCIENTIFIC_MATCHED_ANALYSIS_PLAN_V2)),
    )


def _matched_evidence_policy_version(
    policy: Mapping[str, object], context: str
) -> str:
    """Admit only the Ralph evidence contract."""

    if policy == _MATCHED_RALPH_EVIDENCE_POLICY_V1:
        return _MATCHED_RALPH_EVENT_VOCABULARY_V1
    raise ValueError(f"{context} is unsupported")


def _scientific_analysis_plan_version(
    analysis: Mapping[str, object], context: str
) -> str:
    """Admit the current scientific plans for the two supported comparisons."""

    if analysis == triton_optimization_analysis_plan():
        return "triton_optimization_v1"
    if analysis == _SCIENTIFIC_MATCHED_ANALYSIS_PLAN_V2:
        return "two_part_v2"
    raise ValueError(f"{context} is unsupported")
