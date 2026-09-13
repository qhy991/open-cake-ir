"""Closed prospective task-performance reporting policy, independent of selection."""
from __future__ import annotations

from typing import Mapping

TASK_EFFICIENCY_V1 = "task_efficiency_v1"


def performance_reporting_policy(analysis: Mapping[str, object], claim_scope: str) -> str | None:
    if "performance_reporting" not in analysis:
        return None
    if analysis["performance_reporting"] != TASK_EFFICIENCY_V1:
        raise ValueError("Analysis Plan performance reporting policy differs")
    if claim_scope != "artifact_optimization_only":
        raise ValueError("task_efficiency_v1 requires artifact_optimization_only claim scope")
    return TASK_EFFICIENCY_V1


def analysis_without_performance_policy(analysis: Mapping[str, object], claim_scope: str) -> dict[str, object]:
    """Validate this one optional field; leave all other fields for whole-plan admission."""
    performance_reporting_policy(analysis, claim_scope)
    return {key: value for key, value in analysis.items() if key != "performance_reporting"}
