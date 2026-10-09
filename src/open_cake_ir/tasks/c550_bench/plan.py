"""Allocate one task's fixed budget to original-case Ralph Runs and confirmation.

This preparation plan is not an execution engine or a candidate selector. Each
row becomes an ordinary RunSpecification only after its acceptance gates pass.
"""
from __future__ import annotations

TOTAL_SECONDS = 10800
FINAL_CONFIRMATION_SECONDS = 1080


def case_budget_plan(problem) -> dict:
    uuids = [row.uuid for row in problem.workloads]
    if len(uuids) != 16 or len(set(uuids)) != 16:
        raise ValueError("Shared Bench budget requires the 16 distinct original cases")
    base, remainder = divmod(TOTAL_SECONDS - FINAL_CONFIRMATION_SECONDS, len(uuids))
    return {
        "task": problem.task_id,
        "scope": "in_suite_optimization",
        "status": "prepared_not_launched",
        "total_budget_seconds": TOTAL_SECONDS,
        "final_confirmation_seconds": FINAL_CONFIRMATION_SECONDS,
        "run_budget_semantics": "sum_of_allocated_Ralph_wall_time_including_each_Run_confirmation",
        "overrun_policy": "retain_atomic_operation_overshoot_and_refuse_within_budget_success",
        "failure_policy": "retain_failure_without_retry_or_budget_redistribution",
        "reference_access": "high_level_reference_read",
        "cases": [{"workload_uuid": uuid, "wall_time_seconds": base + (index < remainder),
                   "confirmation_wall_time_seconds": (base + (index < remainder)) / 10,
                   "candidate": None, "run": None}
                  for index, uuid in enumerate(uuids)],
        "final_check": {"workload_scope": "all", "rounds": 10,
                        "expected_checks": 160, "performance": "not_measured"},
    }
