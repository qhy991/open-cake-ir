"""Task-specific input preparation composed outside the Ralph engine."""
from __future__ import annotations

from open_cake_ir.lab.pairing import bind_baseline, comparison_arm
from .flash_kmeans.authoring import prepare_flash_schedule, validate_flash_authoring


def prepare_schedule(schedule, workload, case_id, arm):
    if arm.get("input_format") == "schedule_or_python_v1":
        return bind_baseline(schedule, workload, case_id, backend=arm["lowering_route"]["backend"])
    return prepare_flash_schedule(schedule, workload, case_id)


def validate_authoring(workload, arms, *, empirical_cost_model_path=None):
    comparison = comparison_arm(arms)
    empirical = "candidate_selection" in arms["open_cake"] or empirical_cost_model_path is not None
    if empirical and (
        comparison != "direct_cuda" or workload.document.get("operator") != "flash_kmeans_assign"
    ):
        raise ValueError("empirical selection requires the Flash/direct-CUDA assay")
    if comparison == "direct_cuda":
        validate_flash_authoring(workload, arms)
