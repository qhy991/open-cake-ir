"""Original Bench Workloads for the common Ralph and native Evaluation owners."""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

from open_cake_ir.evaluation.workload import WorkloadContract
from .binding import BenchProblem, OPERATOR, validate_document

REQUIRES_TARGET_PREPARATION = True
PRESERVE_OUTPUT_TENSORS = True


@lru_cache(maxsize=16)
def problem_for(workload):
    binding = workload.document["semantics"]["benchmark"]
    return BenchProblem.open(Path(binding["root"]), binding["task"]), binding["workload_uuid"]


class BenchWorkload(WorkloadContract):
    def compare_output_values(self, expected, observed):
        problem, uuid = problem_for(self)
        check = problem.compare(uuid, expected, observed)
        # A Bench output contract includes its matched-ratio rule. Keep that
        # original verdict; count failed output contracts, not tolerated elements.
        failures = sum(not row["passed"] for row in check["outputs"].values())
        if not check["passed"] and failures == 0:
            failures = 1  # Original shape/name/type refusal has no numeric rows.
        errors = [row.get("max_absolute_error", 0.) for row in check["outputs"].values()]
        finite = [float(value) for value in errors if isinstance(value, (int, float)) and math.isfinite(value)]
        return check["passed"], {
            "output_mismatches": failures, "max_abs_error": max(finite, default=0.),
            "comparison_unit": "original_bench_output_contract",
            "original_bench_check": check,
        }


def prepare_evaluation_case(workload, case_id, *, admission):
    if case_id != "primary" or workload.target != admission.target:
        raise ValueError("Bench preparation differs from its case or admitted target")
    problem, uuid = problem_for(workload)
    inputs, expected, observed = problem.prepare_on_target(uuid, runtime_library=admission.runtime_library)
    if observed != admission:
        raise ValueError("Bench oracle preparation changed the device lease")
    return inputs, expected


validate_contract = validate_document
