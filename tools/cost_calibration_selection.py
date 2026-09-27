"""Pure oracle-scope and empirical candidate-cut admission for calibration Runs."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

from open_cake_ir.compiler import EmpiricalCostModel
from open_cake_ir.lab.selection import _empirical_filter


def _read(path):
    return json.loads(Path(path).read_text())


def _candidate_set_regrets(audit, maximum_candidates_per_turn):
    """Audit every possible Lab three-to-two cut and provider tie order."""
    if type(maximum_candidates_per_turn) is not int or maximum_candidates_per_turn != 3:
        raise ValueError("cost audit requires the Lab's maximum_candidates_per_turn=3")
    regrets = []
    for workload_id in sorted({row["workload_id"] for row in audit}):
        group = sorted((row for row in audit if row["workload_id"] == workload_id), key=lambda row: row["id"])
        for candidate_set in itertools.combinations(group, maximum_candidates_per_turn):
            best_observed = min(row["observed_us"] for row in candidate_set)
            worst = None
            for provider_order in itertools.permutations(candidate_set):
                # Python's stable sort is the Lab rule: equal predictions retain
                # provider order, which can change which member falls below a cut.
                survivors = sorted(provider_order, key=lambda row: row["predicted_us"])[:2]
                regret = min(row["observed_us"] for row in survivors) / best_observed
                if worst is None or regret > worst["top2_regret_ratio"]:
                    worst = {"workload_id": workload_id,
                             "candidate_set": [row["id"] for row in candidate_set],
                             "provider_order": [row["id"] for row in provider_order],
                             "survivors": [row["id"] for row in survivors],
                             "top2_regret_ratio": regret}
            regrets.append(worst)
    return regrets

def _input_scope(plan):
    """Parse the oracle's shape domain rather than trusting free-text equality."""
    try:
        scope = json.loads(plan["input_scope"])
    except (TypeError, ValueError) as error:
        raise ValueError("calibration input_scope must be canonical structured JSON") from error
    if (not isinstance(scope, dict)
            or plan["input_scope"] != json.dumps(scope, sort_keys=True, separators=(",", ":"))
            or set(scope) != {"schema_version", "oracle", "distributions", "families"}
            or type(scope["schema_version"]) is not int or scope["schema_version"] != 1
            or scope["oracle"] != "independent_cpu_case_v1"
            or type(scope["distributions"]) is not list
            or len(scope["distributions"]) != 2
            or any(type(value) is not int for value in scope["distributions"])
            or scope["distributions"] != [0, 1]
            or not isinstance(scope["families"], list) or not scope["families"]):
        raise ValueError("calibration input_scope must be canonical structured JSON")
    admitted = {"fma": {"a", "b", "c", "y"},
                "gemm_bias": {"a", "b", "bias", "c"}}
    families = {}
    for item in scope["families"]:
        if (not isinstance(item, dict) or set(item) != {"family", "buffers"}
                or item["family"] not in admitted or item["family"] in families
                or not isinstance(item["buffers"], dict)
                or set(item["buffers"]) != admitted[item["family"]]):
            raise ValueError("calibration input_scope family fields differ")
        for shape in item["buffers"].values():
            if (not isinstance(shape, list) or not shape
                    or any(value is not None and (type(value) is not int or value <= 0)
                           for value in shape)):
                raise ValueError("calibration input_scope buffer shape differs")
        families[item["family"]] = item["buffers"]
    return families

def _check_input_scope_case(families, case, document):
    patterns = families.get(case["family"])
    actual = {row["name"]: row["shape"] for row in document["buffers"]
              if row["space"] == "global"}
    if patterns is None or set(actual) != set(patterns):
        raise ValueError("Schedule family or global buffers differ from input_scope")
    for name, shape in actual.items():
        pattern = patterns[name]
        if (len(shape) != len(pattern)
                or any(value != (case["extent"] if declared is None else declared)
                       for value, declared in zip(shape, pattern, strict=True))):
            raise ValueError("Schedule shape differs from declared input_scope")

def _candidate_file(candidate, relative, label):
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} path differs")
    path = (candidate / relative).resolve()
    if candidate not in path.parents or not path.is_file():
        raise ValueError(f"{label} must belong to the candidate snapshot")
    return path

def _device_case_indices(plan):
    selection = plan.get("search_selection")
    if selection is None:
        return list(range(len(plan["cases"])))
    by_id = {case["id"]: index for index, case in enumerate(plan["cases"])}
    return [by_id[name] for name in selection["selected_case_ids"]]


def _range_separated_cut(ordered, survivor_count):
    """Admit a cut only when one survivor's descriptive upper bound clears every skip.

    The ranges are empirical error envelopes, not confidence guarantees. Overlap is
    a reason to spend GPU time on the full set instead of acting on a point estimate.
    """
    selected = ordered[:survivor_count]
    skipped = ordered[survivor_count:]
    return bool(selected and skipped) and min(
        row["empirical_cost"]["empirical_range_us"][1] for row in selected
    ) < min(
        row["empirical_cost"]["empirical_range_us"][0] for row in skipped
    )

def _validate_search_selection(candidate, plan):
    """Freeze a complete empirical cut before any GPU stage."""
    selection = plan.get("search_selection")
    if selection is None:
        return
    required = {"kind", "model_path", "predictions_path", "submitted_case_ids",
                "selected_case_ids", "searches_per_workload"}
    if (not isinstance(selection, dict) or set(selection) != required
            or selection["kind"] != "external_empirical_top_k_v1"
            or type(selection["searches_per_workload"]) is not int
            or selection["searches_per_workload"] != 2):
        raise ValueError("empirical search selection fields differ")
    submitted = selection["submitted_case_ids"]
    selected = selection["selected_case_ids"]
    if (not isinstance(submitted, list) or not isinstance(selected, list)
            or any(not isinstance(name, str) or not name for name in submitted + selected)
            or len(set(submitted)) != len(submitted)
            or len(set(selected)) != len(selected)):
        raise ValueError("empirical search candidate ids differ")
    by_id = {case["id"]: case for case in plan["cases"]}
    if any(name not in by_id or by_id[name]["split"] != "audit" for name in submitted):
        raise ValueError("empirical submitted set must name audit Schedules")
    if not set(selected) <= set(submitted):
        raise ValueError("empirical selection names an unsubmitted candidate")
    model_document = _read(_candidate_file(candidate, selection["model_path"], "empirical model"))
    frozen = _read(_candidate_file(candidate, selection["predictions_path"], "frozen predictions"))
    model = EmpiricalCostModel(model_document)
    context = model_document["context"]
    if (model.compiler_revision_id != plan["compiler_revision_id"]
            or model.target != plan["target"]
            or context["timer"] != "PyTorch Kineto CUPTI GPU kernel activity"
            or context["cache_protocol"] != f"{plan['sampling']['l2_flush_bytes']}-byte zeroing before each sample on same stream"
            or context["input_scope"] != plan["input_scope"]
            or context["runtime"] != plan["expected_runtime"]):
        raise ValueError("empirical model differs from the frozen assay")
    if (frozen.get("schema_version") != 1
            or frozen.get("kind") != "frozen_prior_model_predictions_for_prospective_audit"
            or frozen.get("prior_model_id") != model.model_id
            or frozen.get("prior_model_compiler_revision_id") != model.compiler_revision_id
            or frozen.get("target") != model.target
            or frozen.get("assay_context") != context
            or frozen.get("acceptance") != plan["model_acceptance"]
            or frozen.get("maximum_candidates_per_turn") != 3
            or frozen.get("searches_per_turn") != 2):
        raise ValueError("frozen empirical predictions differ from model or assay")
    frozen_rows = {row["case_id"]: row for row in frozen["predictions"]}
    if len(frozen_rows) != len(submitted) or set(frozen_rows) != set(submitted):
        raise ValueError("frozen empirical prediction domain differs")
    groups = {}
    for name in submitted:
        case = by_id[name]
        schedule = _read(_candidate_file(candidate, case["schedule"], "submitted Schedule"))
        estimate = model.estimate(schedule, compiler_revision_id=plan["compiler_revision_id"],
                                  target=plan["target"],
                                  compiled_compiler_version=context["runtime"]["compiler_version"])
        recorded = frozen_rows[name]
        if (estimate["covered"] is not True or recorded.get("covered") is not True
                or recorded.get("extent") != case["extent"]
                or recorded.get("predicted_us") != estimate["predicted_kernel_us"]
                or recorded.get("empirical_range_us") != estimate["empirical_range_us"]):
            raise ValueError("frozen empirical prediction differs from model replay")
        groups.setdefault(case["workload_id"], []).append(
            {"candidate_sha256": name, "disposition": "launchable", "empirical_cost": estimate})
    if not groups or any(len(group) != 3 for group in groups.values()):
        raise ValueError("empirical search requires complete three-candidate workloads")
    decisions = []
    for group in groups.values():
        cases = [by_id[row["candidate_sha256"]] for row in group]
        if len({case["extent"] for case in cases}) != 1 or len({case["curve_id"] for case in cases}) != 3:
            raise ValueError("empirical search group mixes extents or repeats a curve")
        ordered, decision = _empirical_filter(group)
        if not decision["order_applied"]:
            raise ValueError("empirical search must have complete comparable coverage")
        if not _range_separated_cut(ordered, selection["searches_per_workload"]):
            raise ValueError("empirical GPU cut lacks separation between descriptive ranges")
        decisions.append({"extent": cases[0]["extent"],
                          "candidate_set": [row["candidate_sha256"] for row in group],
                          "selected_top2": [row["candidate_sha256"] for row in ordered[:2]],
                          "complete_coverage": True})
    if ([name for decision in decisions for name in decision["selected_top2"]] != selected
            or frozen.get("audit_extents") != [row["extent"] for row in decisions]
            or frozen.get("decisions") != decisions):
        raise ValueError("empirical GPU cut differs from frozen predictions")
