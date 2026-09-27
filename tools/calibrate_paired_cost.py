#!/usr/bin/env python3
"""Admit a frozen B300 GEMM paired-cost plan before any GPU allocation.

This first stage only validates the external candidate snapshot. Device collection
and fitting will use this same plan; check-plan alone produces no cost model.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler  # noqa: E402
from open_cake_ir.evaluation.paired import PAIRED_KIND, candidate_identity, paired_protocol  # noqa: E402
from open_cake_ir.lab.bindings import load_baseline_bundle, load_compiler_reference  # noqa: E402
from open_cake_ir.lab.executor import ExecutorRevision, _external_file  # noqa: E402
from open_cake_ir.lab.selection import _paired_empirical_context  # noqa: E402
from open_cake_ir.tasks.launch import parse_launch_manifest  # noqa: E402
from open_cake_ir.tasks.workloads import load_workload  # noqa: E402


def _external(path: str | Path) -> Path:
    resolved = Path(path).resolve(strict=True)
    if resolved == ROOT or ROOT in resolved.parents:
        raise ValueError("paired calibration snapshot must remain outside source")
    return resolved


def _read(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def observation_order(names: list[str]) -> list[dict[str, str]]:
    """Freeze each split before observing any fit or audit outcome."""
    if len(names) != 3 or len(set(names)) != 3:
        raise ValueError("paired calibration needs three distinct candidates")
    return [{"id": f"{split}-{name}", "candidate_id": name, "split": split}
            for split, order in (("fit", names), ("calibration", names[::-1]),
                                 ("audit", names[1:] + names[:1]))
            for name in order]


def check_plan(snapshot: str | Path) -> dict[str, object]:
    """Resolve all non-device authorities from one immutable external snapshot."""
    snapshot = _external(snapshot)
    plan = _read(_external_file(snapshot, "plan.json", "paired cost plan"))
    fields = {"schema_version", "state", "plan_id", "model_id", "compiler_revision",
              "executor_revision", "study_path", "workload", "case_id", "target",
              "baseline_bundle_path", "candidates", "observations",
              "varying_dimensions", "acceptance"}
    if (not isinstance(plan, dict) or set(plan) != fields
            or type(plan["schema_version"]) is not int or plan["schema_version"] != 1
            or plan["state"] != "frozen"
            or any(not isinstance(plan[key], str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", plan[key]) is None
                   for key in ("plan_id", "model_id"))):
        raise ValueError("paired cost plan fields or freeze state differ")
    revision = load_compiler_reference(ROOT, plan["compiler_revision"], "paired_cost.compiler_revision")
    compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
    executor = ExecutorRevision.load_reference(ROOT, plan["executor_revision"], "paired_cost.executor_revision")
    if revision.revision_id != compiler._revision.revision_id or executor.document["target"] != plan["target"]:
        raise ValueError("paired cost Compiler or Executor target differs")
    if (plan["target"] != "sm_103a" or plan["case_id"] != "primary"
            or not isinstance(plan["study_path"], str)):
        raise ValueError("paired cost currently admits the B300 primary case")
    study = _read(_external_file(ROOT, plan["study_path"], "paired cost Study"))
    ref = plan["workload"]
    if (not isinstance(ref, dict) or set(ref) != {"path", "workload_id", "canonical_sha256"}
            or not isinstance(ref["path"], str)):
        raise ValueError("paired cost Workload reference differs")
    workload = load_workload(_external_file(ROOT, ref["path"], "paired cost Workload"))
    if (workload.workload_id != ref["workload_id"]
            or workload.canonical_sha256 != ref["canonical_sha256"]
            or workload.target != plan["target"]
            or workload.document["operator"] != "gemm_bias_bf16_fp32"
            or not isinstance(study, dict)
            or study.get("kind") != "matched_search"
            or study.get("claim_scope") != "scientific_matched_search"
            or study.get("state") != "template"
            or study.get("execution", {}).get("target") != plan["target"]
            or study.get("workload") != {"path": ref["path"],
                                        "canonical_sha256": ref["canonical_sha256"]}):
        raise ValueError("paired cost Study or Workload binding differs")
    policy = study["evaluation_protocol"]
    protocol = paired_protocol(policy)
    if protocol is None or policy["case_id"] != plan["case_id"] or policy["paired_timing"]["kind"] != PAIRED_KIND:
        raise ValueError("paired cost Study does not declare the expected CUPTI case")
    baseline_path = plan["baseline_bundle_path"]
    if not isinstance(baseline_path, str) or Path(baseline_path).is_absolute():
        raise ValueError("paired cost baseline must belong to the candidate snapshot")
    baseline = load_baseline_bundle(
        ROOT, _external_file(snapshot, baseline_path, "paired cost baseline bundle"),
    )
    manifest = parse_launch_manifest(json.loads(baseline.artifact_payloads["launch_manifest"]))
    manifest.check_workload(workload, plan["case_id"])
    if (baseline.target != plan["target"] or baseline.target != manifest.target
            or baseline.entry_point != manifest.kernel_name
            or baseline.launch_spec_sha256 != manifest.canonical_sha256):
        raise ValueError("paired cost baseline target or launch seal differs")
    candidates = plan["candidates"]
    if (not isinstance(candidates, list) or len(candidates) != 3
            or any(not isinstance(item, dict) or set(item) != {"id", "schedule"}
                   or not isinstance(item["id"], str)
                   or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", item["id"]) is None
                   or not isinstance(item["schedule"], str) for item in candidates)):
        raise ValueError("paired cost candidate pool differs")
    names = [item["id"] for item in candidates]
    if (len(set(names)) != 3 or len({item["schedule"] for item in candidates}) != 3
            or plan["observations"] != observation_order(names)):
        raise ValueError("paired cost pool or frozen observation order differs")
    bindings = plan["varying_dimensions"]
    if (not isinstance(bindings, list) or bindings != [{"buffer": name, "dimension": 0} for name in ("a", "c")]):
        raise ValueError("paired cost B300 GEMM varying dimensions differ")
    route = study["arms"]["open_cake"]["lowering_route"]
    expected_abi = [(arg.name, arg.dtype, list(arg.shape), arg.mode)
                    for arg in workload.tensor_abi(plan["case_id"])]
    signatures: set[bytes] = set()
    for item in candidates:
        source = _external_file(snapshot, item["schedule"], "paired cost Schedule")
        assessment = compiler.assess_file(source)
        if assessment.target != plan["target"] or not assessment.accepted or not assessment.lowering_eligible:
            raise ValueError(f"paired cost Schedule {item['id']} is not admitted")
        document = json.loads(assessment.schedule_bytes)
        observed_abi = [(b["name"], b["dtype"], b["shape"], b["mode"])
                        for b in document["buffers"] if b["space"] == "global"]
        if (document["lowering"] != route or observed_abi != expected_abi
                or document["metadata"].get("workload_contract_sha256") != workload.canonical_sha256):
            raise ValueError(f"paired cost Schedule {item['id']} differs from Study route or Workload binding")
        distinct = {**document, "schedule_id": "candidate-display-id"}
        signatures.add(json.dumps(distinct, sort_keys=True, separators=(",", ":")).encode())
    if len(signatures) != 3:
        raise ValueError("paired cost candidates repeat one Schedule under different names")
    limits = plan["acceptance"]
    required_limits = {"maximum_baseline_drift_ratio", "maximum_mape",
                       "maximum_relative_error", "maximum_top2_regret_ratio", "envelope_allowance"}
    if not isinstance(limits, dict) or set(limits) != required_limits:
        raise ValueError("paired cost acceptance fields differ")
    for key, value in limits.items():
        minimum = 0 if key == "envelope_allowance" else 1 if key.endswith("ratio") else 0
        if (type(value) not in (int, float) or not math.isfinite(value)
                or value < minimum or key in {"maximum_mape", "maximum_relative_error"} and value == 0):
            raise ValueError(f"paired cost acceptance {key} differs")
    if limits["envelope_allowance"] >= 1:
        raise ValueError("paired cost envelope allowance differs")
    context = _paired_empirical_context(
        executor, workload_sha256=workload.canonical_sha256,
        case_id=plan["case_id"], evaluation_protocol=policy,
        baseline_identity=candidate_identity(baseline),
    )
    return {"plan_id": plan["plan_id"], "compiler_revision_id": revision.revision_id,
            "executor_revision_id": executor.executor_id, "workload_id": workload.workload_id,
            "target": plan["target"], "candidate_count": 3,
            "observation_count": len(plan["observations"]), "context": context}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check-plan",))
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    print(json.dumps(check_plan(args.snapshot), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
