#!/usr/bin/env python3
"""Prepare the create-only fixed-copy Lab canary package."""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
from pathlib import Path
from typing import Any


CASE_ID = (
    "data_movement_and_layout__copy_vectorized__analysis__l000007_b200_v1"
    "__directderived_sol_ultra_v2"
)
TASK_ID = "open-cake-aka-copy4-n1024-b200-canary-v6"
WORKLOADS = [
    "zeros-n1024",
    "signed-integers-n1024",
    "alternating-binary-fractions-n1024",
    "seeded-scaled-integers-n1024",
]


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected one JSON object: {path}")
    return value


def prepare(
    *, admission_root: Path, output_root: Path, python: Path, judge_cwd: Path
) -> dict[str, Any]:
    admission = admission_root.expanduser().resolve(strict=True)
    output = output_root.expanduser().resolve()
    runtime_python = python.expanduser().absolute()
    runtime_cwd = judge_cwd.expanduser().resolve(strict=True)
    if not runtime_python.is_file() or not os.access(runtime_python, os.X_OK):
        raise RuntimeError(f"runtime Python is not executable: {runtime_python}")
    if output.exists():
        raise RuntimeError(f"output root already exists: {output}")
    manifest = _load(admission / "admission-manifest.json")
    if (
        manifest.get("schema") != "open-cake.aka-ir-lab-admission.v1"
        or manifest.get("canary_case_id") != CASE_ID
        or manifest.get("policy", {}).get("max_in_flight_after_canary") != 5
    ):
        raise RuntimeError("admission manifest or canary policy differs")
    selected = [entry for entry in manifest.get("entries", []) if entry.get("canary")]
    if len(selected) != 1 or selected[0].get("case_id") != CASE_ID:
        raise RuntimeError("admission manifest does not select the fixed canary")
    if (
        selected[0].get("entry_point") != "copy4_contiguous_fp32_n1024"
        or selected[0].get("target") != "sm_100a"
        or selected[0].get("workload_id") != "n1024"
    ):
        raise RuntimeError("canary Schedule binding differs")

    source_candidate = admission / "cases" / CASE_ID / "candidate"
    kernel_source = (source_candidate / "kernel.py").read_text(encoding="utf-8")
    tree = ast.parse(kernel_source)
    if not any(
        isinstance(node, ast.FunctionDef)
        and node.name == "copy4_contiguous_fp32_n1024"
        for node in tree.body
    ):
        raise RuntimeError("canary lowered wrapper differs")

    candidate = output / "candidate"
    candidate.mkdir(parents=True)
    shutil.copyfile(source_candidate / "kernel.py", candidate / "kernel.py")
    shutil.copyfile(source_candidate / "provenance.json", candidate / "provenance.json")
    shutil.copyfile(Path(__file__).with_name("evaluate.py"), candidate / "evaluate.py")
    task = {
        "schema": "kernelinfra.task.v1",
        "task_id": TASK_ID,
        "description": (
            "Virtualenv-entry successor, not a retry: correctness and sanitizer canary "
            "for exactly one fixed n=1024 Open-Cake Triton lowering on B200; no timing, "
            "broader parent, release, or training claim."
        ),
        "workloads": WORKLOADS,
        "comparison": {
            "primary_workloads": WORKLOADS,
            "relative_noise_floor": 0.0,
        },
        "stages": [],
    }
    command = [
        str(runtime_python),
        "-c",
        (
            "import os,runpy; "
            "runpy.run_path(os.path.join(os.environ['KERNELINFRA_CANDIDATE_DIR'], "
            "'evaluate.py'), run_name='__main__')"
        ),
    ]
    task["stages"] = [
        {
            "id": "correctness",
            "kind": "correctness",
            "judge": {
                "identity": "open-cake-aka-copy4-independent-oracle-v1",
                "cwd": str(runtime_cwd),
                "command": command,
            },
            "resources": {
                "mode": "shared",
                "gpu_count": 1,
                "estimate_s": 60,
                "queue_timeout_s": 1800,
                "run_timeout_s": 300,
            },
        },
        {
            "id": "sanitize-memcheck",
            "kind": "sanitize",
            "judge": {
                "identity": "open-cake-aka-copy4-independent-oracle-v1",
                "cwd": str(runtime_cwd),
                "command": command,
            },
            "resources": {
                "mode": "exclusive",
                "gpu_count": 1,
                "estimate_s": 240,
                "queue_timeout_s": 3600,
                "run_timeout_s": 900,
            },
        },
        {
            "id": "sanitize-racecheck",
            "kind": "sanitize",
            "judge": {
                "identity": "open-cake-aka-copy4-independent-oracle-v1",
                "cwd": str(runtime_cwd),
                "command": command,
            },
            "resources": {
                "mode": "exclusive",
                "gpu_count": 1,
                "estimate_s": 240,
                "queue_timeout_s": 3600,
                "run_timeout_s": 900,
            },
        },
    ]
    (output / "task.json").write_text(
        json.dumps(task, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    preflight = {
        "schema": "open-cake.aka-ir-lab-canary-preflight.v1",
        "case_id": CASE_ID,
        "task_id": TASK_ID,
        "entry_point": selected[0]["entry_point"],
        "target": selected[0]["target"],
        "workload_id": selected[0]["workload_id"],
        "correctness_distributions": WORKLOADS,
        "performance_measured": False,
        "max_in_flight_after_canary": 5,
    }
    (output / "preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return preflight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--judge-cwd", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            prepare(
                admission_root=arguments.admission_root,
                output_root=arguments.output_root,
                python=arguments.python,
                judge_cwd=arguments.judge_cwd,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
