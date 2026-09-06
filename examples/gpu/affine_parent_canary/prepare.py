"""Prepare one source-bound affine parent canary; no GPU or provider calls."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
from open_cake_ir.compiler import Compiler
from canary import C, N, PARENT, SPATIAL, TASK_ID, WORKLOAD, inputs, reference, write_json

SOURCE_COMMIT = "26fbf8f3f2bc691707f82d0e79e7327386861feb"
RESULTS = "docs/data/aka-fma-v41-reaudit-20260906/results.jsonl"


def prepare(output: Path, python: Path, judge_cwd: Path) -> dict:
    if not python.is_absolute() or not judge_cwd.is_absolute():
        raise ValueError("judge Python and cwd must be explicit absolute runtime paths")
    if Path(inspect.getfile(Compiler)).resolve() != ROOT / "src/open_cake_ir/compiler/core.py":
        raise ValueError("Compiler imported from a different checkout")
    expected_lock = subprocess.run(
        ["git", "--no-replace-objects", "-C", str(ROOT), "show", f"{SOURCE_COMMIT}:compiler/revision.lock.json"],
        check=True, capture_output=True,
    ).stdout
    if (ROOT / "compiler/revision.lock.json").read_bytes() != expected_lock:
        raise ValueError("canary requires its frozen v43 Compiler source; use its full Git checkout")
    compiler = Compiler.load(ROOT, ROOT / "compiler/revision.lock.json")
    # Read the reviewed candidate from its fixed source object, not mutable data in a
    # different checkout. The generator never edits this Schedule or emitted source.
    result_bytes = subprocess.run(
        ["git", "--no-replace-objects", "-C", str(ROOT), "show", f"{SOURCE_COMMIT}:{RESULTS}"],
        check=True, capture_output=True, text=True,
    ).stdout
    matches = [row for row in map(json.loads, result_bytes.splitlines()) if row["case_id"] == PARENT]
    if len(matches) != 1 or matches[0]["terminal_status"] != "candidate_lowered_compiled":
        raise ValueError("fixed parent candidate is missing")
    schedule = json.loads(matches[0]["schedule_json"])
    assessment = compiler.assess(schedule)
    if not assessment.accepted or not assessment.lowering_eligible:
        raise ValueError("reviewed candidate refused by its Compiler")
    lowering = compiler.lower(assessment)
    compile(lowering.source, "<affine-parent>", "exec")
    if len(reference()) != N * C * SPATIAL:
        raise ValueError("complete-output oracle differs")
    contract = {"task_id": TASK_ID, "parent_case_id": PARENT, "assessment_scope": "fixed_instance",
                "workload": WORKLOAD, "shape": [N, C, SPATIAL], "compiler_source_commit": SOURCE_COMMIT,
                "compiler_revision_id": lowering.compiler_revision_id, "schedule_source": RESULTS,
                "schedule_id": assessment.schedule_id, "entry_point": lowering.route.entry_point,
                "input_bits": inputs(), "correctness": "all eight output bits, unchanged inputs, output guards and supplied-output ABI",
                "claim_boundary": "selected sample only; no dynamic parent ABI, other shapes, timing or serving claim",
                "performance_measured": False}
    output = output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    candidate = output / "candidate"; candidate.mkdir()
    (candidate / "kernel.py").write_text(lowering.source, encoding="utf-8")
    (candidate / "schedule.json").write_text(json.dumps(schedule, indent=2) + "\n")
    shutil.copyfile(HERE / "canary.py", candidate / "canary.py")
    write_json(candidate / "contract.json", contract)
    task = {"schema": "kernelinfra.task.v1", "task_id": TASK_ID,
            "description": "One fixed N2/C2/S2 affine parent correctness canary from Compiler v43; no timing.",
            "workloads": [WORKLOAD], "comparison": {"primary_workloads": [WORKLOAD], "relative_noise_floor": 0.0},
            "stages": [{"id": "correctness", "kind": "correctness",
                        "judge": {"identity": "open-cake-affine-exact-rational-v1", "cwd": str(judge_cwd),
                                  "command": [str(python), "-c", "import os,runpy,sys; p=os.environ['KERNELINFRA_CANDIDATE_DIR']; sys.path.insert(0,p); runpy.run_path(os.path.join(p,'canary.py'),run_name='__main__')"]},
                        "resources": {"mode": "shared", "gpu_count": 1, "estimate_s": 60,
                                      "queue_timeout_s": 600, "run_timeout_s": 240}}]}
    write_json(output / "task.json", task)
    return contract


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--judge-cwd", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_root, args.python, args.judge_cwd), indent=2))
