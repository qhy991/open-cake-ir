"""Create one immutable correctness-only package from the released v41 Compiler."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from open_cake_ir.compiler import Compiler
from oracle import FAMILIES, MODES, SHAPE


def prepare(output: Path, python: Path, judge_cwd: Path) -> dict:
    output = output.absolute()
    python = python.absolute()  # Preserve the virtualenv invocation, not symlink target.
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("runtime Python is not executable")
    if not judge_cwd.is_dir():
        raise ValueError("judge cwd does not exist")
    compiler = Compiler.load(ROOT, ROOT / "compiler/revision.lock.json")
    lock = json.loads((ROOT / "compiler/revision.lock.json").read_text())
    if lock["state"] != "released" or lock["revision_id"] != "open-cake-ir-sm100a-v41":
        raise ValueError("this task requires the released v41 Compiler")
    candidate = output / "candidate"
    output.mkdir(parents=True, exist_ok=False)
    candidate.mkdir()
    schedules = {"ordinary": "fma-b8-smoke.json", "nested": "fma-chain-b8-smoke.json"}
    entries = {}
    for mode, name in schedules.items():
        schedule = json.loads((ROOT / "corpus/schedules" / name).read_text())
        assessment = compiler.assess(schedule)
        if not assessment.accepted or not assessment.lowering_eligible:
            raise ValueError(f"released schedule refused: {name}")
        artifact = compiler.lower(assessment)
        (candidate / f"{mode}.py").write_text(artifact.source)
        entries[mode] = schedule["lowering"]["entry_point"]
    for name in ("evaluate.py", "oracle.py"):
        shutil.copyfile(Path(__file__).with_name(name), candidate / name)
    workloads = [f"{m}-{f}" for m in MODES for f in FAMILIES]
    contract = {
        "compiler_revision_id": lock["revision_id"], "shape": list(SHAPE),
        "entry_points": entries, "workloads": workloads,
        "numeric_acceptance": "bitwise except NaN payload/sign; NaN results must be quiet",
        "oracle": "exact integer binary32 RN-even arithmetic, no FTZ",
        "input_policy": "all original input bits unchanged; four disjoint pointers",
        "output_policy": "return supplied FP32 output with unchanged pointer and shape",
        "performance_measured": False,
    }
    (candidate / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    command = [str(python), "-c",
               "import os,runpy,sys; p=os.environ['KERNELINFRA_CANDIDATE_DIR']; "
               "sys.path.insert(0,p); runpy.run_path(os.path.join(p,'evaluate.py'),run_name='__main__')"]
    task = {"schema": "kernelinfra.task.v1",
            "task_id": "open-cake-fma-v41-b8x128-b200-correctness-v1",
            "description": "Released v41 FMA and nested FMA exact-output correctness; no timing.",
            "workloads": workloads,
            "comparison": {"primary_workloads": workloads, "relative_noise_floor": 0.0},
            "stages": []}
    for stage_id, kind, mode, estimate, timeout in [
        ("correctness", "correctness", "shared", 120, 600),
        ("sanitize-memcheck", "sanitize", "exclusive", 300, 1200),
        ("sanitize-racecheck", "sanitize", "exclusive", 300, 1200),
    ]:
        task["stages"].append({
            "id": stage_id, "kind": kind,
            "judge": {"identity": "open-cake-fma-integer-oracle-v1",
                      "cwd": str(judge_cwd.absolute()), "command": command},
            "resources": {"mode": mode, "gpu_count": 1, "estimate_s": estimate,
                          "queue_timeout_s": 3600, "run_timeout_s": timeout},
        })
    (output / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    return contract


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--judge-cwd", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_root, args.python, args.judge_cwd), indent=2))
