#!/usr/bin/env python3
"""GPU Infra adapter for the existing whole-task development correctness runner."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler
from open_cake_ir.source_identity import checkout_commit
from open_cake_ir.tasks.solx_fib.catalog import TASK_IDS
from tools.compare_flashinfer_reference import admit_judge_source
from tools import test_flashinfer_b300


def selected_tasks(path: Path) -> list[str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("tasks.json must be a regular input")
    document = json.loads(path.read_text())
    if not isinstance(document, dict) or set(document) != {"tasks"}:
        raise ValueError("task selection fields differ")
    tasks = document["tasks"]
    if (not isinstance(tasks, list) or not tasks
            or any(not isinstance(task, str) or task not in TASK_IDS for task in tasks)
            or len(set(tasks)) != len(tasks)):
        raise ValueError("select unique registered pack tasks")
    return tasks


def probe_arguments(root: Path, tasks: list[str]) -> list[str]:
    path = root / 'probe.json'
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file() or len(tasks) != 1:
        raise ValueError('a probe requires one task and regular probe.json')
    config = json.loads(path.read_text())
    if not isinstance(config, dict) or set(config) not in ({'kind', 'rows'}, {'kind', 'variant'}):
        raise ValueError('probe fields differ')
    if config.get('kind') == 'source' and set(config) == {'kind', 'rows'}:
        if type(config['rows']) is not int or config['rows'] <= 0:
            raise ValueError('probe rows must be positive')
        return ['--candidate-source', str(root / 'candidate.py'), '--rows', str(config['rows'])]
    if config.get('kind') == 'plan' and set(config) == {'kind', 'variant'}:
        if config['variant'] not in ('captured', 'boundary'):
            raise ValueError('unknown probe variant')
        return ['--candidate-plan', str(root / 'plan.json'), '--variant', config['variant']]
    raise ValueError('probe kind differs')


def main() -> int:
    stage = Path(os.environ["KERNELINFRA_STAGE_DIR"])
    destination = Path(os.environ["KERNELINFRA_RESULT"])
    result = {"schema": "kernelinfra.stage-result.v1", "status": "failed",
              "validity": "unknown", "summary": "not evaluated", "artifacts": {}}
    try:
        commit = checkout_commit(ROOT)
        admit_judge_source(commit, json.loads(Path(os.environ["KERNELINFRA_TASK"]).read_text()),
                           os.environ["KERNELINFRA_STAGE_ID"])
        candidate = Path(os.environ["KERNELINFRA_CANDIDATE_DIR"])
        tasks = selected_tasks(candidate / "tasks.json")
        probe = probe_arguments(candidate, tasks)
        gate = Compiler.load(ROOT, ROOT / "compiler/revision.json").check_corpus()
        if not gate.passed:
            raise ValueError("Corpus Gate failed before GPU execution")
        output = stage / "checks"
        arguments = ["test_flashinfer_b300.py", "--output", str(output), *probe]
        for task in tasks:
            arguments.extend(("--task", task))
        original = sys.argv
        try:
            sys.argv = arguments
            test_flashinfer_b300.main()
        finally:
            sys.argv = original
        report = json.loads((output / "report.json").read_text())
        if [row["task_id"] for row in report["tasks"]] != tasks:
            raise ValueError("test coverage differs from the submitted selection")
        passed = all(row["status"] == "passed" for row in report["tasks"])
        result.update(status="passed" if passed else "failed",
                      validity="valid" if passed else "unknown",
                      summary=f"{sum(row['status'] == 'passed' for row in report['tasks'])}/{len(tasks)} task checks passed; correctness only, no performance claim",
                      artifacts={"report": "checks/report.json"})
    except Exception as error:
        result["summary"] = f"{type(error).__name__}: {error}"
        with (stage / "adapter-error.txt").open("x") as stream:
            stream.write(traceback.format_exc())
        result["artifacts"]["error"] = "adapter-error.txt"
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
