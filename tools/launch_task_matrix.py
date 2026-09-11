#!/usr/bin/env python3
"""Run registered TaskLab tasks sequentially through the canonical single-task launcher.

This is an outer-loop driver only. It owns no Workload, Study, provider, Compiler,
Executor, Evaluation or acceptance semantics; every task is still prepared and audited by
``tools/launch_task.py``. The first successful provider qualification is reused under the
same matrix treatment. A failure before that authority exists is common setup failure and
stops the matrix; a later task/campaign failure is retained and the next task proceeds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import launch_task  # noqa: E402
ALL_TASKS = (
    "rmsnorm", "layernorm", "residual_rmsnorm", "softmax",
    *launch_task.ACTIVATION_TASKS, *launch_task.ROWWISE_TASKS,
    *launch_task.REDUCTION_TASKS, *launch_task.OPTIMIZER_TASKS,
    *launch_task.CONTRACTION_TASKS, "gemm_bias",
)
DEPTH_TASKS = frozenset((*launch_task.CONTRACTION_TASKS, "gemm_bias"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)


def _append(path: Path, value: dict[str, object]) -> None:
    with path.open("ab") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                ensure_ascii=False, allow_nan=False).encode() + b"\n")


def _tasks(values: list[str] | None) -> tuple[str, ...]:
    selected = tuple(values or ALL_TASKS)
    if len(set(selected)) != len(selected):
        raise ValueError("task matrix cannot contain duplicate tasks")
    return selected


def _command(args, task: str, workspace: Path, qualification: tuple[Path, Path] | None) -> list[str]:
    command = [sys.executable, str(ROOT / "tools/launch_task.py"),
        "--task", task, "--backend", args.backend,
        "--harness", args.harness, "--model", args.model, "--effort", args.effort,
        "--workspace", str(workspace), "--turns", str(args.turns),
        "--token-budget", str(args.token_budget), "--max-candidates", str(args.max_candidates),
        "--searches-per-turn", str(args.searches_per_turn),
        "--dispatches-per-sample", str(args.dispatches_per_sample),
        "--wall-seconds", str(args.wall_seconds)]
    for flag, value in (("--rows", args.rows), ("--columns", args.columns)):
        if value is not None:
            command.extend((flag, str(value)))
    if task in DEPTH_TASKS:
        command.extend(("--depth", str(args.depth)))
    if args.provider_executable is not None:
        command.extend(("--provider-executable", str(args.provider_executable)))
    if args.provider_revision is not None:
        command.extend(("--provider-revision", args.provider_revision))
    if qualification is not None:
        command.extend(("--qualification", str(qualification[0]),
                        "--qualification-anchor", str(qualification[1])))
    return command


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", choices=ALL_TASKS,
                        help="task to run; repeat to select an ordered subset; default is all")
    parser.add_argument("--backend", choices=tuple(launch_task.DEVICE_BACKENDS), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--harness", choices=("codex", "claude-code"), required=True)
    parser.add_argument("--effort", required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--provider-executable", type=Path)
    parser.add_argument("--provider-revision")
    parser.add_argument("--rows", type=int)
    parser.add_argument("--columns", type=int)
    parser.add_argument("--depth", type=int, default=256,
                        help="K extent for contraction and gemm_bias tasks")
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--token-budget", type=int, default=150000)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--searches-per-turn", type=int, default=2)
    parser.add_argument("--dispatches-per-sample", type=int, default=64)
    parser.add_argument("--wall-seconds", type=int, default=14400)
    args = parser.parse_args(argv)

    try:
        selected = _tasks(args.task)
        root = launch_task._new_workspace(args.workspace_root)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if type(args.depth) is not int or args.depth <= 0:
        parser.error("--depth must be a positive integer")
    root.mkdir(mode=0o750, parents=True)
    summary_path = root / "task-results.jsonl"
    _write(root / "matrix.json", json.dumps({
        "schema_version": 1, "tasks": list(selected), "backend": args.backend,
        "provider": {"harness": args.harness, "model": args.model, "effort": args.effort,
                     "revision": args.provider_revision},
        "budget": {"turns": args.turns, "provider_tokens_per_task": args.token_budget,
                   "maximum_candidates_per_turn": args.max_candidates,
                   "searches_per_turn": args.searches_per_turn,
                   "wall_seconds_per_task": args.wall_seconds},
        "shape": {"rows": args.rows, "columns": args.columns, "depth": args.depth},
        "qualification_policy": "qualify_first_task_then_reuse_exact_receipt",
        "failure_policy": "stop_before_first_qualification; retain_and_continue_afterward",
    }, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n")

    qualification = None
    failures = 0
    stopped = False
    for position, task in enumerate(selected, start=1):
        workspace = root / task
        started = _now()
        reused = qualification is not None
        print(f"START {position}/{len(selected)} {task} {started}", flush=True)
        command = _command(args, task, workspace, qualification)
        completed = subprocess.run(command, cwd=ROOT, capture_output=True)
        _write(root / f"{task}.stdout", completed.stdout)
        _write(root / f"{task}.stderr", completed.stderr)
        receipt, anchor = workspace / "provider-qualification.json", workspace / "provider-anchor.json"
        qualified_here = receipt.is_file() and anchor.is_file()
        if qualification is None and qualified_here:
            qualification = (receipt.resolve(strict=True), anchor.resolve(strict=True))
        row = {"schema_version": 1, "position": position, "task": task,
               "workspace": str(workspace), "started_at": started, "finished_at": _now(),
               "exit_code": completed.returncode, "qualification_available": qualification is not None,
               "qualification_reused": reused}
        _append(summary_path, row)
        print(f"DONE {position}/{len(selected)} {task} exit={completed.returncode}", flush=True)
        failures += int(completed.returncode != 0)
        if qualification is None:
            stopped = True
            break

    terminal = {"schema_version": 1, "status": (
        "stopped_before_provider_qualification" if stopped else
        "completed_with_task_faults" if failures else "completed"),
        "task_count_requested": len(selected),
        "task_count_attempted": len(summary_path.read_bytes().splitlines()),
        "task_fault_count": failures, "qualification_reused": qualification is not None,
        "finished_at": _now()}
    _write(root / "terminal.json", json.dumps(terminal, sort_keys=True, indent=2,
        ensure_ascii=False, allow_nan=False).encode() + b"\n")
    print(root / "terminal.json")
    return 1 if stopped or failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
