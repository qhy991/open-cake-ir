#!/usr/bin/env python3
"""Run registered TaskLab tasks with bounded host concurrency through the canonical single-task launcher.

This is an outer-loop driver only. It owns no Workload, Study, provider, Compiler,
Executor, Evaluation or acceptance semantics; every task is still prepared and audited by
``tools/launch_task.py``. All baselines are compiled, sealed and ABI-checked before the
first provider call, then reused by their tasks. Parallel launches qualify once before starting any
optimization Run. Each task keeps its own budget, workspace and evidence; the existing
broker owns device serialization, mapping and admission. A failure before that authority exists is common setup failure and
stops the matrix; a later task/campaign failure is retained and the next task proceeds.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
from threading import Event

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from tools import launch_task  # noqa: E402
from open_cake_ir.tasks.catalog import task_names, matrix_depth as _depth
from open_cake_ir.serialization import canonical_json_bytes  # noqa: E402
ALL_TASKS = task_names(suite="portable")
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)


def _append(path: Path, value: dict[str, object]) -> None:
    with path.open("ab") as stream:
        stream.write(canonical_json_bytes(value) + b"\n")


def _tasks(values: list[str] | None) -> tuple[str, ...]:
    selected = tuple(values or ALL_TASKS)
    if len(set(selected)) != len(selected):
        raise ValueError("task matrix cannot contain duplicate tasks")
    return selected


def _command(args, task: str, workspace: Path, qualification: tuple[Path, Path] | None,
             prepared_baseline: Path | None = None) -> list[str]:
    command = [sys.executable, str(ROOT / "tools/launch_task.py"),
        "--task", task, "--backend", args.backend,
        "--harness", args.harness, "--model", args.model, "--effort", args.effort,
        "--workspace", str(workspace), "--turns", str(args.turns),
        "--max-candidates", str(args.max_candidates),
        "--searches-per-turn", str(args.searches_per_turn),
        "--wall-seconds", str(args.wall_seconds)]
    for flag, value in (("--token-budget", args.token_budget), ("--maximum-cv", args.maximum_cv), ("--required-pair-wins", args.required_pair_wins)):
        if value is not None:
            command.extend((flag, str(value)))
    if args.dispatches_per_sample is not None:
        command.extend(("--dispatches-per-sample", str(args.dispatches_per_sample)))
    for flag, value in (("--gpu-run", args.gpu_run), ("--broker-socket", args.broker_socket)):
        if value is not None:
            command.extend((flag, str(value)))
    for field in ("agents_md", "kernelctl", "infra_socket", "local_device", "local_queue_seconds", "local_lock_scope", "local_runtime_device", "local_expected_pci"):
        value = getattr(args, field, None)
        if value is not None:
            command.extend(("--" + field.replace("_", "-"), str(value)))
    for flag, value in (("--rows", args.rows), ("--columns", args.columns)):
        if value is not None:
            command.extend((flag, str(value)))
    depth = _depth(task, args.depth)
    if depth is not None:
        command.extend(("--depth", str(depth)))
    if args.provider_executable is not None:
        command.extend(("--provider-executable", str(args.provider_executable)))
    for alias in getattr(args, "response_model_alias", ()):
        command.extend(("--response-model-alias", alias))
    if args.provider_revision is not None:
        command.extend(("--provider-revision", args.provider_revision))
    if args.incumbent_registry is not None and prepared_baseline is None:
        command.extend(("--incumbent-registry", str(args.incumbent_registry)))
    if qualification is not None:
        command.extend(("--qualification", str(qualification[0]),
                        "--qualification-anchor", str(qualification[1])))
    if prepared_baseline is not None:
        command.extend(("--prepared-baseline", str(prepared_baseline)))
    return command


def _bounded_map(items, operation, workers, stop):
    """Keep at most workers host jobs active; never own a device allocation."""
    if workers == 1:
        for item in items:
            if stop.is_set(): break
            yield operation(item)
        return
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = set()
        def fill():
            while len(pending) < workers and not stop.is_set():
                item = next(iterator, None)
                if item is None: break
                pending.add(pool.submit(operation, item))
        fill()
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
            fill()


def _qualification(root, workspace):
    receipt, anchor = workspace / "provider-qualification.json", workspace / "provider-anchor.json"
    if receipt.is_file() and anchor.is_file():
        return (launch_task.external_file(root, str(receipt), "provider qualification"),
                launch_task.external_file(root, str(anchor), "provider qualification anchor"))
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", choices=launch_task.TASKS,
                        help="task to run; repeat to select any launchable subset; default is the portable matrix")
    parser.add_argument("--backend", choices=tuple(launch_task.DEVICE_BACKENDS), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--harness", choices=("codex", "claude-code"), required=True)
    parser.add_argument("--effort", required=True)
    parser.add_argument("--response-model-alias", action="append", default=[])
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--agents-md", type=Path)
    parser.add_argument("--kernelctl", type=Path)
    parser.add_argument("--infra-socket", type=Path)
    parser.add_argument('--local-device', type=int)
    parser.add_argument('--local-lock-scope', choices=('user', 'device'), default='user')
    parser.add_argument('--local-runtime-device', type=int)
    parser.add_argument('--local-expected-pci')
    parser.add_argument('--local-queue-seconds', type=float, default=0)
    parser.add_argument("--provider-executable", type=Path)
    parser.add_argument("--provider-revision")
    parser.add_argument(
        "--incumbent-registry",
        type=Path,
        help="external per-task incumbent registry resolved independently by each task",
    )
    parser.add_argument("--rows", type=int)
    parser.add_argument("--columns", type=int)
    parser.add_argument("--depth", type=int,
                        help="K override; portable contractions default to256, TinyGEMM keeps its task default")
    parser.add_argument("--turns", type=int, default=32)
    parser.add_argument("--token-budget", type=int,
                        help="optional per-task token threshold; omitted means usage accounting only")
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--searches-per-turn", type=int, default=2)
    parser.add_argument("--maximum-cv", type=float)
    parser.add_argument("--required-pair-wins", type=int)
    parser.add_argument("--dispatches-per-sample", type=int, help="Metal only; default: 64")
    parser.add_argument("--gpu-run", type=Path)
    parser.add_argument("--broker-socket", type=Path)
    parser.add_argument("--parallel-tasks", type=int, default=1,
                        help="maximum simultaneous host task processes; GPU allocation stays with the broker")
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--qualification-anchor", type=Path)
    parser.add_argument("--wall-seconds", type=int, default=28800)
    args = parser.parse_args(argv)
    if args.parallel_tasks < 1:
        parser.error('--parallel-tasks must be positive')
    if (args.qualification is None) != (args.qualification_anchor is None):
        parser.error('--qualification and --qualification-anchor must be supplied together')
    if not math.isfinite(args.local_queue_seconds) or args.local_queue_seconds < 0:
        parser.error('--local-queue-seconds must be finite and nonnegative')
    if (args.parallel_tasks > 1 and args.kernelctl is None
            and launch_task._allocation_of(args.backend) == 'local_broker'
            and args.local_queue_seconds <= 0):
        parser.error('parallel local tasks require a positive --local-queue-seconds for the existing broker')

    try:
        selected = _tasks(args.task)
        root = launch_task._new_workspace(args.workspace_root)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.depth is not None and (type(args.depth) is not int or args.depth <= 0):
        parser.error("--depth must be a positive integer")
    # Every factory owns its supported backends and shape contract. Resolve the
    # whole requested subset before writing a workspace or spending provider work;
    # do not silently omit an explicitly requested unsupported operator.
    task_shapes = {}
    for task in selected:
        rows, columns = launch_task._default_shape(task, args.rows, args.columns)
        try:
            document, _ = launch_task.create_task(task, backend=args.backend, rows=rows, columns=columns,
                                    depth=_depth(task, args.depth), case_id="primary")
            task_shapes[task] = next(case['shape'] for case in document['cases'] if case['case_id'] == 'primary')
        except (TypeError, ValueError) as error:
            parser.error(f"task {task!r} cannot run on {args.backend!r}: {error}; select an explicit supported subset")
    root.mkdir(mode=0o750, parents=True)
    summary_path = root / "task-results.jsonl"
    qualification = None
    if args.qualification is not None:
        qualification = (launch_task.external_file(ROOT, str(args.qualification), "provider qualification"),
                         launch_task.external_file(ROOT, str(args.qualification_anchor), "provider qualification anchor"))
    _write(root / "matrix.json", json.dumps({
        "schema_version": 1, "tasks": list(selected), "backend": args.backend,
        "provider": {"harness": args.harness, "model": args.model, "effort": args.effort,
                     "revision": args.provider_revision,
                     **({"response_model_aliases": args.response_model_alias} if args.response_model_alias else {})},
        "budget": {"turns": args.turns, "provider_tokens_per_task": args.token_budget,
                   "maximum_candidates_per_turn": args.max_candidates,
                   "searches_per_turn": args.searches_per_turn,
                   "wall_seconds_per_task": args.wall_seconds},
        "shape": {"rows": args.rows, "columns": args.columns, "depth": args.depth},
        "task_shapes": task_shapes,
        "timing_overrides": {"maximum_cv": args.maximum_cv, "required_pair_wins": args.required_pair_wins},
        "qualification_policy": ("reuse_supplied_exact_receipt" if qualification else
            "qualify_before_parallel_runs" if args.parallel_tasks > 1 else
            "qualify_first_task_then_reuse_exact_receipt"),
        "concurrency": {"maximum_host_tasks": args.parallel_tasks,
                        "device_admission": "existing_broker",
                        "local_queue_seconds": args.local_queue_seconds,
                        "budget_accounting": "each_Run_includes_its_broker_wait"},
        "baseline_policy": (
            {
                "kind": "exact_incumbent_or_reference",
                "registry_root": str(args.incumbent_registry),
            }
            if args.incumbent_registry is not None
            else {"kind": "starter_reference"}
        ),
        "failure_policy": "stop_before_first_qualification; retain_and_continue_afterward",
    }, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n")

    failures = 0
    stopped = False
    stop = Event()
    # Complete CPU baseline preparation for the entire subset before provider work.
    baselines = {}
    def prepare(item):
        position, task = item
        preparation = root / "baseline-preflight" / task
        print(f"PREPARE {position}/{len(selected)} {task} {_now()}", flush=True)
        prepared = subprocess.run(
            [*_command(args, task, preparation, None), "--baseline-only"],
            cwd=ROOT, capture_output=True)
        return position, task, preparation, prepared
    failed_baselines = []
    for position, task, preparation, prepared in _bounded_map(
            enumerate(selected, 1), prepare, args.parallel_tasks, stop):
        _write(root / f"{task}.prepare.stdout", prepared.stdout)
        _write(root / f"{task}.prepare.stderr", prepared.stderr)
        _append(root / "baseline-results.jsonl", {"position": position, "task": task,
            "exit_code": prepared.returncode, "workspace": str(preparation)})
        if prepared.returncode != 0:
            failed_baselines.append(task)
            stop.set()  # Record already active preparations, but start no new ones.
        else:
            baselines[task] = launch_task.external_file(
                ROOT, str(preparation / "prepared-baseline.json"), "prepared baseline")
    if failed_baselines:
        _write(root / "terminal.json", json.dumps({"schema_version": 1,
            "status": "stopped_before_baseline_preflight", "failed_task": failed_baselines[0],
            "task_count_requested": len(selected), "task_count_attempted": 0,
            "provider_calls": 0, "finished_at": _now()}, indent=2).encode() + b"\n")
        return 1

    if args.parallel_tasks > 1 and qualification is None:
        preflight = root / "provider-preflight"
        completed = subprocess.run([*_command(args, selected[0], preflight, None,
            baselines[selected[0]]), "--preflight-only"], cwd=ROOT, capture_output=True)
        _write(root / "provider-preflight.stdout", completed.stdout)
        _write(root / "provider-preflight.stderr", completed.stderr)
        qualification = _qualification(ROOT, preflight)
        if completed.returncode != 0 or qualification is None:
            _write(root / "terminal.json", json.dumps({"schema_version": 1,
                "status": "stopped_before_provider_qualification",
                "task_count_requested": len(selected), "task_count_attempted": 0,
                "finished_at": _now()}, indent=2).encode() + b"\n")
            return 1

    def run_task(item):
        position, task = item
        workspace = root / task
        started = _now()
        reused = qualification is not None
        print(f"START {position}/{len(selected)} {task} {started}", flush=True)
        command = _command(args, task, workspace, qualification, baselines[task])
        completed = subprocess.run(command, cwd=ROOT, capture_output=True)
        row = {"schema_version": 1, "position": position, "task": task,
               "workspace": str(workspace), "started_at": started, "finished_at": _now(),
               "exit_code": completed.returncode, "qualification_reused": reused}
        return row, completed
    for row, completed in _bounded_map(enumerate(selected, 1), run_task, args.parallel_tasks, stop):
        task = row['task']
        _write(root / f"{task}.stdout", completed.stdout)
        _write(root / f"{task}.stderr", completed.stderr)
        if qualification is None:
            qualification = _qualification(ROOT, root / task)
        row['qualification_available'] = qualification is not None
        _append(summary_path, row)
        print(f"DONE {row['position']}/{len(selected)} {task} exit={completed.returncode}", flush=True)
        failures += int(completed.returncode != 0)
        if qualification is None:
            stopped = True
            stop.set()

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
