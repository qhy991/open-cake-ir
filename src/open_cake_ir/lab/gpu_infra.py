"""GPU Infra transport for the existing sealed Evaluation command protocol.

One accepted node run is observed, never resubmitted. GPU Infra owns its
snapshot, queue and lifecycle; the worker owns all correctness/timing evidence.
The CLI boundary deliberately needs no kernel_infra Python dependency.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

from open_cake_ir.evaluation.gpuq import backend_for_target, observe_allocation
from open_cake_ir.evaluation.source_bootstrap import module_command
from open_cake_ir.source_identity import checkout_commit

ROOT = Path(__file__).resolve().parents[3]
TERMINAL = {"completed", "rejected", "infra_error", "cancelled", "interrupted"}


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def input_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("GPU Infra artifact path escapes its root")
    path = root / relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("GPU Infra artifact custody differs")
    return path


def snapshot_request(request_path: Path, destination: Path) -> dict:
    request = json.loads(request_path.read_bytes())
    destination.mkdir()
    names = set(request["artifact_paths"].values())
    if "baseline" in request:
        names.update(request["baseline"]["artifact_paths"].values())
    if names & {"request.json", "workload.json"}:
        raise ValueError("candidate artifact collides with task input")
    for name in names:
        source = input_file(request_path.parent, name)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    workload = Path(request["workload_path"])
    if workload.is_symlink() or not workload.is_file():
        raise ValueError("Workload custody differs")
    shutil.copyfile(workload, destination / "workload.json")
    write_json(destination / "request.json", request)
    return request


def task_document(request: dict, *, worker_module: str, timeout: int) -> dict:
    commit = checkout_commit(ROOT)
    target = request["target"]
    backend_for_target(target)  # refuse undeclared targets before submission
    return {"schema": "kernelinfra.task.v1", "task_id": "cake-" + target.replace("_", "-"),
        "description": "One sealed Cake Evaluation; Lab owns acceptance and promotion",
        "workloads": [request["case_id"]],
        "comparison": {"primary_workloads": [request["case_id"]], "guardrails": {},
                       "relative_noise_floor": 0.0},
        "stages": [{"id": "evaluation", "kind": "judge", "execution": "broker",
            "judge": {"identity": "open-cake-ir@" + commit, "cwd": str(ROOT),
                "command": module_command(sys.executable, "open_cake_ir.lab.gpu_infra", "stage",
                    "--target", target, "--commit", commit, "--worker-module", worker_module)},
            "resources": {"mode": "exclusive", "gpu_count": 1, "estimate_s": None,
                          "queue_timeout_s": timeout, "run_timeout_s": timeout}}]}


def stage(args) -> int:
    result_path = Path(os.environ["KERNELINFRA_RESULT"])
    stage_dir = Path(os.environ["KERNELINFRA_STAGE_DIR"])
    work = stage_dir / "worker"
    try:
        if checkout_commit(ROOT) != args.commit:
            raise ValueError("GPU Infra judge source commit differs")
        allocation = observe_allocation(args.target)
        source = Path(os.environ["KERNELINFRA_CANDIDATE_DIR"])
        request = json.loads(input_file(source, "request.json").read_bytes())
        if request["target"] != args.target:
            raise ValueError("GPU Infra task and sealed candidate targets differ")
        # The worker writes only in its stage; the accepted snapshot stays immutable.
        work.mkdir()
        names = set(request["artifact_paths"].values())
        if "baseline" in request:
            names.update(request["baseline"]["artifact_paths"].values())
        if names & {"request.json", "workload.json", "result.json"}:
            raise ValueError("artifact collides with worker protocol files")
        for name in names | {"workload.json"}:
            origin = input_file(source, name)
            target = work / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, target)
        request["workload_path"] = str(work / "workload.json")
        write_json(work / "request.json", request)
        completed = subprocess.run(module_command(sys.executable, args.worker_module,
            "--request", str(work / "request.json"), "--output", str(work / "result.json")),
            cwd=ROOT, check=False, env={k: v for k, v in os.environ.items()
                if k not in {"METAL_JOB_ID", "METAL_BROKER_LOCK_FD"}})
        if completed.returncode:
            raise ValueError(f"Evaluation worker exited {completed.returncode}")
        result = json.loads(input_file(work, "result.json").read_bytes())
        if result.get("job_id") != allocation["job_id"] or result.get("mode") != "exclusive":
            raise ValueError("Evaluation worker allocation differs")
        receipt = result.get("receipt")
        correctness = receipt.get("correctness_passed") if isinstance(receipt, dict) else None
        validity = ("valid" if correctness is True else "invalid" if correctness is False else "unknown")
        if result.get("admitted") is not True or result.get("error") is not None:
            validity = "unknown"
        write_json(result_path, {"schema": "kernelinfra.stage-result.v1",
            "status": "passed" if validity == "valid" else "failed", "validity": validity,
            "summary": "Cake worker evidence; no independent GPU Infra promotion",
            "artifacts": {"cake_result": "worker/result.json"}})
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        write_json(result_path, {"schema": "kernelinfra.stage-result.v1", "status": "failed",
            "validity": "unknown", "summary": str(error)})
        return 1


def invoke(kernelctl: str, arguments: list[str], *, timeout: float = 45):
    return subprocess.run([kernelctl, *arguments], capture_output=True, text=True,
                          timeout=timeout, check=False)


def preflight(kernelctl: str, socket: Path, target: str) -> dict:
    observed = invoke(kernelctl, ["node-status", "--socket", str(socket), "--json"])
    if observed.returncode:
        raise ValueError("GPU Infra node observation unavailable: " + observed.stderr)
    node = json.loads(observed.stdout)
    broker = node.get("broker", {})
    expected = backend_for_target(target)
    if (broker.get("probe_error") or not broker.get("instance_id")
            or not isinstance(broker.get("gpus"), list) or not broker["gpus"]
            or broker.get("backend") != expected
            or broker.get("allocation_environment") != "gpuq_v1"
            or broker.get("occupancy_scope") != ("cooperative" if expected == "metal" else "system")):
        raise ValueError("GPU Infra broker backend, observation or allocation protocol differs")
    return node


def submit(args) -> int:
    root = args.evidence_root.resolve()
    if any((p / ".git").exists() for p in (root, *root.parents)):
        raise ValueError("GPU Infra evidence must be outside source checkouts")
    root.mkdir(parents=True, exist_ok=True)
    evidence = root / uuid.uuid4().hex
    evidence.mkdir()
    request = snapshot_request(args.request.resolve(strict=True), evidence / "input")
    write_json(evidence / "node.json", preflight(args.kernelctl, args.socket, request["target"]))
    task = task_document(request, worker_module=args.worker_module, timeout=args.timeout)
    task_path = evidence / "task.json"
    write_json(task_path, task)
    checked = invoke(args.kernelctl, ["task-check", str(task_path)])
    if checked.returncode:
        raise ValueError("GPU Infra task refused: " + checked.stderr)
    accepted = invoke(args.kernelctl, ["submit", "--socket", str(args.socket),
                                     "--task", str(task_path), str(evidence / "input")])
    (evidence / "submit.stdout").write_text(accepted.stdout)
    (evidence / "submit.stderr").write_text(accepted.stderr)
    run_id = accepted.stdout.strip()
    if accepted.returncode or re.fullmatch(r"[a-z0-9][a-z0-9._-]+-[0-9a-f]{12}", run_id) is None:
        raise ValueError(f"GPU Infra submission outcome unknown; inspect {evidence}; do not resubmit")
    write_json(evidence / "locator.json", {"run_id": run_id, "socket": str(args.socket)})

    def cancel(signum, frame):
        # Only cancel this accepted node run; never retry or select another node.
        result = invoke(args.kernelctl, ["cancel", "--socket", str(args.socket), run_id])
        (evidence / "cancel.stdout").write_text(result.stdout)
        (evidence / "cancel.stderr").write_text(result.stderr)
        raise InterruptedError("GPU Infra observation interrupted")

    previous = {s: signal.signal(s, cancel) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        deadline = time.monotonic() + 2 * args.timeout + 30
        while True:
            observed = invoke(args.kernelctl, ["wait", "--socket", str(args.socket), run_id,
                                              "--timeout", "20", "--json"])
            if observed.returncode not in {0, 3}:
                raise ValueError(f"GPU Infra observation unknown for {run_id}; retained at {evidence}")
            state = json.loads(observed.stdout)
            if state.get("run_id") != run_id:
                raise ValueError("GPU Infra observed another run")
            if state.get("state") in TERMINAL:
                break
            if time.monotonic() > deadline:
                cancel(signal.SIGTERM, None)
        write_json(evidence / "terminal.json", state)
        run_dir = Path(state["run_dir"]).resolve(strict=True)
        if run_dir.name != run_id:
            raise ValueError("GPU Infra run directory identity differs")
        stage_root = run_dir / "stages/evaluation"
        worker = stage_root / "worker"
        if state["state"] not in {"completed", "rejected"}:
            raise ValueError(f"GPU Infra {state['state']}: {run_id}; evidence {evidence}")
        payload = input_file(worker, "result.json").read_bytes()
        result = json.loads(payload)
        job = state.get("broker_job_id")
        if not job or result.get("job_id") != job:
            raise ValueError("GPU Infra and Evaluation job identities differ")
        receipt = result.get("receipt")
        if isinstance(receipt, dict):
            for name in receipt["artifacts"].values():
                destination = args.output.parent / name
                if Path(name).is_absolute() or ".." in Path(name).parts:
                    raise ValueError("Evaluation output path escapes destination")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as stream:
                    stream.write(input_file(worker, name).read_bytes())
        with args.output.open("xb") as stream:
            stream.write(payload)
        # Preserve the existing CommandBrokerSubmitter protocol and raw node logs.
        print(f"[gpu-run] accepted job {job}", file=sys.stderr)
        print(f"GPU Infra evidence: {evidence}", file=sys.stderr)
        return 0
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    client = commands.add_parser("submit")
    client.add_argument("--kernelctl", required=True)
    client.add_argument("--socket", type=Path, required=True)
    client.add_argument("--evidence-root", type=Path, required=True)
    client.add_argument("--request", type=Path, required=True)
    client.add_argument("--output", type=Path, required=True)
    client.add_argument("--timeout", type=int, default=1800)
    worker = commands.add_parser("stage")
    worker.add_argument("--target", required=True)
    worker.add_argument("--commit", required=True)
    for sub in (client, worker):
        sub.add_argument("--worker-module", required=True)
    args = parser.parse_args(argv)
    if re.fullmatch(r"open_cake_ir\.[a-zA-Z0-9_.]+", args.worker_module) is None:
        parser.error("worker must be an open_cake_ir module")
    if args.action == "submit" and args.timeout <= 0:
        parser.error("timeout must be positive")
    return stage(args) if args.action == "stage" else submit(args)


if __name__ == "__main__":
    raise SystemExit(main())
