#!/usr/bin/env python3
"""Install exact local source and submit the two QSA seed arms through GPU Infra."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler  # noqa: E402
from open_cake_ir.evaluation import ProgramContract, WorkloadContract  # noqa: E402
from open_cake_ir.lab import ExecutorRevision  # noqa: E402

_RUNTIME_PATH = ROOT / "runtime/qsa-seed-gpu-infra-verda-v1.json"
_PROGRAM_PATH = ROOT / "contracts/programs/qsa-prefill-t32768-v2.json"
_WORKLOAD_PATH = ROOT / "contracts/workloads/qsa-prefill-t32768-v1.json"
_DIRECT_SOURCE = ROOT / "src/open_cake_ir/evaluation/assets/qsa_direct_reference_v1.cu"
_DIRECT_MANIFEST = ROOT / "src/open_cake_ir/evaluation/assets/qsa_direct_reference_v1.json"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _write_new(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_bytes(_canonical_json_bytes(value) + b"\n")


def _run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(command, check=True, **kwargs)


def _git_commit() -> str:
    commit = _run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True
    ).stdout.decode().strip()
    status = _run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True
    ).stdout
    if status:
        raise ValueError("QSA launch requires a clean committed local worktree")
    return commit


def _current_executor() -> ExecutorRevision:
    inventory = json.loads(
        (ROOT / "inventory/EXECUTOR_REVISIONS.json").read_text(encoding="utf-8")
    )
    current = inventory["current"]
    executor = ExecutorRevision.load(ROOT, ROOT / current["path"])
    if executor.canonical_sha256 != current["canonical_sha256"]:
        raise ValueError("current Executor inventory differs")
    return executor


def _preflight_authorities() -> tuple[ExecutorRevision, ProgramContract]:
    compiler = Compiler.load(ROOT, ROOT / "compiler/revision.lock.json")
    gate = compiler.check_corpus()
    if compiler.state != "released" or not gate.passed:
        raise ValueError("released Compiler Corpus Gate is not current")
    program = ProgramContract.load(ROOT, _PROGRAM_PATH, compiler)
    workload = WorkloadContract.load(_WORKLOAD_PATH)
    if program.workload.canonical_sha256 != workload.canonical_sha256:
        raise ValueError("QSA Program and Workload differ")
    return _current_executor(), program


def _materialize_candidates(root: Path, program: ProgramContract) -> tuple[Path, Path]:
    candidates = root / "candidates"
    candidates.mkdir()
    cake = candidates / "open-cake-seed"
    direct = candidates / "direct-cuda-seed"
    cake.mkdir()
    direct.mkdir()
    nodes: list[dict[str, str]] = []
    for node in program.nodes:
        destination = cake / f"{node.node_id}.json"
        shutil.copyfile(node.schedule_path, destination)
        nodes.append({"id": node.node_id, "schedule": destination.name})
    _write_new(
        cake / "candidate.json",
        {"schema_version": 1, "arm": "open_cake", "nodes": nodes},
    )
    shutil.copyfile(_DIRECT_SOURCE, direct / "program.cu")
    shutil.copyfile(_DIRECT_MANIFEST, direct / "launch.json")
    _write_new(
        direct / "candidate.json",
        {
            "schema_version": 1,
            "arm": "direct_cuda",
            "source": "program.cu",
            "launch_manifest": "launch.json",
        },
    )
    return cake, direct


def _task(
    *,
    remote_root: str,
    runtime: dict[str, object],
    executor: ExecutorRevision,
    protocol: str,
    component_timing: bool,
    profile_kernel: str,
) -> dict[str, object]:
    judge = runtime["judge"]
    assert isinstance(judge, dict)
    command = [
        str(judge["python"]),
        f"{remote_root}/tools/evaluate_qsa_candidate.py",
        "--project-root",
        remote_root,
        "--nvcc",
        str(judge["nvcc"]),
        "--cuobjdump",
        str(judge["cuobjdump"]),
    ]
    if component_timing:
        if protocol != "seed":
            raise ValueError("QSA component timing requires the seed protocol")
        command.append("--component-timing")
    if profile_kernel != "score_topk":
        if protocol != "profile":
            raise ValueError("QSA profile-kernel selection requires the profile protocol")
        command.extend(("--profile-kernel", profile_kernel))
    identity = f"{executor.executor_id}@{executor.canonical_sha256}"
    stages: list[dict[str, object]] = [
        {
            "id": "compile",
            "kind": "compile",
            "execution": "local",
            "judge": {"identity": identity, "cwd": remote_root, "command": command},
        },
        {
            "id": "correctness",
            "kind": "correctness",
            "judge": {"identity": identity, "cwd": remote_root, "command": command},
            "resources": {
                "mode": "exclusive",
                "gpu_count": 1,
                "estimate_s": 300,
                "queue_timeout_s": 1800,
                "run_timeout_s": 1800,
            },
        },
    ]
    if protocol == "seed":
        stages.append(
            {
                "id": "benchmark",
                "kind": "benchmark",
                "judge": {
                    "identity": identity,
                    "cwd": remote_root,
                    "command": command,
                },
                "resources": {
                    "mode": "exclusive",
                    "gpu_count": 1,
                    "estimate_s": 900,
                    "queue_timeout_s": 1800,
                    "run_timeout_s": 3600,
                },
            }
        )
    elif protocol == "profile":
        stages.append(
            {
                "id": "profile",
                "kind": "profile",
                "judge": {
                    "identity": identity,
                    "cwd": remote_root,
                    "command": command,
                },
                "resources": {
                    "mode": "exclusive",
                    "gpu_count": 1,
                    "estimate_s": 900,
                    "queue_timeout_s": 1800,
                    "run_timeout_s": 1800,
                },
            }
        )
    else:
        raise ValueError("QSA execution protocol differs")
    return {
        "schema": "kernelinfra.task.v1",
        "task_id": (
            "open-cake-qsa-prefill-t32768-seed-component-v1"
            if component_timing
            else (
                f"open-cake-qsa-prefill-t32768-profile-{profile_kernel}-v1"
                if protocol == "profile" and profile_kernel != "score_topk"
                else f"open-cake-qsa-prefill-t32768-{protocol}-v1"
            )
        ),
        "description": (
            "QSA target_t32768 seed-path qualification; candidate versus one hidden "
            "fixed direct-CUDA reference with external FP32 oracle and CUPTI cold-L2 timing."
        ),
        "workloads": ["qsa-prefill-t32768"],
        "comparison": {
            "primary_workloads": ["qsa-prefill-t32768"],
            "guardrails": {"qsa-prefill-t32768": 0.5},
            "relative_noise_floor": 0.05,
        },
        "stages": stages,
    }


def _install_remote_source(ssh: str, remote_root: str, commit: str) -> None:
    quoted_root = shlex.quote(remote_root)
    _run(["ssh", "-o", "BatchMode=yes", ssh, f"mkdir {quoted_root}"])
    archive = subprocess.Popen(
        ["git", "archive", "--format=tar.gz", commit],
        cwd=ROOT,
        stdout=subprocess.PIPE,
    )
    assert archive.stdout is not None
    receiver = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", ssh, f"tar -xz -C {quoted_root}"],
        stdin=archive.stdout,
        check=False,
    )
    archive.stdout.close()
    archive_code = archive.wait()
    if archive_code != 0 or receiver.returncode != 0:
        raise RuntimeError("exact-source remote installation failed")


def _start_daemon(
    *,
    ssh: str,
    kernelctl: str,
    socket: str,
    state_root: str,
    runtime: dict[str, object],
) -> None:
    daemon = runtime["daemon"]
    assert isinstance(daemon, dict)
    service_group = str(daemon["service_group"])
    quoted = {name: shlex.quote(value) for name, value in {
        "kernelctl": kernelctl,
        "socket": socket,
        "state": state_root,
        "broker": str(daemon["broker_socket"]),
        "gpu_run": str(daemon["gpu_run"]),
        "log": f"{state_root}/daemon.log",
    }.items()}
    launch = (
        "umask 0007; "
        "export KERNELINFRA_RUN_DIR_MODE=2770; "
        "export KERNELINFRA_RUN_FILE_MODE=660; "
        f"setsid nohup {quoted['kernelctl']} serve --socket {quoted['socket']} "
        f"--state-dir {quoted['state']} --broker-socket {quoted['broker']} "
        f"--gpu-run {quoted['gpu_run']} --local-capacity {int(daemon['local_capacity'])} "
        f">>{quoted['log']} 2>&1 </dev/null &"
    )
    command = (
        f"mkdir -p {quoted['state']} && "
        f"chgrp {shlex.quote(service_group)} {quoted['state']} && "
        f"chmod 2770 {quoted['state']} && "
        f"sg {shlex.quote(service_group)} -c {shlex.quote(launch)}"
    )
    _run(["ssh", "-o", "BatchMode=yes", ssh, command])
    for _ in range(20):
        observed = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                ssh,
                kernelctl,
                "node-status",
                "--socket",
                socket,
                "--json",
            ],
            capture_output=True,
        )
        if observed.returncode == 0:
            status = json.loads(observed.stdout)
            broker = status.get("broker", {})
            if broker.get("probe_error") is None and status.get("active_runs") == []:
                return
        time.sleep(0.5)
    raise RuntimeError("isolated GPU Infra daemon did not become observable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--gpu-infra-root",
        type=Path,
        default=ROOT.parent.parent / "gpu-infra",
    )
    parser.add_argument(
        "--component-timing",
        action="store_true",
        help="append diagnostic ordered node timing after the primary seed benchmark",
    )
    parser.add_argument(
        "--protocol",
        choices=("seed", "profile"),
        default="seed",
        help="run seed timing or score_topk attribution after common correctness",
    )
    parser.add_argument(
        "--profile-kernel",
        default="score_topk",
        help="candidate Program kernel id selected by an explicit profile task",
    )
    parser.add_argument("--runtime", type=Path, default=_RUNTIME_PATH)
    parser.add_argument("--remote-project-root", required=True)
    parser.add_argument("--remote-state-root", required=True)
    parser.add_argument("--remote-socket", required=True)
    parser.add_argument("--remote-inbox", required=True)
    parser.add_argument(
        "--arm",
        choices=("both", "open_cake", "direct_cuda"),
        default="both",
        help="submit both matched seeds or one explicitly diagnostic arm",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    run_root = arguments.run_root.absolute()
    if run_root.exists() or run_root.is_symlink():
        raise FileExistsError(f"refusing to overwrite run root {run_root}")
    executor, program = _preflight_authorities()
    commit = _git_commit()
    runtime = json.loads(arguments.runtime.resolve(strict=True).read_text())
    node = runtime["node"]
    if runtime.get("schema_version") != 1 or not isinstance(node, dict):
        raise ValueError("QSA GPU Infra runtime binding differs")
    run_root.mkdir(parents=True)
    cake, direct = _materialize_candidates(run_root, program)
    task_path = run_root / "task.json"
    _write_new(
        task_path,
        _task(
            remote_root=arguments.remote_project_root,
            runtime=runtime,
            executor=executor,
            protocol=arguments.protocol,
            component_timing=arguments.component_timing,
            profile_kernel=arguments.profile_kernel,
        ),
    )
    catalog_path = run_root / "catalog.json"
    _write_new(
        catalog_path,
        {
            "schema": "kernelinfra.fleet.v1",
            "connect_timeout_s": 8,
            "command_timeout_s": 120,
            "nodes": [
                {
                    "id": node["id"],
                    "ssh": node["ssh"],
                    "kernelctl": node["kernelctl"],
                    "socket": arguments.remote_socket,
                    "inbox": arguments.remote_inbox,
                    "capabilities": node["capabilities"],
                }
            ],
        },
    )
    kernelctl = arguments.gpu_infra_root.resolve(strict=True) / "bin/kernelctl"
    _run([str(kernelctl), "task-check", str(task_path)])
    _run([str(kernelctl), "fleet-check", str(catalog_path)])
    _install_remote_source(str(node["ssh"]), arguments.remote_project_root, commit)
    _start_daemon(
        ssh=str(node["ssh"]),
        kernelctl=str(node["kernelctl"]),
        socket=arguments.remote_socket,
        state_root=arguments.remote_state_root,
        runtime=runtime,
    )
    routes = run_root / "routes"
    selected_candidates = {
        "both": [cake, direct],
        "open_cake": [cake],
        "direct_cuda": [direct],
    }[arguments.arm]
    _run(
        [
            str(kernelctl),
            "fleet-submit-many",
            "--catalog",
            str(catalog_path),
            "--require",
            "b200",
            "--label-prefix",
            "qsa-seed-",
            "--route-dir",
            str(routes),
            str(task_path),
            *(str(candidate) for candidate in selected_candidates),
        ]
    )
    print(
        json.dumps(
            {
                "commit": commit,
                "executor_revision": executor.executor_id,
                "arm": arguments.arm,
                "protocol": arguments.protocol,
                "component_timing": arguments.component_timing,
                "profile_kernel": arguments.profile_kernel,
                "run_root": str(run_root),
                "routes": str(routes),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
