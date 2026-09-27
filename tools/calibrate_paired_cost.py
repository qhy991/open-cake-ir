#!/usr/bin/env python3
"""Admit and CPU-compile a frozen B300 GEMM paired-cost candidate pool.

GPU Infra may call collect-compile only as a local stage. Broker-owned device
collection and fitting will use the same plan; neither action here measures a GPU.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
import re
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler  # noqa: E402
from open_cake_ir.evaluation import EvaluationReceipt  # noqa: E402
from open_cake_ir.evaluation.paired import PAIRED_KIND, candidate_identity, paired_protocol  # noqa: E402
from open_cake_ir.lab.build import TritonToolchainBuilder  # noqa: E402
from open_cake_ir.lab.bindings import load_baseline_bundle, load_compiler_reference  # noqa: E402
from open_cake_ir.lab.environments import CandidateSubmission  # noqa: E402
from open_cake_ir.lab.executor import ExecutorRevision, _external_file  # noqa: E402
from open_cake_ir.lab.selection import _paired_empirical_context  # noqa: E402
from open_cake_ir.lab.process import sanitized_environment  # noqa: E402
from open_cake_ir.lab.triton_build import IsolatedTritonCompiler  # noqa: E402
from open_cake_ir.lab.paired_cost_calibration import derive_paired_cost_model, observed_paired_cost  # noqa: E402
from open_cake_ir.serialization import canonical_json_bytes  # noqa: E402
from open_cake_ir.tasks.environments import TaskOpenCakeEnvironment  # noqa: E402
from open_cake_ir.tasks.launch import parse_launch_manifest  # noqa: E402
from open_cake_ir.tasks.workloads import load_workload  # noqa: E402


class CorrectnessRejected(ValueError):
    """The external oracle rejected one sealed paired candidate."""


def _external(path: str | Path) -> Path:
    path = Path(path).absolute()
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("paired calibration external path contains a symlink")
    resolved = path.resolve(strict=True)
    if (resolved == ROOT or ROOT in resolved.parents
            or any((parent / ".git").exists() for parent in (resolved, *resolved.parents))):
        raise ValueError("paired calibration artifacts must remain outside every checkout")
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
              "varying_dimensions", "acceptance", "toolchain_identity",
              "toolchain_config_path"}
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
    toolchain = plan["toolchain_identity"]
    if (not isinstance(toolchain, dict) or toolchain.get("kind") != "bubblewrap_triton_kernel_v1"
            or toolchain.get("python") != executor.document["host_environment"]["python"]["invocation_path"]
            or toolchain.get("triton_version") != executor.document["host_environment"]["packages"]["triton"]):
        raise ValueError("paired cost toolchain differs from Executor")
    if not isinstance(plan["toolchain_config_path"], str):
        raise ValueError("paired cost toolchain configuration path differs")
    toolchain_config = _read(_external_file(snapshot, plan["toolchain_config_path"], "paired cost toolchain config"))
    expected_config = {"python", "bubblewrap", "runtime_roots", "build_environment",
                       "triton_version", "timeout_seconds"}
    if (not isinstance(toolchain_config, dict)
            or not expected_config <= set(toolchain_config) <= expected_config | {"pointer_alignment"}
            or toolchain_config["python"] != toolchain["python"]
            or toolchain_config["triton_version"] != toolchain["triton_version"]):
        raise ValueError("paired cost toolchain configuration differs from frozen identity")
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


def _write_new(path: Path, value: object) -> None:
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(value))


def prepare_compile(snapshot: str | Path, stage: str | Path, isolated_compiler) -> dict[str, object]:
    """CPU-only build of every frozen Schedule through the Lab's isolated builder."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("GPUQ_JOB_ID"):
        raise ValueError("paired cost compile must not inherit a GPU allocation")
    checked = check_plan(snapshot)
    snapshot, stage = _external(snapshot), _external(stage)
    if not stage.is_dir() or (stage / "compile-index.json").exists():
        raise ValueError("paired cost compile stage already has candidate output")
    plan = _read(_external_file(snapshot, "plan.json", "paired cost plan"))
    if any((stage / spec["id"]).exists() for spec in plan["candidates"]):
        raise ValueError("paired cost compile stage already has candidate output")
    if canonical_json_bytes(isolated_compiler.identity) != canonical_json_bytes(plan["toolchain_identity"]):
        raise ValueError("paired cost isolated toolchain differs from frozen plan")
    executor = ExecutorRevision.load_reference(ROOT, plan["executor_revision"], "paired_cost.executor_revision")
    executor.admit_host()
    isolated_compiler.check_executor(executor, author_workspace=snapshot)
    compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
    workload = load_workload(_external_file(ROOT, plan["workload"]["path"], "paired cost Workload"))
    study = _read(_external_file(ROOT, plan["study_path"], "paired cost Study"))
    builder = TritonToolchainBuilder(
        workload=workload, case_id=plan["case_id"], isolated_compiler=isolated_compiler,
    )
    environment = TaskOpenCakeEnvironment(
        compiler, builder,
        authority_document={"lowering_route": study["arms"]["open_cake"]["lowering_route"]},
        workload=workload, case_id=plan["case_id"], executor=executor,
    )
    rows = []
    for spec in plan["candidates"]:
        source = _external_file(snapshot, spec["schedule"], "paired cost Schedule").read_bytes()
        assessment = compiler.assess(json.loads(source))
        lowering = compiler.lower(assessment)
        built = environment.build(CandidateSubmission.seal(environment.media_type, source))
        if built.disposition != "launchable" or built.launchable is None:
            raise ValueError(f"paired cost candidate {spec['id']} build refused: {built.feedback}")
        candidate = built.launchable
        if candidate.artifact_payloads.get("lowered_source") != lowering.source.encode():
            raise ValueError("paired cost compiled source differs from frozen Schedule")
        directory = stage / spec["id"]
        directory.mkdir()
        with (directory / "schedule.json").open("xb") as stream:
            stream.write(assessment.schedule_bytes)
        paths = {}
        for role, payload in candidate.artifact_payloads.items():
            if not role.isidentifier():
                raise ValueError("paired cost artifact role cannot be a file name")
            filename = f"candidate-{role}.bin"
            with (directory / filename).open("xb") as stream:
                stream.write(payload)
            paths[role] = filename
        _write_new(directory / "candidate.json", {
            "candidate": candidate_identity(candidate), "artifact_paths": paths,
        })
        rows.append({"candidate_id": spec["id"], "schedule": spec["schedule"],
                     "candidate": candidate_identity(candidate)})
    index = {"plan_id": plan["plan_id"], "source_commit": compiler.commit,
             "context": checked["context"], "toolchain_identity": plan["toolchain_identity"],
             "candidates": rows}
    _write_new(stage / "compile-index.json", index)
    return index


def collect_compile() -> int:
    """GPU Infra local-stage entry; never asks for or inherits a GPU lease."""
    stage = _external(os.environ["KERNELINFRA_STAGE_DIR"])
    result_path = Path(os.environ["KERNELINFRA_RESULT"])
    if result_path.is_symlink():
        raise ValueError("paired cost stage result cannot be a symlink")
    result = result_path.resolve(strict=False)
    if (os.environ.get("KERNELINFRA_STAGE_KIND") != "compile"
            or os.environ.get("KERNELINFRA_STAGE_ID") != "compile"
            or result != stage / "result.json"):
        raise ValueError("paired cost GPU Infra local-stage binding differs")
    snapshot = _external(os.environ["KERNELINFRA_CANDIDATE_DIR"])
    try:
        plan = _read(_external_file(snapshot, "plan.json", "paired cost plan"))
        config = _read(_external_file(snapshot, plan["toolchain_config_path"], "paired cost toolchain config"))
        isolated = IsolatedTritonCompiler(**config)
        index = prepare_compile(snapshot, stage, isolated)
        artifacts = {path.relative_to(stage).as_posix(): path.relative_to(stage).as_posix()
                     for path in sorted(stage.rglob("*")) if path.is_file() and path != result}
        _write_new(result, {"schema": "kernelinfra.stage-result.v1", "status": "passed",
                            "validity": "valid", "summary": "paired cost CPU compilation passed",
                            "workloads": [], "artifacts": artifacts,
                            "metrics": {"candidate_count": len(index["candidates"])}})
        return 0
    except Exception as error:
        if not result.exists():
            _write_new(result, {"schema": "kernelinfra.stage-result.v1", "status": "failed",
                                "validity": "unknown", "summary": f"{type(error).__name__}: {error}",
                                "workloads": [], "artifacts": {}, "metrics": {}})
        return 1


def _task(run: Path, plan: dict, executor: ExecutorRevision, workload) -> dict:
    """Bind this run's two stages to one frozen source and Executor."""
    task = _read(_external_file(run, "task.json", "paired cost GPU Infra task"))
    if (task.get("schema") != "kernelinfra.task.v1"
            or task.get("task_id") != plan["plan_id"]
            or task.get("workloads") != [workload.workload_id]
            or not isinstance(task.get("stages"), list)
            or len(task["stages"]) != 2):
        raise ValueError("paired cost GPU Infra task differs")
    stages = task["stages"]
    if [(item.get("id"), item.get("kind"), item.get("execution", "broker")) for item in stages] != [
        ("compile", "compile", "local"), ("collection", "judge", "broker")
    ] or "resources" in stages[0]:
        raise ValueError("paired cost stage order or CPU/GPU resource boundary differs")
    python = executor.document["host_environment"]["python"]["invocation_path"]
    for stage, action in zip(stages, ("collect-compile", "collect-device"), strict=True):
        judge = stage.get("judge")
        if (not isinstance(judge, dict)
                or judge.get("identity") != executor.executor_id
                or judge.get("cwd") != str(ROOT)
                or judge.get("command") != [python, str(ROOT / "tools/calibrate_paired_cost.py"), action]):
            raise ValueError("paired cost judge source or Executor differs")
    resources = stages[1].get("resources")
    if (not isinstance(resources, dict) or resources.get("mode") != "exclusive"
            or type(resources.get("gpu_count")) is not int or resources["gpu_count"] != 1
            or type(resources.get("run_timeout_s")) not in (int, float)
            or not math.isfinite(resources["run_timeout_s"]) or resources["run_timeout_s"] <= 2):
        raise ValueError("paired cost collection requires one exclusive broker GPU")
    return task


def _broker_parent() -> tuple[int, int, int]:
    """The collection controller must be the direct child of the owning broker."""
    if sys.platform != "linux":
        raise ValueError("paired cost broker admission requires Linux")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(10)
        connection.connect("/tmp/agent-gpu-broker.sock")
        peer = struct.unpack("3i", connection.getsockopt(
            socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
    if os.getppid() != peer[0] or os.geteuid() != peer[1]:
        raise ValueError("paired cost stage is not the broker's direct child")
    return peer


def _node_assignment(run: Path, task: dict, *, timeout_s: float = 2) -> dict:
    """Read the node-owned broker job; a missing child env var is not guessed."""
    request = _read(_external_file(run, "request.json", "paired cost node request"))
    run_id = os.environ.get("KERNELINFRA_RUN_ID")
    if (not isinstance(run_id, str) or not run_id
            or request.get("schema") != "kernelinfra.request.v1"
            or request.get("run_id") != run_id
            or request.get("task_id") != task["task_id"]
            or any(not isinstance(request.get(key), str)
                   or re.fullmatch(r"[0-9a-f]{64}", request[key]) is None
                   for key in ("task_sha256", "candidate_sha256"))):
        raise ValueError("paired cost node request identity differs")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not isinstance(visible, str) or re.fullmatch(r"[0-9]+", visible) is None:
        raise ValueError("paired cost requires one broker-visible CUDA device")
    deadline = time.monotonic() + timeout_s
    while True:
        state = _read(_external_file(run, "state.json", "paired cost node state"))
        expected = {"schema": "kernelinfra.state.v1", "run_id": run_id,
                    "task_id": task["task_id"], "task_sha256": request["task_sha256"],
                    "candidate_sha256": request["candidate_sha256"],
                    "stage_id": "collection", "stage_kind": "judge", "stage_index": 1}
        if (any(state.get(key) != value for key, value in expected.items())
                or state.get("run_dir") != str(run)
                or state.get("terminal_at") is not None):
            raise ValueError("paired cost node state belongs to another run or stage")
        job_id = state.get("broker_job_id")
        if (state.get("state") == "running"
                and isinstance(job_id, str)
                and re.fullmatch(r"gpuq-[0-9a-f]{12}", job_id) is not None
                and job_id != "gpuq-000000000000"
                and state.get("gpu_ids") == [int(visible)]):
            exported = os.environ.get("GPUQ_JOB_ID")
            if exported is not None and exported != job_id:
                raise ValueError("paired cost exported broker job differs from node assignment")
            return {"run_id": run_id, "broker_job_id": job_id,
                    "physical_gpu": int(visible), "node_state": expected}
        if (state.get("state") not in {"submitting", "queued"}
                or state.get("gpu_ids") != [] or time.monotonic() >= deadline):
            raise ValueError("paired cost node has no matching running broker assignment")
        time.sleep(.05)


def _compiled_candidates(run: Path, plan: dict, checked: dict, compiler: Compiler, workload) -> dict:
    """Replay every compile seal against the frozen Schedule and current Compiler."""
    stage = run / "stages/compile"
    result = _read(_external_file(stage, "result.json", "paired cost compile stage result"))
    receipt = _read(_external_file(stage, "receipt.json", "paired cost compile stage receipt"))
    if (result.get("schema") != "kernelinfra.stage-result.v1"
            or result.get("status") != "passed" or result.get("validity") != "valid"
            or receipt.get("execution") != "local" or receipt.get("exit_code") != 0
            or receipt.get("judge_result_valid") is not True):
        raise ValueError("paired cost CPU compile stage did not pass")
    index = _read(_external_file(stage, "compile-index.json", "paired cost compile index"))
    if (index.get("plan_id") != plan["plan_id"]
            or index.get("source_commit") != compiler.commit
            or index.get("context") != checked["context"]
            or index.get("toolchain_identity") != plan["toolchain_identity"]
            or not isinstance(index.get("candidates"), list)
            or len(index["candidates"]) != len(plan["candidates"])):
        raise ValueError("paired cost compile index differs from frozen plan")
    admitted = {}
    for spec, row in zip(plan["candidates"], index["candidates"], strict=True):
        source = _external_file(run / "candidate", spec["schedule"], "frozen paired Schedule").read_bytes()
        assessment = compiler.assess(json.loads(source))
        lowering = compiler.lower(assessment)
        if (row.get("candidate_id") != spec["id"] or row.get("schedule") != spec["schedule"]
                or _external_file(stage / spec["id"], "schedule.json", "compiled Schedule").read_bytes()
                != assessment.schedule_bytes):
            raise ValueError("paired cost compiled Schedule differs")
        candidate = load_baseline_bundle(ROOT, stage / spec["id"] / "candidate.json")
        expected_submission = CandidateSubmission.seal(TaskOpenCakeEnvironment.media_type, source)
        manifest = parse_launch_manifest(json.loads(candidate.artifact_payloads["launch_manifest"]))
        manifest.check_workload(workload, plan["case_id"])
        requirements = lowering.toolchain_requirements
        if (row.get("candidate") != candidate_identity(candidate)
                or candidate.candidate_sha256 != expected_submission.sha256
                or candidate.target != plan["target"]
                or candidate.artifact_payloads.get("lowered_source") != lowering.source.encode()
                or candidate.entry_point != requirements["kernel_entry_point"]
                or candidate.launch_spec_sha256 != manifest.canonical_sha256
                or list(manifest.grid) != requirements["grid"]):
            raise ValueError("paired cost compiled artifact differs from frozen lowering")
        admitted[spec["id"]] = candidate
    return admitted


def _paired_request(candidate, baseline, plan: dict, policy: dict) -> tuple[dict, dict[str, bytes]]:
    """Seal both participants in the common evaluator's existing request format."""
    payloads = {}
    candidate_paths = {}
    baseline_paths = {}
    for role, payload in candidate.artifact_payloads.items():
        if not role.isidentifier():
            raise ValueError("paired cost candidate artifact role differs")
        filename = f"candidate-{role}.bin"
        candidate_paths[role] = filename
        payloads[filename] = payload
    for role, payload in baseline.artifact_payloads.items():
        if not role.isidentifier():
            raise ValueError("paired cost baseline artifact role differs")
        filename = f"baseline-{role}.bin"
        baseline_paths[role] = filename
        payloads[filename] = payload
    request = {**candidate_identity(candidate), "artifact_paths": candidate_paths,
               "executor_revision": plan["executor_revision"],
               "compiler_revision": plan["compiler_revision"],
               "workload_path": str(ROOT / plan["workload"]["path"]),
               "workload_sha256": plan["workload"]["canonical_sha256"],
               "case_id": plan["case_id"], "purpose": "search", "attempt": 1,
               "evaluation_protocol": policy,
               "evaluation_protocol_sha256": sha256(canonical_json_bytes(policy)).hexdigest(),
               "baseline": {**candidate_identity(baseline), "artifact_paths": baseline_paths}}
    return request, payloads


def _seal_observation(directory: Path, candidate, baseline, plan: dict, policy: dict) -> dict:
    directory.mkdir()
    request, payloads = _paired_request(candidate, baseline, plan, policy)
    for name, payload in payloads.items():
        with (directory / name).open("xb") as stream:
            stream.write(payload)
    _write_new(directory / "request.json", request)
    return request


def _observed_evaluation(directory: Path, spec: dict, plan: dict, policy: dict,
                         candidate, baseline, job_id: str, schedule: dict) -> dict:
    """Replay the worker's raw paired receipt, including exact participant seals."""
    from open_cake_ir.tasks import evaluate as worker

    authority = worker._load_authority(_external_file(directory, "request.json", "paired evaluator request"))
    expected_request, _ = _paired_request(candidate, baseline, plan, policy)
    if (authority.candidate.canonical_sha256 != candidate.canonical_sha256
            or authority.baseline is None
            or authority.baseline.canonical_sha256 != baseline.canonical_sha256
            or canonical_json_bytes(authority.request) != canonical_json_bytes(expected_request)):
        raise ValueError("paired evaluator request differs from frozen participants")
    result = _read(_external_file(directory, "result.json", "paired evaluator result"))
    if (result.get("schema_version") != 1 or result.get("job_id") != job_id
            or result.get("mode") != "exclusive" or result.get("admitted") is not True
            or result.get("error") is not None or result.get("failure_class") is not None):
        raise ValueError("paired evaluator admission or completion differs")
    value = result.get("receipt")
    names = {"correctness_output": "correctness-output.json",
             "launch_receipt": "launch-receipt.json",
             "timing_samples": "timing-samples.json"}
    if not isinstance(value, dict) or value.get("artifacts") != names:
        raise ValueError("paired evaluator artifact roles differ")
    payloads = {role: _external_file(directory, name, f"paired {role}").read_bytes()
                for role, name in names.items()}
    receipt = EvaluationReceipt(
        candidate_sha256=candidate.candidate_sha256,
        workload_sha256=plan["workload"]["canonical_sha256"],
        evaluation_protocol_sha256=sha256(canonical_json_bytes(policy)).hexdigest(),
        purpose="search", case_id=plan["case_id"],
        correctness_passed=value["correctness_passed"] is True,
        correctness=value["correctness"], kernel_calls=value["kernel_calls"],
        fallback_calls=value["fallback_calls"],
        launch_receipt_sha256=sha256(payloads["launch_receipt"]).hexdigest(),
        timing=value["timing"], artifact_payloads=payloads,
    )
    if not receipt.correctness_passed:
        raise CorrectnessRejected(f"paired external oracle rejected {spec['id']}")
    observed = observed_paired_cost(
        receipt, candidate=candidate, baseline=baseline,
        evaluation_protocol=policy, worker_result=result,
    )
    return {"candidate_id": spec["candidate_id"], "split": spec["split"],
            "schedule": schedule, "observation": observed}


def collect_device() -> int:
    """GPU Infra broker-stage controller; the common evaluator owns device work."""
    run = _external(os.environ["KERNELINFRA_RUN_DIR"])
    stage = _external(os.environ["KERNELINFRA_STAGE_DIR"])
    snapshot = _external(os.environ["KERNELINFRA_CANDIDATE_DIR"])
    result_path = Path(os.environ["KERNELINFRA_RESULT"])
    if result_path.is_symlink():
        raise ValueError("paired cost stage result cannot be a symlink")
    result = result_path.resolve(strict=False)
    if (os.environ.get("KERNELINFRA_STAGE_KIND") != "judge"
            or os.environ.get("KERNELINFRA_STAGE_ID") != "collection"
            or stage != run / "stages/collection" or snapshot != run / "candidate"
            or result != stage / "result.json"
            or _external(os.environ["KERNELINFRA_TASK"]) != run / "task.json"):
        raise ValueError("paired cost GPU Infra broker-stage binding differs")
    try:
        checked = check_plan(snapshot)
        plan = _read(_external_file(snapshot, "plan.json", "paired cost plan"))
        compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        executor = ExecutorRevision.load_reference(ROOT, plan["executor_revision"],
                                                   "paired_cost.executor_revision")
        workload = load_workload(_external_file(ROOT, plan["workload"]["path"],
                                                "paired cost Workload"))
        task = _task(run, plan, executor, workload)
        compiled = _compiled_candidates(run, plan, checked, compiler, workload)
        baseline = load_baseline_bundle(ROOT, _external_file(snapshot, plan["baseline_bundle_path"],
                                                             "paired cost baseline bundle"))
        study = _read(_external_file(ROOT, plan["study_path"], "paired cost Study"))
        policy = study["evaluation_protocol"]
        executor.admit_host()
        peer = _broker_parent()
        assignment = _node_assignment(run, task)
        _write_new(stage / "execution-context.json", {**assignment, "broker_peer": list(peer),
                                                       "uid": os.geteuid()})
        deadline = time.monotonic() + task["stages"][1]["resources"]["run_timeout_s"] - 2
        rows = []
        for spec in plan["observations"]:
            candidate = compiled[spec["candidate_id"]]
            directory = stage / spec["id"]
            _seal_observation(directory, candidate, baseline, plan, policy)
            command = [executor.document["host_environment"]["python"]["invocation_path"],
                       str(ROOT / "src/open_cake_ir/tasks/evaluate.py"),
                       "--request", str(directory / "request.json"),
                       "--output", str(directory / "result.json")]
            remaining = int(deadline - time.monotonic())
            if remaining <= 0:
                raise TimeoutError("paired cost collection deadline exhausted")
            environment = sanitized_environment()
            environment.update(GPUQ_JOB_ID=assignment["broker_job_id"], GPUQ_BACKEND="nvidia",
                               GPUQ_MODE="exclusive", GPUQ_DEVICE_IDS=str(assignment["physical_gpu"]),
                               GPUQ_OCCUPANCY_SCOPE="system",
                               CUDA_VISIBLE_DEVICES=str(assignment["physical_gpu"]))
            with (directory / "stdout.log").open("xb") as stdout, \
                    (directory / "stderr.log").open("xb") as stderr:
                completed = subprocess.run(command, cwd=ROOT, env=environment,
                                           timeout=remaining, stdin=subprocess.DEVNULL,
                                           stdout=stdout, stderr=stderr, check=False)
            if completed.returncode:
                raise ValueError(f"paired evaluator exited {completed.returncode}")
            schedule = _read(_external_file(snapshot,
                                            next(row["schedule"] for row in plan["candidates"]
                                                 if row["id"] == spec["candidate_id"]),
                                            "paired frozen Schedule"))
            row = _observed_evaluation(directory, spec, plan, policy, candidate, baseline,
                                       assignment["broker_job_id"], schedule)
            rows.append(row)
            _write_new(directory / "observation.json", row)
            baseline_us = [item["observation"]["baseline_us"] for item in rows]
            if max(baseline_us) / min(baseline_us) > plan["acceptance"]["maximum_baseline_drift_ratio"]:
                raise ValueError("paired cost baseline drift exceeded the frozen limit")
            if len({item["observation"]["gpu_uuid"] for item in rows}) != 1:
                raise ValueError("paired cost observations span multiple GPU UUIDs")
        _write_new(stage / "observations.json", {"rows": rows, "context": checked["context"],
                                                  "assignment": assignment})
        artifacts = {path.relative_to(stage).as_posix(): path.relative_to(stage).as_posix()
                     for path in sorted(stage.rglob("*")) if path.is_file() and path != result}
        _write_new(result, {"schema": "kernelinfra.stage-result.v1", "status": "passed",
                            "validity": "valid", "summary": "paired B300 cost collection passed",
                            "workloads": [{"id": workload.workload_id, "correct": True}],
                            "artifacts": artifacts,
                            "metrics": {"observation_count": len(rows),
                                        "scope": "paired cost calibration only; no Candidate promotion"}})
        return 0
    except Exception as error:
        if not result.exists():
            artifacts = {path.relative_to(stage).as_posix(): path.relative_to(stage).as_posix()
                         for path in sorted(stage.rglob("*")) if path.is_file() and path != result}
            _write_new(result, {"schema": "kernelinfra.stage-result.v1", "status": "failed",
                                "validity": "invalid" if isinstance(error, CorrectnessRejected) else "unknown",
                                "summary": f"{type(error).__name__}: {error}",
                                "workloads": [], "artifacts": artifacts, "metrics": {}})
        return 1


def _new_external(path: str | Path) -> Path:
    path = Path(path).absolute()
    if path.is_symlink():
        raise ValueError("paired cost derived output cannot be a symlink")
    parent = _external(path.parent)
    output = parent / path.name
    if output.exists():
        raise ValueError("paired cost derived output must be new")
    return output


def _completed_run(run: Path, task: dict, executor: ExecutorRevision) -> tuple[dict, dict]:
    result = _read(_external_file(run, "result.json", "paired cost run result"))
    if (result.get("schema") != "kernelinfra.run-result.v1"
            or result.get("outcome") != "completed" or result.get("validity") != "valid"
            or result.get("task_id") != task["task_id"]
            or result.get("frontier_eligible") is not False
            or not isinstance(result.get("run_id"), str) or not result["run_id"]
            or [(row.get("id"), row.get("kind"), row.get("status"), row.get("validity"))
                for row in result.get("stages", [])] != [
                    ("compile", "compile", "passed", "valid"),
                    ("collection", "judge", "passed", "valid")]):
        raise ValueError("paired cost run is not terminal, valid calibration evidence")
    state = _read(_external_file(run, "state.json", "paired cost terminal state"))
    request = _read(_external_file(run, "request.json", "paired cost node request"))
    if (any(result.get(key) != request.get(key) or state.get(key) != request.get(key)
            for key in ("run_id", "task_id", "task_sha256", "candidate_sha256"))
            or state.get("run_id") != result["run_id"] or state.get("state") != "completed"
            or state.get("terminal_at") is None):
        raise ValueError("paired cost node has not reported terminal completion")
    receipts = {}
    for stage_spec in task["stages"]:
        name = stage_spec["id"]
        stage = run / "stages" / name
        receipt = _read(_external_file(stage, "receipt.json", "paired cost stage receipt"))
        expected = {"schema": "kernelinfra.stage-receipt.v1", "run_id": result["run_id"],
                    "stage_id": name, "stage_kind": stage_spec["kind"],
                    "execution": stage_spec["execution"],
                    "judge_identity": executor.executor_id,
                    "exit_code": 0, "judge_result_valid": True, "error": None}
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError(f"paired cost {name} stage receipt differs")
        stage_result = _read(_external_file(stage, "result.json", "paired cost stage result"))
        if (stage_result.get("schema") != "kernelinfra.stage-result.v1"
                or stage_result.get("status") != "passed"
                or stage_result.get("validity") != "valid"
                or not isinstance(stage_result.get("artifacts"), dict)):
            raise ValueError(f"paired cost {name} stage result differs")
        for relative in stage_result["artifacts"].values():
            _external_file(stage, relative, "paired retained stage artifact")
        receipts[name] = receipt
    ids = receipts["collection"].get("gpu_ids")
    if (not isinstance(ids, list) or len(ids) != 1 or type(ids[0]) is not int
            or ids[0] < 0
            or not isinstance(receipts["collection"].get("broker_job_id"), str)
            or re.fullmatch(r"gpuq-[0-9a-f]{12}", receipts["collection"]["broker_job_id"]) is None):
        raise ValueError("paired cost collection receipt names no exclusive broker allocation")
    return result, receipts


def fit_run(run: str | Path, output: str | Path) -> int:
    """Replay a terminal Run, then publish only a held-out-qualified model."""
    output = _new_external(output)
    output.mkdir()
    try:
        run = _external(run)
        snapshot = run / "candidate"
        checked = check_plan(snapshot)
        plan = _read(_external_file(snapshot, "plan.json", "paired cost plan"))
        executor = ExecutorRevision.load_reference(ROOT, plan["executor_revision"],
                                                   "paired_cost.executor_revision")
        compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        workload = load_workload(_external_file(ROOT, plan["workload"]["path"],
                                                "paired cost Workload"))
        task = _task(run, plan, executor, workload)
        run_result, receipts = _completed_run(run, task, executor)
        compiled = _compiled_candidates(run, plan, checked, compiler, workload)
        baseline = load_baseline_bundle(ROOT, _external_file(snapshot, plan["baseline_bundle_path"],
                                                             "paired cost baseline bundle"))
        study = _read(_external_file(ROOT, plan["study_path"], "paired cost Study"))
        policy = study["evaluation_protocol"]
        context = _read(_external_file(run / "stages/collection", "execution-context.json",
                                       "paired cost retained broker assignment"))
        request = _read(_external_file(run, "request.json", "paired cost node request"))
        assignment = {key: context[key] for key in ("run_id", "broker_job_id", "physical_gpu", "node_state")}
        expected_node = {"schema": "kernelinfra.state.v1", "run_id": run_result["run_id"],
                         "task_id": task["task_id"],
                         "task_sha256": request["task_sha256"],
                         "candidate_sha256": request["candidate_sha256"],
                         "stage_id": "collection", "stage_kind": "judge", "stage_index": 1}
        peer = context.get("broker_peer")
        if (assignment["run_id"] != run_result["run_id"]
                or assignment["broker_job_id"] != receipts["collection"]["broker_job_id"]
                or assignment["physical_gpu"] != receipts["collection"]["gpu_ids"][0]
                or assignment["node_state"] != expected_node
                or not isinstance(peer, list) or len(peer) != 3
                or any(type(item) is not int for item in peer)
                or context.get("uid") != peer[1]):
            raise ValueError("paired cost retained broker assignment differs from node receipts")
        rows = []
        for spec in plan["observations"]:
            candidate = compiled[spec["candidate_id"]]
            schedule = _read(_external_file(snapshot,
                                            next(item["schedule"] for item in plan["candidates"]
                                                 if item["id"] == spec["candidate_id"]),
                                            "paired frozen Schedule"))
            directory = run / "stages/collection" / spec["id"]
            row = _observed_evaluation(directory, spec, plan, policy, candidate, baseline,
                                       assignment["broker_job_id"], schedule)
            if _read(_external_file(directory, "observation.json", "retained paired observation")) != row:
                raise ValueError("paired cost retained observation differs from raw replay")
            rows.append(row)
        index = _read(_external_file(run / "stages/collection", "observations.json",
                                     "paired cost observation index"))
        if index != {"rows": rows, "context": checked["context"], "assignment": assignment}:
            raise ValueError("paired cost observation index differs from raw replay")
        passed, model, audit = derive_paired_cost_model(
            model_id=plan["model_id"], compiler_revision_id=checked["compiler_revision_id"],
            target=plan["target"], executor=executor,
            workload_sha256=workload.canonical_sha256, case_id=plan["case_id"],
            evaluation_protocol=policy, baseline=baseline, rows=rows,
            varying_dimensions=plan["varying_dimensions"], acceptance=plan["acceptance"],
        )
        model["reported_evidence"].update(run_id=run_result["run_id"], plan_id=plan["plan_id"])
        _write_new(output / "audit.json", {**audit, "run_id": run_result["run_id"]})
        if passed:
            _write_new(output / "model.json", model)
        return 0 if passed else 1
    except Exception as error:
        if not (output / "audit.json").exists():
            _write_new(output / "audit.json", {"passed": False,
                                               "failure_class": type(error).__name__,
                                               "reason": str(error)})
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check-plan", "collect-compile", "collect-device", "fit"))
    parser.add_argument("snapshot", type=Path, nargs="?")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "collect-compile":
        if args.snapshot is not None:
            parser.error("collect-compile reads the GPU Infra candidate snapshot from its environment")
        return collect_compile()
    if args.action == "collect-device":
        if args.snapshot is not None:
            parser.error("collect-device reads the GPU Infra run from its environment")
        return collect_device()
    if args.action == "fit":
        if args.snapshot is None or args.output is None:
            parser.error("fit requires a terminal Run and a new --output directory")
        return fit_run(args.snapshot, args.output)
    if args.snapshot is None:
        parser.error("check-plan requires an external candidate snapshot")
    print(json.dumps(check_plan(args.snapshot), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
