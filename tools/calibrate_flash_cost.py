#!/usr/bin/env python3
"""Collect a frozen Flash fixed-case pool through common Evaluation; fit it on CPU.

GPU Infra invokes ``collect`` for local compile and exclusive judge stages. Its
immutable candidate contains plan.json and four complete, seed-specialized
Schedules. The plan binds released Compiler/Executor/Workload references, the
case, ordered pool, baseline, observation order and acceptance thresholds.
``fit RUN --output NEW_DIRECTORY`` replays those bindings and raw observations.
Neither operation allocates a GPU, retries a measurement or promotes a Candidate.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from open_cake_ir.compiler import Compiler, EmpiricalCostModel
from open_cake_ir.evaluation import CudaLaunchManifest, LaunchableCandidate, WorkloadContract, summarize_cohort
from open_cake_ir.lab.core import _empirical_filter
from open_cake_ir.lab.environments import CandidateSubmission, OpenCakeEnvironment, TritonToolchainBuilder, _EmpiricalSelection, _empirical_context
from open_cake_ir.lab.executor import ExecutorRevision, _external_file
from open_cake_ir.lab.process import sanitized_environment


def _read(path):
    return json.loads(Path(path).read_text())


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _write(path, value):
    payload = _bytes(value)
    with Path(path).open("xb") as stream:
        stream.write(payload)


def _equal(actual, expected, label):
    if _bytes(actual) != _bytes(expected):
        raise ValueError(f"{label} differs")


def _external(path):
    path = Path(path).absolute()
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("external output custody differs")
    path = path.resolve()
    if ROOT == path or ROOT in path.parents or any((p / ".git").exists() for p in (path, *path.parents)):
        raise ValueError("calibration artifacts must remain outside every checkout")
    return path


def observation_order(pool, baseline):
    """The finite protocol: fresh calls at one case, never disjoint shapes."""
    result = [{"id": f"aa-{i:02}", "candidate_id": baseline, "split": "canary"} for i in range(1, 5)]
    ids = [row["id"] for row in pool]
    for split, order in (("fit", ids), ("calibration", ids[::-1]), ("audit", ids[1:] + ids[:1])):
        result.extend({"id": f"{split}-{name}", "candidate_id": name, "split": split} for name in order)
    return result + [{"id": "anchor", "candidate_id": baseline, "split": "anchor"}]


def _plan(candidate):
    plan = _read(_external_file(candidate, "plan.json", "plan"))
    if set(plan) != {"schema_version", "plan_id", "state", "model_id", "compiler_revision", "executor_revision", "workload", "case_id", "target", "pool", "baseline", "observations", "acceptance"}:
        raise ValueError("calibration plan fields differ")
    if type(plan["schema_version"]) is not int or plan["schema_version"] != 1 or plan["state"] != "frozen":
        raise ValueError("calibration plan must be frozen schema 1")
    for name in ("plan_id", "model_id", "baseline"):
        if not isinstance(plan[name], str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", plan[name]):
            raise ValueError(f"plan {name} differs")
    if plan["case_id"] != "b32_smoke" or plan["target"] != "sm_100a":
        raise ValueError("calibration supports the fixed B200 Flash case only")
    pool = plan["pool"]
    if not isinstance(pool, list) or len(pool) != 4:
        raise ValueError("calibration requires four fixed candidates")
    for row in pool:
        if set(row) != {"id", "schedule"} or not isinstance(row["id"], str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", row["id"]):
            raise ValueError("pool candidate fields differ")
        _external_file(candidate, row["schedule"], "pool Schedule")
    if len({row["id"] for row in pool}) != 4 or len({row["schedule"] for row in pool}) != 4 or plan["baseline"] not in {row["id"] for row in pool}:
        raise ValueError("pool or baseline ownership differs")
    _equal(plan["observations"], observation_order(pool, plan["baseline"]), "frozen observation order")
    limits = plan["acceptance"]
    if set(limits) != {"maximum_cohort_cv", "maximum_within_observation_median_ratio", "maximum_baseline_drift_ratio", "maximum_mape", "maximum_relative_error", "top_k", "maximum_top_k_regret_ratio"}:
        raise ValueError("acceptance fields differ")
    for key, value in limits.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"acceptance {key} must be finite and positive")
        if key.endswith("ratio") and value < 1:
            raise ValueError(f"acceptance {key} must be at least one")
    if type(limits["top_k"]) is not int or limits["top_k"] != 2:
        raise ValueError("fixed-pool audit requires top_k=2")
    return plan


def _authorities(plan):
    compiler_ref = plan["compiler_revision"]
    if set(compiler_ref) != {"path", "revision_id", "canonical_sha256"}:
        raise ValueError("Compiler reference differs")
    compiler = Compiler.load(ROOT, _external_file(ROOT, compiler_ref["path"], "Compiler"))
    if compiler.state != "released":
        raise ValueError("collection and fitting require the released Compiler")
    executor_ref = plan["executor_revision"]
    executor = ExecutorRevision.load(ROOT, _external_file(ROOT, executor_ref["path"], "Executor"))
    _equal(dict(executor.reference), executor_ref, "Executor reference")
    if not {"tools/calibrate_flash_cost.py", "tools/evaluate_flash_candidate.py"}.issubset({row["path"] for row in executor.document["sources"]}):
        raise ValueError("Executor must bind both collection instrument and common evaluator")
    ref = plan["workload"]
    if set(ref) != {"path", "workload_id", "canonical_sha256"}:
        raise ValueError("Workload reference differs")
    workload = WorkloadContract.load(_external_file(ROOT, ref["path"], "Workload"))
    if workload.workload_id != ref["workload_id"] or workload.canonical_sha256 != ref["canonical_sha256"] or workload.document["operator"] != "flash_kmeans_assign":
        raise ValueError("Flash Workload reference differs")
    workload.case(plan["case_id"])
    return compiler, executor, workload


def _assessment(compiler, plan, document):
    assessment = compiler.assess(document)
    ref = plan["compiler_revision"]
    if (assessment.compiler_revision_id != ref["revision_id"] or assessment.compiler_revision_sha256 != ref["canonical_sha256"] or assessment.target != plan["target"] or not assessment.accepted or not assessment.lowering_eligible):
        raise ValueError("Schedule assessment or frozen Compiler binding differs")
    return assessment, compiler.lower(assessment)


def _task(task, executor, workload):
    if task["schema"] != "kernelinfra.task.v1" or task["workloads"] != [workload.workload_id]:
        raise ValueError("GPU Infra calibration task differs")
    stages = task["stages"]
    if [(s["id"], s["kind"], s.get("execution", "broker")) for s in stages] != [("compile", "compile", "local"), ("collection", "judge", "broker")]:
        raise ValueError("calibration requires local compile and one broker judge stage")
    identity = f"{executor.executor_id}@{executor.canonical_sha256}"
    for stage in stages:
        if stage["judge"]["identity"] != identity:
            raise ValueError("GPU Infra judge Executor identity differs")
        source_root = Path(stage["judge"]["cwd"])
        if not source_root.is_absolute():
            raise ValueError("judge source root must be absolute")
        _equal(stage["judge"]["command"], [executor.document["host_environment"]["python"]["invocation_path"], str(source_root / "tools/calibrate_flash_cost.py"), "collect"], "collection command")
    if stages[0]["judge"]["cwd"] != stages[1]["judge"]["cwd"] or "resources" in stages[0]:
        raise ValueError("stage source root or CPU-only compile differs")
    resources = stages[1]["resources"]
    if resources["mode"] != "exclusive" or type(resources["gpu_count"]) is not int or resources["gpu_count"] != 1 or type(resources["run_timeout_s"]) not in (int, float) or not math.isfinite(resources["run_timeout_s"]) or resources["run_timeout_s"] <= 2:
        raise ValueError("collection requires one exclusive GPU and a finite task timeout")
    return Path(stages[0]["judge"]["cwd"])


def _request(candidate, plan, source_root):
    return {"candidate_sha256": candidate.candidate_sha256, "candidate_record_sha256": candidate.canonical_sha256,
            "target": candidate.target, "entry_point": candidate.entry_point, "artifact_roles": dict(candidate.artifact_roles),
            "artifact_paths": {role: f"candidate-{role}" for role in candidate.artifact_payloads},
            "launch_spec_sha256": candidate.launch_spec_sha256, "executor_revision": plan["executor_revision"],
            "workload_path": str(source_root / plan["workload"]["path"]), "workload_sha256": plan["workload"]["canonical_sha256"],
            "case_id": plan["case_id"], "purpose": "confirmatory", "attempt": 1}


def _seal(directory, candidate, plan, source_root):
    directory.mkdir()
    for role, payload in candidate.artifact_payloads.items():
        with (directory / f"candidate-{role}").open("xb") as stream:
            stream.write(payload)
    _write(directory / "request.json", _request(candidate, plan, source_root))


def _candidate(directory, plan, source_root):
    """CPU replay of the common LaunchableCandidate seal, including mirrored runs.

    Original absolute Workload paths stay in the receipt; the local source-bound
    Workload is admitted separately. No retained request is rewritten for replay.
    """
    request = _read(_external_file(directory, "request.json", "sealed request"))
    payloads = {role: _external_file(directory, path, "artifact").read_bytes() for role, path in request["artifact_paths"].items()}
    candidate = LaunchableCandidate(request["candidate_sha256"], request["target"], request["entry_point"], request["artifact_roles"], request["launch_spec_sha256"], payloads)
    _equal(request, _request(candidate, plan, source_root), "sealed common Evaluation request")
    manifest = CudaLaunchManifest.from_dict(json.loads(payloads["launch_manifest"]))
    if manifest.canonical_sha256 != candidate.launch_spec_sha256 or manifest.kernel_name != candidate.entry_point:
        raise ValueError("sealed launch manifest differs")
    return candidate, manifest


def _compile(candidate_root, stage, plan, compiler, executor, workload):
    # Existing host admission checks source/runtime bytes; it does not select a GPU.
    if os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("GPUQ_JOB_ID"):
        raise ValueError("local compile must not inherit a GPU allocation")
    executor.admit_host()
    environment = OpenCakeEnvironment(compiler, TritonToolchainBuilder(), authority_document={"lowering_route": {"backend": "triton", "entry_point": "cake_flash_kmeans_assign"}}, workload=workload, case_id=plan["case_id"], executor=executor)
    for spec in plan["pool"]:
        source = _external_file(candidate_root, spec["schedule"], "pool Schedule").read_bytes()
        assessment, _ = _assessment(compiler, plan, json.loads(source))
        built = environment.build(CandidateSubmission.seal(environment.media_type, source))
        if built.disposition != "launchable" or built.launchable is None:
            raise ValueError(f"candidate build refused: {spec['id']}: {built.feedback}")
        _seal(stage / spec["id"], built.launchable, plan, ROOT)
        with (stage / spec["id"] / "schedule.json").open("xb") as stream:
            stream.write(assessment.schedule_bytes)
    _write(stage / "build.json", {"source_root": str(ROOT), "context": _empirical_context(executor, workload_sha256=workload.canonical_sha256, case_id=plan["case_id"])})


def _compiled(run, plan, compiler, executor, workload, source_root):
    stage = run / "stages/compile"
    _equal(_read(_external_file(stage, "plan.json", "compile plan")), plan, "compile plan")
    _equal(_read(_external_file(stage, "build.json", "compile binding")), {"source_root": str(source_root), "context": _empirical_context(executor, workload_sha256=workload.canonical_sha256, case_id=plan["case_id"])}, "compile context")
    result = {}
    seen = set()
    for spec in plan["pool"]:
        source = _external_file(run / "candidate", spec["schedule"], "frozen Schedule").read_bytes()
        assessment, lowering = _assessment(compiler, plan, json.loads(source))
        directory = stage / spec["id"]
        _equal(_read(_external_file(directory, "schedule.json", "compiled Schedule")), json.loads(assessment.schedule_bytes), "compiled Schedule")
        candidate, manifest = _candidate(directory, plan, source_root)
        if set(candidate.artifact_payloads) != {"lowered_source", "compiler_expanded_source", "ttir", "ttgir", "llir", "ptx", "cubin", "launch_manifest"}:
            raise ValueError("complete common Flash build artifacts are required")
        submitted = CandidateSubmission.seal(OpenCakeEnvironment.media_type, source)
        if candidate.candidate_sha256 != submitted.sha256 or candidate.artifact_payloads["lowered_source"] != lowering.source.encode():
            raise ValueError("compiled artifact differs from the frozen Schedule lowering")
        requirements = lowering.toolchain_requirements
        if (list(manifest.grid) != requirements["grid"] or manifest.block != (requirements["compile_options"]["num_warps"] * 32, 1, 1) or manifest.kernel_name != requirements["kernel_entry_point"] or manifest.hidden_null_pointer_parameters != 2):
            raise ValueError("compiled launch differs from canonical lowering")
        template = json.loads(assessment.schedule_bytes)
        template["schedule_id"] = "display-id"
        identity = _bytes(template)
        if identity in seen:
            raise ValueError("pool repeats the same exact Schedule")
        seen.add(identity)
        result[spec["id"]] = (candidate, json.loads(assessment.schedule_bytes))
    return result


def _broker_principal():
    if sys.platform != "linux":
        raise ValueError("broker collection requires Linux peer credentials")
    endpoint = os.environ.get("GPUQ_SOCKET", "/tmp/agent-gpu-broker.sock")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(10)
        connection.connect(endpoint)
        peer = struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
    observed = {"pid": os.getpid(), "parent_pid": os.getppid(), "uid": os.geteuid(), "gid": os.getegid(), "broker_peer": list(peer), "broker_socket": endpoint,
                "job_id": os.environ.get("GPUQ_JOB_ID"), "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"), "run_id": os.environ.get("KERNELINFRA_RUN_ID")}
    if list(peer) != [observed["parent_pid"], observed["uid"], observed["gid"]]:
        raise ValueError("controller is not the direct broker execution principal")
    return observed


def _principal(observed, job_id, run_id):
    if (observed["broker_peer"] != [observed["parent_pid"], observed["uid"], observed["gid"]]
        or not isinstance(job_id, str) or re.fullmatch(r"gpuq-[0-9a-f]{12}", job_id) is None or observed["job_id"] != job_id
        or not isinstance(run_id, str) or not run_id or observed["run_id"] != run_id
        or not isinstance(observed["visible_device"], str) or not observed["visible_device"] or "," in observed["visible_device"]):
        raise ValueError("controller is not the observed direct broker execution principal")


def _node_admission(run, task, observed, deadline):
    """Read this run's node-owned allocation; never discover or guess a job ID.

    Broker 0.5.3 starts its child before the asynchronous node event pump may have
    published broker_started. Only that bounded submitting/queued race is waited
    for; missing, malformed, unrelated or terminal state is an unknown admission.
    """
    request = _read(_external_file(run, "request.json", "node request"))
    expected = {"run_id": observed["run_id"], "task_id": task["task_id"]}
    _equal({"schema": request["schema"], **{k: request[k] for k in expected}}, {"schema": "kernelinfra.request.v1", **expected}, "node request identity")
    for key in ("task_sha256", "candidate_sha256"):
        value = request[key]
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("node request content identity differs")
        expected[key] = value
    expected.update(schema="kernelinfra.state.v1", stage_id="collection", stage_kind="judge", stage_index=1)
    publication_deadline = min(deadline, time.monotonic() + 10)
    while True:
        state = _read(_external_file(run, "state.json", "node state"))
        _equal({key: state[key] for key in expected}, expected, "node allocation identity")
        if state["run_dir"] != str(run) or state["terminal_at"] is not None:
            raise ValueError("node allocation run directory or terminal state differs")
        job_id = state["broker_job_id"]
        if job_id is not None and (not isinstance(job_id, str) or re.fullmatch(r"gpuq-[0-9a-f]{12}", job_id) is None):
            raise ValueError("node broker job identity is malformed")
        if state["state"] == "running":
            ids = state["gpu_ids"]
            if job_id is None or not isinstance(ids, list) or len(ids) != 1 or type(ids[0]) is not int or ids[0] < 0 or observed["visible_device"] != str(ids[0]):
                raise ValueError("node allocation differs from the controller-visible GPU")
            exported = observed["job_id"]
            if exported is not None and exported != job_id:
                raise ValueError("exported broker job differs from node allocation")
            observed.update(exported_job_id=exported, job_id=job_id,
                            node_admission={**expected, "state": "running", "broker_job_id": job_id, "gpu_ids": ids})
            _principal(observed, job_id, request["run_id"])
            return observed
        if state["state"] not in {"submitting", "queued"} or state["gpu_ids"] != []:
            raise ValueError("node allocation is not awaiting broker publication")
        if time.monotonic() >= publication_deadline:
            raise TimeoutError("node did not publish this broker allocation in time")
        time.sleep(.05)


class CorrectnessRejected(ValueError):
    """A common external-oracle rejection, distinct from unknown measurement."""


def _observation(directory, spec, plan, candidate, source_root, job_id):
    for name in ("stdout.log", "stderr.log"):
        _external_file(directory, name, "mandatory evaluator stream")
    observed, _ = _candidate(directory, plan, source_root)
    if observed.canonical_sha256 != candidate.canonical_sha256:
        raise ValueError("measured artifact differs from the sealed compile artifact")
    result = _read(_external_file(directory, "result.json", "Evaluation result"))
    if type(result["schema_version"]) is not int or result["schema_version"] != 1 or result["job_id"] != job_id or result["mode"] != "exclusive" or result["admitted"] is not True or result["error"] is not None or result["failure_class"] is not None:
        raise ValueError("common evaluator admission or completion differs")
    receipt = result["receipt"]
    if receipt["correctness_passed"] is not True:
        raise CorrectnessRejected(f"external oracle rejected {spec['id']}")
    artifacts = receipt["artifacts"]
    _equal(artifacts, {"correctness_output": "correctness-output.json", "launch_receipt": "launch-receipt.json", "timing_samples": "timing-samples.json"}, "Evaluation artifacts")
    correctness = _read(_external_file(directory, artifacts["correctness_output"], "correctness output"))
    if (set(correctness) != {"metrics", "output_sha256", "output_size_bytes"}
        or not isinstance(correctness["output_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", correctness["output_sha256"])
        or type(correctness["output_size_bytes"]) is not int or correctness["output_size_bytes"] <= 0):
        raise ValueError("correctness output record differs")
    _equal(correctness["metrics"], receipt["correctness"], "external-oracle metrics")
    launch = _read(_external_file(directory, artifacts["launch_receipt"], "launch receipt"))
    if not isinstance(launch["gpu_uuid"], str) or not launch["gpu_uuid"]:
        raise ValueError("observed GPU UUID is missing")
    _equal(launch, {"job_id": job_id, "gpu_uuid": launch["gpu_uuid"], "candidate_sha256": candidate.candidate_sha256, "correctness_launches": 1, "fallback_calls": 0}, "launch receipt")
    counters = result["counters"]
    for key, expected in {"compiler_invocations": 0, "module_loads": 1, "preflight_calls": 1, "timing_samples": 125, "fallback_calls": 0}.items():
        _equal(counters[key], expected, f"Evaluation {key}")
    if type(counters["kernel_calls"]) is not int or counters["kernel_calls"] < 126 or receipt["kernel_calls"] != 1 or receipt["fallback_calls"] != 0:
        raise ValueError("observed kernel calls differ")
    raw = _read(_external_file(directory, artifacts["timing_samples"], "raw CUPTI samples"))
    if set(raw) != {"cohorts_ms"} or not isinstance(raw["cohorts_ms"], list) or len(raw["cohorts_ms"]) != 5:
        raise ValueError("CUPTI cohort count differs")
    summaries = []
    for values in raw["cohorts_ms"]:
        if not isinstance(values, list) or len(values) != 25 or any(type(v) not in (int, float) for v in values):
            raise ValueError("CUPTI sample count or numeric type differs")
        summaries.append(summarize_cohort(values))
    limits = plan["acceptance"]
    medians = [s["median_ms"] for s in summaries]
    if any(s["cv"] > limits["maximum_cohort_cv"] for s in summaries) or max(medians) / min(medians) > limits["maximum_within_observation_median_ratio"]:
        raise ValueError("CUPTI cohort CV or within-observation drift failed")
    median = statistics.median(v for cohort in raw["cohorts_ms"] for v in cohort)
    _equal(receipt["timing"], {"measurement_quality_passed": True, "pooled_median_ms": median, "cohort_count": 5, "samples_per_cohort": 25}, "timing summary")
    return {**spec, "gpu_uuid": launch["gpu_uuid"], "median_ms": median, "cohorts": summaries}


def _baseline_quality(rows, limits):
    canaries = [r["median_ms"] for r in rows if r["split"] == "canary"]
    if len(canaries) == 4 and max(canaries) / min(canaries) > limits["maximum_baseline_drift_ratio"]:
        raise ValueError("baseline A/A canary failed")
    anchors = [r["median_ms"] for r in rows if r["split"] == "anchor"]
    if anchors:
        medians = [statistics.median(canaries), *anchors]
        if max(medians) / min(medians) > limits["maximum_baseline_drift_ratio"]:
            raise ValueError("baseline final-anchor drift failed")
    if len({r["gpu_uuid"] for r in rows}) > 1:
        raise ValueError("observations did not share one GPU UUID")


def _measure(run, stage, plan, executor, compiled, task):
    deadline = time.monotonic() + task["stages"][1]["resources"]["run_timeout_s"] - 2
    principal = _node_admission(run, task, _broker_principal(), deadline)
    _write(stage / "execution-context.json", principal)
    rows = []
    for spec in plan["observations"]:
        candidate, _ = compiled[spec["candidate_id"]]
        directory = stage / spec["id"]
        _seal(directory, candidate, plan, ROOT)
        command = [executor.document["host_environment"]["python"]["invocation_path"], str(ROOT / "tools/evaluate_flash_candidate.py"), "--request", str(directory / "request.json"), "--output", str(directory / "result.json")]
        # GPU Infra/broker already owns this process group. A new session would
        # let the evaluator escape lease cancellation. run kills/reaps its direct
        # child on timeout; broker cancellation reaches the inherited group.
        remaining = int(deadline - time.monotonic())
        if remaining <= 0:
            raise TimeoutError("collection task deadline exhausted before next observation")
        environment = sanitized_environment()
        environment["GPUQ_JOB_ID"] = principal["job_id"]
        with (directory / "stdout.log").open("xb") as stdout, (directory / "stderr.log").open("xb") as stderr:
            completed = subprocess.run(command, cwd=ROOT, timeout=remaining, env=environment,
                                       stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, check=False)
        if completed.returncode != 0:
            raise ValueError("common evaluator subprocess failed")
        row = _observation(directory, spec, plan, candidate, ROOT, principal["job_id"])
        rows.append(row)
        _write(directory / "observation.json", row)
        _baseline_quality(rows, plan["acceptance"])
    _write(stage / "observations.json", {"rows": rows, "context": _read(run / "stages/compile/build.json")["context"]})


def _stage_result(path, *, status, validity, summary, stage, workload=None):
    artifacts = {file.relative_to(stage).as_posix(): file.relative_to(stage).as_posix() for file in sorted(stage.rglob("*")) if file.is_file() and file != path}
    _write(path, {"schema": "kernelinfra.stage-result.v1", "status": status, "validity": validity, "summary": summary,
                  "workloads": [{"id": workload, "correct": True}] if workload and status == "passed" else [], "artifacts": artifacts,
                  "metrics": {"scope": "fixed-case calibration only; no Candidate promotion or generic timing frontier"}})


def collect():
    run = _external(os.environ["KERNELINFRA_RUN_DIR"])
    stage = _external(os.environ["KERNELINFRA_STAGE_DIR"])
    output = _external(os.environ["KERNELINFRA_RESULT"])
    try:
        kind = os.environ["KERNELINFRA_STAGE_KIND"]
        stage_id = "compile" if kind == "compile" else "collection"
        if kind not in {"compile", "judge"} or os.environ["KERNELINFRA_STAGE_ID"] != stage_id or stage != run / "stages" / stage_id or output != stage / "result.json":
            raise ValueError("GPU Infra stage binding differs")
        if Path(os.environ["KERNELINFRA_CANDIDATE_DIR"]).resolve() != run / "candidate" or Path(os.environ["KERNELINFRA_TASK"]).resolve() != run / "task.json":
            raise ValueError("GPU Infra immutable snapshot paths differ")
        plan = _plan(run / "candidate")
        compiler, executor, workload = _authorities(plan)
        task = _read(run / "task.json")
        if _task(task, executor, workload) != ROOT:
            raise ValueError("live collection source root differs")
        _write(stage / "plan.json", plan)
        if kind == "compile":
            _compile(run / "candidate", stage, plan, compiler, executor, workload)
        else:
            compiled = _compiled(run, plan, compiler, executor, workload, ROOT)
            _measure(run, stage, plan, executor, compiled, task)
        _stage_result(output, status="passed", validity="valid", summary="frozen Flash calibration stage passed", stage=stage, workload=workload.workload_id if kind == "judge" else None)
        return 0
    except Exception as error:
        if not output.exists():
            _stage_result(output, status="failed", validity="invalid" if isinstance(error, CorrectnessRejected) else "unknown", summary=f"{type(error).__name__}: {error}", stage=stage)
        return 1


def _receipts(run, task, result, executor):
    if result["schema"] != "kernelinfra.run-result.v1" or result["outcome"] != "completed" or result["validity"] != "valid" or result["task_id"] != task["task_id"] or result["frontier_eligible"] is not False:
        raise ValueError("whole calibration run did not pass or claims a timing frontier")
    if [(s["id"], s["kind"], s["status"], s["validity"]) for s in result["stages"]] != [("compile", "compile", "passed", "valid"), ("collection", "judge", "passed", "valid")]:
        raise ValueError("required calibration stages did not pass")
    receipts = {}
    for spec in task["stages"]:
        stage = run / "stages" / spec["id"]
        receipt = _read(_external_file(stage, "receipt.json", "stage receipt"))
        expected = {"schema": "kernelinfra.stage-receipt.v1", "run_id": result["run_id"], "stage_id": spec["id"], "stage_kind": spec["kind"], "execution": spec.get("execution", "broker"), "judge_identity": f"{executor.executor_id}@{executor.canonical_sha256}", "exit_code": 0, "judge_result_valid": True, "error": None}
        _equal({key: receipt[key] for key in expected}, expected, "durable stage receipt")
        if spec["id"] == "collection":
            ids = receipt["gpu_ids"]
            if not isinstance(ids, list) or len(ids) != 1 or type(ids[0]) is not int or ids[0] < 0:
                raise ValueError("broker receipt must identify one allocated GPU")
        judge = _read(_external_file(stage, "result.json", "stage result"))
        if judge["schema"] != "kernelinfra.stage-result.v1" or judge["status"] != "passed" or judge["validity"] != "valid":
            raise ValueError("durable judge result did not pass")
        for relative in judge["artifacts"].values():
            _external_file(stage, relative, "retained stage artifact")
        receipts[spec["id"]] = receipt
    return receipts


def _derive(plan, executor, workload, compiled, rows, run_id):
    by_split = {split: {r["candidate_id"]: r for r in rows if r["split"] == split} for split in ("fit", "calibration", "audit")}
    context = _empirical_context(executor, workload_sha256=workload.canonical_sha256, case_id=plan["case_id"])
    ref = plan["compiler_revision"]
    model = {"schema_version": 2, "model_id": plan["model_id"], "compiler_revision_id": ref["revision_id"], "compiler_revision_sha256": ref["canonical_sha256"], "target": plan["target"], "context": context,
             "reported_evidence": {"run_id": run_id, "plan_id": plan["plan_id"], "scope": "same-case independent measurement cohorts; advisory exact-pool prediction only", "range_meaning": "observed calibration deviation, not a probability, guarantee or pruning rule"}, "curves": []}
    for spec in plan["pool"]:
        name = spec["id"]
        template = compiled[name][1]
        buffers = {b["name"]: b for b in template["buffers"]}
        extent = workload.case(plan["case_id"])["shape"]["N"]
        if any(buffers[b]["shape"][1] != extent for b in ("tokens", "assignments")):
            raise ValueError("measured template extent differs from Workload case")
        point = by_split["fit"][name]["median_ms"] * 1000
        envelope = abs(by_split["calibration"][name]["median_ms"] * 1000 / point - 1)
        model["curves"].append({"template": template, "varying_dimensions": [{"buffer": b, "dimension": 1} for b in ("tokens", "assignments")], "extent_multiple": 1, "points": [{"extent": extent, "kernel_us": point}], "relative_error_envelope": envelope})
    EmpiricalCostModel(model)  # Freeze point/envelope before inspecting audit outcomes.
    selection = _EmpiricalSelection({"kind": "external_empirical_advisory_v1", "model": model}, context=context, compiler_revision_id=ref["revision_id"], compiler_revision_sha256=ref["canonical_sha256"], target=plan["target"])
    limits = plan["acceptance"]
    validation = {}
    passed = True
    for split in ("calibration", "audit"):
        report, filter_rows = [], []
        for spec in plan["pool"]:
            name = spec["id"]
            prediction = selection.estimate(compiled[name][1])
            if prediction["covered"] is not True:
                raise ValueError(f"fixed pool model abstained: {prediction['reason']}")
            observed = by_split[split][name]["median_ms"] * 1000
            lower, upper = prediction["empirical_range_us"]
            report.append({"candidate_id": name, "observed_us": observed, "predicted_us": prediction["predicted_kernel_us"], "relative_error": abs(prediction["predicted_kernel_us"] / observed - 1), "empirical_range_us": [lower, upper], "inside_descriptive_range": lower <= observed <= upper})
            filter_rows.append({"candidate_id": name, "disposition": "launchable", "empirical_cost": prediction})
        ordered, order = _empirical_filter(filter_rows)
        selected = [row["candidate_id"] for row in ordered[:limits["top_k"]]]
        metrics = {"mape": statistics.mean(r["relative_error"] for r in report), "maximum_relative_error": max(r["relative_error"] for r in report), "descriptive_range_covered_count": sum(r["inside_descriptive_range"] for r in report), "case_count": len(report)}
        metrics["descriptive_range_coverage_fraction"] = metrics["descriptive_range_covered_count"] / len(report)
        acceptable = metrics["mape"] <= limits["maximum_mape"] and metrics["maximum_relative_error"] <= limits["maximum_relative_error"]
        if split == "audit":
            metrics["top_k_regret_ratio"] = min(r["observed_us"] for r in report if r["candidate_id"] in selected) / min(r["observed_us"] for r in report)
            metrics["predicted_tie_count"] = len(report) - len({r["predicted_us"] for r in report})
            acceptable &= order["order_applied"] and metrics["top_k_regret_ratio"] <= limits["maximum_top_k_regret_ratio"]
        passed &= acceptable
        validation[split] = {"passed": bool(acceptable), "metrics": metrics, "rows": report, "selected_ids": selected, "order": order}
    model["reported_evidence"]["validation"] = validation
    return bool(passed), model, validation


def fit(run, output):
    output = _external(output)
    output.mkdir(exist_ok=False)
    try:
        run = _external(run)
        plan = _plan(run / "candidate")
        compiler, executor, workload = _authorities(plan)
        task = _read(_external_file(run, "task.json", "task"))
        source_root = _task(task, executor, workload)
        result = _read(_external_file(run, "result.json", "run result"))
        receipts = _receipts(run, task, result, executor)
        compiled = _compiled(run, plan, compiler, executor, workload, source_root)
        stage = run / "stages/collection"
        _equal(_read(_external_file(stage, "plan.json", "collection plan")), plan, "collection plan")
        principal = _read(_external_file(stage, "execution-context.json", "broker principal"))
        _principal(principal, receipts["collection"]["broker_job_id"], result["run_id"])
        _equal(principal["node_admission"], {"schema": "kernelinfra.state.v1", "run_id": result["run_id"], "task_id": result["task_id"], "task_sha256": result["task_sha256"], "candidate_sha256": result["candidate_sha256"], "state": "running", "stage_id": "collection", "stage_kind": "judge", "stage_index": 1, "broker_job_id": receipts["collection"]["broker_job_id"], "gpu_ids": receipts["collection"]["gpu_ids"]}, "retained node admission")
        if principal["exported_job_id"] is not None and principal["exported_job_id"] != principal["job_id"]:
            raise ValueError("retained exported broker job differs from node admission")
        if principal["visible_device"] != str(receipts["collection"]["gpu_ids"][0]):
            raise ValueError("controller-visible GPU differs from the broker allocation")
        rows = []
        for spec in plan["observations"]:
            directory = stage / spec["id"]
            row = _observation(directory, spec, plan, compiled[spec["candidate_id"]][0], source_root, principal["job_id"])
            _equal(_read(_external_file(directory, "observation.json", "observation index")), row, "derived observation")
            rows.append(row)
        _baseline_quality(rows, plan["acceptance"])
        _equal(_read(_external_file(stage, "observations.json", "collection index")), {"rows": rows, "context": _empirical_context(executor, workload_sha256=workload.canonical_sha256, case_id=plan["case_id"])}, "derived collection index")
        passed, model, validation = _derive(plan, executor, workload, compiled, rows, result["run_id"])
        _write(output / "audit.json", {"passed": passed, "run_id": result["run_id"], "validation": validation, "range_meaning": model["reported_evidence"]["range_meaning"]})
        if passed:
            _write(output / "model.json", model)
        return 0 if passed else 1
    except Exception as error:
        _write(output / "audit.json", {"passed": False, "failure_class": type(error).__name__, "reason": str(error)})
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("collect")
    fitting = commands.add_parser("fit")
    fitting.add_argument("run", type=Path)
    fitting.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return collect() if args.action == "collect" else fit(args.run, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
