#!/usr/bin/env python3
"""Collect a frozen NVIDIA FMA/GEMM calibration through GPU Infra, or fit/audit it on CPU.

Plans, candidates and output directories are external artifacts. A plan binds this
collector's bytes and the released Compiler. The plan owns domains and thresholds;
the workload oracle below owns the two explicitly supported evaluation contracts.
Run the compile-container action in a CPU-only GPU Infra local stage, followed by
broker-owned collect-container correctness and profile stages. Fitting audits
the retained device observations after the broker releases the GPU.
"""
from __future__ import annotations

import argparse
import atexit
import ctypes
import itertools
import json
import math
import os
import re
import signal
import socket
import statistics
import struct
import subprocess
import sys
import time
import traceback
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from open_cake_ir.compiler import Compiler, EmpiricalCostModel
from open_cake_ir.compiler.toolchain import compile_triton, inspect_triton_resources
from open_cake_ir.compiler.performance.compiled_resources import CompiledResources, load_compiled_resources
from open_cake_ir.evaluation.cuda_driver import _DYNAMIC_SHARED_OPT_IN_THRESHOLD, _driver_call


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _target_contract(compiler, plan):
    """Take the device and binary version from the declared CUDA Target."""
    target_id = plan.get("target")
    target = compiler._revision.targets.get(target_id)
    if target is None or target.compute_capability is None or target_id not in {"sm_100a", "sm_103a"}:
        raise ValueError("calibration requires a declared B200 or B300 CUDA Target")
    return target, target.compute_capability[0] * 10 + target.compute_capability[1]


def _candidate_set_regrets(audit, maximum_candidates_per_turn):
    """Audit every possible Lab three-to-two cut and provider tie order."""
    if type(maximum_candidates_per_turn) is not int or maximum_candidates_per_turn != 3:
        raise ValueError("cost audit requires the Lab's maximum_candidates_per_turn=3")
    regrets = []
    for workload_id in sorted({row["workload_id"] for row in audit}):
        group = sorted((row for row in audit if row["workload_id"] == workload_id), key=lambda row: row["id"])
        for candidate_set in itertools.combinations(group, maximum_candidates_per_turn):
            best_observed = min(row["observed_us"] for row in candidate_set)
            worst = None
            for provider_order in itertools.permutations(candidate_set):
                # Python's stable sort is the Lab rule: equal predictions retain
                # provider order, which can change which member falls below a cut.
                survivors = sorted(provider_order, key=lambda row: row["predicted_us"])[:2]
                regret = min(row["observed_us"] for row in survivors) / best_observed
                if worst is None or regret > worst["top2_regret_ratio"]:
                    worst = {"workload_id": workload_id,
                             "candidate_set": [row["id"] for row in candidate_set],
                             "provider_order": [row["id"] for row in provider_order],
                             "survivors": [row["id"] for row in survivors],
                             "top2_regret_ratio": regret}
            regrets.append(worst)
    return regrets


def _expected_signature(family):
    if family == "fma":
        return {"a": "*fp32", "b": "*fp32", "c": "*fp32", "y": "*fp32"}
    if family == "gemm_bias":
        return {"a": "*bf16", "b": "*bf16", "bias": "*fp32", "c": "*fp32"}
    raise ValueError(f"calibration family {family!r} is unsupported")


def _positive_number(value, label, *, allow_zero=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{label} must be finite and {'nonnegative' if allow_zero else 'positive'}")


def _check_plan(candidate):
    """Reject an invalid frozen calibration before a broker device lease."""
    candidate = _external(candidate)
    plan = _read(candidate / "plan.json")
    if plan.get("state") != "frozen":
        raise ValueError("calibration plan must be frozen")
    if sha256(Path(__file__).read_bytes()).hexdigest() != plan.get("collector_sha256"):
        raise ValueError("calibration plan collector differs")
    compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
    target, _ = _target_contract(compiler, plan)
    if compiler.commit is None or plan.get("compiler_revision_id") != compiler._revision.revision_id:
        raise ValueError("calibration plan requires this clean Compiler commit")
    if plan.get("device_name") not in target.device_names:
        raise ValueError("calibration device name differs from Target")
    for key in ("broker_uid", "multiprocessor_count"):
        if type(plan.get(key)) is not int or plan[key] < 0 or (key == "multiprocessor_count" and plan[key] == 0):
            raise ValueError(f"calibration {key} differs")
    if not isinstance(plan.get("cuobjdump"), str) or not Path(plan["cuobjdump"]).is_absolute():
        raise ValueError("calibration requires an absolute cuobjdump path")
    if not isinstance(plan.get("container_image_id"), str) or re.fullmatch(r"sha256:[0-9a-f]{64}", plan["container_image_id"]) is None:
        raise ValueError("calibration requires an exact container image id")
    if not isinstance(plan.get("expected_runtime"), dict) or not plan["expected_runtime"]:
        raise ValueError("calibration requires frozen runtime facts")
    for key in ("model_id", "input_scope"):
        if not isinstance(plan.get(key), str) or not plan[key].strip():
            raise ValueError(f"calibration {key} differs")
    sampling = plan.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("calibration sampling differs")
    for key in ("warmup", "rounds", "repetitions", "l2_flush_bytes"):
        if type(sampling.get(key)) is not int or sampling[key] < (0 if key == "warmup" else 1):
            raise ValueError(f"calibration sampling.{key} differs")
    acceptance = plan.get("acceptance")
    limits = plan.get("model_acceptance")
    if not isinstance(acceptance, dict) or not isinstance(limits, dict):
        raise ValueError("calibration acceptance limits differ")
    for key in ("maximum_cohort_cv", "maximum_repeat_median_ratio"):
        _positive_number(acceptance.get(key), f"acceptance.{key}")
    for key in ("maximum_mape", "maximum_relative_error", "maximum_top2_regret_ratio", "envelope_allowance"):
        _positive_number(limits.get(key), f"model_acceptance.{key}", allow_zero=key == "envelope_allowance")
    if limits["maximum_top2_regret_ratio"] < 1 or limits["envelope_allowance"] >= 1:
        raise ValueError("calibration model acceptance bounds differ")
    if limits.get("maximum_candidates_per_turn") != 3:
        raise ValueError("calibration requires the Lab's three-candidate set")
    curves = plan.get("curves")
    cases = plan.get("cases")
    if not isinstance(curves, list) or not curves or not isinstance(cases, list) or not cases:
        raise ValueError("calibration curves and cases are required")
    curve_by_id = {curve.get("id"): curve for curve in curves if isinstance(curve, dict)}
    if len(curve_by_id) != len(curves) or any(not isinstance(name, str) or not name for name in curve_by_id):
        raise ValueError("calibration curve ids differ")
    seen_cases = set()
    splits = {name: {"fit": set(), "calibration": set(), "audit": set()} for name in curve_by_id}
    audit_groups = {}
    templates = {}
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"id", "curve_id", "extent", "split", "schedule", "family", "workload_id"}:
            raise ValueError("calibration case fields differ")
        if not isinstance(case["id"], str) or not case["id"] or case["id"] in seen_cases:
            raise ValueError("calibration case ids differ")
        seen_cases.add(case["id"])
        curve = curve_by_id.get(case["curve_id"])
        if curve is None or case["split"] not in splits[case["curve_id"]]:
            raise ValueError("calibration case curve or split differs")
        if type(case["extent"]) is not int or case["extent"] <= 0:
            raise ValueError("calibration case extent differs")
        if not isinstance(case["workload_id"], str) or not case["workload_id"]:
            raise ValueError("calibration workload id differs")
        if type(curve.get("extent_multiple")) is not int or curve["extent_multiple"] <= 0 or case["extent"] % curve["extent_multiple"]:
            raise ValueError("calibration curve extent alignment differs")
        path = (candidate / case["schedule"]).resolve()
        if candidate not in path.parents or not path.is_file():
            raise ValueError("calibration Schedule must belong to candidate snapshot")
        assessment = compiler.assess_file(path)
        if assessment.target != plan["target"] or not assessment.lowering_eligible:
            raise ValueError("calibration Schedule target or admission differs")
        lowering = compiler.lower(assessment)
        signature = _expected_signature(case["family"])
        if lowering.toolchain_requirements.get("signature") != signature:
            raise ValueError("calibration Schedule ABI differs")
        document = json.loads(assessment.schedule_bytes)
        buffers = {row["name"]: row for row in document["buffers"]}
        bindings = curve.get("varying_dimensions")
        if not isinstance(bindings, list) or not bindings:
            raise ValueError("calibration varying dimensions differ")
        seen_bindings = set()
        for binding in bindings:
            if (not isinstance(binding, dict) or set(binding) != {"buffer", "dimension"}
                    or binding["buffer"] not in buffers or type(binding["dimension"]) is not int
                    or not 0 <= binding["dimension"] < len(buffers[binding["buffer"]]["shape"])
                    or buffers[binding["buffer"]]["shape"][binding["dimension"]] != case["extent"]):
                raise ValueError("calibration varying dimension extent differs")
            position = (binding["buffer"], binding["dimension"])
            if position in seen_bindings:
                raise ValueError("calibration varying dimension is duplicated")
            seen_bindings.add(position)
        document["schedule_id"] = "calibration-template"
        for name, dimension in seen_bindings:
            buffers[name]["shape"][dimension] = None
        prior = templates.setdefault(case["curve_id"], document)
        if document != prior:
            raise ValueError("calibration curve template drifts outside its varying dimensions")
        splits[case["curve_id"]][case["split"]].add(case["extent"])
        if case["split"] == "audit":
            audit_groups.setdefault(case["workload_id"], set()).add(case["curve_id"])
    for curve_id, cohorts in splits.items():
        if not all(cohorts.values()) or any(cohorts[a] & cohorts[b] for a, b in (("fit", "calibration"), ("fit", "audit"), ("calibration", "audit"))):
            raise ValueError(f"calibration split extents differ for {curve_id}")
    if not any(len(group) >= 3 for group in audit_groups.values()):
        raise ValueError("calibration audit has no three-candidate decision")
    return plan, compiler


def _external(path):
    path = Path(path).resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError("calibration outputs must remain outside the checkout")
    return path


def _cpu_case(case, document, torch, distribution):
    """Independent CPU references; no generated implementation is inspected."""
    shapes = {buffer["name"]: buffer["shape"] for buffer in document["buffers"] if buffer["space"] == "global"}
    if case["family"] == "fma":
        indices = torch.arange(math.prod(shapes["a"]), dtype=torch.int64)
        inputs = [(((indices * scale + offset + distribution * 19) % 257 - 128).to(torch.float32) / 128).reshape(shapes["a"]) for scale, offset in ((1, 3), (7, 11), (17, 5))]
        expected = (inputs[0].double() * inputs[1].double() + inputs[2].double()).float()
        return inputs, expected, 0.0
    if case["family"] == "gemm_bias":
        generator = torch.Generator(device="cpu").manual_seed(20260906 + shapes["a"][0] + distribution * 100000)
        inputs = [torch.randn(shapes[name], generator=generator, dtype=torch.float32).to(dtype) for name, dtype in (("a", torch.bfloat16), ("b", torch.bfloat16), ("bias", torch.float32))]
        expected = (inputs[0].double() @ inputs[1].double().t() + inputs[2].double()[None, :]).float()
        return inputs, expected, .002
    raise ValueError("unsupported calibration workload contract")


def _trace_samples(stage, repetition, plan, rows):
    """Verify individual flush/target ownership before attributing any duration."""
    order = []
    for round_index in range(plan["sampling"]["rounds"]):
        shift = (round_index + repetition * 7) % len(rows)
        indices = [(shift + offset) % len(rows) for offset in range(len(rows))]
        if repetition % 2:indices.reverse()
        order.extend(indices)
    if _read(stage / f"launch-order-{repetition}.json") != [rows[index]["id"] for index in order]:
        raise ValueError("launch ledger differs from frozen schedule")
    trace = _read(stage / f"cupti-trace-{repetition}.json")
    events = trace["traceEvents"]
    kernels = sorted((event for event in events if event.get("cat") == "kernel"), key=lambda event: event["ts"])
    drivers = [event["args"]["correlation"] for event in events if event.get("cat") == "cuda_driver" and event.get("name") == "cuLaunchKernel"]
    driver_ids = set(drivers)
    if len(kernels) != 2 * len(order) or len(driver_ids) != len(order) or len(drivers) != len(order):
        raise ValueError("CUPTI kernel/driver event count differs")
    if len({(event["args"]["device"], event["args"]["context"], event["args"]["stream"]) for event in kernels}) != 1:
        raise ValueError("CUPTI records span multiple devices/contexts/streams")
    for event in kernels:
        if any(type(event[key]) not in (int, float) or not math.isfinite(event[key]) for key in ("ts", "dur")) or event["dur"] <= 0:
            raise ValueError("CUPTI timestamp/duration is not finite and positive")
    samples = [[] for _ in rows]
    seen = set()
    for position, index in enumerate(order):
        clear, target = kernels[2 * position:2 * position + 2]
        row = rows[index]
        previous = kernels[2 * position - 1] if position else None
        if any(type(value) is not int or value <= 0 for key in ("grid", "block") for value in target["args"][key]):
            raise ValueError("CUPTI launch dimensions are not positive integers")
        if ("FillFunctor<unsigned char>" not in clear["name"]
                or clear["ts"] + clear["dur"] > target["ts"] + .002
                or (previous and previous["ts"] + previous["dur"] > clear["ts"] + .002)
                or target["name"] != row["profile"]["compiled_resources"]["entry_point"]
                or target["args"]["grid"] != row["grid"]
                or target["args"]["block"] != [row["profile"]["compiled_resources"]["threads_per_cta"], 1, 1]):
            raise ValueError("clear/target order or launch identity differs")
        correlation = target["args"]["correlation"]
        if correlation not in driver_ids or correlation in seen:
            raise ValueError("target lacks its unique driver correlation")
        seen.add(correlation)
        duration = target["dur"]
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("CUPTI duration is not positive and finite")
        samples[index].append(duration)
    return samples


def _prepare():
    """Compile and build independent CPU oracles without a device lease."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("GPUQ_JOB_ID"):
        raise ValueError("local calibration compile must not inherit a GPU allocation")
    run = _external(os.environ["KERNELINFRA_RUN_DIR"])
    stage = _external(os.environ["KERNELINFRA_STAGE_DIR"])
    candidate = _external(os.environ["KERNELINFRA_CANDIDATE_DIR"])
    result = _external(os.environ["KERNELINFRA_RESULT"])
    if (stage != run / "stages/compile" or candidate != run / "candidate"
            or result != stage / "result.json" or os.environ.get("KERNELINFRA_STAGE_KIND") != "compile"
            or os.environ.get("KERNELINFRA_STAGE_ID") != "compile"):
        raise ValueError("local calibration compile stage differs")
    plan, compiler = _check_plan(candidate)
    import torch
    torch.set_num_threads(1)
    _write(stage / "plan.json", plan)
    (stage / "collector.py").write_bytes(Path(__file__).read_bytes())
    rows = []
    for index, case in enumerate(plan["cases"]):
        directory = stage / f"{index:04d}"
        directory.mkdir()
        document = _read(candidate / case["schedule"])
        assessment = compiler.assess(document)
        lowering = compiler.lower(assessment)
        compilation = compile_triton(lowering.source.encode(), lowering.toolchain_requirements)
        resources = inspect_triton_resources(compilation, plan["cuobjdump"])
        _write(directory / "schedule.json", json.loads(assessment.schedule_bytes))
        (directory / "lowered.py").write_bytes(compilation.source)
        (directory / "kernel.cubin").write_bytes(compilation.artifacts["cubin"])
        (directory / "kernel.ptx").write_bytes(compilation.artifacts["ptx"])
        for distribution in (0, 1):
            inputs, answer, tolerance = _cpu_case(case, document, torch, distribution)
            torch.save({"inputs": inputs, "answer": answer, "tolerance": tolerance},
                       directory / f"oracle-{distribution}.pt")
        row = {**case, "grid": list(lowering.toolchain_requirements["grid"]),
               "profile": compiler.profile(assessment, compiled_resources=resources).as_dict()}
        rows.append(row)
        _write(directory / "launch.json", {"entry_point": compilation.entry_point,
                                            "threads_per_cta": compilation.threads_per_cta,
                                            "dynamic_shared_bytes": compilation.dynamic_shared_bytes,
                                            "grid": row["grid"]})
    versions = {row["profile"]["compiled_resources"]["compiler_version"] for row in rows}
    inspectors = {row["profile"]["compiled_resources"]["inspector_version"] for row in rows}
    if len(versions) != 1 or len(inspectors) != 1:
        raise ValueError("compilation context changed within local stage")
    _write(stage / "observations.json", {"schema_version": 1, "rows": rows,
                                          "compiler_version": versions.pop(),
                                          "inspector_version": inspectors.pop()})
    artifacts = {path.relative_to(stage).as_posix(): path.relative_to(stage).as_posix()
                 for path in sorted(stage.rglob("*")) if path.is_file() and path != result}
    _write(result, {"schema": "kernelinfra.stage-result.v1", "status": "passed",
                    "validity": "valid", "summary": "CPU calibration preparation passed",
                    "workloads": [], "artifacts": artifacts,
                    "metrics": {"case_count": len(rows), "scope": "compilation and CPU oracle only"}})


def _prepared(run, plan):
    """Read the completed local stage; the fitter audits retained artifact bytes."""
    stage = run / "stages/compile"
    receipt = _read(stage / "receipt.json")
    if receipt["execution"] != "local" or receipt["exit_code"] != 0 or not receipt["judge_result_valid"]:
        raise ValueError("CPU calibration compile did not pass")
    if _read(stage / "plan.json") != plan or (stage / "collector.py").read_bytes() != Path(__file__).read_bytes():
        raise ValueError("CPU calibration compile binding differs")
    observations = _read(stage / "observations.json")
    if len(observations["rows"]) != len(plan["cases"]):
        raise ValueError("CPU calibration pool differs")
    resources = {}
    for row in observations["rows"]:
        resource = CompiledResources.from_dict(row["profile"]["compiled_resources"])
        if resource.source_sha256 in resources:
            raise ValueError("CPU calibration pool repeats a compiled source")
        resources[resource.source_sha256] = resource
    return stage, observations, resources


def _broker_parent(plan):
    """Admit only a direct child of the owning broker on the host."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect("/tmp/agent-gpu-broker.sock")
        peer = struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
    if os.getppid() != peer[0] or os.geteuid() != peer[1] or peer[1] != plan["broker_uid"]:
        raise RuntimeError("collection requires the direct broker execution principal")
    return peer


def _node_broker_assignment(run, stage_id, physical_gpu, *, timeout_s=2.0):
    """Read the daemon's accepted broker job after its started event."""
    deadline = time.monotonic() + timeout_s
    while True:
        state = _read(run / "state.json")
        job_id = state.get("broker_job_id")
        if (state.get("run_id") == os.environ.get("KERNELINFRA_RUN_ID")
                and state.get("stage_id") == stage_id
                and state.get("state") == "running"
                and isinstance(job_id, str) and job_id.startswith("gpuq-")
                and state.get("gpu_ids") == [physical_gpu]):
            return job_id
        if time.monotonic() >= deadline:
            raise ValueError("daemon broker assignment does not match this running stage")
        time.sleep(0.05)


def _run_named_container(command, name):
    """Keep a task-owned Docker child inside the stage process lifecycle."""
    active = True

    def cleanup():
        if active:
            subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=15, check=False)

    atexit.register(cleanup)

    def stop(signum, _frame):
        cleanup()
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, stop)
    result = subprocess.run(command, check=False)
    if result.returncode:
        raise RuntimeError(f"container calibration exited {result.returncode}")
    active = False


def _compile_container():
    """Run the pinned toolchain without exposing any GPU to the local stage."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("GPUQ_JOB_ID"):
        raise ValueError("local calibration compile must not inherit a GPU allocation")
    run = _external(os.environ["KERNELINFRA_RUN_DIR"])
    stage = _external(os.environ["KERNELINFRA_STAGE_DIR"])
    if (stage != run / "stages/compile"
            or _external(os.environ["KERNELINFRA_CANDIDATE_DIR"]) != run / "candidate"
            or _external(os.environ["KERNELINFRA_RESULT"]) != stage / "result.json"
            or os.environ.get("KERNELINFRA_STAGE_KIND") != "compile"
            or os.environ.get("KERNELINFRA_STAGE_ID") != "compile"):
        raise ValueError("local container stage binding differs")
    plan, _ = _check_plan(run / "candidate")
    scratch = stage / "scratch"
    scratch.mkdir()
    name = f"cake-cost-{os.getpid()}-compile"
    command = ["docker", "run", "--rm", "--network", "none", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--pids-limit", "256",
               "--name", name, "--label", f"cake-cost.run_id={os.environ['KERNELINFRA_RUN_ID']}",
               "--label", "cake-cost.stage_id=compile",
               "--user", f"{os.geteuid()}:{os.getegid()}",
               "-v", f"{ROOT}:{ROOT}:ro", "-v", f"{run}:{run}", "-v", f"{scratch}:/tmp",
               "-w", str(ROOT), "-e", "NVIDIA_VISIBLE_DEVICES=void",
               "-e", "CUDA_VISIBLE_DEVICES=", "-e", "HOME=/tmp/fibhome"]
    for key in ("KERNELINFRA_RUN_ID", "KERNELINFRA_RUN_DIR", "KERNELINFRA_TASK",
                "KERNELINFRA_CANDIDATE_DIR", "KERNELINFRA_STAGE_ID", "KERNELINFRA_STAGE_KIND",
                "KERNELINFRA_STAGE_DIR", "KERNELINFRA_RESULT"):
        command.extend(["-e", f"{key}={os.environ[key]}"])
    command.extend([plan["container_image_id"], "python3", str(ROOT / "tools/calibrate_empirical_cost.py"), "compile"])
    _run_named_container(command, name)


def _collect_container():
    """Map exactly one broker-assigned GPU into a pinned container image."""
    run = _external(os.environ["KERNELINFRA_RUN_DIR"])
    stage = _external(os.environ["KERNELINFRA_STAGE_DIR"])
    kind = os.environ.get("KERNELINFRA_STAGE_KIND")
    stage_id = "correctness" if kind == "correctness" else "collection" if kind == "profile" else None
    if (stage_id is None or stage != run / "stages" / stage_id
            or os.environ.get("KERNELINFRA_STAGE_ID") != stage_id
            or _external(os.environ["KERNELINFRA_CANDIDATE_DIR"]) != run / "candidate"
            or _external(os.environ["KERNELINFRA_RESULT"]) != stage / "result.json"):
        raise ValueError("container calibration stage binding differs")
    plan = _read(run / "candidate/plan.json")
    if (plan.get("state") != "frozen"
            or sha256(Path(__file__).read_bytes()).hexdigest() != plan.get("collector_sha256")
            or not isinstance(plan.get("container_image_id"), str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", plan["container_image_id"]) is None):
        raise ValueError("container calibration frozen source or image differs")
    peer = _broker_parent(plan)
    physical = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if re.fullmatch(r"[0-7]", physical) is None:
        raise ValueError("container calibration requires one broker-assigned GPU")
    job_id = _node_broker_assignment(run, stage_id, int(physical))
    context = {"broker_peer": list(peer), "broker_job_id": job_id,
               "physical_gpu": int(physical), "run_id": os.environ["KERNELINFRA_RUN_ID"],
               "stage_id": stage_id}
    _write(stage / "broker-container.json", context)
    scratch = stage / "scratch"
    scratch.mkdir()
    name = f"cake-cost-{os.getpid()}-{stage_id}"
    command = ["docker", "run", "--rm", "--network", "none", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--pids-limit", "256",
               "--name", name, "--label", f"cake-cost.run_id={context['run_id']}",
               "--label", f"cake-cost.stage_id={stage_id}",
               "--gpus", f"device={physical}", "--user", f"{os.geteuid()}:{os.getegid()}",
               "-v", f"{ROOT}:{ROOT}:ro", "-v", f"{run}:{run}", "-v", f"{scratch}:/tmp",
               "-w", str(ROOT), "-e", "CUDA_VISIBLE_DEVICES=0", "-e", "HOME=/tmp/fibhome",
               "-e", "CAKE_BROKER_CONTAINER=1", "-e", f"GPUQ_JOB_ID={context['broker_job_id']}",
               "-e", f"CAKE_PHYSICAL_GPU={physical}"]
    for key in ("KERNELINFRA_RUN_ID", "KERNELINFRA_RUN_DIR", "KERNELINFRA_TASK",
                "KERNELINFRA_CANDIDATE_DIR", "KERNELINFRA_STAGE_ID", "KERNELINFRA_STAGE_KIND",
                "KERNELINFRA_STAGE_DIR", "KERNELINFRA_RESULT"):
        command.extend(["-e", f"{key}={os.environ[key]}"])
    command.extend([plan["container_image_id"], "python3", str(ROOT / "tools/calibrate_empirical_cost.py"), "collect"])
    _run_named_container(command, name)


def _collect():
    import importlib.metadata

    run = _external(os.environ["KERNELINFRA_RUN_DIR"])
    stage = _external(os.environ["KERNELINFRA_STAGE_DIR"])
    kind = os.environ["KERNELINFRA_STAGE_KIND"]
    stage_id = "correctness" if kind == "correctness" else "collection" if kind == "profile" else None
    if (stage_id is None or stage != run / "stages" / stage_id
            or os.environ.get("KERNELINFRA_STAGE_ID") != stage_id
            or _external(os.environ["KERNELINFRA_RESULT"]) != stage / "result.json"):
        raise ValueError("result must belong to the current stage")
    candidate = Path(os.environ["KERNELINFRA_CANDIDATE_DIR"]).resolve()
    if candidate != run / "candidate":
        raise ValueError("calibration candidate snapshot differs")
    plan = _read(candidate / "plan.json")
    if plan.get("state") != "frozen":raise ValueError("collection requires a frozen plan")
    if sha256(Path(__file__).read_bytes()).hexdigest() != plan["collector_sha256"]:
        raise ValueError("collector differs from frozen plan")
    (stage / "collector.py").write_bytes(Path(__file__).read_bytes())
    _write(stage / "plan.json", plan)
    compile_stage, prepared, resources_by_source = _prepared(run, plan)
    if os.environ.get("CAKE_BROKER_CONTAINER") == "1":
        context = _read(stage / "broker-container.json")
        if (context.get("broker_job_id") != os.environ.get("GPUQ_JOB_ID")
                or context.get("run_id") != os.environ.get("KERNELINFRA_RUN_ID")
                or context.get("stage_id") != stage_id
                or str(context.get("physical_gpu")) != os.environ.get("CAKE_PHYSICAL_GPU")
                or os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
                or os.geteuid() != plan["broker_uid"]):
            raise ValueError("container broker assignment differs")
        peer = context["broker_peer"]
        physical_gpu = context["physical_gpu"]
        job_id = context["broker_job_id"]
    else:
        peer = _broker_parent(plan)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if re.fullmatch(r"[0-7]", visible) is None:
            raise ValueError("direct calibration requires one broker-assigned GPU")
        physical_gpu = int(visible)
        job_id = _node_broker_assignment(run, stage_id, physical_gpu)
    _write(stage / "execution-context.json", {"broker_peer": peer,
        "broker_job_id": job_id, "physical_gpu": physical_gpu,
        "uid": os.geteuid(), "gid": os.getegid(),
        "run_id": os.environ["KERNELINFRA_RUN_ID"],
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
    compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
    target, binary_version_expected = _target_contract(compiler, plan)
    import torch
    from cuda.bindings import driver
    torch.set_num_threads(1)
    if (torch.cuda.device_count() != 1
            or torch.cuda.get_device_name(0) != plan["device_name"]
            or plan["device_name"] not in target.device_names
            or tuple(torch.cuda.get_device_capability(0)) != target.compute_capability):
        raise RuntimeError(f"target must be one broker-visible {plan['target']} device")
    properties = torch.cuda.get_device_properties(0)
    if properties.multi_processor_count != plan["multiprocessor_count"]:
        raise RuntimeError("multiprocessor count differs")
    (driver_version,) = _driver_call(driver, "cuDriverGetVersion", outputs=1)
    runtime = {"python": sys.version, "torch": str(torch.__version__), "torch_cuda": str(torch.version.cuda), "cuda_driver": str(driver_version), "cuda_bindings": importlib.metadata.version("cuda-bindings"), "triton_package": importlib.metadata.version("triton")}
    if any(runtime[key] != value for key, value in plan["expected_runtime"].items()):
        raise ValueError("runtime differs from the frozen collection boundary")
    stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
    rows, launches = [], []
    for index, case in enumerate(plan["cases"]):
        directory = compile_stage / f"{index:04d}"
        prepared_row = prepared["rows"][index]
        if any(prepared_row[key] != value for key, value in case.items()):
            raise ValueError("compiled case order differs from frozen plan")
        launch = _read(directory / "launch.json")
        resource = CompiledResources.from_dict(prepared_row["profile"]["compiled_resources"])
        if (resources_by_source.get(resource.source_sha256) != resource
                or launch != {"entry_point": resource.entry_point,
                              "threads_per_cta": resource.threads_per_cta,
                              "dynamic_shared_bytes": resource.dynamic_shared_bytes,
                              "grid": prepared_row["grid"]}):
            raise ValueError("compiled launch differs from retained resource observation")
        cubin = (directory / "kernel.cubin").read_bytes()
        oracle = [torch.load(directory / f"oracle-{distribution}.pt", map_location="cpu", weights_only=True)
                  for distribution in (0, 1)]
        cpu, expected, tolerance = oracle[0]["inputs"], oracle[0]["answer"], oracle[0]["tolerance"]
        inputs = [item.cuda() for item in cpu]
        output = torch.empty_like(expected, device="cuda")
        (module,) = _driver_call(driver, "cuModuleLoadData", cubin, outputs=1)
        (function,) = _driver_call(driver, "cuModuleGetFunction", module, resource.entry_point.encode(), outputs=1)
        (binary_version,) = _driver_call(driver, "cuFuncGetAttribute", driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_BINARY_VERSION, function, outputs=1)
        if int(binary_version) != binary_version_expected:raise ValueError("loaded binary target differs")
        if resource.dynamic_shared_bytes >= _DYNAMIC_SHARED_OPT_IN_THRESHOLD:
            _driver_call(driver, "cuFuncSetAttribute", function, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, resource.dynamic_shared_bytes, outputs=0)
        values = [ctypes.c_void_p(tensor.data_ptr()) for tensor in (*inputs, output)] + [ctypes.c_void_p(0), ctypes.c_void_p(0)]
        parameters = (ctypes.c_void_p * len(values))(*[ctypes.cast(ctypes.pointer(value), ctypes.c_void_p) for value in values])
        grid = launch["grid"]
        def invoke(function=function, parameters=parameters, values=values, resource=resource, grid=grid):
            _driver_call(driver, "cuLaunchKernel", function, *grid, resource.threads_per_cta, 1, 1, resource.dynamic_shared_bytes, stream, parameters, 0, outputs=0)
        deviations = []
        for distribution, prepared_oracle in enumerate(oracle):
            original, answer = prepared_oracle["inputs"], prepared_oracle["answer"]
            for tensor, source in zip(inputs, original, strict=True):tensor.copy_(source)
            invoke(); torch.cuda.synchronize()
            deviation = (output.cpu() - answer).abs().max().item()
            if not math.isfinite(deviation) or deviation > tolerance or not all(torch.equal(tensor.cpu(), source) for tensor, source in zip(inputs, original, strict=True)):
                raise RuntimeError(f"oracle/input-preservation failed: {case['id']}, distribution {distribution}, deviation {deviation}")
            deviations.append(deviation)
        for tensor, original in zip(inputs, cpu, strict=True):tensor.copy_(original)
        invoke(); torch.cuda.synchronize()
        row = {**prepared_row, "correct": True, "inputs_unchanged": True, "max_deviations": deviations, "samples_us": []}
        rows.append(row)
        launches.append((invoke, module, cpu, inputs, output, expected, tolerance))
        print(json.dumps({"loaded": case["id"], "count": index + 1}), flush=True)
    runtime.update(compiler_version=prepared["compiler_version"], inspector_version=prepared["inspector_version"])
    if kind == "profile":
        for invoke, *_ in launches:
            for _ in range(plan["sampling"]["warmup"]):invoke()
        torch.cuda.synchronize()
        flush = torch.empty(plan["sampling"]["l2_flush_bytes"], dtype=torch.uint8, device="cuda")
        from torch.profiler import profile, ProfilerActivity
        for repetition in range(plan["sampling"]["repetitions"]):
            order = []
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as observation:
                for round_index in range(plan["sampling"]["rounds"]):
                    shift = (round_index + repetition * 7) % len(rows)
                    indices = [(shift + offset) % len(rows) for offset in range(len(rows))]
                    if repetition % 2:indices.reverse()
                    for index in indices:
                        order.append(rows[index]["id"])
                        flush.zero_(); launches[index][0]()
                torch.cuda.synchronize()
            observation.export_chrome_trace(str(stage / f"cupti-trace-{repetition}.json"))
            _write(stage / f"launch-order-{repetition}.json", order)
    quality = True
    for row, (_, module, cpu, inputs, output, expected, tolerance) in zip(rows, launches, strict=True):
        deviation = (output.cpu() - expected).abs().max().item()
        row["correct"] = math.isfinite(deviation) and deviation <= tolerance
        row["inputs_unchanged"] = all(torch.equal(tensor.cpu(), original) for tensor, original in zip(inputs, cpu, strict=True))
        row["quality_passed"] = row["correct"] and row["inputs_unchanged"]
        quality &= row["quality_passed"]
        _driver_call(driver, "cuModuleUnload", module, outputs=0)
    _write(stage / "observations.json", {"schema_version": 1, "runtime": runtime,
                                          "device_checks_passed": quality, "rows": rows})
    artifacts = {"observations": "observations.json", "plan": "plan.json", "collector": "collector.py", "execution_context": "execution-context.json"}
    if os.environ.get("CAKE_BROKER_CONTAINER") == "1":
        artifacts["broker_container"] = "broker-container.json"
    for path in sorted(stage.glob("*trace-*.json")):artifacts[path.stem] = path.name
    for path in sorted(stage.glob("launch-order-*.json")):artifacts[path.stem] = path.name
    _write(os.environ["KERNELINFRA_RESULT"], {"schema": "kernelinfra.stage-result.v1", "status": "passed" if quality else "failed", "validity": "valid" if quality else "unknown", "summary": "device checks and trace capture passed; timing audit follows off device" if quality else "device checks failed; do not fit", "workloads": [{"id": row["id"], "correct": row["correct"]} for row in rows] if kind == "correctness" else [], "artifacts": artifacts, "metrics": {"device_checks_passed": quality, "case_count": len(rows)}})


def _fit(run, output):
    output = _external(output)
    if output.exists():raise ValueError("fitting output must be a new external directory")
    run_result = _read(run / "result.json")
    if run_result["outcome"] != "completed" or run_result["validity"] != "valid":raise ValueError("whole collection did not pass")
    outcomes = {row["id"]: row for row in run_result["stages"]}
    if any(outcomes[name]["status"] != "passed" or outcomes[name]["validity"] != "valid" for name in ("compile", "correctness", "collection")):
        raise ValueError("required stage did not pass")
    stage = run / "stages/collection"
    plan, observed = _read(stage / "plan.json"), _read(stage / "observations.json")
    if plan.get("state") != "frozen" or sha256(Path(__file__).read_bytes()).hexdigest() != plan["collector_sha256"]:
        raise ValueError("fitting must use the frozen collection instrument")
    compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
    _target_contract(compiler, plan)
    reference = compiler.assess_file(run / "stages/compile/0000/schedule.json")
    if reference.target != plan["target"]:
        raise ValueError("retained target differs from calibration plan")
    if reference.compiler_revision_id != plan["compiler_revision_id"]:
        raise ValueError(f"fitting Compiler Revision differs: expected {plan['compiler_revision_id']!r}, observed {reference.compiler_revision_id!r}")
    if not observed["device_checks_passed"]:raise ValueError("device checks failed before fitting")
    rows = observed["rows"]
    for row, case in zip(rows, plan["cases"], strict=True):
        if (any(row[key] != value for key, value in case.items())
                or not row["correct"] or not row["inputs_unchanged"]
                or row["samples_us"] != []):
            raise ValueError("collected domain or correctness differs")
    for repetition in range(plan["sampling"]["repetitions"]):
        for row, samples in zip(rows, _trace_samples(stage, repetition, plan, rows), strict=True):
            row["samples_us"].append(samples)
    for row in rows:
        medians = []
        if len(row["samples_us"]) != plan["sampling"]["repetitions"]:raise ValueError("repetition count differs")
        for values in row["samples_us"]:
            if len(values) != plan["sampling"]["rounds"] or statistics.pstdev(values) / statistics.mean(values) > plan["acceptance"]["maximum_cohort_cv"]:raise ValueError("sample quality differs")
            medians.append(statistics.median(values))
        if max(medians) / min(medians) > plan["acceptance"]["maximum_repeat_median_ratio"]:raise ValueError("repetition drift differs")
        row["kernel_us"] = statistics.median(medians)
    for name in ("compile", "correctness", "collection"):
        receipt = _read(run / f"stages/{name}/receipt.json")
        if (receipt["execution"] != ("local" if name == "compile" else "broker")
                or receipt["exit_code"] != 0 or not receipt["judge_result_valid"]
                or (name != "compile" and not receipt["broker_job_id"])):
            raise ValueError("stage receipt differs")
        if name != "compile":
            context = _read(run / f"stages/{name}/execution-context.json")
            if (context.get("broker_job_id") != receipt["broker_job_id"]
                    or context.get("run_id") != run_result["run_id"]
                    or context.get("physical_gpu") not in receipt["gpu_ids"]
                    or context.get("uid") != plan["broker_uid"]):
                raise ValueError("broker assignment differs from retained stage receipt")
            container_context = run / f"stages/{name}/broker-container.json"
            if container_context.exists():
                admitted = _read(container_context)
                if any(admitted[key] != context[key] for key in ("broker_job_id", "physical_gpu", "run_id")) or admitted["stage_id"] != name:
                    raise ValueError("container broker assignment differs from device stage")
    if _read(run / "stages/correctness/observations.json")["runtime"] != observed["runtime"]:raise ValueError("runtime changed between stages")
    _bind_artifacts(run, plan, rows, compiler)
    specifications = {spec["id"]: spec for spec in plan["curves"]}
    if len(specifications) != len(plan["curves"]) or set(specifications) != {row["curve_id"] for row in rows}:
        raise ValueError("curve ownership differs")
    for row in rows:
        buffers = {buffer["name"]: buffer for buffer in row["template"]["buffers"]}
        if type(row["extent"]) is not int or any(buffers[binding["buffer"]]["shape"][binding["dimension"]] != row["extent"] for binding in specifications[row["curve_id"]]["varying_dimensions"]):
            raise ValueError("measured extent differs from the declared Schedule dimensions")
    document = {"schema_version": 3, "model_id": plan["model_id"], "compiler_revision_id": plan["compiler_revision_id"], "target": plan["target"], "context": {"timer": "PyTorch Kineto CUPTI GPU kernel activity", "cache_protocol": f"{plan['sampling']['l2_flush_bytes']}-byte zeroing before each sample on same stream", "runtime": observed["runtime"], "input_scope": plan["input_scope"]}, "reported_evidence": {"run_id": run_result["run_id"], "scope": "fresh-measurement holdout; conditional empirical prediction, not candidate acceptance"}, "curves": []}
    for spec in plan["curves"]:
        group = [row for row in rows if row["curve_id"] == spec["id"]]
        splits = {split: {row["extent"] for row in group if row["split"] == split} for split in ("fit", "calibration", "audit")}
        if any(not extents for extents in splits.values()) or splits["fit"] & splits["calibration"] or splits["fit"] & splits["audit"] or splits["calibration"] & splits["audit"]:raise ValueError("fit/calibration/audit ownership differs")
        fit = sorted([row for row in group if row["split"] == "fit"], key=lambda row: row["extent"])
        document["curves"].append({"template": fit[0]["template"], "varying_dimensions": spec["varying_dimensions"], "extent_multiple": spec["extent_multiple"], "points": [{"extent": row["extent"], "kernel_us": row["kernel_us"]} for row in fit], "relative_error_envelope": 0.0})
    model = EmpiricalCostModel(document)
    def predict(row):
        value = model.estimate(row["template"], compiler_revision_id=plan["compiler_revision_id"], target=plan["target"], compiled_compiler_version=observed["runtime"]["compiler_version"])
        if not value["covered"]:raise ValueError(value["reason"])
        return value
    # Every fitted observation must describe the same template as its curve too.
    for row in rows:
        if row["split"] == "fit":predict(row)
    for index, spec in enumerate(plan["curves"]):
        errors = [abs(row["kernel_us"] / predict(row)["predicted_kernel_us"] - 1) for row in rows if row["curve_id"] == spec["id"] and row["split"] == "calibration"]
        document["curves"][index]["relative_error_envelope"] = max(errors) + plan["model_acceptance"]["envelope_allowance"]
    model = EmpiricalCostModel(document)
    audit = []
    for row in rows:
        if row["split"] == "audit":
            value = predict(row)
            audit.append({"id": row["id"], "workload_id": row["workload_id"], "observed_us": row["kernel_us"], "predicted_us": value["predicted_kernel_us"], "range_us": value["empirical_range_us"], "relative_error": abs(value["predicted_kernel_us"] / row["kernel_us"] - 1)})
    limits = plan["model_acceptance"]
    regrets = _candidate_set_regrets(audit, limits["maximum_candidates_per_turn"])
    metrics = {"audit_case_count": len(audit), "candidate_set_count": len(regrets),
               "mean_relative_error": statistics.mean(row["relative_error"] for row in audit),
               "max_relative_error": max(row["relative_error"] for row in audit),
               "max_top2_regret_ratio": max((row["top2_regret_ratio"] for row in regrets), default=None)}
    passed = bool(regrets) and metrics["mean_relative_error"] <= limits["maximum_mape"] and metrics["max_relative_error"] <= limits["maximum_relative_error"] and metrics["max_top2_regret_ratio"] <= limits["maximum_top2_regret_ratio"]
    output.mkdir(exist_ok=False)
    _write(output / "audit.json", {"passed": passed, "metrics": metrics, "audit": audit, "regrets": regrets, "run_id": run_result["run_id"]})
    if passed:
        document["reported_evidence"].update(validation=metrics)
        _write(output / "model.json", document)
    print(json.dumps({"passed": passed, **metrics}, indent=2))
    return 0 if passed else 1


def _bind_artifacts(run, plan, rows, compiler):
    """Replay canonical candidates and reuse the existing compiled-report owner."""
    target, _ = _target_contract(compiler, plan)
    candidate = (run / "candidate").resolve()
    if _read(candidate / "plan.json") != plan:
        raise ValueError("stage plan differs from the candidate snapshot")
    compile_stage = run / "stages/compile"
    if (_read(compile_stage / "plan.json") != plan
            or (compile_stage / "collector.py").read_bytes() != Path(__file__).read_bytes()):
        raise ValueError("local compile binding differs")
    observations = load_compiled_resources(compile_stage / "observations.json")
    compiled_rows = _read(compile_stage / "observations.json")["rows"]
    phase_rows = {}
    for phase in ("correctness", "collection"):
        directory = run / "stages" / phase
        if _read(directory / "plan.json") != plan:
            raise ValueError("stage plans differ")
        if (directory / "collector.py").read_bytes() != Path(__file__).read_bytes():
            raise ValueError("stage collector differs from the frozen fitter")
        phase_document = _read(directory / "observations.json")
        phase_rows[phase] = phase_document["rows"]
        if phase_document["device_checks_passed"] is not True or any(row["correct"] is not True or row["inputs_unchanged"] is not True for row in phase_rows[phase]):
            raise ValueError("stage correctness or input preservation differs")
        if len(phase_rows[phase]) != len(rows) or len(compiled_rows) != len(rows):
            raise ValueError("stage case count differs")
    for index, row in enumerate(rows):
        path = (candidate / row["schedule"]).resolve()
        if candidate not in path.parents:
            raise ValueError("Schedule escapes candidate snapshot")
        assessment = compiler.assess_file(path)
        if assessment.target != plan["target"]:
            raise ValueError("candidate target differs from calibration plan")
        if assessment.compiler_revision_id != plan["compiler_revision_id"]:
            raise ValueError(f"candidate Compiler Revision differs: expected {plan['compiler_revision_id']!r}, observed {assessment.compiler_revision_id!r}")
        lowering = compiler.lower(assessment)
        expected = json.loads(assessment.schedule_bytes)
        compiled_directory = compile_stage / f"{index:04d}"
        if json.dumps(_read(compiled_directory / "schedule.json"), sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise ValueError("local compiled Schedule differs from canonical candidate snapshot")
        resource = observations.get(lowering.source_sha256)
        if resource is None or resource != CompiledResources.from_dict(compiled_rows[index]["profile"]["compiled_resources"]):
            raise ValueError("local compiled observation differs from canonical candidate lowering")
        requirements = lowering.toolchain_requirements
        if any(compiled_rows[index][key] != value for key, value in plan["cases"][index].items()):
            raise ValueError("local compiled case differs from frozen plan")
        if ((resource.target, resource.entry_point, resource.threads_per_cta) !=
                (assessment.target, requirements["kernel_entry_point"], requirements["compile_options"]["num_warps"] * target.warp_size)
                or compiled_rows[index]["grid"] != list(requirements["grid"])):
            raise ValueError("local compiled launch differs from canonical lowering")
        if _read(compiled_directory / "launch.json") != {
                "entry_point": resource.entry_point,
                "threads_per_cta": resource.threads_per_cta,
                "dynamic_shared_bytes": resource.dynamic_shared_bytes,
                "grid": compiled_rows[index]["grid"]}:
            raise ValueError("local compiled launch metadata differs")
        for phase in ("correctness", "collection"):
            recorded = phase_rows[phase][index]
            if any(recorded[key] != row[key] for key in plan["cases"][index]):
                raise ValueError("stage case ownership differs")
            if resource != CompiledResources.from_dict(recorded["profile"]["compiled_resources"]):
                raise ValueError("compiled observation differs from canonical candidate lowering")
            if recorded["grid"] != compiled_rows[index]["grid"]:
                raise ValueError("recorded launch differs from canonical lowering")
        row["template"] = expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    check = sub.add_parser("check-plan", help="validate a frozen candidate pool without a GPU")
    check.add_argument("candidate", type=Path)
    sub.add_parser("compile", help="CPU-only preparation in a GPU Infra local stage")
    sub.add_parser("compile-container", help="run CPU preparation in the pinned image without a GPU")
    sub.add_parser("collect")
    sub.add_parser("collect-container", help="broker-owned GPU stage with exact container mapping")
    fit = sub.add_parser("fit")
    fit.add_argument("run", type=Path)
    fit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "check-plan":
        plan, _ = _check_plan(args.candidate)
        print(json.dumps({"target": plan["target"], "case_count": len(plan["cases"]), "curve_count": len(plan["curves"])}))
        return 0
    if args.action == "fit":return _fit(args.run, args.output)
    try:
        if args.action == "compile":
            _prepare()
        elif args.action == "compile-container":
            _compile_container()
        elif args.action == "collect-container":
            _collect_container()
        else:
            _collect()
        return 0
    except Exception as error:
        traceback.print_exc()
        result = _external(os.environ["KERNELINFRA_RESULT"])
        if not result.exists():_write(result, {"schema": "kernelinfra.stage-result.v1", "status": "failed", "validity": "unknown", "summary": str(error), "workloads": [], "artifacts": {}, "metrics": {}})
        return 1


if __name__ == "__main__":raise SystemExit(main())
