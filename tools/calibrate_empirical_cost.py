#!/usr/bin/env python3
"""Collect a frozen FMA/GEMM calibration through GPU Infra, or fit/audit it on CPU.

Plans, candidates and output directories are external artifacts. A plan binds this
collector's bytes and the released Compiler. The plan owns domains and thresholds;
the workload oracle below owns the two explicitly supported evaluation contracts.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import socket
import statistics
import struct
import sys
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


def _collect():
    import importlib.metadata

    stage = _external(os.environ["KERNELINFRA_STAGE_DIR"])
    if _external(os.environ["KERNELINFRA_RESULT"]) != stage / "result.json":
        raise ValueError("result must belong to the current stage")
    candidate = Path(os.environ["KERNELINFRA_CANDIDATE_DIR"]).resolve()
    plan = _read(candidate / "plan.json")
    if plan.get("state") != "frozen":raise ValueError("collection requires a frozen plan")
    if sha256(Path(__file__).read_bytes()).hexdigest() != plan["collector_sha256"]:
        raise ValueError("collector differs from frozen plan")
    (stage / "collector.py").write_bytes(Path(__file__).read_bytes())
    _write(stage / "plan.json", plan)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect("/tmp/agent-gpu-broker.sock")
        peer = struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
    if os.getppid() != peer[0] or os.geteuid() != peer[1] or peer[1] != plan["broker_uid"]:
        raise RuntimeError("collection requires the direct broker execution principal")
    kind = os.environ["KERNELINFRA_STAGE_KIND"]
    if kind not in {"correctness", "profile"}:
        raise ValueError("collection requires a correctness/profile stage")
    _write(stage / "execution-context.json", {"broker_peer": peer, "uid": os.geteuid(), "gid": os.getegid(), "run_id": os.environ["KERNELINFRA_RUN_ID"], "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
    compiler = Compiler.load(ROOT, ROOT / "compiler/revision.lock.json")
    import torch
    from cuda.bindings import driver
    torch.set_num_threads(1)
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != plan["device_name"] or list(torch.cuda.get_device_capability(0)) != [10, 0]:
        raise RuntimeError("target must be one broker-visible B200 sm_100a")
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
        directory = stage / f"{index:04d}"
        directory.mkdir()
        schedule_path = (candidate / case["schedule"]).resolve()
        if candidate not in schedule_path.parents:
            raise ValueError("Schedule must belong to the candidate snapshot")
        document = _read(schedule_path)
        assessment = compiler.assess(document)
        if (assessment.compiler_revision_id != plan["compiler_revision_id"] or assessment.compiler_revision_sha256 != plan["compiler_revision_sha256"]):
            raise ValueError("Compiler Revision differs")
        lowering = compiler.lower(assessment)
        signature = {"a": "*fp32", "b": "*fp32", "c": "*fp32", "y": "*fp32"} if case["family"] == "fma" else {"a": "*bf16", "b": "*bf16", "bias": "*fp32", "c": "*fp32"}
        if lowering.toolchain_requirements["signature"] != signature or list(lowering.toolchain_requirements["signature"]) != list(signature):
            raise ValueError("evaluation contract requires its exact four-pointer ABI")
        compilation = compile_triton(lowering.source.encode(), lowering.toolchain_requirements)
        resources = inspect_triton_resources(compilation, plan["cuobjdump"])
        _write(directory / "schedule.json", json.loads(assessment.schedule_bytes))
        (directory / "lowered.py").write_bytes(compilation.source)
        (directory / "kernel.cubin").write_bytes(compilation.artifacts["cubin"])
        (directory / "kernel.ptx").write_bytes(compilation.artifacts["ptx"])
        cpu, expected, tolerance = _cpu_case(case, document, torch, 0)
        inputs = [item.cuda() for item in cpu]
        output = torch.empty_like(expected, device="cuda")
        (module,) = _driver_call(driver, "cuModuleLoadData", compilation.artifacts["cubin"], outputs=1)
        (function,) = _driver_call(driver, "cuModuleGetFunction", module, compilation.entry_point.encode(), outputs=1)
        (binary_version,) = _driver_call(driver, "cuFuncGetAttribute", driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_BINARY_VERSION, function, outputs=1)
        if int(binary_version) != 100:raise ValueError("loaded binary target differs")
        if compilation.dynamic_shared_bytes >= _DYNAMIC_SHARED_OPT_IN_THRESHOLD:
            _driver_call(driver, "cuFuncSetAttribute", function, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, compilation.dynamic_shared_bytes, outputs=0)
        values = [ctypes.c_void_p(tensor.data_ptr()) for tensor in (*inputs, output)] + [ctypes.c_void_p(0), ctypes.c_void_p(0)]
        parameters = (ctypes.c_void_p * len(values))(*[ctypes.cast(ctypes.pointer(value), ctypes.c_void_p) for value in values])
        grid = list(lowering.toolchain_requirements["grid"])
        def invoke(function=function, parameters=parameters, values=values, compilation=compilation, grid=grid):
            _driver_call(driver, "cuLaunchKernel", function, *grid, compilation.threads_per_cta, 1, 1, compilation.dynamic_shared_bytes, stream, parameters, 0, outputs=0)
        deviations = []
        for distribution in (0, 1):
            original, answer, _ = _cpu_case(case, document, torch, distribution)
            for tensor, source in zip(inputs, original, strict=True):tensor.copy_(source)
            invoke(); torch.cuda.synchronize()
            deviation = (output.cpu() - answer).abs().max().item()
            if not math.isfinite(deviation) or deviation > tolerance or not all(torch.equal(tensor.cpu(), source) for tensor, source in zip(inputs, original, strict=True)):
                raise RuntimeError(f"oracle/input-preservation failed: {case['id']}, distribution {distribution}, deviation {deviation}")
            deviations.append(deviation)
        for tensor, original in zip(inputs, cpu, strict=True):tensor.copy_(original)
        invoke(); torch.cuda.synchronize()
        row = {**case, "grid": grid, "profile": compiler.profile(assessment, compiled_resources=resources).as_dict(), "correct": True, "inputs_unchanged": True, "max_deviations": deviations, "samples_us": []}
        rows.append(row)
        launches.append((invoke, module, cpu, inputs, output, expected, tolerance))
        print(json.dumps({"prepared": case["id"], "count": index + 1}), flush=True)
    versions = {row["profile"]["compiled_resources"]["compiler_version"] for row in rows}
    inspectors = {row["profile"]["compiled_resources"]["inspector_version"] for row in rows}
    if len(versions) != 1 or len(inspectors) != 1:raise ValueError("compilation context changed within collection")
    runtime.update(compiler_version=versions.pop(), inspector_version=inspectors.pop())
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
            for row, samples in zip(rows, _trace_samples(stage, repetition, plan, rows), strict=True):row["samples_us"].append(samples)
    quality = True
    for row, (_, module, cpu, inputs, output, expected, tolerance) in zip(rows, launches, strict=True):
        deviation = (output.cpu() - expected).abs().max().item()
        row["correct"] = math.isfinite(deviation) and deviation <= tolerance
        row["inputs_unchanged"] = all(torch.equal(tensor.cpu(), original) for tensor, original in zip(inputs, cpu, strict=True))
        row["quality_passed"] = row["correct"] and row["inputs_unchanged"]
        if kind == "profile":
            row["summaries"] = [{"median_us": statistics.median(values), "cv": statistics.pstdev(values) / statistics.mean(values)} for values in row["samples_us"]]
            medians = [item["median_us"] for item in row["summaries"]]
            row["repeat_median_ratio"] = max(medians) / min(medians)
            row["quality_passed"] &= row["repeat_median_ratio"] <= plan["acceptance"]["maximum_repeat_median_ratio"] and all(item["cv"] <= plan["acceptance"]["maximum_cohort_cv"] for item in row["summaries"])
        quality &= row["quality_passed"]
        _driver_call(driver, "cuModuleUnload", module, outputs=0)
    _write(stage / "observations.json", {"schema_version": 1, "runtime": runtime, "quality_passed": quality, "rows": rows})
    artifacts = {"observations": "observations.json", "plan": "plan.json", "collector": "collector.py", "execution_context": "execution-context.json"}
    for index in range(len(rows)):
        for role, name in (("schedule", "schedule.json"), ("source", "lowered.py"), ("cubin", "kernel.cubin"), ("ptx", "kernel.ptx")):
            artifacts[f"{index:04d}-{role}"] = f"{index:04d}/{name}"
    for path in sorted(stage.glob("*trace-*.json")):artifacts[path.stem] = path.name
    for path in sorted(stage.glob("launch-order-*.json")):artifacts[path.stem] = path.name
    _write(os.environ["KERNELINFRA_RESULT"], {"schema": "kernelinfra.stage-result.v1", "status": "passed" if quality else "failed", "validity": "valid" if quality else "unknown", "summary": "all cases passed" if quality else "quality gate failed; do not fit", "workloads": [{"id": row["id"], "correct": row["correct"]} for row in rows] if kind == "correctness" else [], "artifacts": artifacts, "metrics": {"quality_passed": quality, "case_count": len(rows)}})


def _fit(run, output):
    output = _external(output)
    if output.exists():raise ValueError("fitting output must be a new external directory")
    run_result = _read(run / "result.json")
    if run_result["outcome"] != "completed" or run_result["validity"] != "valid":raise ValueError("whole collection did not pass")
    outcomes = {row["id"]: row for row in run_result["stages"]}
    if any(outcomes[name]["status"] != "passed" or outcomes[name]["validity"] != "valid" for name in ("correctness", "collection")):
        raise ValueError("required stage did not pass")
    stage = run / "stages/collection"
    plan, observed = _read(stage / "plan.json"), _read(stage / "observations.json")
    if plan.get("state") != "frozen" or sha256(Path(__file__).read_bytes()).hexdigest() != plan["collector_sha256"]:
        raise ValueError("fitting must use the frozen collection instrument")
    compiler = Compiler.load(ROOT, ROOT / "compiler/revision.lock.json")
    reference = compiler.assess_file(stage / "0000/schedule.json")
    if reference.compiler_revision_id != plan["compiler_revision_id"] or reference.compiler_revision_sha256 != plan["compiler_revision_sha256"]:
        raise ValueError("fitting Compiler Revision differs")
    if not observed["quality_passed"]:raise ValueError("quality failed before fitting")
    rows = observed["rows"]
    for row, case in zip(rows, plan["cases"], strict=True):
        if any(row[key] != value for key, value in case.items()) or not row["correct"] or not row["inputs_unchanged"]:
            raise ValueError("collected domain or correctness differs")
        medians = []
        if len(row["samples_us"]) != plan["sampling"]["repetitions"]:raise ValueError("repetition count differs")
        for values in row["samples_us"]:
            if len(values) != plan["sampling"]["rounds"] or statistics.pstdev(values) / statistics.mean(values) > plan["acceptance"]["maximum_cohort_cv"]:raise ValueError("sample quality differs")
            medians.append(statistics.median(values))
        if max(medians) / min(medians) > plan["acceptance"]["maximum_repeat_median_ratio"]:raise ValueError("repetition drift differs")
        row["kernel_us"] = statistics.median(medians)
    for repetition in range(plan["sampling"]["repetitions"]):
        for row, values in zip(rows, _trace_samples(stage, repetition, plan, rows), strict=True):
            if row["samples_us"][repetition] != values:raise ValueError("retained durations differ from trace")
    for name in ("correctness", "collection"):
        receipt = _read(run / f"stages/{name}/receipt.json")
        if receipt["execution"] != "broker" or receipt["exit_code"] != 0 or not receipt["judge_result_valid"] or not receipt["broker_job_id"]:raise ValueError("stage receipt differs")
    if _read(run / "stages/correctness/observations.json")["runtime"] != observed["runtime"]:raise ValueError("runtime changed between stages")
    _bind_artifacts(run, plan, rows, compiler)
    specifications = {spec["id"]: spec for spec in plan["curves"]}
    if len(specifications) != len(plan["curves"]) or set(specifications) != {row["curve_id"] for row in rows}:
        raise ValueError("curve ownership differs")
    for row in rows:
        buffers = {buffer["name"]: buffer for buffer in row["template"]["buffers"]}
        if type(row["extent"]) is not int or any(buffers[binding["buffer"]]["shape"][binding["dimension"]] != row["extent"] for binding in specifications[row["curve_id"]]["varying_dimensions"]):
            raise ValueError("measured extent differs from the declared Schedule dimensions")
    document = {"schema_version": 2, "model_id": plan["model_id"], "compiler_revision_id": plan["compiler_revision_id"], "compiler_revision_sha256": plan["compiler_revision_sha256"], "target": "sm_100a", "context": {"timer": "PyTorch Kineto CUPTI GPU kernel activity", "cache_protocol": f"{plan['sampling']['l2_flush_bytes']}-byte zeroing before each sample on same stream", "runtime": observed["runtime"], "input_scope": plan["input_scope"]}, "reported_evidence": {"run_id": run_result["run_id"], "scope": "fresh-measurement holdout; conditional empirical prediction, not candidate acceptance"}, "curves": []}
    for spec in plan["curves"]:
        group = [row for row in rows if row["curve_id"] == spec["id"]]
        splits = {split: {row["extent"] for row in group if row["split"] == split} for split in ("fit", "calibration", "audit")}
        if any(not extents for extents in splits.values()) or splits["fit"] & splits["calibration"] or splits["fit"] & splits["audit"] or splits["calibration"] & splits["audit"]:raise ValueError("fit/calibration/audit ownership differs")
        fit = sorted([row for row in group if row["split"] == "fit"], key=lambda row: row["extent"])
        document["curves"].append({"template": fit[0]["template"], "varying_dimensions": spec["varying_dimensions"], "extent_multiple": spec["extent_multiple"], "points": [{"extent": row["extent"], "kernel_us": row["kernel_us"]} for row in fit], "relative_error_envelope": 0.0})
    model = EmpiricalCostModel(document)
    def predict(row):
        value = model.estimate(row["template"], compiler_revision_id=plan["compiler_revision_id"], compiler_revision_sha256=plan["compiler_revision_sha256"], target="sm_100a", compiled_compiler_version=observed["runtime"]["compiler_version"])
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
    regrets = []
    for workload_id in sorted({row["workload_id"] for row in audit}):
        group = sorted([row for row in audit if row["workload_id"] == workload_id], key=lambda row: row["predicted_us"])
        tie = group[1]["predicted_us"] == group[2]["predicted_us"]
        regrets.append({"workload_id": workload_id, "abstained": tie, "survivors": [row["id"] for row in (group if tie else group[:2])], "top2_regret_ratio": None if tie else min(row["observed_us"] for row in group[:2]) / min(row["observed_us"] for row in group)})
    decisive = [row["top2_regret_ratio"] for row in regrets if not row["abstained"]]
    metrics = {"audit_case_count": len(audit), "mean_relative_error": statistics.mean(row["relative_error"] for row in audit), "max_relative_error": max(row["relative_error"] for row in audit), "max_top2_regret_ratio": max(decisive) if decisive else None}
    limits = plan["model_acceptance"]
    passed = bool(decisive) and metrics["mean_relative_error"] <= limits["maximum_mape"] and metrics["max_relative_error"] <= limits["maximum_relative_error"] and metrics["max_top2_regret_ratio"] <= limits["maximum_top2_regret_ratio"]
    output.mkdir(exist_ok=False)
    _write(output / "audit.json", {"passed": passed, "metrics": metrics, "audit": audit, "regrets": regrets, "run_id": run_result["run_id"]})
    if passed:
        document["reported_evidence"].update(validation=metrics)
        _write(output / "model.json", document)
    print(json.dumps({"passed": passed, **metrics}, indent=2))
    return 0 if passed else 1


def _bind_artifacts(run, plan, rows, compiler):
    """Replay canonical candidates and reuse the existing compiled-report owner."""
    candidate = (run / "candidate").resolve()
    if _read(candidate / "plan.json") != plan:
        raise ValueError("stage plan differs from the candidate snapshot")
    observations = {}
    phase_rows = {}
    for phase in ("correctness", "collection"):
        directory = run / "stages" / phase
        if _read(directory / "plan.json") != plan:
            raise ValueError("stage plans differ")
        if (directory / "collector.py").read_bytes() != Path(__file__).read_bytes():
            raise ValueError("stage collector differs from the frozen fitter")
        # Validates retained source/CUBIN against their existing identities, including
        # missing files, source/binary drift and symlink/escape refusal.
        observations[phase] = load_compiled_resources(directory / "observations.json")
        phase_document = _read(directory / "observations.json")
        phase_rows[phase] = phase_document["rows"]
        if phase_document["quality_passed"] is not True or any(row["correct"] is not True or row["inputs_unchanged"] is not True for row in phase_rows[phase]):
            raise ValueError("stage correctness or input preservation differs")
        if len(phase_rows[phase]) != len(rows):
            raise ValueError("stage case count differs")
    for index, row in enumerate(rows):
        path = (candidate / row["schedule"]).resolve()
        if candidate not in path.parents:
            raise ValueError("Schedule escapes candidate snapshot")
        assessment = compiler.assess_file(path)
        if (assessment.compiler_revision_id != plan["compiler_revision_id"] or assessment.compiler_revision_sha256 != plan["compiler_revision_sha256"]):
            raise ValueError("candidate Compiler Revision differs")
        lowering = compiler.lower(assessment)
        expected = json.loads(assessment.schedule_bytes)
        bound = []
        for phase in ("correctness", "collection"):
            directory = run / "stages" / phase / f"{index:04d}"
            if json.dumps(_read(directory / "schedule.json"), sort_keys=True) != json.dumps(expected, sort_keys=True):
                raise ValueError("stage Schedule differs from canonical candidate snapshot")
            recorded = phase_rows[phase][index]
            if any(recorded[key] != row[key] for key in plan["cases"][index]):
                raise ValueError("stage case ownership differs")
            resource = observations[phase].get(lowering.source_sha256)
            if resource is None or resource != CompiledResources.from_dict(recorded["profile"]["compiled_resources"]):
                raise ValueError("compiled observation differs from canonical candidate lowering")
            requirements = lowering.toolchain_requirements
            if ((resource.target, resource.entry_point, resource.threads_per_cta) !=
                    (assessment.target, requirements["kernel_entry_point"], requirements["compile_options"]["num_warps"] * 32)
                    or json.dumps(recorded["grid"]) != json.dumps(list(requirements["grid"]))):
                raise ValueError("recorded launch differs from canonical lowering")
            bound.append(resource)
        if bound[0] != bound[1]:
            raise ValueError("correctness and collection compiled artifacts differ")
        row["template"] = expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("collect")
    fit = sub.add_parser("fit")
    fit.add_argument("run", type=Path)
    fit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "fit":return _fit(args.run, args.output)
    try:
        _collect()
        return 0
    except Exception as error:
        traceback.print_exc()
        result = _external(os.environ["KERNELINFRA_RESULT"])
        if not result.exists():_write(result, {"schema": "kernelinfra.stage-result.v1", "status": "failed", "validity": "unknown", "summary": str(error), "workloads": [], "artifacts": {}, "metrics": {}})
        return 1


if __name__ == "__main__":raise SystemExit(main())
