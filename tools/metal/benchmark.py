#!/usr/bin/env python3
"""Bounded local RMSNorm measurement with independent matched confirmation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from open_cake_ir.compiler import frontend
from open_cake_ir.lab.routing import route_rejection
from tools.metal import rmsnorm
from tools.metal.adapter import compile_runner
from tools.metal.check_correctness import evaluate_case, fresh_receipt, prepare_case, released_compiler, runtime_source

PROTOCOL = {
    "warmups": 3, "pilot_samples": 3, "target_command_seconds": 0.001,
    "batch_powers": [1, 2, 4, 8, 16, 32, 64, 128], "sweeps_per_round": 8,
    "search_rounds": 1, "confirmation_rounds": 2, "order_seed": 260906,
    "selection": "lowest search median GPU command-buffer time; lexical tie break",
    "pilot_scope": "handwritten reference slots only, before candidate comparison",
    "cache_policy": "warm buffer reuse; no cache flush",
    "dispatch_ordering": "MTLDispatchType.serial; one encoder and completion per batch",
    "host_interval": "encode through command completion; excludes output poison, validation and file I/O",
    "gpu_interval": "GPUStartTime to GPUEndTime for entire command buffer",
    "amortized_interval": "GPU command-buffer seconds divided by batch dispatches, not pure kernel latency",
    "null_ratio_bounds": [0.95, 1.05], "max_relative_iqr": 0.10,
    "material_gain": 1.05, "confirmation_rule": "search and both independent confirmations exceed material gain with passing noise controls",
    "profiling": "separate instrumented compute-stage observation after ordinary timing",
    "claim_scope": "engineering qualification under this local protocol; no p-value, full Study or serving claim",
}
REFERENCES = ("serial_reference", "simd_reference", "simd_reference_null")


def orders(candidate_ids: list[str]) -> list:
    rng = random.Random(PROTOCOL["order_seed"])
    result = []
    for round_index in range(3):
        slots = [*REFERENCES, *(candidate_ids if round_index == 0 else ["__selected__"])]
        sweeps = []
        for _ in range(PROTOCOL["sweeps_per_round"]):
            order = list(slots)
            rng.shuffle(order)
            sweeps.append(order)
        result.append(sweeps)
    return result


def describe(values: list[float]) -> dict:
    if len(values) < 2 or any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("timing/ratio samples must be finite and positive")
    median = statistics.median(values)
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return {"samples": len(values), "median": median, "min": min(values), "max": max(values),
            "relative_iqr": (quartiles[2] - quartiles[0]) / median}


def analyze(result: dict, job: dict) -> dict:
    if result.get("status") != "completed" or result.get("ordinary_samples_instrumented") is not False:
        raise ValueError("ordinary measurement did not complete without instrumentation")
    if len(job["orders"]) != 3 or any(len(r) != PROTOCOL["sweeps_per_round"] for r in job["orders"]):
        raise ValueError("matched round/sweep count differs from fixed protocol")
    selected = result.get("selected_candidate")
    candidates = [a["id"] for a in job["artifacts"] if a["origin"] == "compiler_generated"]
    if selected not in candidates:
        raise ValueError("selected candidate is not a compiled treatment")
    samples = result["raw_samples"]
    expected = [(r, s, p, selected if slot == "__selected__" else slot)
                for r, sweeps in enumerate(job["orders"]) for s, order in enumerate(sweeps)
                for p, slot in enumerate(order)]
    observed = [(s["round"], s["sweep"], s["position"], s["artifact"]) for s in samples]
    if observed != expected:
        raise ValueError("raw sample identity/order differs from frozen matched protocol")
    count = result["batch_dispatches"]
    if type(count) is not int or count not in PROTOCOL["batch_powers"]:
        raise ValueError("batch count is outside predeclared powers")
    for sample in samples:
        if sample["dispatches"] != count or sample["command_status"] != "completed":
            raise ValueError("sample dispatch count/completion differs")
        validation = sample.get("validation", {})
        if not isinstance(validation, dict) or validation.get("gpu_correctness") != "passed" or validation.get("input_immutability") != "passed":
            raise ValueError("ordinary sample lacks successful output/input validation")
        for field in ("warmed_host_call_seconds", "gpu_command_buffer_seconds", "amortized_dispatch_seconds"):
            if not math.isfinite(sample[field]) or sample[field] <= 0:
                raise ValueError(f"invalid timer: {field}")
        if not math.isclose(sample["amortized_dispatch_seconds"], sample["gpu_command_buffer_seconds"] / count, rel_tol=1e-12):
            raise ValueError("amortization differs from actual dispatch count")
    search_times = {id: describe([s["gpu_command_buffer_seconds"] for s in samples if s["round"] == 0 and s["artifact"] == id])
                    for id in candidates}
    observed_best = min(candidates, key=lambda id: (search_times[id]["median"], id))
    if selected != observed_best:
        raise ValueError("selection differs from predeclared search rule")
    by_slot = {(s["round"], s["sweep"], s["artifact"]): s for s in samples}
    arm_stats = {}
    null = []
    for r in range(3):
        ids = [*REFERENCES, *(candidates if r == 0 else [selected])]
        arm_stats[str(r)] = {id: {
            field: describe([by_slot[r, sweep, id][field] for sweep in range(PROTOCOL["sweeps_per_round"])])
            for field in ("warmed_host_call_seconds", "gpu_command_buffer_seconds", "amortized_dispatch_seconds")}
            for id in ids}
        ratio = describe([by_slot[r, sweep, "simd_reference"]["gpu_command_buffer_seconds"] /
                          by_slot[r, sweep, "simd_reference_null"]["gpu_command_buffer_seconds"]
                          for sweep in range(PROTOCOL["sweeps_per_round"])])
        limits = PROTOCOL["null_ratio_bounds"]
        quality = limits[0] <= ratio["median"] <= limits[1] and all(
            arm_stats[str(r)][id]["gpu_command_buffer_seconds"]["relative_iqr"] <= PROTOCOL["max_relative_iqr"]
            for id in ("simd_reference", "simd_reference_null"))
        null.append({"round": r, "paired_ratio": ratio, "passed": quality})
    comparisons = {}
    for reference in ("serial_reference", "simd_reference"):
        rounds = []
        for r in range(3):
            ratio = describe([by_slot[r, sweep, reference]["gpu_command_buffer_seconds"] /
                              by_slot[r, sweep, selected]["gpu_command_buffer_seconds"]
                              for sweep in range(PROTOCOL["sweeps_per_round"])])
            quality = null[r]["passed"] and all(
                arm_stats[str(r)][id]["gpu_command_buffer_seconds"]["relative_iqr"] <= PROTOCOL["max_relative_iqr"]
                for id in (reference, selected))
            rounds.append({"round": r, "purpose": "search" if r == 0 else "independent_confirmation",
                           "paired_reference_over_candidate_ratio": ratio, "noise_passed": quality})
        qualified = all(r["noise_passed"] and r["paired_reference_over_candidate_ratio"]["median"] > PROTOCOL["material_gain"] for r in rounds)
        comparisons[reference] = {"qualified_speedup": qualified, "rounds": rounds,
            "measurement_quality": "qualified_local_improvement" if qualified else "inconclusive_or_no_material_gain"}
        if qualified:
            comparisons[reference]["qualified_confirmation_ratio"] = statistics.median(
                r["paired_reference_over_candidate_ratio"]["median"] for r in rounds[1:])
    return {"selected_candidate": selected, "search": search_times, "arm_statistics": arm_stats,
            "null_control": null, "comparisons": comparisons,
            "calibrated_ranker": "unavailable; no ranking inversion inferred"}


def invoke_batch(binary: Path, receipt: Path, job: dict, *, compile_only: bool = False) -> dict:
    (receipt / "batch.json").write_text(json.dumps(job, indent=2) + "\n")
    start = time.perf_counter()
    result = subprocess.run([str(binary), "--compile-batch" if compile_only else "--batch", str(receipt / "batch.json")],
                            capture_output=True, text=True, timeout=600)
    (receipt / "batch.stdout.json").write_text(result.stdout)
    (receipt / "batch.stderr.txt").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError("Metal batch failed; see batch.stderr.txt and append-only events.jsonl")
    output = json.loads(result.stdout)
    output["process_wall_seconds"] = time.perf_counter() - start
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target", choices=("apple_gpu_family7", "apple_gpu_family8"),
                        default="apple_gpu_family8", help="exact target; no device fallback")
    args = parser.parse_args()
    receipt = fresh_receipt(args.output_root, prefix="metal-benchmark-")
    summary = {"status": "failed", "started_at": datetime.now(timezone.utc).isoformat(),
               "contract": rmsnorm.CONTRACT, "protocol": PROTOCOL, "candidates": [], "held_out_correctness": []}
    stage = "release"
    current = None
    try:
        summary["runtime_source"] = runtime_source()
        if not summary["runtime_source"]["tracked"] or not summary["runtime_source"]["clean"]:
            raise ValueError("benchmark requires committed clean runtime/example sources")
        compiler, lock, device_names = released_compiler(receipt, args.target)
        summary["compiler_revision_id"] = lock["revision_id"]
        summary["target"] = args.target
        (receipt / "protocol.json").write_text(json.dumps({"contract": rmsnorm.CONTRACT, "protocol": PROTOCOL}, indent=2) + "\n")
        begin = time.perf_counter()
        binary = compile_runner(receipt)
        summary["swift_host_build_seconds"] = time.perf_counter() - begin
        inputs, oracles = rmsnorm.inputs_and_oracle(*rmsnorm.PRIMARY_SHAPE, "uniform")
        artifacts = []
        stage = "assess"
        for formula in rmsnorm.FORMULAS:
            current = {"id": formula, "origin": "compiler_generated", "kind": "evaluation", "candidate_disposition": "pending", "gpu_correctness": "not_run"}
            summary["candidates"].append(current)
            directory = receipt / formula
            directory.mkdir()
            source_path = receipt / f"{formula}.py"
            source_path.write_text(rmsnorm.source(*rmsnorm.PRIMARY_SHAPE, formula, target=args.target))
            authored = frontend.read_schedule(source_path)
            current["authored_source"] = str(source_path)
            prepare_case(compiler, authored.document, inputs, oracles, directory, device_names, current)
            current["finding_locations"] = [asdict(location) if (location := authored.location_for(f["path"])) else None
                                            for f in current["findings"]]
            artifacts.append({"id": formula, "manifest_path": str(directory / "manifest.json"),
                              "oracle_path": str(directory / "oracle.json"), "origin": "compiler_generated"})
        for id in REFERENCES:
            stage = "reference_prepare"
            current = {"id": id, "origin": "handwritten_reference"}
            directory = receipt / id
            directory.mkdir()
            rmsnorm.reference(*rmsnorm.PRIMARY_SHAPE, "serial" if id == "serial_reference" else "simd",
                              inputs, directory, device_names, target=args.target)
            (directory / "oracle.json").write_text(json.dumps(oracles, allow_nan=False) + "\n")
            artifacts.append({"id": id, "manifest_path": str(directory / "manifest.json"),
                              "oracle_path": str(directory / "oracle.json"), "origin": "handwritten_reference"})
        job = {"artifacts": artifacts, "warmups": PROTOCOL["warmups"], "pilot_samples": PROTOCOL["pilot_samples"],
               "max_batch_dispatches": max(PROTOCOL["batch_powers"]), "target_command_seconds": PROTOCOL["target_command_seconds"],
               "orders": orders(list(rmsnorm.FORMULAS))}
        stage = "batch"
        current = None
        for candidate in summary["candidates"]:
            candidate.update(gpu_execution="requested", gpu_correctness="unknown", candidate_disposition="pending_runtime")
        observed = invoke_batch(binary, receipt, job)
        summary["runtime_observation"] = observed
        if observed.get("status") == "completed":
            for candidate in summary["candidates"]:
                if observed.get("correctness", {}).get(candidate["id"], {}).get("gpu_correctness") == "passed":
                    candidate.update(gpu_execution="completed", gpu_correctness="passed")
        stage = "measurement"
        summary["analysis"] = analyze(observed, job)
        selected = summary["analysis"]["selected_candidate"]
        for candidate in summary["candidates"]:
            candidate.update(gpu_correctness="passed", candidate_disposition="selected_for_confirmation" if candidate["id"] == selected else "search_only",
                measurement_quality="local_engineering_protocol", calibrated_ranker="unavailable")
            candidate["search_measurement"] = summary["analysis"]["search"][candidate["id"]]
            if candidate["id"] == selected:
                candidate["confirmation"] = summary["analysis"]["comparisons"]
        stage = "held_out_correctness"
        for rows, columns in rmsnorm.CORRECTNESS_SHAPES:
            for distribution in rmsnorm.DISTRIBUTIONS:
                name = f"held-out-{rows}x{columns}-{distribution}"
                directory = receipt / name
                directory.mkdir()
                current = {"case": name, "origin": "compiler_generated", "formula": selected, "gpu_correctness": "not_run"}
                summary["held_out_correctness"].append(current)
                case_inputs, case_oracles = rmsnorm.inputs_and_oracle(rows, columns, distribution)
                evaluate_case(compiler, binary, rmsnorm.document(rows, columns, selected, target=args.target), case_inputs, case_oracles,
                              directory, device_names, current)
        summary["status"] = "completed"
    except Exception as error:
        feedback = {"kind": "rejection", "stage": stage, "error": f"{type(error).__name__}: {error}",
                    "findings": current.get("findings", []) if current else [],
                    "artifact": current.get("id", current.get("case")) if current else None,
                    "origin": current.get("origin", "unknown") if current else "unknown"}
        if stage == "batch" and (receipt / "events.jsonl").exists():
            events = [json.loads(line) for line in (receipt / "events.jsonl").read_text().splitlines()]
            feedback["last_runtime_event"] = events[-1] if events else None
            for event in events:
                if event.get("stage") == "correctness" and event.get("status") == "passed":
                    for candidate in summary["candidates"]:
                        if candidate["id"] == event.get("artifact"):
                            candidate.update(gpu_execution="completed", gpu_correctness="passed")
            if events and events[-1].get("status") == "started":
                artifact = events[-1].get("artifact")
                known = next((a for a in artifacts if a["id"] == artifact), None)
                if known is not None:
                    feedback.update(artifact=artifact, origin=known["origin"])
                    matching = next((c for c in summary["candidates"] if c["id"] == artifact), None)
                    feedback["findings"] = matching.get("findings", []) if matching else []
                    if events[-1].get("stage") == "compile":
                        feedback["stage"] = "compile" if known["origin"] == "compiler_generated" else "reference_compile"
        if feedback["origin"] == "compiler_generated" and feedback["stage"] in {"assess", "compile", "held_out_correctness"}:
            feedback["route"] = asdict(route_rejection(feedback))
        else:
            feedback["measurement_quality"] = "invalid; no candidate or cost-model attribution established"
        summary["failure"] = feedback
    finally:
        (receipt / "receipt.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        (receipt / "feedback.json").write_text(json.dumps({"scope": PROTOCOL["claim_scope"],
            "candidates": summary["candidates"], "failure": summary.get("failure")}, indent=2, allow_nan=False) + "\n")
        print(receipt / "receipt.json")
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
