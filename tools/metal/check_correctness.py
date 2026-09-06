#!/usr/bin/env python3
"""Bounded Apple M2 correctness only; requires an approved released Compiler."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from open_cake_ir.compiler import Compiler, frontend
from tools.metal.adapter import compile_runner, invoke, manifest

SHAPES = ((1, 1), (2, 7), (3, 32), (4, 65), (2, 257))
DISTRIBUTIONS = ("zero", "uniform", "alternating", "mixed_magnitude")


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def values(count: int, distribution: str, seed: int) -> list[float]:
    rng = random.Random(seed)
    if distribution == "zero":
        return [0.0 if i % 2 else -0.0 for i in range(count)]
    if distribution == "uniform":
        return [f32(rng.uniform(-2.0, 2.0)) for _ in range(count)]
    if distribution == "alternating":
        return [f32((-1 if i % 2 else 1) * (1 + (i % 17) / 16)) for i in range(count)]
    if distribution == "mixed_magnitude":
        return [f32(rng.choice((-1, 1)) * 2.0 ** rng.randint(-12, 8)) for _ in range(count)]
    raise ValueError(f"unknown input distribution: {distribution}")


def schedule_document(operator: str, rows: int, columns: int) -> dict:
    """Vary the annotated shapes while retaining the same Python authoring path."""
    example = "row_sum" if operator == "row_max" else operator
    path = ROOT / "examples/python" / f"metal_{example}.py"
    text = path.read_text()
    if operator == "elementwise":
        text = text.replace("(3, 37)", f"({rows}, {columns})")
    elif operator in {"row_sum", "row_max"}:
        text = text.replace("(5, 65)", f"({rows}, {columns})").replace("(5,)", f"({rows},)")
        if operator == "row_max":
            text = text.replace('op="sum"', 'op="max"').replace("row_sum", "row_max").replace("row-sum", "row-max")
    else:
        raise ValueError("unknown oracle operator")
    return frontend.parse(text, filename=str(path)).document


def oracle(operator: str, x: list[float], y: list[float], columns: int) -> tuple[list[float], list[float]]:
    if operator == "elementwise":
        expected = [f32(f32(a + b) * 2.0) for a, b in zip(x, y)]
        return expected, [1e-6 + 2e-6 * abs(v) for v in expected]
    # Independent high-precision summation, with the standard FP32 accumulation bound.
    # MSL permits round-to-zero as well as nearest; use the larger error bound.
    unit_roundoff = 2.0 ** -23
    gamma = columns * unit_roundoff / (1 - columns * unit_roundoff)
    rows = [x[start:start + columns] for start in range(0, len(x), columns)]
    if operator == "row_max":
        return [max(row) for row in rows], [0.0 for _ in rows]
    return ([math.fsum(row) for row in rows],
            [1e-6 + gamma * math.fsum(abs(v) for v in row) for row in rows])


def compare(payload: bytes, expected: list[float], tolerance: list[float]) -> dict:
    if len(payload) != len(expected) * 4:
        raise ValueError("output byte length differs from the external oracle")
    actual = [v[0] for v in struct.iter_unpack("<f", payload)]
    differences = [abs(a - b) for a, b in zip(actual, expected)]
    bad = [i for i, (a, error, bound) in enumerate(zip(actual, differences, tolerance))
           if not math.isfinite(a) or error > bound]
    if bad:
        i = bad[0]
        raise ValueError(f"output[{i}]={actual[i]} expected={expected[i]} tolerance={tolerance[i]}")
    return {"elements": len(expected), "max_abs_error": max(differences, default=0.0),
            "max_allowed_error": max(tolerance, default=0.0)}


def fresh_receipt(output_root: Path) -> Path:
    if not output_root.is_absolute():
        raise ValueError("--output-root must be absolute")
    output_root = output_root.resolve()
    result = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=ROOT,
                            capture_output=True, text=True)
    worktrees = [ROOT]
    if result.returncode == 0:
        worktrees += [Path(line.removeprefix("worktree ")).resolve()
                      for line in result.stdout.splitlines() if line.startswith("worktree ")]
    if any(output_root == path or path in output_root.parents for path in worktrees):
        raise ValueError("--output-root must be outside project worktrees")
    output_root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="metal-correctness-", dir=output_root))


def released_compiler(receipt: Path) -> tuple[Compiler, dict, list[str]]:
    lock_path = ROOT / "compiler/revision.lock.json"
    lock = json.loads(lock_path.read_text())
    if lock.get("state") != "released":
        raise ValueError("GPU correctness requires a reviewed released Compiler")
    # Canonical release verification owns reviewer independence and the bound closure.
    command = [sys.executable, str(ROOT / "tools/release_compiler.py"), "--project-root", str(ROOT),
               "--proposal", str(ROOT / "compiler/revision.json"),
               "--source-set", str(ROOT / "compiler/source_set.json"),
               "--gate-report", str(ROOT / "compiler/corpus-gate-report.json"),
               "--approval", str(ROOT / "compiler/release-approval.json"),
               "--output", str(lock_path), "--verify"]
    check = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=300)
    (receipt / "release-verification.log").write_text(check.stdout + check.stderr)
    if check.returncode:
        raise ValueError("reviewed release verification failed; see release-verification.log")
    compiler = Compiler.load(ROOT, lock_path)
    target_ref = lock["target_definitions"].get("apple_gpu_family8")
    if target_ref is None:
        raise ValueError("released Compiler does not bind apple_gpu_family8")
    target = json.loads((ROOT / target_ref["path"]).read_text())
    return compiler, lock, target["device_names"]


def runtime_source() -> dict:
    """Git owns the host/harness source identity; the Compiler lock owns lowering."""
    paths = ["tools/metal/adapter.py", "tools/metal/runner.swift", "tools/metal/check_correctness.py",
             "examples/python/metal_elementwise.py", "examples/python/metal_row_sum.py"]
    head = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True, check=True)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "--", *paths], cwd=ROOT,
                             capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", *paths],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    return {"git_commit": head.stdout.strip(), "paths": paths,
            "tracked": tracked.returncode == 0, "clean": not status.stdout.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    receipt = fresh_receipt(args.output_root)
    summary = {"started_at": datetime.now(timezone.utc).isoformat(),
               "scope": "local Metal output correctness; no timing or framework acceptance",
               "status": "failed", "cases": []}
    try:
        summary["runtime_source"] = runtime_source()
        if not summary["runtime_source"]["tracked"] or not summary["runtime_source"]["clean"]:
            raise ValueError("correctness requires committed, clean runtime and example sources")
        compiler, lock, device_names = released_compiler(receipt)
        summary["compiler_revision_id"] = lock["revision_id"]
        binary = compile_runner(receipt)
        for operator in ("elementwise", "row_sum", "row_max"):
            for rows, columns in SHAPES:
                for distribution in DISTRIBUTIONS:
                    name = f"{operator}-{rows}x{columns}-{distribution}"
                    directory = receipt / name
                    directory.mkdir()
                    document = schedule_document(operator, rows, columns)
                    assessment = compiler.assess(document)
                    case = {"case": name, "static_accepted": assessment.accepted,
                            "lowering_eligible": assessment.lowering_eligible,
                            "findings": [{"code": f.code, "path": f.path, "message": f.message}
                                         for f in assessment.findings],
                            "gpu_correctness": "not_run"}
                    summary["cases"].append(case)
                    if not assessment.accepted or not assessment.lowering_eligible:
                        raise ValueError(f"Compiler refused {name}; see receipt findings")
                    lowering = compiler.lower(assessment)
                    x = values(rows * columns, distribution, 7001)
                    y = values(rows * columns, distribution, 9001)
                    inputs = {"x": struct.pack(f"<{len(x)}f", *x)}
                    if operator == "elementwise":
                        inputs["y"] = struct.pack(f"<{len(y)}f", *y)
                    launch = manifest(assessment, lowering, inputs, directory, device_names=device_names)
                    (directory / "schedule.json").write_text(json.dumps(document, indent=2) + "\n")
                    case["gpu_execution"] = "requested"
                    case["gpu_correctness"] = "unknown"
                    runtime = invoke(binary, directory)
                    case["device"] = runtime["device"]
                    case["command_status"] = runtime["command_status"]
                    case["gpu_execution"] = "completed"
                    case["gpu_correctness"] = "failed"
                    for buffer in launch["buffers"]:
                        if buffer["mode"] == "input" and Path(buffer["output_path"]).read_bytes() != inputs[buffer["name"]]:
                            raise ValueError(f"input buffer was mutated: {buffer['name']}")
                    output = next(b for b in launch["buffers"] if b["name"] == "out")
                    reference, tolerance = oracle(operator, x, y, columns)
                    (directory / "oracle.json").write_text(json.dumps({"expected": reference,
                        "absolute_tolerance": tolerance}, allow_nan=False) + "\n")
                    case["comparison"] = compare(Path(output["output_path"]).read_bytes(), reference, tolerance)
                    case["gpu_correctness"] = "passed"
        summary["status"] = "gpu_correctness_passed"
    except Exception as error:
        summary["error"] = f"{type(error).__name__}: {error}"
    finally:
        (receipt / "receipt.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print(receipt / "receipt.json")
    return 0 if summary["status"] == "gpu_correctness_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
