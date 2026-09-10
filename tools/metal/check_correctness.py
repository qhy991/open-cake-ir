#!/usr/bin/env python3
"""Bounded exact-target Apple Metal correctness only; requires an approved released Compiler."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from open_cake_ir.compiler import Compiler, frontend
from tools.metal.adapter import compile_runner, invoke, manifest
from tools.metal import rmsnorm

SHAPES = ((1, 1), (2, 7), (3, 32), (4, 65), (2, 257))
DISTRIBUTIONS = ("zero", "uniform", "alternating", "mixed_magnitude")


from tools.metal.numerics import f32, values, compare


def schedule_document(operator: str, rows: int, columns: int, target: str = "apple_gpu_family8") -> dict:
    """Vary the annotated shapes while retaining the same Python authoring path."""
    example = "row_sum" if operator == "row_max" else operator
    path = ROOT / "examples/python" / f"metal_{example}.py"
    text = path.read_text().replace("apple_gpu_family8", target)
    if operator == "elementwise":
        text = text.replace("(3, 37)", f"({rows}, {columns})")
    elif operator in {"row_sum", "row_max"}:
        text = text.replace("(5, 65)", f"({rows}, {columns})").replace("(5,)", f"({rows},)")
        if operator == "row_max":
            text = text.replace('op="sum"', 'op="max"').replace("row_sum", "row_max").replace("row-sum", "row-max")
    elif operator == "rmsnorm":
        return rmsnorm.document(rows, columns, target=target)
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



def fresh_receipt(output_root: Path, *, prefix: str = "metal-correctness-") -> Path:
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
    return Path(tempfile.mkdtemp(prefix=prefix, dir=output_root))


def released_compiler(receipt: Path, target_id: str = "apple_gpu_family8") -> tuple[Compiler, dict, list[str]]:
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
    target_ref = lock["target_definitions"].get(target_id)
    if target_ref is None:
        raise ValueError(f"released Compiler does not bind {target_id}")
    target = json.loads((ROOT / target_ref["path"]).read_text())
    return compiler, lock, target["device_names"]


def runtime_source() -> dict:
    """Git owns the host/harness source identity; the Compiler lock owns lowering."""
    paths = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "tools/metal").glob("*")
                   if p.suffix in {".py", ".swift", ".metal"})
    paths += ["examples/python/metal_elementwise.py", "examples/python/metal_row_sum.py",
              "examples/python/metal_rmsnorm.py", "src/open_cake_ir/lab/routing.py"]
    head = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True, check=True)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "--", *paths], cwd=ROOT,
                             capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", *paths],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    return {"git_commit": head.stdout.strip(), "paths": paths,
            "tracked": tracked.returncode == 0, "clean": not status.stdout.strip()}


def prepare_case(compiler, document, inputs, oracles, directory, device_names, case):
    assessment = compiler.assess(document)
    case.update(static_accepted=assessment.accepted, lowering_eligible=assessment.lowering_eligible,
                findings=[{"code": f.code, "path": f.path, "message": f.message,
                           "blocks_acceptance": f.blocks_acceptance, "blocks_lowering": f.blocks_lowering}
                          for f in assessment.findings])
    if not assessment.accepted or not assessment.lowering_eligible:
        (directory / "assessment.json").write_text(json.dumps(case, indent=2) + "\n")
        raise ValueError("Compiler refused the candidate; see assessment findings")
    ranked, unranked = compiler.rank([assessment])
    case["pre_gpu_cost_ranking"] = {
        "ranked_count": len(ranked), "unranked_schedule_ids": list(unranked),
        "coverage": "unavailable" if unranked else "calibrated",
        "domain": "no Apple occupancy, physical registers, spill, bandwidth or latency estimate",
    }
    lowering = compiler.lower(assessment)
    launch = manifest(assessment, lowering, inputs, directory, device_names=device_names)
    (directory / "source-map.json").write_text(json.dumps(dict(lowering.source_map)) + "\n")
    (directory / "schedule.json").write_text(json.dumps(document, indent=2) + "\n")
    (directory / "oracle.json").write_text(json.dumps(oracles, allow_nan=False) + "\n")
    (directory / "assessment.json").write_text(json.dumps(case, indent=2) + "\n")
    return launch


def evaluate_case(compiler, binary, document, inputs, oracles, directory, device_names, case):
    launch = prepare_case(compiler, document, inputs, oracles, directory, device_names, case)
    case.update(gpu_execution="requested", gpu_correctness="unknown")
    runtime = invoke(binary, directory)
    case.update(device=runtime["device"], command_status=runtime["command_status"],
                gpu_execution="completed", gpu_correctness="failed")
    comparisons = {}
    for buffer in launch["buffers"]:
        payload = Path(buffer["output_path"]).read_bytes()
        if buffer["mode"] == "input":
            if payload != inputs[buffer["name"]]:
                raise ValueError(f"input buffer was mutated: {buffer['name']}")
        else:
            reference = oracles[buffer["name"]]
            comparisons[buffer["name"]] = compare(payload, reference["expected"], reference["absolute_tolerance"])
    case.update(comparisons=comparisons, gpu_correctness="passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target", choices=("apple_gpu_family7", "apple_gpu_family8", "apple_gpu_family9"),
                        default="apple_gpu_family8", help="exact target; no device fallback")
    args = parser.parse_args()
    receipt = fresh_receipt(args.output_root)
    summary = {"started_at": datetime.now(timezone.utc).isoformat(),
               "scope": "local Metal output correctness; no timing or framework acceptance",
               "status": "failed", "cases": []}
    try:
        summary["runtime_source"] = runtime_source()
        if not summary["runtime_source"]["tracked"] or not summary["runtime_source"]["clean"]:
            raise ValueError("correctness requires committed, clean runtime and example sources")
        compiler, lock, device_names = released_compiler(receipt, args.target)
        summary["compiler_revision_id"] = lock["revision_id"]
        summary["target"] = args.target
        binary = compile_runner(receipt)
        for operator in ("elementwise", "row_sum", "row_max", "rmsnorm"):
            shapes = rmsnorm.CORRECTNESS_SHAPES if operator == "rmsnorm" else SHAPES
            for rows, columns in shapes:
                for distribution in (rmsnorm.DISTRIBUTIONS if operator == "rmsnorm" else DISTRIBUTIONS):
                    name = f"{operator}-{rows}x{columns}-{distribution}"
                    directory = receipt / name
                    directory.mkdir()
                    document = schedule_document(operator, rows, columns, args.target)
                    if operator == "rmsnorm":
                        inputs, oracles = rmsnorm.inputs_and_oracle(rows, columns, distribution)
                    else:
                        x = values(rows * columns, distribution, 7001)
                        y = values(rows * columns, distribution, 9001)
                        inputs = {"x": struct.pack(f"<{len(x)}f", *x)}
                        if operator == "elementwise":
                            inputs["y"] = struct.pack(f"<{len(y)}f", *y)
                        expected, tolerance = oracle(operator, x, y, columns)
                        oracles = {"out": {"expected": expected, "absolute_tolerance": tolerance}}
                    case = {"case": name, "gpu_correctness": "not_run"}
                    summary["cases"].append(case)
                    evaluate_case(compiler, binary, document, inputs, oracles, directory, device_names, case)
        summary["status"] = "gpu_correctness_passed"
    except Exception as error:
        summary["error"] = f"{type(error).__name__}: {error}"
    finally:
        (receipt / "receipt.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print(receipt / "receipt.json")
    return 0 if summary["status"] == "gpu_correctness_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
