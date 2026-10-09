"""Emit/check a bounded gfx938 capability probe outside the Compiler checkout.

This is component qualification, not a Lab Campaign or whole-model result.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def write(path: Path, document) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(document, stream, indent=2)
        stream.write("\n")


def emit(destination: Path) -> None:
    from open_cake_ir.compiler import Compiler
    engine = Compiler.load(ROOT, ROOT / "compiler/revision.json")
    if engine.commit is None:
        raise ValueError("qualification requires a clean Compiler commit")
    destination.mkdir(parents=True, exist_ok=False)
    entries = []

    def lower(document, name, kind, **description):
        assessment = engine.assess(document)
        if not assessment.lowering_eligible:
            raise ValueError([f.to_dict() for f in assessment.findings])
        result = engine.lower(assessment)
        folder = destination / name
        folder.mkdir()
        (folder / "source.py").write_text(result.source)
        write(folder / "schedule.json", document)
        entries.append({"name": name, "kind": kind, "entry_point": result.route.entry_point,
                        "source": str(folder / "source.py"),
                        "toolchain": dict(result.toolchain_requirements), **description})

    base = json.loads((ROOT / "corpus/schedules/gfx938-cast-bf16-fp32-b8-smoke.json").read_text())
    for source, target in (("bf16", "fp32"), ("fp32", "bf16"), ("int32", "fp32")):
        d = copy.deepcopy(base)
        d["schedule_id"] = "qualify-cast-" + source + "-" + target
        d["lowering"]["entry_point"] = "qualify_cast_" + source + "_" + target
        for buffer in d["buffers"]:
            buffer["dtype"] = source if buffer["name"] in ("x", "x_tile") else target
        next(o for o in d["operations"] if o["kind"] == "cast")["parameters"]["to"] = target
        lower(d, d["schedule_id"], "cast", source_dtype=source, target_dtype=target)

    for m, n, k in ((64, 64, 128), (128, 128, 128), (512, 256, 256), (128, 96, 192)):
        for dtype in ("bf16", "fp32"):
            d = json.loads((ROOT / "corpus/schedules/gfx938-gemm-bias-bf16-b1-smoke.json").read_text())
            d["schedule_id"] = f"qualify-gemm-{dtype}-{m}-{n}-{k}"
            d["lowering"]["entry_point"] = f"qualify_gemm_{dtype}_{m}_{n}_{k}"
            for buffer in d["buffers"]:
                if buffer["name"] in ("a", "b", "a_tile", "b_tile"):
                    buffer["dtype"] = dtype
                if buffer["name"] == "a": buffer["shape"] = [m, k]
                if buffer["name"] == "b": buffer["shape"] = [n, k]
                if buffer["name"] == "bias": buffer["shape"] = [n]
                if buffer["name"] == "c": buffer["shape"] = [m, n]
            next(o for o in d["operations"] if o["kind"] == "mma")["parameters"]["instruction"] = {
                "contract": "triton.dot.bf16_fp32" if dtype == "bf16" else "triton.dot.fp32_ieee"}
            # Same legal geometry and pipeline for both arms; FP32 stage2 can exceed64KiB.
            d["tile_loops"][0]["range_options"]["num_stages"] = 1
            lower(d, d["schedule_id"], "gemm", shape=[m, n, k], operand_dtype=dtype)
    write(destination / "manifest.json", {"compiler_commit": engine.commit, "target": "gfx938",
                                          "entries": entries, "scope": "bounded component qualification"})


def check(destination: Path) -> None:
    import torch
    torch.manual_seed(200)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if arch != "gfx938": raise ValueError("exact gfx938 required")
    manifest = json.loads((destination / "manifest.json").read_text())
    modules = {}
    for entry in manifest["entries"]:
        spec = importlib.util.spec_from_file_location(entry["name"].replace("-", "_"), entry["source"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[entry["name"]] = (module, getattr(module, entry["entry_point"]))
    dtypes = {"bf16": torch.bfloat16, "fp32": torch.float32, "int32": torch.int32}
    checks, timings = [], []
    for entry in manifest["entries"]:
        module, run = modules[entry["name"]]
        if entry["kind"] == "cast":
            x = torch.arange(-512, 512, device="cuda").reshape(8, 128).to(dtypes[entry["source_dtype"]])
            if entry["source_dtype"] != "int32": x = x / 37.0
            original = x.clone()
            actual = run(x)
            expected = x.to(dtypes[entry["target_dtype"]])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if not torch.equal(x, original): raise ValueError("cast mutated input")
            checks.append({"name": entry["name"], "passed": True, "values": x.numel(), "exact": True})
            continue
        if entry["operand_dtype"] != "bf16": continue
        m, n, k = entry["shape"]
        fp32_name = entry["name"].replace("bf16", "fp32")
        old = modules[fp32_name][1]
        def widened(x, y, z): return old(x.float(), y.float(), z)
        for scale in (1.0, 2.0**40):
            a = (torch.randn((m, k), device="cuda") * scale).to(torch.bfloat16)
            b = (torch.randn((n, k), device="cuda") / scale).to(torch.bfloat16)
            bias = torch.randn((n,), device="cuda", dtype=torch.float32)
            snapshots = [t.clone() for t in (a, b, bias)]
            expected = a.float() @ b.float().T + bias
            actual = run(a, b, bias)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-4)
            torch.testing.assert_close(widened(a, b, bias), expected, rtol=2e-5, atol=2e-4)
            if not all(torch.equal(t, s) for t, s in zip((a, b, bias), snapshots)):
                raise ValueError("GEMM mutated input")
            checks.append({"name": entry["name"], "scale": scale, "passed": True,
                           "values": actual.numel(), "max_abs": (actual-expected).abs().max().item()})
        # Capture the actual generated native assembly for dtype/ISA inspection.
        kernel = getattr(module, entry["toolchain"]["kernel_entry_point"])
        compiled = kernel.warmup(a, b, bias, torch.empty((m,n), device="cuda"),
                                grid=(math.ceil(m/64), math.ceil(n/64), 1),
                                **entry["toolchain"]["compile_constants"],
                                **entry["toolchain"]["compile_options"])
        assembly = compiled.asm["amdgcn"]
        (destination / (entry["name"] + ".amdgcn")).write_text(assembly)
        native = sorted(set(line.strip() for line in assembly.splitlines() if "v_mmac" in line or "v_mfma" in line))
        if not any("bf16" in line for line in native):
            raise ValueError("BF16 matrix instruction missing in actual native assembly")
        for _ in range(10): run(a,b,bias); widened(a,b,bias)
        samples = {key: [] for key in ("nf", "of", "nr", "orr", "aa1", "aa2")}
        def timed(fn):
            torch.cuda.synchronize(); start=time.perf_counter(); fn(a,b,bias); torch.cuda.synchronize()
            return (time.perf_counter()-start)*1e6
        for _ in range(30): samples["of"].append(timed(widened)); samples["nf"].append(timed(run))
        for _ in range(30): samples["nr"].append(timed(run)); samples["orr"].append(timed(widened))
        for _ in range(30): samples["aa1"].append(timed(widened)); samples["aa2"].append(timed(widened))
        med = {key: statistics.median(value) for key,value in samples.items()}
        timings.append({"shape": [m,n,k], "bf16_direct_us": statistics.median(samples["nf"]+samples["nr"]),
                        "fp32_widening_us": statistics.median(samples["of"]+samples["orr"]),
                        "forward_speedup": med["of"]/med["nf"], "reverse_speedup": med["orr"]/med["nr"],
                        "aa_abs_drift_us": abs(med["aa1"]-med["aa2"]), "native_matrix_instructions": native,
                        "samples_us": samples})
    write(destination / "device-result.json", {"compiler_commit": manifest["compiler_commit"], "arch": arch,
          "status": "passed", "checks": checks, "paired_component_timings": timings,
          "measurement": {"timer": "host perf_counter with device synchronization before and after each callable",
                          "boundary": "allocations, explicit widening and kernel dispatch included",
                          "cache_policy": "warm cache; no explicit cache flush", "warmup_pairs": 10,
                          "samples_per_arm_per_order": 30, "external_gpu_activity": "not_excluded"},
          "performance_scope": "BF16 generated GEMM+bias versus same-geometry FP32 widening route; not community-SOTA or serving speedup"})
    print(json.dumps({"status":"passed", "checks":len(checks), "component_pairs":len(timings)}))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("emit", "check"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args=parser.parse_args()
    destination=args.output_dir.resolve()
    if destination == ROOT or ROOT in destination.parents:
        parser.error("generated evidence must be outside the Compiler checkout")
    emit(destination) if args.mode == "emit" else check(destination)
