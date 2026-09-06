"""The fixed local weighted-RMSNorm contract, candidate DAGs and external oracle."""
from __future__ import annotations

from dataclasses import replace
import math
import random
from pathlib import Path
import struct

from open_cake_ir.compiler import Schedule, frontend
from tools.metal.adapter import reference_manifest
from tools.metal.numerics import values, f32

ROOT = Path(__file__).resolve().parents[2]
PRIMARY_SHAPE = (128, 1024)
CORRECTNESS_SHAPES = ((1, 1), (3, 7), (2, 32), (5, 65), (3, 257), (128, 1024), (2, 4096))
DISTRIBUTIONS = ("zero", "uniform", "alternating", "mixed_magnitude", "epsilon_dominated")
FORMULAS = ("canonical", "weight_first", "prescaled_square")
CONTRACT = {
    "operator": "weighted_fp32_rmsnorm", "primary_shape": list(PRIMARY_SHAPE),
    "equation": "out[r,c] = x[r,c] * rsqrt(sum_j(x[r,j]^2)/C + 1e-5) * weight[c]",
    "epsilon": 1e-5, "atol": 2e-5, "rtol": 2e-5,
    "dtype": "fp32", "storage": "contiguous", "input_domain": "the bounded normal/zero distributions declared below",
    "input_seed": 7001, "weight_seed": 17001,
    "input_distributions": {"zero": "signed zeros", "uniform": "uniform[-2,2]",
        "alternating": "alternating sign, magnitude 1+(i mod 17)/16",
        "mixed_magnitude": "signed powers of two, exponent[-12,8]",
        "epsilon_dominated": "uniform[-1e-4,1e-4]"},
    "weight_distribution": "FP32 uniform[-1.5,1.5], every17th element zero",
    "authoring_environment": "known-kernel reproduction/optimization",
    "claim_scope": "local artifact output and matched warm command-buffer measurements, not a qualified Study",
    "correctness_shapes": [list(shape) for shape in CORRECTNESS_SHAPES],
}


def source(rows: int, columns: int, formula: str = "canonical") -> str:
    if formula not in FORMULAS or type(rows) is not int or type(columns) is not int or rows <= 0 or columns <= 0:
        raise ValueError("unsupported RMSNorm formula or shape")
    text = (ROOT / "examples/python/metal_rmsnorm.py").read_text()
    text = text.replace("(128, 1024)", f"({rows}, {columns})").replace("(1024,)", f"({columns},)")
    text = text.replace("total / 1024.0", f"total / {float(columns)!r}")
    if formula == "weight_first":
        text = text.replace("normalized = values * inverse\n        result = normalized * weights",
                            "normalized = values * weights\n        result = normalized * inverse")
    elif formula == "prescaled_square":
        text = text.replace('squared = lm.square(values, id="square")',
                            f'scaled = values * {1 / math.sqrt(columns)!r}\n        squared = lm.square(scaled, id="square")')
        text = text.replace(f"total / {float(columns)!r}", "total * 1.0")
    return text.replace('name="metal-rmsnorm"', f'name="metal-rmsnorm-{formula}"')


def document(rows: int, columns: int, formula: str = "canonical") -> dict:
    return frontend.parse(source(rows, columns, formula), filename=f"metal_rmsnorm:{formula}").document


def inputs_and_oracle(rows: int, columns: int, distribution: str) -> tuple[dict, dict]:
    if distribution == "epsilon_dominated":
        rng = random.Random(CONTRACT["input_seed"])
        x = [f32(rng.uniform(-1e-4, 1e-4)) for _ in range(rows * columns)]
    else:
        x = values(rows * columns, distribution, CONTRACT["input_seed"])
    rng = random.Random(CONTRACT["weight_seed"])
    weight = [0.0 if (i + 1) % 17 == 0 else f32(rng.uniform(-1.5, 1.5)) for i in range(columns)]
    expected = []
    for row in range(rows):
        values_row = x[row * columns:(row + 1) * columns]
        inverse = 1.0 / math.sqrt(math.fsum(value * value for value in values_row) / columns + CONTRACT["epsilon"])
        expected.extend(value * inverse * scale for value, scale in zip(values_row, weight))
    tolerances = [CONTRACT["atol"] + CONTRACT["rtol"] * abs(v) for v in expected]
    return ({"x": struct.pack(f"<{len(x)}f", *x), "weight": struct.pack(f"<{len(weight)}f", *weight)},
            {"out": {"expected": expected, "absolute_tolerance": tolerances}})


def reference(rows: int, columns: int, execution: str, inputs: dict,
              directory: Path, device_names: list[str]) -> dict:
    if execution not in {"serial", "simd"}:
        raise ValueError("unknown reference execution")
    schedule = Schedule.from_dict(document(rows, columns))
    route = replace(schedule.lowering, entry_point=f"reference_{execution}")
    schedule = replace(schedule, lowering=route)
    path = ROOT / "tools/metal/rmsnorm_reference.metal"
    toolchain = {"target": schedule.target, "source_language": "metal", "compiler": "MTLDevice.makeLibrary",
        "language_standard": "metal2.3", "fast_math_enabled": False, "threadgroup_memory_bytes": 0,
        "execution_model": f"{execution}_program_tile", "active_threads_per_threadgroup": 1 if execution == "serial" else 32,
        "buffer_order": ["x", "weight", "out"], "threadgroups_per_grid": [rows, 1, 1], "threads_per_threadgroup": [32, 1, 1]}
    return reference_manifest(schedule, path.read_text().replace("__COLUMNS__", str(columns)), toolchain,
        inputs, directory, device_names=device_names, provenance={"kind": "handwritten_reference",
            "source_path": "tools/metal/rmsnorm_reference.metal", "entry_point": route.entry_point,
            "basis": "mathematical weighted RMSNorm; authored here under known-kernel reproduction",
            "reduction_order": "increasing columns" if execution == "serial" else "striped partial sums followed by simd_sum"})
