"""Deterministic finite FP32 inputs and external numeric comparisons."""
from __future__ import annotations
import math
import random
import struct

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

