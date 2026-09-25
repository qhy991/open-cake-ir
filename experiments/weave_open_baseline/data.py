"""CPU-owned deterministic inputs and independent FP64 MoE oracle.

Only NumPy is needed. Inputs are recreated from the versioned contract after
the broker has released the GPUs; no device result is used to form the oracle.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


CONTRACT = Path(__file__).with_name("contract.json")


def load_contract(path: Path = CONTRACT) -> dict:
    document = json.loads(path.read_text())
    shape = document["geometry"]
    if (document["target"] != "sm_103a"
            or shape["expert_parallel_size"] != 4
            or shape["tokens_total"] != 4 * shape["tokens_per_rank"]
            or shape["experts"] % 4 != 0
            or shape["top_k"] > shape["experts"]
            or shape["dtype"] != "bfloat16"):
        raise ValueError("Unsupported EP4 BF16 contract geometry")
    if document["upstream"]["commit"] != "63de69e48dde17f32b0ee80ba83901c6950404cd":
        raise ValueError("Upstream source differs from reviewed commit")
    return document


def round_bf16(values: np.ndarray) -> np.ndarray:
    """Round finite float32 values to BF16 with round-to-nearest-even."""
    a = np.ascontiguousarray(values, dtype=np.float32)
    bits = a.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def make_rank(document: dict, rank: int) -> dict[str, np.ndarray]:
    shape = document["geometry"]
    if rank not in range(4):
        raise ValueError("Rank must be in EP4")
    seed = document["input"]["seed"]
    t, e, k = (shape[key] for key in ("tokens_per_rank", "experts", "top_k"))
    h, f = shape["hidden"], shape["intermediate"]
    input_rng = np.random.default_rng(seed + rank)
    weight_rng = np.random.default_rng(seed + 1000 + rank)
    hidden = round_bf16(input_rng.standard_normal((t, h), dtype=np.float32) * 0.5)
    scores = input_rng.random((t, e), dtype=np.float32)
    ids = np.argsort(-scores, axis=1, kind="stable")[:, :k].astype(np.int32)
    weights = input_rng.random((t, k), dtype=np.float32) + np.float32(0.1)
    weights /= weights.sum(axis=1, keepdims=True)
    local_e = e // 4
    gate = round_bf16(weight_rng.standard_normal((local_e, f, h), dtype=np.float32) * 0.1)
    up = round_bf16(weight_rng.standard_normal((local_e, f, h), dtype=np.float32) * 0.1)
    down = round_bf16(weight_rng.standard_normal((local_e, h, f), dtype=np.float32) * 0.1)
    return {"hidden": hidden, "ids": ids, "weights": weights,
            "gate": gate, "up": up, "down": down}


def reference(document: dict) -> np.ndarray:
    """Compute every routed contribution in FP64 on the CPU, then BF16 round."""
    shape = document["geometry"]
    t, h, e = (shape[key] for key in ("tokens_per_rank", "hidden", "experts"))
    ranks = [make_rank(document, rank) for rank in range(4)]
    output = np.zeros((4, t, h), dtype=np.float64)
    ids = np.stack([rank["ids"] for rank in ranks])
    for expert in range(e):
        source, token, slot = np.nonzero(ids == expert)
        if not source.size:
            continue
        owner, local = divmod(expert, e // 4)
        x = np.stack([ranks[int(s)]["hidden"][int(tok)]
                      for s, tok in zip(source, token, strict=True)]).astype(np.float64)
        gate = x @ ranks[owner]["gate"][local].astype(np.float64).T
        up = x @ ranks[owner]["up"][local].astype(np.float64).T
        activated = (gate / (1.0 + np.exp(-gate))) * up
        projected = activated @ ranks[owner]["down"][local].astype(np.float64).T
        route_weight = np.array([ranks[int(s)]["weights"][int(tok), int(sl)]
                                 for s, tok, sl in zip(source, token, slot, strict=True)])
        np.add.at(output, (source, token), projected * route_weight[:, None])
    return round_bf16(output.astype(np.float32))


def compare_outputs(document: dict, output_dir: Path) -> dict:
    expected = reference(document)
    actual = np.stack([np.load(output_dir / f"rank{rank}-output.npy")
                       for rank in range(4)])
    if actual.shape != expected.shape:
        raise ValueError(f"Output shape {actual.shape} != oracle {expected.shape}")
    if not np.all(np.isfinite(actual)):
        raise ValueError("Non-finite device output")
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    tolerance = document["oracle"]["atol"] + document["oracle"]["rtol"] * np.abs(expected)
    return {"pass": bool(np.all(difference <= tolerance)),
            "max_absolute_error": float(difference.max()),
            "failing_elements": int(np.count_nonzero(difference > tolerance)),
            "elements": int(difference.size),
            "numpy_version": np.__version__}
