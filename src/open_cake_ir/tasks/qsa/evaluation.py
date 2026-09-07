"""External QSA workload materialization, oracle, and correctness audit.

The Workload Contract owns semantics and the declared geometry.  This module
only executes that contract; neither the Compiler nor a candidate imports it.
The operator begins after Q/K/V and index Q/K projection, so projection,
query-side normalization, and RoPE are deliberate non-goals of this task.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, cast

from .workload import WorkloadContract


@dataclass(frozen=True)
class QsaCorrectnessObservation:
    """Canonical tolerance result for one complete QSA output."""

    matched: int
    total: int
    match_fraction: float
    max_abs_diff: float
    passed: bool

    def as_dict(self) -> dict[str, int | float | bool]:
        return {
            "matched": self.matched,
            "total": self.total,
            "match_fraction": self.match_fraction,
            "max_abs_diff": self.max_abs_diff,
            "passed": self.passed,
        }


def _qsa_document(workload: WorkloadContract) -> Mapping[str, object]:
    document = workload.document
    if document.get("operator") != "qsa_prefill":
        raise ValueError("QSA evaluation requires the qsa_prefill workload")
    return document


def _geometry(workload: WorkloadContract) -> Mapping[str, int]:
    semantics = cast(Mapping[str, object], _qsa_document(workload)["semantics"])
    geometry = cast(Mapping[str, object], semantics["task_declared_geometry"])
    if not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in geometry.values()
    ):
        raise ValueError("QSA task geometry must contain integers")
    return cast(Mapping[str, int], geometry)


def materialize_qsa_case(workload: WorkloadContract, case_id: str, device):
    """Create deterministic BF16/FP32 inputs from one frozen case seed."""

    import torch

    case = workload.case(case_id)
    seed = case["seed"]
    if not isinstance(seed, int):
        raise ValueError("QSA cases require an integer seed")
    shape = cast(Mapping[str, int], case["shape"])
    if shape.get("B") != 1:
        raise ValueError("the first QSA workload supports batch one")
    tokens = int(shape["T"])
    geometry = _geometry(workload)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    def randn(tensor_shape, dtype=torch.bfloat16):
        return torch.randn(
            tensor_shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )

    index_dim = geometry["index_head_dim"]
    return {
        "q": randn((1, tokens, geometry["num_heads"], geometry["head_dim"])),
        "k": randn((1, tokens, geometry["num_kv_heads"], geometry["head_dim"])),
        "v": randn((1, tokens, geometry["num_kv_heads"], geometry["head_dim"])),
        "index_q": randn((1, tokens, geometry["index_n_heads"], index_dim)),
        "index_k": randn((1, tokens, geometry["index_kv_heads"], index_dim)),
        "k_norm_weight": torch.ones(index_dim, device=device, dtype=torch.float32)
        + 0.02 * randn((index_dim,), torch.float32),
        "scale": 1.0 / math.sqrt(geometry["head_dim"]),
    }


def _layernorm(x, weight, epsilon: float):
    import torch

    x = x.float()
    centered = x - x.mean(-1, keepdim=True)
    return (
        centered
        * torch.rsqrt(centered.square().mean(-1, keepdim=True) + epsilon)
        * weight
    )


def qsa_block_scores(workload: WorkloadContract, inputs):
    """Return FP32 scores with ReLU before the index-head reduction."""

    import torch

    geometry = _geometry(workload)
    ratio = geometry["compress_ratio"]
    dimension = geometry["index_head_dim"]
    tokens = int(inputs["index_k"].shape[1])
    blocks = tokens // ratio
    raw = inputs["index_k"][0, :, 0, :]
    pooled = raw.view(blocks, ratio, dimension).float().mean(dim=1)
    semantics = cast(Mapping[str, object], _qsa_document(workload)["semantics"])
    pooled = _layernorm(
        pooled,
        inputs["k_norm_weight"],
        float(semantics["layernorm_epsilon"]),
    )
    agreement = torch.relu(
        torch.einsum("thd,bd->tbh", inputs["index_q"][0].float(), pooled)
    )
    return agreement.sum(-1) / math.sqrt(dimension)


def qsa_selection_mask(workload: WorkloadContract, inputs):
    """Materialize the task's causal complete-block token mask."""

    import torch

    geometry = _geometry(workload)
    ratio = geometry["compress_ratio"]
    blocks_to_keep = geometry["token_budget"] // ratio
    tokens = int(inputs["q"].shape[1])
    device = inputs["q"].device
    blocks = tokens // ratio
    scores = qsa_block_scores(workload, inputs)
    positions = torch.arange(tokens, device=device)
    block_end = (torch.arange(blocks, device=device) + 1) * ratio - 1
    candidate = block_end[None, :] <= positions[:, None]
    masked = torch.where(candidate, scores, torch.finfo(scores.dtype).min)
    selected = masked.topk(min(blocks_to_keep, blocks), dim=-1).indices
    block_mask = torch.zeros((tokens, blocks), dtype=torch.bool, device=device)
    block_mask.scatter_(1, selected, True)
    block_mask &= candidate
    token_mask = block_mask[:, positions // ratio]
    return token_mask & (positions[None, :] <= positions[:, None])


def reference_qsa_output(workload: WorkloadContract, inputs):
    """Independent dense FP32 GQA oracle for the selected-token semantics."""

    import torch

    geometry = _geometry(workload)
    query = inputs["q"][0].float()
    key = inputs["k"][0].float()
    value = inputs["v"][0].float()
    tokens = int(query.shape[0])
    heads = geometry["num_heads"]
    group = heads // geometry["num_kv_heads"]
    mask = qsa_selection_mask(workload, inputs)
    output = torch.empty(
        (tokens, heads, geometry["head_dim"]),
        device=query.device,
        dtype=torch.float32,
    )
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for head in range(heads):
            kv_head = head // group
            logits = (
                query[:, head, :] @ key[:, kv_head, :].transpose(0, 1)
            ) * float(inputs["scale"])
            logits = torch.where(mask, logits, torch.finfo(torch.float32).min)
            output[:, head, :] = torch.softmax(logits, dim=-1) @ value[:, kv_head, :]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return output.unsqueeze(0)


def audit_qsa_output(
    workload: WorkloadContract,
    actual,
    expected,
) -> QsaCorrectnessObservation:
    """Apply the Workload-owned elementwise tolerance and match-fraction gate."""

    import torch

    validation = cast(Mapping[str, object], _qsa_document(workload)["validation"])
    actual = actual.float()
    expected = expected.float()
    close = torch.isclose(
        actual,
        expected,
        atol=float(validation["atol"]),
        rtol=float(validation["rtol"]),
    )
    matched = int(close.sum().item())
    total = int(close.numel())
    fraction = matched / total if total else 0.0
    difference = (actual - expected).abs()
    return QsaCorrectnessObservation(
        matched=matched,
        total=total,
        match_fraction=fraction,
        max_abs_diff=float(difference.max().item()),
        passed=fraction >= float(validation["minimum_match_fraction"]),
    )
