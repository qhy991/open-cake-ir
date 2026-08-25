"""Current correctness oracles for emitted-kernel observations.

`kernel_cases.ORACLES` predates the ranking calibration and its exact file bytes are now
part of frozen calibration authority. Rewriting those authorities would invalidate
historical replay. This module is the current extension point: it projects the retained
base registry and owns only oracles added after that freeze.
"""

from __future__ import annotations

from kernel_cases import ORACLES as _RETAINED_ORACLES


def _swiglu_oracle(inputs, torch):
    """The tanh identity used by the KDA v12 SwiGLU delta.

    Written from the operator definition rather than from the generated decomposition.
    It proves the standalone arithmetic slice, not grouped GEMMs or routed scatter.
    """

    up, gate, _ = inputs
    return up * gate * (0.5 * (torch.tanh(0.5 * gate) + 1.0)), None


def _top_k_oracle(inputs, torch):
    """Stable descending source indices for the standalone selection slice.

    The fixed prefix makes ties and negative infinity part of every observation rather
    than hoping random input happens to cover them. Stable sort is independent of the
    lowering's repeated reductions and defines the declared lowest-index tie break.
    """

    scores, _ = inputs
    scores.fill_(float("-inf"))
    prefix = torch.tensor(
        [10.0, 10.0, 9.0, 9.0, 8.0, 8.0, float("-inf"), float("-inf")],
        dtype=scores.dtype,
        device=scores.device,
    )
    scores[:, : prefix.numel()] = prefix
    return torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :8].to(
        torch.int32
    ), None


def _block_scaled_gemm_oracle(inputs, torch):
    """Dequantize both operands, then contract them in float32.

    This is deliberately not the lowering's partial-dot decomposition. The profile
    relation says A scales each row/128-wide K block and B has one 128-row group with
    the same K granularity; expanding that relation produces ordinary dense tensors
    whose matmul is an independent answer.
    """

    a, b, scale_a, scale_b, _ = inputs
    a_dequant = a.to(torch.float32) * scale_a.t().repeat_interleave(128, dim=1)
    b_dequant = b.to(torch.float32) * (
        scale_b.expand(b.shape[0], -1).repeat_interleave(128, dim=1)
    )
    return a_dequant @ b_dequant.t(), None


def _ragged_zero_pad_oracle(inputs, torch):
    """Dense materialization from the declared prefix lengths."""

    ragged, lengths, _ = inputs
    rows = torch.arange(ragged.shape[1], device=ragged.device)
    valid = rows[None, :] < lengths[:, None]
    return torch.where(valid[:, :, None], ragged, torch.zeros_like(ragged)), None


def _ragged_grouped_gemm_oracle(inputs, torch):
    """Mask routed rows, then evaluate one independent dense GEMM per group."""

    a, b, lengths, _ = inputs
    rows = torch.arange(a.shape[1], device=a.device)
    masked_a = torch.where(
        (rows[None, :] < lengths[:, None])[:, :, None],
        a,
        torch.zeros_like(a),
    )
    return torch.bmm(masked_a.float(), b.float().transpose(1, 2)), None


def _indexed_gather_oracle(inputs, torch):
    """KDA-style expert/position tuples select rows; invalid routes contribute zero.

    Advanced indexing states the mathematical answer independently of the lowering's
    flattened pointer arithmetic and broadcasting. Clamping is only for memory safety;
    the explicit validity predicate owns the result for KDA's -1 sentinel.
    """

    expert_rows, expert_ids, row_ids, _ = inputs
    valid = (
        (expert_ids >= 0)
        & (expert_ids < expert_rows.shape[0])
        & (row_ids >= 0)
        & (row_ids < expert_rows.shape[1])
    )
    selected = expert_rows[
        expert_ids.clamp(0, expert_rows.shape[0] - 1).long(),
        row_ids.clamp(0, expert_rows.shape[1] - 1).long(),
    ]
    return torch.where(valid[:, :, None], selected, torch.zeros_like(selected)), None


ORACLES = {
    **_RETAINED_ORACLES,
    "swiglu_b8_smoke": _swiglu_oracle,
    "top_k_b8_smoke": _top_k_oracle,
    "block_scaled_gemm_b1_smoke": _block_scaled_gemm_oracle,
    "ragged_zero_pad_b1_smoke": _ragged_zero_pad_oracle,
    "ragged_grouped_gemm_b1_smoke": _ragged_grouped_gemm_oracle,
    "indexed_gather_b8_smoke": _indexed_gather_oracle,
}

# Observation tools select an oracle by the executable interface the Compiler returns.
# The retained registry remains keyed by its historical calibration vocabulary because
# its bytes are frozen evidence; this is the sole current projection from lowering routes.
_WORKLOAD_BY_ENTRY_POINT = {
    "cake_flash_kmeans_assign": "flash_kmeans_b32_smoke",
    "cake_flash_kmeans_assignment_full": "flash_kmeans_assignment_full",
    "cake_softmax_b8_smoke": "softmax_b8_smoke",
    "cake_rmsnorm_b8_smoke": "rmsnorm_b8_smoke",
    "cake_layernorm_b8_smoke": "layernorm_b8_smoke",
    "cake_gemm_bias_b1_smoke": "gemm_bias_b1_smoke",
    "cake_block_scaled_gemm_b1_smoke": "block_scaled_gemm_b1_smoke",
    "cake_ragged_zero_pad_b1_smoke": "ragged_zero_pad_b1_smoke",
    "cake_ragged_grouped_gemm_b1_smoke": "ragged_grouped_gemm_b1_smoke",
    "cake_indexed_gather_b8_smoke": "indexed_gather_b8_smoke",
    "cake_swiglu_b8_smoke": "swiglu_b8_smoke",
    "cake_top_k_b8_smoke": "top_k_b8_smoke",
}
ORACLE_BY_ENTRY_POINT = {
    entry_point: ORACLES[workload]
    for entry_point, workload in _WORKLOAD_BY_ENTRY_POINT.items()
}
