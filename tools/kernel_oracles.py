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


ORACLES = {
    **_RETAINED_ORACLES,
    "swiglu_b8_smoke": _swiglu_oracle,
    "top_k_b8_smoke": _top_k_oracle,
    "block_scaled_gemm_b1_smoke": _block_scaled_gemm_oracle,
}
