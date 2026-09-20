"""Fused Add + RMSNorm (DeepSeek-V3/R1, H=7168) — single Triton kernel.

Materialized operator code for the KerSor run-1 optimization round.

Semantics (fp32-internal, bf16-IO, EPS=1e-6, hidden_size fixed at 7168):
    x       = hidden_states.fp32 + residual.fp32
    inv_rms = rsqrt( mean(x*x, dim=-1) + EPS )
    y       = (x * inv_rms) * weight.fp32
    return y.to(bfloat16)

Design (per retrieved-context priors):
  - One CTA per row (program axis = batch). Row width H=7168 is contiguous.
  - Vectorized full-width tl.load over the row (128-bit bf16 transactions).
  - Squares accumulated in fp32 in registers; bf16 cannot hold the reduction.
  - tl.sum performs the warp/CTA shuffle reduce; rsqrt+eps in fp32; scale+store bf16.
  - Single fused launch (replaces ~10 eager elementwise kernels).
  - Triton requires BLOCK_SIZE to be a power of 2; H=7168 is covered by the
    next power of 2 (8192) with a tail mask. Autotune varies num_warps/num_stages
    over that fixed block: more warps give more lanes to hide HBM latency on the
    large prefill rows; fewer warps cut launch overhead on the tiny decode rows.

Contract: module exports run(hidden_states, residual, weight) -> output.
The harness owns input allocation; this kernel must NOT mutate inputs and must
NOT export anything other than `run`.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

HIDDEN_SIZE = 7168
EPS = 1e-6

# Next power of 2 >= 7168. The 1024-element tail is masked out.
BLOCK = 8192


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=3),
        triton.Config({}, num_warps=32, num_stages=2),
        triton.Config({}, num_warps=32, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=2),
    ],
    key=[],
)
@triton.jit
def _fused_add_rmsnorm_kernel(
    hidden_ptr,         # *bf16 [B, N]
    residual_ptr,       # *bf16 [B, N]
    weight_ptr,         # *bf16 [N]
    out_ptr,            # *bf16 [B, N]
    N,                  # row width (7168)
    inv_N,              # 1.0 / N (fp32 constant for the mean)
    eps,                # 1e-6
    stride_hb,          # stride from row to row for hidden/residual/out (elements)
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    row_off = row * stride_hb

    h = tl.load(hidden_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(residual_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    x = h + r
    # Masked zeros contribute nothing to the sum; inv_N uses the real N=7168.
    mean_sq = tl.sum(x * x, axis=0) * inv_N
    inv_rms = 1.0 / tl.sqrt(mean_sq + eps)

    y = (x * inv_rms) * w
    tl.store(out_ptr + row_off + cols, y.to(tl.bfloat16), mask=mask)


def run(hidden_states: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    B, N = hidden_states.shape
    assert N == HIDDEN_SIZE, f"expected hidden_size={HIDDEN_SIZE}, got {N}"

    out = torch.empty_like(hidden_states)
    inv_N = 1.0 / float(N)

    grid = (B,)
    _fused_add_rmsnorm_kernel[grid](
        hidden_states, residual, weight, out,
        N, inv_N, EPS,
        hidden_states.stride(0),
        BLOCK=BLOCK,
    )
    return out
