"""bf16 RMSNorm Triton kernel for hidden_size=2048, NVIDIA B200.

Design:
* Single Triton kernel processes a 2D tile [ROWS_PER_PROGRAM, HIDDEN] per CTA.
* Reduction (sum of squares) is along the hidden axis (axis=1) in fp32.
* Autotuned over (ROWS_PER_PROGRAM, num_warps, num_stages) keyed on batch_size:
  - Large-B (B>=1000): ROWS_PER_PROGRAM=1, num_warps=8 - HBM-bound, one row/CTA.
  - Small-B (B<=79): pack many rows per CTA to amortize launch overhead.
* All math (sum, mean, rsqrt, scale) is fp32 with an fp32 accumulator.
* Single bf16 cast happens exactly once on tl.store (no intermediate bf16).
* eps=1e-6 is INSIDE the rsqrt argument as required.
* INV_H = 1/2048 = 0.00048828125 is exact in fp32 (no divide at runtime).

The numerical contract follows the spec exactly:
    inv_rms = rsqrt(sum(x_fp32^2) / 2048 + 1e-6)
    y = ((x_fp32 * inv_rms) * w_fp32).to(bf16)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


HIDDEN_SIZE: int = 2048
EPS: float = 1e-6
INV_H: float = 1.0 / 2048.0  # 0.00048828125 - exact in fp32.


# ---------------------------------------------------------------------------
# Triton kernel: 2D row-packed tile.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        # Large-B configs: 1 row per CTA, maximize bandwidth.
        triton.Config({"ROWS_PER_PROGRAM": 1}, num_warps=8, num_stages=1),
        triton.Config({"ROWS_PER_PROGRAM": 1}, num_warps=4, num_stages=1),
        triton.Config({"ROWS_PER_PROGRAM": 1}, num_warps=8, num_stages=2),
        triton.Config({"ROWS_PER_PROGRAM": 1}, num_warps=16, num_stages=1),
        # Medium-B configs: amortize launch.
        triton.Config({"ROWS_PER_PROGRAM": 2}, num_warps=8, num_stages=1),
        triton.Config({"ROWS_PER_PROGRAM": 4}, num_warps=8, num_stages=1),
        triton.Config({"ROWS_PER_PROGRAM": 4}, num_warps=4, num_stages=1),
        # Small-B configs: pack many rows per CTA so few CTAs cover B.
        triton.Config({"ROWS_PER_PROGRAM": 8}, num_warps=8, num_stages=1),
        triton.Config({"ROWS_PER_PROGRAM": 8}, num_warps=4, num_stages=1),
        triton.Config({"ROWS_PER_PROGRAM": 16}, num_warps=8, num_stages=1),
        triton.Config({"ROWS_PER_PROGRAM": 16}, num_warps=4, num_stages=1),
    ],
    key=["batch_size"],
)
@triton.jit
def _rmsnorm_kernel(
    x_ptr,           # *bf16 [batch_size, HIDDEN]
    w_ptr,           # *bf16 [HIDDEN]
    y_ptr,           # *bf16 [batch_size, HIDDEN]
    batch_size,
    BLOCK_H: tl.constexpr,           # = HIDDEN_SIZE = 2048
    ROWS_PER_PROGRAM: tl.constexpr,  # autotuned
    INV_H_C: tl.constexpr,           # = 1/2048
    EPS_C: tl.constexpr,             # = 1e-6
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROGRAM

    # Column offsets - constant across all rows in this tile.
    col = tl.arange(0, BLOCK_H)  # [BLOCK_H]

    # Load weight once (bf16 -> fp32). Broadcast across rows.
    w = tl.load(w_ptr + col).to(tl.float32)  # [BLOCK_H]

    # Row offsets within the tile.
    rows = row_start + tl.arange(0, ROWS_PER_PROGRAM)  # [ROWS_PER_PROGRAM]
    row_mask = rows < batch_size                       # [ROWS_PER_PROGRAM]

    # 2D address grid: [ROWS_PER_PROGRAM, BLOCK_H]
    offs = rows[:, None] * BLOCK_H + col[None, :]
    mask2d = row_mask[:, None]

    # Load tile in bf16, immediately promote to fp32 BEFORE any math.
    x = tl.load(x_ptr + offs, mask=mask2d, other=0.0).to(tl.float32)

    # Per-row sum of squares in fp32, reduce along hidden axis.
    xsq = x * x                                    # [ROWS_PER_PROGRAM, BLOCK_H] fp32
    sumsq = tl.sum(xsq, axis=1)                    # [ROWS_PER_PROGRAM] fp32

    # mean(x^2) = sumsq / H  -> use INV_H to avoid divide.
    mean_sq = sumsq * INV_H_C                      # [ROWS_PER_PROGRAM] fp32

    # eps INSIDE the rsqrt argument as required by the spec.
    inv_rms = tl.rsqrt(mean_sq + EPS_C)            # [ROWS_PER_PROGRAM] fp32

    # Compute y in fp32: y = (x * inv_rms) * w. Broadcast inv_rms along H,
    # w along rows. Single bf16 cast exactly once on the store.
    y = (x * inv_rms[:, None]) * w[None, :]        # [ROWS_PER_PROGRAM, BLOCK_H] fp32

    # Store with row-mask so out-of-bounds rows in the last tile are skipped.
    tl.store(y_ptr + offs, y.to(tl.bfloat16), mask=mask2d)


# ---------------------------------------------------------------------------
# Host wrapper.
# ---------------------------------------------------------------------------
def run(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Apply RMSNorm: y = (x * rsqrt(mean(x^2) + 1e-6)) * weight in bf16.

    Args:
        hidden_states: [batch_size, 2048] bfloat16 on CUDA.
        weight: [2048] bfloat16 on CUDA.

    Returns:
        [batch_size, 2048] bfloat16 on CUDA.
    """
    assert hidden_states.is_cuda and weight.is_cuda, "inputs must be on CUDA"
    assert hidden_states.dtype == torch.bfloat16, "hidden_states must be bf16"
    assert weight.dtype == torch.bfloat16, "weight must be bf16"
    assert hidden_states.ndim == 2 and hidden_states.shape[1] == HIDDEN_SIZE, (
        f"hidden_states must be [B, {HIDDEN_SIZE}], got {tuple(hidden_states.shape)}"
    )
    assert weight.ndim == 1 and weight.shape[0] == HIDDEN_SIZE, (
        f"weight must be [{HIDDEN_SIZE}], got {tuple(weight.shape)}"
    )

    x = hidden_states.contiguous()
    w = weight.contiguous()
    batch_size = x.shape[0]
    y = torch.empty_like(x)

    # Grid depends on autotuned ROWS_PER_PROGRAM.
    grid = lambda meta: (triton.cdiv(batch_size, meta["ROWS_PER_PROGRAM"]),)

    _rmsnorm_kernel[grid](
        x,
        w,
        y,
        batch_size,
        BLOCK_H=HIDDEN_SIZE,
        INV_H_C=INV_H,
        EPS_C=EPS,
    )
    return y


# Test harness expects this binding (entry_point = kernel.py::run).
kernel_function = run
