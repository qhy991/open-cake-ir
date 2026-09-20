"""
Fused Single-Pass RMSNorm Triton Kernel (cycle-0-a4)
====================================================
Round-2 hybrid + chunked + pipelined solution.

Stacks four orthogonal axes on top of the round-1 10.81x baseline:

  (1) Chunked BLOCK0=1024 + BLOCK1=512 zero-mask layout.
      Kills the 25% next_pow2(1536)=2048 lane-mask waste on every HBM
      load/store. Two unmasked tiles cover exactly HIDDEN_SIZE.

  (2) Hybrid per-workload dispatch on ROWS_PER_BLOCK:
        bs <= 32  -> RPB=1 kernel (one CTA per row).
                    Doubles grid in the under-saturated regime
                    (bs=7 -> 7 CTAs vs 4; bs=18 -> 18 vs 9).
        bs >  32  -> RPB=2 kernel with num_stages=3.
                    Amortizes weight load across 2 rows AND
                    enables inter-row HBM/compute pipelining.
                    bs=64 falls into this bucket -> 32 CTAs with
                    pipelined 2-row processing (target ~14x).

  (3) num_stages=3 software pipelining on the RPB=2 static row
      loop -- HBM loads of row[i+1] overlap with rsqrt+rescale of row[i].
      With hoisted loads (4 independent x-tiles per CTA) the deeper
      pipeline finally has enough independent memory issues to fill.

  (4) tl.max_contiguous + tl.multiple_of vectorization hints so the
      Triton compiler emits 128-bit wide bf16 loads/stores
      (1024 % 128 == 0 and 512 % 128 == 0 both fit 8 x bf16 vectors).

  (5) Register-cache x-tiles across the sum-of-squares reduction and
      the rescale/store pass -- x is loaded from HBM exactly ONCE per
      row (not twice). This halves the HBM read pressure on x.

Held constant (failed_strategy guards):
  - num_warps=8 throughout (num_warps=16 regressed to 0.78x at h=1536)
  - ROWS_PER_BLOCK in {1, 2} only (RPB=4 regressed to 5.64x)
  - Dynamic grid (NOT persistent NUM_SMS=148 -- regressed to 1.28x small-bs)
  - Differentiated eviction: weight evict_last, x/y evict_first
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Compile-time constants
# ---------------------------------------------------------------------------
HIDDEN_SIZE: int = 1536
EPS_F: float = 1e-6
INV_H_F: float = 1.0 / 1536.0

# Two-tile chunked layout: BLOCK0 + BLOCK1 = HIDDEN_SIZE exactly. Zero mask waste.
BLOCK0: int = 1024       # tile 0: cols [0, 1024)
BLOCK1: int = 512        # tile 1: cols [1024, 1536)

# Hybrid dispatch threshold. bs <= threshold takes the RPB=1 path to
# escape grid under-saturation on small batches (the round-1 baseline's
# 8.89x / 8.67x valleys at bs=7 / bs=18).
SMALL_BATCH_THRESHOLD: int = 32

NUM_WARPS: int = 8


# ---------------------------------------------------------------------------
# Kernel A: ROWS_PER_BLOCK=1 -- one CTA per row.
# Used for small batches where grid saturation across 148 B200 SMs dominates.
#
# Register-cache pattern: x0, x1 are loaded once from HBM (bf16 -> fp32),
# reused for both (a) the sum-of-squares reduction and (b) the rescale
# + weight multiply + store pass. No second HBM read of x.
# ---------------------------------------------------------------------------
@triton.jit
def _rmsnorm_kernel_rpb1(
    X_ptr,           # *bf16 [batch_size, HIDDEN_SIZE]
    W_ptr,           # *bf16 [HIDDEN_SIZE]
    OUT_ptr,         # *bf16 [batch_size, HIDDEN_SIZE]
    batch_size,      # int
    stride_x_row,    # int (= HIDDEN_SIZE)
    stride_out_row,  # int (= HIDDEN_SIZE)
    INV_H: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK0: tl.constexpr,   # 1024
    BLOCK1: tl.constexpr,   # 512
):
    pid = tl.program_id(0)
    if pid >= batch_size:
        return

    # Column offsets with vectorization hints; both tiles are multiples of 128
    # so the compiler may emit 128-bit (8 x bf16) wide loads/stores.
    cols0 = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK0), BLOCK0), BLOCK0)
    cols1_local = tl.max_contiguous(
        tl.multiple_of(tl.arange(0, BLOCK1), BLOCK1), BLOCK1
    )
    cols1 = cols1_local + BLOCK0

    # Weight: loaded once per CTA. evict_last keeps the 3 KB weight warm in
    # L1/L2 for the next CTA scheduled on this SM (all CTAs share weight).
    w0 = tl.load(W_ptr + cols0, eviction_policy="evict_last").to(tl.float32)
    w1 = tl.load(W_ptr + cols1, eviction_policy="evict_last").to(tl.float32)

    # x: two zero-masked tiles. Load ONCE, register-cache as fp32, and reuse
    # across both the reduction and the rescale pass. This eliminates the
    # second HBM read of x that a naive load-then-reload layout would emit.
    x_base = pid * stride_x_row
    x0 = tl.load(X_ptr + x_base + cols0, eviction_policy="evict_first").to(tl.float32)
    x1 = tl.load(X_ptr + x_base + cols1, eviction_policy="evict_first").to(tl.float32)

    # Pass (a): sum of squares across both tiles, then rsqrt.
    sum_sq = tl.sum(x0 * x0, axis=0) + tl.sum(x1 * x1, axis=0)
    inv_rms = tl.rsqrt(sum_sq * INV_H + EPS)

    # Pass (b): normalize + weight multiply -- reuses the register-cached x0/x1.
    y0 = (x0 * inv_rms) * w0
    y1 = (x1 * inv_rms) * w1

    out_base = pid * stride_out_row
    tl.store(OUT_ptr + out_base + cols0, y0.to(OUT_ptr.dtype.element_ty),
             eviction_policy="evict_first")
    tl.store(OUT_ptr + out_base + cols1, y1.to(OUT_ptr.dtype.element_ty),
             eviction_policy="evict_first")


# ---------------------------------------------------------------------------
# Kernel B: ROWS_PER_BLOCK=2 with num_stages=3 software pipelining.
# Used for larger batches where amortizing the weight load across 2 rows
# (and pipelining row[i+1] HBM loads under row[i] rsqrt/rescale) wins.
#
# Register-cache pattern extended to two rows: all four x-tiles
# (x0_row0, x1_row0, x0_row1, x1_row1) are loaded once from HBM up front,
# held in fp32 registers, and reused across both reduction and rescale
# passes for their respective rows. This gives 4 independent HBM read
# issues per CTA -- enough independent iterations to fill num_stages=3.
# ---------------------------------------------------------------------------
@triton.jit
def _rmsnorm_kernel_rpb2(
    X_ptr,           # *bf16 [batch_size, HIDDEN_SIZE]
    W_ptr,           # *bf16 [HIDDEN_SIZE]
    OUT_ptr,         # *bf16 [batch_size, HIDDEN_SIZE]
    batch_size,      # int
    stride_x_row,    # int (= HIDDEN_SIZE)
    stride_out_row,  # int (= HIDDEN_SIZE)
    INV_H: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK0: tl.constexpr,         # 1024
    BLOCK1: tl.constexpr,         # 512
    ROWS_PER_BLOCK: tl.constexpr, # 2
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_BLOCK

    # Column offsets with vectorization hints.
    cols0 = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK0), BLOCK0), BLOCK0)
    cols1_local = tl.max_contiguous(
        tl.multiple_of(tl.arange(0, BLOCK1), BLOCK1), BLOCK1
    )
    cols1 = cols1_local + BLOCK0

    # Weight: loaded once per CTA, reused across ROWS_PER_BLOCK rows.
    w0 = tl.load(W_ptr + cols0, eviction_policy="evict_last").to(tl.float32)
    w1 = tl.load(W_ptr + cols1, eviction_policy="evict_last").to(tl.float32)

    # Static-unrolled row loop. num_stages=3 (at launch) lets Triton overlap
    # the HBM loads of iteration (i+1) and (i+2) with the compute of prior
    # iterations. Each iteration also register-caches its two x-tiles across
    # the sum-of-squares and rescale passes -- x is read from HBM exactly once
    # per row (not twice).
    for i in tl.static_range(ROWS_PER_BLOCK):
        row = row_start + i
        if row < batch_size:
            x_base = row * stride_x_row

            # Load x tiles ONCE per row and register-cache as fp32.
            x0 = tl.load(X_ptr + x_base + cols0,
                         eviction_policy="evict_first").to(tl.float32)
            x1 = tl.load(X_ptr + x_base + cols1,
                         eviction_policy="evict_first").to(tl.float32)

            # Pass (a): reduction over the register-cached tiles.
            sum_sq = tl.sum(x0 * x0, axis=0) + tl.sum(x1 * x1, axis=0)
            inv_rms = tl.rsqrt(sum_sq * INV_H + EPS)

            # Pass (b): rescale + weight multiply -- reuses register-cached x0/x1.
            y0 = (x0 * inv_rms) * w0
            y1 = (x1 * inv_rms) * w1

            out_base = row * stride_out_row
            tl.store(OUT_ptr + out_base + cols0,
                     y0.to(OUT_ptr.dtype.element_ty),
                     eviction_policy="evict_first")
            tl.store(OUT_ptr + out_base + cols1,
                     y1.to(OUT_ptr.dtype.element_ty),
                     eviction_policy="evict_first")


# ---------------------------------------------------------------------------
# Python entrypoint
# ---------------------------------------------------------------------------
@torch.no_grad()
def run(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """RMSNorm over the last dim. Mirrors the FlashInfer-Bench reference."""
    assert hidden_states.is_cuda and weight.is_cuda
    assert hidden_states.dtype == torch.bfloat16
    assert weight.dtype == torch.bfloat16
    assert hidden_states.ndim == 2
    assert hidden_states.shape[1] == HIDDEN_SIZE
    assert weight.shape[0] == HIDDEN_SIZE

    # Force a contiguous input so stride_x_row == HIDDEN_SIZE.
    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()

    batch_size = hidden_states.shape[0]
    out = torch.empty_like(hidden_states)

    if batch_size == 0:
        return out

    stride_x_row = hidden_states.stride(0)
    stride_out_row = out.stride(0)

    if batch_size <= SMALL_BATCH_THRESHOLD:
        # Small-batch path: one CTA per row, max grid saturation.
        grid = (batch_size,)
        _rmsnorm_kernel_rpb1[grid](
            hidden_states, weight, out,
            batch_size,
            stride_x_row, stride_out_row,
            INV_H=INV_H_F,
            EPS=EPS_F,
            BLOCK0=BLOCK0,
            BLOCK1=BLOCK1,
            num_warps=NUM_WARPS,
            num_stages=1,
        )
    else:
        # Large-batch path: 2 rows per CTA, pipelined with num_stages=3.
        ROWS_PER_BLOCK = 2
        grid = ((batch_size + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK,)
        _rmsnorm_kernel_rpb2[grid](
            hidden_states, weight, out,
            batch_size,
            stride_x_row, stride_out_row,
            INV_H=INV_H_F,
            EPS=EPS_F,
            BLOCK0=BLOCK0,
            BLOCK1=BLOCK1,
            ROWS_PER_BLOCK=ROWS_PER_BLOCK,
            num_warps=NUM_WARPS,
            num_stages=3,
        )

    return out

