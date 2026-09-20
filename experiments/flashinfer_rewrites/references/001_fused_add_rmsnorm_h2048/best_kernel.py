# Optimization: Large-M compute-bubble fill via 2-row CTA unrolling with prefetch overlap
# (variant 1/2).
#
# Key transformations vs. parent kernel (iter-1-single-pass-fused-triton):
#   1. The large-M one-CTA-per-row kernel (_fused_add_rmsnorm_large) is replaced by
#      a 2-row unrolled kernel (_fused_add_rmsnorm_large_2row) where each CTA handles
#      exactly 2 rows (or 1 for the last odd CTA when M is odd).
#   2. Compute-bubble overlap: after loading hs_a+res_a for row N, the loads for row
#      N+1 (hs_b, res_b) are issued BEFORE the tl.sum and tl.rsqrt for row N execute.
#      Triton's memory scheduler dispatches these row-N+1 loads into the HBM pipeline
#      while the tl.sum cross-warp reduction (~50 cycles) and rsqrt SFU (~20 cycles)
#      execute for row N — filling the ~37-50 ns compute bubble with useful HBM
#      prefetch work.
#   3. Weight is loaded once per CTA (not once per row as in the parent), applied to
#      both rows — same L2 reuse pattern as the parent but across 2 rows now.
#   4. Grid shrinks by ~2x: ceil(M/2) CTAs vs M CTAs in the parent, halving kernel
#      launch and scheduler overhead for the large-M path.
#   5. The small-M path (ROWS_PER_CTA=4) is unchanged from the parent kernel.
#   6. Both kernel variants are pre-warmed at import time.
#
# Execution sequence within each CTA (2-row interleave):
#   (1) Load hs_a, res_a for row N           → HBM access
#   (2) Issue loads for hs_b, res_b (row N+1) → dispatched to HBM pipeline
#   (3) Compute x_a = hs_a + res_a           → ALU, overlaps with (2) in flight
#   (4) Compute var_a = tl.sum(x_a*x_a)      → cross-warp reduction (50 cycles)
#       During this reduction, row N+1 loads resolve from HBM concurrently
#   (5) inv_rms_a = tl.rsqrt(var_a + EPS)   → SFU (20 cycles), loads still overlap
#   (6) Load weight w                         → small (4KB), likely L2 hit
#   (7) y_a = x_a * inv_rms_a * w; store     → ALU + HBM write
#   (8) Compute x_b, var_b, inv_rms_b, y_b   → row N+1 data already in registers
#   (9) store y_b                             → HBM write

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel A: large-M path — 2-row unrolled CTA with prefetch overlap
# ---------------------------------------------------------------------------
@triton.jit
def _fused_add_rmsnorm_large_2row(
    HS_ptr, RES_ptr, W_ptr, OUT_ptr,
    M,
    hidden_size: tl.constexpr,
    EPS: tl.constexpr,
):
    """2-row unrolled large-M kernel: each CTA processes 2 consecutive rows.
    Row N+1 loads are issued before row N's tl.sum/rsqrt, overlapping HBM
    access with the mandatory compute bubble in the cross-warp reduction path.
    Grid = ceil(M / 2) CTAs. Weight loaded once per CTA, reused across both rows.
    """
    cta_id = tl.program_id(0)
    offs = tl.arange(0, hidden_size)

    # Row indices for this CTA
    row_a = cta_id * 2
    row_b = cta_id * 2 + 1

    # Load weight once per CTA — reused across both rows (4KB, L2 resident)
    w = tl.load(W_ptr + offs).to(tl.float32)

    # ----- Row A -----
    # Step 1: Load row A data from HBM
    hs_a  = tl.load(HS_ptr  + row_a * hidden_size + offs).to(tl.float32)
    res_a = tl.load(RES_ptr + row_a * hidden_size + offs).to(tl.float32)

    # Step 2: Issue row B loads NOW — before row A's reduction.
    # Triton's memory scheduler will dispatch these into the HBM pipeline.
    # They will resolve during steps 3-5 (row A compute bubble).
    # Use a mask to handle odd M (last CTA may have only 1 valid row).
    row_b_valid = row_b < M
    hs_b  = tl.load(HS_ptr  + row_b * hidden_size + offs,
                    mask=row_b_valid, other=0.0).to(tl.float32)
    res_b = tl.load(RES_ptr + row_b * hidden_size + offs,
                    mask=row_b_valid, other=0.0).to(tl.float32)

    # Step 3: Compute row A — row B loads are in-flight concurrently
    x_a  = hs_a + res_a
    x2_a = x_a * x_a
    # Step 4: cross-warp reduction (~50 cycles) — row B HBM loads resolve here
    var_a    = tl.sum(x2_a, axis=0) / hidden_size
    # Step 5: SFU rsqrt (~20 cycles) — row B loads fully resolved after this
    inv_rms_a = tl.rsqrt(var_a + EPS)

    # Step 7: Apply weight and store row A
    y_a = x_a * inv_rms_a * w
    tl.store(OUT_ptr + row_a * hidden_size + offs, y_a.to(tl.bfloat16))

    # ----- Row B (data already in registers from prefetch) -----
    if row_b_valid:
        x_b  = hs_b + res_b
        x2_b = x_b * x_b
        var_b     = tl.sum(x2_b, axis=0) / hidden_size
        inv_rms_b = tl.rsqrt(var_b + EPS)
        y_b = x_b * inv_rms_b * w
        tl.store(OUT_ptr + row_b * hidden_size + offs, y_b.to(tl.bfloat16))


# ---------------------------------------------------------------------------
# Kernel B: small-M path (ROWS_PER_CTA rows per CTA, weight reused)
# Unchanged from parent kernel.
# ---------------------------------------------------------------------------
@triton.jit
def _fused_add_rmsnorm_small(
    HS_ptr, RES_ptr, W_ptr, OUT_ptr,
    M,
    hidden_size: tl.constexpr,
    ROWS_PER_CTA: tl.constexpr,
    EPS: tl.constexpr,
):
    """One CTA handles ROWS_PER_CTA rows. Reduces grid size for small M."""
    cta_id = tl.program_id(0)
    offs = tl.arange(0, hidden_size)

    # Load weight once per CTA (4 KB, fits in L1; reused across all rows)
    w = tl.load(W_ptr + offs).to(tl.float32)

    # Process up to ROWS_PER_CTA rows in this CTA
    for i in tl.static_range(ROWS_PER_CTA):
        row = cta_id * ROWS_PER_CTA + i
        # Guard for the last CTA which may have fewer rows
        if row < M:
            hs  = tl.load(HS_ptr  + row * hidden_size + offs).to(tl.float32)
            res = tl.load(RES_ptr + row * hidden_size + offs).to(tl.float32)

            x  = hs + res
            x2 = x * x
            var     = tl.sum(x2, axis=0) / hidden_size
            inv_rms = tl.rsqrt(var + EPS)

            y = x * inv_rms * w
            tl.store(OUT_ptr + row * hidden_size + offs, y.to(tl.bfloat16))


# ---------------------------------------------------------------------------
# Dispatch threshold (same as parent kernel)
# ---------------------------------------------------------------------------
_SMALL_M_THRESHOLD = 128


# ---------------------------------------------------------------------------
# Pre-warm: compile both kernel variants at import time
# ---------------------------------------------------------------------------
def _prewarm():
    """Pre-JIT both kernel variants so compilation is not timed."""
    try:
        _hs  = torch.zeros(4, 2048, dtype=torch.bfloat16, device="cuda")
        _res = torch.zeros(4, 2048, dtype=torch.bfloat16, device="cuda")
        _w   = torch.ones(2048, dtype=torch.bfloat16, device="cuda")
        _out = torch.empty_like(_hs)

        # Warm large-M 2-row kernel (grid = (2,) for 4 rows)
        _fused_add_rmsnorm_large_2row[(2,)](
            _hs, _res, _w, _out,
            M=4,
            hidden_size=2048,
            EPS=1e-6,
            num_warps=8,
            num_stages=2,
        )

        # Warm small-M kernel (ROWS_PER_CTA=4, grid = (1,))
        _fused_add_rmsnorm_small[(1,)](
            _hs, _res, _w, _out,
            M=4,
            hidden_size=2048,
            ROWS_PER_CTA=4,
            EPS=1e-6,
            num_warps=4,
            num_stages=2,
        )

        torch.cuda.synchronize()
    except Exception:
        # Silent failure — pre-warm is best-effort
        pass


# Run pre-warm at import time
_prewarm()


# ---------------------------------------------------------------------------
# Public entry-point (signature unchanged from baseline)
# ---------------------------------------------------------------------------
def fused_add_rmsnorm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float = 1e-6,
) -> torch.Tensor:
    """Fused add-residual + RMSNorm kernel.

    Args:
        hidden_states: (M, hidden_size) bf16 tensor
        residual:      (M, hidden_size) bf16 tensor
        weight:        (hidden_size,) bf16 weight vector
        variance_epsilon: epsilon for numerical stability

    Returns:
        output: (M, hidden_size) bf16 tensor
    """
    M, hidden_size = hidden_states.shape
    output = torch.empty_like(hidden_states)

    if M <= _SMALL_M_THRESHOLD:
        # Small-M path: ROWS_PER_CTA=4, num_warps=4
        ROWS_PER_CTA = 4
        grid = ((M + ROWS_PER_CTA - 1) // ROWS_PER_CTA,)
        _fused_add_rmsnorm_small[grid](
            hidden_states, residual, weight, output,
            M=M,
            hidden_size=hidden_size,
            ROWS_PER_CTA=ROWS_PER_CTA,
            EPS=variance_epsilon,
            num_warps=4,
            num_stages=2,
        )
    else:
        # Large-M path: 2-row unrolled CTA with prefetch overlap
        # Grid is ceil(M/2) — half the CTAs vs the parent's M CTAs
        grid = ((M + 1) // 2,)
        _fused_add_rmsnorm_large_2row[grid](
            hidden_states, residual, weight, output,
            M=M,
            hidden_size=hidden_size,
            EPS=variance_epsilon,
            num_warps=8,
            num_stages=2,
        )

    return output

