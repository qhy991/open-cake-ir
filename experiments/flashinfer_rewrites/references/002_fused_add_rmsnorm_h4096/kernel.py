# Optimization: Bimodal host-side dispatch into two structurally distinct
# single-pass Triton kernels for fused Add(residual) + row-wise RMSNorm.
#
# Structural strategy (node-1-L, large-step proposal):
#
#   (A) Small-batch path  (M <= SMALL_M_THRESHOLD):
#         Non-persistent kernel, grid = (ceil(M / ROWS_PER_CTA),).
#         ROWS_PER_CTA = 2.  Each CTA processes 2 rows of 4096 elements,
#         entire row held in registers across the sum and normalize phases.
#         num_warps=8 (256 threads) -> 4096/256 = 16 fp32 regs per thread
#         for the row data, plus 16 fp32 regs for the row weight (re-used
#         across both rows of the CTA).  Selected because small-M
#         (M <= 170 in the workload distribution) would *under-saturate*
#         a persistent 148-CTA grid (skill rule: persistent regressed to
#         1.28x on small batches).  Grid stays light and launch overhead
#         is small (32 CTAs for M=64, 86 for M=170).
#
#   (B) Large-batch persistent path  (M > SMALL_M_THRESHOLD):
#         Persistent kernel, grid = (min(2*num_SMs, M),) = 296 CTAs.
#         Each CTA loads weight ONCE into fp32 registers, then grid-strides
#         over rows reusing that in-register weight tile.  2x oversubscription
#         (296 vs 148) increases occupancy because each CTA holds only one
#         row + weight (~32 KB fp32 in regs) and the launch is host-free.
#         For M in {8804, 10827, 11832, 14418, 14509} each CTA handles
#         ~30..49 rows, fully amortizing launch + the one-time weight load.
#
# Both kernels share the single-pass invariant:
#   - x = hs + res computed once in fp32 registers (no HBM intermediate)
#   - variance from x*x via tl.sum
#   - inv_rms = rsqrt(var + EPS)
#   - y = x * inv_rms * w   (x reused from registers, no reload)
#   - one tl.load per input tensor per row, one tl.store per row
#
# Numerical: EPS = 1e-5 to match the PyTorch reference (1e-2 atol/rtol).
# fp32 accumulator throughout; bf16 only at the I/O boundary.
#
# Pure Triton 3.5.0 (no CUTLASS, no cpp_extension, no wgmma).
from __future__ import annotations

import torch
import triton
import triton.language as tl


HIDDEN_SIZE: int = 4096
EPS: float = 1e-5
INV_H: float = 1.0 / 4096.0  # 0.000244140625
SMALL_M_THRESHOLD: int = 512  # below: non-persistent; above: persistent x2
ROWS_PER_CTA_SMALL: int = 2

_NUM_SMS: int | None = None


def _get_num_sms() -> int:
    try:
        return torch.cuda.get_device_properties(0).multi_processor_count
    except Exception:
        return 148


# ---------------------------------------------------------------------------
# (A) Small-M kernel: non-persistent, ROWS_PER_CTA rows per CTA
# ---------------------------------------------------------------------------
@triton.jit
def _fused_add_rmsnorm_small(
    HS_ptr, RES_ptr, W_ptr, OUT_ptr, M,
    HIDDEN_SIZE: tl.constexpr,
    INV_H: tl.constexpr,
    EPS: tl.constexpr,
    ROWS_PER_CTA: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, HIDDEN_SIZE)

    # Weight loaded once per CTA, reused across ROWS_PER_CTA rows.
    w = tl.load(W_ptr + offs).to(tl.float32)

    for i in tl.static_range(ROWS_PER_CTA):
        row = pid * ROWS_PER_CTA + i
        if row < M:
            row_offs = row * HIDDEN_SIZE + offs
            hs = tl.load(HS_ptr + row_offs).to(tl.float32)
            res = tl.load(RES_ptr + row_offs).to(tl.float32)
            x = hs + res
            var = tl.sum(x * x, axis=0) * INV_H
            inv_rms = tl.rsqrt(var + EPS)
            out = (x * inv_rms * w).to(tl.bfloat16)
            tl.store(OUT_ptr + row_offs, out)


# ---------------------------------------------------------------------------
# (B) Large-M kernel: persistent, 2x SM oversubscription, grid-strided rows
# ---------------------------------------------------------------------------
@triton.jit
def _fused_add_rmsnorm_persistent(
    HS_ptr, RES_ptr, W_ptr, OUT_ptr, M,
    HIDDEN_SIZE: tl.constexpr,
    INV_H: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    offs = tl.arange(0, HIDDEN_SIZE)

    # Weight: loaded ONCE per CTA, reused across all rows this CTA handles.
    w = tl.load(W_ptr + offs).to(tl.float32)

    row = pid
    while row < M:
        row_offs = row * HIDDEN_SIZE + offs
        hs = tl.load(HS_ptr + row_offs).to(tl.float32)
        res = tl.load(RES_ptr + row_offs).to(tl.float32)
        x = hs + res
        var = tl.sum(x * x, axis=0) * INV_H
        inv_rms = tl.rsqrt(var + EPS)
        out = (x * inv_rms * w).to(tl.bfloat16)
        tl.store(OUT_ptr + row_offs, out)
        row += grid_size


# ---------------------------------------------------------------------------
# Python wrapper — bimodal host-side dispatch
# ---------------------------------------------------------------------------
@torch.no_grad()
def run(hidden_states: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = _get_num_sms()

    M, H = hidden_states.shape
    assert H == HIDDEN_SIZE, f"expected hidden_size={HIDDEN_SIZE}, got {H}"

    out = torch.empty_like(hidden_states)

    if M <= SMALL_M_THRESHOLD:
        grid_size = (M + ROWS_PER_CTA_SMALL - 1) // ROWS_PER_CTA_SMALL
        _fused_add_rmsnorm_small[(grid_size,)](
            hidden_states, residual, weight, out, M,
            HIDDEN_SIZE=HIDDEN_SIZE,
            INV_H=INV_H,
            EPS=EPS,
            ROWS_PER_CTA=ROWS_PER_CTA_SMALL,
            num_warps=8,
            num_stages=1,
        )
    else:
        # 2x SM oversubscription to hide latency on large-M sweeps.
        grid_size = min(2 * _NUM_SMS, M)
        _fused_add_rmsnorm_persistent[(grid_size,)](
            hidden_states, residual, weight, out, M,
            HIDDEN_SIZE=HIDDEN_SIZE,
            INV_H=INV_H,
            EPS=EPS,
            num_warps=8,
            num_stages=1,
        )
    return out


# ---------------------------------------------------------------------------
# Pre-warm: JIT compile both kernels outside the measured window.
# ---------------------------------------------------------------------------
def _warmup():
    try:
        if not torch.cuda.is_available():
            return
        dev = "cuda"
        w = torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16, device=dev)
        # Small path
        h_s = torch.zeros(4, HIDDEN_SIZE, dtype=torch.bfloat16, device=dev)
        r_s = torch.zeros(4, HIDDEN_SIZE, dtype=torch.bfloat16, device=dev)
        run(h_s, r_s, w)
        # Large path
        h_l = torch.zeros(1024, HIDDEN_SIZE, dtype=torch.bfloat16, device=dev)
        r_l = torch.zeros(1024, HIDDEN_SIZE, dtype=torch.bfloat16, device=dev)
        run(h_l, r_l, w)
        torch.cuda.synchronize()
    except Exception:
        pass


_warmup()
