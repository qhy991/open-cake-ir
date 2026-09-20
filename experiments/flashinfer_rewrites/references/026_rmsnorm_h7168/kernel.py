import torch
import triton
import triton.language as tl

HIDDEN_SIZE = 7168
BLOCK_SIZE = 8192  # next pow2 >= 7168; single block covers the whole row
EPS = 1e-6


@triton.jit
def _rmsnorm_fused_kernel(
    X_ptr,           # *bf16 [B, H]
    W_ptr,           # *bf16 [H]
    Y_ptr,           # *bf16 [B, H]
    stride_xb,
    stride_yb,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    # One program per row. BLOCK is a fully-unrolled constexpr covering the row.
    row = tl.program_id(0)
    x_row_ptr = X_ptr + row * stride_xb
    y_row_ptr = Y_ptr + row * stride_yb

    offs = tl.arange(0, BLOCK)
    mask = offs < HIDDEN

    # ---- Fused single-pass RMSNorm ----
    # Load x (bf16) once, cast to fp32, HOLD IN REGISTERS across the
    # sum-of-squares reduction AND the scale/write epilogue.  This eliminates
    # the pass-2 HBM re-load that the parent (2-pass NUM_CHUNKS=2) incurred.
    x = tl.load(x_row_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Preload weight once as fp32 (hint 2: bf16*bf16 loses mantissa on tail).
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Reduction in fp32 (hint 1: fp32 accumulator kept for rsqrt).
    ss = tl.sum(x * x, axis=0)
    inv_rms = tl.rsqrt(ss / HIDDEN + EPS)

    # Fused epilogue: multiply in fp32, cast to bf16 only at store.
    y = (x * inv_rms) * w
    tl.store(y_row_ptr + offs, y.to(tl.bfloat16), mask=mask)


def rmsnorm_triton(hidden_states: torch.Tensor,
                   weight: torch.Tensor,
                   output: torch.Tensor) -> torch.Tensor:
    assert hidden_states.is_cuda and weight.is_cuda and output.is_cuda
    assert hidden_states.dtype == torch.bfloat16
    assert weight.dtype == torch.bfloat16
    assert output.dtype == torch.bfloat16
    B, H = hidden_states.shape
    assert H == HIDDEN_SIZE, f"expected hidden_size={HIDDEN_SIZE}, got {H}"
    assert output.shape == hidden_states.shape

    # Ensure contiguity along last dim (128-bit vectorized loads).
    if hidden_states.stride(-1) != 1:
        hidden_states = hidden_states.contiguous()
    if weight.stride(-1) != 1:
        weight = weight.contiguous()

    grid = (B,)
    _rmsnorm_fused_kernel[grid](
        hidden_states,
        weight,
        output,
        hidden_states.stride(0),
        output.stride(0),
        HIDDEN=HIDDEN_SIZE,
        BLOCK=BLOCK_SIZE,
        EPS=EPS,
        num_warps=8,   # hint 6: 8 warps = 256 threads, ~32 elems each
        num_stages=2,
    )
    return output


def run(hidden_states: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor) -> torch.Tensor:
    """DPS entry point: run(hidden_states, weight, output)."""
    return rmsnorm_triton(hidden_states, weight, output)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self,
                hidden_states: torch.Tensor,
                weight: torch.Tensor,
                output: torch.Tensor = None) -> torch.Tensor:
        if output is None:
            output = torch.empty_like(hidden_states)
        return rmsnorm_triton(hidden_states, weight, output)
