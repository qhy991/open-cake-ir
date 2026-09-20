"""
GEMM C = A @ B.T for NVIDIA B200 (sm_100). Lean cuBLAS DPS dispatch.

Winning technique (preserved from parent, locked in):
    torch.mm(A, B.T, out=C)
writes the product into the pre-allocated DPS buffer C.
"""

import torch


# ---- B200 sm_100 constants (informational; not used in the fast path) -----
NUM_SMS = 148

# ---- Pre-bound function references (skip per-call torch attr lookup) -----
_mm = torch.mm


@torch.inference_mode()
def run(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
    """DPS entry: C := A @ B.T.

    A: [M, K] fp16 contiguous
    B: [N, K] fp16 contiguous (N=5120, K=2048)
    C: [M, N] fp16 contiguous, pre-allocated by the harness
    """
    _mm(A, B.T, out=C)
