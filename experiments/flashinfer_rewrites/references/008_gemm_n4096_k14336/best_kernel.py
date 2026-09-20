import torch


# Round-5 large-step EDIT (a): torch.compile(mode='max-autotune') pre-warmed
# large-M dispatch, cached per (M, N, K) shape. Small-M (<128) stays on eager
# torch.matmul (1.0x baseline floor). No Triton kernel is invoked; Inductor
# picks its own codegen (may differ from eager cuBLASLt heuristic).
#
# Harness runs 10 warmup rounds before 50 timed iters, so max-autotune JIT
# compile time is amortized and does not enter the measured window.
#
# Rationale for structural novelty vs pool (Triton K-inner-product with M<128
# fallback): the large-M path here is a purely library/compiler path with no
# hand-written Triton. This tests whether Inductor's autotuned matmul on B200
# sm_100 can pick a better cuBLASLt algorithm or a fused codegen that beats
# eager torch.matmul(A, B.T) heuristic.

_SMALL_M_THRESHOLD = 128     # below this, eager is best (round-4 finding)
_COMPILE_M_THRESHOLD = 512   # above this, try compiled path

_COMPILED = {}


def _matmul_at_bt(A, B):
    # Recomputes C = A @ B.T on each call; torch.compile traces this closure.
    return torch.matmul(A, B.T)


def _get_compiled(shape_key):
    fn = _COMPILED.get(shape_key)
    if fn is None:
        try:
            fn = torch.compile(_matmul_at_bt, mode="max-autotune", dynamic=False, fullgraph=True)
        except Exception:
            # Older torch or unsupported mode; fall back to a no-compile shim.
            fn = _matmul_at_bt
        _COMPILED[shape_key] = fn
    return fn


def run(A, B):
    # Defensive: only handle the exact fp16 2D CUDA path; else eager.
    if (A.dtype != torch.float16 or B.dtype != torch.float16
            or A.dim() != 2 or B.dim() != 2
            or not A.is_cuda or not B.is_cuda):
        return torch.matmul(A, B.T)

    M, K = A.shape
    N = B.shape[0]

    # Small-M: eager is fastest per round-4 measurement.
    if M < _SMALL_M_THRESHOLD:
        return torch.matmul(A, B.T)

    # Mid-M (128 <= M < 512): eager remains the floor; no compile overhead risk.
    if M < _COMPILE_M_THRESHOLD:
        return torch.matmul(A, B.T)

    # Large-M: shape-cached compiled path with defensive try/except.
    try:
        A_c = A if A.is_contiguous() else A.contiguous()
        B_c = B if B.is_contiguous() else B.contiguous()
        compiled = _get_compiled((M, N, K))
        out = compiled(A_c, B_c)
        # torch.compile output should already be fp16 (M, N); guard shape/dtype.
        if out.dtype != torch.float16:
            out = out.to(torch.float16)
        return out
    except Exception:
        return torch.matmul(A, B.T)
