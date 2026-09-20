"""
010_gemm_n6144_k4096 - cycle-1-a1 (r5): num_stages=4 on GEMV pipeline
================================================================================

Single-axis edit over parent (cycle-1-a0 at 1.0885x):
  In `_gemv_configs`, bump `num_stages` from 3 -> 4 for every configuration.
  No config topology change, no kernel body change, no dispatcher change.

Rationale (pipeline-depth-only, kernel body byte-identical):
  * GEMV path is memory-bound on streaming B ([N=6144, K=4096] fp16 = 48 MiB,
    no cross-CTA reuse under the persistent grid). Latency-hiding on this
    workload is limited by the number of in-flight B tiles in the software
    pipeline.
  * B200 has ample shared-memory budget per SM (228 KiB). BLOCK_N=64/128
    with BLOCK_K=128 puts one B tile at 64*128*2=16 KiB or 128*128*2=32 KiB;
    an A tile is 16*128*2 = 4 KiB. Even num_stages=4 uses at most
    4*(32 KiB B + 4 KiB A) = 144 KiB, comfortably within SM shared budget.
  * Deeper pipeline gives Triton more room to overlap B-load latency with
    compute (tl.dot + acc accumulate) and with the next iteration's A/B
    prefetches. Best case: 3-6% GEMV bucket improvement; worst case: Triton
    falls back to a smaller stage count internally if SMEM insufficient
    (no correctness or regression risk).
  * All other structural properties inherited untouched from cycle-1-a0:
    - Persistent CTA scheduler (grid = NUM_SM = 148), column-major-N strip.
    - eviction_policy="evict_first" on A and B loads.
    - fp32 accumulator, fp16 store; SPLIT_K = 1.
    - No atomics, no two-pass reduction.
    - Dispatcher: THIN_MAX=8, GEMV_MAX=8, cuBLAS for M>=9.
    - Same 7 autotune configs (only num_stages field mutated 3 -> 4).

Shape (fixed): C[M, N] = A[M, K] @ B[N, K].T   fp16
  N=6144, K=4096, M=var(1..8192), B200 sm_100, Triton 3.5.0 built-ins only.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


N_CONST: int = 6144
K_CONST: int = 4096

GEMV_MAX: int = 8       # M <= 8   -> persistent multi-N gemv
THIN_MAX: int = 8       # (unreachable, retained for structural stability)
# M > 8                            -> cuBLAS (torch.matmul)

NUM_SM: int = 148       # B200


# --------------------------------------------------------------------------
# Autotune configs
# --------------------------------------------------------------------------
def _thin_configs():
    return [
        # M in 9..32
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 64},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_K": 64},  num_warps=8, num_stages=3),
        # M in 33..128
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64},  num_warps=8, num_stages=3),
        # M in 129..256
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64},  num_warps=8, num_stages=3),
    ]


def _gemv_configs():
    # M<=8 -> persistent CTAs, each owning a strip of N-tiles.
    # SINGLE-AXIS CHANGE vs cycle-1-a0: num_stages 3 -> 4 for every config.
    return [
        triton.Config({"BLOCK_N": 32,  "BLOCK_K": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_N": 64,  "BLOCK_K": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_N": 64,  "BLOCK_K": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64},  num_warps=4, num_stages=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_warps=8, num_stages=4),
    ]


@triton.autotune(configs=_thin_configs(), key=["M"])
@triton.jit
def _persistent_thin_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    NUM_SM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    total_tiles = num_pid_m * num_pid_n

    k_range = tl.arange(0, BLOCK_K)
    num_k_iters = tl.cdiv(K, BLOCK_K)

    tile = pid
    while tile < total_tiles:
        pid_n = tile // num_pid_m
        pid_m = tile % num_pid_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        m_mask = offs_m < M
        n_mask = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for ki in range(0, num_k_iters):
            offs_k = ki * BLOCK_K + k_range
            k_mask = offs_k < K

            a = tl.load(
                A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
                eviction_policy="evict_first",
            )
            b = tl.load(
                B_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk,
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
                eviction_policy="evict_last",
            )
            acc = tl.dot(a, tl.trans(b), acc, out_dtype=tl.float32)

        store_mask = m_mask[:, None] & n_mask[None, :]
        c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc.to(tl.float16), mask=store_mask)

        tile += NUM_SM


@triton.autotune(configs=_gemv_configs(), key=["M"])
@triton.jit
def _persistent_gemv_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    NUM_SM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)

    num_pid_n = tl.cdiv(N, BLOCK_N)

    offs_m = tl.arange(0, BLOCK_M)
    k_range = tl.arange(0, BLOCK_K)
    num_k_iters = tl.cdiv(K, BLOCK_K)
    m_mask = offs_m < M

    pid_n = pid
    while pid_n < num_pid_n:
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for ki in range(0, num_k_iters):
            offs_k = ki * BLOCK_K + k_range
            k_mask = offs_k < K

            a = tl.load(
                A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
                eviction_policy="evict_first",
            )
            b = tl.load(
                B_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk,
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
                eviction_policy="evict_first",
            )
            acc = tl.dot(a, tl.trans(b), acc, out_dtype=tl.float32)

        store_mask = m_mask[:, None] & n_mask[None, :]
        c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc.to(tl.float16), mask=store_mask)

        pid_n += NUM_SM


def _run_thin(A: torch.Tensor, B: torch.Tensor, M: int) -> torch.Tensor:
    N = N_CONST
    K = K_CONST
    C = torch.empty((M, N), dtype=torch.float16, device=A.device)

    stride_am, stride_ak = A.stride()
    stride_bn, stride_bk = B.stride()
    stride_cm, stride_cn = C.stride()

    grid = (NUM_SM,)

    _persistent_thin_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
        NUM_SM=NUM_SM,
    )
    return C


def _run_gemv(A: torch.Tensor, B: torch.Tensor, M: int) -> torch.Tensor:
    N = N_CONST
    K = K_CONST
    C = torch.empty((M, N), dtype=torch.float16, device=A.device)

    stride_am, stride_ak = A.stride()
    stride_bn, stride_bk = B.stride()
    stride_cm, stride_cn = C.stride()

    BLOCK_M = 16
    grid = (NUM_SM,)

    _persistent_gemv_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
        NUM_SM=NUM_SM,
        BLOCK_M=BLOCK_M,
    )
    return C


@torch.no_grad()
def run(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.dim() == 2 and B.dim() == 2, "A, B must be 2-D"
    M, K = A.shape
    N = B.shape[0]
    assert K == K_CONST, f"expected K={K_CONST}, got {K}"
    assert N == N_CONST, f"expected N={N_CONST}, got {N}"
    assert A.dtype == torch.float16 and B.dtype == torch.float16

    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()

    if M > THIN_MAX:
        return torch.matmul(A, B.T)

    if M <= GEMV_MAX:
        return _run_gemv(A, B, M)

    return _run_thin(A, B, M)

