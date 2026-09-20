// ============================================================================
// gemm_n4096_k14336 — candidate sol_2_1 (iter-2 / cand-1)
// Hybrid of the two measured iter-1 candidates (B300 SXM6 AC / sm_103 /
// 148 SMs, torch 2.11.0+cu130):
//   * sol_1_1 (cand-1): 1.0198x geomean, 43/43 PASS — GEMV for M<=4
//   * sol_1_2 (cand-2): 0.9971x geomean, 43/43 PASS — GEMV for M<=8
//
// C = A @ B.T   ·  A:[M,K] fp16   ·  B:[N,K] fp16   ·  C:[M,N] fp16
// N=4096, K=14336 fixed; M variable in [1, 8192] (43 workloads).
//
// Per-workload evidence driving this hybrid:
//   M=1 : cand-2 kernel measured 31.26 us vs cand-1 35.07 us (1.193x vs
//         1.053x) — the wider per-lane B loads + __ldcs streaming hint won.
//   M=2 : cand-1 31.42 us vs cand-2 31.71 us — a tie within noise (~1.17x).
//   M=4 : BOTH custom paths LOSE to cuBLAS (cand-1 0.925x, cand-2 0.840x).
//         cuBLAS reference is 36.8 us there; the custom path pays for extra
//         A-load issue slots and MSHR entries that delay the B stream.
//   M=7,8: cand-2's extension was a large regression (0.701x / 0.617x);
//         cuBLAS measured 1.028x for both.
//   M>=15: cuBLAS passthrough measured 1.00x-1.04x throughout; the M>=2053
//         workloads are compute-bound and sit on the roofline (M=8192:
//         0.5797 ms vs 0.5801 ms ideal), so there is nothing to win.
//
// Strategy: keep cand-2's improved GEMV exactly where it measured a win
// (M in {1,2}) and route everything else through at::matmul(A, B.t()) — the
// identical op to the reference, preserving the per-workload >= 1.0x floor
// for 41/43 workloads. Expected geomean vs cand-1's 1.0198x:
//   x (1.000/0.9252 * 1.1934/1.0529)^(1/43) ~= 1.025x.
//
// No CUTLASS dependency, no machine-specific include paths, no state
// retained across calls. Portable CUDA only (uint4 loads, half2 conversions,
// warp shuffles), so it builds for sm_103 regardless of the gencode flags
// torch's arch list provides.
// ============================================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {

constexpr int kSplitK      = 4;                        // K slices for the tiny-M path
constexpr int kThreads     = 256;                      // 8 warps per block
constexpr int kWarpsPerBlk = kThreads / 32;            // 8
constexpr int kNPerWarp    = 2;                        // rows of B owned by one warp
constexpr int kNPerBlk     = kWarpsPerBlk * kNPerWarp; // 16 rows of B per block
constexpr int kLaneHalves  = 16;                       // 32B (2x uint4) of B per lane per row per iteration
constexpr int kChunkK      = 32 * kLaneHalves;         // 512 k-values per warp iteration

__device__ __forceinline__ float dot8(uint4 a, uint4 b) {
    const __half2* ah = reinterpret_cast<const __half2*>(&a);
    const __half2* bh = reinterpret_cast<const __half2*>(&b);
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 af = __half22float2(ah[i]);
        const float2 bf = __half22float2(bh[i]);
        s += af.x * bf.x + af.y * bf.y;
    }
    return s;
}

// ---------------------------------------------------------------------------
// Stage 1: partial[s][m][n] = sum over the s-th K-slice of A[m,k]*B[n,k].
//
// One warp owns two rows of B (two columns of C for every A row). Each lane
// issues 32 bytes (2x uint4) per B row per iteration with warp stride 512
// along K, so each B element is touched exactly once across the whole grid.
// The four B loads of an iteration are independent and are all issued before
// the first dependent conversion — the memory-level parallelism that took
// M=1 from 35.07 us (sol_1_1) to 31.26 us (sol_2_1's kernel body). B is read
// exactly once, hence __ldcs (evict-first / streaming). A (<= 2 x 28 KB) is
// read through the read-only L2-cached path (__ldg); each A element is
// re-read by the x-blocks but stays L2 resident.
//
// Requires (checked host-side): N % 16 == 0, K % (kChunkK * kSplitK) == 0,
// M in {1, 2}. All 32 lanes of every warp are always active, so the shuffle
// reduction below is divergence-free.
// ---------------------------------------------------------------------------
template <int M>
__global__ void gemv_splitk_stage1(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ partial,
    const int N,
    const int K)
{
    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    const int n0    = blockIdx.x * kNPerBlk + warp * kNPerWarp;  // n0, n0+1 < N
    const int split = blockIdx.y;

    const int k_per_split = K / kSplitK;              // multiple of kChunkK
    const int k_base      = split * k_per_split;

    const __half* __restrict__ Br0 = B + (int64_t)n0 * K;
    const __half* __restrict__ Br1 = B + (int64_t)(n0 + 1) * K;

    float acc[kNPerWarp][M];
#pragma unroll
    for (int nn = 0; nn < kNPerWarp; ++nn)
#pragma unroll
        for (int m = 0; m < M; ++m)
            acc[nn][m] = 0.0f;

    for (int c = 0; c < k_per_split; c += kChunkK) {
        const int k = k_base + c + lane * kLaneHalves;

        // Four independent streaming loads of B, issued back to back before
        // any dependent use. B is read exactly once, hence __ldcs.
        const uint4 b00 = __ldcs(reinterpret_cast<const uint4*>(Br0 + k));
        const uint4 b01 = __ldcs(reinterpret_cast<const uint4*>(Br0 + k + 8));
        const uint4 b10 = __ldcs(reinterpret_cast<const uint4*>(Br1 + k));
        const uint4 b11 = __ldcs(reinterpret_cast<const uint4*>(Br1 + k + 8));

#pragma unroll
        for (int m = 0; m < M; ++m) {
            const uint4 a0 = __ldg(reinterpret_cast<const uint4*>(A + (int64_t)m * K + k));
            const uint4 a1 = __ldg(reinterpret_cast<const uint4*>(A + (int64_t)m * K + k + 8));

            acc[0][m] += dot8(a0, b00) + dot8(a1, b01);
            acc[1][m] += dot8(a0, b10) + dot8(a1, b11);
        }
    }

    // Warp-level reduction; lane 0 writes both n columns of this slice.
#pragma unroll
    for (int nn = 0; nn < kNPerWarp; ++nn) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
            float v = acc[nn][m];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                v += __shfl_down_sync(0xffffffffu, v, off);
            if (lane == 0)
                partial[((int64_t)split * M + m) * N + (n0 + nn)] = v;
        }
    }
}

// ---------------------------------------------------------------------------
// Stage 2: C[m,n] = half(sum_s partial[s][m][n]). Reads <= 128 KB.
// Both partial and C are [.., M, N] row-major, so a linear index maps to the
// same (m, n) in both. Same stream as stage 1 -> ordered after it.
// ---------------------------------------------------------------------------
__global__ void gemv_splitk_stage2(
    const float* __restrict__ partial,
    __half* __restrict__ C,
    const int MN)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= MN) return;
    float v = 0.0f;
#pragma unroll
    for (int s = 0; s < kSplitK; ++s)
        v += partial[(int64_t)s * MN + idx];
    C[idx] = __float2half(v);
}

bool use_custom_path(int M, int64_t N, int64_t K) {
    return (M == 1 || M == 2)
        && (N % kNPerBlk == 0)
        && (K % (int64_t)(kChunkK * kSplitK) == 0);
}

} // namespace

torch::Tensor run(const torch::Tensor& A, const torch::Tensor& B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2-D");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == torch::kHalf, "A must be float16");
    TORCH_CHECK(B.scalar_type() == torch::kHalf, "B must be float16");
    TORCH_CHECK(A.size(1) == B.size(1), "inner dimensions must match");

    const int M = (int)A.size(0);
    const int N = (int)B.size(0);
    const int K = (int)A.size(1);

    if (!use_custom_path(M, N, K)) {
        // cuBLAS passthrough — the identical op to the reference, so this is
        // the per-workload >= 1.0x floor for every M > 2 workload. Measured
        // evidence: both custom GEMV variants LOSE to cuBLAS at M >= 4
        // (sol_1_1 M=4: 0.925x; sol_1_2 M=4/7/8: 0.840x/0.701x/0.617x).
        return torch::matmul(A, B.t());
    }

    auto Ac = A.is_contiguous() ? A : A.contiguous();
    auto Bc = B.is_contiguous() ? B : B.contiguous();

    auto C = torch::empty({(int64_t)M, (int64_t)N}, Ac.options());
    auto partial = torch::empty(
        {(int64_t)kSplitK, (int64_t)M, (int64_t)N},
        Ac.options().dtype(torch::kFloat));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const __half* A_ptr = reinterpret_cast<const __half*>(Ac.data_ptr<at::Half>());
    const __half* B_ptr = reinterpret_cast<const __half*>(Bc.data_ptr<at::Half>());
    float* P_ptr        = partial.data_ptr<float>();
    __half* C_ptr       = reinterpret_cast<__half*>(C.data_ptr<at::Half>());

    dim3 grid((unsigned)(N / kNPerBlk), (unsigned)kSplitK);

    switch (M) {
        case 1:
            gemv_splitk_stage1<1><<<grid, kThreads, 0, stream>>>(A_ptr, B_ptr, P_ptr, N, K);
            break;
        default:
            gemv_splitk_stage1<2><<<grid, kThreads, 0, stream>>>(A_ptr, B_ptr, P_ptr, N, K);
            break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int MN = M * N;
    gemv_splitk_stage2<<<(MN + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        P_ptr, C_ptr, MN);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "gemm_n4096_k14336: C = A @ B.T (fp16; split-K GEMV for M<=2, "
          "cuBLAS otherwise)");
}
