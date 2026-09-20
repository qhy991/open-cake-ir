/*
 * fused_add_rmsnorm_h2048 — single-file candidate (iter0-s1)
 *
 * Selected features (CUDA-LLM FSR):
 *   - tile_row_cta_256t        : one full row (2048 elems, 4096 B) per CTA,
 *                                256 threads, 8 bf16/thread, grid = batch.
 *                                No inter-CTA communication; r1-verified
 *                                geometry, credited 4.27x in iter0-s0.
 *   - vec_ld_st_128bit         : each thread moves exactly 8 bf16 = 16 B =
 *                                one 128-bit vector per operand (hidden,
 *                                residual, weight, output). hidden_size = 2048
 *                                = 256 * 8, so the no-tail static_assert holds
 *                                and every access is 16 B-aligned on the
 *                                256 B-aligned torch allocations.
 *   - warp_block_reduce_smem   : the row spans 8 warps, so the sum-of-squares
 *                                is reduced with __shfl_xor_sync inside each
 *                                warp, the 8 per-warp partials meet in shared
 *                                memory, warp 0 finishes the reduction and the
 *                                result is broadcast through smem[0].
 *   - stcs_streaming_store     : the output row is stored with the streaming
 *                                (evict-first, st.global.cs) store intrinsic
 *                                __stcs instead of a plain vector store. The
 *                                output is written once and never re-read by
 *                                this kernel, so on B300's large L2 the store
 *                                streams through instead of write-allocating
 *                                ~66 MB of L2 on the 12383/16254-row
 *                                workloads, leaving L2 free for the input
 *                                streams. This is the README's explicit hint
 *                                for this chip.
 *
 * Dropped vs iter0-s0: host_l2_warm_clone. Its reward was suppressed as a
 * timing-harness-sensitive hack (rule 4), and the clones conflict with the
 * streaming-store philosophy (the kernel now reads the caller's tensors
 * directly from HBM/L2 and writes its output through). Squares accumulate via
 * fmaf so multiply-adds co-issue with the loads.
 *
 * Compile target: sm_103. No PyTorch calls from device code; the host side
 * only validates tensors, allocates output, and launches one kernel.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// ---------------------------------------------------------------------------
// Geometry constants — tile_row_cta_256t + vec_ld_st_128bit
// ---------------------------------------------------------------------------
constexpr int kHiddenSize     = 2048;                       // fixed by the task
constexpr int kThreadsPerCta  = 256;                        // 256-thread CTA
constexpr int kElemsPerThread = kHiddenSize / kThreadsPerCta;   // 8 bf16
constexpr int kWarpsPerCta    = kThreadsPerCta / 32;            // 8 warps

// 8 bf16 = 16 bytes = one 128-bit vector; no tail iteration exists.
static_assert(kElemsPerThread == 8, "expected 8 bf16 per 128-bit vector");
static_assert(kHiddenSize == kThreadsPerCta * kElemsPerThread,
              "hidden_size must tile exactly: no tail handling");
static_assert(kElemsPerThread * sizeof(__nv_bfloat16) == 16,
              "128-bit vector load/store requires 16 bytes per thread");

// ---------------------------------------------------------------------------
// warp_block_reduce_smem (warp shuffle partials + shared-memory meet)
// ---------------------------------------------------------------------------
__inline__ __device__ float warp_reduce_sum(float v)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

// Reduce `v` across the whole CTA: shuffle-reduce inside each warp, land the
// per-warp partials in shared memory, let warp 0 reduce those, and broadcast
// the final value to every thread through smem[0].
__inline__ __device__ float block_reduce_sum_smem(float v, float* warp_sums)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    v = warp_reduce_sum(v);

    if (lane == 0) {
        warp_sums[warp] = v;                // smem_reduce_scratch
    }
    __syncthreads();

    if (warp == 0) {
        float partial = (lane < kWarpsPerCta) ? warp_sums[lane] : 0.0f;
        partial = warp_reduce_sum(partial);
        if (lane == 0) {
            warp_sums[0] = partial;         // broadcast slot
        }
    }
    __syncthreads();

    return warp_sums[0];
}

// ---------------------------------------------------------------------------
// Kernel — one row per CTA (grid.x == batch), 128-bit vector I/O,
// streaming (evict-first) output stores.
// ---------------------------------------------------------------------------
__global__ void fused_add_rmsnorm_h2048_kernel(
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
          __nv_bfloat16* __restrict__ output,
    const float eps)
{
    // Row this CTA owns: 2048 elements = 4096 B, fully covered by 256 threads.
    const int    row  = blockIdx.x;
    const size_t base = static_cast<size_t>(row) * kHiddenSize +
                        static_cast<size_t>(threadIdx.x) * kElemsPerThread;

    // vec_ld_st_128bit: one 16-byte vector per operand per thread. The inputs
    // are read directly from the caller's tensors (no host-side clone).
    const uint4 h_vec = *reinterpret_cast<const uint4*>(hidden   + base);
    const uint4 r_vec = *reinterpret_cast<const uint4*>(residual + base);
    // weight is 4 KB, reused by every row: read through the read-only path.
    const uint4 w_vec = __ldg(
        reinterpret_cast<const uint4*>(weight +
            static_cast<size_t>(threadIdx.x) * kElemsPerThread));

    const __nv_bfloat16* h8 = reinterpret_cast<const __nv_bfloat16*>(&h_vec);
    const __nv_bfloat16* r8 = reinterpret_cast<const __nv_bfloat16*>(&r_vec);
    const __nv_bfloat16* w8 = reinterpret_cast<const __nv_bfloat16*>(&w_vec);

    // x = hidden + residual in fp32; accumulate squares with FMA while the
    // values are still in registers.
    float x[kElemsPerThread];
    float sumsq = 0.0f;
    #pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
        x[i] = __bfloat162float(h8[i]) + __bfloat162float(r8[i]);
        sumsq = fmaf(x[i], x[i], sumsq);
    }

    // Cross-warp reduction of the sum of squares.
    __shared__ float warp_sums[kWarpsPerCta];
    sumsq = block_reduce_sum_smem(sumsq, warp_sums);

    // inv_rms = rsqrt(mean(x^2) + eps); mean = sumsq / 2048 (exact power of 2).
    const float inv_rms = rsqrtf(sumsq * (1.0f / static_cast<float>(kHiddenSize)) + eps);

    // y = (x * inv_rms) * weight, cast once to bf16, emitted as one 128-bit
    // vector.
    uint4 o_vec;
    __nv_bfloat16* o8 = reinterpret_cast<__nv_bfloat16*>(&o_vec);
    #pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
        o8[i] = __float2bfloat16(x[i] * inv_rms * __bfloat162float(w8[i]));
    }

    // stcs_streaming_store: __stcs lowers to st.global.cs — an evict-first,
    // streaming store. The output is written exactly once and never re-read
    // by this kernel, so on B300's large L2 it streams through to HBM instead
    // of write-allocating L2 lines (~66 MB of output on the 12383/16254-row
    // workloads), preserving L2 capacity for the incoming hidden/residual
    // streams. Correctness is identical to a plain store.
    __stcs(reinterpret_cast<uint4*>(output + base), o_vec);
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
void fused_add_rmsnorm_h2048_launch(
    const void* hidden_states,
    const void* residual,
    const void* weight,
          void* output,
    int M,
    float eps,
    cudaStream_t stream)
{
    if (M <= 0) {
        return;
    }
    fused_add_rmsnorm_h2048_kernel<<<M, kThreadsPerCta, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden_states),
        static_cast<const __nv_bfloat16*>(residual),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_bfloat16*>(output),
        eps);
}

// ---------------------------------------------------------------------------
// Python-visible entry point (return-value style; matches reference.py)
// ---------------------------------------------------------------------------
torch::Tensor run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& residual,
    const torch::Tensor& weight)
{
    // Input validation
    TORCH_CHECK(hidden_states.is_cuda(),         "hidden_states must be CUDA tensor");
    TORCH_CHECK(residual.is_cuda(),              "residual must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(),                "weight must be CUDA tensor");

    TORCH_CHECK(hidden_states.dtype() == torch::kBFloat16, "hidden_states must be bfloat16");
    TORCH_CHECK(residual.dtype()      == torch::kBFloat16, "residual must be bfloat16");
    TORCH_CHECK(weight.dtype()        == torch::kBFloat16, "weight must be bfloat16");

    TORCH_CHECK(hidden_states.is_contiguous(), "hidden_states must be contiguous");
    TORCH_CHECK(residual.is_contiguous(),      "residual must be contiguous");
    TORCH_CHECK(weight.is_contiguous(),        "weight must be contiguous");

    TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be 2-D");
    TORCH_CHECK(residual.dim()      == 2, "residual must be 2-D");
    TORCH_CHECK(weight.dim()        == 1, "weight must be 1-D");

    const int M = static_cast<int>(hidden_states.size(0));
    const int N = static_cast<int>(hidden_states.size(1));
    TORCH_CHECK(N == 2048, "hidden_size must be 2048, got ", N);
    TORCH_CHECK(residual.size(0) == M && residual.size(1) == N, "residual shape mismatch");
    TORCH_CHECK(weight.size(0) == N, "weight size mismatch");

    // No host-side L2-warming clone: host_l2_warm_clone was dropped from the
    // feature set (timing-harness-sensitive reward, suppressed per rule 4).
    // The kernel reads the caller's hidden_states/residual tensors directly.

    // Allocate output with torch::empty_like — no GPU work enqueued.
    auto output = torch::empty_like(hidden_states);

    // Launch the single fused kernel (the only CUPTI-measured span).
    constexpr float EPS = 1e-6f;

    fused_add_rmsnorm_h2048_launch(
        static_cast<const void*>(hidden_states.data_ptr<at::BFloat16>()),
        static_cast<const void*>(residual.data_ptr<at::BFloat16>()),
        static_cast<const void*>(weight.data_ptr<at::BFloat16>()),
        static_cast<      void*>(output.data_ptr<at::BFloat16>()),
        M,
        EPS,
        at::cuda::getCurrentCUDAStream()
    );

    return output;
}

// ---------------------------------------------------------------------------
// Module registration — harness detects PYBIND11_MODULE and extracts "run"
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run",
          &run,
          "Fused residual add + RMSNorm, hidden_size=2048 (bfloat16, single-kernel)");
}
