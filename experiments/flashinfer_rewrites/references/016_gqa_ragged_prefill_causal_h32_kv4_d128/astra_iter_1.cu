/*
 * GQA Ragged Prefill (causal) Kernel — Candidate 5 (single-launch mma).
 *
 * Op: 016_gqa_ragged_prefill_causal (h=32, kv=4, d=128, causal, ragged KV).
 * Compile target: sm_103 (B300); mma.sync m16n8k16 bf16 requires sm_80+.
 *
 * Candidate-4 mma design (unchanged inner loops) with the launch path
 * collapsed to a single kernel: the build_tile_meta_kernel pre-pass, the
 * torch::zeros(num_tiles) and torch::full(lse) fill launches, and the
 * meta_gpu allocation are gone. Each CTA resolves its own tile metadata
 * in-kernel from qo_indptr/kv_indptr (thread-0 serial scan over the batch,
 * O(batch) trivial math; batch <= 34 in all contract workloads), and grid.y
 * is sized to a provable host-side upper bound on num_tiles instead of
 * total_q, so dead CTAs are never scheduled rather than launched to exit.
 *
 * Design:
 *   - Grid:  (NUM_KV_HEADS, num_tiles_ub)   [ub = batch + ceil(total_q/16)]
 *   - Block: 256 threads = 8 warps; warp w handles qh = kv_head*8 + w.
 *   - BRQ=16 query rows per CTA tile (mma M=16), T_PIPE=32 KV tokens per tile.
 *   - K/V staged in SMEM with +8-element row padding so the K B-fragment
 *     loads (32-bit each) are bank-conflict-free.
 *   - Online softmax at tile granularity: per-row tile max reduced over the
 *     4-lane fragment group (shfl_xor 1,2), one exp2 pass, one rescale of
 *     the fp32 O accumulators per 32-token tile.
 *   - Causal + ragged-tail masking applied on the fp32 logits in registers:
 *     masked entries set to -INFINITY before exp2 (exp2f(-inf) = 0 keeps
 *     rsum and O exact for masked/padded columns).
 *   - P rounded to bf16 for the P·V mma (same numeric class as
 *     FlashAttention-style kernels); QK and O accumulate in fp32.
 *   - Base-2 LSE = rmax + log2f(rsum); empty rows keep -INFINITY.
 *
 * m16n8k16 fragment layouts (PTX ISA, lane = g*4 + t, g = lane>>2, t = lane&3):
 *   A/Q: Ra0={A[g][2t],A[g][2t+1]} Ra1={A[g+8][2t],..} Ra2={A[g][2t+8],..}
 *        Ra3={A[g+8][2t+8],..}
 *   B/K: Rb0={B[2t][g],B[2t+1][g]} Rb1={B[2t+8][g],B[2t+9][g]},
 *        with B[k][n] = K[n][k] for QK^T (k = head dim, n = kv token).
 *   B/V: B[k][n] = V[k][n] directly (k = kv token, n = head dim).
 *   C:   c0=(g,2t) c1=(g,2t+1) c2=(g+8,2t) c3=(g+8,2t+1).
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <cstdint>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

static constexpr int HEAD_DIM      = 128;
static constexpr int NUM_QO_HEADS  = 32;
static constexpr int NUM_KV_HEADS  = 4;          // 016: kv_heads=4
static constexpr int GQA_RATIO     = 8;          // 32 / 4
static constexpr int WARPS_PER_BLOCK = 8;        // 8 warps = 8 q_heads per kv_head
static constexpr int BLOCK_THREADS = WARPS_PER_BLOCK * 32; // 256
static constexpr int BRQ           = 16;         // query rows per CTA tile (mma M=16)
static constexpr int T_PIPE        = 32;         // KV tokens per SMEM tile
static constexpr int ROWS_PER_WARP = T_PIPE / WARPS_PER_BLOCK; // = 4
static constexpr int KV_STRIDE     = HEAD_DIM + 8;  // +8 elem pad: conflict-free frag loads
static constexpr int NUM_KBLK      = HEAD_DIM / 16; // 8 k-blocks (head dim) for QK^T
static constexpr int NUM_SBLK      = T_PIPE / 8;    // 4 n-blocks (kv tokens) for S
static constexpr int NUM_OBLK      = HEAD_DIM / 8;  // 16 n-blocks (head dim) for O
static constexpr int NUM_PKB       = T_PIPE / 16;   // 2 k-blocks (kv tokens) for P·V

static_assert(BRQ == 16, "BRQ must match the mma m16 M dimension");
static_assert(T_PIPE % 16 == 0, "T_PIPE must be a multiple of the mma k16");

static constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ uint32_t pack_bf16_ff(float lo, float hi)
{
    // .x (low half) is the lower-index fragment element, as mma expects.
    __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&p);
}

__device__ __forceinline__ uint32_t pack_bf16_hh(__nv_bfloat16 lo, __nv_bfloat16 hi)
{
    unsigned short l = *reinterpret_cast<unsigned short*>(&lo);
    unsigned short h = *reinterpret_cast<unsigned short*>(&hi);
    return (uint32_t)l | ((uint32_t)h << 16);
}

// D = A (m16k16 bf16, row) * B (k16n8 bf16, col) + C, fp32 accumulate.
__device__ __forceinline__ void mma_m16n8k16_bf16(
    float* c,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

/*
 * SMEM layout (single buffer, T_PIPE=32, padded stride):
 *   smem_k[T_PIPE][KV_STRIDE], smem_v[T_PIPE][KV_STRIDE]
 *   = 32 * 136 * 2 * 2 = 17408 bytes = 17 KB
 * (plus 24 B of static s_meta, which coexists with the dynamic buffer).
 */
__global__ void __launch_bounds__(BLOCK_THREADS, 1)
gqa_ragged_prefill_mma(
    const __nv_bfloat16* __restrict__ q,         // [total_q, 32, 128]
    const __nv_bfloat16* __restrict__ k,         // [total_kv, 4, 128]
    const __nv_bfloat16* __restrict__ v,         // [total_kv, 4, 128]
    float                             sm_scale,
    const int*           __restrict__ qo_indptr, // [batch + 1]
    const int*           __restrict__ kv_indptr, // [batch + 1]
    int                               batch,
    __nv_bfloat16*       __restrict__ output,    // [total_q, 32, 128]
    float*               __restrict__ lse)       // [total_q, 32]
{
    extern __shared__ __nv_bfloat16 smem_raw[];
    __nv_bfloat16* smem_k = smem_raw;
    __nv_bfloat16* smem_v = smem_raw + T_PIPE * KV_STRIDE;

    const int kv_head = blockIdx.x;
    const int tidx    = blockIdx.y;
    const int tid      = threadIdx.x;
    const int warp_id  = tid >> 5;
    const int lane_id  = tid & 31;
    // m16n8k16 fragment group coordinates: lane = g*4 + t.
    const int g = lane_id >> 2;
    const int t = lane_id & 3;

    // ---- In-kernel tile resolution (replaces build_tile_meta_kernel) ----
    // Thread 0 scans qo_indptr serially, accumulating per-seq tile counts
    // (identical arithmetic to the old pre-pass), and fills the 6-int tile
    // meta this CTA owns. Valid tiles always have qlen >= 1, so qlen <= 0
    // is an unambiguous dead-CTA marker. The scan is O(batch); batch <= 34
    // in all contract workloads.
    __shared__ int s_meta[6];
    if (tid == 0) {
        int acc = 0;
        int found = 0;
        for (int b = 0; b < batch; b++) {
            const int q_start = __ldg(qo_indptr + b);
            const int seq_q   = __ldg(qo_indptr + b + 1) - q_start;
            const int cnt     = (seq_q > 0) ? ((seq_q + BRQ - 1) / BRQ) : 0;
            if (tidx < acc + cnt) {
                const int t_in_seq = tidx - acc;
                const int kv_start = __ldg(kv_indptr + b);
                const int num_kv   = __ldg(kv_indptr + b + 1) - kv_start;
                const int rem      = seq_q - t_in_seq * BRQ;
                s_meta[0] = q_start + t_in_seq * BRQ;
                s_meta[1] = (rem < BRQ) ? rem : BRQ;
                s_meta[2] = t_in_seq * BRQ;
                s_meta[3] = kv_start;
                s_meta[4] = num_kv;
                s_meta[5] = num_kv - seq_q;
                found = 1;
                break;
            }
            acc += cnt;
        }
        if (!found) s_meta[1] = 0;  // blockIdx.y >= num_tiles: dead CTA
    }
    // The barrier publishes s_meta before the uniform exit; tidx is uniform
    // per CTA, so all threads take the same branch and no one hangs.
    __syncthreads();

    const int q_start_global = s_meta[0];
    const int q_len          = s_meta[1];
    const int q_start_local  = s_meta[2];
    const int kv_start       = s_meta[3];
    const int num_kv         = s_meta[4];
    const int delta          = s_meta[5];
    if (q_len <= 0) return;

    const int qh = kv_head * GQA_RATIO + warp_id;
    const float sm_scale_log2 = sm_scale * LOG2E;
    const int kv_head_offset  = kv_head * HEAD_DIM;
    const int kv_stride       = NUM_KV_HEADS * HEAD_DIM;

    // ---- Q as bf16 A-fragments (16 rows x 128 dims), loaded once per CTA ----
    uint32_t qfrag[NUM_KBLK][4];
    {
        const bool has_lo = (g < q_len);
        const bool has_hi = (g + 8 < q_len);
        const __nv_bfloat16* qlo =
            q + ((q_start_global + g) * NUM_QO_HEADS + qh) * HEAD_DIM;
        const __nv_bfloat16* qhi =
            q + ((q_start_global + g + 8) * NUM_QO_HEADS + qh) * HEAD_DIM;
        #pragma unroll
        for (int kb = 0; kb < NUM_KBLK; kb++) {
            const int d0 = kb * 16 + 2 * t;
            qfrag[kb][0] = has_lo ? __ldg(reinterpret_cast<const uint32_t*>(qlo + d0))     : 0u;
            qfrag[kb][1] = has_hi ? __ldg(reinterpret_cast<const uint32_t*>(qhi + d0))     : 0u;
            qfrag[kb][2] = has_lo ? __ldg(reinterpret_cast<const uint32_t*>(qlo + d0 + 8)) : 0u;
            qfrag[kb][3] = has_hi ? __ldg(reinterpret_cast<const uint32_t*>(qhi + d0 + 8)) : 0u;
        }
    }

    // ---- O accumulators (16 rows x 128 dims fp32) + per-row running stats ----
    // Rows g (frag regs c0/c1) and g+8 (regs c2/c3) carry separate softmax state.
    float o[NUM_OBLK][4];
    #pragma unroll
    for (int nb = 0; nb < NUM_OBLK; nb++) {
        o[nb][0] = 0.f; o[nb][1] = 0.f; o[nb][2] = 0.f; o[nb][3] = 0.f;
    }
    float rmax_lo = -FLT_MAX, rmax_hi = -FLT_MAX;
    float rsum_lo = 0.0f,     rsum_hi = 0.0f;

    int max_max_kv = q_start_local + q_len + delta;
    if (max_max_kv > num_kv) max_max_kv = num_kv;
    if (max_max_kv < 0) max_max_kv = 0;
    const int num_kv_tiles = (max_max_kv + T_PIPE - 1) / T_PIPE;

    for (int tile = 0; tile < num_kv_tiles; tile++) {
        const int kv_lo      = tile * T_PIPE;
        const int tile_kv_count = (num_kv - kv_lo < T_PIPE) ? (num_kv - kv_lo) : T_PIPE;

        if (tile > 0) __syncthreads();

        // ---- Cooperative load of K/V tile into padded SMEM (8 warps * 4 rows) ----
        // Tail rows are zero-filled with the same uint2 store so the loop is
        // branch-uniform; bf16 0x0000 = +0 keeps padded mma columns harmless.
        const int warp_row_start = warp_id * ROWS_PER_WARP;
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; r++) {
            const int row = warp_row_start + r;
            __nv_bfloat16* dk = smem_k + row * KV_STRIDE + lane_id * 4;
            __nv_bfloat16* dv = smem_v + row * KV_STRIDE + lane_id * 4;
            if (row < tile_kv_count) {
                const int base =
                    (kv_start + kv_lo + row) * kv_stride + kv_head_offset + lane_id * 4;
                *reinterpret_cast<uint2*>(dk) =
                    __ldg(reinterpret_cast<const uint2*>(k + base));
                *reinterpret_cast<uint2*>(dv) =
                    __ldg(reinterpret_cast<const uint2*>(v + base));
            } else {
                *reinterpret_cast<uint2*>(dk) = make_uint2(0u, 0u);
                *reinterpret_cast<uint2*>(dv) = make_uint2(0u, 0u);
            }
        }

        __syncthreads();

        // ---- Causal limits for this tile (rows g and g+8) ----
        // lim = min(causal limit, num_kv, end of valid tokens in this tile).
        const int kv_end_tile = kv_lo + tile_kv_count;
        const int lim_lo = min(min(q_start_local + g + delta + 1,     num_kv), kv_end_tile);
        const int lim_hi = min(min(q_start_local + g + 8 + delta + 1, num_kv), kv_end_tile);

        // ---- S = Q K^T: mma over 8 dim k-blocks x 4 token n-blocks ----
        float s[NUM_SBLK][4];
        #pragma unroll
        for (int nb = 0; nb < NUM_SBLK; nb++) {
            s[nb][0] = 0.f; s[nb][1] = 0.f; s[nb][2] = 0.f; s[nb][3] = 0.f;
        }
        #pragma unroll
        for (int nb = 0; nb < NUM_SBLK; nb++) {
            // B[k][n] = K[n][k]: token nb*8+g, contiguous dims at 2t (and 2t+8).
            const __nv_bfloat16* krow = smem_k + (nb * 8 + g) * KV_STRIDE;
            #pragma unroll
            for (int kb = 0; kb < NUM_KBLK; kb++) {
                const int d0 = kb * 16 + 2 * t;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(krow + d0);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(krow + d0 + 8);
                mma_m16n8k16_bf16(s[nb],
                                  qfrag[kb][0], qfrag[kb][1],
                                  qfrag[kb][2], qfrag[kb][3],
                                  b0, b1);
            }
        }

        // ---- scale, causal/padding mask, per-row tile max ----
        float m_lo = -INFINITY, m_hi = -INFINITY;
        #pragma unroll
        for (int nb = 0; nb < NUM_SBLK; nb++) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int col = kv_lo + nb * 8 + 2 * t + (i & 1);
                const int lim = (i >= 2) ? lim_hi : lim_lo;
                float logit = s[nb][i] * sm_scale_log2;
                if (col >= lim) logit = -INFINITY;
                s[nb][i] = logit;
                if (i >= 2) m_hi = fmaxf(m_hi, logit);
                else        m_lo = fmaxf(m_lo, logit);
            }
        }
        // Reduce over the 4 lanes of the fragment group (t = lane & 3);
        // rmax init at -FLT_MAX (never -INF) keeps exp2f inputs finite-safe.
        m_lo = fmaxf(m_lo, __shfl_xor_sync(0xffffffff, m_lo, 1));
        m_lo = fmaxf(m_lo, __shfl_xor_sync(0xffffffff, m_lo, 2));
        m_hi = fmaxf(m_hi, __shfl_xor_sync(0xffffffff, m_hi, 1));
        m_hi = fmaxf(m_hi, __shfl_xor_sync(0xffffffff, m_hi, 2));

        const float new_max_lo = fmaxf(rmax_lo, m_lo);
        const float new_max_hi = fmaxf(rmax_hi, m_hi);
        const float ep_lo = exp2f(rmax_lo - new_max_lo);
        const float ep_hi = exp2f(rmax_hi - new_max_hi);

        // ---- exp2 pass; P (e-values in [0,1]) kept in s[] ----
        float sum_lo = 0.f, sum_hi = 0.f;
        #pragma unroll
        for (int nb = 0; nb < NUM_SBLK; nb++) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const float e = exp2f(s[nb][i] - ((i >= 2) ? new_max_hi : new_max_lo));
                s[nb][i] = e;
                if (i >= 2) sum_hi += e;
                else        sum_lo += e;
            }
        }
        sum_lo += __shfl_xor_sync(0xffffffff, sum_lo, 1);
        sum_lo += __shfl_xor_sync(0xffffffff, sum_lo, 2);
        sum_hi += __shfl_xor_sync(0xffffffff, sum_hi, 1);
        sum_hi += __shfl_xor_sync(0xffffffff, sum_hi, 2);

        rsum_lo = rsum_lo * ep_lo + sum_lo;
        rsum_hi = rsum_hi * ep_hi + sum_hi;
        rmax_lo = new_max_lo;
        rmax_hi = new_max_hi;

        // ---- rescale O once per 32-token tile (row g by ep_lo, g+8 by ep_hi) ----
        #pragma unroll
        for (int nb = 0; nb < NUM_OBLK; nb++) {
            o[nb][0] *= ep_lo; o[nb][1] *= ep_lo;
            o[nb][2] *= ep_hi; o[nb][3] *= ep_hi;
        }

        // ---- O += P (bf16) · V: 2 token k-blocks x 16 dim n-blocks ----
        #pragma unroll
        for (int kb = 0; kb < NUM_PKB; kb++) {
            // P A-fragments from the S fragments: tokens kb*16+{2t,2t+1} live
            // in s[2*kb] (cols 2t..2t+1) and tokens kb*16+{2t+8,2t+9} in
            // s[2*kb+1]; regs 0/2 are row g, regs 1/3 are row g+8.
            const uint32_t pa[4] = {
                pack_bf16_ff(s[2 * kb + 0][0], s[2 * kb + 0][1]),
                pack_bf16_ff(s[2 * kb + 0][2], s[2 * kb + 0][3]),
                pack_bf16_ff(s[2 * kb + 1][0], s[2 * kb + 1][1]),
                pack_bf16_ff(s[2 * kb + 1][2], s[2 * kb + 1][3]),
            };
            #pragma unroll
            for (int nb = 0; nb < NUM_OBLK; nb++) {
                // B[k][n] = V[k][n]: token rows 2t,2t+1 (b0) / 2t+8,2t+9 (b1),
                // column dim nb*8 + g.
                const __nv_bfloat16* v0 = smem_v + (kb * 16 + 2 * t) * KV_STRIDE + nb * 8 + g;
                const __nv_bfloat16* v1 = v0 + KV_STRIDE;
                const __nv_bfloat16* v2 = v0 + 8 * KV_STRIDE;
                const __nv_bfloat16* v3 = v2 + KV_STRIDE;
                const uint32_t b0 = pack_bf16_hh(*v0, *v1);
                const uint32_t b1 = pack_bf16_hh(*v2, *v3);
                mma_m16n8k16_bf16(o[nb], pa[0], pa[1], pa[2], pa[3], b0, b1);
            }
        }
    }

    // ---- epilogue: normalize, write bf16 O as paired dims + base-2 LSE ----
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        const int r = (half == 0) ? g : g + 8;
        if (r >= q_len) continue;
        const float rmax    = (half == 0) ? rmax_lo : rmax_hi;
        const float rsum    = (half == 0) ? rsum_lo : rsum_hi;
        const float inv_sum = (rsum > 0.0f) ? 1.0f / rsum : 0.0f;
        const int q_row_global = q_start_global + r;
        __nv_bfloat16* out_ptr =
            output + (q_row_global * NUM_QO_HEADS + qh) * HEAD_DIM + 2 * t;
        #pragma unroll
        for (int nb = 0; nb < NUM_OBLK; nb++) {
            // c0/c1 (row g) sit at dims nb*8+2t, +1; c2/c3 (row g+8) likewise.
            *reinterpret_cast<uint32_t*>(out_ptr + nb * 8) =
                pack_bf16_ff(o[nb][2 * half + 0] * inv_sum,
                             o[nb][2 * half + 1] * inv_sum);
        }
        if (t == 0) {
            float* lse_ptr = lse + (q_row_global * NUM_QO_HEADS + qh);
            if (rsum > 0.0f && isfinite(rsum)) {
                *lse_ptr = rmax + log2f(rsum);
            } else {
                *lse_ptr = -INFINITY;
            }
        }
    }
}

std::vector<torch::Tensor> run(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor qo_indptr,
    torch::Tensor kv_indptr,
    double sm_scale)
{
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    if (!q.is_contiguous())          q          = q.contiguous();
    if (!k.is_contiguous())          k          = k.contiguous();
    if (!v.is_contiguous())          v          = v.contiguous();
    if (!qo_indptr.is_contiguous())  qo_indptr  = qo_indptr.contiguous();
    if (!kv_indptr.is_contiguous())  kv_indptr  = kv_indptr.contiguous();

    const int total_q = (int)q.size(0);
    auto f32_opts  = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    auto bf16_opts = torch::TensorOptions().dtype(torch::kBFloat16).device(q.device());
    auto output = torch::empty({total_q, NUM_QO_HEADS, HEAD_DIM}, bf16_opts);
    // No pre-fill needed: every q row belongs to exactly one tile of a
    // non-empty seq, and each tile CTA covers all 32 heads (4 kv_heads x
    // 8 warps) with lane t==0 writing LSE; rows with no visible KV (empty
    // seq / num_kv == 0 -> rsum == 0) get -INFINITY from the epilogue itself.
    auto lse    = torch::empty({total_q, NUM_QO_HEADS}, f32_opts);

    const int batch = (int)qo_indptr.size(0) - 1;
    if (total_q == 0 || batch <= 0) {
        return {output, lse};
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Padded SMEM stride: 2 * T_PIPE * (HEAD_DIM + 8) bf16 elements = 17 KB.
    const size_t smem_size = (size_t)T_PIPE * KV_STRIDE * sizeof(__nv_bfloat16) * 2;
    static bool s_smem_attr_set = false;
    if (!s_smem_attr_set) {
        cudaFuncSetAttribute(gqa_ragged_prefill_mma,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);
        s_smem_attr_set = true;
    }

    // Host-side upper bound on num_tiles = sum over non-empty seqs of
    // ceil(seq_q / BRQ). ceil(s/16) <= 1 + (s-1)/16 for s >= 1, so the sum
    // is <= #non-empty + (total_q - #non-empty)/16 <= batch + total_q/16,
    // in turn <= batch + ceil(total_q/16). Never undershoots, so no tile
    // is dropped; excess CTAs resolve to the dead marker and exit.
    const int num_tiles_ub = batch + (total_q + BRQ - 1) / BRQ;

    dim3 grid(NUM_KV_HEADS, num_tiles_ub);
    dim3 block(BLOCK_THREADS);
    gqa_ragged_prefill_mma<<<grid, block, smem_size, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
        (float)sm_scale,
        qo_indptr.data_ptr<int>(),
        kv_indptr.data_ptr<int>(),
        batch,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        lse.data_ptr<float>());

    return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "GQA Ragged Prefill (causal) mma, single launch: BRQ=16, T_PIPE=32, "
          "mma.sync m16n8k16 bf16 QK^T/PV with tile-granular online softmax, "
          "in-kernel ragged tile resolution from indptrs, base-2 LSE, "
          "explicit -inf causal mask, contiguous ragged KV, GQA 8:1");
}
