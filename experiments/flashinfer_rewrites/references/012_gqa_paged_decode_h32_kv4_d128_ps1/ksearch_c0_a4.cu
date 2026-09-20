/*
 * GQA Paged KV Decode Kernel for task 012 (h=32, kv=4, d=128, page_size=1, bf16).
 *
 * Adapted from the proven 013 campaign best kernel (h=32, kv=8, GQA_RATIO=4) for the
 * 8:1 GQA ratio of task 012 (kv=4 -> GQA_RATIO=8). Block = 256 threads = 8 warps,
 * one warp per query head of this KV head's GQA group. Each warp's 32 lanes cover
 * the 128-dim head (DIMS_PER_LANE=4 bf16 per lane = 8 bytes = one uint2 / LDS.64).
 *
 * Round-5 changes on top of the A1 adaptive split-K base:
 *
 * 1. Two-stage double-buffered cp.async pipeline (D4 tile occupancy / memory
 *    latency). The base tile loop was single-buffered: every T_PIPE tile paid the
 *    full random-gather latency before any compute ran, and K used a synchronous
 *    ld.global.cg round-trip through registers. Now K and V both stream through
 *    cp.async into a 2-deep smem ring: while tile t is being processed by the 8
 *    warps, tile t+1's K/V rows are already in flight. Per-tile cost drops from
 *    (load_latency + compute) to max(load_latency, compute). T_PIPE is raised
 *    16 -> 32 (ROWS_PER_WARP = 4) so each in-flight stage carries 16 KB and the
 *    barrier count per token halves. SMEM = 2 buffers * (K+V) = 32 KB (below the
 *    48 KB default limit, no opt-in needed). Rows past tile_tokens are never read
 *    (process_smem_tile only touches the first tile_tokens rows), so the old
 *    zero-fill of padding rows is dropped.
 *
 * 2. Contiguous chunk assignment per split (A2). The strided pick
 *    idx_base[i * seq_splits] made every index load and every K/V row gather land
 *    seq_splits pages apart; split s now owns the contiguous token range
 *    [s*chunk, (s+1)*chunk) with chunk = ceil(num_tokens / seq_splits), so the
 *    kv_indices reads are perfectly coalesced and coverage stays exact.
 *
 * 3. Parallelism-first split policy. Host picks
 *        SPLIT_CNT = family_round(max(ceil(TARGET_CTAS / (4*batch)),
 *                                      ceil(avg_tokens / 64)))
 *    so the grid fills ~4 waves of the 148 SMs (dominant for small batch, where
 *    base CTAs are scarce) AND the per-(seq, split) token chain stays near 64
 *    tokens = 2 tiles (dominant for large batch, where the longest sequence's
 *    serial chain used to set the makespan under length skew). Workloads below
 *    SPLIT_FLOOR_TOTAL tokens keep the SPLIT_CNT==1 direct path (one launch, no
 *    reduce kernel). The in-kernel refinement floor drops 64 -> 16 tokens so the
 *    extra z-blocks a long sequence earns are actually fed, while short
 *    sequences still early-out their surplus splits cheaply.
 *
 * LSE semantics: reference uses log2-base LSE = logsumexp(logits*scale)/log(2).
 * In the split path the partial kernel writes natural-log LSE (running_max/LOG2E +
 * log(sum)) and the reduce kernel merges and converts to log2 base
 * ((g_max + log(W)) * LOG2E); in the SPLIT_CNT==1 direct path the kernel writes
 * log2 base immediately (running_max is already in log2 units).
 *
 * Target hardware: NVIDIA B300 SXM6 AC (sm_103, 148 SMs).
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <float.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

static constexpr int HEAD_DIM      = 128;
static constexpr int NUM_QO_HEADS  = 32;
static constexpr int NUM_KV_HEADS  = 4;
static constexpr int GQA_RATIO     = 8;            // 32 / 4
static constexpr int BLOCK_THREADS = 256;          // 8 warps, one per Q head
static constexpr int DIMS_PER_LANE = HEAD_DIM / 32; // = 4

// Adaptive split-K family: the template parameter SPLIT_CNT takes these values.
static constexpr int MAX_SPLITS            = 32;
// In-kernel refinement floor: a sequence activates at most
// ceil(num_tokens / MIN_TOKENS_PER_SPLIT) splits, so surplus z-blocks early out.
static constexpr int MIN_TOKENS_PER_SPLIT  = 16;
// Host-side chain-length target: ~64 tokens (2 T_PIPE tiles) per (seq, split).
static constexpr int TARGET_CHAIN_TOKENS   = 64;
// Host-side grid-fill target: ~4 waves of the 148 B300 SMs.
static constexpr int NUM_SMS               = 148;  // B300 SXM6 AC
static constexpr int TARGET_TOTAL_CTAS     = NUM_SMS * 4;
// Below this total token count the launch floor dominates: single launch,
// direct write, no reduce kernel.
static constexpr int SPLIT_FLOOR_TOTAL     = 96;

// Pipeline tile size: 32 tokens per tile. Must be >= GQA_RATIO and divisible by
// it so ROWS_PER_WARP is integral (32 lanes of a warp cooperatively load one
// 256 B row; 8 warps * 4 rows = 32 rows). Double-buffered K+V tiles:
// SMEM = 2 * 2 * T_PIPE * HEAD_DIM * sizeof(bf16) = 32 KB.
static constexpr int T_PIPE = 32;
// Rows per warp in cooperative tile loading: T_PIPE / 8 warps = 32 / 8 = 4.
static constexpr int ROWS_PER_WARP = T_PIPE / GQA_RATIO;  // = 4

static constexpr float LOG2E = 1.4426950408889634f;

// Vectorized load of 4 bf16 values from global memory via __ldg (L1/texture cache).
// 8-byte (uint2) transaction.
__device__ __forceinline__ void load4_bf16_ldg(
    const __nv_bfloat16* ptr,
    float* r0, float* r1, float* r2, float* r3)
{
    uint2 v = __ldg(reinterpret_cast<const uint2*>(ptr));
    *r0 = __bfloat162float(__nv_bfloat16_raw{(unsigned short)(v.x & 0xFFFF)});
    *r1 = __bfloat162float(__nv_bfloat16_raw{(unsigned short)(v.x >> 16)});
    *r2 = __bfloat162float(__nv_bfloat16_raw{(unsigned short)(v.y & 0xFFFF)});
    *r3 = __bfloat162float(__nv_bfloat16_raw{(unsigned short)(v.y >> 16)});
}

// Vectorized load of 4 bf16 values from SMEM using uint2 (LDS.64).
__device__ __forceinline__ void load4_bf16_smem_u2(
    const __nv_bfloat16* __restrict__ ptr,
    float* r0, float* r1, float* r2, float* r3)
{
    uint2 raw = *reinterpret_cast<const uint2*>(ptr);
    *r0 = __bfloat162float(__nv_bfloat16_raw{(unsigned short)(raw.x & 0xFFFF)});
    *r1 = __bfloat162float(__nv_bfloat16_raw{(unsigned short)(raw.x >> 16)});
    *r2 = __bfloat162float(__nv_bfloat16_raw{(unsigned short)(raw.y & 0xFFFF)});
    *r3 = __bfloat162float(__nv_bfloat16_raw{(unsigned short)(raw.y >> 16)});
}

__device__ __forceinline__ float warp_reduce_sum(float val)
{
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return __shfl_sync(0xffffffff, val, 0);
}

// Process a 4-token sub-group from smem: Q@K, online softmax update, P@V accumulate.
__device__ __forceinline__ void process_4tok(
    const __nv_bfloat16* __restrict__ smem_k_buf,
    const __nv_bfloat16* __restrict__ smem_v_buf,
    int tok_base,
    const float* q_reg,
    float* acc,
    float& running_max,
    float& running_sum,
    float sm_scale_log2,
    int lane_id)
{
    const int lane_off = lane_id * DIMS_PER_LANE;

    float k0[DIMS_PER_LANE], k1[DIMS_PER_LANE], k2[DIMS_PER_LANE], k3[DIMS_PER_LANE];
    load4_bf16_smem_u2(smem_k_buf + (tok_base+0)*HEAD_DIM + lane_off, &k0[0], &k0[1], &k0[2], &k0[3]);
    load4_bf16_smem_u2(smem_k_buf + (tok_base+1)*HEAD_DIM + lane_off, &k1[0], &k1[1], &k1[2], &k1[3]);
    load4_bf16_smem_u2(smem_k_buf + (tok_base+2)*HEAD_DIM + lane_off, &k2[0], &k2[1], &k2[2], &k2[3]);
    load4_bf16_smem_u2(smem_k_buf + (tok_base+3)*HEAD_DIM + lane_off, &k3[0], &k3[1], &k3[2], &k3[3]);

    float dot0=0.f, dot1=0.f, dot2=0.f, dot3=0.f;
    #pragma unroll
    for (int d = 0; d < DIMS_PER_LANE; d++) {
        dot0 += q_reg[d] * k0[d]; dot1 += q_reg[d] * k1[d];
        dot2 += q_reg[d] * k2[d]; dot3 += q_reg[d] * k3[d];
    }
    dot0 = warp_reduce_sum(dot0); dot1 = warp_reduce_sum(dot1);
    dot2 = warp_reduce_sum(dot2); dot3 = warp_reduce_sum(dot3);

    float logit0 = dot0 * sm_scale_log2, logit1 = dot1 * sm_scale_log2;
    float logit2 = dot2 * sm_scale_log2, logit3 = dot3 * sm_scale_log2;

    float v0[DIMS_PER_LANE], v1[DIMS_PER_LANE], v2[DIMS_PER_LANE], v3[DIMS_PER_LANE];
    load4_bf16_smem_u2(smem_v_buf + (tok_base+0)*HEAD_DIM + lane_off, &v0[0], &v0[1], &v0[2], &v0[3]);
    load4_bf16_smem_u2(smem_v_buf + (tok_base+1)*HEAD_DIM + lane_off, &v1[0], &v1[1], &v1[2], &v1[3]);
    load4_bf16_smem_u2(smem_v_buf + (tok_base+2)*HEAD_DIM + lane_off, &v2[0], &v2[1], &v2[2], &v2[3]);
    load4_bf16_smem_u2(smem_v_buf + (tok_base+3)*HEAD_DIM + lane_off, &v3[0], &v3[1], &v3[2], &v3[3]);

    float new_max = fmaxf(running_max, fmaxf(fmaxf(logit0, logit1), fmaxf(logit2, logit3)));
    float exp_prev = exp2f(running_max - new_max);
    float e0 = exp2f(logit0 - new_max), e1 = exp2f(logit1 - new_max);
    float e2 = exp2f(logit2 - new_max), e3 = exp2f(logit3 - new_max);
    running_sum = running_sum * exp_prev + e0 + e1 + e2 + e3;
    running_max = new_max;

    #pragma unroll
    for (int d = 0; d < DIMS_PER_LANE; d++) {
        acc[d] = acc[d] * exp_prev + e0 * v0[d] + e1 * v1[d] + e2 * v2[d] + e3 * v3[d];
    }
}

__device__ __forceinline__ void process_2tok(
    const __nv_bfloat16* __restrict__ smem_k_buf,
    const __nv_bfloat16* __restrict__ smem_v_buf,
    int tok_base,
    const float* q_reg,
    float* acc,
    float& running_max,
    float& running_sum,
    float sm_scale_log2,
    int lane_id)
{
    const int lane_off = lane_id * DIMS_PER_LANE;

    float k0[DIMS_PER_LANE], k1[DIMS_PER_LANE];
    load4_bf16_smem_u2(smem_k_buf + (tok_base+0)*HEAD_DIM + lane_off, &k0[0], &k0[1], &k0[2], &k0[3]);
    load4_bf16_smem_u2(smem_k_buf + (tok_base+1)*HEAD_DIM + lane_off, &k1[0], &k1[1], &k1[2], &k1[3]);
    float dot0 = 0.f, dot1 = 0.f;
    #pragma unroll
    for (int d = 0; d < DIMS_PER_LANE; d++) {
        dot0 += q_reg[d] * k0[d]; dot1 += q_reg[d] * k1[d];
    }
    dot0 = warp_reduce_sum(dot0); dot1 = warp_reduce_sum(dot1);
    float logit0 = dot0 * sm_scale_log2, logit1 = dot1 * sm_scale_log2;

    float v0[DIMS_PER_LANE], v1[DIMS_PER_LANE];
    load4_bf16_smem_u2(smem_v_buf + (tok_base+0)*HEAD_DIM + lane_off, &v0[0], &v0[1], &v0[2], &v0[3]);
    load4_bf16_smem_u2(smem_v_buf + (tok_base+1)*HEAD_DIM + lane_off, &v1[0], &v1[1], &v1[2], &v1[3]);
    float new_max = fmaxf(running_max, fmaxf(logit0, logit1));
    float exp_prev = exp2f(running_max - new_max);
    float e0 = exp2f(logit0 - new_max), e1 = exp2f(logit1 - new_max);
    running_sum = running_sum * exp_prev + e0 + e1;
    running_max = new_max;
    #pragma unroll
    for (int d = 0; d < DIMS_PER_LANE; d++) {
        acc[d] = acc[d] * exp_prev + e0 * v0[d] + e1 * v1[d];
    }
}

__device__ __forceinline__ void process_1tok(
    const __nv_bfloat16* __restrict__ smem_k_buf,
    const __nv_bfloat16* __restrict__ smem_v_buf,
    int tok_base,
    const float* q_reg,
    float* acc,
    float& running_max,
    float& running_sum,
    float sm_scale_log2,
    int lane_id)
{
    const int lane_off = lane_id * DIMS_PER_LANE;

    float k_reg[DIMS_PER_LANE];
    load4_bf16_smem_u2(smem_k_buf + tok_base*HEAD_DIM + lane_off, &k_reg[0], &k_reg[1], &k_reg[2], &k_reg[3]);
    float dot = 0.f;
    #pragma unroll
    for (int d = 0; d < DIMS_PER_LANE; d++) dot += q_reg[d] * k_reg[d];
    dot = warp_reduce_sum(dot);
    float logit = dot * sm_scale_log2;

    float v_reg[DIMS_PER_LANE];
    load4_bf16_smem_u2(smem_v_buf + tok_base*HEAD_DIM + lane_off, &v_reg[0], &v_reg[1], &v_reg[2], &v_reg[3]);
    float new_max  = fmaxf(running_max, logit);
    float exp_prev = exp2f(running_max - new_max);
    float exp_cur  = exp2f(logit - new_max);
    running_sum = running_sum * exp_prev + exp_cur;
    running_max = new_max;
    #pragma unroll
    for (int d = 0; d < DIMS_PER_LANE; d++) {
        acc[d] = acc[d] * exp_prev + exp_cur * v_reg[d];
    }
}

__device__ __forceinline__ void process_smem_tile(
    const __nv_bfloat16* __restrict__ smem_k_buf,
    const __nv_bfloat16* __restrict__ smem_v_buf,
    int tile_tokens,
    const float* q_reg,
    float* acc,
    float& running_max,
    float& running_sum,
    float sm_scale_log2,
    int lane_id)
{
    int n8 = tile_tokens >> 3;
    int n4 = (tile_tokens >> 2) & 1;
    int n2 = (tile_tokens >> 1) & 1;
    int n1 = tile_tokens & 1;
    int tok = 0;

    for (int gi = 0; gi < n8; gi++, tok += 8) {
        process_4tok(smem_k_buf, smem_v_buf, tok,
                     q_reg, acc, running_max, running_sum, sm_scale_log2, lane_id);
        process_4tok(smem_k_buf, smem_v_buf, tok + 4,
                     q_reg, acc, running_max, running_sum, sm_scale_log2, lane_id);
    }

    if (n4) {
        process_4tok(smem_k_buf, smem_v_buf, tok,
                     q_reg, acc, running_max, running_sum, sm_scale_log2, lane_id);
        tok += 4;
    }

    if (n2) {
        process_2tok(smem_k_buf, smem_v_buf, tok,
                     q_reg, acc, running_max, running_sum, sm_scale_log2, lane_id);
        tok += 2;
    }

    if (n1) {
        process_1tok(smem_k_buf, smem_v_buf, tok,
                     q_reg, acc, running_max, running_sum, sm_scale_log2, lane_id);
    }
}

/*
 * Issue one tile's K and V rows into a smem buffer pair via cp.async (8 B per
 * lane = one 256 B row per warp load). Called for tile t+1 while tile t is still
 * being processed, so the random-page gather latency overlaps with compute.
 * Rows past tile_tokens are never issued and never read back.
 */
__device__ __forceinline__ void issue_tile_async(
    __nv_bfloat16*                   smem_k,
    __nv_bfloat16*                   smem_v,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int*           __restrict__ idx_base,
    int  tile_start,     // local token index of the tile's first row
    int  tile_tokens,    // valid rows in this tile
    int  kv_head_offset,
    int  kv_stride,
    int  warp_id,
    int  lane_id)
{
    const int row_base  = warp_id * ROWS_PER_WARP;
    const int lane_base = lane_id * DIMS_PER_LANE;

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        const int row = row_base + r;
        if (row < tile_tokens) {
            // Contiguous chunk assignment (A2): consecutive rows read consecutive
            // kv_indices entries (coalesced, warp-uniform broadcast).
            const int pg   = __ldg(idx_base + tile_start + row);
            const int goff = pg * kv_stride + kv_head_offset + lane_base;
            __pipeline_memcpy_async(smem_k + row * HEAD_DIM + lane_base,
                                    k_cache + goff,
                                    sizeof(__nv_bfloat16) * DIMS_PER_LANE);
            __pipeline_memcpy_async(smem_v + row * HEAD_DIM + lane_base,
                                    v_cache + goff,
                                    sizeof(__nv_bfloat16) * DIMS_PER_LANE);
        }
    }
}

/*
 * Main split-K decode kernel (template on the host-chosen adaptive split count).
 * grid = (NUM_KV_HEADS, batch, SPLIT_CNT), block = 256 threads (8 warps).
 * warp_id in [0,8): one warp per query head of this KV head's GQA group.
 *
 * out_dst/lse_dst are the partial buffers when SPLIT_CNT > 1; when SPLIT_CNT == 1
 * they are the final output/lse tensors (the SPLIT_CNT=1 partial layout is exactly
 * the output layout), and the kernel writes normalized output + log2-base LSE
 * directly so the reduce launch is skipped.
 */
template <int SPLIT_CNT>
__global__ void __launch_bounds__(BLOCK_THREADS, 1)
gqa_paged_decode_splitk(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int*           __restrict__ kv_indptr,
    const int*           __restrict__ kv_indices,
    float                             sm_scale,
    __nv_bfloat16*       __restrict__ out_dst,
    float*               __restrict__ lse_dst,
    int                               batch_size)
{
    extern __shared__ __nv_bfloat16 smem_raw[];
    // Two-stage K/V ring: [buf0 K | buf0 V | buf1 K | buf1 V], each stage
    // T_PIPE*HEAD_DIM bf16. Tile t lives in stage (t & 1).
    const int BUF_ELEMS = T_PIPE * HEAD_DIM;
    __nv_bfloat16* smem_k[2] = { smem_raw,             smem_raw + 2 * BUF_ELEMS };
    __nv_bfloat16* smem_v[2] = { smem_raw + BUF_ELEMS, smem_raw + 3 * BUF_ELEMS };

    const int kv_head = blockIdx.x;
    const int b       = blockIdx.y;
    const int split   = blockIdx.z;
    const int tid     = threadIdx.x;

    const int warp_id   = tid >> 5;
    const int lane_id   = tid & 31;
    const int lane_base = lane_id * DIMS_PER_LANE;

    const int page_start = __ldg(kv_indptr + b);
    const int page_end   = __ldg(kv_indptr + b + 1);
    const int num_tokens = page_end - page_start;

    // Per-sequence refinement of the host-chosen split count: activate only as many
    // splits as this sequence can feed with MIN_TOKENS_PER_SPLIT tokens each (still
    // bounded by the compile-time SPLIT_CNT). Splits above the refined count
    // early-out below instead of running near-empty bodies.
    const int tok_cap = (num_tokens + MIN_TOKENS_PER_SPLIT - 1) / MIN_TOKENS_PER_SPLIT;
    int seq_splits = (tok_cap < SPLIT_CNT) ? tok_cap : SPLIT_CNT;
    if (seq_splits < 1) seq_splits = 1;

    // Contiguous chunk assignment (A2): split s owns tokens [s*chunk, (s+1)*chunk)
    // of this sequence, chunk = ceil(num_tokens / seq_splits). The seq_splits
    // active splits exactly cover [0, num_tokens); an interior split can be empty
    // when the ceil rounding overshoots (handled by the local_count == 0 path).
    const int chunk = (num_tokens + seq_splits - 1) / seq_splits;
    const int tok0  = split * chunk;

    int local_count = 0;
    if (split < seq_splits && tok0 < num_tokens) {
        const int rem = num_tokens - tok0;
        local_count = (chunk < rem) ? chunk : rem;
    }

    const int qh_global = kv_head * GQA_RATIO + warp_id;
    const int lse_idx = (b * NUM_QO_HEADS + qh_global) * SPLIT_CNT + split;

    if (local_count == 0 || split >= seq_splits) {
        if (SPLIT_CNT == 1) {
            // Direct (no-reduce) path: empty sequence -> zero output, -inf LSE.
            #pragma unroll
            for (int d = 0; d < DIMS_PER_LANE; d++) {
                out_dst[(b * NUM_QO_HEADS + qh_global) * HEAD_DIM + lane_base + d]
                    = __float2bfloat16(0.f);
            }
            if (lane_id == 0) {
                lse_dst[b * NUM_QO_HEADS + qh_global] = -__builtin_huge_valf();
            }
        } else if (lane_id == 0) {
            lse_dst[lse_idx] = -FLT_MAX;
        }
        return;
    }

    const __nv_bfloat16* q_row = q + (b * NUM_QO_HEADS + qh_global) * HEAD_DIM;
    float q_reg[DIMS_PER_LANE];
    load4_bf16_ldg(q_row + lane_base, &q_reg[0], &q_reg[1], &q_reg[2], &q_reg[3]);

    float acc[DIMS_PER_LANE] = {0.f, 0.f, 0.f, 0.f};
    float running_max = -FLT_MAX;
    float running_sum = 0.0f;

    const float sm_scale_log2 = sm_scale * LOG2E;
    const int kv_head_offset = kv_head * HEAD_DIM;
    const int kv_stride = NUM_KV_HEADS * HEAD_DIM;

    // Contiguous chunk: row i of this split is kv_indices[page_start + tok0 + i].
    const int* idx_base = kv_indices + page_start + tok0;

    const int num_tiles = (local_count + T_PIPE - 1) / T_PIPE;

    // Pipeline prologue: start tile 0 into stage 0.
    {
        const int tt0 = (T_PIPE < local_count) ? T_PIPE : local_count;
        issue_tile_async(smem_k[0], smem_v[0], k_cache, v_cache, idx_base,
                         0, tt0, kv_head_offset, kv_stride, warp_id, lane_id);
    }
    __pipeline_commit();

    for (int t = 0; t < num_tiles; t++) {
        const int ts = t * T_PIPE;
        const int tt = (T_PIPE < local_count - ts) ? T_PIPE : (local_count - ts);

        if (t + 1 < num_tiles) {
            // Issue tile t+1 into stage (t+1)&1 (which held tile t-1; the trailing
            // __syncthreads of iteration t-1 proves all reads of it are done).
            // Its kv_indices __ldg latency overlaps tile t's in-flight loads.
            const int ts1 = (t + 1) * T_PIPE;
            const int tt1 = (T_PIPE < local_count - ts1) ? T_PIPE : (local_count - ts1);
            const int nxt = (t + 1) & 1;
            issue_tile_async(smem_k[nxt], smem_v[nxt], k_cache, v_cache, idx_base,
                             ts1, tt1, kv_head_offset, kv_stride, warp_id, lane_id);
            __pipeline_commit();
            // Keep tile t+1's group in flight; wait only for tile t's group.
            __pipeline_wait_prior(1);
        } else {
            // Last tile: no new group committed, so drain everything.
            __pipeline_wait_prior(0);
        }
        __syncthreads();

        const int cur = t & 1;
        process_smem_tile(smem_k[cur], smem_v[cur], tt,
                          q_reg, acc, running_max, running_sum,
                          sm_scale_log2, lane_id);

        // Barrier before this stage is overwritten by tile t+2's issue.
        __syncthreads();
    }

    // Write partial output (per-split, unnormalized; reduce kernel normalizes).
    // For SPLIT_CNT == 1 out_dst is the final output and acc * inv_sum is already
    // the normalized softmax @ V, so no reduce pass is needed.
    float inv_sum = (running_sum > 0.0f) ? 1.0f / running_sum : 0.0f;

    const int out_base = (b * NUM_QO_HEADS + qh_global) * (SPLIT_CNT * HEAD_DIM)
                         + split * HEAD_DIM + lane_base;
    __nv_bfloat16* out_ptr = out_dst + out_base;
    #pragma unroll
    for (int d = 0; d < DIMS_PER_LANE; d++) {
        out_ptr[d] = __float2bfloat16(acc[d] * inv_sum);
    }

    if (lane_id == 0) {
        if (SPLIT_CNT == 1) {
            // Direct path: log2-base LSE (running_max is already in log2 units).
            lse_dst[b * NUM_QO_HEADS + qh_global] =
                (running_sum > 0.f && isfinite(running_sum))
                ? (running_max + __logf(running_sum) * LOG2E)
                : -__builtin_huge_valf();
        } else {
            float lse_val = (running_sum > 0.f && isfinite(running_sum))
                ? (running_max / LOG2E + __logf(running_sum))
                : -FLT_MAX;
            lse_dst[lse_idx] = lse_val;
        }
    }
}

// Reduce kernel: merge SPLIT_CNT partial outputs + LSE into final output (log2-base
// LSE). Templated on the same SPLIT_CNT as the main kernel so the merge loop is
// compile-time-known and always matches the partial layout. Only launched for
// SPLIT_CNT > 1.
template <int SPLIT_CNT>
__global__ void __launch_bounds__(HEAD_DIM, 8)
gqa_splitk_reduce(
    const __nv_bfloat16* __restrict__ out_partial,
    const float*         __restrict__ lse_partial,
    __nv_bfloat16*       __restrict__ output,
    float*               __restrict__ lse_out)
{
    const int b   = blockIdx.x;
    const int qh  = blockIdx.y;
    const int tid = threadIdx.x;

    const float* lse_base = lse_partial + (b * NUM_QO_HEADS + qh) * SPLIT_CNT;

    float lse_vals[SPLIT_CNT];
    float g_max = -FLT_MAX;

    #pragma unroll
    for (int s = 0; s < SPLIT_CNT; s++) {
        float lse_s = __ldg(lse_base + s);
        lse_vals[s] = lse_s;
        if (lse_s > g_max) g_max = lse_s;
    }

    if (g_max <= -FLT_MAX * 0.5f) {
        output[(b * NUM_QO_HEADS + qh) * HEAD_DIM + tid] = __float2bfloat16(0.f);
        if (tid == 0) lse_out[b * NUM_QO_HEADS + qh] = -__builtin_huge_valf();
        return;
    }

    const __nv_bfloat16* partial_base =
        out_partial + (b * NUM_QO_HEADS + qh) * (SPLIT_CNT * HEAD_DIM);

    float out_acc = 0.0f;
    float W       = 0.0f;

    #pragma unroll
    for (int s = 0; s < SPLIT_CNT; s++) {
        if (lse_vals[s] <= -FLT_MAX * 0.5f) continue;
        float w = __expf(lse_vals[s] - g_max);
        W += w;
        out_acc += w * __bfloat162float(__ldg(partial_base + s * HEAD_DIM + tid));
    }

    float inv_W = (W > 0.0f) ? 1.0f / W : 0.0f;
    output[(b * NUM_QO_HEADS + qh) * HEAD_DIM + tid] = __float2bfloat16(out_acc * inv_W);

    if (tid == 0) {
        lse_out[b * NUM_QO_HEADS + qh] = (g_max + __logf(W)) * LOG2E;
    }
}

/*
 * Host-side adaptive split-count selection.
 *   num_kv_indices is known from the kv_indices tensor shape (== kv_indptr[batch]),
 *   so no D2H sync on indptr is needed.
 *   - launch-floor regime (total < SPLIT_FLOOR_TOTAL): SPLIT_CNT = 1, direct write,
 *     single launch — these workloads are dominated by launch/dispatch latency;
 *   - fill-the-machine term: ceil(TARGET_TOTAL_CTAS / (NUM_KV_HEADS * batch)) so
 *     small batches still spread over ~4 waves of the 148 SMs (batch=1 reaches the
 *     32-split cap => 128 CTAs instead of 4);
 *   - chain-length term: ceil(avg_tokens / TARGET_CHAIN_TOKENS) so every
 *     (seq, split) chain is ~2 T_PIPE tiles; this bounds the makespan under
 *     sequence-length skew, where the longest chain used to dominate;
 *   - result rounded down into the template family {1,2,4,8,16,32}.
 */
static int pick_num_splits(int batch_size, int total_tokens)
{
    if (batch_size < 1) batch_size = 1;
    if (total_tokens < SPLIT_FLOOR_TOTAL) return 1;

    const int avg_tokens = (total_tokens + batch_size - 1) / batch_size;

    const int s_parallel = TARGET_TOTAL_CTAS / (NUM_KV_HEADS * batch_size);
    const int s_chain    = (avg_tokens + TARGET_CHAIN_TOKENS - 1) / TARGET_CHAIN_TOKENS;

    int s = (s_parallel > s_chain) ? s_parallel : s_chain;
    if (s > MAX_SPLITS) s = MAX_SPLITS;
    if (s < 1) s = 1;
    if (s >= 32) return 32;
    if (s >= 16) return 16;
    if (s >= 8)  return 8;
    if (s >= 4)  return 4;
    if (s >= 2)  return 2;
    return 1;
}

// Launch the SPLIT_CNT-instantiated main kernel (+ reduce kernel iff SPLIT_CNT > 1).
template <int SPLIT_CNT>
static void run_splitk(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache,
    const int*           kv_indptr,
    const int*           kv_indices,
    float                sm_scale,
    __nv_bfloat16*       out_dst,
    float*               lse_dst,
    __nv_bfloat16*       out_partial,
    float*               lse_partial,
    __nv_bfloat16*       output,
    float*               lse,
    int                  batch_size,
    cudaStream_t         stream)
{
    // 2-stage K/V ring: 2 buffers * (K tile + V tile) = 32 KB.
    const size_t smem_size = T_PIPE * HEAD_DIM * sizeof(__nv_bfloat16) * 4;

    // 32 KB is below the 48 KB default limit; set the attribute only once per
    // instantiation to keep the tiny-workload launch path lean.
    static bool smem_attr_done = false;
    if (!smem_attr_done) {
        cudaFuncSetAttribute(gqa_paged_decode_splitk<SPLIT_CNT>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);
        smem_attr_done = true;
    }

    dim3 main_grid(NUM_KV_HEADS, batch_size, SPLIT_CNT);
    dim3 main_block(BLOCK_THREADS);

    gqa_paged_decode_splitk<SPLIT_CNT><<<main_grid, main_block, smem_size, stream>>>(
        q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale,
        out_dst, lse_dst, batch_size);

    if (SPLIT_CNT > 1) {
        dim3 red_grid(batch_size, NUM_QO_HEADS);
        dim3 red_block(HEAD_DIM);

        gqa_splitk_reduce<SPLIT_CNT><<<red_grid, red_block, 0, stream>>>(
            out_partial, lse_partial, output, lse);
    }
}

std::vector<torch::Tensor> forward(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor kv_indptr,
    torch::Tensor kv_indices,
    float         sm_scale)
{
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    if (!q.is_contiguous())          q          = q.contiguous();
    if (!k_cache.is_contiguous())    k_cache    = k_cache.contiguous();
    if (!v_cache.is_contiguous())    v_cache    = v_cache.contiguous();
    if (!kv_indptr.is_contiguous())  kv_indptr  = kv_indptr.contiguous();
    if (!kv_indices.is_contiguous()) kv_indices = kv_indices.contiguous();

    const int batch_size   = q.size(0);
    const int total_tokens = (int)kv_indices.size(0);  // == kv_indptr[batch_size]

    auto f32_opts  = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    auto bf16_opts = torch::TensorOptions().dtype(torch::kBFloat16).device(q.device());

    auto output = torch::empty({batch_size, NUM_QO_HEADS, HEAD_DIM}, bf16_opts);
    auto lse    = torch::empty({batch_size, NUM_QO_HEADS}, f32_opts);

    // Adaptive split count: host-side, from tensor-shape data only.
    const int num_splits = pick_num_splits(batch_size, total_tokens);

    __nv_bfloat16* out_dst;
    float*         lse_dst;
    torch::Tensor  out_partial, lse_partial;

    if (num_splits > 1) {
        out_partial = torch::empty({batch_size, NUM_QO_HEADS, num_splits, HEAD_DIM}, bf16_opts);
        lse_partial = torch::empty({batch_size, NUM_QO_HEADS, num_splits}, f32_opts);
        out_dst = reinterpret_cast<__nv_bfloat16*>(out_partial.data_ptr<at::BFloat16>());
        lse_dst = lse_partial.data_ptr<float>();
    } else {
        // Single-split fast path: the main kernel writes the final output and
        // log2-base LSE directly; the reduce kernel is not launched at all.
        out_dst = reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());
        lse_dst = lse.data_ptr<float>();
    }

    const __nv_bfloat16* q_ptr = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>());
    const __nv_bfloat16* k_ptr = reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr<at::BFloat16>());
    const __nv_bfloat16* v_ptr = reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr<at::BFloat16>());
    const int* indptr_ptr  = kv_indptr.data_ptr<int>();
    const int* indices_ptr = kv_indices.data_ptr<int>();
    __nv_bfloat16* out_partial_ptr = (num_splits > 1)
        ? reinterpret_cast<__nv_bfloat16*>(out_partial.data_ptr<at::BFloat16>()) : nullptr;
    float* lse_partial_ptr = (num_splits > 1) ? lse_partial.data_ptr<float>() : nullptr;
    __nv_bfloat16* out_ptr = reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());
    float* lse_ptr = lse.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    switch (num_splits) {
        case 1:
            run_splitk<1>(q_ptr, k_ptr, v_ptr, indptr_ptr, indices_ptr, sm_scale,
                          out_dst, lse_dst, out_partial_ptr, lse_partial_ptr,
                          out_ptr, lse_ptr, batch_size, stream);
            break;
        case 2:
            run_splitk<2>(q_ptr, k_ptr, v_ptr, indptr_ptr, indices_ptr, sm_scale,
                          out_dst, lse_dst, out_partial_ptr, lse_partial_ptr,
                          out_ptr, lse_ptr, batch_size, stream);
            break;
        case 4:
            run_splitk<4>(q_ptr, k_ptr, v_ptr, indptr_ptr, indices_ptr, sm_scale,
                          out_dst, lse_dst, out_partial_ptr, lse_partial_ptr,
                          out_ptr, lse_ptr, batch_size, stream);
            break;
        case 8:
            run_splitk<8>(q_ptr, k_ptr, v_ptr, indptr_ptr, indices_ptr, sm_scale,
                          out_dst, lse_dst, out_partial_ptr, lse_partial_ptr,
                          out_ptr, lse_ptr, batch_size, stream);
            break;
        case 16:
            run_splitk<16>(q_ptr, k_ptr, v_ptr, indptr_ptr, indices_ptr, sm_scale,
                           out_dst, lse_dst, out_partial_ptr, lse_partial_ptr,
                           out_ptr, lse_ptr, batch_size, stream);
            break;
        case 32:
            run_splitk<32>(q_ptr, k_ptr, v_ptr, indptr_ptr, indices_ptr, sm_scale,
                           out_dst, lse_dst, out_partial_ptr, lse_partial_ptr,
                           out_ptr, lse_ptr, batch_size, stream);
            break;
        default:
            TORCH_CHECK(false, "pick_num_splits returned a value outside the "
                               "{1,2,4,8,16,32} template family");
    }

    return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward,
          "GQA Paged KV Decode (task 012): adaptive split-K, contiguous chunks, "
          "2-stage double-buffered cp.async K/V pipeline, warp-per-qhead, "
          "SPLIT_CNT in {1,2,4,8,16,32} picked on host");
    m.def("run", &forward,
          "GQA Paged KV Decode (task 012), adaptive split-K entry point");
}