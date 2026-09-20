/*
 * MLA Paged Decode Attention -- V21 (attempt 0 of this activation)
 * Target: RTX 4090 (sm_89, Ada Lovelace), CUDA 12+
 *
 * V21 CHANGES vs V20 (1.040x):
 *
 * 1. ELIMINATED FILL KERNEL LAUNCHES (torch::zeros -> torch::empty,
 *    torch::full -> torch::empty):
 *    Every dispatch path writes the ENTIRE output and lse grids:
 *      - smalll grid is (batch, 16) and its L_tokens<=0 early-return writes
 *        zeros to output and -inf to lse for those (b,h);
 *      - splitk pass2 grid is (batch, 16) with the identical early-return.
 *    Therefore the pre-fill kernels launched by torch::zeros/torch::full were
 *    pure overhead: two extra kernel launches (plus their memsets) per call
 *    on the critical path of a ~0.1ms operation. Replaced with torch::empty
 *    (allocation only, no fill). The batch==0 early-exit path keeps
 *    zeros/full since no kernel runs there.
 *
 * 2. V20 lineage retained verbatim otherwise:
 *    - ld.global.ca CKV (v2.u32) + KPE (u16) gathers for inter-head L1/L2 KV
 *      reuse (16 q-heads share 1 KV head -> ~16x effective traffic cut).
 *    - 8-token unrolled hot loop, batched online softmax (9 exp2f / 8 tokens).
 *    - Double-buffered wpart smem: 1 __syncthreads per 8 tokens.
 *    - Small-L direct-register path (avg_L <= 32) with double-buffered warp_part.
 *    - Host dispatch on avg_L: smalll (<=32), split-K k=4 (32..63),
 *      split-K k=16 (>=64) / k=32 (>=2048).
 *
 * Lineage: V8 -> V11 (8-token unroll) -> V13 (double-buffered wpart) ->
 *          V20 (ld.global.ca + batched softmax) -> V21 (no fill launches)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// ============================================================
// Compile-time constants
// ============================================================
static constexpr int NUM_HEADS             = 16;
static constexpr int DIM_CKV              = 512;
static constexpr int DIM_KPE              = 64;
static constexpr int BLOCK_THREADS        = 128;
static constexpr int NUM_WARPS            = 4;
static constexpr float LOG2E_F            = 1.4426950408889634f;
static constexpr int ELEMS_PER_THREAD_CKV = DIM_CKV / BLOCK_THREADS;  // 4

// wpart: 8 token slots x NUM_WARPS = 32 floats = 128 bytes per buffer.
// Double-buffered (2 x 128 = 256 bytes) so consecutive 8-token iterations
// write to alternating buffers -> only 1 __syncthreads per 8 tokens.
static constexpr int WPART_8TOK    = 8 * NUM_WARPS;          // 32 floats
static constexpr int WPART_BUF     = WPART_8TOK;             // one buffer = 32 floats
static constexpr int SMEM_PASS1    = 2 * WPART_BUF * (int)sizeof(float);  // 256 bytes

// Split-K workspace: [m, d, acc[512]] per (b, h, k)
static constexpr int K_MAX_CHUNKS        = 16;
static constexpr int K_MAX_CHUNKS_LONG   = 32;
static constexpr int WORKSPACE_SLOT_SIZE = 2 + DIM_CKV;

// Dispatch thresholds
static constexpr int SMALLL_THRESH  = 32;
static constexpr int SPLIT_L_THRESH = 64;
static constexpr int LONG_L_THRESH  = 2048;

// small-L SMEM
static constexpr int SMEM_SMALLL   = 2 * NUM_WARPS * (int)sizeof(float);   // 32 bytes

// ============================================================
// Warp-level sum reduction
// ============================================================
__device__ __forceinline__
float warp_reduce_sum_f32(float v) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, mask);
    return v;
}

// ============================================================
// CKV load macro: ld.global.ca into registers (L1+L2 cached for
// inter-head KV page reuse: the 16 (b,h,*) blocks of a fixed b read
// the same KV pages).
// ============================================================
#define LOAD_CKV_CS(pi, a0, a1, a2, a3, t_id)                                    \
{                                                                                  \
    const __nv_bfloat16* _src = ckv_cache + (int64_t)(pi) * DIM_CKV             \
                                           + (t_id) * ELEMS_PER_THREAD_CKV;      \
    uint2 _v;                                                                     \
    asm volatile("ld.global.ca.v2.u32 {%0,%1},[%2];"                            \
                 :"=r"(_v.x),"=r"(_v.y):"l"(_src));                             \
    const __nv_bfloat16* _r = reinterpret_cast<const __nv_bfloat16*>(&_v);     \
    (a0) = __bfloat162float(_r[0]); (a1) = __bfloat162float(_r[1]);            \
    (a2) = __bfloat162float(_r[2]); (a3) = __bfloat162float(_r[3]);            \
}

// ============================================================
// KPE load macro: ld.global.ca.u16 (L1+L2 cached, only for t < DIM_KPE)
// ============================================================
#define LOAD_KPE_CS(pi, p, t_id)                                                  \
{                                                                                  \
    if ((t_id) < DIM_KPE) {                                                       \
        unsigned short _v;                                                        \
        asm volatile("ld.global.ca.u16 %0,[%1];"                                \
                     :"=h"(_v):"l"(kpe_cache+(int64_t)(pi)*DIM_KPE+(t_id)));   \
        (p) = __bfloat162float(__nv_bfloat16_raw{_v});                          \
    } else { (p) = 0.f; }                                                         \
}

// ============================================================
// Online softmax update macro (2-base, in-line)
// ============================================================
#define SOFTMAX_UPDATE(l, kc0, kc1, kc2, kc3)                                \
{                                                                              \
    const float _nm = fmaxf(m, (l));                                          \
    const float _al = exp2f(m - _nm);                                         \
    const float _ev = exp2f((l) - _nm);                                       \
    d    = d   * _al + _ev;                                                   \
    m    = _nm;                                                                \
    acc0 = acc0 * _al + _ev * (kc0);                                          \
    acc1 = acc1 * _al + _ev * (kc1);                                          \
    acc2 = acc2 * _al + _ev * (kc2);                                          \
    acc3 = acc3 * _al + _ev * (kc3);                                          \
}

// ============================================================
// Small-L fast-path kernel
// Grid: (batch_size, num_heads)
// Block: 128 threads, 32 bytes SMEM
// V20: direct register loads (no CKV/KPE smem) + double-buffered
// warp_part => 1 sync/token.
// ============================================================
__global__ __launch_bounds__(BLOCK_THREADS, 8)
void mla_paged_decode_smalll(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv_cache,
    const __nv_bfloat16* __restrict__ kpe_cache,
    const int32_t*        __restrict__ kv_indptr,
    const int32_t*        __restrict__ kv_indices,
    __nv_bfloat16*        __restrict__ output,
    float*                __restrict__ lse,
    float                              sm_scale
) {
    const int b       = blockIdx.x;
    const int h       = blockIdx.y;
    const int t       = threadIdx.x;
    const int warp_id = t >> 5;
    const int lane_id = t & 31;

    const int page_beg = kv_indptr[b];
    const int page_end = kv_indptr[b + 1];
    const int L_tokens = page_end - page_beg;

    __nv_bfloat16* out_base =
        output + ((int64_t)b * NUM_HEADS + h) * DIM_CKV + t * ELEMS_PER_THREAD_CKV;

    if (L_tokens <= 0) {
        *reinterpret_cast<uint2*>(out_base) = make_uint2(0u, 0u);
        if (t == 0) lse[(int64_t)b * NUM_HEADS + h] = -__builtin_inff();
        return;
    }

    extern __shared__ float warp_part[];

    const __nv_bfloat16* qn_ptr =
        q_nope + ((int64_t)b * NUM_HEADS + h) * DIM_CKV + t * ELEMS_PER_THREAD_CKV;
    uint2 qn_vec = __ldg(reinterpret_cast<const uint2*>(qn_ptr));
    const __nv_bfloat16* qnr = reinterpret_cast<const __nv_bfloat16*>(&qn_vec);
    const float qn0 = __bfloat162float(qnr[0]);
    const float qn1 = __bfloat162float(qnr[1]);
    const float qn2 = __bfloat162float(qnr[2]);
    const float qn3 = __bfloat162float(qnr[3]);

    float qp = 0.0f;
    if (t < DIM_KPE)
        qp = __bfloat162float(__ldg(q_pe + ((int64_t)b * NUM_HEADS + h) * DIM_KPE + t));

    const float sm2 = sm_scale * LOG2E_F;
    float m = -FLT_MAX, d = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    int buf = 0;
    for (int tok = 0; tok < L_tokens; tok++) {
        const int page_idx = kv_indices[page_beg + tok];

        float kc0, kc1, kc2, kc3, kp_val;
        LOAD_CKV_CS(page_idx, kc0, kc1, kc2, kc3, t)
        LOAD_KPE_CS(page_idx, kp_val, t)

        float dot = qn0*kc0 + qn1*kc1 + qn2*kc2 + qn3*kc3;
        if (t < DIM_KPE) dot += qp * kp_val;

        const float ws = warp_reduce_sum_f32(dot);
        float* wp = warp_part + buf * NUM_WARPS;
        if (lane_id == 0) wp[warp_id] = ws;
        __syncthreads();

        float tot = 0.0f;
#pragma unroll
        for (int w = 0; w < NUM_WARPS; w++) tot += wp[w];
        const float logit = tot * sm2;

        SOFTMAX_UPDATE(logit, kc0, kc1, kc2, kc3)
        buf ^= 1;
    }

    const float inv_d = (d > 0.0f) ? (1.0f / d) : 0.0f;
    __nv_bfloat16 ov[4];
    ov[0] = __float2bfloat16(acc0 * inv_d);
    ov[1] = __float2bfloat16(acc1 * inv_d);
    ov[2] = __float2bfloat16(acc2 * inv_d);
    ov[3] = __float2bfloat16(acc3 * inv_d);
    *reinterpret_cast<uint2*>(out_base) = *reinterpret_cast<const uint2*>(ov);
    if (t == 0)
        lse[(int64_t)b * NUM_HEADS + h] = (d > 0.0f) ? (m + log2f(d)) : -__builtin_inff();
}

// ============================================================
// Split-K Pass 1 — 8-token unrolled + ld.global.ca + batched softmax
// Grid: (batch_size, num_heads, k_total_chunks)
// Block: 128 threads (4 warps), 256 bytes SMEM
// ============================================================
__global__ __launch_bounds__(BLOCK_THREADS, 8)
void mla_paged_decode_splitk_pass1(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv_cache,
    const __nv_bfloat16* __restrict__ kpe_cache,
    const int32_t*        __restrict__ kv_indptr,
    const int32_t*        __restrict__ kv_indices,
    float*                __restrict__ workspace,
    float                              sm_scale,
    int                                k_total_chunks
) {
    const int b        = blockIdx.x;
    const int h        = blockIdx.y;
    const int k_chunk  = blockIdx.z;
    const int t        = threadIdx.x;
    const int warp_id  = t >> 5;
    const int lane_id  = t & 31;

    const int page_beg  = kv_indptr[b];
    const int page_end  = kv_indptr[b + 1];
    const int L_tokens  = page_end - page_beg;
    const int chunk_size = (L_tokens + k_total_chunks - 1) / k_total_chunks;
    const int rel_beg   = k_chunk * chunk_size;
    const int rel_end   = min(rel_beg + chunk_size, L_tokens);
    const int chunk_len = max(0, rel_end - rel_beg);

    float* ws = workspace +
        ((int64_t)(b * NUM_HEADS + h) * k_total_chunks + k_chunk) * WORKSPACE_SLOT_SIZE;

    if (chunk_len <= 0) {
        if (t == 0) { ws[0] = -FLT_MAX; ws[1] = 0.0f; }
        ws[2 + t * ELEMS_PER_THREAD_CKV + 0] = 0.0f;
        ws[2 + t * ELEMS_PER_THREAD_CKV + 1] = 0.0f;
        ws[2 + t * ELEMS_PER_THREAD_CKV + 2] = 0.0f;
        ws[2 + t * ELEMS_PER_THREAD_CKV + 3] = 0.0f;
        return;
    }

    const int abs_beg = page_beg + rel_beg;
    extern __shared__ float smem_wpart[];

    // Load query
    const __nv_bfloat16* qn_ptr =
        q_nope + ((int64_t)b * NUM_HEADS + h) * DIM_CKV + t * ELEMS_PER_THREAD_CKV;
    uint2 qn_vec = __ldg(reinterpret_cast<const uint2*>(qn_ptr));
    const __nv_bfloat16* qnr = reinterpret_cast<const __nv_bfloat16*>(&qn_vec);
    const float qn0 = __bfloat162float(qnr[0]);
    const float qn1 = __bfloat162float(qnr[1]);
    const float qn2 = __bfloat162float(qnr[2]);
    const float qn3 = __bfloat162float(qnr[3]);

    float qp = 0.0f;
    if (t < DIM_KPE)
        qp = __bfloat162float(__ldg(q_pe + ((int64_t)b * NUM_HEADS + h) * DIM_KPE + t));

    const float sm2 = sm_scale * LOG2E_F;
    float m = -FLT_MAX, d = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    // ── 8-token unrolled main loop ─────────────────────────────────────────────
    int local_tok = 0;
    const int chunk_len8 = (chunk_len >> 3) << 3;  // round down to multiple of 8
    int buf = 0;  // double-buffer flip: 0 or 1, selects smem wpart region

    for (; local_tok < chunk_len8; local_tok += 8) {
        const int pi0 = kv_indices[abs_beg + local_tok + 0];
        const int pi1 = kv_indices[abs_beg + local_tok + 1];
        const int pi2 = kv_indices[abs_beg + local_tok + 2];
        const int pi3 = kv_indices[abs_beg + local_tok + 3];
        const int pi4 = kv_indices[abs_beg + local_tok + 4];
        const int pi5 = kv_indices[abs_beg + local_tok + 5];
        const int pi6 = kv_indices[abs_beg + local_tok + 6];
        const int pi7 = kv_indices[abs_beg + local_tok + 7];

        // Issue all 8 CKV loads simultaneously (outstanding requests)
        float k0_0, k0_1, k0_2, k0_3;
        float k1_0, k1_1, k1_2, k1_3;
        float k2_0, k2_1, k2_2, k2_3;
        float k3_0, k3_1, k3_2, k3_3;
        float k4_0, k4_1, k4_2, k4_3;
        float k5_0, k5_1, k5_2, k5_3;
        float k6_0, k6_1, k6_2, k6_3;
        float k7_0, k7_1, k7_2, k7_3;

        LOAD_CKV_CS(pi0, k0_0, k0_1, k0_2, k0_3, t)
        LOAD_CKV_CS(pi1, k1_0, k1_1, k1_2, k1_3, t)
        LOAD_CKV_CS(pi2, k2_0, k2_1, k2_2, k2_3, t)
        LOAD_CKV_CS(pi3, k3_0, k3_1, k3_2, k3_3, t)
        LOAD_CKV_CS(pi4, k4_0, k4_1, k4_2, k4_3, t)
        LOAD_CKV_CS(pi5, k5_0, k5_1, k5_2, k5_3, t)
        LOAD_CKV_CS(pi6, k6_0, k6_1, k6_2, k6_3, t)
        LOAD_CKV_CS(pi7, k7_0, k7_1, k7_2, k7_3, t)

        // Issue 8 KPE loads (L1+L2 cached)
        float p0, p1, p2, p3, p4, p5, p6, p7;
        LOAD_KPE_CS(pi0, p0, t) LOAD_KPE_CS(pi1, p1, t) LOAD_KPE_CS(pi2, p2, t) LOAD_KPE_CS(pi3, p3, t)
        LOAD_KPE_CS(pi4, p4, t) LOAD_KPE_CS(pi5, p5, t) LOAD_KPE_CS(pi6, p6, t) LOAD_KPE_CS(pi7, p7, t)

        // 8 partial dot products
        float d0 = qn0*k0_0 + qn1*k0_1 + qn2*k0_2 + qn3*k0_3;
        float d1 = qn0*k1_0 + qn1*k1_1 + qn2*k1_2 + qn3*k1_3;
        float d2 = qn0*k2_0 + qn1*k2_1 + qn2*k2_2 + qn3*k2_3;
        float d3 = qn0*k3_0 + qn1*k3_1 + qn2*k3_2 + qn3*k3_3;
        float d4 = qn0*k4_0 + qn1*k4_1 + qn2*k4_2 + qn3*k4_3;
        float d5 = qn0*k5_0 + qn1*k5_1 + qn2*k5_2 + qn3*k5_3;
        float d6 = qn0*k6_0 + qn1*k6_1 + qn2*k6_2 + qn3*k6_3;
        float d7 = qn0*k7_0 + qn1*k7_1 + qn2*k7_2 + qn3*k7_3;
        if (t < DIM_KPE) {
            d0 += qp*p0; d1 += qp*p1; d2 += qp*p2; d3 += qp*p3;
            d4 += qp*p4; d5 += qp*p5; d6 += qp*p6; d7 += qp*p7;
        }

        // 8 warp reduces
        const float ws0 = warp_reduce_sum_f32(d0);
        const float ws1 = warp_reduce_sum_f32(d1);
        const float ws2 = warp_reduce_sum_f32(d2);
        const float ws3 = warp_reduce_sum_f32(d3);
        const float ws4 = warp_reduce_sum_f32(d4);
        const float ws5 = warp_reduce_sum_f32(d5);
        const float ws6 = warp_reduce_sum_f32(d6);
        const float ws7 = warp_reduce_sum_f32(d7);

        // Write 8 warp partial sums to smem (region selected by `buf`).
        // Next iteration writes to the OTHER region -> no second sync needed.
        float* wp = smem_wpart + buf * WPART_BUF;
        if (lane_id == 0) {
            wp[0 * NUM_WARPS + warp_id] = ws0;
            wp[1 * NUM_WARPS + warp_id] = ws1;
            wp[2 * NUM_WARPS + warp_id] = ws2;
            wp[3 * NUM_WARPS + warp_id] = ws3;
            wp[4 * NUM_WARPS + warp_id] = ws4;
            wp[5 * NUM_WARPS + warp_id] = ws5;
            wp[6 * NUM_WARPS + warp_id] = ws6;
            wp[7 * NUM_WARPS + warp_id] = ws7;
        }
        __syncthreads();  // the only sync per 8 tokens

        // Read cross-warp sums and compute logits
        float tot0=0.f, tot1=0.f, tot2=0.f, tot3=0.f;
        float tot4=0.f, tot5=0.f, tot6=0.f, tot7=0.f;
#pragma unroll
        for (int w = 0; w < NUM_WARPS; w++) {
            tot0 += wp[0 * NUM_WARPS + w];
            tot1 += wp[1 * NUM_WARPS + w];
            tot2 += wp[2 * NUM_WARPS + w];
            tot3 += wp[3 * NUM_WARPS + w];
            tot4 += wp[4 * NUM_WARPS + w];
            tot5 += wp[5 * NUM_WARPS + w];
            tot6 += wp[6 * NUM_WARPS + w];
            tot7 += wp[7 * NUM_WARPS + w];
        }
        const float l0 = tot0 * sm2, l1 = tot1 * sm2;
        const float l2 = tot2 * sm2, l3 = tot3 * sm2;
        const float l4 = tot4 * sm2, l5 = tot5 * sm2;
        const float l6 = tot6 * sm2, l7 = tot7 * sm2;

        // Batched online softmax update for all 8 tokens at once:
        // new max over (m, l0..l7), ONE rescale by alpha=exp2(m-m_new),
        // plus 8 exp2 for the new token weights = 9 exp2f per 8 tokens.
        float m_new = fmaxf(fmaxf(fmaxf(fmaxf(m, l0), l1), l2), l3);
        m_new = fmaxf(fmaxf(fmaxf(fmaxf(m_new, l4), l5), l6), l7);
        const float alpha = exp2f(m - m_new);
        const float e0 = exp2f(l0 - m_new);
        const float e1 = exp2f(l1 - m_new);
        const float e2 = exp2f(l2 - m_new);
        const float e3 = exp2f(l3 - m_new);
        const float e4 = exp2f(l4 - m_new);
        const float e5 = exp2f(l5 - m_new);
        const float e6 = exp2f(l6 - m_new);
        const float e7 = exp2f(l7 - m_new);
        d    = d   * alpha + (e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7);
        acc0 = acc0 * alpha + (e0*k0_0 + e1*k1_0 + e2*k2_0 + e3*k3_0 + e4*k4_0 + e5*k5_0 + e6*k6_0 + e7*k7_0);
        acc1 = acc1 * alpha + (e0*k0_1 + e1*k1_1 + e2*k2_1 + e3*k3_1 + e4*k4_1 + e5*k5_1 + e6*k6_1 + e7*k7_1);
        acc2 = acc2 * alpha + (e0*k0_2 + e1*k1_2 + e2*k2_2 + e3*k3_2 + e4*k4_2 + e5*k5_2 + e6*k6_2 + e7*k7_2);
        acc3 = acc3 * alpha + (e0*k0_3 + e1*k1_3 + e2*k2_3 + e3*k3_3 + e4*k4_3 + e5*k5_3 + e6*k6_3 + e7*k7_3);
        m = m_new;

        buf ^= 1;
    }
    // Separate the loop from the tail (tail reuses smem region 0).
    __syncthreads();

    // ── 4-token tail ──────────────────────────────────────────────────────────
    if (local_tok + 4 <= chunk_len) {
        const int pi0 = kv_indices[abs_beg + local_tok + 0];
        const int pi1 = kv_indices[abs_beg + local_tok + 1];
        const int pi2 = kv_indices[abs_beg + local_tok + 2];
        const int pi3 = kv_indices[abs_beg + local_tok + 3];

        float k0_0, k0_1, k0_2, k0_3;
        float k1_0, k1_1, k1_2, k1_3;
        float k2_0, k2_1, k2_2, k2_3;
        float k3_0, k3_1, k3_2, k3_3;
        LOAD_CKV_CS(pi0, k0_0, k0_1, k0_2, k0_3, t)
        LOAD_CKV_CS(pi1, k1_0, k1_1, k1_2, k1_3, t)
        LOAD_CKV_CS(pi2, k2_0, k2_1, k2_2, k2_3, t)
        LOAD_CKV_CS(pi3, k3_0, k3_1, k3_2, k3_3, t)

        float p0=0.f, p1=0.f, p2=0.f, p3=0.f;
        LOAD_KPE_CS(pi0, p0, t) LOAD_KPE_CS(pi1, p1, t) LOAD_KPE_CS(pi2, p2, t) LOAD_KPE_CS(pi3, p3, t)

        float d0=qn0*k0_0+qn1*k0_1+qn2*k0_2+qn3*k0_3;
        float d1=qn0*k1_0+qn1*k1_1+qn2*k1_2+qn3*k1_3;
        float d2=qn0*k2_0+qn1*k2_1+qn2*k2_2+qn3*k2_3;
        float d3=qn0*k3_0+qn1*k3_1+qn2*k3_2+qn3*k3_3;
        if (t < DIM_KPE) { d0+=qp*p0; d1+=qp*p1; d2+=qp*p2; d3+=qp*p3; }

        const float ws0=warp_reduce_sum_f32(d0), ws1=warp_reduce_sum_f32(d1);
        const float ws2=warp_reduce_sum_f32(d2), ws3=warp_reduce_sum_f32(d3);
        if (lane_id == 0) {
            smem_wpart[0*NUM_WARPS+warp_id]=ws0; smem_wpart[1*NUM_WARPS+warp_id]=ws1;
            smem_wpart[2*NUM_WARPS+warp_id]=ws2; smem_wpart[3*NUM_WARPS+warp_id]=ws3;
        }
        __syncthreads();
        float tot0=0.f,tot1=0.f,tot2=0.f,tot3=0.f;
#pragma unroll
        for (int w=0;w<NUM_WARPS;w++) {
            tot0+=smem_wpart[0*NUM_WARPS+w]; tot1+=smem_wpart[1*NUM_WARPS+w];
            tot2+=smem_wpart[2*NUM_WARPS+w]; tot3+=smem_wpart[3*NUM_WARPS+w];
        }
        // Batched softmax for 4-token tail (5 exp2f vs 8 sequential).
        {
            const float l0=tot0*sm2, l1=tot1*sm2, l2=tot2*sm2, l3=tot3*sm2;
            float m_new = fmaxf(fmaxf(fmaxf(m, l0), l1), l2); m_new = fmaxf(m_new, l3);
            const float alpha = exp2f(m - m_new);
            const float e0 = exp2f(l0 - m_new);
            const float e1 = exp2f(l1 - m_new);
            const float e2 = exp2f(l2 - m_new);
            const float e3 = exp2f(l3 - m_new);
            d    = d   * alpha + (e0 + e1 + e2 + e3);
            acc0 = acc0 * alpha + (e0*k0_0 + e1*k1_0 + e2*k2_0 + e3*k3_0);
            acc1 = acc1 * alpha + (e0*k0_1 + e1*k1_1 + e2*k2_1 + e3*k3_1);
            acc2 = acc2 * alpha + (e0*k0_2 + e1*k1_2 + e2*k2_2 + e3*k3_2);
            acc3 = acc3 * alpha + (e0*k0_3 + e1*k1_3 + e2*k2_3 + e3*k3_3);
            m = m_new;
        }
        __syncthreads();
        local_tok += 4;
    }

    // ── 2-token tail ──────────────────────────────────────────────────────────
    if (local_tok + 2 <= chunk_len) {
        const int pi0 = kv_indices[abs_beg + local_tok];
        const int pi1 = kv_indices[abs_beg + local_tok + 1];
        float k0_0,k0_1,k0_2,k0_3,k1_0,k1_1,k1_2,k1_3,p0=0.f,p1=0.f;
        LOAD_CKV_CS(pi0, k0_0, k0_1, k0_2, k0_3, t)
        LOAD_CKV_CS(pi1, k1_0, k1_1, k1_2, k1_3, t)
        LOAD_KPE_CS(pi0, p0, t) LOAD_KPE_CS(pi1, p1, t)
        float d0=qn0*k0_0+qn1*k0_1+qn2*k0_2+qn3*k0_3;
        float d1=qn0*k1_0+qn1*k1_1+qn2*k1_2+qn3*k1_3;
        if (t<DIM_KPE) { d0+=qp*p0; d1+=qp*p1; }
        const float ws0=warp_reduce_sum_f32(d0), ws1=warp_reduce_sum_f32(d1);
        if (lane_id==0) { smem_wpart[0*NUM_WARPS+warp_id]=ws0; smem_wpart[1*NUM_WARPS+warp_id]=ws1; }
        __syncthreads();
        float tot0=0.f,tot1=0.f;
        for (int w=0;w<NUM_WARPS;w++) { tot0+=smem_wpart[0*NUM_WARPS+w]; tot1+=smem_wpart[1*NUM_WARPS+w]; }
        // Batched softmax for 2-token tail (3 exp2f vs 4 sequential).
        {
            const float l0=tot0*sm2, l1=tot1*sm2;
            const float m_new = fmaxf(fmaxf(m, l0), l1);
            const float alpha = exp2f(m - m_new);
            const float e0 = exp2f(l0 - m_new);
            const float e1 = exp2f(l1 - m_new);
            d    = d   * alpha + (e0 + e1);
            acc0 = acc0 * alpha + (e0*k0_0 + e1*k1_0);
            acc1 = acc1 * alpha + (e0*k0_1 + e1*k1_1);
            acc2 = acc2 * alpha + (e0*k0_2 + e1*k1_2);
            acc3 = acc3 * alpha + (e0*k0_3 + e1*k1_3);
            m = m_new;
        }
        __syncthreads();
        local_tok += 2;
    }

    // ── 1-token tail ──────────────────────────────────────────────────────────
    if (local_tok < chunk_len) {
        const int pi = kv_indices[abs_beg + local_tok];
        float kc0, kc1, kc2, kc3, kp = 0.f;
        LOAD_CKV_CS(pi, kc0, kc1, kc2, kc3, t)
        LOAD_KPE_CS(pi, kp, t)
        float dot = qn0*kc0+qn1*kc1+qn2*kc2+qn3*kc3;
        if (t < DIM_KPE) dot += qp*kp;
        const float ws_ = warp_reduce_sum_f32(dot);
        if (lane_id==0) smem_wpart[warp_id]=ws_;
        __syncthreads();
        float tot=0.f;
        for (int w=0;w<NUM_WARPS;w++) tot+=smem_wpart[w];
        SOFTMAX_UPDATE(tot*sm2, kc0, kc1, kc2, kc3)
    }

    // Write workspace
    if (t == 0) { ws[0] = m; ws[1] = d; }
    ws[2 + t * ELEMS_PER_THREAD_CKV + 0] = acc0;
    ws[2 + t * ELEMS_PER_THREAD_CKV + 1] = acc1;
    ws[2 + t * ELEMS_PER_THREAD_CKV + 2] = acc2;
    ws[2 + t * ELEMS_PER_THREAD_CKV + 3] = acc3;
}

// ============================================================
// Split-K Pass 2 — Streaming single-pass online merge
// Grid: (batch_size, num_heads), 128 threads
// ============================================================
__global__ __launch_bounds__(BLOCK_THREADS, 4)
void mla_paged_decode_splitk_pass2(
    const float*          __restrict__ workspace,
    __nv_bfloat16*        __restrict__ output,
    float*                __restrict__ lse,
    const int32_t*        __restrict__ kv_indptr,
    int                                k_total_chunks
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int t = threadIdx.x;

    const int page_beg = kv_indptr[b];
    const int page_end = kv_indptr[b + 1];
    const int L_tokens = page_end - page_beg;

    __nv_bfloat16* out_base =
        output + ((int64_t)b * NUM_HEADS + h) * DIM_CKV + t * ELEMS_PER_THREAD_CKV;

    if (L_tokens <= 0) {
        *reinterpret_cast<uint2*>(out_base) = make_uint2(0u, 0u);
        if (t == 0) lse[(int64_t)b * NUM_HEADS + h] = -__builtin_inff();
        return;
    }

    const float* ws_bh = workspace +
        (int64_t)(b * NUM_HEADS + h) * k_total_chunks * WORKSPACE_SLOT_SIZE;
    const int acc_off = 2 + t * ELEMS_PER_THREAD_CKV;

    float m_global = -FLT_MAX;
    float l_global = 0.0f;
    float out0 = 0.0f, out1 = 0.0f, out2 = 0.0f, out3 = 0.0f;

    for (int k = 0; k < k_total_chunks; k++) {
        const float* wk = ws_bh + k * WORKSPACE_SLOT_SIZE;
        const float mk = wk[0];
        const float lk = wk[1];
        if (lk <= 0.0f) continue;

        const float new_m = fmaxf(m_global, mk);
        const float alpha = exp2f(m_global - new_m);
        const float beta  = exp2f(mk - new_m);

        const float* ack = wk + acc_off;
        out0 = alpha * out0 + beta * ack[0];
        out1 = alpha * out1 + beta * ack[1];
        out2 = alpha * out2 + beta * ack[2];
        out3 = alpha * out3 + beta * ack[3];
        l_global = alpha * l_global + beta * lk;
        m_global = new_m;
    }

    const float inv_l = (l_global > 0.0f) ? (1.0f / l_global) : 0.0f;
    __nv_bfloat16 ov[4];
    ov[0] = __float2bfloat16(out0 * inv_l);
    ov[1] = __float2bfloat16(out1 * inv_l);
    ov[2] = __float2bfloat16(out2 * inv_l);
    ov[3] = __float2bfloat16(out3 * inv_l);
    *reinterpret_cast<uint2*>(out_base) = *reinterpret_cast<const uint2*>(ov);

    if (t == 0) {
        lse[(int64_t)b * NUM_HEADS + h] =
            (l_global > 0.0f && m_global > -1e37f)
                ? (m_global + log2f(l_global))
                : -__builtin_inff();
    }
}

// ============================================================
// Host launcher (entry point re-exported by kernel_binding.cpp)
// ============================================================
py::dict run(
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor ckv_cache,
    torch::Tensor kpe_cache,
    torch::Tensor kv_indptr,
    torch::Tensor kv_indices,
    float sm_scale
) {
    TORCH_CHECK(q_nope.is_cuda(),     "q_nope must be a CUDA tensor");
    TORCH_CHECK(q_nope.dim() == 3,    "q_nope must be 3-D");
    TORCH_CHECK(q_pe.dim() == 3,      "q_pe must be 3-D");
    TORCH_CHECK(q_nope.scalar_type() == torch::kBFloat16, "q_nope must be bf16");
    TORCH_CHECK(q_pe.scalar_type()   == torch::kBFloat16, "q_pe must be bf16");

    const int batch = q_nope.size(0);

    auto ckv_flat = ckv_cache.reshape({-1, DIM_CKV}).contiguous();
    auto kpe_flat = kpe_cache.reshape({-1, DIM_KPE}).contiguous();

    if (batch == 0) {
        auto output = torch::zeros(
            {batch, NUM_HEADS, DIM_CKV},
            torch::TensorOptions().dtype(torch::kBFloat16).device(q_nope.device()));
        auto lse_out = torch::full(
            {batch, NUM_HEADS},
            -std::numeric_limits<float>::infinity(),
            torch::TensorOptions().dtype(torch::kFloat32).device(q_nope.device()));
        py::dict result;
        result["output"] = output;
        result["lse"]    = lse_out;
        return result;
    }

    // V21: every kernel path below writes the FULL output and lse grids
    // (grid covers (batch, NUM_HEADS); the L_tokens<=0 early-return in both
    // smalll and pass2 writes zeros/-inf), so torch::empty (no fill kernel)
    // replaces torch::zeros / torch::full. This removes two fill launches
    // from the critical path of a ~0.1ms call.
    auto output = torch::empty(
        {batch, NUM_HEADS, DIM_CKV},
        torch::TensorOptions().dtype(torch::kBFloat16).device(q_nope.device()));
    auto lse_out = torch::empty(
        {batch, NUM_HEADS},
        torch::TensorOptions().dtype(torch::kFloat32).device(q_nope.device()));

    auto q_nope_c     = q_nope.contiguous();
    auto q_pe_c       = q_pe.contiguous();
    auto kv_indptr_c  = kv_indptr.contiguous();
    auto kv_indices_c = kv_indices.contiguous();
    auto stream = at::cuda::getCurrentCUDAStream();

    const int total_kv = (int)kv_indices_c.size(0);
    const int avg_L    = (batch > 0) ? (total_kv / batch) : 0;

    const bool use_smalll  = (avg_L <= SMALLL_THRESH);
    const bool use_splitk  = !use_smalll && (avg_L >= SPLIT_L_THRESH);
    const bool use_long_k  = use_splitk && (avg_L >= LONG_L_THRESH);
    const int  k_chunks    = use_long_k ? K_MAX_CHUNKS_LONG : K_MAX_CHUNKS;

    const __nv_bfloat16* ckv_ptr = reinterpret_cast<const __nv_bfloat16*>(ckv_flat.data_ptr());
    const __nv_bfloat16* kpe_ptr = reinterpret_cast<const __nv_bfloat16*>(kpe_flat.data_ptr());
    const __nv_bfloat16* qn_ptr  = reinterpret_cast<const __nv_bfloat16*>(q_nope_c.data_ptr());
    const __nv_bfloat16* qp_ptr  = reinterpret_cast<const __nv_bfloat16*>(q_pe_c.data_ptr());
    __nv_bfloat16*       out_ptr = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
    float*               lse_ptr = lse_out.data_ptr<float>();

    if (use_smalll) {
        dim3 grid(batch, NUM_HEADS);
        dim3 block(BLOCK_THREADS);
        static bool once = false;
        if (!once) {
            cudaFuncSetAttribute(mla_paged_decode_smalll,
                cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SMALLL);
            once = true;
        }
        mla_paged_decode_smalll<<<grid, block, SMEM_SMALLL, stream>>>(
            qn_ptr, qp_ptr, ckv_ptr, kpe_ptr,
            kv_indptr_c.data_ptr<int32_t>(), kv_indices_c.data_ptr<int32_t>(),
            out_ptr, lse_ptr, sm_scale);
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "smalll launch failed");

    } else if (use_splitk) {
        auto workspace = torch::empty(
            {(int64_t)batch * NUM_HEADS * k_chunks * WORKSPACE_SLOT_SIZE},
            torch::TensorOptions().dtype(torch::kFloat32).device(q_nope.device()));

        dim3 grid1(batch, NUM_HEADS, k_chunks);
        dim3 block1(BLOCK_THREADS);
        static bool once1 = false;
        if (!once1) {
            cudaFuncSetAttribute(mla_paged_decode_splitk_pass1,
                cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_PASS1);
            once1 = true;
        }
        mla_paged_decode_splitk_pass1<<<grid1, block1, SMEM_PASS1, stream>>>(
            qn_ptr, qp_ptr, ckv_ptr, kpe_ptr,
            kv_indptr_c.data_ptr<int32_t>(), kv_indices_c.data_ptr<int32_t>(),
            reinterpret_cast<float*>(workspace.data_ptr()),
            sm_scale, k_chunks);
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "pass1 launch failed");

        dim3 grid2(batch, NUM_HEADS);
        dim3 block2(BLOCK_THREADS);
        mla_paged_decode_splitk_pass2<<<grid2, block2, 0, stream>>>(
            reinterpret_cast<const float*>(workspace.data_ptr()),
            out_ptr, lse_ptr,
            kv_indptr_c.data_ptr<int32_t>(), k_chunks);
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "pass2 launch failed");

    } else {
        // Middle range: 32 < avg_L < 64 -> split-K with k_small=4
        const int k_small = 4;
        auto workspace = torch::empty(
            {(int64_t)batch * NUM_HEADS * k_small * WORKSPACE_SLOT_SIZE},
            torch::TensorOptions().dtype(torch::kFloat32).device(q_nope.device()));

        dim3 grid1(batch, NUM_HEADS, k_small);
        dim3 block1(BLOCK_THREADS);
        static bool once_mid = false;
        if (!once_mid) {
            cudaFuncSetAttribute(mla_paged_decode_splitk_pass1,
                cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_PASS1);
            once_mid = true;
        }
        mla_paged_decode_splitk_pass1<<<grid1, block1, SMEM_PASS1, stream>>>(
            qn_ptr, qp_ptr, ckv_ptr, kpe_ptr,
            kv_indptr_c.data_ptr<int32_t>(), kv_indices_c.data_ptr<int32_t>(),
            reinterpret_cast<float*>(workspace.data_ptr()),
            sm_scale, k_small);
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "mid pass1 launch failed");

        dim3 grid2(batch, NUM_HEADS);
        dim3 block2(BLOCK_THREADS);
        mla_paged_decode_splitk_pass2<<<grid2, block2, 0, stream>>>(
            reinterpret_cast<const float*>(workspace.data_ptr()),
            out_ptr, lse_ptr,
            kv_indptr_c.data_ptr<int32_t>(), k_small);
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "mid pass2 launch failed");
    }

    py::dict result;
    result["output"] = output;
    result["lse"]    = lse_out;
    return result;
}

// kernel_binding.cpp — pybind11 bindings for the MLA paged decode CUDA
// extension (kernel.cu). The kernels and the host launcher `run` are defined
// in kernel.cu; this translation unit only declares and exposes `run` to
// Python.

#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Host launcher defined in kernel.cu.
py::dict run(
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor ckv_cache,
    torch::Tensor kpe_cache,
    torch::Tensor kv_indptr,
    torch::Tensor kv_indices,
    float sm_scale
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "MLA Paged Decode V21: 8-token unrolled ld.global.ca KV gathers "
          "(L1+L2 cached for inter-head KV reuse) + double-buffered wpart "
          "(1 sync per 8 tokens) + batched online softmax (9 exp2f/8tok) + "
          "fill-kernel-free output allocation. Split-KV pass1/pass2 + small-L "
          "direct-register path. V20 was 1.040x over V13.",
          py::arg("q_nope"), py::arg("q_pe"),
          py::arg("ckv_cache"), py::arg("kpe_cache"),
          py::arg("kv_indptr"), py::arg("kv_indices"),
          py::arg("sm_scale"));
}
