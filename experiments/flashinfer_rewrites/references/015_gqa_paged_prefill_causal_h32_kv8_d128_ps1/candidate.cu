/*
 * GQA Paged Prefill (causal) Kernel — Candidate r1-s2 (tensor_core_utilization).
 *
 * Op: gqa_paged_prefill_causal (h=32, kv=8, d=128, page_size=1, causal).
 * Target: RTX 4090 / sm_89 and B300 / sm_103 — both support
 *         mma.sync.aligned.m16n8k16 (bf16 in, fp32 acc) and ldmatrix, so the
 *         tensor-core path is portable across both measured targets.
 *
 * Design (changes vs v1_base are the compute path only):
 *   - Grid:  (NUM_KV_HEADS, num_q_tiles_total)                      [unchanged]
 *   - Block: 128 threads = 4 warps; warp w handles q_head qh = kv_head*4 + w
 *            (one warp per q_head, 4 heads per kv_head per CTA).    [unchanged]
 *   - BRQ widened 8 -> 16 so the Q tile matches one mma M=16 row tile.
 *   - Q tile (16x128 bf16) is staged per warp into SMEM (coalesced uint2
 *     loads) and loaded ONCE into persistent mma A-fragments (8 frags for
 *     d=128), replacing q_reg[8][4].
 *   - KV tokens still stream through SMEM in T_PIPE=16 tiles (cooperative
 *     load by the 4 warps, page_size=1 gather via kv_indices, zero-filled
 *     padding rows, same __syncthreads structure).                   [unchanged]
 *   - Q@K^T: K streamed from the existing SMEM K buffer via ldmatrix into
 *     B-fragments; 16 mma m16n8k16 ops per tile produce S=16x16 in fp32
 *     C-fragments. The scalar FMA + 5-step __shfl_down warp_reduce_sum issue
 *     bottleneck (old process_1tok/2tok/4tok) is deleted entirely.
 *   - Causal mask applied on the S fragment registers: each thread owns fixed
 *     (row, col) elements; tokens beyond the row's valid count — same rule as
 *     v1_base, valid = clamp(min(num_kv, q_local+delta+1) - kv_lo, 0, 16) —
 *     get -inf, so masking semantics (causal + KV padding) are identical.
 *   - Online softmax in log2 space on the fragment elements (same running
 *     max / running sum / rescale math as v1_base, evaluated once per
 *     16-token tile). A row's 16 tokens live in the 4 lanes of a quad, so the
 *     row max/sum reductions are 2-step __shfl_xor ops.
 *   - P@V: P=16x16 converted to bf16 A-fragments (P elements are
 *     probabilities <= 1 — the same bf16 quantization FlashInfer makes),
 *     V streamed via ldmatrix.trans from the same SMEM V buffer; 16 mma ops
 *     accumulate O=16x128 in fp32 C-fragments.
 *   - Final normalize + bf16 store (staged through the Q SMEM region for
 *     coalesced 256B row writes) and base-2 LSE output path are semantically
 *     unchanged: lse = running_max + log2(running_sum); a row with no valid
 *     KV token gets output 0 and lse -inf.
 *
 * SMEM layout (single buffered, rows padded +8 bf16 so ldmatrix row segments
 * land on distinct banks):
 *   smem_k[16][136], smem_v[16][136], smem_q[4 warps][16][136]
 *   = (2*16 + 4*16) * 136 * 2 B = 26112 B. The per-warp Q region is reused
 *   for the output staging at the end of the kernel.
 *
 * Tile meta (q_start_global, q_len, q_start_local, kv_start, num_kv, delta) is
 * built on the GPU from qo_indptr/kv_indptr (build_tile_meta_kernel) so the
 * grid is compact (no ragged/empty CTAs) and no host sync is needed.
 *
 * Edge cases:
 *   - Row with 0 valid KV tokens (max_kv_r <= 0): output 0, lse -inf.
 *   - Padding KV rows beyond num_kv in a tile: zero-filled in SMEM and masked
 *     to -inf in the S fragments.
 *   - q_len < 16 (last tile of a sequence): pad Q rows are zero-filled; their
 *     garbage results are never written out.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

static constexpr int HEAD_DIM      = 128;
static constexpr int NUM_QO_HEADS  = 32;
static constexpr int NUM_KV_HEADS  = 8;
static constexpr int GQA_RATIO     = 4;              // 32 / 8
static constexpr int BLOCK_THREADS = 128;            // 4 warps
static constexpr int DIMS_PER_LANE = HEAD_DIM / 32;  // = 4
static constexpr int BRQ           = 16;             // query rows per CTA tile (= mma M)
static constexpr int T_PIPE        = 16;             // KV tokens per SMEM tile
static constexpr int ROWS_PER_WARP = T_PIPE / 4;     // = 4 (4 warps load T_PIPE rows)
static constexpr int SMEM_STRIDE   = HEAD_DIM + 8;   // 136 bf16 per SMEM row (bank pad)

static constexpr float   LOG2E          = 1.4426950408889634f;
static constexpr unsigned FULL_WARP_MASK = 0xffffffffu;

// ---- 32-bit shared-memory address for ldmatrix ----
__device__ __forceinline__ unsigned smem_u32(const void* ptr)
{
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

// ---- ldmatrix x4: four 8x8 bf16 matrices -> 4 .b32 regs (mma A/B fragments) ----
__device__ __forceinline__ void ldmatrix_x4(
    unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, unsigned addr)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
}

// ---- ldmatrix x4 transposed (V: stored [token][dim], B-frag needs [dim][token]) ----
__device__ __forceinline__ void ldmatrix_x4_trans(
    unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, unsigned addr)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
}

// ---- D = A @ B + D ; mma.sync m16n8k16, bf16 in, fp32 accumulate ----
__device__ __forceinline__ void mma_m16n8k16_bf16(
    float& d0, float& d1, float& d2, float& d3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ---- pack two fp32 into one bf16x2 register (fragment element pair) ----
__device__ __forceinline__ unsigned pack_bf16x2(float lo, float hi)
{
    __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<unsigned*>(&h);
}

/*
 * Fragment ownership (m16n8k16, lane = 32*warps + l):
 *   groupID g = l >> 2, tid-in-group t = l & 3.
 *   A (16x16): a0 = A[g][2t..2t+1], a1 = A[g+8][2t..2t+1],
 *              a2 = A[g][8+2t..],   a3 = A[g+8][8+2t..]
 *   B (16x8) : b0 = B[2t..2t+1][g], b1 = B[8+2t..][g]
 *   C (16x8) : c0 = C[g][2t], c1 = C[g][2t+1], c2 = C[g+8][2t], c3 = C[g+8][2t+1]
 *
 * S = Q@K^T: A = Q (rows = queries, k = dims), B = K^T (k = dims, n = tokens).
 *   K is stored [token][dim], so a NON-trans ldmatrix of K rows delivers the
 *   B fragment directly (b0 = K[g][kb*16+2t], b1 = K[g][kb*16+8+2t]).
 * O = P@V: A = P (rows = queries, k = tokens), B = V (k = tokens, n = dims).
 *   V is stored [token][dim], so ldmatrix.trans is used for B.
 */
__global__ void __launch_bounds__(BLOCK_THREADS, 2)
gqa_paged_prefill_mma(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int*           __restrict__ kv_indices,
    float                             sm_scale,
    const int*           __restrict__ tile_meta,   // [num_tiles, 6]
    __nv_bfloat16*       __restrict__ output,
    float*               __restrict__ lse,
    const int*           __restrict__ d_num_tiles, // device-resident real tile count
    int                               num_tiles_bound) // grid upper bound (total_q)
{
    extern __shared__ __nv_bfloat16 smem_raw[];
    __nv_bfloat16* smem_k = smem_raw;                              // [T_PIPE][SMEM_STRIDE]
    __nv_bfloat16* smem_v = smem_raw + T_PIPE * SMEM_STRIDE;       // [T_PIPE][SMEM_STRIDE]
    __nv_bfloat16* smem_q = smem_raw + 2 * T_PIPE * SMEM_STRIDE;   // [4 warps][BRQ][SMEM_STRIDE]

    const int kv_head = blockIdx.x;
    const int tidx    = blockIdx.y;
    // Real num_tiles lives in device memory (computed by build_tile_meta_kernel).
    // CTAs beyond the real count early-return — no host sync needed to size grid.
    const int num_tiles = __ldg(d_num_tiles);
    if (tidx >= num_tiles) return;
    (void)num_tiles_bound;
    const int tid      = threadIdx.x;
    const int warp_id  = tid >> 5;
    const int lane_id  = tid & 31;
    const int lane_base = lane_id * DIMS_PER_LANE;
    const int g   = lane_id >> 2;   // fragment row group (rows g and g+8)
    const int tig = lane_id & 3;    // thread-in-group

    // tile_meta[tidx*6 + 0..5] = q_start_global, q_len, q_start_local, kv_start, num_kv, delta
    const int* meta = tile_meta + tidx * 6;
    const int q_start_global = meta[0];
    const int q_len          = meta[1];
    const int q_start_local  = meta[2];
    const int kv_start       = meta[3];
    const int num_kv         = meta[4];
    const int delta          = meta[5];

    const int qh = kv_head * GQA_RATIO + warp_id;
    const float sm_scale_log2 = sm_scale * LOG2E;
    const int kv_head_offset  = kv_head * HEAD_DIM;
    const int kv_stride       = NUM_KV_HEADS * HEAD_DIM;

    __nv_bfloat16* qsw = smem_q + warp_id * BRQ * SMEM_STRIDE;  // this warp's Q tile

    // ---- Stage this warp's Q tile (16 x 128) into SMEM (coalesced uint2/lane) ----
    #pragma unroll
    for (int r = 0; r < BRQ; r++) {
        uint2 v;
        if (r < q_len) {
            const __nv_bfloat16* q_row =
                q + ((size_t)(q_start_global + r) * NUM_QO_HEADS + qh) * HEAD_DIM;
            v = __ldg(reinterpret_cast<const uint2*>(q_row + lane_base));
        } else {
            v = make_uint2(0u, 0u);   // pad rows: zero Q -> finite S, never written out
        }
        *reinterpret_cast<uint2*>(qsw + r * SMEM_STRIDE + lane_base) = v;
    }
    __syncwarp();

    // ---- Load Q once into persistent mma A-fragments (8 frags for d=128) ----
    // ldmatrix x4 lane addressing: lane l -> row l&15, col kb*16 + (l>>4)*8.
    unsigned qa[8][4];
    {
        const int arow = lane_id & 15;
        const int acol = (lane_id >> 4) << 3;
        #pragma unroll
        for (int kb = 0; kb < 8; kb++) {
            ldmatrix_x4(qa[kb][0], qa[kb][1], qa[kb][2], qa[kb][3],
                        smem_u32(qsw + arow * SMEM_STRIDE + kb * 16 + acol));
        }
    }

    // ---- Accumulators: O = 16 x 128 fp32 C-fragments ----
    // o[db][0..3]: rows g/g+8, dims db*8 + tig*2 + {0,1}
    float o[16][4];
    float rmax0 = -FLT_MAX, rmax1 = -FLT_MAX;   // online-softmax state, rows g / g+8
    float rsum0 = 0.0f,     rsum1 = 0.0f;       // (replicated across the quad)
    #pragma unroll
    for (int db = 0; db < 16; db++) {
        o[db][0] = o[db][1] = o[db][2] = o[db][3] = 0.f;
    }

    // ---- KV loop bound: max KV index needed by any row in this tile ----
    // last row q_local = q_start_local + q_len - 1; max_kv = q_local+delta+1
    int max_max_kv = q_start_local + q_len + delta;   // = last_row_q_local + delta + 1
    if (max_max_kv > num_kv) max_max_kv = num_kv;
    if (max_max_kv < 0) max_max_kv = 0;
    const int num_kv_tiles = (max_max_kv + T_PIPE - 1) / T_PIPE;

    const int* idx_base = kv_indices + kv_start;

    // ldmatrix lane addressing constants (see fragment comment above):
    //   K (non-trans): token = (l&7) + ((l>>4)<<3), dim = ((l>>3)&1)<<3
    //   V (trans):     token = l&15,                dim = (l>>4)<<3
    const int krow_lane = (lane_id & 7) + ((lane_id >> 4) << 3);
    const int kcol_lane = ((lane_id >> 3) & 1) << 3;
    const int vrow_lane = lane_id & 15;
    const int vcol_lane = (lane_id >> 4) << 3;

    for (int tile = 0; tile < num_kv_tiles; tile++) {
        const int kv_lo        = tile * T_PIPE;
        const int tile_kv_count = (num_kv - kv_lo < T_PIPE) ? (num_kv - kv_lo) : T_PIPE;

        // guard: ensure previous tile's compute finished reading smem
        if (tile > 0) __syncthreads();

        // ---- Cooperative load of K/V tile into SMEM (4 warps * 4 rows) ----
        const int warp_row_start = warp_id * ROWS_PER_WARP;
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; r++) {
            int row = warp_row_start + r;
            if (row < tile_kv_count) {
                int tok_global = kv_lo + row;
                int pg = __ldg(idx_base + tok_global);
                int base_offset = pg * kv_stride + kv_head_offset + lane_base;
                // K (synchronous __ldg)
                *reinterpret_cast<uint2*>(smem_k + row * SMEM_STRIDE + lane_base) =
                    __ldg(reinterpret_cast<const uint2*>(k_cache + base_offset));
                // V (synchronous __ldg)
                *reinterpret_cast<uint2*>(smem_v + row * SMEM_STRIDE + lane_base) =
                    __ldg(reinterpret_cast<const uint2*>(v_cache + base_offset));
            } else {
                // padding row: zero
                #pragma unroll
                for (int d = 0; d < DIMS_PER_LANE; d++) {
                    smem_k[row * SMEM_STRIDE + lane_base + d] = __float2bfloat16(0.f);
                    smem_v[row * SMEM_STRIDE + lane_base + d] = __float2bfloat16(0.f);
                }
            }
        }

        __syncthreads();   // load complete

        // ---- Per-row valid token count in this tile (baseline masking rule) ----
        // valid = clamp(min(num_kv, q_local+delta+1) - kv_lo, 0, T_PIPE);
        // (<= tile_kv_count automatically because max_kv_r <= num_kv.)
        int mk0 = q_start_local + g     + delta + 1;
        int mk1 = q_start_local + g + 8 + delta + 1;
        if (mk0 > num_kv) mk0 = num_kv;
        if (mk1 > num_kv) mk1 = num_kv;
        int vr0 = mk0 - kv_lo;
        int vr1 = mk1 - kv_lo;
        if (vr0 < 0) vr0 = 0; if (vr0 > T_PIPE) vr0 = T_PIPE;
        if (vr1 < 0) vr1 = 0; if (vr1 > T_PIPE) vr1 = T_PIPE;

        // ---- S = Q @ K^T : 16x16 fp32 C-fragments (2 n-blocks x 8 k-blocks) ----
        // s[nb][0..3]: rows g/g+8, tokens nb*8 + tig*2 + {0,1}
        float s[2][4];
        #pragma unroll
        for (int nb = 0; nb < 2; nb++)
            #pragma unroll
            for (int j = 0; j < 4; j++) s[nb][j] = 0.f;
        #pragma unroll
        for (int kb = 0; kb < 8; kb++) {
            unsigned bk0, bk1, bk2, bk3;
            ldmatrix_x4(bk0, bk1, bk2, bk3,
                        smem_u32(smem_k + krow_lane * SMEM_STRIDE + kb * 16 + kcol_lane));
            mma_m16n8k16_bf16(s[0][0], s[0][1], s[0][2], s[0][3],
                              qa[kb][0], qa[kb][1], qa[kb][2], qa[kb][3], bk0, bk1);
            mma_m16n8k16_bf16(s[1][0], s[1][1], s[1][2], s[1][3],
                              qa[kb][0], qa[kb][1], qa[kb][2], qa[kb][3], bk2, bk3);
        }

        // ---- Scale (log2 space) + causal/padding mask on fragment elements ----
        #pragma unroll
        for (int nb = 0; nb < 2; nb++) {
            int tok = nb * 8 + tig * 2;
            s[nb][0] = (tok     < vr0) ? s[nb][0] * sm_scale_log2 : -INFINITY;
            s[nb][1] = (tok + 1 < vr0) ? s[nb][1] * sm_scale_log2 : -INFINITY;
            s[nb][2] = (tok     < vr1) ? s[nb][2] * sm_scale_log2 : -INFINITY;
            s[nb][3] = (tok + 1 < vr1) ? s[nb][3] * sm_scale_log2 : -INFINITY;
        }

        // ---- Row max over the 16 tokens (each row's tokens live in 4 lanes) ----
        float m0 = fmaxf(fmaxf(s[0][0], s[0][1]), fmaxf(s[1][0], s[1][1]));  // row g
        float m1 = fmaxf(fmaxf(s[0][2], s[0][3]), fmaxf(s[1][2], s[1][3]));  // row g+8
        m0 = fmaxf(m0, __shfl_xor_sync(FULL_WARP_MASK, m0, 1));
        m0 = fmaxf(m0, __shfl_xor_sync(FULL_WARP_MASK, m0, 2));
        m1 = fmaxf(m1, __shfl_xor_sync(FULL_WARP_MASK, m1, 1));
        m1 = fmaxf(m1, __shfl_xor_sync(FULL_WARP_MASK, m1, 2));

        float nm0 = fmaxf(rmax0, m0), nm1 = fmaxf(rmax1, m1);
        float ep0 = exp2f(rmax0 - nm0), ep1 = exp2f(rmax1 - nm1);

        // ---- P = exp2(S - new_max); masked elements -> 0 ----
        float p[2][4];
        #pragma unroll
        for (int nb = 0; nb < 2; nb++) {
            p[nb][0] = exp2f(s[nb][0] - nm0);
            p[nb][1] = exp2f(s[nb][1] - nm0);
            p[nb][2] = exp2f(s[nb][2] - nm1);
            p[nb][3] = exp2f(s[nb][3] - nm1);
        }

        // ---- Row sums over the quad, then online-softmax state update ----
        float ps0 = p[0][0] + p[0][1] + p[1][0] + p[1][1];
        float ps1 = p[0][2] + p[0][3] + p[1][2] + p[1][3];
        ps0 += __shfl_xor_sync(FULL_WARP_MASK, ps0, 1);
        ps0 += __shfl_xor_sync(FULL_WARP_MASK, ps0, 2);
        ps1 += __shfl_xor_sync(FULL_WARP_MASK, ps1, 1);
        ps1 += __shfl_xor_sync(FULL_WARP_MASK, ps1, 2);

        rsum0 = rsum0 * ep0 + ps0;
        rsum1 = rsum1 * ep1 + ps1;
        rmax0 = nm0;
        rmax1 = nm1;

        // ---- Rescale O rows by exp(prev_max - new_max) ----
        #pragma unroll
        for (int db = 0; db < 16; db++) {
            o[db][0] *= ep0; o[db][1] *= ep0;
            o[db][2] *= ep1; o[db][3] *= ep1;
        }

        // ---- P -> bf16 A-fragment (same layout as the Q A-fragment) ----
        unsigned pa0 = pack_bf16x2(p[0][0], p[0][1]);
        unsigned pa1 = pack_bf16x2(p[0][2], p[0][3]);
        unsigned pa2 = pack_bf16x2(p[1][0], p[1][1]);
        unsigned pa3 = pack_bf16x2(p[1][2], p[1][3]);

        // ---- O += P @ V : 16 mma ops (16 d-blocks of 8) ----
        #pragma unroll
        for (int dbp = 0; dbp < 8; dbp++) {
            unsigned bv0, bv1, bv2, bv3;
            ldmatrix_x4_trans(bv0, bv1, bv2, bv3,
                              smem_u32(smem_v + vrow_lane * SMEM_STRIDE + dbp * 16 + vcol_lane));
            mma_m16n8k16_bf16(o[dbp * 2][0], o[dbp * 2][1], o[dbp * 2][2], o[dbp * 2][3],
                              pa0, pa1, pa2, pa3, bv0, bv1);
            mma_m16n8k16_bf16(o[dbp * 2 + 1][0], o[dbp * 2 + 1][1],
                              o[dbp * 2 + 1][2], o[dbp * 2 + 1][3],
                              pa0, pa1, pa2, pa3, bv2, bv3);
        }
    }

    // ---- Normalize + store O (stage through this warp's Q SMEM region) ----
    float inv0 = (rsum0 > 0.0f) ? 1.0f / rsum0 : 0.0f;
    float inv1 = (rsum1 > 0.0f) ? 1.0f / rsum1 : 0.0f;
    #pragma unroll
    for (int db = 0; db < 16; db++) {
        *reinterpret_cast<unsigned*>(qsw + g * SMEM_STRIDE + db * 8 + tig * 2) =
            pack_bf16x2(o[db][0] * inv0, o[db][1] * inv0);
        *reinterpret_cast<unsigned*>(qsw + (g + 8) * SMEM_STRIDE + db * 8 + tig * 2) =
            pack_bf16x2(o[db][2] * inv1, o[db][3] * inv1);
    }
    __syncwarp();

    // Coalesced row writes: one 256B row per store instruction.
    #pragma unroll
    for (int r = 0; r < BRQ; r++) {
        if (r < q_len) {
            __nv_bfloat16* out_row =
                output + ((size_t)(q_start_global + r) * NUM_QO_HEADS + qh) * HEAD_DIM;
            *reinterpret_cast<uint2*>(out_row + lane_base) =
                *reinterpret_cast<const uint2*>(qsw + r * SMEM_STRIDE + lane_base);
        }
    }

    // ---- LSE (base-2), written by the quad leaders (state is quad-replicated) ----
    if (tig == 0) {
        #pragma unroll
        for (int rr = 0; rr < 2; rr++) {
            int r = g + rr * 8;
            if (r < q_len) {
                float rs = (rr == 0) ? rsum0 : rsum1;
                float rm = (rr == 0) ? rmax0 : rmax1;
                float* lse_ptr = lse + (size_t)(q_start_global + r) * NUM_QO_HEADS + qh;
                if (rs > 0.0f && isfinite(rs)) {
                    // base-2 LSE: max2 + log2(sum2)
                    *lse_ptr = rm + log2f(rs);
                } else {
                    *lse_ptr = -INFINITY;
                }
            }
        }
    }
}

// ---- GPU-side tile_meta construction (eliminates D2H/H2D sync) ----
// One block of <= 1024 threads, one thread per batch element. Computes the
// per-batch tile count, an exclusive prefix sum (tile_base[b]), and writes the
// 6-int tile_meta records for every tile of every non-empty batch. Also writes
// the total tile count to *d_num_tiles. Because ceil(seq_q/BRQ) <= seq_q for
// seq_q>=1 and tiles partition query rows, num_tiles <= total_q always holds,
// so the main kernel may safely launch gridDim.y = total_q and early-return
// CTAs whose tidx >= num_tiles (num_tiles read from *d_num_tiles in device
// memory — no host sync required before launch).
__global__ void build_tile_meta_kernel(
    const int* __restrict__ qo_indptr,   // [batch+1]
    const int* __restrict__ kv_indptr,   // [batch+1]
    int*       __restrict__ tile_meta,   // [total_q*6] (upper bound)
    int*       __restrict__ d_num_tiles,
    int                     batch)
{
    extern __shared__ int s_meta[];      // [batch] tile counts, then [batch] prefix
    int* s_cnt  = s_meta;                // per-batch tile count
    int* s_base = s_meta + batch;        // exclusive prefix sum of tile counts

    const int tid = threadIdx.x;
    int my_cnt = 0;
    if (tid < batch) {
        int q_start = qo_indptr[tid];
        int q_end   = qo_indptr[tid + 1];
        int seq_q   = q_end - q_start;
        my_cnt = (seq_q > 0) ? ((seq_q + BRQ - 1) / BRQ) : 0;
    }
    s_cnt[tid] = my_cnt;
    __syncthreads();

    // Serial exclusive prefix sum in thread 0 (batch is tiny in practice).
    if (tid == 0) {
        int acc = 0;
        for (int b = 0; b < batch; b++) {
            s_base[b] = acc;
            acc += s_cnt[b];
        }
        *d_num_tiles = acc;
    }
    __syncthreads();

    // ---- Each thread writes the tile_meta records for its batch ----
    if (tid < batch && my_cnt > 0) {
        int q_start  = qo_indptr[tid];
        int q_end    = qo_indptr[tid + 1];
        int kv_start = kv_indptr[tid];
        int kv_end   = kv_indptr[tid + 1];
        int seq_q    = q_end - q_start;
        int num_kv   = kv_end - kv_start;
        int delta    = num_kv - seq_q;
        int base = s_base[tid];
        for (int t = 0; t < my_cnt; t++) {
            int* m = tile_meta + (base + t) * 6;
            int qlen = (seq_q - t * BRQ < BRQ) ? (seq_q - t * BRQ) : BRQ;
            m[0] = q_start + t * BRQ;   // q_start_global
            m[1] = qlen;                // q_len
            m[2] = t * BRQ;             // q_start_local
            m[3] = kv_start;            // kv_start
            m[4] = num_kv;              // num_kv
            m[5] = delta;               // delta
        }
    }
}

std::vector<torch::Tensor> run(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor qo_indptr,
    torch::Tensor kv_indptr,
    torch::Tensor kv_indices,
    double sm_scale)
{
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    if (!q.is_contiguous())          q          = q.contiguous();
    if (!k_cache.is_contiguous())    k_cache    = k_cache.contiguous();
    if (!v_cache.is_contiguous())    v_cache    = v_cache.contiguous();
    if (!qo_indptr.is_contiguous())  qo_indptr  = qo_indptr.contiguous();
    if (!kv_indptr.is_contiguous())  kv_indptr  = kv_indptr.contiguous();
    if (!kv_indices.is_contiguous()) kv_indices = kv_indices.contiguous();

    const int total_q = (int)q.size(0);
    auto f32_opts  = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    auto bf16_opts = torch::TensorOptions().dtype(torch::kBFloat16).device(q.device());
    auto output = torch::empty({total_q, NUM_QO_HEADS, HEAD_DIM}, bf16_opts);
    auto lse    = torch::full({total_q, NUM_QO_HEADS}, -INFINITY, f32_opts);

    const int batch = (int)qo_indptr.size(0) - 1;
    if (total_q == 0 || batch <= 0) {
        return {output, lse};
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // ---- Build tile_meta entirely on GPU (no D2H/H2D sync) ----
    // num_tiles <= total_q always, so allocate total_q*6 ints (upper bound).
    auto meta_gpu    = torch::empty({(long long)total_q * 6},
                                    torch::TensorOptions().dtype(torch::kInt).device(q.device()));
    auto num_tiles_t = torch::zeros({1},
                                    torch::TensorOptions().dtype(torch::kInt).device(q.device()));

    const int scan_threads = batch < 1024 ? batch : 1024;
    const size_t scan_smem = (size_t)batch * sizeof(int) * 2;   // s_cnt + s_base
    build_tile_meta_kernel<<<1, scan_threads, scan_smem, stream>>>(
        qo_indptr.data_ptr<int>(),
        kv_indptr.data_ptr<int>(),
        meta_gpu.data_ptr<int>(),
        num_tiles_t.data_ptr<int>(),
        batch);

    // ---- Set MaxDynamicSharedMemorySize once per process ----
    // smem = K tile + V tile + 4 per-warp Q tiles, rows padded to SMEM_STRIDE.
    const size_t smem_size =
        (size_t)(2 * T_PIPE + 4 * BRQ) * SMEM_STRIDE * sizeof(__nv_bfloat16);
    static bool s_smem_attr_set = false;
    if (!s_smem_attr_set) {
        cudaFuncSetAttribute(gqa_paged_prefill_mma,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);
        s_smem_attr_set = true;
    }

    // gridDim.y = total_q (safe upper bound for num_tiles). The kernel reads
    // the real num_tiles from *d_num_tiles (device memory) and early-returns
    // any CTA whose blockIdx.y >= num_tiles — so NO host sync is needed.
    dim3 grid(NUM_KV_HEADS, total_q);
    dim3 block(BLOCK_THREADS);
    gqa_paged_prefill_mma<<<grid, block, smem_size, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr<at::BFloat16>()),
        kv_indices.data_ptr<int>(),
        (float)sm_scale,
        meta_gpu.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        lse.data_ptr<float>(),
        num_tiles_t.data_ptr<int>(),
        total_q);

    return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "GQA Paged Prefill (causal) r1-s2: mma.sync m16n8k16 (bf16->fp32) "
          "Q@K^T and P@V, BRQ=16, T_PIPE=16, online softmax, base-2 LSE, "
          "per-row causal mask, paged KV page_size=1");
}
