/*
 * MLA Paged Prefill Causal Attention  --  tensor-core candidate (attempt 0)
 *
 *   num_qo_heads = 16, head_dim_ckv = 512, head_dim_kpe = 64, page_size = 1
 *
 *   s[h][j] = (q_nope[i,h,:] . Kc[j,:] + q_pe[i,h,:] . Kp[j,:]) * sm_scale
 *   causal : s[h][j] = -inf for j > (kv_len - q_len) + i
 *   out[i,h,:] = softmax_j(s[h][:]) @ Kc     (bf16)
 *   lse[i,h]   = (row_max + log(row_sum)) * log2(e)   (fp32, 2-base)
 *
 * Design notes (what differs from the FFMA baseline):
 *
 *  D1  One CTA per query token, holding ALL 16 heads.  The 16 heads are the M
 *      dimension of both GEMMs, which is exactly one mma.m16n8k16 M-tile, so
 *      (a) the KV tile is gathered once for all heads instead of 16 times,
 *      (b) the causal bound is a single per-CTA scalar (all 16 rows share it)
 *          and no row of the M-tile is ever a masked dummy.
 *
 *  D2  Both GEMMs run on mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32.
 *        score : A = Q[16][576] row-major, B = K[576][32] col-major, and
 *                "B col-major" is literally the [token][dim] KV tile, so B
 *                fragments come from plain ldmatrix.x4.
 *        out   : A = P[16][32] row-major, B = V[32][512] col-major, which is
 *                the transpose of the same [token][dim] tile, so B fragments
 *                come from ldmatrix.x4.trans.  No smem transpose pass.
 *
 *  D3  Warp split: 8 warps.  Scores split the K=576 reduction 4 ways (144 dims
 *      = 9 k-steps per warp, so Q's A-fragments stay in registers for the whole
 *      KV loop) x tokens 2 ways; partials are summed through smem.  The PV
 *      GEMM splits the N=512 output dims 8 ways, so each warp carries only
 *      8 n-tiles * 4 = 32 fp32 accumulators.
 *
 *  D4  No device->host sync in the launcher.  grid.x = total_q is known from
 *      q_nope.size(0), and every (token, head) row is written by exactly one
 *      CTA, so the outputs are allocated with empty() instead of zeros().
 *
 *  D5  Smem strides are padded for conflict-free access:
 *        sQ/sK stride 584 halves -> 292 words, 292 % 32 == 4, so the 8 row
 *          addresses of an ldmatrix phase tile all 32 banks exactly once
 *          (this holds for the .trans form too).
 *        sP stride 40 halves -> 20 words: rows 0..7 start at banks
 *          0,20,8,28,16,4,24,12, again tiling all 32 banks.
 *        sS stride 36 floats -> 4 words per row, so the float2 fragment
 *          stores of the score partials do not collide.
 *
 *  Total dynamic smem = 66752 B (needs the MaxDynamicSharedMemorySize opt-in).
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <tuple>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// ============================================================
// Compile-time shape constants
// ============================================================
#define MLA_H        16                 // num_qo_heads == M rows of both GEMMs
#define MLA_DC       512                // head_dim_ckv
#define MLA_DP       64                 // head_dim_kpe
#define MLA_D        576                // MLA_DC + MLA_DP : score reduction dim
#define MLA_TKV      32                 // KV tile (tokens)
#define MLA_WARPS    8
#define MLA_THREADS  (MLA_WARPS * 32)   // 256
#define MLA_SD       584                // padded row stride, halves, for sQ/sK
#define MLA_SP       40                 // padded row stride, halves, for sP
#define MLA_SS       36                 // padded row stride, floats, for sS
#define MLA_KCHUNK   144                // MLA_D / 4 : score K dims per warp
#define MLA_KSTEPS   9                  // MLA_KCHUNK / 16
#define MLA_PVN      64                 // MLA_DC / MLA_WARPS : out dims per warp
#define MLA_PVNT     8                   // MLA_PVN / 8 : n-tiles per warp

static constexpr float MLA_LOG2E = 1.4426950408889634f;

// Shared-memory block offsets (bytes); every entry is 16 B aligned.
#define MLA_OFF_Q     0
#define MLA_OFF_K     (MLA_OFF_Q + MLA_H   * MLA_SD * 2)      // 18688
#define MLA_OFF_P     (MLA_OFF_K + MLA_TKV * MLA_SD * 2)      // 56064
#define MLA_OFF_S     (MLA_OFF_P + MLA_H   * MLA_SP * 2)      // 57344
#define MLA_OFF_MAX   (MLA_OFF_S + 4 * MLA_H * MLA_SS * 4)    // 66560
#define MLA_OFF_SUM   (MLA_OFF_MAX + MLA_H * 4)
#define MLA_OFF_ALPHA (MLA_OFF_SUM + MLA_H * 4)
#define MLA_SMEM      (MLA_OFF_ALPHA + MLA_H * 4)             // 66752

// ============================================================
// PTX helpers
// ============================================================
__device__ __forceinline__ uint32_t mla_sptr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// 4 x 8x8 b16 tiles -> 4 registers.  Lanes 8m..8m+7 supply the row addresses
// of tile m; lane L then owns tile-row (L>>2), tile-cols 2*(L&3) and +1.
__device__ __forceinline__ void mla_ldm_x4(uint32_t& r0, uint32_t& r1,
                                           uint32_t& r2, uint32_t& r3,
                                           uint32_t addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
#else
    r0 = r1 = r2 = r3 = 0u; (void)addr;
#endif
}

// Same, but each 8x8 tile is transposed on the way out: lane L owns
// tile-col (L>>2) and tile-rows 2*(L&3), +1.  This is what turns the
// [token][dim] KV tile into the col-major B operand of the PV GEMM.
__device__ __forceinline__ void mla_ldm_x4_trans(uint32_t& r0, uint32_t& r1,
                                                 uint32_t& r2, uint32_t& r3,
                                                 uint32_t addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
#else
    r0 = r1 = r2 = r3 = 0u; (void)addr;
#endif
}

// D += A[16x16] * B[16x8], bf16 in / fp32 accumulate.
__device__ __forceinline__ void mla_mma(float& d0, float& d1, float& d2, float& d3,
                                        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
#endif
}

// ============================================================
// Kernel: one CTA == one query token == 16 head-rows
// ============================================================
__global__ void __launch_bounds__(MLA_THREADS, 2)
mla_paged_prefill_kernel(
    const __nv_bfloat16* __restrict__ q_nope,     // [total_q, 16, 512]
    const __nv_bfloat16* __restrict__ q_pe,       // [total_q, 16,  64]
    const __nv_bfloat16* __restrict__ ckv_cache,  // [num_pages, 512]
    const __nv_bfloat16* __restrict__ kpe_cache,  // [num_pages,  64]
    const int* __restrict__ qo_indptr,            // [batch+1]
    const int* __restrict__ kv_indptr,            // [batch+1]
    const int* __restrict__ kv_indices,           // [num_kv_indices]
    const float sm_scale,
    const int   batch_size,
    const int   total_q,
    __nv_bfloat16* __restrict__ out,              // [total_q, 16, 512]
    float* __restrict__ lse)                      // [total_q, 16]
{
    // Reverse the block -> token map: within a long sequence the causal work
    // grows with the token index, so the heaviest CTAs must be dispatched
    // first or they land in the tail.
    const int token = total_q - 1 - (int)blockIdx.x;
    if (token < 0) return;

    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // Which sequence owns this token.  Picking the *last* b with
    // qo_indptr[b] <= token is also correct when empty sequences are present.
    int lo = 0, hi = batch_size;
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (qo_indptr[mid] <= token) lo = mid; else hi = mid;
    }
    const int q_start  = qo_indptr[lo];
    const int q_len    = qo_indptr[lo + 1] - q_start;
    const int kv_start = kv_indptr[lo];
    const int kv_len   = kv_indptr[lo + 1] - kv_start;
    const int qi       = token - q_start;

    // Causal bound: j <= (kv_len - q_len) + qi, i.e. kv_end tokens are live.
    const int kv_end = min(kv_len, kv_len - q_len + qi + 1);

    __nv_bfloat16* obase = out + (long long)token * (MLA_H * MLA_DC);

    // Nothing attends here (empty KV, or q_len > kv_len at the head of the
    // chunk).  Write the row explicitly -- the outputs are empty(), not zeros().
    if (kv_end <= 0) {
        const uint4 z = make_uint4(0u, 0u, 0u, 0u);
        for (int g = tid; g < (MLA_H * MLA_DC) / 8; g += MLA_THREADS)
            *reinterpret_cast<uint4*>(obase + g * 8) = z;
        if (tid < MLA_H) lse[(long long)token * MLA_H + tid] = -INFINITY;
        return;
    }

    extern __shared__ __align__(16) char mla_smem[];
    __nv_bfloat16* sQ     = reinterpret_cast<__nv_bfloat16*>(mla_smem + MLA_OFF_Q);
    __nv_bfloat16* sK     = reinterpret_cast<__nv_bfloat16*>(mla_smem + MLA_OFF_K);
    __nv_bfloat16* sP     = reinterpret_cast<__nv_bfloat16*>(mla_smem + MLA_OFF_P);
    float*         sS     = reinterpret_cast<float*>(mla_smem + MLA_OFF_S);
    float*         sMax   = reinterpret_cast<float*>(mla_smem + MLA_OFF_MAX);
    float*         sSum   = reinterpret_cast<float*>(mla_smem + MLA_OFF_SUM);
    float*         sAlpha = reinterpret_cast<float*>(mla_smem + MLA_OFF_ALPHA);

    // ---- stage Q for all 16 heads: [16][576] = nope ++ pe -----------------
    {
        const __nv_bfloat16* qn = q_nope + (long long)token * (MLA_H * MLA_DC);
        #pragma unroll
        for (int it = 0; it < 4; ++it) {
            const int g  = it * MLA_THREADS + tid;     // 0..1023 uint4 of [16][512]
            const int h  = g >> 6;
            const int c8 = (g & 63) * 8;
            *reinterpret_cast<uint4*>(&sQ[h * MLA_SD + c8]) =
                *reinterpret_cast<const uint4*>(qn + g * 8);
        }
        if (tid < 128) {
            const __nv_bfloat16* qp = q_pe + (long long)token * (MLA_H * MLA_DP);
            const int h  = tid >> 3;
            const int c8 = (tid & 7) * 8;
            *reinterpret_cast<uint4*>(&sQ[h * MLA_SD + MLA_DC + c8]) =
                *reinterpret_cast<const uint4*>(qp + tid * 8);
        }
        if (tid < MLA_H) { sMax[tid] = -INFINITY; sSum[tid] = 0.0f; }
    }
    __syncthreads();

    // ldmatrix address patterns (per lane).
    //  pattern A -- tiles are {rows 0-7, rows 8-15} x {cols +0, cols +8}:
    //               used for the score A operand (Q), the PV A operand (P)
    //               and the PV B operand (V, transposed).
    //  pattern B -- tiles are {cols +0, cols +8} x {rows 0-7, rows 8-15}:
    //               used for the score B operand (K).
    const int paRow = (lane & 7) + (((lane >> 3) & 1) << 3);
    const int paOff = (lane >> 4) << 3;
    const int pbRow = (lane & 7) + ((lane >> 4) << 3);
    const int pbOff = ((lane >> 3) & 1) << 3;

    const int kc     = warp & 3;            // score K chunk: 144 dims
    const int nc     = warp >> 2;           // score token chunk: 16 tokens
    const int tbase  = nc * 16;
    const int pvDim0 = warp * MLA_PVN;      // PV output dim chunk: 64 dims

    // Q A-fragments live in registers across the whole KV loop (36 regs).
    uint32_t qa[MLA_KSTEPS][4];
    #pragma unroll
    for (int ks = 0; ks < MLA_KSTEPS; ++ks) {
        const int kbase = kc * MLA_KCHUNK + ks * 16;
        mla_ldm_x4(qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3],
                   mla_sptr(&sQ[paRow * MLA_SD + kbase + paOff]));
    }

    float acc[MLA_PVNT][4];
    #pragma unroll
    for (int nt = 0; nt < MLA_PVNT; ++nt) {
        acc[nt][0] = 0.0f; acc[nt][1] = 0.0f; acc[nt][2] = 0.0f; acc[nt][3] = 0.0f;
    }

    const int ntiles = (kv_end + MLA_TKV - 1) / MLA_TKV;

    for (int tile = 0; tile < ntiles; ++tile) {
        const int ts    = tile * MLA_TKV;
        const int valid = min(MLA_TKV, kv_end - ts);

        __syncthreads();   // previous tile's readers of sK / sP are done

        // ---- paged gather of the KV tile (page_size == 1 -> page == token) --
        // 64 consecutive lanes cover one 1024 B ckv row, so each row is a
        // fully coalesced burst even though the row addresses are random.
        #pragma unroll 4
        for (int it = 0; it < 8; ++it) {
            const int g  = it * MLA_THREADS + tid;
            const int t  = g >> 6;
            const int c8 = (g & 63) * 8;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (t < valid) {
                const int page = kv_indices[kv_start + ts + t];
                v = *reinterpret_cast<const uint4*>(ckv_cache + (long long)page * MLA_DC + c8);
            }
            *reinterpret_cast<uint4*>(&sK[t * MLA_SD + c8]) = v;
        }
        {
            const int t  = tid >> 3;
            const int c8 = (tid & 7) * 8;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (t < valid) {
                const int page = kv_indices[kv_start + ts + t];
                v = *reinterpret_cast<const uint4*>(kpe_cache + (long long)page * MLA_DP + c8);
            }
            *reinterpret_cast<uint4*>(&sK[t * MLA_SD + MLA_DC + c8]) = v;
        }
        __syncthreads();

        // ---- score GEMM: S[16][32] partials over this warp's 144 K dims ----
        if (tbase < valid) {
            float s0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float s1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int ks = 0; ks < MLA_KSTEPS; ++ks) {
                const int kbase = kc * MLA_KCHUNK + ks * 16;
                uint32_t b0, b1, b2, b3;
                mla_ldm_x4(b0, b1, b2, b3,
                           mla_sptr(&sK[(tbase + pbRow) * MLA_SD + kbase + pbOff]));
                mla_mma(s0[0], s0[1], s0[2], s0[3],
                        qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], b0, b1);
                mla_mma(s1[0], s1[1], s1[2], s1[3],
                        qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], b2, b3);
            }
            float* sSk = sS + kc * (MLA_H * MLA_SS);
            const int r0 = lane >> 2, r1 = r0 + 8;
            const int cb = tbase + ((lane & 3) << 1);
            *reinterpret_cast<float2*>(&sSk[r0 * MLA_SS + cb])     = make_float2(s0[0], s0[1]);
            *reinterpret_cast<float2*>(&sSk[r1 * MLA_SS + cb])     = make_float2(s0[2], s0[3]);
            *reinterpret_cast<float2*>(&sSk[r0 * MLA_SS + cb + 8]) = make_float2(s1[0], s1[1]);
            *reinterpret_cast<float2*>(&sSk[r1 * MLA_SS + cb + 8]) = make_float2(s1[2], s1[3]);
        }
        __syncthreads();

        // ---- cross-warp sum, causal mask, online softmax --------------------
        // One row of 16 lanes per head, so every head lives inside one warp
        // and the running (max, sum) update needs no extra barrier.
        {
            const int row = tid >> 4;        // head
            const int c   = tid & 15;
            const int col = c << 1;
            float v0 = 0.0f, v1 = 0.0f;
            #pragma unroll
            for (int k4 = 0; k4 < 4; ++k4) {
                const float2 v = *reinterpret_cast<const float2*>(
                    &sS[k4 * (MLA_H * MLA_SS) + row * MLA_SS + col]);
                v0 += v.x; v1 += v.y;
            }
            v0 *= sm_scale;
            v1 *= sm_scale;
            // Plain assignment, so a skipped (never-written) partial slot can
            // never leak a NaN into the softmax.
            if (ts + col     >= kv_end) v0 = -INFINITY;
            if (ts + col + 1 >= kv_end) v1 = -INFINITY;

            float m = fmaxf(v0, v1);
            #pragma unroll
            for (int msk = 1; msk < 16; msk <<= 1)
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, msk));

            const float oldm  = sMax[row];
            const float newm  = fmaxf(oldm, m);          // finite: col 0 is live
            const float alpha = (oldm == -INFINITY) ? 0.0f : __expf(oldm - newm);
            const float p0    = __expf(v0 - newm);
            const float p1    = __expf(v1 - newm);

            float ps = p0 + p1;
            #pragma unroll
            for (int msk = 1; msk < 16; msk <<= 1)
                ps += __shfl_xor_sync(0xffffffffu, ps, msk);

            if (c == 0) {
                sMax[row]   = newm;
                sSum[row]   = sSum[row] * alpha + ps;
                sAlpha[row] = alpha;
            }
            __nv_bfloat162 pp;
            pp.x = __float2bfloat16(p0);
            pp.y = __float2bfloat16(p1);
            *reinterpret_cast<__nv_bfloat162*>(&sP[row * MLA_SP + col]) = pp;
        }
        __syncthreads();

        // ---- PV GEMM: O[16][64 dims of this warp] += P[16][32] * V[32][.] ---
        {
            const int r0 = lane >> 2;
            const float a0 = sAlpha[r0];
            const float a1 = sAlpha[r0 + 8];
            #pragma unroll
            for (int nt = 0; nt < MLA_PVNT; ++nt) {
                acc[nt][0] *= a0; acc[nt][1] *= a0;
                acc[nt][2] *= a1; acc[nt][3] *= a1;
            }
            #pragma unroll
            for (int kt = 0; kt < MLA_TKV; kt += 16) {
                if (kt >= valid) break;      // uniform: skip the dead half tile
                uint32_t p0, p1, p2, p3;
                mla_ldm_x4(p0, p1, p2, p3,
                           mla_sptr(&sP[paRow * MLA_SP + kt + paOff]));
                #pragma unroll
                for (int j = 0; j < MLA_PVNT / 2; ++j) {
                    const int dim = pvDim0 + j * 16;
                    uint32_t v0, v1, v2, v3;
                    mla_ldm_x4_trans(v0, v1, v2, v3,
                                     mla_sptr(&sK[(kt + paRow) * MLA_SD + dim + paOff]));
                    mla_mma(acc[2 * j][0], acc[2 * j][1], acc[2 * j][2], acc[2 * j][3],
                            p0, p1, p2, p3, v0, v1);
                    mla_mma(acc[2 * j + 1][0], acc[2 * j + 1][1],
                            acc[2 * j + 1][2], acc[2 * j + 1][3],
                            p0, p1, p2, p3, v2, v3);
                }
            }
        }
    }

    // ---- epilogue: normalise, store bf16 output, store 2-base LSE ----------
    {
        const int r0 = lane >> 2, r1 = r0 + 8;
        const float sum0 = sSum[r0], sum1 = sSum[r1];
        const float inv0 = (sum0 > 0.0f) ? (1.0f / sum0) : 0.0f;
        const float inv1 = (sum1 > 0.0f) ? (1.0f / sum1) : 0.0f;
        #pragma unroll
        for (int nt = 0; nt < MLA_PVNT; ++nt) {
            const int dim = pvDim0 + nt * 8 + ((lane & 3) << 1);
            __nv_bfloat162 o0, o1;
            o0.x = __float2bfloat16(acc[nt][0] * inv0);
            o0.y = __float2bfloat16(acc[nt][1] * inv0);
            o1.x = __float2bfloat16(acc[nt][2] * inv1);
            o1.y = __float2bfloat16(acc[nt][3] * inv1);
            *reinterpret_cast<__nv_bfloat162*>(obase + r0 * MLA_DC + dim) = o0;
            *reinterpret_cast<__nv_bfloat162*>(obase + r1 * MLA_DC + dim) = o1;
        }
        if (warp == 0 && lane < MLA_H) {
            const float sm = sMax[lane];
            const float su = sSum[lane];
            lse[(long long)token * MLA_H + lane] =
                (su > 0.0f) ? ((sm + __logf(su)) * MLA_LOG2E) : -INFINITY;
        }
    }
}

// ============================================================
// Host entry point -- no device->host synchronisation anywhere.
// ============================================================
std::tuple<torch::Tensor, torch::Tensor> mla_paged_prefill_run(
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor ckv_cache,
    torch::Tensor kpe_cache,
    torch::Tensor qo_indptr,
    torch::Tensor kv_indptr,
    torch::Tensor kv_indices,
    double sm_scale)
{
    TORCH_CHECK(q_nope.is_cuda(),    "q_nope must be a CUDA tensor");
    TORCH_CHECK(q_nope.dim() == 3,   "q_nope must be [total_q, 16, 512]");
    TORCH_CHECK(q_nope.scalar_type()    == torch::kBFloat16, "q_nope must be bf16");
    TORCH_CHECK(q_pe.scalar_type()      == torch::kBFloat16, "q_pe must be bf16");
    TORCH_CHECK(ckv_cache.scalar_type() == torch::kBFloat16, "ckv_cache must be bf16");
    TORCH_CHECK(kpe_cache.scalar_type() == torch::kBFloat16, "kpe_cache must be bf16");
    TORCH_CHECK(qo_indptr.scalar_type()  == torch::kInt32, "qo_indptr must be int32");
    TORCH_CHECK(kv_indptr.scalar_type()  == torch::kInt32, "kv_indptr must be int32");
    TORCH_CHECK(kv_indices.scalar_type() == torch::kInt32, "kv_indices must be int32");
    TORCH_CHECK(q_nope.size(1) == MLA_H,  "num_qo_heads must be 16");
    TORCH_CHECK(q_nope.size(2) == MLA_DC, "head_dim_ckv must be 512");
    TORCH_CHECK(q_pe.size(2)   == MLA_DP, "head_dim_kpe must be 64");
    TORCH_CHECK(ckv_cache.size(1) == 1,   "page_size must be 1");
    TORCH_CHECK(q_nope.is_contiguous() && q_pe.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(ckv_cache.is_contiguous() && kpe_cache.is_contiguous(),
                "KV cache must be contiguous");

    const int total_q    = (int)q_nope.size(0);
    const int batch_size = (int)qo_indptr.size(0) - 1;

    auto out = torch::empty({total_q, MLA_H, MLA_DC},
        torch::TensorOptions().dtype(torch::kBFloat16).device(q_nope.device()));
    auto lse = torch::empty({total_q, MLA_H},
        torch::TensorOptions().dtype(torch::kFloat32).device(q_nope.device()));

    if (total_q == 0 || batch_size <= 0) return std::make_tuple(out, lse);

    // 66752 B of dynamic smem needs the opt-in; do it once per process.
    static const bool smem_opted_in = [] {
        cudaFuncSetAttribute(mla_paged_prefill_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)MLA_SMEM);
        return true;
    }();
    (void)smem_opted_in;

    auto stream = at::cuda::getCurrentCUDAStream();

    // grid.x == total_q: one CTA per query token, all 16 heads.  Every output
    // row is covered exactly once, which is what lets the tensors be empty().
    mla_paged_prefill_kernel<<<dim3(total_q), dim3(MLA_THREADS), MLA_SMEM, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr()),
        reinterpret_cast<const int*>(qo_indptr.data_ptr()),
        reinterpret_cast<const int*>(kv_indptr.data_ptr()),
        reinterpret_cast<const int*>(kv_indices.data_ptr()),
        (float)sm_scale,
        batch_size,
        total_q,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        reinterpret_cast<float*>(lse.data_ptr()));

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(out, lse);
}

// The harness builds kernel.cu on its own (entry point kernel.cu::run), so the
// module lives here by default.  Building kernel_binding.cpp alongside it with
// -DMLA_EXTERNAL_BINDING moves the module there instead.
#ifndef MLA_EXTERNAL_BINDING
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &mla_paged_prefill_run,
          "MLA paged prefill causal attention (h16, ckv512, kpe64, page_size 1)",
          py::arg("q_nope"), py::arg("q_pe"),
          py::arg("ckv_cache"), py::arg("kpe_cache"),
          py::arg("qo_indptr"), py::arg("kv_indptr"), py::arg("kv_indices"),
          py::arg("sm_scale"));
}
#endif

// kernels/kernel_binding.cpp
//
// pybind11 surface for the MLA paged-prefill kernel.
//
// kernel.cu already carries a PYBIND11_MODULE, because the sol-execbench
// harness compiles that single translation unit with entry point
// kernel.cu::run.  Defining a second module here unconditionally would give
// the extension two PyInit_ symbols, so the two are mutually exclusive:
//
//   * default flags        -> kernel.cu owns the module, this file only
//                             declares the launcher (safe to co-compile).
//   * -DMLA_EXTERNAL_BINDING -> kernel.cu stays silent, this file owns it.

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <tuple>

namespace py = pybind11;

// Defined in kernel.cu.
std::tuple<torch::Tensor, torch::Tensor> mla_paged_prefill_run(
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor ckv_cache,
    torch::Tensor kpe_cache,
    torch::Tensor qo_indptr,
    torch::Tensor kv_indptr,
    torch::Tensor kv_indices,
    double sm_scale);

// Tensor/dtype/shape validation lives inside mla_paged_prefill_run (TORCH_CHECK
// on dtype, layout, the four fixed constants and page_size == 1) so that both
// binding paths enforce exactly the same contract.
std::tuple<torch::Tensor, torch::Tensor> run_forward(
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor ckv_cache,
    torch::Tensor kpe_cache,
    torch::Tensor qo_indptr,
    torch::Tensor kv_indptr,
    torch::Tensor kv_indices,
    double sm_scale)
{
    return mla_paged_prefill_run(q_nope, q_pe, ckv_cache, kpe_cache,
                                 qo_indptr, kv_indptr, kv_indices, sm_scale);
}

#ifdef MLA_EXTERNAL_BINDING
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run_forward,
          "MLA paged prefill causal attention (h16, ckv512, kpe64, page_size 1)",
          py::arg("q_nope"), py::arg("q_pe"),
          py::arg("ckv_cache"), py::arg("kpe_cache"),
          py::arg("qo_indptr"), py::arg("kv_indptr"), py::arg("kv_indices"),
          py::arg("sm_scale"));
}
#endif
