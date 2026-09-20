// kernel.cu — GQA paged decode (num_qo_heads=32, num_kv_heads=8, head_dim=128, page_size=1)
// for NVIDIA B300 / sm_103. Attempt 4: stabilization-first rebuild.
//
// Structure: the attempt-0 two-kernel FlashDecoding skeleton — a split-K main kernel
// (grid (S, B*8), 128 threads) plus a separate LSE-weighted reduce kernel (grid (B*8),
// 32 threads). No device-side arrival counters, no persistent workspace, no
// cp.async.bulk.prefetch PTX (the features that made attempts 2/3 unverifiable).
// Layered on top, only the low-risk high-return pieces:
//   * Page-id smem ring (3 slots x 64 ids): each tile's kv_indices are read once as a
//     coalesced int4-per-lane load (scalar clamped fallback otherwise) and broadcast
//     from smem in issue_tile — the 16x-redundant global kv_indices reads on the
//     random-gather critical path are gone.
//   * Tiny-workload warp kernel (gqa_paged_decode_small) for nkv < B*64: one warp per
//     q head, __ldg 8-B row slices, fp32 FMA dot with a 32-lane butterfly, base-2 online
//     softmax, direct bf16 out + 2-based lse. Dispatched purely from host-known shapes.
//   * Split heuristic retuned for 148 SMs: S = clamp(ceil(444/(B*8)), 1,
//     ceil(nkv/(B*chunk))) with whole-tile chunk quantization (64-token chunks, 128-token
//     chunks when B*8 >= 148), so kv_indices reads stay stride-1 and L2-local.
//
// Core math: cp.async.cg 16-B gathers with 16 consecutive lanes forming each full 256-B
// (token, kv_head) row, KSTAGE=2 double-buffered XOR-swizzled smem tiles, and
// mma.m16n8k8 bf16->f32 tensor cores for BOTH QK^T and O += P*V.
//
// Why m16n8k8 and not m16n8k16: the m16n8k8 fragment layouts are the fully specified
// base case (A: a0 = A[g][2c+{0,1}], a1 = A[g+8][2c+{0,1}]; B: b0 = {B[2c][g],
// B[2c+1][g]}; D: d0,d1 = D[g][2c+{0,1}], d2,d3 = D[g+8][2c+{0,1}]), so every
// register->element mapping in this file is unambiguous and directly anchored to the
// documented base case. The k16 A-operand quadrant ordering is the one layout that
// cannot be anchored with certainty, and it feeds BOTH the Q ldmatrix and the in-register
// P packing — a silent numerics killer. This kernel is memory-bound (~4 FLOP per KV
// byte streamed; ~100 effective TFLOP/s at the 6.434 TB/s floor vs ~1658 measured), so
// the 2x mma instruction count costs essentially nothing, while removing the largest
// correctness risk. Stabilization first: an unverified aggressive feature is worth zero.
//
// Softmax is in base 2: qk_scale = sm_scale*log2(e) is folded in on the host, exp2f
// yields the probabilities, and lse = m* + log2(lsum) falls out directly in the
// reference's 2-based convention (reference.py:54) with no post-hoc log conversion.
// Empty chunks exit before touching the big smem; empty sequences give out=0 / lse=-inf,
// exactly like the reference.

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <vector>

namespace gqa_ps1 {

constexpr int HEAD_DIM    = 128;
constexpr int NUM_KVH     = 8;
constexpr int GQA         = 4;               // 32 qo heads / 8 kv heads
constexpr int T_TILE      = 64;              // tokens per KV tile
constexpr int NUM_WARPS   = 4;               // each warp owns a 16-token slice
constexpr int KSTAGE      = 2;               // cp.async double buffer (KV tiles)
constexpr int IDX_RING    = 3;               // page-id ring: one tile deeper than the KV pipe
constexpr int NTHREADS    = 128;
constexpr int SMEM_Q      = 16 * 256;        // 16 head-rows (4 real + 12 zero pad) x 128 bf16
constexpr int KV_STAGE    = T_TILE * 256;    // 64 token rows x 256 B, one cp.async stage
constexpr int SMEM_IDX    = IDX_RING * T_TILE * (int)sizeof(int);  // 768 B page-id ring
constexpr int SMEM_BYTES  = SMEM_Q + 2 * KSTAGE * KV_STAGE + SMEM_IDX;  // 70400 B
constexpr int TARGET_CTAS = 444;             // 148 SMs x ~3 resident CTAs (sm_103)
constexpr float LOG2E     = 1.44269504088896340736f;
constexpr int PSTRIDE     = 4 * 130;         // fp32 partial per (b, kv_head, split)

// 16-B-chunk XOR swizzle over a 256 B row: chunk c of row r lands at (c ^ (r & 15)).
__device__ __forceinline__ int kv_swz(int row, int chunk) {
  return row * 256 + ((chunk ^ (row & 15)) << 4);
}

__device__ __forceinline__ void cp_async16(void* smem_dst, const void* gmem_src) {
  unsigned s = (unsigned)__cvta_generic_to_shared(smem_dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem_src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t r[4], const void* p) {
  unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldmatrix_x4_t(uint32_t r[4], const void* p) {
  unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

// D(16x8, f32) += A(16x8, bf16) * B(8x8, bf16), mma.m16n8k8.row.col.
// Fragment layout (the fully specified base case): thread t with g = t>>2, c = t&3
//   A: a0 = {A[g][2c], A[g][2c+1]}, a1 = {A[g+8][2c], A[g+8][2c+1]}
//   B: b0 = {B[2c][g], B[2c+1][g]}
//   D: d0,d1 = {D[g][2c], D[g][2c+1]}, d2,d3 = {D[g+8][2c], D[g+8][2c+1]}
__device__ __forceinline__ void mma_k8(float d[4], const uint32_t a[2], uint32_t b) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(b));
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

} // namespace gqa_ps1

// ---------------------------------------------------------------------------
// Main split-K decode kernel. grid = (num_splits, batch*8), block = 128.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(128, 2)
gqa_paged_decode_splitk(
    const __nv_bfloat16* __restrict__ q,        // [B, 32, 128]
    const __nv_bfloat16* __restrict__ k_cache,  // [P, 1, 8, 128]
    const __nv_bfloat16* __restrict__ v_cache,  // [P, 1, 8, 128]
    const int* __restrict__ kv_indptr,          // [B+1]
    const int* __restrict__ kv_indices,         // [nkv]
    __nv_bfloat16* __restrict__ out,            // [B, 32, 128] (only when S == 1)
    float* __restrict__ lse,                    // [B, 32]      (only when S == 1)
    float* __restrict__ partial,                // [B*8*S, 4, 130] (only when S > 1)
    int num_splits, float qk_scale) {

  using namespace gqa_ps1;

  const int tid  = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int s    = blockIdx.x;             // split id
  const int bh   = blockIdx.y;             // b * 8 + hkv
  const int b    = bh / NUM_KVH;
  const int hkv  = bh % NUM_KVH;

  extern __shared__ __align__(16) char smem_raw[];
  char* sQ   = smem_raw;
  char* sK   = smem_raw + SMEM_Q;
  char* sV   = sK + KSTAGE * KV_STAGE;
  int*  sidx = reinterpret_cast<int*>(sV + KSTAGE * KV_STAGE);
  // Epilogue merge buffers overlay the K stage (free after the tile loop).
  float* mO  = reinterpret_cast<float*>(sK);       // [NUM_WARPS][GQA][128]
  float* mML = mO + NUM_WARPS * GQA * 128;         // [NUM_WARPS][GQA][2]

  const int ps  = kv_indptr[b];
  const int pe  = kv_indptr[b + 1];
  const int len = pe - ps;
  // Contiguous whole-64-token chunk per split: kv_indices reads stay stride-1.
  const int ntiles_seq = (len + T_TILE - 1) / T_TILE;
  const int cl_tiles   = (ntiles_seq + num_splits - 1) / num_splits;
  const int cl  = cl_tiles * T_TILE;
  const int cs  = ps + s * cl;
  const int ce  = min(cs + cl, pe);
  const int clen = ce - cs;                 // tokens owned by this split

  if (cs >= ce) {
    // Empty chunk: CTA-uniform early exit, never touches the big smem.
    if (num_splits == 1) {
      for (int i = tid; i < GQA * HEAD_DIM; i += NTHREADS) {
        int h = i / HEAD_DIM, d = i % HEAD_DIM;
        out[((size_t)b * 32 + hkv * GQA + h) * HEAD_DIM + d] = __float2bfloat16(0.f);
      }
      if (tid < GQA) lse[(size_t)b * 32 + hkv * GQA + tid] = -INFINITY;
    } else if (tid < GQA) {
      float* dst = partial + ((size_t)bh * num_splits + s) * PSTRIDE + tid * 130;
      dst[128] = -INFINITY;   // empty split: reduce skips its O entirely (m == -inf)
      dst[129] = 0.f;
    }
    return;
  }

  // ---- Q staging: rows 0..3 real (heads hkv*4 .. hkv*4+3), rows 4..15 zero ----
  {
    const char* qbase =
        reinterpret_cast<const char*>(q) + ((size_t)(b * 32 + hkv * GQA)) * 256;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      int slot = tid + i * NTHREADS;        // 0..255 = 16 rows x 16 chunks
      int row = slot >> 4, c = slot & 15;
      uint4 v = make_uint4(0u, 0u, 0u, 0u);
      if (row < GQA) v = *reinterpret_cast<const uint4*>(qbase + row * 256 + (c << 4));
      *reinterpret_cast<uint4*>(sQ + kv_swz(row, c)) = v;
    }
  }

  const int ntiles = (clen + T_TILE - 1) / T_TILE;

  // Stage one tile's 64 page ids into ring slot t%IDX_RING. Coalesced int4-per-lane
  // when the range is 16-B aligned and fully in-sequence; scalar clamped fallback
  // otherwise (never reads kv_indices past pe-1, so never past the array end).
  auto stage_idx = [&](int t) {
    int slot = t % IDX_RING;
    int* dst = sidx + slot * T_TILE;
    int gs = cs + t * T_TILE;
    bool fast = ((((uintptr_t)kv_indices) & 15) == 0) && ((gs & 3) == 0) &&
                (gs + T_TILE <= pe);
    if (fast) {
      if (tid < 16) {
        int4 v = reinterpret_cast<const int4*>(kv_indices + gs)[tid];
        dst[tid * 4 + 0] = v.x;
        dst[tid * 4 + 1] = v.y;
        dst[tid * 4 + 2] = v.z;
        dst[tid * 4 + 3] = v.w;
      }
    } else {
      for (int i = tid; i < T_TILE; i += NTHREADS)
        dst[i] = kv_indices[(gs + i < pe) ? (gs + i) : (pe - 1)];
    }
  };

  // Issue one full K+V tile (64 tokens x 512 B) into KV stage st via cp.async.cg.
  // 16 consecutive lanes form one 256 B (token, kv_head) row — every gather is a
  // complete cache-line pair. Tail rows clamp the page id to a valid one (their
  // scores are masked to -inf after QK^T). Ids come from the smem ring.
  auto issue_tile = [&](int t, int st) {
    char* kb = sK + st * KV_STAGE;
    char* vb = sV + st * KV_STAGE;
    const int* idx = sidx + (t % IDX_RING) * T_TILE;
    int tstart = t * T_TILE;
    int tvalid = min(T_TILE, clen - tstart);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int row = (tid >> 4) + (i << 3);      // tid/16 + 8*i  -> rows 0..63
      int c   = tid & 15;
      int rr  = row < tvalid ? row : (tvalid - 1);
      int pg  = idx[rr];
      size_t base = ((size_t)pg * NUM_KVH + hkv) * 256 + (c << 4);
      cp_async16(kb + kv_swz(row, c), reinterpret_cast<const char*>(k_cache) + base);
      cp_async16(vb + kv_swz(row, c), reinterpret_cast<const char*>(v_cache) + base);
    }
    cp_async_commit();
  };

  // Pipeline: idx for tiles 0,1 staged up front; at iteration t we stage idx(t+2)
  // (one ring slot deeper than the KV pipe) and issue the cp.asyncs for tile t+1
  // into KV stage (t+1)%2 — that stage was last computed at iteration t-1 and has
  // been retired by the end-of-iteration __syncthreads, so the schedule is race-free.
  stage_idx(0);
  if (ntiles > 1) stage_idx(1);
  __syncthreads();
  issue_tile(0, 0);

  // ---- accumulators (per thread) ----
  float O[16][4];                            // 16 n8-chunks of head_dim x (2 rows x 2 dims)
#pragma unroll
  for (int nc = 0; nc < 16; ++nc)
#pragma unroll
    for (int j = 0; j < 4; ++j) O[nc][j] = 0.f;
  float m_run = -INFINITY, l_run = 0.f;

  const int tok0 = warp * 16;                // warp's token slice within the tile
  const int g    = lane >> 2;                // head row 0..7 (real only when g < 4)
  const int tp   = (lane & 3) * 2;           // first token of this lane's pair

  for (int t = 0; t < ntiles; ++t) {
    if (t + 2 < ntiles) {
      stage_idx(t + 2);
      __syncthreads();
    }
    if (t + 1 < ntiles) {
      issue_tile(t + 1, (t + 1) % KSTAGE);
      cp_async_wait<KSTAGE - 1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();                          // tile t's cp.asyncs complete and visible

    const char* kbuf = sK + (t % KSTAGE) * KV_STAGE;
    const char* vbuf = sV + (t % KSTAGE) * KV_STAGE;
    int tvalid = min(T_TILE, clen - t * T_TILE);

    // ---- S = Q K^T on tensor cores (m16n8k8, 8 pairs of 8-dim k-blocks) ----
    // A = Q: 16 head-rows x 8 dims per mma. ldmatrix.x4 matrices land as
    //   a[0]=(rows 0-7, dims 8*(2m)+0..7)   -> a0 of k-block 2m
    //   a[1]=(rows 8-15, same dims)         -> a1 of k-block 2m
    //   a[2]=(rows 0-7, dims 8*(2m+1))      -> a0 of k-block 2m+1
    //   a[3]=(rows 8-15, dims 8*(2m+1))     -> a1 of k-block 2m+1
    // B = K^T (k=dim, n=token): non-trans ldmatrix over the token-major K rows gives
    // thread t the pair {K[token g][dim 2c], K[token g][dim 2c+1]} = {B[2c][g],
    // B[2c+1][g]} — exactly the m16n8k8 B fragment:
    //   b[0]=(tokens 0-7,  dims 2m),  b[1]=(tokens 0-7,  dims 2m+1)
    //   b[2]=(tokens 8-15, dims 2m),  b[3]=(tokens 8-15, dims 2m+1)
    float S1[4] = {0.f, 0.f, 0.f, 0.f};      // tokens tok0 .. tok0+7
    float S2[4] = {0.f, 0.f, 0.f, 0.f};      // tokens tok0+8 .. tok0+15
#pragma unroll
    for (int m = 0; m < 8; ++m) {
      uint32_t a[4], b[4];
      {
        int p = lane >> 3, lr = lane & 7;
        int row = lr + ((p & 1) << 3);
        int ch  = 2 * m + (p >> 1);
        ldmatrix_x4(a, sQ + kv_swz(row, ch));
      }
      {
        int p = lane >> 3, lr = lane & 7;
        int row = tok0 + lr + ((p >> 1) << 3);
        int ch  = 2 * m + (p & 1);
        ldmatrix_x4(b, kbuf + kv_swz(row, ch));
      }
      mma_k8(S1, a,     b[0]);               // k-block 2m,   tokens 0..7
      mma_k8(S1, a + 2, b[1]);               // k-block 2m+1, tokens 0..7
      mma_k8(S2, a,     b[2]);               // k-block 2m,   tokens 8..15
      mma_k8(S2, a + 2, b[3]);               // k-block 2m+1, tokens 8..15
    }

    // ---- fragment online softmax, base 2 ----
    // Thread's real head row is g; its 4 tokens are tp, tp+1 (S1) and tp+8, tp+9 (S2).
    int tv = tvalid - tok0;                  // valid tokens within this warp's slice
    float s0 = S1[0] * qk_scale, s1 = S1[1] * qk_scale;
    float s2 = S2[0] * qk_scale, s3 = S2[1] * qk_scale;
    if (tp      >= tv) s0 = -INFINITY;
    if (tp + 1  >= tv) s1 = -INFINITY;
    if (tp + 8  >= tv) s2 = -INFINITY;
    if (tp + 9  >= tv) s3 = -INFINITY;
    float m_t = fmaxf(fmaxf(s0, s1), fmaxf(s2, s3));
    m_t = fmaxf(m_t, __shfl_xor_sync(0xffffffffu, m_t, 1));   // reduce over 4-lane group
    m_t = fmaxf(m_t, __shfl_xor_sync(0xffffffffu, m_t, 2));   // -> per-head max, 16 tokens
    float m_new = fmaxf(m_run, m_t);
    float alpha = (m_run == -INFINITY) ? 0.f : exp2f(m_run - m_new);
    float p0 = (s0 == -INFINITY) ? 0.f : exp2f(s0 - m_new);
    float p1 = (s1 == -INFINITY) ? 0.f : exp2f(s1 - m_new);
    float p2 = (s2 == -INFINITY) ? 0.f : exp2f(s2 - m_new);
    float p3 = (s3 == -INFINITY) ? 0.f : exp2f(s3 - m_new);
    // Rows g+8 of S are pure padding (Q rows 8..15 are zero); their p values only
    // feed discarded O rows. Guard against m_new == -inf producing exp2(+inf).
    float pr0, pr1, pr2, pr3;
    if (m_new == -INFINITY) {
      pr0 = pr1 = pr2 = pr3 = 0.f;
    } else {
      pr0 = exp2f(S1[2] * qk_scale - m_new);
      pr1 = exp2f(S1[3] * qk_scale - m_new);
      pr2 = exp2f(S2[2] * qk_scale - m_new);
      pr3 = exp2f(S2[3] * qk_scale - m_new);
    }
    float l_t = p0 + p1 + p2 + p3;
    l_t += __shfl_xor_sync(0xffffffffu, l_t, 1);
    l_t += __shfl_xor_sync(0xffffffffu, l_t, 2);
    l_run = l_run * alpha + l_t;
    m_run = m_new;
#pragma unroll
    for (int nc = 0; nc < 16; ++nc) {
      O[nc][0] *= alpha; O[nc][1] *= alpha;
      O[nc][2] *= alpha; O[nc][3] *= alpha;
    }

    // ---- O += P V on tensor cores (m16n8k8, 8 pairs of 8-dim n-blocks) ----
    // A = P built in-register from this thread's S fragments (no smem round-trip):
    //   k8 A fragment over tokens 0..7:  a0 = (row g,   k = 2tp+{0,1}) = {p0, p1}
    //                                     a1 = (row g+8, k = 2tp+{0,1}) = {pr0, pr1}
    //   k8 A fragment over tokens 8..15: a0 = {p2, p3}, a1 = {pr2, pr3}
    // B = V (k = 16 tokens, n = 8 dims per mma): ldmatrix.trans over the token-major
    // V rows gives thread t {V[token 2c][dim g], V[token 2c+1][dim g]} = the B
    // fragment; the x4 covers 2 n-blocks x 2 token-halves:
    //   v[0]=(tokens 0-7,  dims 2u),  v[1]=(tokens 0-7,  dims 2u+1)
    //   v[2]=(tokens 8-15, dims 2u),  v[3]=(tokens 8-15, dims 2u+1)
    uint32_t ph0[2] = {pack_bf16(p0, p1), pack_bf16(pr0, pr1)};
    uint32_t ph1[2] = {pack_bf16(p2, p3), pack_bf16(pr2, pr3)};
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      uint32_t v[4];
      int p = lane >> 3, lr = lane & 7;
      int row = tok0 + lr + ((p >> 1) << 3);
      int ch  = 2 * u + (p & 1);
      ldmatrix_x4_t(v, vbuf + kv_swz(row, ch));
      mma_k8(O[2 * u],     ph0, v[0]);
      mma_k8(O[2 * u + 1], ph0, v[1]);
      mma_k8(O[2 * u],     ph1, v[2]);
      mma_k8(O[2 * u + 1], ph1, v[3]);
    }
    __syncthreads();                          // all warps done with stage t%KSTAGE before reissue
  }

  // ---- CTA epilogue: 4-warp LSE merge through smem ----
  // Each warp stores its per-head O (rescaled to its own m) and (m, l).
#pragma unroll
  for (int nc = 0; nc < 16; ++nc) {
    if (g < GQA) {
      mO[(warp * GQA + g) * 128 + nc * 8 + (lane & 3) * 2]     = O[nc][0];
      mO[(warp * GQA + g) * 128 + nc * 8 + (lane & 3) * 2 + 1] = O[nc][1];
    }
  }
  if (g < GQA && (lane & 3) == 0) {
    mML[(warp * GQA + g) * 2]     = m_run;
    mML[(warp * GQA + g) * 2 + 1] = l_run;
  }
  __syncthreads();

  {
    const int h    = tid >> 5;               // 0..3
    const int dloc = tid & 31;
    float mstar = -INFINITY;
#pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w)
      mstar = fmaxf(mstar, mML[(w * GQA + h) * 2]);
    float wgt[NUM_WARPS], lsum = 0.f;
#pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w) {
      float mw = mML[(w * GQA + h) * 2];
      wgt[w] = (mw == -INFINITY || mstar == -INFINITY) ? 0.f : exp2f(mw - mstar);
      lsum += wgt[w] * mML[(w * GQA + h) * 2 + 1];
    }
    const bool direct = (num_splits == 1);
#pragma unroll
    for (int it = 0; it < 4; ++it) {
      int d = dloc + 32 * it;
      float o = 0.f;
#pragma unroll
      for (int w = 0; w < NUM_WARPS; ++w) o += wgt[w] * mO[(w * GQA + h) * 128 + d];
      if (direct) {
        float v = lsum > 0.f ? o * (1.f / lsum) : 0.f;
        out[((size_t)b * 32 + hkv * GQA + h) * HEAD_DIM + d] = __float2bfloat16(v);
      } else {
        partial[((size_t)bh * num_splits + s) * PSTRIDE + h * 130 + d] = o;
      }
    }
    if (dloc == 0) {
      if (direct) {
        lse[(size_t)b * 32 + hkv * GQA + h] =
            lsum > 0.f ? mstar + log2f(lsum) : -INFINITY;
      } else {
        float* dst = partial + ((size_t)bh * num_splits + s) * PSTRIDE + h * 130;
        dst[128] = mstar;
        dst[129] = lsum;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Split-K reduce: one warp per (b, kv_head) combines the S partials.
// O = sum_s w_s O_s / sum_s w_s l_s with w_s = exp2(m_s - m*); lse = m* + log2(lsum).
// Fully empty sequences write out=0 / lse=-inf, matching the reference exactly.
// ---------------------------------------------------------------------------
__global__ void gqa_splitk_reduce(
    const float* __restrict__ partial,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ lse,
    int num_splits) {
  using namespace gqa_ps1;
  const int bh = blockIdx.x;
  const int b = bh / NUM_KVH, hkv = bh % NUM_KVH;
  const int lane = threadIdx.x;
  const float* base = partial + (size_t)bh * num_splits * PSTRIDE;

#pragma unroll
  for (int h = 0; h < GQA; ++h) {
    const float* ph = base + h * 130;
    float mstar = -INFINITY;
    for (int sp = lane; sp < num_splits; sp += 32)
      mstar = fmaxf(mstar, ph[(size_t)sp * PSTRIDE + 128]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      mstar = fmaxf(mstar, __shfl_xor_sync(0xffffffffu, mstar, off));

    float lsum = 0.f;
    float o[4] = {0.f, 0.f, 0.f, 0.f};
    if (mstar > -INFINITY) {
      for (int sp = 0; sp < num_splits; ++sp) {
        const float* ps = ph + (size_t)sp * PSTRIDE;
        float mval = ps[128];
        if (mval == -INFINITY) continue;     // empty split: never touch its O
        float w = exp2f(mval - mstar);
        lsum += w * ps[129];
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] += w * ps[lane * 4 + j];
      }
    }

    const int qh = hkv * GQA + h;
    if (lsum > 0.f) {
      float inv = 1.f / lsum;
#pragma unroll
      for (int j = 0; j < 4; ++j)
        out[((size_t)b * 32 + qh) * HEAD_DIM + lane * 4 + j] =
            __float2bfloat16(o[j] * inv);
      if (lane == 0) lse[(size_t)b * 32 + qh] = mstar + log2f(lsum);
    } else {
      // fully empty sequence: output 0, lse -inf (matches the reference)
#pragma unroll
      for (int j = 0; j < 4; ++j)
        out[((size_t)b * 32 + qh) * HEAD_DIM + lane * 4 + j] =
            __float2bfloat16(0.f);
      if (lane == 0) lse[(size_t)b * 32 + qh] = -INFINITY;
    }
  }
}

// ---------------------------------------------------------------------------
// Tiny-workload kernel: grid (B*8) x 128 threads, one warp per q head.
// Q row slice in registers (4 dims per lane), per-token __ldg 8-B K/V row slices,
// fp32 FMA dot with a 32-lane butterfly reduction, base-2 online softmax, direct
// bf16 out + 2-based lse. Length-agnostic, so the host's mean-length threshold
// (nkv < B*T_TILE) stays correct on uneven batches. Zero dynamic smem.
// ---------------------------------------------------------------------------
__global__ void gqa_paged_decode_small(
    const __nv_bfloat16* __restrict__ q,        // [B, 32, 128]
    const __nv_bfloat16* __restrict__ k_cache,  // [P, 1, 8, 128]
    const __nv_bfloat16* __restrict__ v_cache,  // [P, 1, 8, 128]
    const int* __restrict__ kv_indptr,          // [B+1]
    const int* __restrict__ kv_indices,         // [nkv]
    __nv_bfloat16* __restrict__ out,            // [B, 32, 128]
    float* __restrict__ lse,                    // [B, 32]
    float qk_scale) {
  using namespace gqa_ps1;
  const int bh   = blockIdx.x;
  const int b    = bh / NUM_KVH;
  const int hkv  = bh % NUM_KVH;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int qh   = hkv * GQA + warp;        // this warp's q head
  const int ps   = kv_indptr[b];
  const int pe   = kv_indptr[b + 1];
  const int len  = pe - ps;

  __nv_bfloat16* orow = out + ((size_t)b * 32 + qh) * HEAD_DIM;
  float* lrow = lse + (size_t)b * 32 + qh;

  if (len <= 0) {
    // empty sequence: out 0 / lse -inf
    *reinterpret_cast<uint2*>(orow + lane * 4) = make_uint2(0u, 0u);
    if (lane == 0) *lrow = -INFINITY;
    return;
  }

  const __nv_bfloat16* qrow = q + ((size_t)b * 32 + qh) * HEAD_DIM + lane * 4;
  uint2 qu = __ldg(reinterpret_cast<const uint2*>(qrow));
  float2 qa = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&qu.x));
  float2 qb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&qu.y));
  float q0 = qa.x, q1 = qa.y, q2 = qb.x, q3 = qb.y;

  float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
  float m_run = -INFINITY, l_run = 0.f;

  for (int t = 0; t < len; ++t) {
    int pg = __ldg(kv_indices + ps + t);
    const __nv_bfloat16* krow =
        k_cache + ((size_t)pg * NUM_KVH + hkv) * HEAD_DIM + lane * 4;
    const __nv_bfloat16* vrow =
        v_cache + ((size_t)pg * NUM_KVH + hkv) * HEAD_DIM + lane * 4;
    uint2 ku = __ldg(reinterpret_cast<const uint2*>(krow));
    float2 ka = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ku.x));
    float2 kb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ku.y));
    float acc = q0 * ka.x + q1 * ka.y + q2 * kb.x + q3 * kb.y;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc += __shfl_xor_sync(0xffffffffu, acc, off);
    float s = acc * qk_scale;
    float m_new = fmaxf(m_run, s);
    float alpha = (m_run == -INFINITY) ? 0.f : exp2f(m_run - m_new);
    float p = exp2f(s - m_new);
    uint2 vu = __ldg(reinterpret_cast<const uint2*>(vrow));
    float2 va = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vu.x));
    float2 vb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vu.y));
    o0 = o0 * alpha + p * va.x;
    o1 = o1 * alpha + p * va.y;
    o2 = o2 * alpha + p * vb.x;
    o3 = o3 * alpha + p * vb.y;
    l_run = l_run * alpha + p;
    m_run = m_new;
  }

  float inv = 1.f / l_run;                  // l_run >= 1 whenever len > 0
  __nv_bfloat162 r0 = __floats2bfloat162_rn(o0 * inv, o1 * inv);
  __nv_bfloat162 r1 = __floats2bfloat162_rn(o2 * inv, o3 * inv);
  *reinterpret_cast<uint2*>(orow + lane * 4) =
      make_uint2(*reinterpret_cast<uint32_t*>(&r0), *reinterpret_cast<uint32_t*>(&r1));
  if (lane == 0) *lrow = m_run + log2f(l_run);
}

// ---------------------------------------------------------------------------
// Host wrapper. All dispatch decisions come from host-known shapes only — there is
// no D2H sync anywhere. Split count targets ~444 resident CTAs (148 SMs) with
// whole-tile chunk quantization; nkv < B*T_TILE routes to the tiny-workload kernel.
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> run(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor kv_indptr,
    torch::Tensor kv_indices,
    double sm_scale) {

  using namespace gqa_ps1;

  TORCH_CHECK(q.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda() &&
              kv_indptr.is_cuda() && kv_indices.is_cuda(),
              "all tensors must be CUDA");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16 &&
              k_cache.scalar_type() == torch::kBFloat16 &&
              v_cache.scalar_type() == torch::kBFloat16,
              "q/k_cache/v_cache must be bfloat16");
  TORCH_CHECK(kv_indptr.scalar_type() == torch::kInt32 &&
              kv_indices.scalar_type() == torch::kInt32,
              "kv_indptr/kv_indices must be int32");
  TORCH_CHECK(q.is_contiguous() && k_cache.is_contiguous() &&
              v_cache.is_contiguous() && kv_indptr.is_contiguous() &&
              kv_indices.is_contiguous(),
              "all tensors must be contiguous");

  const int B = (int)q.size(0);
  TORCH_CHECK(q.dim() == 3 && q.size(1) == 32 && q.size(2) == HEAD_DIM,
              "q must be [B, 32, 128]");
  TORCH_CHECK(k_cache.dim() == 4 && k_cache.size(1) == 1 &&
              k_cache.size(2) == NUM_KVH && k_cache.size(3) == HEAD_DIM,
              "k_cache must be [P, 1, 8, 128]");
  TORCH_CHECK(v_cache.dim() == 4 && v_cache.size(1) == 1 &&
              v_cache.size(2) == NUM_KVH && v_cache.size(3) == HEAD_DIM,
              "v_cache must be [P, 1, 8, 128]");
  TORCH_CHECK(kv_indptr.dim() == 1 && kv_indptr.size(0) == B + 1,
              "kv_indptr must be [B+1]");
  TORCH_CHECK(kv_indices.dim() == 1, "kv_indices must be [nkv]");

  const long nkv = kv_indices.numel();

  auto opts = q.options();
  torch::Tensor out = torch::empty({B, 32, HEAD_DIM}, opts);
  torch::Tensor lse = torch::empty({B, 32}, opts.dtype(torch::kFloat32));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const float qk_scale = (float)sm_scale * LOG2E;

  // ---- tiny-workload path: mean sequence length < one KV tile ----
  if (nkv < (long)B * T_TILE) {
    gqa_paged_decode_small<<<dim3(B * NUM_KVH), NTHREADS, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr()),
        kv_indptr.data_ptr<int>(), kv_indices.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        lse.data_ptr<float>(), qk_scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, lse};
  }

  // ---- FlashDecoding split count: fill ~444 CTAs, never split below one chunk ----
  const int bh = B * NUM_KVH;
  const int chunk_tok = (bh >= 148) ? 128 : 64;   // 128-token chunks for large batches
  int target = (TARGET_CTAS + bh - 1) / bh;
  if (target < 1) target = 1;
  long denom = (long)B * chunk_tok;
  int smax = (int)((nkv + denom - 1) / denom);
  if (smax < 1) smax = 1;
  int S = (target < smax) ? target : smax;

  torch::Tensor partial;
  float* partial_ptr = nullptr;
  if (S > 1) {
    partial = torch::empty({(long)B * NUM_KVH * S * PSTRIDE},
                           opts.dtype(torch::kFloat32));
    partial_ptr = partial.data_ptr<float>();
  }

  static int smem_configured = 0;    // one-time opt-in for the 70400 B dynamic smem
  if (!smem_configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        (const void*)gqa_paged_decode_splitk,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    smem_configured = 1;
  }

  dim3 grid(S, bh);
  gqa_paged_decode_splitk<<<grid, NTHREADS, SMEM_BYTES, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr()),
      kv_indptr.data_ptr<int>(), kv_indices.data_ptr<int>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      lse.data_ptr<float>(), partial_ptr, S, qk_scale);

  if (S > 1) {
    gqa_splitk_reduce<<<dim3(bh), dim3(32), 0, stream>>>(
        partial_ptr, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        lse.data_ptr<float>(), S);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {out, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run,
        "GQA Paged KV Decode (attempt 4, stabilization-first): two-kernel "
        "FlashDecoding split-K (mma.m16n8k8 bf16 tensor cores, KSTAGE=2 cp.async.cg "
        "256-B-row gathers, page-id smem ring) + separate LSE-weighted reduce kernel; "
        "warp-per-head FMA kernel for tiny workloads (nkv < B*64); base-2 online "
        "softmax so lse is 2-based like the reference; empty sequences -> out 0 / "
        "lse -inf. Returns (output[bf16], lse[f32]).");
}