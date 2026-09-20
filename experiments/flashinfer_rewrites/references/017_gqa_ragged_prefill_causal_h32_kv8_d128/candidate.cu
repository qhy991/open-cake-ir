/* GQA Ragged Prefill (causal) Kernel — task 017.
 * Op: 017_gqa_ragged_prefill_causal_h32_kv8_d128
 *   num_qo_heads=32, num_kv_heads=8 (GQA 4:1), head_dim=128.
 *   Ragged (contiguous, NON-paged) KV: NO kv_indices page table.
 *   Signature: run(q, k, v, qo_indptr, kv_indptr, sm_scale) -> (output, lse).
 *
 * Adapted from the proven 015 paged prefill winner (2.07x vs FlashInfer) for the
 * ragged layout: removed the kv_indices page table entirely. KV access is
 * contiguous and coalesced:
 *     base_offset = (kv_start + tok_global) * (NUM_KV_HEADS*HEAD_DIM)
 *                   + kv_head*HEAD_DIM + lane_base
 * eliminating the per-token __ldg(idx_base + tok) indirection (no dependent
 * gather, better L2 reuse) — a strict win over the paged version.
 *
 * Design (FlashAttention-2 style, fully fused — no logits/softmax materialized):
 *   - Grid:  (NUM_KV_HEADS=8, num_q_tiles_total) (real num_tiles read from
 *     device memory via *d_num_tiles; gridDim.y=total_q is a safe upper bound,
 *     CTAs beyond num_tiles early-return — NO host sync before launch).
 *   - Block: 128 threads = 4 warps; warp w handles q_head qh = kv_head*4 + w.
 *   - Each CTA processes BRQ=4 query rows for all 4 q_heads of one kv_head.
 *     Q rows' Q vectors loaded into registers once and kept resident.
 *   - KV tokens stream through SMEM in tiles of T_PIPE=16 (cooperative __ldg
 *     load by the 4 warps), double-buffered. K/V tile reused across all BRQ=4
 *     rows and all 4 heads of the kv_head.
 *   - CRITICAL: each lane owns 4 bf16 (8 bytes) of a 128-dim KV row, loaded via
 *     __ldg uint2 at col = lane*4 — matching the reader's lane*DIMS_PER_LANE
 *     (=lane*4) access. 32 lanes * 4 = 128 = HEAD_DIM (no row overflow).
 *   - Online softmax in log2 space with exp2f (running max + rescale):
 *       scale logits by sm_scale*LOG2E so rmax is in log2 domain;
 *       base-2 LSE = rmax + log2(rsum); -inf for fully-masked rows.
 *   - Per-row causal mask folded into the loop as a KV-count bound:
 *       for query row q_local, valid KV count = min(num_kv, q_local+1+delta),
 *       delta = num_kv - num_q_tokens (per sequence, constant within a tile).
 *
 * Occupancy tuning (this variant): BRQ 8->4 halves per-thread softmax state
 * (q_reg+acc drops 64->32 live fp32 regs), and __launch_bounds__(128, 6)
 * targets ~<=85 regs/thread so ~6 CTAs fit the 64K regfile instead of ~4.
 * smem stays 16KB/CTA, which caps residency at 6 CTAs on sm_89's 100KB/SM ->
 * 24 warps/SM = 50% achieved occupancy (up from 25%). Per-row KV consumption
 * order (T_PIPE=16 tile sequence, process_4tok/2tok/1tok order) is unchanged,
 * so rmax/rsum/acc accumulate identically — bit-identical output/lse. K/V
 * tiles are re-read once more per 4 rows, which is free (memory ~4% HBM peak).
 *
 * Tile meta (q_start_global, q_len, q_start_local, kv_start, num_kv, delta) is
 * precomputed on the GPU from qo_indptr/kv_indptr (build_tile_meta_kernel) so
 * the grid is compact and no D2H/H2D sync is needed (num_tiles <= total_q).
 *
 * Edge cases:
 *   - Row with 0 valid KV tokens (max_kv_r <= 0): output 0, lse -inf.
 *   - Padding KV rows beyond num_kv in a tile: zero-filled in SMEM.
 *   - Empty batch element (q_start>=q_end or kv_start>=kv_end): emits no tiles.
 *
 * Target: RTX 4090 / sm_89.
 * Verified: 21/21 PASSED, geomean 0.0595 ms = 1.72x vs FlashInfer 0.1023 ms.
 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

static constexpr int HEAD_DIM      = 128;
static constexpr int NUM_QO_HEADS  = 32;
static constexpr int NUM_KV_HEADS  = 8;
static constexpr int GQA_RATIO     = 4;
static constexpr int BLOCK_THREADS = 128;
static constexpr int DIMS_PER_LANE = HEAD_DIM / 32;
static constexpr int BRQ           = 4;
static constexpr int T_PIPE        = 16;
static constexpr int ROWS_PER_WARP = T_PIPE / 4;
static constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ void load4_bf16_ldg(const __nv_bfloat16* ptr, float* r0, float* r1, float* r2, float* r3) {
    uint2 v = __ldg(reinterpret_cast<const uint2*>(ptr));
    *r0 = __bfloat162float(__nv_bfloat16_raw{(unsigned short)(v.x & 0xFFFF)});
    *r1 = __bfloat162float(__nv_bfloat16_raw{(unsigned short)(v.x >> 16)});
    *r2 = __bfloat162float(__nv_bfloat16_raw{(unsigned short)(v.y & 0xFFFF)});
    *r3 = __bfloat162float(__nv_bfloat16_raw{(unsigned short)(v.y >> 16)});
}
__device__ __forceinline__ void load4_bf16_smem_u2(const __nv_bfloat16* __restrict__ ptr, float* r0, float* r1, float* r2, float* r3) {
    uint2 raw = *reinterpret_cast<const uint2*>(ptr);
    *r0 = __bfloat162float(__nv_bfloat16_raw{(unsigned short)(raw.x & 0xFFFF)});
    *r1 = __bfloat162float(__nv_bfloat16_raw{(unsigned short)(raw.x >> 16)});
    *r2 = __bfloat162float(__nv_bfloat16_raw{(unsigned short)(raw.y & 0xFFFF)});
    *r3 = __bfloat162float(__nv_bfloat16_raw{(unsigned short)(raw.y >> 16)});
}
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return __shfl_sync(0xffffffff, val, 0);
}

__device__ __forceinline__ void process_4tok(const __nv_bfloat16* __restrict__ sk, const __nv_bfloat16* __restrict__ sv, int tb, const float* q_reg, float* acc, float& rmax, float& rsum, float sm_sl2, int lane_id) {
    const int lo = lane_id * DIMS_PER_LANE;
    float k0[DIMS_PER_LANE], k1[DIMS_PER_LANE], k2[DIMS_PER_LANE], k3[DIMS_PER_LANE];
    load4_bf16_smem_u2(sk+(tb+0)*HEAD_DIM+lo, &k0[0], &k0[1], &k0[2], &k0[3]);
    load4_bf16_smem_u2(sk+(tb+1)*HEAD_DIM+lo, &k1[0], &k1[1], &k1[2], &k1[3]);
    load4_bf16_smem_u2(sk+(tb+2)*HEAD_DIM+lo, &k2[0], &k2[1], &k2[2], &k2[3]);
    load4_bf16_smem_u2(sk+(tb+3)*HEAD_DIM+lo, &k3[0], &k3[1], &k3[2], &k3[3]);
    float d0=0.f, d1=0.f, d2=0.f, d3=0.f;
    #pragma unroll
    for (int d=0; d<DIMS_PER_LANE; d++) { d0+=q_reg[d]*k0[d]; d1+=q_reg[d]*k1[d]; d2+=q_reg[d]*k2[d]; d3+=q_reg[d]*k3[d]; }
    d0=warp_reduce_sum(d0); d1=warp_reduce_sum(d1); d2=warp_reduce_sum(d2); d3=warp_reduce_sum(d3);
    float l0=d0*sm_sl2, l1=d1*sm_sl2, l2=d2*sm_sl2, l3=d3*sm_sl2;
    float v0[DIMS_PER_LANE], v1[DIMS_PER_LANE], v2[DIMS_PER_LANE], v3[DIMS_PER_LANE];
    load4_bf16_smem_u2(sv+(tb+0)*HEAD_DIM+lo, &v0[0], &v0[1], &v0[2], &v0[3]);
    load4_bf16_smem_u2(sv+(tb+1)*HEAD_DIM+lo, &v1[0], &v1[1], &v1[2], &v1[3]);
    load4_bf16_smem_u2(sv+(tb+2)*HEAD_DIM+lo, &v2[0], &v2[1], &v2[2], &v2[3]);
    load4_bf16_smem_u2(sv+(tb+3)*HEAD_DIM+lo, &v3[0], &v3[1], &v3[2], &v3[3]);
    float nm=fmaxf(rmax, fmaxf(fmaxf(l0,l1), fmaxf(l2,l3)));
    float ep=exp2f(rmax-nm);
    float e0=exp2f(l0-nm), e1=exp2f(l1-nm), e2=exp2f(l2-nm), e3=exp2f(l3-nm);
    rsum=rsum*ep+e0+e1+e2+e3; rmax=nm;
    #pragma unroll
    for (int d=0; d<DIMS_PER_LANE; d++) acc[d]=acc[d]*ep+e0*v0[d]+e1*v1[d]+e2*v2[d]+e3*v3[d];
}
__device__ __forceinline__ void process_2tok(const __nv_bfloat16* __restrict__ sk, const __nv_bfloat16* __restrict__ sv, int tb, const float* q_reg, float* acc, float& rmax, float& rsum, float sm_sl2, int lane_id) {
    const int lo = lane_id * DIMS_PER_LANE;
    float k0[DIMS_PER_LANE], k1[DIMS_PER_LANE];
    load4_bf16_smem_u2(sk+(tb+0)*HEAD_DIM+lo, &k0[0], &k0[1], &k0[2], &k0[3]);
    load4_bf16_smem_u2(sk+(tb+1)*HEAD_DIM+lo, &k1[0], &k1[1], &k1[2], &k1[3]);
    float d0=0.f, d1=0.f;
    #pragma unroll
    for (int d=0; d<DIMS_PER_LANE; d++) { d0+=q_reg[d]*k0[d]; d1+=q_reg[d]*k1[d]; }
    d0=warp_reduce_sum(d0); d1=warp_reduce_sum(d1);
    float l0=d0*sm_sl2, l1=d1*sm_sl2;
    float v0[DIMS_PER_LANE], v1[DIMS_PER_LANE];
    load4_bf16_smem_u2(sv+(tb+0)*HEAD_DIM+lo, &v0[0], &v0[1], &v0[2], &v0[3]);
    load4_bf16_smem_u2(sv+(tb+1)*HEAD_DIM+lo, &v1[0], &v1[1], &v1[2], &v1[3]);
    float nm=fmaxf(rmax, fmaxf(l0,l1));
    float ep=exp2f(rmax-nm);
    float e0=exp2f(l0-nm), e1=exp2f(l1-nm);
    rsum=rsum*ep+e0+e1; rmax=nm;
    #pragma unroll
    for (int d=0; d<DIMS_PER_LANE; d++) acc[d]=acc[d]*ep+e0*v0[d]+e1*v1[d];
}
__device__ __forceinline__ void process_1tok(const __nv_bfloat16* __restrict__ sk, const __nv_bfloat16* __restrict__ sv, int tb, const float* q_reg, float* acc, float& rmax, float& rsum, float sm_sl2, int lane_id) {
    const int lo = lane_id * DIMS_PER_LANE;
    float k_reg[DIMS_PER_LANE];
    load4_bf16_smem_u2(sk+tb*HEAD_DIM+lo, &k_reg[0], &k_reg[1], &k_reg[2], &k_reg[3]);
    float dot=0.f;
    #pragma unroll
    for (int d=0; d<DIMS_PER_LANE; d++) dot+=q_reg[d]*k_reg[d];
    dot=warp_reduce_sum(dot);
    float l=dot*sm_sl2;
    float v_reg[DIMS_PER_LANE];
    load4_bf16_smem_u2(sv+tb*HEAD_DIM+lo, &v_reg[0], &v_reg[1], &v_reg[2], &v_reg[3]);
    float nm=fmaxf(rmax, l);
    float ep=exp2f(rmax-nm), ec=exp2f(l-nm);
    rsum=rsum*ep+ec; rmax=nm;
    #pragma unroll
    for (int d=0; d<DIMS_PER_LANE; d++) acc[d]=acc[d]*ep+ec*v_reg[d];
}
__device__ __forceinline__ void process_smem_tile(const __nv_bfloat16* __restrict__ sk, const __nv_bfloat16* __restrict__ sv, int tt, const float* q_reg, float* acc, float& rmax, float& rsum, float sm_sl2, int lane_id) {
    int n8=tt>>3, n4=(tt>>2)&1, n2=(tt>>1)&1, n1=tt&1, tok=0;
    for (int gi=0; gi<n8; gi++, tok+=8) {
        process_4tok(sk, sv, tok,     q_reg, acc, rmax, rsum, sm_sl2, lane_id);
        process_4tok(sk, sv, tok + 4, q_reg, acc, rmax, rsum, sm_sl2, lane_id);
    }
    if (n4) { process_4tok(sk, sv, tok, q_reg, acc, rmax, rsum, sm_sl2, lane_id); tok += 4; }
    if (n2) { process_2tok(sk, sv, tok, q_reg, acc, rmax, rsum, sm_sl2, lane_id); tok += 2; }
    if (n1) { process_1tok(sk, sv, tok, q_reg, acc, rmax, rsum, sm_sl2, lane_id); }
}

// Synchronous cooperative __ldg load (proven in 015 winner). Each lane loads
// 4 bf16 (8 bytes, uint2) at col=lane*4; 32*4=128=HEAD_DIM (matches reader).
__device__ __forceinline__ void load_kv_tile_sync(__nv_bfloat16* __restrict__ buf, const __nv_bfloat16* __restrict__ gmem, int kv_lo, int tkc, int kv_start, int kv_head_offset, int kv_stride) {
    const int tid=threadIdx.x, warp_id=tid>>5, lane_id=tid&31;
    const int wrs=warp_id*ROWS_PER_WARP;
    const int col=lane_id*DIMS_PER_LANE;
    #pragma unroll
    for (int r=0; r<ROWS_PER_WARP; r++) {
        int row=wrs+r;
        __nv_bfloat16* dst=buf+row*HEAD_DIM+col;
        if (row<tkc) {
            int tg=kv_lo+row;
            int bo=(kv_start+tg)*kv_stride+kv_head_offset+col;
            uint2 v = __ldg(reinterpret_cast<const uint2*>(gmem+bo));
            *reinterpret_cast<uint2*>(dst)=v;
        } else {
            *reinterpret_cast<uint2*>(dst)=make_uint2(0u, 0u);
        }
    }
}

__global__ void __launch_bounds__(BLOCK_THREADS, 6)
gqa_ragged_prefill_v1(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k_cache, const __nv_bfloat16* __restrict__ v_cache, float sm_scale, const int* __restrict__ tile_meta, __nv_bfloat16* __restrict__ output, float* __restrict__ lse, const int* __restrict__ d_num_tiles, int num_tiles_bound) {
    extern __shared__ __nv_bfloat16 smem_raw[];
    __nv_bfloat16* smem_k = smem_raw;
    __nv_bfloat16* smem_v = smem_raw + 2*T_PIPE*HEAD_DIM;
    const int kv_head=blockIdx.x, tidx=blockIdx.y;
    const int num_tiles=__ldg(d_num_tiles);
    if (tidx>=num_tiles) return;
    (void)num_tiles_bound;
    const int tid=threadIdx.x, warp_id=tid>>5, lane_id=tid&31, lane_base=lane_id*DIMS_PER_LANE;
    const int* meta=tile_meta+tidx*6;
    const int q_start_global=meta[0], q_len=meta[1], q_start_local=meta[2], kv_start=meta[3], num_kv=meta[4], delta=meta[5];
    const int qh=kv_head*GQA_RATIO+warp_id;
    const float sm_sl2=sm_scale*LOG2E;
    const int kv_head_offset=kv_head*HEAD_DIM, kv_stride=NUM_KV_HEADS*HEAD_DIM;

    float q_reg[BRQ][DIMS_PER_LANE];
    #pragma unroll
    for (int r=0; r<BRQ; r++) {
        if (r<q_len) {
            int qrg=q_start_global+r;
            const __nv_bfloat16* qr=q+(qrg*NUM_QO_HEADS+qh)*HEAD_DIM;
            load4_bf16_ldg(qr+lane_base, &q_reg[r][0], &q_reg[r][1], &q_reg[r][2], &q_reg[r][3]);
        } else {
            #pragma unroll
            for (int d=0; d<DIMS_PER_LANE; d++) q_reg[r][d]=0.f;
        }
    }

    float acc[BRQ][DIMS_PER_LANE];
    float rmax[BRQ], rsum[BRQ];
    #pragma unroll
    for (int r=0; r<BRQ; r++) {
        #pragma unroll
        for (int d=0; d<DIMS_PER_LANE; d++) acc[r][d]=0.f;
        rmax[r]=-FLT_MAX; rsum[r]=0.0f;
    }

    int mmk=q_start_local+q_len+delta;
    if (mmk>num_kv) mmk=num_kv;
    if (mmk<0) mmk=0;
    const int nkt=(mmk+T_PIPE-1)/T_PIPE;
    if (nkt==0) {
        #pragma unroll
        for (int r=0; r<BRQ; r++) {
            if (r>=q_len) break;
            int qrg=q_start_global+r;
            __nv_bfloat16* op=output+(qrg*NUM_QO_HEADS+qh)*HEAD_DIM+lane_base;
            #pragma unroll
            for (int d=0; d<DIMS_PER_LANE; d++) op[d]=__float2bfloat16(0.f);
            if (lane_id==0) lse[qrg*NUM_QO_HEADS+qh]=-INFINITY;
        }
        return;
    }

    {
        int kl0=0;
        int tk0=(num_kv-kl0<T_PIPE)?(num_kv-kl0):T_PIPE;
        load_kv_tile_sync(smem_k, k_cache, kl0, tk0, kv_start, kv_head_offset, kv_stride);
        load_kv_tile_sync(smem_v, v_cache, kl0, tk0, kv_start, kv_head_offset, kv_stride);
    }

    int cur=0;
    for (int tile=0; tile<nkt; tile++) {
        const int kl=tile*T_PIPE;
        const int tkc=(num_kv-kl<T_PIPE)?(num_kv-kl):T_PIPE;
        if (tile+1<nkt) {
            const int nxt=1-cur;
            int kl1=(tile+1)*T_PIPE;
            int tk1=(num_kv-kl1<T_PIPE)?(num_kv-kl1):T_PIPE;
            load_kv_tile_sync(smem_k+nxt*T_PIPE*HEAD_DIM, k_cache, kl1, tk1, kv_start, kv_head_offset, kv_stride);
            load_kv_tile_sync(smem_v+nxt*T_PIPE*HEAD_DIM, v_cache, kl1, tk1, kv_start, kv_head_offset, kv_stride);
        }
        __syncthreads();

        __nv_bfloat16* skb=smem_k+cur*T_PIPE*HEAD_DIM;
        __nv_bfloat16* svb=smem_v+cur*T_PIPE*HEAD_DIM;
        #pragma unroll
        for (int r=0; r<BRQ; r++) {
            if (r>=q_len) break;
            int ql=q_start_local+r;
            int mkr=ql+delta+1;
            if (mkr>num_kv) mkr=num_kv;
            if (mkr<=0) continue;
            int vr=mkr-kl;
            if (vr>tkc) vr=tkc;
            if (vr<=0) continue;
            process_smem_tile(skb, svb, vr, q_reg[r], acc[r], rmax[r], rsum[r], sm_sl2, lane_id);
        }
        cur=1-cur;
        __syncthreads();
    }

    #pragma unroll
    for (int r=0; r<BRQ; r++) {
        if (r>=q_len) break;
        int qrg=q_start_global+r;
        float inv=(rsum[r]>0.0f)?(1.0f/rsum[r]):0.0f;
        __nv_bfloat16* op=output+(qrg*NUM_QO_HEADS+qh)*HEAD_DIM+lane_base;
        #pragma unroll
        for (int d=0; d<DIMS_PER_LANE; d++) op[d]=__float2bfloat16(acc[r][d]*inv);
        if (lane_id==0) {
            float* lp=lse+(qrg*NUM_QO_HEADS+qh);
            if (rsum[r]>0.0f && isfinite(rsum[r])) *lp=rmax[r]+log2f(rsum[r]);
            else *lp=-INFINITY;
        }
    }
}

__global__ void build_tile_meta_kernel(const int* __restrict__ qo_indptr, const int* __restrict__ kv_indptr, int* __restrict__ tile_meta, int* __restrict__ d_num_tiles, int batch) {
    extern __shared__ int s_meta[];
    int* s_cnt=s_meta;
    int* s_base=s_meta+batch;
    const int tid=threadIdx.x;
    int my_cnt=0;
    if (tid<batch) {
        int qs=qo_indptr[tid], qe=qo_indptr[tid+1], kvs=kv_indptr[tid], kve=kv_indptr[tid+1];
        int sq=qe-qs, nkv=kve-kvs;
        my_cnt=(sq>0 && nkv>0)?((sq+BRQ-1)/BRQ):0;
    }
    s_cnt[tid]=my_cnt;
    __syncthreads();
    if (tid==0) {
        int acc=0;
        for (int b=0; b<batch; b++) { s_base[b]=acc; acc+=s_cnt[b]; }
        *d_num_tiles=acc;
    }
    __syncthreads();
    if (tid<batch && my_cnt>0) {
        int qs=qo_indptr[tid], qe=qo_indptr[tid+1], kvs=kv_indptr[tid], kve=kv_indptr[tid+1];
        int sq=qe-qs, nkv=kve-kvs, delta=nkv-sq;
        int base=s_base[tid];
        for (int t=0; t<my_cnt; t++) {
            int* m=tile_meta+(base+t)*6;
            int qlen=(sq-t*BRQ<BRQ)?(sq-t*BRQ):BRQ;
            m[0]=qs+t*BRQ; m[1]=qlen; m[2]=t*BRQ; m[3]=kvs; m[4]=nkv; m[5]=delta;
        }
    }
}

std::vector<torch::Tensor> run(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor qo_indptr, torch::Tensor kv_indptr, double sm_scale) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(q.dtype()==torch::kBFloat16, "q must be bf16");
    TORCH_CHECK(k.dtype()==torch::kBFloat16, "k must be bf16");
    TORCH_CHECK(v.dtype()==torch::kBFloat16, "v must be bf16");
    TORCH_CHECK(qo_indptr.dtype()==torch::kInt32, "qo_indptr must be int32");
    TORCH_CHECK(kv_indptr.dtype()==torch::kInt32, "kv_indptr must be int32");
    if (!q.is_contiguous()) q=q.contiguous();
    if (!k.is_contiguous()) k=k.contiguous();
    if (!v.is_contiguous()) v=v.contiguous();
    if (!qo_indptr.is_contiguous()) qo_indptr=qo_indptr.contiguous();
    if (!kv_indptr.is_contiguous()) kv_indptr=kv_indptr.contiguous();
    const int total_q=(int)q.size(0);
    auto f32opts=torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    auto bf16opts=torch::TensorOptions().dtype(torch::kBFloat16).device(q.device());
    auto output=torch::empty({total_q, NUM_QO_HEADS, HEAD_DIM}, bf16opts);
    auto lse=torch::full({total_q, NUM_QO_HEADS}, -INFINITY, f32opts);
    const int batch=(int)qo_indptr.size(0)-1;
    if (total_q==0 || batch<=0) return {output, lse};
    cudaStream_t stream=at::cuda::getCurrentCUDAStream();
    auto meta_gpu=torch::empty({(long long)total_q*6}, torch::TensorOptions().dtype(torch::kInt).device(q.device()));
    auto num_tiles_t=torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt).device(q.device()));
    const int scan_threads=batch<1024?batch:1024;
    const size_t scan_smem=(size_t)batch*sizeof(int)*2;
    build_tile_meta_kernel<<<1, scan_threads, scan_smem, stream>>>(qo_indptr.data_ptr<int>(), kv_indptr.data_ptr<int>(), meta_gpu.data_ptr<int>(), num_tiles_t.data_ptr<int>(), batch);
    const size_t smem_size=2*T_PIPE*HEAD_DIM*sizeof(__nv_bfloat16)*2;
    static bool s_smem_attr_set=false;
    if (!s_smem_attr_set) {
        cudaFuncSetAttribute(gqa_ragged_prefill_v1, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        s_smem_attr_set=true;
    }
    dim3 grid(NUM_KV_HEADS, total_q);
    dim3 block(BLOCK_THREADS);
    gqa_ragged_prefill_v1<<<grid, block, smem_size, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
        (float)sm_scale, meta_gpu.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        lse.data_ptr<float>(), num_tiles_t.data_ptr<int>(), total_q);
    return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "GQA Ragged Prefill (causal) 017: fused FA2, BRQ=4, T_PIPE=16, online softmax log2, base-2 LSE, contiguous ragged KV, GQA 4:1");
}