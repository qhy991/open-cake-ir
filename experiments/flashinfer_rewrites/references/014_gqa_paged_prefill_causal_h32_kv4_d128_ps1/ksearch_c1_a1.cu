/*
 * GQA Paged Prefill (causal) — Split-KV FA2 with direct-output fast path.
 * Op: 014_gqa_paged_prefill_causal (h=32, kv=4, d=128, page_size=1, causal).
 * Target: B300.
 *
 * Paged KV gather through kv_indices (page_size=1 => 1 page == 1 token).
 * Split-KV: each sequence's KV range partitioned into chunks of KV_SPLIT
 * tokens. One tile_meta entry per (Q-tile, KV-chunk). Each CTA processes one
 * chunk. When the sequence has exactly 1 KV chunk, the CTA writes final
 * output+LSE directly (no partials, no reduction). When >1 chunk, it writes
 * partial (acc_bf16, rmax, rsum) to scratch; a reduction kernel merges them.
 *
 * Occupancy: single SMEM buffer (N_BUF=1, 32 KB) so 2 CTAs fit per SM
 * (100 KB). Memory latency is hidden by inter-CTA occupancy rather than
 * intra-CTA double-buffering.
 *
 * Host path: sync-free. The partial-scratch sizing uses a shape-only upper
 * bound, max_chunks_bound = ceil(num_kv_indices / KV_SPLIT); the exact
 * max_chunks is computed on device by build_tile_meta_kernel into
 * d_counters[1] and used as the partial-buffer stride by the split and
 * reduce kernels.
 *
 * This round — launch/metadata overhaul (all shape-only, math unchanged):
 *  1. chunk_counts is per-ROW ([total_q], one value shared by the 32 q heads
 *     of a row) and filled cooperatively by ALL threads of the metadata
 *     kernel. The old [total_q,32] fill was done by ONE thread per sequence
 *     with 128B-stride stores: total_q*32 serialized stores dominated
 *     single-sequence workloads (wl11: ~524k stores ~= 1.9 ms of a 2.06 ms
 *     total) and inflated every total_q>10k workload.
 *  2. KV-less sequences (nkv<=0, sq>0) now emit one zero-work chunk, so every
 *     query row is written by the split kernel (zero output / -inf lse).
 *     output/lse/chunk_counts therefore need no host-side fill: the
 *     torch::zeros (a 134 MB memset on 16k-row shapes) / torch::full /
 *     num_tiles fill kernels are gone.
 *  3. Shape-only tile-count bound replacing the loose product bound:
 *       sum_b ceil(sq/8)*ceil(nkv/2048)
 *         <= q*kv/16384 + ceil(q/8) + ceil(kv/2048) + 3*batch   (+ slack)
 *     (from ceil(a/8)*ceil(b/2048) <= (a/8+1)(b/2048+1) and
 *      sum sq_b*nkv_b <= total_q*num_kv_indices). Cuts launched CTAs ~50x on
 *     ragged multi-batch workloads (wl7: 436k -> ~8k) by removing dead CTAs
 *     that only read a counter and exit.
 *  4. Partial scratch and the reduce kernel are only allocated/launched when
 *     max_chunks_bound > 1: any multi-chunk row requires exact max_chunks>=2
 *     and exact <= bound, so bound==1 proves no partial path is ever taken
 *     (nullptr partials are never dereferenced). Removes one launch + three
 *     allocations for the ~32/34 workloads with num_kv_indices <= 2048.
 *  5. 8-byte packed bf16 stores in the epilogue (4 scalar 2B stores -> 1).
 *
 * This round (a1/cycle-1) — dead initialization removal:
 *  6. run() allocated output=torch::zeros(...) and lse=torch::full(...,-inf)
 *     and then IMMEDIATELY reassigned both to torch::empty: the zero-fill and
 *     -inf-fill kernels plus their allocator round-trips executed dead every
 *     call (a 134 MB-class output memset on 16k-row shapes). output/lse are
 *     now allocated ONCE as torch::empty at a single site, before the
 *     degenerate-path guard. The kernels fully overwrite every row (zero-work
 *     chunks cover KV-less sequences), so the fills were provably redundant.
 *     The degenerate early-return path stays well-defined: total_q==0 gives
 *     zero-sized tensors (torch::empty is trivially valid), and batch<=0 with
 *     total_q>0 cannot occur (qo_indptr[batch]==total_q forces total_q==0
 *     when batch<=0); it is still guarded with an explicit in-place fill so
 *     the returned tensors are always well-defined.
 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <float.h>
#include <math.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

static constexpr int HEAD_DIM        = 128;
static constexpr int NUM_QO_HEADS    = 32;
static constexpr int NUM_KV_HEADS    = 4;
static constexpr int GQA_RATIO       = 8;
static constexpr int WARPS_PER_BLOCK = 8;
static constexpr int BLOCK_THREADS   = WARPS_PER_BLOCK * 32;
static constexpr int DIMS_PER_LANE   = HEAD_DIM / 32;
static constexpr int BRQ             = 8;
static constexpr int T_PIPE          = 64;
static constexpr int N_BUF           = 1;
static constexpr int KV_SPLIT        = 2048;
// (row, q-head) pairs merged per reduce CTA. Keeps the reduce grid small
// (total_q*32/64 CTAs) in the common single-chunk case.
static constexpr int REDUCE_PAIRS_PER_CTA = 64;
// Threads for the metadata kernel: enough to parallelize the tile-meta and
// per-row chunk_counts fills even when batch is tiny (batch==1 was the
// worst case of the old single-thread fill).
static constexpr int SCAN_MIN_THREADS = 256;

static constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ void cp_async_16B(void* __restrict__ dst, const void* __restrict__ src) {
    // V-cache path: cp.async.ca (cache-all, L1+L2). V is read AFTER scores are
    // computed and benefits from L1 residency to overlap the V->acc accumulation
    // with K->score work of neighbouring rows.
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
        : : "r"(static_cast<unsigned>(__cvta_generic_to_shared(dst))),
             "l"(reinterpret_cast<unsigned long long>(src)));
}
__device__ __forceinline__ void cp_async_cg_16B(void* __restrict__ dst, const void* __restrict__ src) {
    // K-cache path: cp.async.cg (cache-global, L2-only => L1-bypass). K is
    // consumed once for the Q.K dot product then discarded; caching it in L1
    // pollutes L1 capacity that V reads depend on. Bypassing L1 for K frees that
    // capacity for V. Identical DMA semantics to cp.async.ca (same commit/wait
    // groups) so N_BUF=1 overlap and occupancy are preserved.
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
        : : "r"(static_cast<unsigned>(__cvta_generic_to_shared(dst))),
             "l"(reinterpret_cast<unsigned long long>(src)));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}
__device__ __forceinline__ void load4_bf16_smem(const __nv_bfloat16* __restrict__ ptr,
    float* r0, float* r1, float* r2, float* r3) {
    uint2 raw = *reinterpret_cast<const uint2*>(ptr);
    *r0 = __bfloat162float(__nv_bfloat16_raw{(unsigned short)(raw.x & 0xFFFF)});
    *r1 = __bfloat162float(__nv_bfloat16_raw{(unsigned short)(raw.x >> 16)});
    *r2 = __bfloat162float(__nv_bfloat16_raw{(unsigned short)(raw.y & 0xFFFF)});
    *r3 = __bfloat162float(__nv_bfloat16_raw{(unsigned short)(raw.y >> 16)});
}
__device__ __forceinline__ void load4_bf16_ldg(const __nv_bfloat16* ptr,
    float* r0, float* r1, float* r2, float* r3) {
    uint2 v = __ldg(reinterpret_cast<const uint2*>(ptr));
    *r0 = __bfloat162float(__nv_bfloat16_raw{(unsigned short)(v.x & 0xFFFF)});
    *r1 = __bfloat162float(__nv_bfloat16_raw{(unsigned short)(v.x >> 16)});
    *r2 = __bfloat162float(__nv_bfloat16_raw{(unsigned short)(v.y & 0xFFFF)});
    *r3 = __bfloat162float(__nv_bfloat16_raw{(unsigned short)(v.y >> 16)});
}
// Packed 8B store of the 4 contiguous bf16 elements a lane owns (lane_base..+4).
// Replaces 4 scalar 2B stores per row-chunk in every epilogue.
__device__ __forceinline__ void store4_bf16(__nv_bfloat16* __restrict__ ptr,
    float v0, float v1, float v2, float v3) {
    __nv_bfloat162 lo = __floats2bfloat162_rn(v0, v1);
    __nv_bfloat162 hi = __floats2bfloat162_rn(v2, v3);
    reinterpret_cast<__nv_bfloat162*>(ptr)[0] = lo;
    reinterpret_cast<__nv_bfloat162*>(ptr)[1] = hi;
}
__device__ __forceinline__ void store4_bf16_zero(__nv_bfloat16* __restrict__ ptr) {
    *reinterpret_cast<unsigned long long*>(ptr) = 0ull;
}
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return __shfl_sync(0xffffffff, val, 0);
}

__device__ __forceinline__ void load_kv_tile_async(
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int*           __restrict__ kv_indices,
    __nv_bfloat16* __restrict__ smem_k, __nv_bfloat16* __restrict__ smem_v,
    int kv_lo, int tile_kv_count, int kv_start, int chunk_lo, int kv_head, int tid)
{
    constexpr int XFER_PER_ROW = HEAD_DIM / 8;
    constexpr int TOTAL_XFER   = T_PIPE * XFER_PER_ROW;
    constexpr int XFER_PER_THR = TOTAL_XFER / BLOCK_THREADS;
    const int page_stride = NUM_KV_HEADS * HEAD_DIM;
    const __nv_bfloat16* k_head_base = k_cache + (size_t)kv_head * HEAD_DIM;
    const __nv_bfloat16* v_head_base = v_cache + (size_t)kv_head * HEAD_DIM;
    #pragma unroll
    for (int i = 0; i < XFER_PER_THR; i++) {
        const int xid  = tid * XFER_PER_THR + i;
        const int row  = xid / XFER_PER_ROW;
        const int col8 = (xid % XFER_PER_ROW) * 8;
        __nv_bfloat16* sk = smem_k + row * HEAD_DIM + col8;
        __nv_bfloat16* sv = smem_v + row * HEAD_DIM + col8;
        if (row < tile_kv_count) {
            // chunk_lo is the chunk's offset within the sequence; without it
            // every chunk reads chunk-0's pages (max_rel_err 21.83 on wl11).
            const int page_id = __ldg(kv_indices + kv_start + chunk_lo + kv_lo + row);
            const __nv_bfloat16* gk = k_head_base + (size_t)page_id * page_stride + col8;
            const __nv_bfloat16* gv = v_head_base + (size_t)page_id * page_stride + col8;
            // Cache-policy differentiation: K via cp.async.cg (L1-bypass, L2-only)
            // since K is consumed once then discarded; V via cp.async.ca (L1+L2)
            // since V is read after scores and benefits from L1 residency. Both
            // share the same async commit/wait group, so the N_BUF=1 single-buffer
            // pipeline and 2-CTA/SM occupancy are unchanged.
            cp_async_cg_16B(sk, gk); cp_async_16B(sv, gv);
        } else {
            *reinterpret_cast<uint4*>(sk) = make_uint4(0,0,0,0);
            *reinterpret_cast<uint4*>(sv) = make_uint4(0,0,0,0);
        }
    }
}

__device__ __forceinline__ void process_4tok(
    const __nv_bfloat16* __restrict__ smem_k, const __nv_bfloat16* __restrict__ smem_v,
    int tok_base, const float* q_reg, float* acc,
    float& running_max, float& running_sum, float sm_scale_log2, int lane_base) {
    float k0[DIMS_PER_LANE], k1[DIMS_PER_LANE], k2[DIMS_PER_LANE], k3[DIMS_PER_LANE];
    load4_bf16_smem(smem_k+(tok_base+0)*HEAD_DIM+lane_base,&k0[0],&k0[1],&k0[2],&k0[3]);
    load4_bf16_smem(smem_k+(tok_base+1)*HEAD_DIM+lane_base,&k1[0],&k1[1],&k1[2],&k1[3]);
    load4_bf16_smem(smem_k+(tok_base+2)*HEAD_DIM+lane_base,&k2[0],&k2[1],&k2[2],&k2[3]);
    load4_bf16_smem(smem_k+(tok_base+3)*HEAD_DIM+lane_base,&k3[0],&k3[1],&k3[2],&k3[3]);
    float d0=0.f,d1=0.f,d2=0.f,d3=0.f;
    #pragma unroll
    for (int d=0; d<DIMS_PER_LANE; d++) { d0+=q_reg[d]*k0[d]; d1+=q_reg[d]*k1[d]; d2+=q_reg[d]*k2[d]; d3+=q_reg[d]*k3[d]; }
    d0=warp_reduce_sum(d0); d1=warp_reduce_sum(d1); d2=warp_reduce_sum(d2); d3=warp_reduce_sum(d3);
    float l0=d0*sm_scale_log2,l1=d1*sm_scale_log2,l2=d2*sm_scale_log2,l3=d3*sm_scale_log2;
    float v0[DIMS_PER_LANE],v1[DIMS_PER_LANE],v2[DIMS_PER_LANE],v3[DIMS_PER_LANE];
    load4_bf16_smem(smem_v+(tok_base+0)*HEAD_DIM+lane_base,&v0[0],&v0[1],&v0[2],&v0[3]);
    load4_bf16_smem(smem_v+(tok_base+1)*HEAD_DIM+lane_base,&v1[0],&v1[1],&v1[2],&v1[3]);
    load4_bf16_smem(smem_v+(tok_base+2)*HEAD_DIM+lane_base,&v2[0],&v2[1],&v2[2],&v2[3]);
    load4_bf16_smem(smem_v+(tok_base+3)*HEAD_DIM+lane_base,&v3[0],&v3[1],&v3[2],&v3[3]);
    float nm=fmaxf(running_max,fmaxf(fmaxf(l0,l1),fmaxf(l2,l3)));
    float ep=exp2f(running_max-nm);
    float e0=exp2f(l0-nm),e1=exp2f(l1-nm),e2=exp2f(l2-nm),e3=exp2f(l3-nm);
    running_sum=running_sum*ep+e0+e1+e2+e3; running_max=nm;
    #pragma unroll
    for (int d=0; d<DIMS_PER_LANE; d++) acc[d]=acc[d]*ep+e0*v0[d]+e1*v1[d]+e2*v2[d]+e3*v3[d];
}
__device__ __forceinline__ void process_2tok(
    const __nv_bfloat16* __restrict__ smem_k, const __nv_bfloat16* __restrict__ smem_v,
    int tok_base, const float* q_reg, float* acc,
    float& running_max, float& running_sum, float sm_scale_log2, int lane_base) {
    float k0[DIMS_PER_LANE],k1[DIMS_PER_LANE];
    load4_bf16_smem(smem_k+(tok_base+0)*HEAD_DIM+lane_base,&k0[0],&k0[1],&k0[2],&k0[3]);
    load4_bf16_smem(smem_k+(tok_base+1)*HEAD_DIM+lane_base,&k1[0],&k1[1],&k1[2],&k1[3]);
    float d0=0.f,d1=0.f;
    #pragma unroll
    for (int d=0; d<DIMS_PER_LANE; d++) { d0+=q_reg[d]*k0[d]; d1+=q_reg[d]*k1[d]; }
    d0=warp_reduce_sum(d0); d1=warp_reduce_sum(d1);
    float l0=d0*sm_scale_log2,l1=d1*sm_scale_log2;
    float v0[DIMS_PER_LANE],v1[DIMS_PER_LANE];
    load4_bf16_smem(smem_v+(tok_base+0)*HEAD_DIM+lane_base,&v0[0],&v0[1],&v0[2],&v0[3]);
    load4_bf16_smem(smem_v+(tok_base+1)*HEAD_DIM+lane_base,&v1[0],&v1[1],&v1[2],&v1[3]);
    float nm=fmaxf(running_max,fmaxf(l0,l1));
    float ep=exp2f(running_max-nm);
    float e0=exp2f(l0-nm),e1=exp2f(l1-nm);
    running_sum=running_sum*ep+e0+e1; running_max=nm;
    #pragma unroll
    for (int d=0; d<DIMS_PER_LANE; d++) acc[d]=acc[d]*ep+e0*v0[d]+e1*v1[d];
}
__device__ __forceinline__ void process_1tok(
    const __nv_bfloat16* __restrict__ smem_k, const __nv_bfloat16* __restrict__ smem_v,
    int tok_base, const float* q_reg, float* acc,
    float& running_max, float& running_sum, float sm_scale_log2, int lane_base) {
    float k_reg[DIMS_PER_LANE];
    load4_bf16_smem(smem_k+tok_base*HEAD_DIM+lane_base,&k_reg[0],&k_reg[1],&k_reg[2],&k_reg[3]);
    float dot=0.f;
    #pragma unroll
    for (int d=0; d<DIMS_PER_LANE; d++) dot+=q_reg[d]*k_reg[d];
    dot=warp_reduce_sum(dot); float l=dot*sm_scale_log2;
    float v_reg[DIMS_PER_LANE];
    load4_bf16_smem(smem_v+tok_base*HEAD_DIM+lane_base,&v_reg[0],&v_reg[1],&v_reg[2],&v_reg[3]);
    float nm=fmaxf(running_max,l);
    float ep=exp2f(running_max-nm);
    float ec=exp2f(l-nm);
    running_sum=running_sum*ep+ec; running_max=nm;
    #pragma unroll
    for (int d=0; d<DIMS_PER_LANE; d++) acc[d]=acc[d]*ep+ec*v_reg[d];
}
__device__ __forceinline__ void process_tile_row(
    const __nv_bfloat16* __restrict__ smem_k, const __nv_bfloat16* __restrict__ smem_v,
    int valid_r, const float* q_reg, float* acc,
    float& running_max, float& running_sum, float sm_scale_log2, int lane_id) {
    const int lane_base = lane_id * DIMS_PER_LANE;
    int n4=valid_r>>2, rem=valid_r&3, tok=0;
    #pragma unroll
    for (int c=0; c<(T_PIPE>>2); c++) { if(c>=n4)break; process_4tok(smem_k,smem_v,tok,q_reg,acc,running_max,running_sum,sm_scale_log2,lane_base); tok+=4; }
    if (rem>=2) { process_2tok(smem_k,smem_v,tok,q_reg,acc,running_max,running_sum,sm_scale_log2,lane_base); tok+=2; rem-=2; }
    if (rem>=1) process_1tok(smem_k,smem_v,tok,q_reg,acc,running_max,running_sum,sm_scale_log2,lane_base);
}

// meta entry (9 ints): {q_start_global, q_len, q_start_local, kv_start, num_kv,
//                        delta, chunk_lo, chunk_kv_count, num_chunks}
// d_counters (3 ints): [0]=num_tiles (exact), [1]=max_chunks (exact,
//                      device-side partial-scratch stride), [2]=any_multi.
__global__ void gqa_paged_prefill_splitkv_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int*           __restrict__ kv_indices,
    float                             sm_scale,
    const int*           __restrict__ tile_meta,
    __nv_bfloat16*       __restrict__ partial_acc,
    float*               __restrict__ partial_max,
    float*               __restrict__ partial_sum,
    __nv_bfloat16*       __restrict__ output,
    float*               __restrict__ lse,
    const int*           __restrict__ d_counters)
{
    extern __shared__ __nv_bfloat16 smem_raw[];
    __nv_bfloat16* smem_buf[N_BUF];
    #pragma unroll
    for (int b=0; b<N_BUF; b++) smem_buf[b] = smem_raw + (size_t)b*(2*T_PIPE*HEAD_DIM);
    #define SMEM_K(b) (smem_buf[b])
    #define SMEM_V(b) (smem_buf[b] + T_PIPE*HEAD_DIM)

    // Grid flattened as (max_tiles_bound, NUM_KV_HEADS); the exact tile count
    // is read on device, so over-launched CTAs exit here. The shape-only bound
    // (host) is now tight enough that only a handful of CTAs take this exit.
    const int tidx = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int num_tiles = __ldg(d_counters);
    if (tidx >= num_tiles) return;
    // Exact device-side partial-scratch stride (<= the host-side allocation
    // bound, so all indices stay in range; nullptr partials are safe because
    // num_chunks>1 implies exact max_chunks>=2 implies the host bound>1).
    const int max_chunks = __ldg(d_counters + 1);

    const int tid=threadIdx.x, warp_id=tid>>5, lane_id=tid&31, lane_base=lane_id*DIMS_PER_LANE;
    const int* meta = tile_meta + tidx*9;
    const int q_start_global=meta[0], q_len=meta[1], q_start_local=meta[2];
    const int kv_start=meta[3], num_kv=meta[4], delta=meta[5];
    const int chunk_lo=meta[6], chunk_kv_count=meta[7], num_chunks=meta[8];
    const int qh = kv_head*GQA_RATIO + warp_id;
    const float sm_scale_log2 = sm_scale * LOG2E;
    const int chunk_idx = chunk_lo / KV_SPLIT;

    float q_reg[BRQ][DIMS_PER_LANE];
    #pragma unroll
    for (int r=0; r<BRQ; r++) {
        if (r<q_len) {
            const int qrg = q_start_global + r;
            load4_bf16_ldg(q+(size_t)qrg*NUM_QO_HEADS*HEAD_DIM+qh*HEAD_DIM+lane_base,
                           &q_reg[r][0],&q_reg[r][1],&q_reg[r][2],&q_reg[r][3]);
        } else {
            #pragma unroll
            for (int d=0; d<DIMS_PER_LANE; d++) q_reg[r][d]=0.f;
        }
    }
    float acc[BRQ][DIMS_PER_LANE], rmax[BRQ], rsum[BRQ];
    #pragma unroll
    for (int r=0; r<BRQ; r++) {
        #pragma unroll
        for (int d=0; d<DIMS_PER_LANE; d++) acc[r][d]=0.f;
        rmax[r]=-FLT_MAX;
        rsum[r]=0.0f;
    }

    int last_q_causal = q_start_local + (q_len-1) + delta + 1;
    if (last_q_causal > num_kv) last_q_causal = num_kv;
    if (last_q_causal < 0) last_q_causal = 0;
    int chunk_end = chunk_lo + chunk_kv_count;
    if (chunk_end > last_q_causal) chunk_end = last_q_causal;
    int chunk_tokens = chunk_end - chunk_lo;
    const int num_kv_tiles = (chunk_tokens + T_PIPE - 1) / T_PIPE;

    if (num_kv_tiles == 0) {
        // Empty chunk (also the zero-work chunks emitted for KV-less
        // sequences): write -inf/0 partial (only needed if multi-chunk).
        if (num_chunks > 1) {
            #pragma unroll
            for (int r=0; r<BRQ; r++) {
                if (r>=q_len) break;
                const int qrg=q_start_global+r;
                const size_t pb=((size_t)qrg*NUM_QO_HEADS+qh)*max_chunks;
                store4_bf16_zero(partial_acc+(pb+chunk_idx)*HEAD_DIM+lane_base);
                if (lane_id==0) { partial_max[pb+chunk_idx]=-INFINITY; partial_sum[pb+chunk_idx]=0.0f; }
            }
        } else {
            // single chunk, no visible KV: write zero output, -inf lse directly
            // (this is what makes torch::empty output/lse sufficient).
            #pragma unroll
            for (int r=0; r<BRQ; r++) {
                if (r>=q_len) break;
                const int qrg=q_start_global+r;
                store4_bf16_zero(output+(size_t)qrg*NUM_QO_HEADS*HEAD_DIM+qh*HEAD_DIM+lane_base);
                if (lane_id==0) lse[(size_t)qrg*NUM_QO_HEADS+qh]=-INFINITY;
            }
        }
        return;
    }

    // ---- Tile loop ----
    // With N_BUF=1 (single SMEM buffer, chosen to reach 2 CTAs/SM occupancy),
    // loads and compute cannot overlap within one CTA: each tile is loaded,
    // waited on, then consumed before the next tile reuses the buffer. The
    // memory latency is hidden by occupancy (a second resident CTA issues
    // loads while this one computes) rather than by intra-CTA double-buffering.
    for (int tile=0; tile<num_kv_tiles; tile++) {
        const int kv_lo=tile*T_PIPE;
        const int tkc=(chunk_tokens-kv_lo<T_PIPE)?(chunk_tokens-kv_lo):T_PIPE;
        load_kv_tile_async(k_cache,v_cache,kv_indices,SMEM_K(0),SMEM_V(0),kv_lo,tkc,kv_start,chunk_lo,kv_head,tid);
        cp_async_commit();
        cp_async_wait_group<0>();
        __syncthreads();
        __nv_bfloat16* sk=SMEM_K(0), *sv=SMEM_V(0);
        #pragma unroll
        for (int r=0; r<BRQ; r++) {
            if (r>=q_len) break;
            const int q_local=q_start_local+r;
            int mkr=q_local+delta+1;
            if (mkr>num_kv) mkr=num_kv;
            if (mkr<=0) continue;
            int abs_lo=chunk_lo+kv_lo, abs_end=abs_lo+tkc;
            if (abs_end>mkr) abs_end=mkr;
            int vr=abs_end-abs_lo;
            if (vr<=0) continue;
            process_tile_row(sk,sv,vr,q_reg[r],acc[r],rmax[r],rsum[r],sm_scale_log2,lane_id);
        }
        __syncthreads();  // compute done before buffer reuse
    }

    // Epilogue: direct output if single chunk, else partial.
    if (num_chunks == 1) {
        #pragma unroll
        for (int r=0; r<BRQ; r++) {
            if (r>=q_len) break;
            const int qrg=q_start_global+r;
            const float inv=(rsum[r]>0.f)?(1.f/rsum[r]):0.f;
            store4_bf16(output+(size_t)qrg*NUM_QO_HEADS*HEAD_DIM+qh*HEAD_DIM+lane_base,
                        acc[r][0]*inv,acc[r][1]*inv,acc[r][2]*inv,acc[r][3]*inv);
            if (lane_id==0) {
                float* lp=lse+(size_t)qrg*NUM_QO_HEADS+qh;
                if (rsum[r]>0.f && isfinite(rmax[r])) *lp=rmax[r]+log2f(rsum[r]);
                else *lp=-INFINITY;
            }
        }
    } else {
        #pragma unroll
        for (int r=0; r<BRQ; r++) {
            if (r>=q_len) break;
            const int qrg=q_start_global+r;
            const size_t pb=((size_t)qrg*NUM_QO_HEADS+qh)*max_chunks;
            #pragma unroll
            for (int d=0; d<DIMS_PER_LANE; d++)
                partial_acc[(pb+chunk_idx)*HEAD_DIM+lane_base+d]=__float2bfloat16(acc[r][d]);
            if (lane_id==0) { partial_max[pb+chunk_idx]=rmax[r]; partial_sum[pb+chunk_idx]=rsum[r]; }
        }
    }
}

// Launched only when the host-side chunk bound > 1 (multi-chunk possible);
// additionally gated by the device-side any_multi flag (d_counters[2]) so the
// dominant all-single-chunk case exits after one 4-byte load. Flattened 1D
// grid over (row*32+qh) pairs, REDUCE_PAIRS_PER_CTA pairs per CTA.
__global__ void reduce_partial_kernel(
    const __nv_bfloat16* __restrict__ partial_acc,
    const float*         __restrict__ partial_max,
    const float*         __restrict__ partial_sum,
    const int*           __restrict__ chunk_counts,   // per-ROW: [total_q]
    const int*           __restrict__ d_counters,
    __nv_bfloat16*       __restrict__ output,
    float*               __restrict__ lse,
    int                               total_pairs)
{
    if (__ldg(d_counters+2) == 0) return;  // no multi-chunk row anywhere
    const int max_chunks = __ldg(d_counters+1);
    const int lane=threadIdx.x, lane_base=lane*DIMS_PER_LANE;
    const int pair_lo = blockIdx.x * REDUCE_PAIRS_PER_CTA;
    const int pair_hi = (pair_lo + REDUCE_PAIRS_PER_CTA < total_pairs)
                        ? (pair_lo + REDUCE_PAIRS_PER_CTA) : total_pairs;
    for (int pair=pair_lo; pair<pair_hi; ++pair) {
        const int qh   = pair % NUM_QO_HEADS;
        const int qrow = pair / NUM_QO_HEADS;
        const int nchunks = chunk_counts[qrow];
        if (nchunks <= 1) continue;  // single-chunk rows already written directly
        const size_t pb=((size_t)qrow*NUM_QO_HEADS+qh)*max_chunks;
        float m_i=-INFINITY;
        for (int c=0; c<nchunks; c++) { float pm=partial_max[pb+c]; if (pm>m_i) m_i=pm; }
        if (!isfinite(m_i)) {
            store4_bf16_zero(output+(size_t)qrow*NUM_QO_HEADS*HEAD_DIM+qh*HEAD_DIM+lane_base);
            if (lane==0) lse[(size_t)qrow*NUM_QO_HEADS+qh]=-INFINITY;
            continue;
        }
        float acc[DIMS_PER_LANE], l_i=0.f;
        #pragma unroll
        for (int d=0; d<DIMS_PER_LANE; d++) acc[d]=0.f;
        for (int c=0; c<nchunks; c++) {
            float pm=partial_max[pb+c], ps=partial_sum[pb+c];
            float scale=exp2f(pm-m_i);
            l_i += ps*scale;
            const __nv_bfloat16* pa=partial_acc+(pb+c)*HEAD_DIM+lane_base;
            float a0,a1,a2,a3; load4_bf16_smem(pa,&a0,&a1,&a2,&a3);
            acc[0]+=a0*scale; acc[1]+=a1*scale; acc[2]+=a2*scale; acc[3]+=a3*scale;
        }
        const float inv=(l_i>0.f)?(1.f/l_i):0.f;
        store4_bf16(output+(size_t)qrow*NUM_QO_HEADS*HEAD_DIM+qh*HEAD_DIM+lane_base,
                    acc[0]*inv,acc[1]*inv,acc[2]*inv,acc[3]*inv);
        if (lane==0) {
            float* lp=lse+(size_t)qrow*NUM_QO_HEADS+qh;
            if (l_i>0.f && isfinite(m_i)) *lp=m_i+log2f(l_i); else *lp=-INFINITY;
        }
    }
}

__global__ void build_tile_meta_kernel(
    const int* __restrict__ qo_indptr, const int* __restrict__ kv_indptr,
    int*       __restrict__ tile_meta, int* __restrict__ d_counters,
    int*       __restrict__ chunk_counts,
    int batch, int total_q)
{
    extern __shared__ int s_meta[];   // [ s_cnt | s_base | s_nnc | s_multi(1) ]
    int* s_cnt=s_meta, *s_base=s_meta+batch, *s_nnc=s_meta+2*batch, *s_multi=s_meta+3*batch;
    const int tid=threadIdx.x;
    if (tid==0) s_multi[0]=0;   // init before any multi-flag write below
    __syncthreads();
    if (tid<batch) {
        const int qs=qo_indptr[tid], qe=qo_indptr[tid+1], sq=qe-qs;
        const int ks=kv_indptr[tid], ke=kv_indptr[tid+1], nkv=ke-ks;
        const int nqt=(sq>0)?((sq+BRQ-1)/BRQ):0;
        // KV chunks; KV-less sequences with query rows emit ONE zero-work
        // chunk so their rows are still written (zero output / -inf lse) by
        // the split kernel — output/lse/chunk_counts need no host-side fill.
        const int nnc=(nkv>0)?((nkv+KV_SPLIT-1)/KV_SPLIT):((sq>0)?1:0);
        s_nnc[tid]=nnc;
        s_cnt[tid]=nqt*nnc;
        // Device-side gate replacing the host any_multi: set iff some
        // sequence with query rows needs >1 KV chunk. Benign race — only the
        // value 1 is ever written by racing threads.
        if (nnc>1 && nqt>0) s_multi[0]=1;
    }
    __syncthreads();
    if (tid==0) {
        int a=0, mx=1;
        for (int b=0;b<batch;b++) { s_base[b]=a; a+=s_cnt[b]; if (s_nnc[b]>mx) mx=s_nnc[b]; }
        d_counters[0]=a;          // num_tiles (exact)
        d_counters[1]=mx;         // exact max_chunks: partial-scratch stride
        d_counters[2]=s_multi[0]; // any multi-chunk sequence
    }
    __syncthreads();
    // ---- Cooperative tile-meta + per-row chunk_counts fill ----
    // The old kernel filled both with ONE thread per sequence; with
    // batch==1 and total_q=16384 that serialized total_q*32 128B-stride
    // stores (~1.9 ms). Now every thread of the block participates, and
    // chunk_counts is per-ROW (the 32 q heads of a row share one value).
    for (int b=0;b<batch;b++) {
        const int cnt=s_cnt[b];
        if (cnt<=0) continue;
        const int base=s_base[b];
        const int qs=qo_indptr[b], sq=qo_indptr[b+1]-qs;
        const int ks=kv_indptr[b], nkv=kv_indptr[b+1]-ks, delta=nkv-sq;
        const int nqt=(sq+BRQ-1)/BRQ, nnc=s_nnc[b];
        for (int t=base+tid; t<base+cnt; t+=blockDim.x) {
            const int li=t-base, qt=li/nnc, ch=li-qt*nnc;
            int ql=sq-qt*BRQ; if (ql>BRQ) ql=BRQ;
            int clo=ch*KV_SPLIT, ckc=nkv-clo;
            if (ckc>KV_SPLIT) ckc=KV_SPLIT;
            if (ckc<0) ckc=0;
            int* m=tile_meta+t*9;
            m[0]=qs+qt*BRQ; m[1]=ql; m[2]=qt*BRQ; m[3]=ks; m[4]=nkv; m[5]=delta;
            m[6]=clo; m[7]=ckc; m[8]=nnc;
        }
        for (int r=qs+tid; r<qs+sq; r+=blockDim.x) chunk_counts[r]=nnc;
    }
    (void)total_q;
}

std::vector<torch::Tensor> run(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor qo_indptr, torch::Tensor kv_indptr, torch::Tensor kv_indices,
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
    const int batch=(int)qo_indptr.size(0)-1;
    auto f32=torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    auto bf16=torch::TensorOptions().dtype(torch::kBFloat16).device(q.device());
    auto i32=torch::TensorOptions().dtype(torch::kInt).device(q.device());
    // Single allocation site: the kernels fully overwrite every row (KV-less
    // sequences emit zero-work chunks), so torch::empty is sufficient. The old
    // torch::zeros/torch::full fill launches (a 134 MB-class memset on 16k-row
    // shapes) executed dead — the tensors were reassigned to torch::empty
    // before any kernel launch — and are deleted along with the reassignment.
    auto output=torch::empty({total_q,NUM_QO_HEADS,HEAD_DIM}, bf16);
    auto lse=torch::empty({total_q,NUM_QO_HEADS}, f32);
    if (total_q==0) return {output,lse};  // zero-sized tensors, trivially valid
    if (batch<=0) {
        // Degenerate: rows exist but no sequence can address them. Unreachable
        // under the harness contract (qo_indptr[batch]==total_q forces
        // total_q==0 when batch<=0), but kept so the returned tensors are
        // always well-defined (zero output / -inf lse) even if it ever fires.
        output.zero_();
        lse.fill_(-INFINITY);
        return {output,lse};
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // ---- Sync-free worst-case bounds from tensor shapes ----
    // Per-sequence chunk count is ceil(nkv_b/KV_SPLIT) with nkv_b bounded by
    // the total number of gathered KV tokens, so
    //     max_chunks <= ceil(num_kv_indices / KV_SPLIT)
    // (a multi-chunk row needs exact max_chunks>=2, hence bound>=2: partials
    // and the reduce launch below are gated on exactly that).
    const int total_kv_idx=(int)kv_indices.size(0);
    int max_chunks_bound=(total_kv_idx+KV_SPLIT-1)/KV_SPLIT;
    if (max_chunks_bound<1) max_chunks_bound=1;

    // Shape-only tile-count bound. For every sequence:
    //   ceil(sq/8)*ceil(nkv/2048) <= (sq/8+1)*(nkv/2048+1)
    //                             =  sq*nkv/16384 + sq/8 + nkv/2048 + 1
    // and sum_b sq_b*nkv_b <= total_q*num_kv_indices, so
    //   num_tiles <= ceil(total_q*kv/16384) + ceil(total_q/8) + ceil(kv/2048)
    //               + 3*batch                       (+ 64 slack)
    // This replaces the loose (ceil(q/8)+B)*(ceil(kv/S)+B) product bound,
    // which over-launched ~50x more CTAs than exist on ragged multi-batch
    // workloads (every excess CTA costs a dispatch + counter read + exit).
    const long long tiles_ll = (((long long)total_q*total_kv_idx)+16383)/16384
                             + (long long)((total_q+BRQ-1)/BRQ)
                             + (long long)((total_kv_idx+KV_SPLIT-1)/KV_SPLIT)
                             + 3LL*batch + 64;
    int max_tiles_cap = (tiles_ll > 4000000LL) ? 4000000 : (int)tiles_ll;
    if (max_tiles_cap < 1) max_tiles_cap = 1;

    // All outputs are fully written by the kernels: every query row belongs
    // to a sequence that emits at least one tile (zero-work chunks cover
    // KV-less sequences), so NO fill kernels are needed for output/lse/
    // chunk_counts/counters — output/lse were allocated ONCE as torch::empty
    // above (no dead zeros/full init, no reassignment, no extra allocator
    // round-trip).
    auto meta_gpu=torch::empty({(long long)max_tiles_cap*9}, i32);
    auto counters=torch::empty({3}, i32);          // written by build kernel
    auto chunk_counts=torch::empty({total_q}, i32); // per-ROW, written by build

    // Partial scratch at the shape-derived bound, allocated only when a
    // multi-chunk row is possible (bound>1). Indexed on device with the
    // EXACT max_chunks (<= bound), so every slot the reducer reads is
    // written by the producer kernel: torch::empty, no memset, no sync.
    torch::Tensor partial_acc, partial_max, partial_sum;
    __nv_bfloat16* partial_acc_ptr=nullptr;
    float* partial_max_ptr=nullptr;
    float* partial_sum_ptr=nullptr;
    if (max_chunks_bound > 1) {
        partial_acc=torch::empty({(long long)total_q,NUM_QO_HEADS,max_chunks_bound,HEAD_DIM}, bf16);
        partial_max=torch::empty({(long long)total_q,NUM_QO_HEADS,max_chunks_bound}, f32);
        partial_sum=torch::empty({(long long)total_q,NUM_QO_HEADS,max_chunks_bound}, f32);
        partial_acc_ptr=(__nv_bfloat16*)partial_acc.data_ptr<at::BFloat16>();
        partial_max_ptr=partial_max.data_ptr<float>();
        partial_sum_ptr=partial_sum.data_ptr<float>();
    }

    // >=256 threads even for tiny batches so the cooperative meta /
    // chunk_counts fills are wide; per-seq work still guards tid<batch.
    int scan_threads = batch<SCAN_MIN_THREADS ? SCAN_MIN_THREADS : batch;
    if (scan_threads > 1024) scan_threads = 1024;
    const size_t scan_smem=(size_t)(3*batch+1)*sizeof(int);
    build_tile_meta_kernel<<<1,scan_threads,scan_smem,stream>>>(
        qo_indptr.data_ptr<int>(), kv_indptr.data_ptr<int>(),
        meta_gpu.data_ptr<int>(), counters.data_ptr<int>(),
        chunk_counts.data_ptr<int>(), batch, total_q);

    const size_t smem_size=(size_t)N_BUF*2*T_PIPE*HEAD_DIM*sizeof(__nv_bfloat16);
    static bool s_smem_attr_set=false;
    if (!s_smem_attr_set) {
        cudaFuncSetAttribute(gqa_paged_prefill_splitkv_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size);
        s_smem_attr_set=true;
    }

    dim3 grid(max_tiles_cap, NUM_KV_HEADS);
    dim3 block(BLOCK_THREADS);
    gqa_paged_prefill_splitkv_kernel<<<grid,block,smem_size,stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr<at::BFloat16>()),
        kv_indices.data_ptr<int>(), (float)sm_scale,
        meta_gpu.data_ptr<int>(),
        partial_acc_ptr, partial_max_ptr, partial_sum_ptr,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        lse.data_ptr<float>(),
        counters.data_ptr<int>());

    // Reduction only when a multi-chunk row is possible (bound>1). It is
    // additionally gated by the device flag (counters[2]) so the dominant
    // all-single-chunk case exits after one 4B load.
    if (max_chunks_bound > 1) {
        const long long total_pairs_ll=(long long)total_q*NUM_QO_HEADS;
        const int total_pairs=(int)total_pairs_ll;
        const int reduce_ctas=(int)((total_pairs_ll+REDUCE_PAIRS_PER_CTA-1)/REDUCE_PAIRS_PER_CTA);
        dim3 rgrid(reduce_ctas>0?reduce_ctas:1); dim3 rblock(32);
        reduce_partial_kernel<<<rgrid,rblock,0,stream>>>(
            partial_acc_ptr, partial_max_ptr, partial_sum_ptr,
            chunk_counts.data_ptr<int>(), counters.data_ptr<int>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            lse.data_ptr<float>(), total_pairs);
    }
    return {output,lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "GQA Paged Prefill (causal) split-KV FA2 with direct-output fast path: "
          "paged kv_indices gather, KV_SPLIT=2048, single-buffer occ=2 cp.async, "
          "online softmax, partial-LSE reduction (multi-chunk only), base-2 LSE, GQA 8:1, "
          "sync-free host path (shape-bounded partial scratch + device-side reduce gate), "
          "fill-free outputs (single torch::empty site, no dead zeros/full init), "
          "cooperative per-row chunk_counts, tight shape-only grid bound");
}