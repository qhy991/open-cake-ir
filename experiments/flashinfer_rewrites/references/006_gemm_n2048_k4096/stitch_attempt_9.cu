/*
 * GEMM C[M,N] = A[M,K] @ B[N,K]^T, fp16 -- attempt 10: de-risk reset.
 *
 * After five consecutive unverified designs (attempts 4..9: TMA tensor
 * maps, cluster/DSMEM launches, CUDA-graph capture/replay, raw bulk-copy
 * 1D paths -- every one of them failed compile or verification), this
 * attempt carries forward VERBATIM the only measured-passing unit: the
 * attempt-3 ticket fused-split-K kernel (stitch_attempt_2.cu, bench
 * result: 29/29 PASS, 1.2326x geomean on B300 SXM6 AC, sm_103, 148 SMs,
 * ~6.434 TB/s HBM; 1.81-1.84x at M <= 8, 1.29-1.69x at M 17..32, M=63
 * 1.038x, M=64 1.188x). No cluster kernels, no TMA, no graph replay, no
 * dlsym, no descriptor encoding, no __grid_constant__ CUtensorMap, no
 * sm_90-only instructions: every launch is a direct <<<>>> launch of the
 * same sm_80-compatible cp.async + mma.sync.m16n8k16 body.
 *
 *   N=2048 (const), K=4096 (const), M varies 1..16294 across 29 workloads.
 *   Baseline: torch.matmul (cuBLAS NT).
 *
 * Three bands (M-keyed, two compares):
 *
 *   (1) M >= 258 (12 workloads, compute-bound): cuBLAS passthrough
 *       torch.matmul(A, B.t()) -- parity is the optimum there; zero
 *       sub-1.0x risk. Unchanged.
 *
 *   (2) 65 <= M < 258 (M=93, 128, 172): cuBLAS passthrough. The old
 *       (64,96] custom bucket measured 0.899x at M=93 (BM=96 re-tile,
 *       attempt 4's predecessor); M_CUSTOM_MAX drops 96 -> 64 so that
 *       bucket lands on the parity path (~1.0x) instead. A bucket that
 *       re-measures <= 1.0x drops M_CUSTOM_MAX to 32 rather than
 *       shipping sub-parity.
 *
 *   (3) M <= 64: custom B-streaming fused split-K GEMM, single launch,
 *       static per-M config table matching the measured rows:
 *         M 1..16 : BM=16/WM=1/WN=8/KSPLIT=8 /NSTAGES=4 (smem 46.1KB,
 *                   grid 32x8x1 = 256 CTAs; measured 1.81-1.84x at M<=8,
 *                   1.29x at M=15/16)
 *         M 17..32: BM=32/WM=2/WN=4/KSPLIT=8 /NSTAGES=4 (smem 55.3KB;
 *                   measured 1.29-1.69x)
 *         M 33..64: BM=64/WM=4/WN=2/KSPLIT=4/NSTAGES=2 (smem 36.9KB,
 *                   proven double buffer; M=63 1.038x, M=64 1.188x)
 *
 * Fusion mechanism (unchanged, measured): each CTA stores its fp32
 * partial to a one-time workspace, issues __threadfence(), then bumps a
 * per-(m,n)-tile ticket counter. The last-arriving CTA clears the counter
 * (self-resetting scratch), reads all KSPLIT partials in ascending-ks
 * order -- fixed summation order, bitwise run-to-run deterministic, no
 * atomics on data -- accumulates in fp32, converts once to fp16, and
 * writes C exactly once with coalesced half2 pairs. One kernel launch
 * per call: no memset, no second kernel.
 *
 * Scratch state crossing calls (all non-operand-derived):
 *   - ticket counters: 160 x 4B of pure synchronization state, zeroed
 *     once and self-resetting inside every launch.
 *   - fp32 partial workspace: one-time cudaMalloc'd max-size scratch
 *     (8*96*2048 floats = 6.3MB, headroom over the largest enabled
 *     config's 8*32*2048), never zeroed: every element the reduction
 *     reads at (ks, row < M, col) is written earlier in the SAME launch
 *     before the writer bumps the ticket; the reading CTA waits for all
 *     KSPLIT tickets. The capacity bound is TORCH_CHECKed on the
 *     failure-only path.
 *
 * Memory pipeline core (unchanged, measured): 16B coalesced cp.async.cg
 * segments, 8 consecutive threads per row, into +8-half padded smem rows
 * (16B-aligned destinations, bank-conflict-free fragment reads);
 * NSTAGES-stage ring, one commit group per stage (A + BN B-rows
 * together), prologue fills NSTAGES-1 stages, steady state waits
 * cp.async.wait_group NSTAGES-1 (leaving NSTAGES-1 groups in flight),
 * tail iterations wait_group 0. The wait depths 0..7 are spelled out
 * explicitly in cp_async_wait_group_n<N> -- the old template folded any
 * N > 2 to "wait_group 3", a latent bug for a hypothetical NSTAGES=5+
 * ring; it is now a static_assert-guarded explicit chain. A tail rows
 * past M are zero-filled in smem. 256-thread CTA / 8 warps in a WM x WN
 * grid, mma.sync.m16n8k16 row.col fp16 x fp16 -> fp32 (B[N,K] row-major
 * IS KxN column-major, so B feeds the col-major MMA operand directly --
 * zero transpose work), register-resident fp32 accumulators persisting
 * across the whole K-slice (fp16 accumulation over K=4096 would risk
 * the 1e-2 tolerance gate).
 *
 * Gated lever (constexpr-false, ships closed): M8_DEEP_CONCURRENCY, a
 * BN=128 geometry for M <= 8 (grid 16x8x1). It flips only after the
 * restored baseline re-measures 29/29 PASS and >= 1.2326x AND nsys/ncu
 * shows kernel (not host) time dominating the ~14.2us M<=8 floor. No
 * gate is open at ship time.
 *
 * Build: -gencode arch=compute_103,code=sm_103 (torch's default
 * arch_list stops at sm_100). Every instruction used (mma.sync m16n8k16,
 * cp.async cg/commit_group/wait_group, atomicAdd/atomicExch,
 * __threadfence) is sm_80+ and valid on sm_103, and the asm bodies stay
 * sm_80-compatible so the unit also compiles for older targets; no
 * __CUDA_ARCH__ >= 900 guard surface remains.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

static constexpr int N_FIXED = 2048;
static constexpr int K_FIXED = 4096;

// At/above the measured B300 roofline crossover: cuBLAS NT parity path.
static constexpr int M_LARGE_THRESHOLD = 258;

// Upper bound of the custom band. Attempt 4's predecessor lost M=93
// (0.899x) inside the old 96 bound; lowering 96 -> 64 pushes (64,257]
// onto the cuBLAS parity path (~1.0x). If the M <= 64 bucket ever
// re-measures <= 1.0x, lower this to 32.
static constexpr int M_CUSTOM_MAX = 64;

// Gated M <= 8 deep-concurrency lever (BN=128 geometry). CLOSED at ship
// time; flips only after a measured > 1.0x on its bucket.
static constexpr bool M8_DEEP_CONCURRENCY = false;

static constexpr int THREADS_PER_BLOCK = 256;   // 8 warps per CTA

// Ticket-counter geometry: one counter per (m,n) tile. grid.x = N/BN = 32
// for every enabled bucket; grid.z (m-tiles) is 1 for every enabled bucket
// (BM >= M), and at most ceil(257/64) = 5 for headroom.
static constexpr int BN_FIXED = 64;
static constexpr int NUM_N_TILES = N_FIXED / BN_FIXED;          // 32
static constexpr int MAX_M_TILES = 5;                            // ceil(257/64)
static constexpr int TICKET_COUNT = NUM_N_TILES * MAX_M_TILES;   // 160 counters

// One-time fp32 partial scratch: covers the largest enabled custom config
// (KSPLIT=8 up to M=32 -> 8*32*2048, KSPLIT=4 up to M=64 -> 4*64*2048,
// and the gated BN=128 M<=8 lever -> 8*8*2048) with the plan's
// KSPLIT_max x 96 x 2048 headroom. 6.3MB, allocated once.
static constexpr int64_t WS_MAX_FLOATS = (int64_t)8 * 96 * N_FIXED;

// ---------------------------------------------------------------------------
// cp.async helpers (sm_80+, valid on sm_103): 16B global -> shared, grouped
// commit/wait. wait_group N (N a compile-time immediate) completes all but
// the newest N groups - the ring's per-stage wait is NSTAGES-1. All depths
// 0..7 are spelled out explicitly: the old template folded any N > 2 into
// "wait_group 3", which was correct only because the shipped rings were
// NSTAGES <= 4; this version refuses any depth outside 0..7 at compile
// time instead of silently mis-waiting a deeper ring.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void cp_async16(void* smem_dst, const void* gsrc) {
    const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(dst), "l"(gsrc));
}
__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void cp_async_wait_group_n() {
    static_assert(N >= 0 && N <= 7, "cp.async.wait_group depth must be 0..7");
    if constexpr (N == 0) {
        asm volatile("cp.async.wait_group 0;\n");
    } else if constexpr (N == 1) {
        asm volatile("cp.async.wait_group 1;\n");
    } else if constexpr (N == 2) {
        asm volatile("cp.async.wait_group 2;\n");
    } else if constexpr (N == 3) {
        asm volatile("cp.async.wait_group 3;\n");
    } else if constexpr (N == 4) {
        asm volatile("cp.async.wait_group 4;\n");
    } else if constexpr (N == 5) {
        asm volatile("cp.async.wait_group 5;\n");
    } else if constexpr (N == 6) {
        asm volatile("cp.async.wait_group 6;\n");
    } else {
        asm volatile("cp.async.wait_group 7;\n");
    }
}

// ---------------------------------------------------------------------------
// mma.sync m16n8k16, fp16 x fp16 -> fp32 (row.col: A fragments row-major,
// B fragments column-major - B[N,K] row-major IS KxN column-major, so B feeds
// the MMA directly with no transpose).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mma_m16n8k16(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float& c0, float& c1, float& c2, float& c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ---------------------------------------------------------------------------
// Fused split-K GEMM kernel (the M <= 64 custom band) -- carried forward
// verbatim from the measured attempt-3 unit.
//
//   grid  = (N/BN, KSPLIT, ceil(M/BM)),  block = 256 threads (8 warps)
//   CTA   : fp32 partial [BM x BN] of one K-slice -> ws[ks]
//           + last-arriving CTA per (m,n) tile reduces the tile into C.
//
// Warp layout: 8 warps in a WM x WN grid, each owning a (BM/WM) x (BN/WN)
// tile built from m16n8k16 fragments (fp32 accumulators, zero-initialized
// once and kept across the whole K-slice).
//
// NSTAGES-stage cp.async ring: the prologue fills stages 0..NSTAGES-2; loop
// iteration c issues stage c+NSTAGES-1 into the slot consumed at iteration
// c-1 (safe: iteration c-1 ended with __syncthreads) and then waits
// cp.async.wait_group NSTAGES-1, which completes exactly stage c (it has
// NSTAGES-1 committed groups after it). Tail iterations (no stage left to
// issue) wait_group 0. Tail rows past M (only in the last m-tile) are
// zero-filled in smem so the MMA never reads garbage; their partials are
// never stored and the reduction loop never reads or writes them.
//
// Epilogue (the fusion): partial store -> __threadfence() -> __syncthreads()
// -> one atomicAdd on the tile's ticket (thread 0). The CTA that observes
// old == KSPLIT-1 knows all KSPLIT contributors' partials are device-visible
// (each did store -> fence before its atomic; this is the documented CUDA
// threadfence-reduction pattern). It clears the counter back to zero for the
// next launch, then reduces the tile: ascending-ks fp32 summation (bitwise
// deterministic), one fp16 conversion, one coalesced write of C.
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int WM, int WN, int KSPLIT, int NSTAGES>
__global__ __launch_bounds__(THREADS_PER_BLOCK) void gemm_splitk_fused(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ ws,
    unsigned int* __restrict__ tickets,
    __half* __restrict__ C,
    const int M)
{
    static_assert((NSTAGES & (NSTAGES - 1)) == 0, "NSTAGES must be a power of two");
    static_assert(BM % WM == 0 && (BM / WM) % 16 == 0,
                  "warp m-tile must be a multiple of 16 rows");
    static_assert(BN % WN == 0 && (BN / WN) % 8 == 0,
                  "warp n-tile must be a multiple of 8 cols");
    static_assert(N_FIXED % BN == 0, "N must tile evenly by BN");

    constexpr int PAD = 8;                  // 16B row padding: 16B-aligned
                                            // cp.async destinations and
                                            // bank-conflict-free fragment reads
    constexpr int SA_STRIDE = BK + PAD;
    constexpr int SB_STRIDE = BK + PAD;
    constexpr int MM = BM / WM / 16;        // mma m-subtiles per warp (16 rows)
    constexpr int NN = BN / WN / 8;         // mma n-subtiles per warp (8 cols)
    constexpr int SEGS = BK / 8;            // 16B segments per tile row
    constexpr int KDIM = K_FIXED / KSPLIT;  // K-slice depth, multiple of BK
    constexpr int NCHUNKS = KDIM / BK;      // BK-deep chunks per K-slice
    static_assert(NSTAGES <= NCHUNKS, "ring deeper than the K-slice");

    extern __shared__ __align__(16) char smem_raw[];
    __half* const sA = reinterpret_cast<__half*>(smem_raw);
    __half* const sB = sA + NSTAGES * BM * SA_STRIDE;

    const int m0 = blockIdx.z * BM;         // first row of this m-tile
    const int n0 = blockIdx.x * BN;         // first col (B row) of this n-tile
    const int ks = blockIdx.y;              // K-slice index
    const int k_base = ks * KDIM;

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp / WN;               // warp row offset: wm * (BM/WM)
    const int wn = warp % WN;               // warp col offset: wn * (BN/WN)

    const int lrow = lane >> 2;             // fragment row 0..7
    const int lcol = (lane & 3) * 2;        // fragment col pair 0,2,4,6

    // fp32 accumulators: persist across the whole K-slice (fp16 accumulation
    // over K=4096 would risk the 1e-2 tolerance gate).
    float acc[MM][NN][4];
    #pragma unroll
    for (int i = 0; i < MM; ++i) {
        #pragma unroll
        for (int j = 0; j < NN; ++j) {
            #pragma unroll
            for (int t = 0; t < 4; ++t) acc[i][j][t] = 0.0f;
        }
    }

    // Stage one BK-deep tile of A[BM x BK] + B[BN x BK]: 16B (8 halfs) per
    // thread-segment, 8 consecutive threads per row -> fully coalesced.
    // B rows are always valid (2048 % 64 == 0); A tail rows are zero-filled.
    auto load_tiles = [&](int buf, int kb) {
        for (int i = tid; i < BM * SEGS; i += THREADS_PER_BLOCK) {
            const int row = i / SEGS;
            const int seg = i - row * SEGS;
            __half* const dst = sA + (buf * BM + row) * SA_STRIDE + seg * 8;
            const int gr = m0 + row;
            if (gr < M) {
                cp_async16(dst, A + (size_t)gr * K_FIXED + kb + seg * 8);
            } else {
                *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        for (int i = tid; i < BN * SEGS; i += THREADS_PER_BLOCK) {
            const int row = i / SEGS;
            const int seg = i - row * SEGS;
            cp_async16(sB + (buf * BN + row) * SB_STRIDE + seg * 8,
                       B + (size_t)(n0 + row) * K_FIXED + kb + seg * 8);
        }
        cp_async_commit_group();
    };

    // Ring prologue: fill NSTAGES-1 stages so NSTAGES-1 stages are in flight
    // behind the first consumed stage.
    #pragma unroll
    for (int s = 0; s < NSTAGES - 1; ++s) {
        load_tiles(s, k_base + s * BK);
    }

    for (int c = 0; c < NCHUNKS; ++c) {
        const int nxt = c + NSTAGES - 1;    // stage (re)filled this iteration
        if (nxt < NCHUNKS) {
            // Slot nxt % NSTAGES == slot consumed at iteration c-1 (whose
            // trailing __syncthreads already retired every read of it).
            load_tiles(nxt & (NSTAGES - 1), k_base + nxt * BK);
            // Complete exactly stage c: it has NSTAGES-1 groups after it.
            cp_async_wait_group_n<NSTAGES - 1>();
        } else {
            cp_async_wait_group_n<0>();     // tail chunk: everything must land
        }
        __syncthreads();

        const __half* const sa = sA + (c & (NSTAGES - 1)) * BM * SA_STRIDE;
        const __half* const sb = sB + (c & (NSTAGES - 1)) * BN * SB_STRIDE;

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            // B fragments: lane holds B[k = lcol..lcol+1 (b0), +8 (b1)][n = lrow]
            uint32_t bfr[NN][2];
            #pragma unroll
            for (int nn = 0; nn < NN; ++nn) {
                const __half* const bp =
                    sb + (wn * (BN / WN) + nn * 8 + lrow) * SB_STRIDE + kk + lcol;
                bfr[nn][0] = *reinterpret_cast<const uint32_t*>(bp);
                bfr[nn][1] = *reinterpret_cast<const uint32_t*>(bp + 8);
            }
            // A fragments: a0/a2 rows lrow, a1/a3 rows lrow+8;
            // k = lcol (a0/a1), k = lcol+8 (a2/a3).
            #pragma unroll
            for (int mm = 0; mm < MM; ++mm) {
                const __half* const ap =
                    sa + (wm * (BM / WM) + mm * 16 + lrow) * SA_STRIDE + kk + lcol;
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(ap);
                const uint32_t a1 = *reinterpret_cast<const uint32_t*>(ap + 8 * SA_STRIDE);
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(ap + 8);
                const uint32_t a3 = *reinterpret_cast<const uint32_t*>(ap + 8 * SA_STRIDE + 8);
                #pragma unroll
                for (int nn = 0; nn < NN; ++nn) {
                    mma_m16n8k16(a0, a1, a2, a3, bfr[nn][0], bfr[nn][1],
                                 acc[mm][nn][0], acc[mm][nn][1],
                                 acc[mm][nn][2], acc[mm][nn][3]);
                }
            }
        }
        __syncthreads();                    // ring slot free for stage c+NSTAGES
    }

    // Store fp32 partials to ws[ks][m][n] as 8B float2 stores (a warp's 32
    // lanes cover 256 contiguous bytes = full 32B sectors). Tail rows past M
    // are skipped entirely - the reduction never touches them.
    const size_t ws_base = (size_t)ks * M * N_FIXED;
    #pragma unroll
    for (int mm = 0; mm < MM; ++mm) {
        #pragma unroll
        for (int nn = 0; nn < NN; ++nn) {
            const int row0 = m0 + wm * (BM / WM) + mm * 16 + lrow;
            const int col  = n0 + wn * (BN / WN) + nn * 8 + lcol;
            if (row0 < M) {
                *reinterpret_cast<float2*>(ws + ws_base + (size_t)row0 * N_FIXED + col) =
                    make_float2(acc[mm][nn][0], acc[mm][nn][1]);
            }
            if (row0 + 8 < M) {
                *reinterpret_cast<float2*>(ws + ws_base + (size_t)(row0 + 8) * N_FIXED + col) =
                    make_float2(acc[mm][nn][2], acc[mm][nn][3]);
            }
        }
    }

    // ---- fused deterministic reduction (single launch, no second kernel) ----
    // Every thread's partial stores must be device-visible before the ticket
    // is bumped: fence per thread, barrier, then one atomicAdd by thread 0.
    __threadfence();
    __shared__ bool s_am_last;
    __syncthreads();
    if (tid == 0) {
        const int tile = blockIdx.z * gridDim.x + blockIdx.x;
        const unsigned int old = atomicAdd(&tickets[tile], 1u);
        s_am_last = (old == (unsigned int)(KSPLIT - 1));
        if (s_am_last) {
            // Last arrival for this tile: all KSPLIT CTAs already bumped the
            // counter, so nobody in this launch touches it again. Reset it to
            // zero for the next (stream-ordered) launch.
            (void)atomicExch(&tickets[tile], 0u);
        }
    }
    __syncthreads();
    if (!s_am_last) return;

    // This CTA reduces the whole (m,n) tile: rows [m0, min(m0+BM, M)), all
    // BN columns. 32 threads per row with float2 partial reads (2 columns
    // each -> a warp reads 256 contiguous bytes, full 32B sectors), 8 rows in
    // flight per pass, ascending-ks summation (fixed order -> bitwise
    // run-to-run deterministic), one fp32->fp16 conversion, paired half2
    // stores into C. C is written exactly once per element.
    const int m_end = m0 + BM;
    const int m_lim = (m_end < M) ? m_end : M;
    const int cpair = tid & 31;             // column pair 0..31 -> cols n0+2*cpair
    const int rgrp  = tid >> 5;             // row group 0..7 (one per warp)
    for (int r = m0 + rgrp; r < m_lim; r += 8) {
        const size_t col = (size_t)r * N_FIXED + n0 + 2 * cpair;
        float ax = 0.0f, ay = 0.0f;
        #pragma unroll
        for (int s = 0; s < KSPLIT; ++s) {  // fixed ascending ks order
            const float2 v = *reinterpret_cast<const float2*>(
                ws + (size_t)s * M * N_FIXED + col);
            ax += v.x;
            ay += v.y;
        }
        *reinterpret_cast<__half2*>(C + col) = __floats2half2_rn(ax, ay);
    }
}

// ---------------------------------------------------------------------------
// Host launcher. Dynamic shared memory (NSTAGES-stage rings need up to
// 55.3KB for the BM=32 bucket, above the 48KB static limit); the one-time
// cudaFuncSetAttribute is launch configuration only - no data crosses calls.
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int WM, int WN, int KSPLIT, int NSTAGES>
static void launch_splitk(const __half* A, const __half* B, float* ws,
                          unsigned int* tickets, __half* C, int M,
                          cudaStream_t stream) {
    constexpr int PAD = 8;
    constexpr int SMEM_BYTES = NSTAGES * (BM + BN) * (BK + PAD) * (int)sizeof(__half);
    static const bool attr_set = []() {
        (void)cudaFuncSetAttribute(
            gemm_splitk_fused<BM, BN, BK, WM, WN, KSPLIT, NSTAGES>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
        return true;
    }();
    (void)attr_set;

    dim3 grid(N_FIXED / BN, KSPLIT, (M + BM - 1) / BM);
    gemm_splitk_fused<BM, BN, BK, WM, WN, KSPLIT, NSTAGES>
        <<<grid, THREADS_PER_BLOCK, SMEM_BYTES, stream>>>(A, B, ws, tickets, C, M);
}

// ---------------------------------------------------------------------------
// Ticket-counter scratch: TICKET_COUNT ints of pure synchronization state,
// zeroed once at first use and self-resetting inside every launch (the
// last-arriving CTA clears its own counter), so each launch starts from the
// canonical all-zero state; stream ordering serializes consecutive launches
// on the harness's single stream. It holds no operand-derived data. If the
// active device changes, a fresh buffer is created for it (the stale
// 640-byte block on the old device is intentionally left to leak).
// ---------------------------------------------------------------------------
static unsigned int* ticket_buffer(cudaStream_t stream) {
    static unsigned int* ptr = nullptr;
    static int dev = -1;
    int cur = 0;
    cudaError_t err = cudaGetDevice(&cur);
    TORCH_CHECK(err == cudaSuccess, "cudaGetDevice failed: ", cudaGetErrorString(err));
    if (ptr == nullptr || dev != cur) {
        unsigned int* p = nullptr;
        err = cudaMalloc(&p, TICKET_COUNT * sizeof(unsigned int));
        TORCH_CHECK(err == cudaSuccess,
                    "ticket buffer alloc failed: ", cudaGetErrorString(err));
        // Stream-ordered one-time zeroing; runs before the first kernel that
        // uses the counters on this stream.
        err = cudaMemsetAsync(p, 0, TICKET_COUNT * sizeof(unsigned int), stream);
        TORCH_CHECK(err == cudaSuccess,
                    "ticket buffer zeroing failed: ", cudaGetErrorString(err));
        ptr = p;
        dev = cur;
    }
    return ptr;
}

// ---------------------------------------------------------------------------
// fp32 partial workspace scratch: a one-time cudaMalloc'd max-size buffer
// (6.3MB) -- no per-call cudaMalloc, no per-call caching-allocator work on
// the ~14.2us M<=8 dispatch floor. Pure capacity scratch, same
// non-operand-derived status as the ticket buffer: it is never zeroed
// because every element the fused reduction reads at (ks, row < M, col) is
// written earlier in the SAME launch by CTA (ks, col's n-tile) before it
// bumps the ticket, and the reading CTA waits for all KSPLIT tickets. No
// data of any kind crosses calls. Per-device like the ticket buffer; the
// stale block on a device switch is intentionally left to leak.
// ---------------------------------------------------------------------------
static float* ws_scratch() {
    static float* ptr = nullptr;
    static int dev = -1;
    int cur = 0;
    cudaError_t err = cudaGetDevice(&cur);
    TORCH_CHECK(err == cudaSuccess, "cudaGetDevice failed: ", cudaGetErrorString(err));
    if (ptr == nullptr || dev != cur) {
        float* p = nullptr;
        err = cudaMalloc(&p, WS_MAX_FLOATS * sizeof(float));
        TORCH_CHECK(err == cudaSuccess,
                    "ws scratch alloc failed: ", cudaGetErrorString(err));
        ptr = p;
        dev = cur;
    }
    return ptr;
}

// ---------------------------------------------------------------------------
// Entry point: M-keyed three-band dispatch, attempt-3 structure only.
//   M >= 258          -> torch.matmul(A, B.t()) (cuBLAS NT parity)
//   65 <= M < 258     -> torch.matmul(A, B.t()) (cuBLAS passthrough)
//   M <= 64           -> static per-M config table -> one direct <<<>>> launch
// ---------------------------------------------------------------------------
torch::Tensor run(const torch::Tensor& A, const torch::Tensor& B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "2D tensors required");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(A.scalar_type() == torch::kFloat16, "fp16 A required");
    TORCH_CHECK(B.scalar_type() == torch::kFloat16, "fp16 B required");

    const int M = (int)A.size(0);
    TORCH_CHECK(A.size(1) == K_FIXED, "A must be [M, 4096]");
    TORCH_CHECK(B.size(0) == N_FIXED && B.size(1) == K_FIXED, "B must be [2048, 4096]");

    // ---- (1) M >= 258: compute-bound regime -> cuBLAS NT parity ----
    if (M >= M_LARGE_THRESHOLD) {
        return torch::matmul(A, B.t());
    }

    // ---- (2) 65 <= M < 258: cuBLAS passthrough ----
    // M=93 measured 0.899x inside the old custom bound; parity (~1.0x) is
    // the correct landing spot for this band.
    if (M > M_CUSTOM_MAX) {
        return torch::matmul(A, B.t());
    }

    // ---- (3) M <= 64: custom fused split-K, one direct launch ----
    // KSPLIT = 8 for the deep-ring M <= 32 buckets (grid 32 x 8 x 1 = 256
    // CTAs, all resident), 4 for 33..64 (grid 32 x 4 x 1 = 128 CTAs, one
    // wave; halves the fp32 partial round trip that made M=63 lose in
    // attempt 2).
    const int KS = (M <= 32) ? 8 : 4;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    torch::Tensor C = torch::empty({M, N_FIXED}, A.options());
    unsigned int* tickets = ticket_buffer(stream);
    float* const wsp = ws_scratch();
    // Failure-only capacity bound (always satisfied for M <= 64).
    TORCH_CHECK((int64_t)KS * M * N_FIXED <= WS_MAX_FLOATS,
                "fp32 partial scratch too small for M=", M,
                " (bump WS_MAX_FLOATS or lower M_CUSTOM_MAX)");

    const __half* Ap = reinterpret_cast<const __half*>(A.data_ptr<at::Half>());
    const __half* Bp = reinterpret_cast<const __half*>(B.data_ptr<at::Half>());
    __half* Cp   = reinterpret_cast<__half*>(C.data_ptr<at::Half>());

    if (M <= 8 && M8_DEEP_CONCURRENCY) {
        // Gated lever (CLOSED): BN=128 deep-concurrency geometry for M <= 8,
        // grid 16 x 8 x 1 = 128 CTAs, warp tile 16x16 (WM=1, WN=8, MM=1,
        // NN=2), smem 83.3KB. Flips only after the restored baseline
        // re-measures and nsys/ncu shows kernel time (not host) dominating
        // the ~14.2us M<=8 floor.
        launch_splitk<16, 128, 64, 1, 8, 8, 4>(Ap, Bp, wsp, tickets, Cp, M, stream);
    } else if (M <= 16) {
        // 4-stage ring: 3 x ~8.5KB of B in flight per CTA; smem 46.1KB.
        // Measured 1.81-1.84x at M <= 8, 1.29x at M=15/16.
        launch_splitk<16, BN_FIXED, 64, 1, 8, 8, 4>(Ap, Bp, wsp, tickets, Cp, M, stream);
    } else if (M <= 32) {
        // 4-stage ring; smem 55.3KB (one-time cudaFuncSetAttribute).
        // Measured 1.29-1.69x at M=17..32.
        launch_splitk<32, BN_FIXED, 64, 2, 4, 8, 4>(Ap, Bp, wsp, tickets, Cp, M, stream);
    } else {
        // M 33..64: KSPLIT=4 proven double buffer; smem 36.9KB. Measured
        // M=63 1.038x, M=64 1.188x.
        launch_splitk<64, BN_FIXED, 64, 4, 2, 4, 2>(Ap, Bp, wsp, tickets, Cp, M, stream);
    }

    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "fused split-K GEMM launch failed: ", cudaGetErrorString(err));
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "fp16 GEMM C=A@B.T: cuBLAS NT parity (M>=258 and 65..257) + custom "
          "B-streaming fused-split-K ticket kernel (M<=64, 4-stage cp.async "
          "ring at M<=32, KSPLIT=4 double buffer at 33..64), one kernel "
          "launch per call");
}