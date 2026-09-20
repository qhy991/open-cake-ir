/*
 * kernel.cu -- fp16 GEMM C = A @ B.T  (N=5120, K=2048, M variable 1..16294)
 * Target: NVIDIA B300 SXM6 AC (sm_103), CUDA 12.8+, torch 2.x.
 * (History: developed on RTX 4090 / sm_89; the mma/cp.async/ldmatrix
 * primitives below are the ones validated end-to-end on the scoring B300 --
 * exp/sol_r1_s1_0.bench.jsonl, 25/25 PASSED, min matched_ratio 1.0.)
 *
 * STRATEGY (r2-s3, "tensor_core_utilization"): DOUBLE PER-WARP MMA DENSITY in
 * the sub-crossover split-K kernel -- m16n8 -> m16n16 per warp -- for the
 * MT_TOTAL >= 2 instantiations only (the band M = 17..172). Previously each
 * of the 4 warps owned an 8-row B strip (SK_N_TILE = 32) and per k-step
 * issued 1 ldmatrix.x2 (B) + MT_TOTAL ldmatrix.x4 (A) + MT_TOTAL
 * mma.sync.m16n8k16 -- one A-fragment smem load per mma, competing for issue
 * slots with the cp.async B/A staging while the tensor pipe sat nearly idle.
 * The widened form is a SEPARATE instantiation (sk_kernel_w, SPLITS = 2
 * only): SK_N_TILE 32 -> 64 (grid.x 160 -> 80), warp w owns B rows
 * 16w..16w+15 as two 8-row sub-strips (brow = warp*16 + s*8 + (lane&7)), per
 * k-step two ldmatrix.x2 (one per sub-strip) + MT_TOTAL ldmatrix.x4 and
 * 2*MT_TOTAL mma -- every A fragment feeds TWO mma, halving A-fragment
 * ldmatrix instructions per mma (12/11 -> 13/22 at MT = 11) and cutting the
 * non-tensor-core smem-load issue in the k-step loop. Epilogue: the
 * quad-gather is unchanged per strip; each warp writes 16 columns
 * (cbase = n0 + warp*16, two SkHalf8/float4 packets per row, all
 * 16B-aligned). NUMERICS IDENTICAL: the same
 * mma.sync.aligned.m16n8k16.f32.f16.f16.f32 instruction and fp32 register
 * accumulation produce the same value per element -- only operand reuse and
 * CTA ownership of B rows change; C coverage is unchanged. The B tile per
 * pipeline stage doubles (8KB -> 16KB/stage; MT = 8-11 footprint
 * 168KB -> 193KB, still < 227KB at 1 CTA/SM); to avoid halving CTA-level
 * parallelism the widened form is instantiated ONLY with SPLITS = 2
 * (80x2 = 160 CTAs >= 148 SMs) while the SPLITS = 1 form stays at
 * SK_N_TILE = 32. The MT_TOTAL = 1 instantiations (M <= 16, the
 * r1_s1_0-identical stream-floor form) stay byte-identical, and the existing
 * per-M decide_sk_impl gate (>= 99.5% matched at task tolerance, 2% timing
 * margin, min-of-5 CUDA events, cuBLAS fallback) picks the winner per M at
 * warmup among FOUR candidates (cuBLAS, narrow SPLITS=1, narrow SPLITS=2,
 * widened SPLITS=2) -- any M where the widened form loses keeps the previous
 * dispatch, so worst case is parity. The M > 258 cuBLAS ALGO_MAP dispatch is
 * untouched.
 *
 * STRATEGY (r2-s2, "vectorized_memory_access"): the r2-s1 candidate is kept
 * with THREE width-only edits, all outside the mma k-loop (the mma.sync
 * sequence, fp32 register accumulation order, B/A cp.async.cg 16B staging and
 * the M > 258 cuBLAS dispatch are untouched). They widen the sub-16B global
 * transactions that serve exactly the M <= 258 band's non-B traffic: the
 * SPLITS=2 fp32 workspace round trip (2*M*N*4 write+read ~= 14.1 MB at M=172,
 * ~40% of band traffic) and both epilogue stores. The band measures 15.6-22
 * us against a 3.26 us B-stream floor (~21% of the HBM roofline) on what is
 * a pure streaming workload, so per-thread transaction width and LSU issue
 * rate set the achieved fraction of peak bandwidth:
 *   (1) sk_reduce_kernel: one thread per EIGHT adjacent columns instead of
 *       one per column-pair -- two 16B float4 loads per split, 8 fp32
 *       accumulators summed in the same fixed ascending split order, one
 *       16B packed 8-half store. Grid shrinks M*2560 -> M*640 threads; LSU
 *       instructions per byte drop 2x (loads) / 4x (stores).
 *   (2) sk_kernel SPLITS>1 epilogue ws store: quad-level __shfl_sync repack
 *       (the quad's lanes t=0..3 already hold contiguous cols 2t..2t+1 of
 *       rows g and g+8) so each lane issues one 16B float4 instead of 4x8B
 *       float2 per quad-row.
 *   (3) sk_kernel SPLITS==1 epilogue C store: same repack, one 16B store of
 *       8 halves per row instead of 4x4B.
 * NUMERICS BIT-IDENTICAL: the shuffle only moves registers (same values),
 * the same addresses are stored (all widened addresses 16B-aligned: C row
 * 10240 B, ws row 20480 B, n0+warp*8 column offset a multiple of 8 halves),
 * and the summation/conversion order per element is unchanged; the existing
 * per-M decide_sk_impl gate (>= 99.5% matched, 2% timing margin, cuBLAS
 * fallback) re-validates at warmup, so worst case is parity.
 *
 * STRATEGY (r2-s1, "register_pressure_reduction"): the r2-s0 sub-crossover
 * swap (B-streaming split-K skinny kernel for M <= 258, byte-identical
 * cuBLAS algo-map dispatch for M > 258; geomean 1.2066x, 25/25,
 * exp/sol_r2_s0_1.bench.jsonl) is retained UNCHANGED except for two
 * register-file changes to sk_kernel<MT_TOTAL,SPLITS>, the custom path that
 * owns the 16 latency-bound sub-crossover workloads (M <= 172, measured
 * 13.8-18.5 us vs the 3.26 us B-stream floor = 18-24% of the HBM roofline;
 * grid 160xSPLITS CTAs of 128 threads over 148 SMs -> ~4 warps/SM, ~6% warp
 * occupancy at 1 resident CTA/SM, and SPLITS=2's 2.17 waves collapse to ~1
 * wave at >= 2 CTAs/SM):
 *   (1) The bare __launch_bounds__(SK_THREADS) is replaced by a
 *       per-instantiation minBlocksPerMultiprocessor that mirrors the smem
 *       co-residency budget derivable from the SK_* constants (227 KB
 *       usable/SM); see SK_MIN_BLOCKS / SK_MIN_BLOCKS_W.
 *   (2) The live register set of the k-step mma loop is trimmed: the
 *       warp/lane-invariant fragment-address scalars (brow, arow, bsel,
 *       ahalf and their pre-multiplied byte offsets) are hoisted out of the
 *       unrolled body so their live ranges stop interleaving with the
 *       fp32 accumulator set.
 *
 * STRATEGY (r2-s0, inherited): SUB-CROSSOVER ALGORITHM SWAP. B-streaming
 * split-K skinny GEMM replaces the dense-tile cuBLAS GEMM for M <= 258; the
 * cuBLAS TN algo-map dispatch is kept byte-identical for the ceiling-
 * saturated M > 258 region. Evidence (exp/sol_r1_s1_0.bench.jsonl, B300,
 * 25/25, geomean 1.2216x): the 9 compute-bound workloads (M >= 8828) run at
 * 96-97% of the measured 1658.6 TFLOP/s cuBLAS fp16 ceiling (TASK.json
 * roofline_b300) -- no algorithmic headroom there -- while the 16 workloads
 * below the M_crossover = 258 B-stream knee sat on a flat ~17.5-18 us
 * plateau against a 3.26 us B-stream floor (B = 20.97 MB at 6.434 TB/s
 * HBM): the dense-tile algorithm re-fetches B once per 16-row m-tile (up
 * to 11x, ~230 MB of L2 traffic at M=172) and burns tile work the skinny
 * shape never needed.
 *
 * THE SWAPPED-IN ALGORITHM (M <= 258):
 *   (1) B-streaming, exactly once: every (n, k) element of B belongs to
 *       exactly ONE CTA's pipeline stage, so B's 20.97 MB streams from HBM
 *       exactly once per call at full rate across the 148 B300 SMs. The
 *       k-stage loop is OUTERMOST and the m-tile loop is INSIDE it: each
 *       128-deep B k-stage is consumed by ALL ceil(M/16) m-tiles before its
 *       buffer recycles. The B smem tile is the XOR-swizzled pad-free
 *       256B-row SoA layout measured fastest (r1_s1_0): cp.async source
 *       addresses stay byte-identical and coalesced; only the smem
 *       placement swizzles.
 *   (2) A resident: A (M x 2048, <= 1.05 MB fp16 at M=258) is L2-resident;
 *       each CTA stages its k-stage A slice (ALL M rows, 128 k deep, padded
 *       272B rows) in smem D-deep ahead of use via the same cp.async groups,
 *       so an A element is read once per CTA and never re-fetched per
 *       m-tile.
 *   (3) Split-K, deterministic two-pass reduction, no atomics: with
 *       SPLITS=2, pass 1 accumulates fp32 partials in registers and writes
 *       them to a [SPLITS][M][N] fp32 workspace; pass 2 (sk_reduce_kernel)
 *       sums the splits in a fixed ascending order and fuses the single
 *       fp16 rounding at the store. SPLITS=1 is the degenerate single-split
 *       form: the reduction fuses into pass 1's epilogue (direct fp16
 *       store) -- for M <= 16 that instantiation is work-identical to the
 *       validated r1_s1_0 kernel that already sits on the B-stream floor.
 *   (4) Dispatch: for M <= 258 the gate measures every custom candidate
 *       against the cuBLAS path this M would otherwise take (one-time per M
 *       on the warmup call, CUDA-event min-of-5, plus a >= 99.5%-matched
 *       correctness check at task tolerance) and caches the winner; the
 *       resident-A form covers M <= 176 (smem budget); M in 177..258 and any
 *       gate loss keep cuBLAS. For M > 258 the existing ALGO_MAP /
 *       CUBLAS_GEMM_DEFAULT_TENSOR_OP cuBLAS TN dispatch is byte-identical
 *       to the previous round (no-regression; unseen M falls back to
 *       DEFAULT == torch.matmul parity).
 *
 * ROBUSTNESS / NO-REGRESSION GUARANTEE:
 *   Known M > 258 uses the measured algo; unseen M > 258 falls back to
 *   CUBLAS_GEMM_DEFAULT_TENSOR_OP (== torch.matmul). M <= 258 uses a custom
 *   kernel only where it MEASURED a win (2% margin) AND matched the cuBLAS
 *   output at tolerance; everything else keeps the cuBLAS path. All gating
 *   runs once per M per process, never in steady state. No cross-call
 *   caching of B or any input-derived data (the harness shifts pointers);
 *   the only cross-call state is the per-M dispatch decision.
 *
 * Correctness: fp32 accumulation (registers) + one fp16 rounding at the
 *   store on the single-split path; fp32 workspace partials + one fp16
 *   rounding in the fused reduce on the split path -- CUBLAS_COMPUTE_32F
 *   numerics class, atol=1e-2, rtol=1e-2, 99% matched.
 * Harness: entry_point="kernel.cu::forward" (also bound as "run"),
 * forward/run(A,B)->C, return-value style.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/TensorOptions.h>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <mutex>

static constexpr int N_FIXED = 5120;
static constexpr int K_FIXED = 2048;

// ---------------------------------------------------------------------------
// cuBLAS machinery (UNCHANGED / byte-identical to the previous round: the
// M > 258 dispatch and every fallback path below run exactly as before).
// ---------------------------------------------------------------------------

// Persistent cuBLAS handle (creation is ~expensive; reuse across calls).
static cublasHandle_t s_handle = nullptr;
static bool s_handle_init = false;
static std::mutex s_mu;

static cublasHandle_t get_handle(cudaStream_t stream) {
    if (!s_handle_init) {
        cublasStatus_t st = cublasCreate(&s_handle);
        TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasCreate failed: ", (int)st);
        cublasSetMathMode(s_handle, CUBLAS_TENSOR_OP_MATH);
        s_handle_init = true;
    }
    cublasSetStream(s_handle, stream);
    return s_handle;
}

// Per-M cuBLAS algorithm map (TN layout, fp32 accum).
//   -1  -> CUBLAS_GEMM_DEFAULT_TENSOR_OP  (== torch.matmul; guaranteed parity)
//   0..23 -> CUBLAS_GEMM_ALGO0_TENSOR_OP + idx
// Each entry was the fastest of all 24 tensor-op algos for that exact M
// (measured on the original target, seed=200, fresh inputs), all validated
// correct (100% matched). These serve as-is; the gate below independently
// decides whether a custom kernel beats this path per M.
struct AlgoEntry { int M; int algo; };

static const AlgoEntry ALGO_MAP[] = {
    {    1,  8 }, {    2,  6 }, {    4,  1 }, {    5,  2 }, {    6,  3 },
    {    8,  4 }, {   16,  6 }, {   17,  5 }, {   25, -1 }, {   32,  5 },
    {   34, 13 }, {   63,  0 }, {   64,  2 }, {   93,  0 }, {  128, 12 },
    {  172,  2 }, {  289, 14 }, {  492,  0 }, {  952, 11 }, { 8828,  1 },
    {11006,  8 }, {12251,  7 }, {12853,  6 }, {14915,  5 }, {16294, -1 },
};
static constexpr int ALGO_MAP_LEN = sizeof(ALGO_MAP) / sizeof(AlgoEntry);

static int lookup_algo(int M) {
    for (int i = 0; i < ALGO_MAP_LEN; i++) {
        if (ALGO_MAP[i].M == M) return ALGO_MAP[i].algo;
    }
    return -1;  // CUBLAS_GEMM_DEFAULT_TENSOR_OP -> parity with torch.matmul
}

static cublasGemmAlgo_t to_algo(int idx) {
    if (idx < 0) return CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    return (cublasGemmAlgo_t)(CUBLAS_GEMM_ALGO0_TENSOR_OP + idx);
}

// cuBLAS TN: C = A @ B^T row-major <=> col-major:
//   opA=T on B(lda=K), opB=N on A(ldb=K), m=N, n=M, k=K, ldc=N
// B read directly as [N,K] row-major (transpose-free; B^T folded into TN load).
static cublasStatus_t cublas_tn(cublasHandle_t handle,
                                const __half* A, const __half* B, __half* C,
                                int M, cublasGemmAlgo_t algo) {
    const float alpha = 1.0f, beta = 0.0f;
    return cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        N_FIXED, M, K_FIXED,
                        &alpha, B, CUDA_R_16F, K_FIXED,
                        A, CUDA_R_16F, K_FIXED,
                        &beta, C, CUDA_R_16F, N_FIXED,
                        CUBLAS_COMPUTE_32F, algo);
}

// ---------------------------------------------------------------------------
// CUDA-event timing helper (one warmup call + timed reps, min of reps).
// Used ONLY for one-time-per-M dispatch decisions, never in the hot path.
// ---------------------------------------------------------------------------
static float time_reps(int reps, cudaStream_t stream,
                       const std::function<void()>& fn) {
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    if (cudaEventCreate(&e0) != cudaSuccess || cudaEventCreate(&e1) != cudaSuccess) {
        if (e0) cudaEventDestroy(e0);
        if (e1) cudaEventDestroy(e1);
        return 1e30f;  // caller treats as "do not switch"
    }
    fn();  // warmup
    cudaStreamSynchronize(stream);
    float best = 1e30f;
    for (int r = 0; r < reps; ++r) {
        cudaEventRecord(e0, stream);
        fn();
        cudaEventRecord(e1, stream);
        cudaEventSynchronize(e1);
        float ms = 1e30f;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return best;
}

// ---------------------------------------------------------------------------
// (1) B-streaming split-K skinny kernel (M <= SK_MAX_MT * 16 = 176 in its
// A-resident form; dispatched for the M <= 258 sub-crossover band).
//
// C is [M,5120] fp16, A is [M,2048] fp16 row-major, B is [5120,2048] fp16
// row-major. Both operands are K-contiguous -> the mma-native TN layout:
// A rows (m) and B rows (n) are staged [rows][K]-major in smem and consumed
// with plain (non-.trans) ldmatrix:
//   A fragment (m16k16): ldmatrix.x4, lane l addresses matrix m_l = l/8:
//     row = (m_l & 1)*8 + (l & 7), 16B chunk at byte 32*i + 16*(m_l >> 1) of
//     the 128-k stage. (Validated path, unchanged.)
//   B fragment (k16n8): ldmatrix.x2, lane l (lanes 0..15) addresses the
//     warp's B row base + (l & 7), logical 16B chunk 2*i + (l >> 3), at the
//     XOR-swizzled position (chunk ^ (row & 7)) of the 256B row. (Validated
//     path, unchanged; the r2-s3 widened form shifts the row base by
//     warp*16 + s*8 instead of warp*8 -- the same addressing pattern per
//     8-row strip, applied twice.)
//   mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32, fp32 accumulators.
//
// TWO SHAPES share this machinery:
//   * sk_kernel<MT_TOTAL, SPLITS>: the validated narrow form. CTA tile
//     SK_N_TILE = 32 B rows; each of the 4 warps owns ONE 8-row strip and
//     per k-step issues 1 ldmatrix.x2 (B) + MT_TOTAL ldmatrix.x4 (A) +
//     MT_TOTAL mma. Used for every SPLITS=1 instantiation and BOTH SPLITS=2
//     instantiations at MT_TOTAL = 1 (those stay byte-identical to the
//     previous round).
//   * sk_kernel_w<MT_TOTAL> (r2-s3): the widened m16n16-per-warp form,
//     SPLITS = 2 and MT_TOTAL >= 2 only. CTA tile SK_N_TILE_W = 64 B rows
//     (grid.x 80, grid.y 2 -> 160 CTAs >= 148 SMs, so CTA-level parallelism
//     is not halved); each warp owns TWO 8-row sub-strips and per k-step
//     issues 2 ldmatrix.x2 (B) + MT_TOTAL ldmatrix.x4 (A) + 2*MT_TOTAL mma
//     -- every A fragment feeds two mma.
//
// The k-stage loop (KS/128 stages of 128 k) is OUTERMOST; each B stage is
// consumed by all MT_TOTAL m-tiles before its buffer recycles, so no B
// re-fetch per m-tile. A is the resident operand: the k-stage A slice
// covering ALL M rows is staged D-deep in smem through the same
// one-commit-group-per-stage pipeline (rows >= M zero-filled via cp.async
// src-size 0 with a clamped source address: no OOB global reads, bit-exact
// padding). Group discipline is the validated one: D groups in flight,
// wait_group D-1 retires stage `st` at each iteration top, the
// end-of-iteration barrier orders every read of the buffers before stage
// st+D reissues into them (st+D === st mod D).
//
// D = 6 for MT_TOTAL == 1 (the M <= 16 instantiation is work-identical to
// the validated r1_s1_0 kernel) and D = 3 for MT_TOTAL >= 2.
//
// REGISTER BUDGET: both kernels are compiled with a per-instantiation
// __launch_bounds__(SK_THREADS, minBlocks) whose minBlocksPerMultiprocessor
// mirrors the smem co-residency budget (227 KB usable/SM); see SK_MIN_BLOCKS
// (narrow) and SK_MIN_BLOCKS_W (widened).
//
// Epilogue: SPLITS == 1 -> fp32 accumulators rounded once to fp16 and stored
// straight to C. SPLITS > 1 -> fp32 partials stored to ws[si][m][n]; the
// deterministic pass-2 kernel (sk_reduce_kernel) then sums the splits in
// fixed ascending split order and fuses the single fp16 rounding. No
// atomics anywhere. (r2-s2: both store paths are widened to 16B global
// transactions via a quad-level shuffle repack; r2-s3: the widened kernel
// writes two 16-column strips per warp through the same per-strip repack.)
// ---------------------------------------------------------------------------
static constexpr int SK_N_TILE     = 32;   // narrow form: B rows / C cols per CTA
static constexpr int SK_N_TILE_W   = 64;   // r2-s3 widened form (SPLITS=2, MT>=2)
static constexpr int SK_K_TILE     = 128;  // k depth staged per pipeline step
static constexpr int SK_THREADS    = 128;  // 4 warps
static constexpr int SK_M_TILE     = 16;   // A rows per m-tile (mma m extent)
static constexpr int SK_MAX_MT     = 11;   // ceil(176/16): resident-A budget
static constexpr int SK_BAND_M     = 258;  // plan's M_crossover (TASK.json)
static constexpr int SK_MAX_SPLITS = 2;
static constexpr int SK_ROW_BYTES     = SK_K_TILE * 2;      // 256B of k per row
static constexpr int SK_A_ROW_STRIDE  = SK_ROW_BYTES + 16;  // 272B: validated
                                                            // padded A layout
                                                            // (ldmatrix.x4 +
                                                            // cp.async, kept)
static constexpr int SK_B_ROW_STRIDE  = SK_ROW_BYTES;       // 256B: XOR-swizzled
                                                            // SoA B tile, pad
                                                            // removed
// Pipeline depth per m-tile count (mirrored by sk_smem_bytes / sk_w_smem_bytes
// on the host).
static constexpr int SK_DEPTH(int mt) { return (mt == 1) ? 6 : 3; }

// Co-resident CTAs per SM the NARROW dynamic-smem footprint permits
// (227 KB usable/SM), declared to ptxas as minBlocksPerMultiprocessor so the
// per-thread register file is capped at 65536 / (blocks * SK_THREADS):
//   MT_TOTAL=1   : D=6,  75,264 B/CTA -> 3 CTAs (225,792 B)  -> 170-reg cap
//   MT_TOTAL=2   : D=3,  50,688 B/CTA -> 4 CTAs (202,752 B)  -> 128-reg cap
//   MT_TOTAL=3-4 : D=3, <= 76,800 B/CTA -> 3 CTAs (<= 230,400 B)
//   MT_TOTAL=5-7 : D=3, <= 115,968 B/CTA -> 2 CTAs (<= 231,936 B)
//   MT_TOTAL=8-11: D=3,  168,192 B/CTA -> 1 CTA (smem-forced; reg cap 512,
//                  i.e. the architecture per-thread maximum anyway)
static constexpr int SK_MIN_BLOCKS(int mt) {
    return (mt == 1) ? 3
         : (mt == 2) ? 4
         : (mt <= 4) ? 3
         : (mt <= 7) ? 2
         : 1;
}

// Co-resident CTAs per SM the WIDENED (r2-s3) dynamic-smem footprint permits
// (227 KB usable/SM; D = 3, per stage: B tile 64*256 = 16,384 B + A slice
// mt*16*272 = mt*4,352 B):
//   MT_TOTAL=2   :  75,264 B/CTA -> 3 CTAs (225,792 B)  -> 170-reg cap
//   MT_TOTAL=3-5 :  88,320..114,432 B/CTA -> 2 CTAs (<= 228,864 B) -> 256-reg
//   MT_TOTAL=6-11: 127,488..192,768 B/CTA -> 1 CTA (two CTAs exceed the
//                  budget from MT=6 up; reg cap 512 = architectural max).
// MT=8-11 lands at 193 KB -- still under the 227 KB single-CTA budget.
static constexpr int SK_MIN_BLOCKS_W(int mt) {
    return (mt <= 2) ? 3
         : (mt <= 5) ? 2
         : 1;
}

// XOR-swizzled SoA placement of the 16B chunk holding B[row][chunk*8 .. +7]
// within a pipeline buffer: bufB + row*256 + (chunk ^ (row & 7)) * 16.
// Applied identically on the cp.async write side and the ldmatrix read side.
__device__ __forceinline__ int b_chunk_off(int row, int chunk) {
    return row * SK_B_ROW_STRIDE + (chunk ^ (row & 7)) * 16;
}

__device__ __forceinline__ void cp_async16(void* dst_smem, const void* src, int src_bytes) {
    unsigned d = (unsigned)__cvta_generic_to_shared(dst_smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(d), "l"(src), "r"(src_bytes));
}

__device__ __forceinline__ void cp_commit() {
    asm volatile("cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__ void cp_wait_n() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void cp_wait_all() {
    asm volatile("cp.async.wait_group 0;\n");
}

__device__ __forceinline__ void ldmatrix_a(unsigned& r0, unsigned& r1,
                                           unsigned& r2, unsigned& r3,
                                           const void* addr) {
    unsigned a = (unsigned)__cvta_generic_to_shared(addr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}

__device__ __forceinline__ void ldmatrix_b(unsigned& r0, unsigned& r1,
                                           const void* addr) {
    unsigned a = (unsigned)__cvta_generic_to_shared(addr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1) : "r"(a));
}

// mma.sync m16n8k16 (fp32 accumulators; passed by reference so the per-m-tile
// register array never has its address taken).
__device__ __forceinline__ void mma_m16n8k16(float& c0, float& c1,
                                             float& c2, float& c3,
                                             unsigned a0, unsigned a1,
                                             unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// 16B packet of eight fp16 values -- the widened (r2-s2) epilogue / reduction
// store unit. The __align__(16) keeps the by-value store a single 16B global
// transaction (st.global.v4.b32).
struct __align__(16) SkHalf8 { __half2 h0, h1, h2, h3; };

// Stage one 128-deep k-slice of B (NT rows) AND of A (ALL M rows of this
// CTA's k-slice, rows >= M zero-filled) into pipeline buffer `buf` as ONE
// commit group. NT is the CTA's B-row count: SK_N_TILE (32) for the narrow
// sk_kernel, SK_N_TILE_W (64) for the r2-s3 widened sk_kernel_w. `stage` is
// the global stage index within the CTA's k-slice; stages past the slice
// (the wait-accounting padding tail) commit an EMPTY group: exactly one
// group is committed per k-stage, which keeps the wait_group(D-1) FIFO
// accounting invariant, and their buffers are never read. B destinations go
// through the XOR swizzle; A keeps the validated 272B padded layout.
template <int MT_TOTAL, int SPLITS, int NT>
__device__ __forceinline__ void sk_issue_stage(
        char* smB, char* smA, int buf, int stage,
        const __half* __restrict__ A, int M,
        const __half* __restrict__ B, int n0, int kbase, int tid) {
    constexpr int M_PAD = MT_TOTAL * SK_M_TILE;
    constexpr int NST   = (K_FIXED / SPLITS) / SK_K_TILE;
    const bool valid = stage < NST;
    const int k0 = valid ? kbase + stage * SK_K_TILE : kbase;  // keep srcs
                                                               // in-bounds
    char* bufB = smB + buf * (NT * SK_B_ROW_STRIDE);
    char* bufA = smA + buf * (M_PAD * SK_A_ROW_STRIDE);
    // B: NT rows x 16 chunks (16B) = NT*16 cp.async, (NT*16)/SK_THREADS per
    // thread (4 at NT=32, 8 at NT=64).
    #pragma unroll
    for (int j = 0; j < (NT * 16) / SK_THREADS; ++j) {
        const int op = tid + SK_THREADS * j;
        const int row = op >> 4, chunk = op & 15;
        const __half* src = B + (size_t)(n0 + row) * K_FIXED
                            + (size_t)k0 + chunk * 8;
        cp_async16(bufB + b_chunk_off(row, chunk), src, valid ? 16 : 0);
    }
    // A: M_PAD rows x 16 chunks = 2*MT_TOTAL per thread; rows >= M
    // (out-of-range rows of the last m-tile) zfill via src-size 0.
    #pragma unroll
    for (int j = 0; j < (M_PAD * 16) / SK_THREADS; ++j) {
        const int op = tid + SK_THREADS * j;
        const int row = op >> 4, chunk = op & 15;
        const int srow = row < M ? row : (M > 0 ? M - 1 : 0);
        const __half* src = A + (size_t)srow * K_FIXED
                            + (size_t)k0 + chunk * 8;
        cp_async16(bufA + row * SK_A_ROW_STRIDE + chunk * 16, src,
                   (row < M && valid) ? 16 : 0);
    }
    cp_commit();
}

// NARROW form (validated path, byte-identical to the previous round for the
// SPLITS=1 instantiations and both MT_TOTAL=1 instantiations).
template <int MT_TOTAL, int SPLITS>
__global__ __launch_bounds__(SK_THREADS, SK_MIN_BLOCKS(MT_TOTAL))
void sk_kernel(const __half* __restrict__ A,   // [M, K]
               const __half* __restrict__ B,   // [N, K]
               float* __restrict__ ws,         // [SPLITS, M, N] (SPLITS > 1)
               __half* __restrict__ C,         // [M, N]         (SPLITS == 1)
               int M) {
    constexpr int D     = SK_DEPTH(MT_TOTAL);
    constexpr int M_PAD = MT_TOTAL * SK_M_TILE;
    constexpr int NST   = (K_FIXED / SPLITS) / SK_K_TILE;

    extern __shared__ __align__(16) char smem[];
    char* smB = smem;
    char* smA = smem + D * (SK_N_TILE * SK_B_ROW_STRIDE);

    const int tid    = threadIdx.x;
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int n0     = blockIdx.x * SK_N_TILE;
    const int si     = blockIdx.y;                 // k-split id
    const int kbase  = si * (K_FIXED / SPLITS);

    // fp32 accumulators for EVERY m-tile, live across the whole k walk
    // (44 floats/thread at MT_TOTAL = 11).
    float acc[MT_TOTAL][4];
    #pragma unroll
    for (int mt = 0; mt < MT_TOTAL; ++mt) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc[mt][i] = 0.f;
    }

    // Hoisted warp/lane-invariant fragment-address scalars (r2-s1 register-
    // pressure trim): every value below is invariant across BOTH the k-stage
    // loop and the unrolled k-step mma loop. Computing them once here, ahead
    // of the pipeline, keeps their live ranges from interleaving with the
    // acc[MT_TOTAL][4] fp32 accumulator set inside the unrolled body, which
    // trims the peak live register set ptxas must accommodate under the
    // SK_MIN_BLOCKS register cap. All resulting addresses are byte-identical
    // to the previous per-iteration formulation:
    //   B: b_row + ((bc8 * 16) ^ b_swz) == b_chunk_off(brow, bc8), since XOR
    //      of two 4-bit values commutes with the 4-bit left shift
    //      ((x ^ y) << 4 == (x << 4) ^ (y << 4) for x, y < 16).
    //   A: a_row + abyte + mt*(SK_M_TILE*SK_A_ROW_STRIDE)
    //        == (size_t)(mt*SK_M_TILE + arow)*SK_A_ROW_STRIDE + abyte.
    const int brow  = warp * 8 + (lane & 7);               // B row in CTA tile
    const int arow  = ((lane >> 3) & 1) * 8 + (lane & 7);  // A row in m-tile
    const int bsel  = lane >> 3;                           // B 16B-chunk
                                                           // selector bit
    const int ahalf = lane >> 4;                           // A 16B-half
                                                           // selector
    const int b_row = brow * SK_B_ROW_STRIDE;              // B row byte offset
    const int b_swz = (brow & 7) * 16;                     // XOR-swizzle
                                                           // pattern, pre-
                                                           // shifted 4 bits
    const int a_row = arow * SK_A_ROW_STRIDE;              // A row byte offset

    // Prologue: fill the first D stage buffers (one commit group each).
    #pragma unroll
    for (int s = 0; s < D; ++s) {
        sk_issue_stage<MT_TOTAL, SPLITS, SK_N_TILE>(smB, smA, s, s, A, M, B,
                                                    n0, kbase, tid);
    }

    // k-stage loop OUTERMOST: B[st] is fetched once and consumed by ALL
    // m-tiles; its buffer is recycled for B[st+D] only after this stage's
    // end-of-iteration barrier ordered every read of it.
    #pragma unroll 1
    for (int st = 0; st < NST; ++st) {
        // Oldest commit group (stage st) has completed; exactly D-1 groups
        // remain outstanding.
        cp_wait_n<D - 1>();
        __syncthreads();
        // Per-stage buffer base pointers: computed once per stage and kept
        // live across the unrolled k-step body below (hoisted out of it, so
        // their offset arithmetic is not re-materialized per k-step).
        const char* bufB = smB + (st % D) * (SK_N_TILE * SK_B_ROW_STRIDE);
        const char* bufA = smA + (st % D) * (M_PAD * SK_A_ROW_STRIDE);

        // 8 mma k-steps per stage; the B fragment is loaded ONCE per k-step
        // and reused by every m-tile of this stage (the B stream is the
        // shared, persistent operand; A restages per m-tile from the
        // resident k-stage slice). Only the k-step-varying terms (bc8's low
        // bits, abyte's 32*i term) and the unrolled m-tile constant offsets
        // are computed inside this body.
        #pragma unroll
        for (int i = 0; i < SK_K_TILE / 16; ++i) {
            const int bc8   = (2 * i + bsel) & 15;
            const int b_off = b_row + (bc8 * 16 ^ b_swz);
            unsigned b0, b1;
            ldmatrix_b(b0, b1, bufB + b_off);

            // A fragment: ldmatrix.x4 (m16 x k16) at this m-tile's row
            // offset inside the resident all-M k-stage slice; the m-tile
            // term is the compile-time constant mt*(SK_M_TILE*272).
            const int abyte = 32 * i + 16 * ahalf;
            #pragma unroll
            for (int mt = 0; mt < MT_TOTAL; ++mt) {
                unsigned a0, a1, a2, a3;
                ldmatrix_a(a0, a1, a2, a3,
                           bufA + a_row + abyte
                                + mt * (SK_M_TILE * SK_A_ROW_STRIDE));
                mma_m16n8k16(acc[mt][0], acc[mt][1], acc[mt][2], acc[mt][3],
                             a0, a1, a2, a3, b0, b1);
            }
        }
        __syncthreads();  // all warps done reading stage st's A and B buffers
        sk_issue_stage<MT_TOTAL, SPLITS, SK_N_TILE>(smB, smA, (st + D) % D,
                                                    st + D, A, M, B, n0,
                                                    kbase, tid);
    }

    // Drain the (empty) tail padding groups; nothing else reads smem.
    cp_wait_all();
    __syncthreads();

    // Epilogue. m16n8 accumulator layout: lane l holds c0,c1 at (row l>>2,
    // cols 2t, 2t+1) and c2,c3 at (row (l>>2)+8, same cols), t = l&3, within
    // the warp's 8-column strip -- so the quad's four lanes (t = 0..3)
    // already hold ALL eight columns of both rows, as contiguous column
    // pairs. r2-s2 widens the global stores: a quad-level __shfl_sync repack
    // moves those pairs into per-lane 16B packets before the store. The
    // shuffle only moves registers -- values, addresses, summation and
    // rounding order are unchanged -- and every lane of the quad executes
    // it unconditionally (the row guards below wrap only the stores, so the
    // full-warp __shfl_sync mask never diverges). Padded rows (global row
    // >= M) are never stored. SPLITS == 1: one fp16 rounding per element,
    // one 16B store of the row's 8 halves. SPLITS > 1: fp32 partials to
    // ws[si][m][n], one 16B float4 per lane, for the deterministic pass-2
    // reduction. All widened addresses are 16B-aligned: C row stride 10240 B,
    // ws row stride 20480 B, and the cbase column offset is a multiple of 8
    // halves (= 16 B).
    const int g = lane >> 2, t = lane & 3;
    const int q0 = lane & ~3;                       // quad's t == 0 lane id
    const size_t cbase = (size_t)n0 + warp * 8;     // warp's 8-col strip
    #pragma unroll
    for (int mt = 0; mt < MT_TOTAL; ++mt) {
        const int r0 = mt * SK_M_TILE + g;
        // Quad gather: v0[c] = column c of row r0 (lives on lane
        // q0 + (c >> 1), register (c & 1) ? c1 : c0); v8[c] = column c of
        // row r0 + 8 (c2/c3). Every lane ends up with the full 8 columns of
        // both rows; the stores below then pick one 16B packet per lane.
        float v0[8], v8[8];
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            v0[c] = __shfl_sync(0xffffffffu, (c & 1) ? acc[mt][1] : acc[mt][0],
                                q0 + (c >> 1));
            v8[c] = __shfl_sync(0xffffffffu, (c & 1) ? acc[mt][3] : acc[mt][2],
                                q0 + (c >> 1));
        }
        if (SPLITS == 1) {
            // One 16B store of 8 halves per row (was 4x 4B half2): t == 0
            // covers row r0, t == 1 covers row r0 + 8. Per-value
            // __float2half rounding identical to the validated path.
            if (t == 0 && r0 < M) {
                SkHalf8 v;
                v.h0 = __halves2half2(__float2half(v0[0]), __float2half(v0[1]));
                v.h1 = __halves2half2(__float2half(v0[2]), __float2half(v0[3]));
                v.h2 = __halves2half2(__float2half(v0[4]), __float2half(v0[5]));
                v.h3 = __halves2half2(__float2half(v0[6]), __float2half(v0[7]));
                *reinterpret_cast<SkHalf8*>(
                        C + (size_t)r0 * N_FIXED + cbase) = v;
            } else if (t == 1 && r0 + 8 < M) {
                SkHalf8 v;
                v.h0 = __halves2half2(__float2half(v8[0]), __float2half(v8[1]));
                v.h1 = __halves2half2(__float2half(v8[2]), __float2half(v8[3]));
                v.h2 = __halves2half2(__float2half(v8[4]), __float2half(v8[5]));
                v.h3 = __halves2half2(__float2half(v8[6]), __float2half(v8[7]));
                *reinterpret_cast<SkHalf8*>(
                        C + (size_t)(r0 + 8) * N_FIXED + cbase) = v;
            }
        } else {
            // One 16B float4 per lane (was 4x 8B float2 per quad-row):
            // t = 0/1 cover row r0's cols 0..3 / 4..7, t = 2/3 cover row
            // r0 + 8's likewise.
            const size_t wbase = (size_t)si * M * N_FIXED;
            if (t == 0 && r0 < M) {
                *reinterpret_cast<float4*>(
                        ws + wbase + (size_t)r0 * N_FIXED + cbase) =
                    make_float4(v0[0], v0[1], v0[2], v0[3]);
            } else if (t == 1 && r0 < M) {
                *reinterpret_cast<float4*>(
                        ws + wbase + (size_t)r0 * N_FIXED + cbase + 4) =
                    make_float4(v0[4], v0[5], v0[6], v0[7]);
            } else if (t == 2 && r0 + 8 < M) {
                *reinterpret_cast<float4*>(
                        ws + wbase + (size_t)(r0 + 8) * N_FIXED + cbase) =
                    make_float4(v8[0], v8[1], v8[2], v8[3]);
            } else if (t == 3 && r0 + 8 < M) {
                *reinterpret_cast<float4*>(
                        ws + wbase + (size_t)(r0 + 8) * N_FIXED + cbase + 4) =
                    make_float4(v8[4], v8[5], v8[6], v8[7]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// (r2-s3) WIDENED split-K kernel: m16n16 per warp. Instantiated ONLY with
// SPLITS = 2 and MT_TOTAL >= 2 (the M = 17..172 band). Differences vs the
// narrow sk_kernel<MT_TOTAL, 2> are exactly:
//   * CTA tile: SK_N_TILE_W = 64 B rows (n0 = blockIdx.x * 64; grid =
//     dim3(80, 2) = 160 CTAs >= 148 SMs, so CTA-level parallelism is NOT
//     halved by the doubled tile).
//   * Warp w owns B rows [16w, 16w+16) as TWO 8-row sub-strips; per k-step
//     the warp issues TWO ldmatrix.x2 (one per sub-strip) + MT_TOTAL
//     ldmatrix.x4, then 2*MT_TOTAL mma -- every A fragment feeds two mma,
//     halving A-fragment ldmatrix instructions per mma (12/11 -> 13/22 at
//     MT = 11) and cutting the non-tensor-core smem-load issue in the loop
//     that shares issue slots with the cp.async staging.
//   * Accumulators double: acc[MT_TOTAL][2][4], one m16n8 fp32 set per
//     sub-strip (88 floats/thread at MT_TOTAL = 11).
//   * Epilogue: the same quad-gather per 8-column strip; each warp writes
//     16 columns (cbase = n0 + warp*16 + s*8), two 16B float4 packets per
//     row, all 16B-aligned (ws row 20480 B; cbase is a multiple of 8 fp32).
//   * B tile per pipeline stage doubles (8KB -> 16KB; MT = 8-11 footprint
//     168KB -> 193KB, still < 227KB at 1 CTA/SM).
// The mma.sync.aligned.m16n8k16.f32.f16.f16.f32 instruction, the fp32
// register accumulation and the cp.async pipeline discipline are identical
// to the narrow form, so per-element values are bit-identical; only operand
// reuse and CTA ownership of B rows change (C coverage is unchanged: every
// (m, n) is still produced by exactly one CTA with exactly one accumulator).
// ---------------------------------------------------------------------------
template <int MT_TOTAL>
__global__ __launch_bounds__(SK_THREADS, SK_MIN_BLOCKS_W(MT_TOTAL))
void sk_kernel_w(const __half* __restrict__ A,   // [M, K]
                 const __half* __restrict__ B,   // [N, K]
                 float* __restrict__ ws,         // [2, M, N]
                 __half* __restrict__ C,         // unused (SPLITS == 2 form)
                 int M) {
    constexpr int SPLITS = 2;                   // widened form: 2 splits only
    constexpr int NT     = SK_N_TILE_W;         // 64
    constexpr int D      = SK_DEPTH(MT_TOTAL);  // 3 for MT_TOTAL >= 2
    constexpr int M_PAD  = MT_TOTAL * SK_M_TILE;
    constexpr int NST    = (K_FIXED / SPLITS) / SK_K_TILE;

    extern __shared__ __align__(16) char smem[];
    char* smB = smem;
    char* smA = smem + D * (NT * SK_B_ROW_STRIDE);

    const int tid    = threadIdx.x;
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int n0     = blockIdx.x * NT;
    const int si     = blockIdx.y;                 // k-split id
    const int kbase  = si * (K_FIXED / SPLITS);

    // fp32 accumulators for EVERY m-tile and BOTH 8-column sub-strips of the
    // warp's m16n16 output, live across the whole k walk (88 floats/thread
    // at MT_TOTAL = 11).
    float acc[MT_TOTAL][2][4];
    #pragma unroll
    for (int mt = 0; mt < MT_TOTAL; ++mt) {
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[mt][s][i] = 0.f;
        }
    }

    // Hoisted warp/lane-invariant fragment-address scalars (same r2-s1
    // discipline as the narrow form). The warp's two 8-row sub-strips sit at
    // B rows warp*16 + s*8 + (lane&7): the sub-strip base warp*16 + s*8 is a
    // multiple of 8, so the XOR-swizzle term (brow & 7) reduces to (lane&7)
    // for BOTH sub-strips, and their row byte offsets differ by the constant
    // 8*SK_B_ROW_STRIDE. The A-side scalars are identical to the narrow
    // form (A staging and the A fragment layout are untouched).
    const int arow   = ((lane >> 3) & 1) * 8 + (lane & 7);  // A row in m-tile
    const int bsel   = lane >> 3;                           // B 16B-chunk
                                                            // selector bit
    const int ahalf  = lane >> 4;                           // A 16B-half
                                                            // selector
    const int b_swz  = (lane & 7) * 16;                     // XOR-swizzle
                                                            // pattern, pre-
                                                            // shifted 4 bits
    const int b_row0 = (warp * 16 + (lane & 7)) * SK_B_ROW_STRIDE;
    const int b_row1 = b_row0 + 8 * SK_B_ROW_STRIDE;        // sub-strip 1
    const int a_row  = arow * SK_A_ROW_STRIDE;              // A row byte off

    // Prologue: fill the first D stage buffers (one commit group each).
    #pragma unroll
    for (int s = 0; s < D; ++s) {
        sk_issue_stage<MT_TOTAL, SPLITS, NT>(smB, smA, s, s, A, M, B, n0,
                                             kbase, tid);
    }

    // k-stage loop OUTERMOST: identical pipeline discipline to the narrow
    // form (B[st] fetched once, consumed by ALL m-tiles, buffer recycled
    // only after the end-of-iteration barrier ordered every read of it).
    #pragma unroll 1
    for (int st = 0; st < NST; ++st) {
        cp_wait_n<D - 1>();
        __syncthreads();
        const char* bufB = smB + (st % D) * (NT * SK_B_ROW_STRIDE);
        const char* bufA = smA + (st % D) * (M_PAD * SK_A_ROW_STRIDE);

        // 8 mma k-steps per stage; the TWO B fragments (one per 8-row
        // sub-strip) are loaded once per k-step and reused by every m-tile,
        // so each ldmatrix.x4 A fragment feeds TWO mma -- the r2-s3
        // issue-density win (13 smem-load instructions for 2*MT_TOTAL mma
        // vs the narrow form's 12 for MT_TOTAL).
        #pragma unroll
        for (int i = 0; i < SK_K_TILE / 16; ++i) {
            const int bc8 = (2 * i + bsel) & 15;
            const int bsw = bc8 * 16 ^ b_swz;
            unsigned b00, b01, b10, b11;
            ldmatrix_b(b00, b01, bufB + b_row0 + bsw);  // sub-strip 0
            ldmatrix_b(b10, b11, bufB + b_row1 + bsw);  // sub-strip 1

            const int abyte = 32 * i + 16 * ahalf;
            #pragma unroll
            for (int mt = 0; mt < MT_TOTAL; ++mt) {
                unsigned a0, a1, a2, a3;
                ldmatrix_a(a0, a1, a2, a3,
                           bufA + a_row + abyte
                                + mt * (SK_M_TILE * SK_A_ROW_STRIDE));
                mma_m16n8k16(acc[mt][0][0], acc[mt][0][1],
                             acc[mt][0][2], acc[mt][0][3],
                             a0, a1, a2, a3, b00, b01);
                mma_m16n8k16(acc[mt][1][0], acc[mt][1][1],
                             acc[mt][1][2], acc[mt][1][3],
                             a0, a1, a2, a3, b10, b11);
            }
        }
        __syncthreads();  // all warps done reading stage st's A and B buffers
        sk_issue_stage<MT_TOTAL, SPLITS, NT>(smB, smA, (st + D) % D, st + D,
                                             A, M, B, n0, kbase, tid);
    }

    // Drain the (empty) tail padding groups; nothing else reads smem.
    cp_wait_all();
    __syncthreads();

    // Epilogue (SPLITS == 2 form only): the same quad-gather as the narrow
    // form, applied PER 8-column sub-strip; the warp's 16 columns split at
    // cbase_s = n0 + warp*16 + s*8 (a multiple of 8 fp32 = 16 B). Two 16B
    // float4 packets per row: t = 0/1 cover row r0's cols 0..3 / 4..7 of the
    // strip, t = 2/3 cover row r0 + 8's likewise. Padded rows (global row
    // >= M) are never stored. The shuffles execute unconditionally on all
    // 32 lanes (full mask); only the stores are row-guarded.
    const int g = lane >> 2, t = lane & 3;
    const int q0 = lane & ~3;                       // quad's t == 0 lane id
    const size_t wbase = (size_t)si * M * N_FIXED;
    #pragma unroll
    for (int mt = 0; mt < MT_TOTAL; ++mt) {
        const int r0 = mt * SK_M_TILE + g;
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            const size_t cbase = (size_t)n0 + warp * 16 + s * 8;
            float v0[8], v8[8];
            #pragma unroll
            for (int c = 0; c < 8; ++c) {
                v0[c] = __shfl_sync(0xffffffffu,
                                    (c & 1) ? acc[mt][s][1] : acc[mt][s][0],
                                    q0 + (c >> 1));
                v8[c] = __shfl_sync(0xffffffffu,
                                    (c & 1) ? acc[mt][s][3] : acc[mt][s][2],
                                    q0 + (c >> 1));
            }
            if (t == 0 && r0 < M) {
                *reinterpret_cast<float4*>(
                        ws + wbase + (size_t)r0 * N_FIXED + cbase) =
                    make_float4(v0[0], v0[1], v0[2], v0[3]);
            } else if (t == 1 && r0 < M) {
                *reinterpret_cast<float4*>(
                        ws + wbase + (size_t)r0 * N_FIXED + cbase + 4) =
                    make_float4(v0[4], v0[5], v0[6], v0[7]);
            } else if (t == 2 && r0 + 8 < M) {
                *reinterpret_cast<float4*>(
                        ws + wbase + (size_t)(r0 + 8) * N_FIXED + cbase) =
                    make_float4(v8[0], v8[1], v8[2], v8[3]);
            } else if (t == 3 && r0 + 8 < M) {
                *reinterpret_cast<float4*>(
                        ws + wbase + (size_t)(r0 + 8) * N_FIXED + cbase + 4) =
                    make_float4(v8[4], v8[5], v8[6], v8[7]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// (3) Pass 2: deterministic split-K reduction. Sums the fp32 workspace
// partials in fixed ascending split order (no atomics, run-to-run
// deterministic) and fuses the single fp16 rounding at the store. r2-s2: one
// thread per EIGHT adjacent columns of C (was one per column-pair) -- two
// 16B float4 loads per split, eight fp32 accumulators summed in the same
// fixed ascending split order, one 16B packed 8-half store; the grid shrinks
// from M*2560 to M*640 threads and the LSU instructions per byte drop 2x
// (loads) / 4x (stores) on the workspace round trip. Every address is
// 16B-aligned (ws row 20480 B, C row 10240 B, column offset a multiple of 8
// halves); per-column values and summation order are bit-identical to the
// previous one-pair-per-thread form. Serves both the narrow and the r2-s3
// widened pass-1 kernels unchanged (identical [S, M, N] workspace layout).
// ---------------------------------------------------------------------------
__global__ void sk_reduce_kernel(const float* __restrict__ ws,  // [S, M, N]
                                 __half* __restrict__ C,        // [M, N]
                                 int M, int S) {
    const size_t vecs_per_row = N_FIXED / 8;               // 640
    const size_t total = (size_t)M * vecs_per_row;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const size_t row  = i / vecs_per_row;
    const size_t vp   = i % vecs_per_row;
    const size_t base = row * N_FIXED + 8 * vp;            // 16B-aligned
    float s[8];
    #pragma unroll
    for (int c = 0; c < 8; ++c) s[c] = 0.f;
    for (int sidx = 0; sidx < S; ++sidx) {                 // fixed order
        const float* w = ws + (size_t)sidx * (size_t)M * N_FIXED + base;
        const float4 lo = *reinterpret_cast<const float4*>(w);
        const float4 hi = *reinterpret_cast<const float4*>(w + 4);
        s[0] += lo.x; s[1] += lo.y; s[2] += lo.z; s[3] += lo.w;
        s[4] += hi.x; s[5] += hi.y; s[6] += hi.z; s[7] += hi.w;
    }
    SkHalf8 v;
    v.h0 = __floats2half2_rn(s[0], s[1]);
    v.h1 = __floats2half2_rn(s[2], s[3]);
    v.h2 = __floats2half2_rn(s[4], s[5]);
    v.h3 = __floats2half2_rn(s[6], s[7]);
    *reinterpret_cast<SkHalf8*>(C + base) = v;
}

// ---------------------------------------------------------------------------
// Host launch: compile-time MT_TOTAL dispatch (static acc indexing -> pure
// registers), per-instantiation dynamic-smem opt-in cache, pass-2 launch.
// (r2-s3: a second table/launcher pair serves the widened m16n16-per-warp
// kernel; the narrow sk_kernel grid, smem opt-in and dispatch are unchanged.
// Unchanged by r2-s1: the register-pressure change was carried entirely by
// the kernels' __launch_bounds__ attributes.)
// ---------------------------------------------------------------------------
using SkFn = void (*)(const __half*, const __half*, float*, __half*, int);

static SkFn sk_kernel_for(int mt, int splits) {
    if (splits != 1 && splits != 2) return nullptr;
    switch (mt) {
        case  1: return splits == 1 ? &sk_kernel< 1, 1> : &sk_kernel< 1, 2>;
        case  2: return splits == 1 ? &sk_kernel< 2, 1> : &sk_kernel< 2, 2>;
        case  3: return splits == 1 ? &sk_kernel< 3, 1> : &sk_kernel< 3, 2>;
        case  4: return splits == 1 ? &sk_kernel< 4, 1> : &sk_kernel< 4, 2>;
        case  5: return splits == 1 ? &sk_kernel< 5, 1> : &sk_kernel< 5, 2>;
        case  6: return splits == 1 ? &sk_kernel< 6, 1> : &sk_kernel< 6, 2>;
        case  7: return splits == 1 ? &sk_kernel< 7, 1> : &sk_kernel< 7, 2>;
        case  8: return splits == 1 ? &sk_kernel< 8, 1> : &sk_kernel< 8, 2>;
        case  9: return splits == 1 ? &sk_kernel< 9, 1> : &sk_kernel< 9, 2>;
        case 10: return splits == 1 ? &sk_kernel<10, 1> : &sk_kernel<10, 2>;
        case 11: return splits == 1 ? &sk_kernel<11, 1> : &sk_kernel<11, 2>;
        default: return nullptr;
    }
}

// r2-s3: widened (m16n16-per-warp) kernel table -- SPLITS = 2, MT >= 2 only.
// The MT_TOTAL = 1 instantiation deliberately has NO widened form: the
// M <= 16 stream-floor path stays byte-identical to the validated kernel.
static SkFn sk_w_kernel_for(int mt) {
    switch (mt) {
        case  2: return &sk_kernel_w< 2>;
        case  3: return &sk_kernel_w< 3>;
        case  4: return &sk_kernel_w< 4>;
        case  5: return &sk_kernel_w< 5>;
        case  6: return &sk_kernel_w< 6>;
        case  7: return &sk_kernel_w< 7>;
        case  8: return &sk_kernel_w< 8>;
        case  9: return &sk_kernel_w< 9>;
        case 10: return &sk_kernel_w<10>;
        case 11: return &sk_kernel_w<11>;
        default: return nullptr;
    }
}

// Dynamic smem footprint of the narrow <mt, splits> instantiation (mirrors
// the kernel's SK_DEPTH/D layout: D * (B tile + all-M A slice)).
static int sk_smem_bytes(int mt) {
    const int d = SK_DEPTH(mt);
    return d * ((mt * SK_M_TILE) * SK_A_ROW_STRIDE
                + SK_N_TILE * SK_B_ROW_STRIDE);
}

// Dynamic smem footprint of the widened <mt> instantiation (D * (64-row B
// tile + all-M A slice); MT = 8-11 -> 193 KB, still < 227 KB at 1 CTA/SM).
static int sk_w_smem_bytes(int mt) {
    const int d = SK_DEPTH(mt);
    return d * ((mt * SK_M_TILE) * SK_A_ROW_STRIDE
                + SK_N_TILE_W * SK_B_ROW_STRIDE);
}

static bool s_sk_attr[SK_MAX_MT * SK_MAX_SPLITS];   // zero-init (narrow)
static bool s_skw_attr[SK_MAX_MT + 1];              // zero-init (widened;
                                                    // index = mt, 2..11)

// Returns true only if both kernels were actually configured and issued
// (dynamic smem > 48KB requires an opt-in via cudaFuncSetAttribute).
static bool launch_sk(const __half* A, const __half* B, float* ws, __half* C,
                      int M, int splits, cudaStream_t stream) {
    if (M < 1 || M > SK_MAX_MT * SK_M_TILE) return false;  // resident-A budget
    if (splits != 1 && splits != 2) return false;
    const int mt = (M + SK_M_TILE - 1) / SK_M_TILE;
    SkFn kern = sk_kernel_for(mt, splits);
    if (kern == nullptr) return false;
    const int idx = (mt - 1) * SK_MAX_SPLITS + (splits - 1);
    if (!s_sk_attr[idx]) {
        if (cudaFuncSetAttribute((const void*)kern,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 sk_smem_bytes(mt)) != cudaSuccess) {
            return false;  // caller falls back to the cuBLAS path
        }
        s_sk_attr[idx] = true;
    }
    kern<<<dim3(N_FIXED / SK_N_TILE, splits), SK_THREADS,
          sk_smem_bytes(mt), stream>>>(A, B, ws, C, M);
    if (cudaGetLastError() != cudaSuccess) return false;
    if (splits > 1) {
        // r2-s2: one thread per 8 columns (was per column-pair): M*640
        // threads total, two 16B float4 loads per split, one 16B packed
        // 8-half store.
        const size_t total = (size_t)M * (N_FIXED / 8);
        const unsigned grid = (unsigned)((total + 255) / 256);
        sk_reduce_kernel<<<grid, 256, 0, stream>>>(ws, C, M, splits);
        if (cudaGetLastError() != cudaSuccess) return false;
    }
    return true;
}

// r2-s3: launch the widened m16n16-per-warp SPLITS=2 kernel (grid 80x2 =
// 160 CTAs) plus the unchanged pass-2 reduce. MT_TOTAL >= 2 only; M <= 16
// keeps the byte-identical narrow form (sk_kernel_for covers it).
static bool launch_skw(const __half* A, const __half* B, float* ws, __half* C,
                       int M, cudaStream_t stream) {
    if (M < 1 || M > SK_MAX_MT * SK_M_TILE) return false;
    const int mt = (M + SK_M_TILE - 1) / SK_M_TILE;
    if (mt < 2) return false;  // no widened MT_TOTAL = 1 instantiation
    SkFn kern = sk_w_kernel_for(mt);
    if (kern == nullptr) return false;
    if (!s_skw_attr[mt]) {
        if (cudaFuncSetAttribute((const void*)kern,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 sk_w_smem_bytes(mt)) != cudaSuccess) {
            return false;  // caller falls back (gate keeps previous dispatch)
        }
        s_skw_attr[mt] = true;
    }
    kern<<<dim3(N_FIXED / SK_N_TILE_W, 2), SK_THREADS,
          sk_w_smem_bytes(mt), stream>>>(A, B, ws, C, M);
    if (cudaGetLastError() != cudaSuccess) return false;
    // Same widened pass-2 reduce as the narrow SPLITS=2 form (identical
    // [S, M, N] fp32 workspace layout).
    const size_t total = (size_t)M * (N_FIXED / 8);
    const unsigned grid = (unsigned)((total + 255) / 256);
    sk_reduce_kernel<<<grid, 256, 0, stream>>>(ws, C, M, 2);
    if (cudaGetLastError() != cudaSuccess) return false;
    return true;
}

// ---------------------------------------------------------------------------
// (4) One-time per-M decision: for M <= 258, pick among the cuBLAS path this
// M would otherwise take, the single-split kernel (SPLITS=1), the split-K
// two-pass kernel (SPLITS=2) and -- r2-s3, MT_TOTAL >= 2 only -- the widened
// m16n16-per-warp split-K two-pass kernel. A candidate is eligible only if
// it matches the cuBLAS output at the task tolerance (>= 99.5% matched, NaN
// compares false -> unmatched); the winner must beat cuBLAS by 2% (min of 5
// timed reps). Returns the cached state: 1 = cuBLAS, 2 = SPLITS=1,
// 3 = SPLITS=2 (narrow), 4 = SPLITS=2 (widened). On every return where
// state == 1, C already holds a valid cuBLAS result.
// ---------------------------------------------------------------------------
static int decide_sk_impl(int M, torch::Tensor& C,
                          const __half* Aptr, const __half* Bptr, __half* Cptr,
                          cublasHandle_t handle, cudaStream_t stream) {
    static int s_cc = -1;
    if (s_cc < 0) {
        int dev = 0, major = 0, minor = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        s_cc = major * 10 + minor;
    }
    cublasGemmAlgo_t alt = to_algo(lookup_algo(M));
    if (cublas_tn(handle, Aptr, Bptr, Cptr, M, alt) != CUBLAS_STATUS_SUCCESS) {
        return 1;
    }
    if (s_cc < 80) return 1;
    if (M > SK_MAX_MT * SK_M_TILE) return 1;
    cudaStreamSynchronize(stream);

    // Correctness gate (task tolerance; NaN compares false -> unmatched).
    auto check = [&](const torch::Tensor& out) -> bool {
        torch::Tensor diff = (out - C).abs();
        torch::Tensor tol  = C.abs() * 0.01 + 0.01;
        long matched = (diff <= tol).sum().item<long>();
        return (double)matched >= 0.995 * (double)C.numel();
    };

    float t_blas = 1e30f, t1 = 1e30f, t2 = 1e30f, t4 = 1e30f;

    // SPLITS=1 candidate (fused single-pass; direct fp16 store).
    torch::Tensor out1 = torch::empty_like(C);
    __half* p1 = reinterpret_cast<__half*>(out1.data_ptr<at::Half>());
    if (launch_sk(Aptr, Bptr, nullptr, p1, M, 1, stream)) {
        cudaStreamSynchronize(stream);
        if (check(out1)) {
            t1 = time_reps(5, stream,
                           [&] { launch_sk(Aptr, Bptr, nullptr, p1, M, 1, stream); });
        }
    }

    // SPLITS=2 candidate (narrow two-pass split-K: fp32 workspace + fused
    // cast). The timed callable includes the per-call workspace allocation
    // so the decision reflects the true steady-state cost.
    torch::Tensor out2 = torch::empty_like(C);
    __half* p2 = reinterpret_cast<__half*>(out2.data_ptr<at::Half>());
    {
        torch::Tensor wst = torch::empty(
            {2, M, N_FIXED},
            torch::TensorOptions().dtype(torch::kFloat32).device(C.device()));
        float* wsp = wst.data_ptr<float>();
        if (launch_sk(Aptr, Bptr, wsp, p2, M, 2, stream)) {
            cudaStreamSynchronize(stream);
            if (check(out2)) {
                t2 = time_reps(5, stream, [&] {
                    torch::Tensor w = torch::empty(
                        {2, M, N_FIXED},
                        torch::TensorOptions().dtype(torch::kFloat32).device(C.device()));
                    launch_sk(Aptr, Bptr, w.data_ptr<float>(), p2, M, 2, stream);
                });
            }
        }
    }

    // r2-s3: widened m16n16-per-warp SPLITS=2 candidate (MT_TOTAL >= 2 only;
    // M <= 16 has no widened instantiation and keeps the candidates above).
    {
        const int mt = (M + SK_M_TILE - 1) / SK_M_TILE;
        if (mt >= 2) {
            torch::Tensor out4 = torch::empty_like(C);
            __half* p4 = reinterpret_cast<__half*>(out4.data_ptr<at::Half>());
            torch::Tensor wst = torch::empty(
                {2, M, N_FIXED},
                torch::TensorOptions().dtype(torch::kFloat32).device(C.device()));
            float* wsp = wst.data_ptr<float>();
            if (launch_skw(Aptr, Bptr, wsp, p4, M, stream)) {
                cudaStreamSynchronize(stream);
                if (check(out4)) {
                    t4 = time_reps(5, stream, [&] {
                        torch::Tensor w = torch::empty(
                            {2, M, N_FIXED},
                            torch::TensorOptions().dtype(torch::kFloat32).device(C.device()));
                        launch_skw(Aptr, Bptr, w.data_ptr<float>(), p4, M, stream);
                    });
                }
            }
        }
    }

    // cuBLAS timing last (its reps rewrite C with the same deterministic
    // result, so the correctness references above stay valid either way).
    t_blas = time_reps(5, stream,
                       [&] { cublas_tn(handle, Aptr, Bptr, Cptr, M, alt); });

    // Winner among the eligible candidates, subject to the 2% margin over
    // cuBLAS; any M where the widened form loses keeps the previous dispatch
    // (narrow or cuBLAS), so worst case is parity.
    int best_state = 1;
    float best = 1e30f;
    if (t1 < best) { best = t1; best_state = 2; }
    if (t2 < best) { best = t2; best_state = 3; }
    if (t4 < best) { best = t4; best_state = 4; }
    if (best < 0.98f * t_blas) return best_state;
    return 1;
}

// ---------------------------------------------------------------------------
// Per-M gate cache: 0 = unchecked, 1 = cuBLAS, 2 = SPLITS=1 (narrow),
// 3 = SPLITS=2 (narrow), 4 = SPLITS=2 (r2-s3 widened). Only Ms <= SK_BAND_M
// (258) ever get an entry (at most 32 distinct per process), so the cache can
// never fill; every M > 258 skips the gate entirely and keeps the
// byte-identical cuBLAS dispatch.
// ---------------------------------------------------------------------------
static constexpr int SK_GATE_SLOTS = 32;
static int s_gate_m[SK_GATE_SLOTS];
static int s_gate_state[SK_GATE_SLOTS];   // zero-init = unchecked
static int s_gate_n = 0;

static int gate_find(int M) {
    for (int i = 0; i < s_gate_n; ++i) {
        if (s_gate_m[i] == M) return i;
    }
    return -1;
}

// ---------------------------------------------------------------------------
// forward(A, B) -> C   where C = A @ B.T, all fp16.
// ---------------------------------------------------------------------------
torch::Tensor forward(const torch::Tensor& A, const torch::Tensor& B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "2D tensors required");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(A.scalar_type() == torch::kFloat16, "fp16 A required");
    TORCH_CHECK(B.scalar_type() == torch::kFloat16, "fp16 B required");
    TORCH_CHECK(A.size(1) == K_FIXED, "A K dim must be ", K_FIXED);
    TORCH_CHECK(B.size(0) == N_FIXED, "B N dim must be ", N_FIXED);
    TORCH_CHECK(B.size(1) == K_FIXED, "B K dim must be ", K_FIXED);

    const int M = (int)A.size(0);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const __half* Aptr = reinterpret_cast<const __half*>(A.data_ptr<at::Half>());
    const __half* Bptr = reinterpret_cast<const __half*>(B.data_ptr<at::Half>());

    auto C = torch::empty({M, N_FIXED},
                          torch::TensorOptions().dtype(torch::kFloat16).device(A.device()));
    __half* Cptr = reinterpret_cast<__half*>(C.data_ptr<at::Half>());

    cublasHandle_t handle = get_handle(stream);

    // Sub-crossover band (M <= 258): the split-K skinny kernels compete with
    // the cuBLAS path this M would otherwise take; the decision is measured
    // once per M on the warmup call and cached. M > 258 never enters this
    // block and keeps the untouched dispatch below.
    if (M <= SK_BAND_M) {
        int gi = gate_find(M);
        if (gi < 0) {
            std::lock_guard<std::mutex> lock(s_mu);
            gi = gate_find(M);               // re-check under the lock
            if (gi < 0 && s_gate_n < SK_GATE_SLOTS) {
                gi = s_gate_n++;
                s_gate_m[gi] = M;
                s_gate_state[gi] = 0;        // unchecked
            }
        }
        if (gi >= 0 && s_gate_state[gi] == 0) {
            std::lock_guard<std::mutex> lock(s_mu);
            if (s_gate_state[gi] == 0) {     // double-checked
                s_gate_state[gi] = decide_sk_impl(M, C, Aptr, Bptr, Cptr,
                                                  handle, stream);
            }
        }
        if (gi >= 0) {
            const int st = s_gate_state[gi];
            if (st == 2) {
                if (launch_sk(Aptr, Bptr, nullptr, Cptr, M, 1, stream)) return C;
                // Unreachable in practice (the gate verified this launch);
                // fall through and recompute via cuBLAS so C is always correct.
            } else if (st == 3) {
                torch::Tensor ws = torch::empty(
                    {2, M, N_FIXED},
                    torch::TensorOptions().dtype(torch::kFloat32).device(A.device()));
                if (launch_sk(Aptr, Bptr, ws.data_ptr<float>(), Cptr, M, 2, stream))
                    return C;
            } else if (st == 4) {
                // r2-s3: widened m16n16-per-warp SPLITS=2 dispatch.
                torch::Tensor ws = torch::empty(
                    {2, M, N_FIXED},
                    torch::TensorOptions().dtype(torch::kFloat32).device(A.device()));
                if (launch_skw(Aptr, Bptr, ws.data_ptr<float>(), Cptr, M, stream))
                    return C;
            }
            // st == 1: the gate already left the cuBLAS result in C during
            // the warmup decide call; the recompute below is idempotent.
        }
    }

    // Untouched cuBLAS dispatch (M > 258 and every fallback): per-M algo for
    // known M, DEFAULT (== torch.matmul parity) for unseen M.
    // Portable variant: the per-M TENSOR_OP algos are checked, with a
    // CUBLAS_GEMM_DEFAULT retry - the legacy CUBLAS_GEMM_ALGO*_TENSOR_OP
    // enums are rejected by some cuBLAS builds (CUDA 13 defines only
    // ALGO0..15_TENSOR_OP; the official GPU server rejects more), and an
    // unchecked rejection leaves C unwritten. All valid algos measure the
    // same latency on B300 (probe 2026-09-15), so the fallback costs nothing.
    if (cublas_tn(handle, Aptr, Bptr, Cptr, M, to_algo(lookup_algo(M)))
        != CUBLAS_STATUS_SUCCESS) {
        cublasStatus_t st = cublas_tn(handle, Aptr, Bptr, Cptr, M,
                                      CUBLAS_GEMM_DEFAULT);
        TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS,
                    "cublasGemmEx failed even with CUBLAS_GEMM_DEFAULT: ",
                    (int)st);
    }
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward,
          "fp16 GEMM C=A@B.T (N=5120,K=2048): B-streaming split-K skinny "
          "mma.sync kernels for the M<=258 sub-crossover band (grid = N-tiles "
          "x K-splits, B streamed from HBM exactly once, A k-stage slices "
          "smem-resident, fp32 accumulators, deterministic two-pass fp32-"
          "workspace reduction with fused fp16 cast, no atomics); r2-s3 adds "
          "a widened m16n16-per-warp SPLITS=2 form (SK_N_TILE 32->64, two "
          "8-row B sub-strips per warp, every ldmatrix.x4 A fragment feeding "
          "two mma.sync.m16n8k16 -> halved A-fragment smem-load issue per "
          "mma, MT_TOTAL>=2 instantiations only so the M<=16 stream-floor "
          "path stays byte-identical), still register-pressure-tuned via "
          "per-instantiation __launch_bounds__ minBlocksPerMultiprocessor "
          "matching the smem co-residency budget plus hoisted lane-invariant "
          "fragment addressing, still r2-s2-widened 16B global transactions "
          "in the epilogue and pass-2 reduce, gated per-M against cuBLAS "
          "with no-regression fallback; M>258 keeps the byte-identical "
          "cuBLAS-TN per-shape algo dispatch.");
    m.def("run", &forward,
          "Alias of forward (task entry point run(A, B) -> C).");
}