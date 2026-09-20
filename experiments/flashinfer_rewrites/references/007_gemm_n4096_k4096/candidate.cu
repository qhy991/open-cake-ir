/*
 * GEMM C[M,N] = A[M,K] @ B[N,K]^T  (fp16, N=4096 K=4096, M variable 1..8192)
 * Task 007_gemm_n4096_k4096 on B300 SXM6 (sm_103, 148 SMs).
 *
 * Reference: torch.matmul(A, B.T)  -> cuBLAS NT layout.
 *
 * Strategy: occupancy_tuning.
 *   The 40 workloads with M < 2048 are all below the M~258 memory/compute
 *   crossover: their cost is "stream B once" (33.55 MB / 6.434 TB/s =
 *   0.005215 ms floor), and they measured ~0.0199 ms, i.e. 3.8x above it.
 *   The cause is SM fill, not bandwidth: cuBLAS's default 128x128 tile on a
 *   4096-wide output launches ceil(4096/128)*ceil(M/128) = 32 CTAs for every
 *   M <= 128 -- 32/148 = 21.6% of the SMs -- and the measured B-stream
 *   throughput at those M is 23.5%-27.1% of peak.  Achieved bandwidth tracks
 *   predicted SM fill to within ~2 points, so the knob is the GEMM's CTA grid.
 *
 *   We therefore drive the small-M GEMM through cuBLASLt and pick the
 *   algorithm by CTA count instead of taking the library's default: among the
 *   heuristic candidates we take the lowest-wavesCount one that reaches at
 *   least 148 CTAs (one per SM), falling back to whichever candidate fills the
 *   most SMs.  Split-K is allowed as a second occupancy source but capped, so
 *   the fp32 partials it writes and re-reads cannot cost more HBM traffic than
 *   the extra parallelism buys back.
 *
 *   Retired in this revision (measured losers on B300, both sweeps):
 *   - The cached-B_T + NN-layout path.  Its premise ("cuBLAS picks better
 *     algorithms for NN than for NT at small M") was an RTX 4090 finding and
 *     is inverted here: over the 21 workloads it covered it geomeans 0.935x,
 *     while the plain at::mm(A, B.t()) pass-through over its 19 workloads
 *     geomeans 1.049x.  Every M in [104,256] was below 1.0x in both runs.
 *   - transpose_half_kernel and its vectorised 64x64 replacement.  The
 *     same_pool cache already amortised the transpose to ~zero, so retuning
 *     its launch geometry moved nothing (M=128 0.950 -> 0.904).
 *   - The s_prev_B_ptr pool-shift detector: cross-call state that inflates
 *     measured speedup on repeated inputs (TASK.json known_issues).
 *
 * Untouched (79-100% of the fp16 roofline, no headroom):
 *   - M in [2048,4096): direct cublasGemmEx NT.
 *   - M >= 4096:        delegate to the reference NT path.
 *
 * Hard-won constraints (DO NOT regress):
 *   - CUBLAS_COMPUTE_16F returns INVALID_VALUE -> use COMPUTE_32F everywhere.
 *   - No per-M algorithm search inside the timed loop.  The cuBLASLt search
 *     runs once per distinct M, on the first (warmup) call, and the winner is
 *     cached in s_algo_cache; timed iterations only do the matmul.
 *   - Nothing here reads at::globalContext().allowFP16ReductionCuBLAS(), whose
 *     return type changed to at::CuBLASReductionOption in torch 2.11.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/TensorOptions.h>
#include <cstdint>

static constexpr int N_FIXED = 4096;
static constexpr int K_FIXED = 4096;

// Direct-NT window: for M in [DIRECT_NT_LO, DIRECT_NT_HI) the manual
// cublasGemmEx NT path is already at the roofline; left exactly as measured.
static constexpr int DIRECT_NT_LO = 2048;
static constexpr int DIRECT_NT_HI = 4096;
// Very-large-M threshold: compute-bound (M=8192 measures 100.3% of the
// 1658.6 TFLOP/s peak), so delegate to the reference NT path.
static constexpr int LARGE_M_REFERENCE_THRESHOLD = 4096;

// ---------------------------------------------------------------------------
// Occupancy-tuning parameters.
// ---------------------------------------------------------------------------
// B300 SXM6 SM count: the CTA target that defines "one wave covers the GPU".
static constexpr long long SM_COUNT = 148;
// Heuristic candidates to rank.  16 is enough to reach past the default
// 128x128 tile into the small-tile / split-K part of the algorithm space.
static constexpr int HEURISTIC_COUNT = 16;
// Split-K partials are fp32 and cost 2 * M * N * 4 * splitk bytes (the write
// plus the reduction's read).  Keeping that under 25% of the 33.554 MB
// B-stream gives cap = max(1, 256/M): 32 at M=8, 4 at M=64, 2 at M=128,
// 1 at M>=256.  Above the cap split-K re-imports more traffic than the extra
// occupancy saves.
static constexpr int SPLITK_TRAFFIC_BUDGET = 256;
// One workspace, allocated once, sized for the split-K partials plus slack.
static constexpr size_t LT_WORKSPACE_BYTES = 32ull << 20;
// Per-M algorithm cache: 43 distinct M in the sweep, 40 of them below 2048.
static constexpr int ALGO_CACHE_CAP = 64;

// Persistent cuBLAS handle for the large-M direct path.
static cublasHandle_t s_handle = nullptr;
static bool s_handle_init = false;

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

// ---------------------------------------------------------------------------
// cuBLASLt state.
//
// The operand mapping is the one already proven correct by the cublasGemmEx
// block below: cuBLAS's "A" argument is our B matrix with OP_T, cuBLAS's "B"
// argument is our A matrix with OP_N, so the matmul dims are
//   m = N_FIXED (4096), n = M, k = K_FIXED (4096),
// and a column-major D of (N_FIXED x M) with ld = N_FIXED is bit-for-bit the
// row-major C[M][N] we must return.  Only the two M-dependent layouts change
// between calls, and they change by one COLS attribute, so all four objects
// are created once and kept in statics.
// ---------------------------------------------------------------------------
static cublasLtHandle_t s_lt = nullptr;
static cublasLtMatmulDesc_t s_desc = nullptr;
static cublasLtMatrixLayout_t s_lay_B = nullptr;  // cuBLAS A-arg: [K x N], ld=K
static cublasLtMatrixLayout_t s_lay_A = nullptr;  // cuBLAS B-arg: [K x M], ld=K
static cublasLtMatrixLayout_t s_lay_C = nullptr;  // C and D:      [N x M], ld=N
static at::Tensor s_workspace;
static bool s_lt_ok = false;
static bool s_lt_tried = false;
static int s_layout_M = -1;   // M currently programmed into s_lay_A / s_lay_C

struct AlgoEntry {
    int M;
    bool has_algo;            // false => this M is on the at::mm fallback
    cublasLtMatmulAlgo_t algo;
};
static AlgoEntry s_algo_cache[ALGO_CACHE_CAP];
static int s_algo_count = 0;

static bool lt_init(const torch::Tensor& ref) {
    if (s_lt_tried) return s_lt_ok;
    s_lt_tried = true;

    if (cublasLtCreate(&s_lt) != CUBLAS_STATUS_SUCCESS) return false;

    // COMPUTE_32F with a 32F scale type: same accumulation as the reference,
    // and strictly more accurate than the fp16 reduction the M<208 workloads
    // used to run with.
    if (cublasLtMatmulDescCreate(&s_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F)
        != CUBLAS_STATUS_SUCCESS) return false;

    const cublasOperation_t op_t = CUBLAS_OP_T;   // on B (stored [N,K])
    const cublasOperation_t op_n = CUBLAS_OP_N;   // on A (stored [M,K])
    if (cublasLtMatmulDescSetAttribute(s_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                       &op_t, sizeof(op_t)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(s_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                       &op_n, sizeof(op_n)) != CUBLAS_STATUS_SUCCESS) {
        return false;
    }

    // op(A-arg) is m x k = N x K, so the stored matrix is k x m = K x N.
    if (cublasLtMatrixLayoutCreate(&s_lay_B, CUDA_R_16F, K_FIXED, N_FIXED, K_FIXED)
        != CUBLAS_STATUS_SUCCESS) return false;
    // op(B-arg) is k x n = K x M; cols are re-set per call.
    if (cublasLtMatrixLayoutCreate(&s_lay_A, CUDA_R_16F, K_FIXED, 1, K_FIXED)
        != CUBLAS_STATUS_SUCCESS) return false;
    // D is m x n = N x M with ld = N; cols are re-set per call.
    if (cublasLtMatrixLayoutCreate(&s_lay_C, CUDA_R_16F, N_FIXED, 1, N_FIXED)
        != CUBLAS_STATUS_SUCCESS) return false;

    s_workspace = torch::empty({(long)LT_WORKSPACE_BYTES},
                               torch::TensorOptions()
                                   .dtype(torch::kUInt8)
                                   .device(ref.device()));

    s_lt_ok = true;
    return true;
}

// Program the two M-dependent layouts.  Cheap, but skipped when M repeats
// (which it does for every iteration of a workload).
static bool lt_set_M(int M) {
    if (s_layout_M == M) return true;
    const uint64_t cols = (uint64_t)M;
    if (cublasLtMatrixLayoutSetAttribute(s_lay_A, CUBLASLT_MATRIX_LAYOUT_COLS,
                                         &cols, sizeof(cols)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutSetAttribute(s_lay_C, CUBLASLT_MATRIX_LAYOUT_COLS,
                                         &cols, sizeof(cols)) != CUBLAS_STATUS_SUCCESS) {
        s_layout_M = -1;
        return false;
    }
    s_layout_M = M;
    return true;
}

// Decode CUBLASLT_ALGO_CONFIG_TILE_ID into its (tile_m, tile_n) extents.
// Only the enumerators that have been stable since CUDA 11 are listed; newer
// ids (the 8xN / 16xN families) are reported as unknown on purpose rather than
// guessed at, and the caller then falls back to the library's own wavesCount.
static bool tile_extent(uint32_t tile_id, int& tm, int& tn) {
    switch (tile_id) {
        case  1: tm =   8; tn =   8; return true;
        case  2: tm =   8; tn =  16; return true;
        case  3: tm =  16; tn =   8; return true;
        case  4: tm =   8; tn =  32; return true;
        case  5: tm =  16; tn =  16; return true;
        case  6: tm =  32; tn =   8; return true;
        case  7: tm =   8; tn =  64; return true;
        case  8: tm =  16; tn =  32; return true;
        case  9: tm =  32; tn =  16; return true;
        case 10: tm =  64; tn =   8; return true;
        case 11: tm =  32; tn =  32; return true;
        case 12: tm =  32; tn =  64; return true;
        case 13: tm =  64; tn =  32; return true;
        case 14: tm =  32; tn = 128; return true;
        case 15: tm =  64; tn =  64; return true;
        case 16: tm = 128; tn =  32; return true;
        case 17: tm =  64; tn = 128; return true;
        case 18: tm = 128; tn =  64; return true;
        case 19: tm =  64; tn = 256; return true;
        case 20: tm = 128; tn = 128; return true;
        case 21: tm = 256; tn =  64; return true;
        case 22: tm =  64; tn = 512; return true;
        case 23: tm = 128; tn = 256; return true;
        case 24: tm = 256; tn = 128; return true;
        case 25: tm = 512; tn =  64; return true;
        case 26: tm =  64; tn =  96; return true;
        case 27: tm =  96; tn =  64; return true;
        case 28: tm =  96; tn = 128; return true;
        case 29: tm = 128; tn = 160; return true;
        case 30: tm = 160; tn = 128; return true;
        case 31: tm = 192; tn = 128; return true;
        case 32: tm = 128; tn = 192; return true;
        case 33: tm = 128; tn =  96; return true;
        case 34: tm =  32; tn = 256; return true;
        case 35: tm = 256; tn =  32; return true;
        default: return false;
    }
}

// CTAs launched by a candidate.  The matmul's m dimension is N_FIXED and its
// n dimension is M (see the descriptor above), so the grid is
// ceil(N_FIXED/tile_m) * ceil(M/tile_n), times the split-K factor.  This is
// the total CTA count the plan ranks on; the axis labels follow this
// descriptor's m/n, which is what decides the grid.
static long long cta_count(uint32_t tile_id, uint32_t splitk, int M, float waves) {
    int tm = 0, tn = 0;
    if (tile_extent(tile_id, tm, tn) && tm > 0 && tn > 0) {
        const long long grid_m = (N_FIXED + tm - 1) / tm;
        const long long grid_n = ((long long)M + tn - 1) / tn;
        return grid_m * grid_n * (long long)splitk;
    }
    // Unknown tile id: wavesCount is cuBLASLt's own fill metric (1.0 == the
    // launch covers the GPU once), so waves * SM_COUNT estimates the same
    // quantity without guessing at the enum.
    const long long est = (long long)((double)waves * (double)SM_COUNT + 0.5);
    return est > 0 ? est : 1;
}

static uint32_t splitk_cap_for(int M) {
    const int cap = SPLITK_TRAFFIC_BUDGET / (M > 0 ? M : 1);
    return (uint32_t)(cap > 1 ? cap : 1);
}

static uint32_t algo_splitk(const cublasLtMatmulAlgo_t& algo) {
    uint32_t splitk = 1;
    size_t written = 0;
    if (cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                             &splitk, sizeof(splitk), &written)
            != CUBLAS_STATUS_SUCCESS || written != sizeof(splitk) || splitk < 1) {
        return 1;
    }
    return splitk;
}

static uint32_t algo_tile_id(const cublasLtMatmulAlgo_t& algo) {
    uint32_t tile = 0;
    size_t written = 0;
    if (cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID,
                                             &tile, sizeof(tile), &written)
            != CUBLAS_STATUS_SUCCESS || written != sizeof(tile)) {
        return 0;
    }
    return tile;
}

// The occupancy knob.  Ask for 16 candidates, drop the ones whose split-K
// exceeds the traffic cap, then take the lowest-wavesCount candidate that
// reaches one CTA per SM; if nothing does, take the one that fills the most
// SMs.  Runs once per distinct M, outside the timed iterations.
static bool lt_select_algo(int M, cublasLtMatmulAlgo_t* out) {
    cublasLtMatmulPreference_t pref = nullptr;
    if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) return false;
    size_t ws = LT_WORKSPACE_BYTES;
    if (cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                             &ws, sizeof(ws)) != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatmulPreferenceDestroy(pref);
        return false;
    }

    cublasLtMatmulHeuristicResult_t res[HEURISTIC_COUNT];
    int returned = 0;
    const cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
        s_lt, s_desc, s_lay_B, s_lay_A, s_lay_C, s_lay_C,
        pref, HEURISTIC_COUNT, res, &returned);
    cublasLtMatmulPreferenceDestroy(pref);
    if (st != CUBLAS_STATUS_SUCCESS || returned <= 0) return false;

    const uint32_t cap = splitk_cap_for(M);

    int best = -1;              // >= SM_COUNT CTAs, minimal wavesCount
    float best_waves = 0.0f;
    int fullest = -1;           // most CTAs, used when nothing reaches SM_COUNT
    long long fullest_ctas = -1;

    for (int i = 0; i < returned; ++i) {
        if (res[i].state != CUBLAS_STATUS_SUCCESS) continue;
        if (res[i].workspaceSize > LT_WORKSPACE_BYTES) continue;

        const uint32_t splitk = algo_splitk(res[i].algo);
        if (splitk > cap) continue;   // split-K traffic cap

        const long long ctas =
            cta_count(algo_tile_id(res[i].algo), splitk, M, res[i].wavesCount);

        if (ctas >= SM_COUNT && (best < 0 || res[i].wavesCount < best_waves)) {
            best = i;
            best_waves = res[i].wavesCount;
        }
        if (ctas > fullest_ctas) {
            fullest_ctas = ctas;
            fullest = i;
        }
    }

    const int pick = (best >= 0) ? best : fullest;
    if (pick < 0) return false;

    cublasLtMatmulAlgo_t algo = res[pick].algo;

    // Split-K reduces through fp32 partials rather than fp16 ones.  Validate
    // the edited config before trusting it; if the library rejects it, keep
    // the candidate exactly as the heuristic returned it.
    if (algo_splitk(algo) > 1) {
        cublasLtMatmulAlgo_t tuned = algo;
        uint32_t scheme = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
        if (cublasLtMatmulAlgoConfigSetAttribute(&tuned, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                                                 &scheme, sizeof(scheme)) == CUBLAS_STATUS_SUCCESS) {
            cublasLtMatmulHeuristicResult_t chk;
            if (cublasLtMatmulAlgoCheck(s_lt, s_desc, s_lay_B, s_lay_A, s_lay_C, s_lay_C,
                                        &tuned, &chk) == CUBLAS_STATUS_SUCCESS &&
                chk.state == CUBLAS_STATUS_SUCCESS &&
                chk.workspaceSize <= LT_WORKSPACE_BYTES) {
                algo = tuned;
            }
        }
    }

    *out = algo;
    return true;
}

// Per-M algorithm table.  Returns the cache slot for this M, running the
// search on first sight; slot->has_algo == false means "this M goes to the
// at::mm fallback", which is remembered so no timed iteration ever repeats a
// failed search.
static AlgoEntry* lt_algo_for(int M) {
    for (int i = 0; i < s_algo_count; ++i) {
        if (s_algo_cache[i].M == M) return &s_algo_cache[i];
    }

    cublasLtMatmulAlgo_t algo;
    const bool found = lt_select_algo(M, &algo);

    if (s_algo_count >= ALGO_CACHE_CAP) {
        // Table full (never happens for this 43-workload sweep).  Do not cache;
        // returning null sends this call to the fallback rather than paying for
        // a search inside the timed loop.
        return nullptr;
    }
    AlgoEntry* e = &s_algo_cache[s_algo_count++];
    e->M = M;
    e->has_algo = found;
    if (found) e->algo = algo;
    return e;
}

torch::Tensor run(const torch::Tensor& A, const torch::Tensor& B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "2D tensors required");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(A.scalar_type() == torch::kFloat16, "fp16 A required");
    TORCH_CHECK(B.scalar_type() == torch::kFloat16, "fp16 B required");

    const int M = (int)A.size(0);
    TORCH_CHECK(A.size(1) == K_FIXED, "A must be [M, 4096]");
    TORCH_CHECK(B.size(0) == N_FIXED && B.size(1) == K_FIXED, "B must be [4096, 4096]");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const __half* Bptr = reinterpret_cast<const __half*>(B.data_ptr<at::Half>());

    if (M >= DIRECT_NT_LO && M < DIRECT_NT_HI) {
        // Mid-large M window (2048..4095): direct cublasGemmEx fp32 compute,
        // NT layout.  Measured at 79-100% of the fp16 roofline; untouched.
        const __half* Aptr = reinterpret_cast<const __half*>(A.data_ptr<at::Half>());
        auto C = torch::empty({M, N_FIXED},
                              torch::TensorOptions()
                                  .dtype(torch::kFloat16)
                                  .device(A.device()));
        __half* Cptr = reinterpret_cast<__half*>(C.data_ptr<at::Half>());

        cublasHandle_t handle = get_handle(stream);
        const float alpha = 1.0f, beta = 0.0f;
        cublasStatus_t status = cublasGemmEx(
            handle,
            CUBLAS_OP_T,      // opA on B (B stored [N,K], opT -> reads K-major)
            CUBLAS_OP_N,      // opB on A (A stored [M,K])
            N_FIXED,          // m  (rows of op(A)=B -> N)
            M,                // n  (cols of op(B)=A -> M)
            K_FIXED,          // k
            &alpha,
            Bptr,             // A-arg of cublas = B data
            CUDA_R_16F,
            K_FIXED,          // lda = K
            Aptr,             // B-arg of cublas = A data
            CUDA_R_16F,
            K_FIXED,          // ldb = K
            &beta,
            Cptr,
            CUDA_R_16F,
            N_FIXED,          // ldc = N
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
        TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cublasGemmEx failed: ", (int)status);
        return C;
    }

    if (M >= LARGE_M_REFERENCE_THRESHOLD) {
        // Very large M (>=4096, i.e. M=8192 here): compute-bound and already at
        // 100.3% of the measured fp16 peak.  Delegate to the reference NT path.
        return at::mm(A, B.t());
    }

    // -----------------------------------------------------------------------
    // Below the crossover (M < 2048, 40 of the 43 workloads): occupancy-ranked
    // cuBLASLt NT matmul.  Every exit from this block that is not a successful
    // cublasLtMatmul falls through to at::mm(A, B.t()) -- the identical-work
    // pass-through that measured 1.049x -- so the floor is the best currently
    // measured behaviour rather than a regression.
    // -----------------------------------------------------------------------
    const bool operands_ok =
        A.is_contiguous() && B.is_contiguous() &&
        (reinterpret_cast<uintptr_t>(A.data_ptr<at::Half>()) & 15) == 0 &&
        (reinterpret_cast<uintptr_t>(Bptr) & 15) == 0;

    if (operands_ok && lt_init(A) && lt_set_M(M)) {
        AlgoEntry* entry = lt_algo_for(M);
        if (entry != nullptr && entry->has_algo) {
            auto C = torch::empty({M, N_FIXED},
                                  torch::TensorOptions()
                                      .dtype(torch::kFloat16)
                                      .device(A.device()));
            // Column-major D of (N x M) with ld = N is exactly this row-major
            // C[M][N]; beta = 0 so C and D may alias.
            void* Cptr = C.data_ptr<at::Half>();
            const float alpha = 1.0f, beta = 0.0f;

            const cublasStatus_t st = cublasLtMatmul(
                s_lt, s_desc,
                &alpha,
                Bptr,   s_lay_B,                   // cuBLAS A-arg = B data, OP_T
                A.data_ptr<at::Half>(), s_lay_A,   // cuBLAS B-arg = A data, OP_N
                &beta,
                Cptr,   s_lay_C,
                Cptr,   s_lay_C,
                &entry->algo,
                s_workspace.data_ptr(), LT_WORKSPACE_BYTES,
                stream);

            if (st == CUBLAS_STATUS_SUCCESS) return C;

            // Nothing was launched on a non-success status, so C is simply
            // discarded.  Retire this M to the fallback permanently.
            entry->has_algo = false;
        }
    }

    // Fallback: metadata-only transpose -> cuBLAS NT, i.e. the reference path.
    return at::mm(A, B.t());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "fp16 GEMM 007: occupancy-ranked cuBLASLt NT below the crossover, cublasGemmEx NT / reference above");
}
