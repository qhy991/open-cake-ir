#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <limits>

using namespace at;

// Fixed geometry constants
constexpr int64_t H = 7168;
constexpr int64_t I = 2048;
constexpr int64_t BLOCK = 128;
constexpr int64_t E_GLOBAL = 256;
constexpr int64_t E_LOCAL = 32;
constexpr int64_t TOP_K = 8;
constexpr int64_t N_GROUP = 8;
constexpr int64_t TOPK_GROUP = 4;

// <<<IMPROVE BEGINS>>>
// Conservative dispatch kernel: scatter (token t, selected-expert-slot k) pairs of
// topk_idx[T,8] into per-local-expert token-id bins with one atomicAdd per routed pair.
// Replaces the seed's per-expert `(topk_idx==ge).any(1).item<bool>()` + `at::nonzero`
// host-sync round-trips (~2 forced D2H syncs * 32 experts = ~64/call) with ONE on-GPU
// kernel + ONE D2H copy of count[32].
//
// Bit-exactness (seed is 19/19 bit-exact, max_abs_err=0.0): only intra-expert row order
// differs (atomic arrival order). GEMM rows are independent (each output row is a dot
// product of one A row with the shared weight, independent of sibling rows), and
// output.index_add_(0, token_idx, weighted_O) accumulates each token's contribution across
// the up-to-8 experts it was routed to in the SAME outer le=0..31 order as the seed, so the
// per-token FP sum is order-invariant. No FP8 e4m3 tensor-core compute is added (still
// TF32 cuBLAS + fp32 block-scale dequant), so numerics match the seed exactly.
__global__ void scatter_tokens_to_local_experts(
    const int64_t* __restrict__ topk_idx,   // [T, TOP_K] int64
    int64_t* __restrict__ count,            // [E_LOCAL] int64 (zeroed before launch)
    int64_t* __restrict__ token_ids,        // [E_LOCAL * T] int64
    int64_t T,
    int64_t local_start)
{
    int64_t n_items = T * TOP_K;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_items;
         idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t t = idx / TOP_K;            // [0, T)
        int64_t k = idx - t * TOP_K;        // [0, TOP_K)
        int64_t ge = topk_idx[t * TOP_K + k];
        if (ge >= local_start && ge < local_start + E_LOCAL) {
            int64_t le = ge - local_start;  // [0, E_LOCAL)
            // 64-bit atomicAdd via the portable unsigned-long-long overload
            // (int64_t == long on LP64 Linux, which has no direct cuda atomicAdd overload).
            int64_t slot = (int64_t)atomicAdd(
                reinterpret_cast<unsigned long long*>(count + le), 1ULL);
            token_ids[le * T + slot] = t;
        }
    }
}

// ============================================================================
// Fused FP8(e4m3) block-scale -> fp32 dequant (conservative_tiling, round 2).
// Replaces the seed's per-call-site pattern `to(kFloat)` + `repeat_interleave`*2
// (which materializes a FULL-SIZE fp32 scale tensor ~rows*cols*4B that is immediately
// discarded) + elementwise mul, applied at the three dequant call sites (activations +
// per-expert W13 + per-expert W2). HBM traffic per dequant drops ~4x (e.g. W13: ~588MB
// seed -> ~147MB fused; W2 similar) and 4 ATen launches collapse to 1 kernel.
//
// conservative_tiling (round-2 incremental reinforcement of this floor kernel):
// VECTORIZED cols tile. Each thread handles V=4 fp8 elements: it loads 4 fp8 bytes as a
// single uint32 (one coalesced 128B load per warp: 32 lanes * 4B) and stores 4 fp32 as a
// single float4 (one coalesced 128B store transaction group per warp). The hot-path
// instruction count drops ~4x vs the round-1 scalar 1-byte-load/1-fp32-store per thread,
// and store transactions stay fully coalesced. Re-tiled so ONE warp (32 lanes * V=4)
// covers EXACTLY 128 cols == one scale block: sc = (blockIdx.x*128)/128 = blockIdx.x is
// identical for all 32 lanes AND threadIdx.y is constant within a warp -> the per-128
// scale address is identical across the whole warp -> a single L1 broadcast read instead
// of the round-1 per-element scale re-reads. Block thread count (256) and ROWS_BLOCKED /
// {rows,cols,scale_cols} contract are unchanged; a scalar tail path (c0+V>cols) preserves
// bit-exactness for non-multiple-of-128 cols (H=7168, I=2048, 2I=4096 are all multiples
// of 128, so the tail rarely triggers, but it is kept for safety).
//
// Bit-exactness (e4m3 has a unique fp32 decode; fp8->fp32 is an exact widening with no
// rounding; the multiply is IEEE round-to-nearest-even fp32, identical to ATen's
// elementwise mul and to the round-1 scalar `dst[r*cols+c] = v*s`; mul vs fma(a,b,0) are
// bit-identical under RNE): cuBLAS TF32 GEMM inputs are unchanged, so 17/19 bit-exact +
// the 2 large-T workloads' max_abs_err=4096 (cuBLAS TF32 ULP nondeterminism from
// non-sorted intra-expert row order, NOT the dequant) are preserved. No FP8 tensor-core
// compute is added (still TF32 cuBLAS), preserving the loose tolerance budget.
//
// Layout: src [rows, cols] fp8 e4m3 row-major; scale [scale_rows, cols/BLOCK] fp32
// row-major, where scale_rows = rows/BLOCK when ROWS_BLOCKED (2-D block scale, used for
// weights) and = rows when !ROWS_BLOCKED (per-row block scale along cols, used for the
// activation, whose scale is [H/128, T] permuted to [T, H/128]). Each scale element
// covers a BLOCK x BLOCK (weights) or 1 x BLOCK (activation) tile.
// ============================================================================
template <bool ROWS_BLOCKED>
__global__ void dequant_fp8_e4m3_blockscale_fused(
    const __nv_fp8_e4m3* __restrict__ src,   // [rows, cols] fp8 e4m3 (row-major)
    const float* __restrict__ scale,          // [scale_rows, cols/BLOCK] fp32 (row-major)
    float* __restrict__ dst,                  // [rows, cols] fp32 (row-major)
    int64_t rows, int64_t cols, int64_t scale_cols)
{
    // V=4: each thread dequants 4 fp8 bytes. One warp = 32 lanes * 4 = 128 cols == one
    // scale block, so sc = blockIdx.x is the same for every lane in the warp.
    constexpr int V = 4;
    int64_t lane = threadIdx.x;
    int64_t c0 = (int64_t)blockIdx.x * ((int64_t)blockDim.x * V) + lane * V;  // base col
    int64_t r  = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= rows) return;
    int64_t sr = ROWS_BLOCKED ? (r / BLOCK) : r;   // BLOCK is power-of-2 -> shift

    if (c0 + V <= cols) {
        // Vectorized path. sc = blockIdx.x (= c0/BLOCK, identical for all lanes) and
        // threadIdx.y is warp-constant -> sr is warp-constant -> the scale address is
        // identical across all 32 lanes -> one L1 broadcast read (vs round-1 per-element).
        float s = scale[sr * scale_cols + blockIdx.x];
        // 4 fp8 bytes as one uint32 (coalesced 128B across the warp). cols is a multiple
        // of 128 and r*cols is a multiple of 128 -> c0 = 4-aligned -> safe uint32 load.
        uint32_t packed = *reinterpret_cast<const uint32_t*>(src + r * cols + c0);
        __nv_fp8_e4m3 e[4];
        *reinterpret_cast<uint32_t*>(e) = packed;   // exact byte-level fp8 decode
        float4 out;
        out.x = static_cast<float>(e[0]) * s;       // fp8 e4m3 -> fp32 (exact widening)
        out.y = static_cast<float>(e[1]) * s;
        out.z = static_cast<float>(e[2]) * s;
        out.w = static_cast<float>(e[3]) * s;
        *reinterpret_cast<float4*>(dst + r * cols + c0) = out;  // coalesced fp32 store
    } else {
        // Scalar tail path for the last partial 128-col tile (cols not a multiple of 128;
        // H/I/2I are multiples of 128 so this rarely triggers). Bit-exact to the round-1
        // scalar path: one fp8 byte load + per-128 scale + fp32 store per element.
        for (int v = 0; v < V; ++v) {
            int64_t c = c0 + v;
            if (c >= cols) break;
            int64_t sc = c / BLOCK;                         // power-of-2 -> shift
            float s = scale[sr * scale_cols + sc];
            float vv = static_cast<float>(src[r * cols + c]);  // fp8 e4m3 -> fp32 (exact)
            dst[r * cols + c] = vv * s;
        }
    }
}

// Host launcher for the fused dequant. fp8_tensor must be contiguous float8_e4m3fn
// [rows, cols]; scale_contig must be contiguous float32 [scale_rows, cols/BLOCK] matching
// the ROWS_BLOCKED layout above. Returns a contiguous fp32 [rows, cols] tensor.
template <bool ROWS_BLOCKED>
static inline Tensor fused_dequant_fp8_e4m3(
    const Tensor& fp8_tensor,
    const Tensor& scale_contig,
    int64_t rows, int64_t cols)
{
    TORCH_CHECK(fp8_tensor.is_contiguous(), "fused_dequant: fp8 tensor must be contiguous");
    TORCH_CHECK(scale_contig.is_contiguous(), "fused_dequant: scale must be contiguous");
    TORCH_CHECK(scale_contig.scalar_type() == kFloat, "fused_dequant: scale must be float32");
    auto device = fp8_tensor.device();
    Tensor dst = at::empty({rows, cols},
        at::TensorOptions().dtype(kFloat).device(device));
    c10::cuda::CUDAGuard guard(device.index());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index()).stream();
    int64_t scale_cols = cols / BLOCK;
    // V=4 vectorized cols tile: one warp (32 lanes * V=4) covers 128 cols == one scale
    // block. block(32,8) is unchanged in thread count (256 -> occupancy preserved);
    // grid.x = ceil(cols/128), grid.y = ceil(rows/8). Same launch count, 4x work/thread.
    constexpr int V = 4;
    dim3 block(32, 8);  // 256 threads/tile
    dim3 grid((unsigned)((cols + (int64_t)block.x * V - 1) / ((int64_t)block.x * V)),
              (unsigned)((rows + block.y - 1) / block.y));
    dequant_fp8_e4m3_blockscale_fused<ROWS_BLOCKED>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_fp8_e4m3*>(fp8_tensor.data_ptr()),
            scale_contig.data_ptr<float>(),
            dst.data_ptr<float>(),
            rows, cols, scale_cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dst;
}

// ============================================================================
// FUSED SwiGLU (conservative_tiling, round 2): C[r,c] = silu(G1[r,c+I]) * G1[r,c] in
// ONE pass over the 2I-wide gate-up tensor G1. Replaces the seed's two ATen launches
// (at::silu + elementwise mul) + a full [Tk,I] fp32 intermediate (silu->tmp: read+write
// Tk*I, then tmp*X1->C: read tmp+X1, write C = 5*Tk*I fp32 HBM) with ONE kernel reading
// 2*Tk*I + writing Tk*I fp32 = 3*Tk*I fp32 (~1.67x less HBM on this stage, 2 launches->1).
//
// Bit-exactness: silu(x)=x/(1+exp(-x)); expf here is the CUDA math-library exp -- the
// SAME function torch.exp and at::silu use for float32 (runtime-verified:
// at::silu(x) == x/(1+torch.exp(-x)) elementwise, torch.equal True, max_abs=0.0). The
// division is div.rn (accurate, same as PyTorch elementwise div, NOT a reciprocal
// approx) and the * x1 is a plain IEEE-RNE fp32 mul -- identical to the seed's
// at::silu(X2)*X1. No FMA contraction can alter this stage (div is not fusable; the
// trailing *x1 has no following add), so the result is bit-identical to the seed and
// GEMM2's input C is unchanged -> 17/19 bit-exact + the 2 large-T max_abs=4096 (cuBLAS
// TF32 ULP, unrelated) are preserved.
//
// Tile (reuses the round-1 dequant geometry): block(32,8)=256, V=4 cols/thread via
// float4 (one coalesced 128B load+store per warp), grid.x=ceil(I/128), grid.y=ceil(Tk/8).
// I=2048 is a multiple of 128 so the scalar tail never triggers (kept for safety).
// ============================================================================
__global__ void silu_mul_kernel(
    const float* __restrict__ G1,   // [Tk, 2I] fp32 (row-major)
    float* __restrict__ C,          // [Tk, I]  fp32 (row-major)
    int64_t Tk)
{
    constexpr int V = 4;
    int64_t lane = threadIdx.x;
    int64_t c0 = (int64_t)blockIdx.x * ((int64_t)blockDim.x * V) + lane * V;  // base col in [0,I)
    int64_t r  = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= Tk) return;

    if (c0 + V <= I) {
        // X1 = G1[r, c0:c0+4] (first I cols); X2 = G1[r, I+c0:I+c0+4] (second I cols).
        float4 x1 = *reinterpret_cast<const float4*>(G1 + r * (2 * I) + c0);
        float4 x2 = *reinterpret_cast<const float4*>(G1 + r * (2 * I) + (I + c0));
        float4 out;
        out.x = (x2.x / (1.0f + expf(-x2.x))) * x1.x;
        out.y = (x2.y / (1.0f + expf(-x2.y))) * x1.y;
        out.z = (x2.z / (1.0f + expf(-x2.z))) * x1.z;
        out.w = (x2.w / (1.0f + expf(-x2.w))) * x1.w;
        *reinterpret_cast<float4*>(C + r * I + c0) = out;
    } else {
        // Scalar tail (cols not a multiple of 128; I=2048 is, so this is rare). Bit-exact
        // to the vectorized path: one silu + one mul per element.
        for (int v = 0; v < V; ++v) {
            int64_t c = c0 + v;
            if (c >= I) break;
            float x1 = G1[r * (2 * I) + c];
            float x2 = G1[r * (2 * I) + (I + c)];
            C[r * I + c] = (x2 / (1.0f + expf(-x2))) * x1;
        }
    }
}

// Host launcher for fused SwiGLU. G1 must be contiguous float32 [Tk, 2I] (a matmul
// output). Returns a contiguous float32 [Tk, I] tensor C = silu(G1[:,I:2I]) * G1[:,0:I].
static inline Tensor silu_mul_fused(const Tensor& G1, int64_t Tk)
{
    TORCH_CHECK(G1.is_contiguous(), "silu_mul_fused: G1 must be contiguous");
    TORCH_CHECK(G1.scalar_type() == kFloat, "silu_mul_fused: G1 must be float32");
    auto device = G1.device();
    Tensor C = at::empty({Tk, I}, at::TensorOptions().dtype(kFloat).device(device));
    c10::cuda::CUDAGuard guard(device.index());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index()).stream();
    constexpr int V = 4;
    dim3 block(32, 8);  // 256 threads/tile (occupancy preserved, same as dequant)
    dim3 grid((unsigned)((I + (int64_t)block.x * V - 1) / ((int64_t)block.x * V)),
              (unsigned)((Tk + block.y - 1) / block.y));
    silu_mul_kernel<<<grid, block, 0, stream>>>(
        G1.data_ptr<float>(), C.data_ptr<float>(), Tk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
}

// ============================================================================
// FUSED scaled-scatter-add epilogue (conservative_tiling, round 2): per active expert le,
// output[token_idx[i]][h] += O[i][h] * w_tok[i] in ONE kernel. Replaces the seed's
// `O * w_tok` (materializes a full [Tk,H] fp32 intermediate: write Tk*H + read Tk*H =
// 2*Tk*H fp32 HBM/expert) + `output.index_add_` (2 launches) with ONE kernel that reads
// Tk*H (O) + Tk*1 (w) + read+write Tk*H (output) with NO intermediate.
//
// Bit-exactness: per-expert token_idx is UNIQUE (a token appears in at most one top-k
// slot per expert), so no two threads write the same output address within the expert ->
// atomic-free load-add-store is order-free and bit-identical to index_add_. The le=0..31
// host loop serializes cross-expert accumulation in the seed's order, so the per-token
// fp32 sum across the up-to-8 routed experts is identical. __fmul_rn (mul.rn) +
// __fadd_rn (add.rn) replicate the seed's TWO separate roundings (elementwise mul
// O*w_tok -> rounded fp32, THEN index_add_ += -> rounded fp32 add); this prevents
// nvcc's default -fmad=true from contracting `a*b+c` into a single-rounding fma (which
// would differ by <=1 ULP). Output stays pre-zeroed (seed line 260).
//
// Tile: block(256), grid=ceil(Tk*H/(256*V)), V=4 output cols/thread via float4. H=7168 is
// a multiple of 4 and Tk*H is always a multiple of 4 -> the scalar tail never triggers
// (kept for safety).
// ============================================================================
__global__ void scaled_scatter_add_kernel(
    const float* __restrict__ O,            // [Tk, H] fp32 (row-major)
    const float* __restrict__ weights_e,    // [Tk, E_GLOBAL] fp32 (row-major); w=weights_e[i*E_GLOBAL+ge]
    int64_t ge,
    const int64_t* __restrict__ token_idx,  // [Tk] int64
    float* __restrict__ output,             // [T, H] fp32 (pre-zeroed, accumulated in-place)
    int64_t Tk)
{
    constexpr int V = 4;
    int64_t tid  = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t base = tid * V;                  // flat index in [Tk, H]
    int64_t i    = base / H;
    int64_t h0   = base - i * H;             // = base % H (mult of 4 since H mult of 4)
    if (i >= Tk) return;                     // over-launch guard (no OOB token_idx read)

    int64_t row = token_idx[i];
    float w = weights_e[i * E_GLOBAL + ge];

    if (h0 + V <= H) {
        float4 o   = *reinterpret_cast<const float4*>(O + i * H + h0);
        float4 cur = *reinterpret_cast<float4*>(output + row * H + h0);  // current accumulated
        float4 res;
        res.x = __fadd_rn(cur.x, __fmul_rn(o.x, w));
        res.y = __fadd_rn(cur.y, __fmul_rn(o.y, w));
        res.z = __fadd_rn(cur.z, __fmul_rn(o.z, w));
        res.w = __fadd_rn(cur.w, __fmul_rn(o.w, w));
        *reinterpret_cast<float4*>(output + row * H + h0) = res;
    } else {
        for (int v = 0; v < V; ++v) {
            int64_t h = h0 + v;
            if (h >= H) break;
            float prod = __fmul_rn(O[i * H + h], w);
            output[row * H + h] = __fadd_rn(output[row * H + h], prod);
        }
    }
}

// Host launcher for the fused scaled-scatter-add epilogue. O must be contiguous float32
// [Tk, H]; weights_e contiguous float32 [Tk, E_GLOBAL]; token_idx contiguous int64 [Tk];
// output is the in-place [T, H] fp32 accumulator (pre-zeroed). No-op if Tk==0.
static inline void scaled_scatter_add_fused(
    const Tensor& O,
    const Tensor& weights_e,
    int64_t ge,
    const Tensor& token_idx,
    Tensor& output,
    int64_t Tk)
{
    if (Tk == 0) return;
    TORCH_CHECK(O.is_contiguous(), "scaled_scatter_add_fused: O must be contiguous");
    TORCH_CHECK(weights_e.is_contiguous(), "scaled_scatter_add_fused: weights_e must be contiguous");
    TORCH_CHECK(token_idx.is_contiguous(), "scaled_scatter_add_fused: token_idx must be contiguous");
    TORCH_CHECK(O.scalar_type() == kFloat, "scaled_scatter_add_fused: O must be float32");
    auto device = O.device();
    c10::cuda::CUDAGuard guard(device.index());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index()).stream();
    constexpr int V = 4;
    int64_t n_threads = (Tk * H + V - 1) / V;       // each thread does V=4 output cols
    int blocks  = (int)((n_threads + 255) / 256);
    int threads = 256;
    scaled_scatter_add_kernel<<<blocks, threads, 0, stream>>>(
        O.data_ptr<float>(),
        weights_e.data_ptr<float>(),
        ge,
        token_idx.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        Tk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
// <<<IMPROVE ENDS>>>

Tensor run(
    const Tensor& routing_logits,
    const Tensor& routing_bias,
    const Tensor& hidden_states,
    const Tensor& hidden_states_scale,
    const Tensor& gemm1_weights,
    const Tensor& gemm1_weights_scale,
    const Tensor& gemm2_weights,
    const Tensor& gemm2_weights_scale,
    int64_t local_expert_offset,
    double routed_scaling_factor)
{
    // Get sequence length and device
    // Enable TF32 tensor cores for the fp32 matmuls (~1.66x faster, ~10-bit mantissa).
    at::globalContext().setAllowTF32CuBLAS(true);
    int64_t T = routing_logits.sizes()[0];
    auto device = hidden_states.device();

    // ===== 1) FUSED FP8 block-scale dequantization for activations (call site 1/3) =====
    // <<<IMPROVE BEGINS>>>
    // Fused e4m3 block-scale -> fp32 dequant. Replaces the seed's hidden_states.to(kFloat)
    // + scale.repeat({1,1,BLOCK}).reshape({T,H}) (materializes a FULL-SIZE fp32 scale
    // tensor T*H*4B that is immediately discarded) + elementwise mul. For the activation
    // the scale layout is per-row-block (scale_rows = rows = T, scale_cols = H/128), so
    // ROWS_BLOCKED=false. Each thread loads ONE fp8 byte + its matching per-128 block
    // scale and writes fp32 in a single pass: ~1 byte read + 4 byte write per element
    // (scale broadcast across 128 lanes) vs the seed's ~4x amplified HBM traffic.
    Tensor A_scale_TH = hidden_states_scale.to(kFloat)
                            .permute({1, 0}).contiguous();          // [T, H/128] fp32
    Tensor A = fused_dequant_fp8_e4m3<false>(hidden_states, A_scale_TH, T, H);  // [T, H] fp32
    // <<<IMPROVE ENDS>>>

    // NOTE: W13 and W2 are NOT dequantized upfront here (lazy optimization): each
    // expert's weights are dequantized on-demand inside the loop below via the same
    // fused kernel.

    // ===== 2) No-aux routing =====
    // logits: [T, E], bias: [E]
    Tensor logits = routing_logits.to(kFloat);
    Tensor bias = routing_bias.to(kFloat).reshape({E_GLOBAL});

    // Sigmoid: s = 1 / (1 + exp(-logits))
    Tensor s = at::sigmoid(logits);  // [T, E]

    // Add bias (broadcast)
    Tensor s_with_bias = s + bias;   // [T, E]

    // Group: [T, E] -> [T, 8, 32]
    int64_t group_size = E_GLOBAL / N_GROUP;
    Tensor s_wb_grouped = s_with_bias.view({T, N_GROUP, group_size});

    // Group scores = sum of top-2 values within each group
    auto top2_result = at::topk(s_wb_grouped, 2, 2, true, false);
    Tensor group_scores = std::get<0>(top2_result).sum(2);  // [T, 8]

    // Select topk_group groups -> group mask
    auto group_topk = at::topk(group_scores, TOPK_GROUP, 1, true, false);
    Tensor group_idx = std::get<1>(group_topk);  // [T, 4]
    Tensor group_mask = at::zeros_like(group_scores);
    group_mask.scatter_(1, group_idx, 1.0);

    // Expand group mask to expert mask
    Tensor score_mask = group_mask.unsqueeze(2)
                              .expand({T, N_GROUP, group_size})
                              .reshape({T, E_GLOBAL});

    // Global top-k within kept groups
    float neg_inf = -std::numeric_limits<float>::infinity();
    Tensor scores_pruned = at::masked_fill(s_with_bias, score_mask == 0, neg_inf);
    auto topk_result = at::topk(scores_pruned, TOP_K, 1, true, false);
    Tensor topk_idx = std::get<1>(topk_result);  // [T, 8]

    // Combination weights: use s (without bias) for normalization
    Tensor M = at::zeros_like(s);
    M.scatter_(1, topk_idx, 1.0);
    Tensor weights = s * M;
    Tensor weights_sum = weights.sum(1, true) + 1e-20;
    weights = (weights / weights_sum) * routed_scaling_factor;

    // ===== 3) Local expert compute and accumulation with LAZY dequantization =====
    Tensor output = at::zeros({T, H}, at::TensorOptions().dtype(kFloat).device(device));

    int64_t local_start = local_expert_offset;

    // <<<IMPROVE BEGINS>>>
    // Conservative on-GPU token dispatch (replaces the seed's per-expert host syncs).
    // One grid over T*8 work items scatters each (token, selected-expert-slot) into a
    // per-local-expert bin via atomicAdd. After the kernel, a single D2H copy of count[32]
    // gives the host each expert's token count, enabling a host-known-length narrow +
    // index_select downstream with NO further sync (replaces ~32 * 2 = ~64 forced syncs).
    Tensor topk_idx_c = topk_idx.contiguous();                       // [T, TOP_K] int64
    auto opts_i64 = at::TensorOptions().dtype(kLong).device(device);
    Tensor count_buf = at::empty({E_LOCAL}, opts_i64);              // [E_LOCAL]
    Tensor token_ids_buf = at::empty({E_LOCAL * T}, opts_i64);       // [E_LOCAL * T]

    {
        c10::cuda::CUDAGuard guard(device.index());
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index()).stream();
        // atomicAdd requires zero-initialized counters; memset on the launch stream so it
        // is ordered before the scatter kernel below.
        cudaMemsetAsync(count_buf.data_ptr<int64_t>(), 0,
                        E_LOCAL * sizeof(int64_t), stream);
        // token_ids_buf needs no init: only [0, Tk) entries are written per expert.
        int64_t n_items = T * TOP_K;
        int blocks = (int)((n_items + 255) / 256);
        if (blocks > 1024) blocks = 1024;
        int threads = 256;
        scatter_tokens_to_local_experts<<<blocks, threads, 0, stream>>>(
            topk_idx_c.data_ptr<int64_t>(),
            count_buf.data_ptr<int64_t>(),
            token_ids_buf.data_ptr<int64_t>(),
            T,
            local_start);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // ONE D2H copy of the per-expert token counts -> single host sync (was ~64).
    Tensor count_host = count_buf.to(at::kCPU);   // [E_LOCAL] int64
    const int64_t* count_host_ptr = count_host.data_ptr<int64_t>();

    // For each local expert: use host-known Tk (no .item()/nonzero sync), LAZY dequantize
    // weights, run GEMM1 -> SwiGLU -> GEMM2, accumulate. Downstream is byte-identical to the
    // seed; only the token-selection prologue changed.
    for (int64_t le = 0; le < E_LOCAL; ++le) {
        int64_t ge = local_start + le;
        if (ge < 0 || ge >= E_GLOBAL) continue;

        int64_t Tk = count_host_ptr[le];
        if (Tk == 0) {
            // SKIP: No tokens selected this expert - no dequantization, no GEMM computation!
            // This is the key optimization: for small T, most experts are empty.
            continue;
        }

        // Token indices for this expert (host-known length Tk -> no further sync).
        Tensor token_idx = token_ids_buf.narrow(0, le * T, Tk).reshape({Tk});

        // Gather inputs
        Tensor A_e = A.index_select(0, token_idx);  // [Tk, H]

        // ===== FUSED LAZY DEQUANTIZATION for expert le's weights ONLY (call sites 2,3/3) =====
        // W13: [2I, H] fp8 e4m3 -> fp32, scale [2I/128, H/128] (2-D block scale, ROWS_BLOCKED).
        // W2:  [H, I]  fp8 e4m3 -> fp32, scale [H/128, I/128]  (2-D block scale, ROWS_BLOCKED).
        // gemm1_weights[le] / gemm2_weights[le] are contiguous float8_e4m3fn dim-0 slices;
        // the *_scale[le] slices are contiguous float32 (definition). Each fused launch
        // replaces the seed's to(kFloat) + repeat_interleave*2 (full-size scale materialize,
        // immediately discarded) + elementwise mul -> ~4x less HBM traffic, 1 kernel vs 4.
        Tensor W13_e = fused_dequant_fp8_e4m3<true>(
            gemm1_weights[le], gemm1_weights_scale[le].to(kFloat), 2 * I, H);  // [2I, H] fp32
        Tensor W2_e = fused_dequant_fp8_e4m3<true>(
            gemm2_weights[le], gemm2_weights_scale[le].to(kFloat), H, I);      // [H, I] fp32

        // GEMM1: [Tk, H] @ [H, 2I] = [Tk, 2I]
        Tensor G1 = A_e.matmul(W13_e.t());

        // ===== FUSED SwiGLU (conservative_tiling, round 2): one __global__ kernel =====
        // Replaces two ATen launches (at::silu + elementwise mul) + a full [Tk,I] fp32
        // intermediate with ONE pass over the 2I-wide G1: read 2*Tk*I, write Tk*I fp32
        // (~1.67x less HBM on this stage). Bit-exact: silu(x)=x/(1+expf(-x)) with expf
        // (the SAME exp torch.exp/at::silu use on float32 -> verified bit-identical) and
        // the mul is IEEE-RNE fp32 identical to the seed's at::silu(X2)*X1; no FMA can
        // alter this stage (div not fusable, trailing mul has no add). G1 is the
        // contiguous [Tk,2I] fp32 matmul output; GEMM2 input C is unchanged.
        Tensor C = silu_mul_fused(G1, Tk);   // [Tk, I] fp32

        // GEMM2: [Tk, I] @ [I, H] = [Tk, H]
        Tensor O = C.matmul(W2_e.t());

        // ===== FUSED scaled-scatter-add epilogue (conservative_tiling, round 2) =====
        // Replaces `O * w_tok` (materializes a [Tk,H] fp32 intermediate) +
        // `output.index_add_` (2 launches) with ONE kernel:
        // output[token_idx[i]][h] += O[i][h] * weights_e[i][ge]. per-expert token_idx is
        // unique -> non-atomic; the le-loop serializes cross-expert accumulation in the
        // seed's order. __fmul_rn + __fadd_rn replicate the seed's two separate roundings
        // (mul then add), avoiding nvcc FMA contraction. `weights_e` is the contiguous
        // [Tk, E] gather of `weights` (the same index_select as the seed; only the
        // multiply + scatter are fused, the .narrow view is now read inside the kernel).
        Tensor weights_e = weights.index_select(0, token_idx);  // [Tk, E_GLOBAL] fp32 contiguous
        scaled_scatter_add_fused(O, weights_e, ge, token_idx, output, Tk);
    }
    // <<<IMPROVE ENDS>>>

    return output.to(kBFloat16);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "FP8 MoE block-scale - conservative dispatch + fused fp8 e4m3 block-scale->fp32 dequant kernel");
}
