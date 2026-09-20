// =============================================================================
// kernel.cu — Task 005_gemm_n256_k7168: C = A @ B.T (N=256, K=7168, M var)
// Target: NVIDIA B300 SXM6 AC (sm_103, 148 SMs), CUDA 13.1 / torch 2.11.
//
// Cycle-1-a2 action: "Fix the M=901 mid band: 0.53x -> >=1.0x".
//
//   Parent (cycle-0-a0): 17/17 pass, geomean 1.5281, sole sub-1x workload
//   M=901 at 0.52999x (0.048544ms vs cuBLAS 0.025728ms). The mid-band config
//   Cfg{64,64,SK=2} launches grid_m*4*2 CTAs = 15*8 = 120 CTAs on 148 SMs —
//   28 SMs idle, ~1.9x slower than cuBLAS (exp/ksearch_c0_a0.bench.jsonl).
//
//   Fix: SM-fill gate on the mid band (a principled variant of "route the
//   underfilled band to cuBLAS"). The megakernel is kept only when its CTA
//   grid fills the 148 SMs (grid_m*8 >= 148, i.e. M >= 1153: at least one
//   full wave); every underfilled shape — M=901's 120-CTA launch included —
//   resolves to no config and falls through to the existing at::matmul
//   (cuBLAS) path, exactly like the large band. For the 17-workload set the
//   only mid-band member is M=901, so it moves 0.53x -> ~1.0x (cuBLAS vs
//   itself, matched ratio 1.0 by construction — the max_relative_error 24.5
//   seen in the parent bench disappears with it) and the geomean lifts
//   ~1.528 -> ~1.59. All other bands are untouched.
//
// Cycle-0-a0 action (parent, preserved): "Port baseline to CUDA .cu entry
// with ctypes graph path removed." Pure CUDA C++ re-implementation of the
// H800 baseline's per-band dispatch (solution.py), with the entire
// Triton/ctypes machinery replaced:
//
//   Bands (ported 1:1 from solution.py::_resolve_launcher):
//     * M <= 80  (skinny band, 14/17 workloads): split-K last-writer
//       megakernel in CUDA C++ using mma.sync.m16n8k16 fp16 tensor cores
//       (native to sm_103), cp.async multi-stage global->shared pipeline.
//         - M <= 16 : BM=16 BN=32 BK=128 SK=14, fp32 partials  (cfg (16,14))
//         - M <= 32 : BM=16 BN=32 BK=128 SK=8 , fp32 partials  (cfg (16,8))
//         - M <= 80 : BM=32 BN=32 BK=128 SK=8/5, fp16 partials (cfg (32,8/5))
//           (fp16 partial handoff preserved from baseline: safe for the
//           BM=32 SK<=8 band only — M<=16 SK=14 fails tolerance there)
//     * 81 <= M <= 11947 (mid band): SM-fill gate (this cycle). M >= 1153
//       (grid_m*8 >= 148): BM=64 BN=64 BK=128 SK=2, fp32 partials, as in the
//       baseline _pick_midm_config; 81 <= M < 1153: cuBLAS via at::matmul.
//     * M >= 11948 (large band): forward to at::matmul (cuBLAS). On H800 the
//       baseline's MXFP8/Stream-K/cuBLAS gates resolved to cuBLAS parity
//       (0.998–1.003x; MXFP8 intrinsically failed the 0.99 match gate), so
//       the port resolves this band straight to cuBLAS.
//
//   DELETED (solution.py:585-856 and all indirect-slot launchers): NO
//   cuGraphExecKernelNodeSetParams_v2, NO cuGraphLaunch, NO hand-typed
//   CUeventParam/MEMCPY3D ctypes structs, no Triton. That path SIGSEGVs under
//   B300 + CUDA 13.1 + torch 2.11. With a plain cudaLaunchKernel, the moving
//   A/B pointers of the harness's ShiftingMemoryPoolAllocator are simply
//   fresh kernel arguments on every call — the problem the ctypes graph
//   rebind tried to solve does not exist here. EPOCH is likewise a live arg.
//
//   Hot path = one unordered_map lookup keyed (M, device) (the _RUN_CACHE
//   port; resolved once per M) + one kernel launch on the current torch
//   stream.
//
// Build: requires -gencode arch=compute_103,code=sm_103 (torch's default
// arch_list stops at sm_100). cp.async / ldmatrix / mma.sync are all native
// to sm_103.
// =============================================================================
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>
#include <unordered_map>

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME kernel
#endif

namespace {

// ---------------------------------------------------------------------------
// Device helpers (all native to sm_103)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void cp_async16(void* smem_dst, const void* gmem_src) {
  const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// ldmatrix.x4: 4x 8x8 b16 matrices; lane quarter j supplies matrix j's 8 row
// addresses (in order) and receives matrix j's fragment in register j, with
// the per-matrix distribution "thread t: row t/4, cols 2*(t%4)+{0,1}" —
// exactly the mma A-fragment for a row-major 16x16 tile when the lanes map
// row = lane%16, kcol-half = (lane/16)*8 (register order = rowhalf + 2*khalf).
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&f)[4], const __half* p) {
  const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(f[0]), "=r"(f[1]), "=r"(f[2]), "=r"(f[3])
               : "r"(addr));
}

// ldmatrix.x2: B is kept in shared memory in its natural global layout [n][k]
// (k contiguous). A non-trans ldmatrix.x2 on rows n0..n0+7 at k-offsets 0/8
// yields exactly the mma B-fragment (col-major KxN) for C = A @ B.T:
// thread t: reg0 = B[k=2tg, 2tg+1][n=g], reg1 = B[k+8, k+9][n=g].
__device__ __forceinline__ void ldmatrix_x2(uint32_t (&f)[2], const __half* p) {
  const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(f[0]), "=r"(f[1])
               : "r"(addr));
}

__device__ __forceinline__ void mma_m16n8k16(float (&d)[4], const uint32_t (&a)[4],
                                             const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ---------------------------------------------------------------------------
// Split-K last-writer megakernel (port of _megakernel_lwr / _megakernel_fp16w)
//
// Grid: (GRID_N, GRID_M, SK); blockIdx.{x,y,z} = {pid_n, pid_m, s}.
// Each CTA computes a BM x BN tile of C over its K-slice s (k_iter = s, s+SK,
// ...), accumulates in fp32 tensor-core registers, writes its partial to
// W[s][m][n] (fp32 or fp16), bumps a per-tile counter, and the last of the
// SK CTAs for the tile reduces all partials and writes C in fp16.
//
// Shared-memory row stride is BK+8 halves (16B pad): an unpadded 256B stride
// would put all 8 rows of an ldmatrix gather in the same bank group (8-way
// conflict); 272B rotates the bank group by 4 words per row.
//
// A rows are clamped (not masked) on load: rows m >= M read row M-1 and the
// garbage is masked out on store — outputs only ever come from valid rows.
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int SK, int WM, int WN, int NS, bool WFP16>
__global__ __launch_bounds__(128) void gemm_splitk_lwr(
    const __half* __restrict__ A, const __half* __restrict__ B,
    void* __restrict__ Wv, __half* __restrict__ C, int* __restrict__ cnt,
    const int M, const int K, const int lda, const int ldb,
    const long long epoch) {
  using WT = std::conditional_t<WFP16, __half, float>;
  WT* W = static_cast<WT*>(Wv);

  constexpr int RSB = BK + 8;        // padded row stride, in halves
  constexpr int GRID_N = 256 / BN;   // exact for BN in {32, 64}
  constexpr int MI = WM / 16;        // mma m-blocks per warp
  constexpr int NI = WN / 8;         // mma n-blocks per warp
  constexpr int WARPS_N = BN / WN;   // 4 warps: (BM/WM) x (BN/WN)
  constexpr int AKC = BK / 8;        // 16B chunks per row
  constexpr int A_CHUNKS = BM * AKC;
  constexpr int B_CHUNKS = BN * AKC;

  const int s = blockIdx.z;
  const int pid_m = blockIdx.y;
  const int pid_n = blockIdx.x;
  const int tile = pid_m * GRID_N + pid_n;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int warp_m = warp / WARPS_N;
  const int warp_n = warp % WARPS_N;

  extern __shared__ __align__(16) char smem_raw[];
  __half* const As = reinterpret_cast<__half*>(smem_raw);
  __half* const Bs = As + static_cast<size_t>(NS) * BM * RSB;
  __shared__ int s_old;

  const int num_k_iters = K / BK;
  const int iters = (num_k_iters - s + SK - 1) / SK;  // >= 1 for K=7168

  float acc[MI][NI][4];
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int ni = 0; ni < NI; ++ni)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[mi][ni][r] = 0.0f;

  // ---- global -> shared: one cp.async stage issue (A rows, B rows) ----
  // Captured by value: no local-memory spill of kernel parameters.
  const auto issue = [=](int idx, int stage) {
    const int kbase = (s + idx * SK) * BK;
    __half* const AsS = As + stage * (BM * RSB);
    __half* const BsS = Bs + stage * (BN * RSB);
#pragma unroll
    for (int i = 0; i < A_CHUNKS / 128; ++i) {
      const int c = tid + i * 128;
      const int m = c / AKC;
      const int kk = (c % AKC) * 8;
      int grow = pid_m * BM + m;
      if (grow >= M) grow = M - 1;  // clamp; masked on store
      cp_async16(AsS + m * RSB + kk, A + static_cast<size_t>(grow) * lda + kbase + kk);
    }
#pragma unroll
    for (int i = 0; i < B_CHUNKS / 128; ++i) {
      const int c = tid + i * 128;
      const int n = c / AKC;
      const int kk = (c % AKC) * 8;
      cp_async16(BsS + n * RSB + kk,
                 B + static_cast<size_t>(pid_n * BN + n) * ldb + kbase + kk);
    }
    cp_async_commit();
  };

  const int pro = (NS - 1 < iters) ? (NS - 1) : iters;
  for (int p = 0; p < pro; ++p) issue(p, p);

  for (int i = 0; i < iters; ++i) {
    const int stage = i % NS;
    const bool more = (i + NS - 1 < iters);
    if (more)
      cp_async_wait<NS - 2>();  // steady state: oldest (== stage i) completes
    else
      cp_async_wait<0>();       // tail / short loops: everything completes
    __syncthreads();

    const __half* const AsS = As + stage * (BM * RSB);
    const __half* const BsS = Bs + stage * (BN * RSB);
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      uint32_t af[MI][4];
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
        // lane l -> row l%16 of the 16x16 block, k-half l/16
        ldmatrix_x4(af[mi],
                    AsS + (warp_m * WM + mi * 16 + (lane % 16)) * RSB + kk +
                        (lane / 16) * 8);
      }
      uint32_t bf[NI][2];
#pragma unroll
      for (int ni = 0; ni < NI; ++ni) {
        // lanes 0-7 -> rows n0..n0+7 at kk; lanes 8-15 -> same at kk+8
        ldmatrix_x2(bf[ni],
                    BsS + (warp_n * WN + ni * 8 + (lane % 8)) * RSB + kk +
                        ((lane >> 3) & 1) * 8);
      }
#pragma unroll
      for (int mi = 0; mi < MI; ++mi)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) mma_m16n8k16(acc[mi][ni], af[mi], bf[ni]);
    }
    __syncthreads();  // stage (i-1)%NS is rewritten by the issue below
    if (more) issue(i + NS - 1, (i + NS - 1) % NS);
  }

  // ---- store this CTA's partial to W[s][m][n] ----
  // mma D-fragment: c0=(g,2tg) c1=(g,2tg+1) c2=(g+8,2tg) c3=(g+8,2tg+1)
  {
    const int g = lane >> 2;
    const int tg = lane & 3;
#pragma unroll
    for (int mi = 0; mi < MI; ++mi)
#pragma unroll
      for (int ni = 0; ni < NI; ++ni)
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const int m = warp_m * WM + mi * 16 + g + (r >> 1) * 8;
          const int n = warp_n * WN + ni * 8 + 2 * tg + (r & 1);
          const int gm = pid_m * BM + m;
          if (gm < M) {
            const size_t off =
                (static_cast<size_t>(s) * M + gm) * 256 + (pid_n * BN + n);
            if constexpr (WFP16)
              W[off] = __float2half(acc[mi][ni][r]);
            else
              W[off] = acc[mi][ni][r];
          }
        }
  }

  // ---- last-writer reduce (the baseline's epoch counter protocol) ----
  __threadfence();  // release: this CTA's partials are device-visible
  __syncthreads();
  if (tid == 0) s_old = atomicAdd(&cnt[tile], 1);
  __syncthreads();
  if (s_old == static_cast<int>(epoch * static_cast<long long>(SK) + (SK - 1))) {
    __threadfence();  // acquire: other CTAs' partials are visible
    for (int e = tid; e < BM * BN; e += 128) {
      const int m = e / BN;
      const int n = e % BN;
      const int gm = pid_m * BM + m;
      if (gm < M) {
        const int gn = pid_n * BN + n;
        float racc = 0.0f;
#pragma unroll
        for (int ss = 0; ss < SK; ++ss) {
          const size_t off = (static_cast<size_t>(ss) * M + gm) * 256 + gn;
          if constexpr (WFP16)
            racc += __half2float(W[off]);
          else
            racc += W[off];
        }
        C[static_cast<size_t>(gm) * 256 + gn] = __float2half(racc);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Host side: band dispatch + per-(M, device) launcher cache (_RUN_CACHE port)
// ---------------------------------------------------------------------------
using KernFn = void (*)(const __half*, const __half*, void*, __half*, int*, int,
                        int, int, int, long long);

struct Cfg {
  int bm, bn, sk, ns;
  bool wfp;  // fp16 split-K partials (baseline _use_fp16w gating)
};

// Target: B300 SXM6 AC (sm_103) — 148 SMs (task hardware contract).
constexpr int kNumSMs = 148;

// Port of solution.py::_pick_config / _pick_midm_config band tables, with the
// cycle-1-a2 SM-fill gate on the mid band.
// M <= 80 combos are exhaustive for the workload set's small band.
bool resolve_cfg(const int64_t M, Cfg& c) {
  if (M <= 16) {
    c = Cfg{16, 32, 14, 6, false};
    return true;
  }
  if (M <= 32) {
    c = Cfg{16, 32, 8, 6, false};
    return true;
  }
  if (M <= 80) {
    const int grid_m = static_cast<int>((M + 31) / 32);
    const int base = grid_m * 8;  // GRID_N = 256/32 = 8
    int sk = (120 + base / 2) / base;
    if (sk < 4) sk = 4;
    if (sk > 14) sk = 14;
    if (sk != 8 && sk != 5) return false;  // unreachable for M in 33..80
    // baseline cycle-8-a1: M=80 band (grid_m>=3 AND sk<=5) -> ns=6
    c = Cfg{32, 32, sk, (grid_m >= 3 && sk <= 5) ? 6 : 5, true};
    return true;
  }
  // Mid band (81..11947; M=901 workload): the Cfg{64,64,2,3} megakernel
  // launches grid_m * (256/64) * 2 = grid_m*8 CTAs. Keep it only when that
  // fills the machine (grid_m*8 >= 148, i.e. M >= 1153: at least one full
  // wave); otherwise return false and let run() forward to cuBLAS. M=901 ->
  // 15*8 = 120 CTAs on 148 SMs = 28 idle SMs and a measured 0.52999x
  // (0.048544ms vs cuBLAS 0.025728ms, exp/ksearch_c0_a0.bench.jsonl) —
  // exactly the shapes the gate sends to cuBLAS.
  const int grid_m = static_cast<int>((M + 63) / 64);
  if (grid_m * 8 < kNumSMs) return false;
  c = Cfg{64, 64, 2, 3, false};  // mid band, grid-filling shapes (M >= 1153)
  return true;
}

KernFn pick_kernel(const Cfg& c) {
  if (c.bm == 16) {
    if (c.sk == 14) return &gemm_splitk_lwr<16, 32, 128, 14, 16, 8, 6, false>;
    return &gemm_splitk_lwr<16, 32, 128, 8, 16, 8, 6, false>;
  }
  if (c.bm == 32) {
    if (c.sk == 8) return &gemm_splitk_lwr<32, 32, 128, 8, 16, 16, 5, true>;
    return &gemm_splitk_lwr<32, 32, 128, 5, 16, 16, 6, true>;
  }
  return &gemm_splitk_lwr<64, 64, 128, 2, 32, 32, 3, false>;
}

int smem_bytes(const Cfg& c) { return c.ns * (c.bm + c.bn) * (128 + 8) * 2; }

struct BandState {
  torch::Tensor W, C, cnt;
  long long epoch = 0;
  KernFn kfn = nullptr;
  int grid_m = 0, grid_n = 0, sk = 0, smem = 0;
  bool ready = false;
};

std::unordered_map<long long, BandState>& band_states() {
  static std::unordered_map<long long, BandState> states;
  return states;
}

}  // namespace

// Entry point: run(A, B) -> C, matching the reference signature exactly.
torch::Tensor run(torch::Tensor A, torch::Tensor B) {
  const int64_t M = A.size(0);
  const int64_t K = A.size(1);
  const int64_t N = B.size(0);

  // Large-M band (M >= 11948): straight to cuBLAS. The baseline's H800 gates
  // (MXFP8 / Stream-K / cuBLAS) resolved here to cuBLAS parity anyway, and a
  // wrong-guess custom kernel at large M would drag the geomean.
  if (M > 11947) return at::matmul(A, B.t());

  // Anything off-contract (dtype/shape/contiguity) falls back to cuBLAS.
  const bool shape_ok = A.is_cuda() && B.is_cuda() &&
                        A.scalar_type() == at::kHalf && B.scalar_type() == at::kHalf &&
                        A.dim() == 2 && B.dim() == 2 && N == 256 && K == B.size(1) &&
                        M > 0 && K > 0 && K % 128 == 0 &&
                        A.is_contiguous() && B.is_contiguous();
  if (!shape_ok) return at::matmul(A, B.t());

  Cfg c;
  // Mid-band shapes whose grid underfills the 148 SMs (M=901: 120 CTAs)
  // resolve to no config and take the cuBLAS path here — the cycle-1-a2 fix.
  if (!resolve_cfg(M, c)) return at::matmul(A, B.t());
  KernFn kfn = pick_kernel(c);

  const at::cuda::CUDAGuard guard(A.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Per-(M, device) launcher state, resolved once (the _RUN_CACHE port).
  const long long key = (M << 8) | static_cast<long long>(A.get_device());
  BandState& S = band_states()[key];

  if (!S.ready) {
    const int grid_n = 256 / c.bn;
    const int grid_m = static_cast<int>((M + c.bm - 1) / c.bm);
    const int smem = smem_bytes(c);
    // All configs use > 48KB dynamic smem: opt in per kernel instantiation.
    if (cudaFuncSetAttribute(reinterpret_cast<const void*>(kfn),
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem) != cudaSuccess) {
      return at::matmul(A, B.t());  // safe cuBLAS fallback
    }
    const auto opts = torch::TensorOptions().device(A.device());
    S.W = torch::empty({static_cast<int64_t>(c.sk), M, 256},
                       opts.dtype(c.wfp ? torch::kHalf : torch::kFloat));
    S.C = torch::empty({M, 256}, opts.dtype(torch::kHalf));
    S.cnt = torch::zeros({static_cast<int64_t>(grid_m * grid_n)},
                         opts.dtype(torch::kInt32));
    S.kfn = kfn;
    S.grid_m = grid_m;
    S.grid_n = grid_n;
    S.sk = c.sk;
    S.smem = smem;
    S.epoch = 0;
    S.ready = true;
  }

  long long e = S.epoch;
  if (e > 100000000LL) {  // counter reset, as in the baseline
    cudaMemsetAsync(S.cnt.data_ptr(), 0, S.cnt.numel() * sizeof(int), stream);
    e = 0;
    S.epoch = 0;
  }

  // Plain kernel launch: A/B pointers and EPOCH are live arguments, so the
  // harness's shifting allocator is handled for free (no graph, no rebind).
  const dim3 grid(static_cast<unsigned>(S.grid_n), static_cast<unsigned>(S.grid_m),
                  static_cast<unsigned>(S.sk));
  S.kfn<<<grid, 128, S.smem, stream>>>(
      static_cast<const __half*>(A.data_ptr()),
      static_cast<const __half*>(B.data_ptr()), S.W.data_ptr(),
      static_cast<__half*>(S.C.data_ptr()),
      static_cast<int*>(S.cnt.data_ptr()), static_cast<int>(M),
      static_cast<int>(K), static_cast<int>(K), static_cast<int>(K), e);
  S.epoch = e + 1;
  return S.C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run,
        "005_gemm_n256_k7168: C = A @ B.T (B300 sm_103 CUDA port, no graph path)");
}
