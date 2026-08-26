// CAKE_REPRO_LAUNCH_V1 {"abi":"flash_kmeans_assign_v1","block":[64,1,1],"dynamic_shared_memory_bytes":0,"grid":[512,32,1],"kernel_name":"cake_flash_kmeans_assign","schema_version":1,"target":"sm_100a"}
#include <cuda_bf16.h>
#include <mma.h>
#include <stdint.h>

extern "C" __global__ void cake_flash_kmeans_assign(
    const __nv_bfloat16* tokens,
    const __nv_bfloat16* centroids,
    const float* centroid_sq,
    int32_t* assignments) {
  using namespace nvcuda;

  constexpr int kN = 65536;
  constexpr int kD = 128;
  constexpr int kK = 1024;
  constexpr int kTokensPerBlock = 128;
  constexpr int kTokensPerWarp = 64;

  __shared__ __nv_bfloat16 token_tile[kTokensPerBlock * kD];
  __shared__ float score_tile[2][4][16 * 16];

  const int b = static_cast<int>(blockIdx.y);
  const int block_token = static_cast<int>(blockIdx.x) * kTokensPerBlock;
  const long long token_base =
      (static_cast<long long>(b) * kN + block_token) * kD;

  for (int i = static_cast<int>(threadIdx.x);
       i < kTokensPerBlock * kD;
       i += static_cast<int>(blockDim.x)) {
    token_tile[i] = tokens[token_base + i];
  }
  __syncthreads();

  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp_token = warp * kTokensPerWarp;

  float best0 = 3.402823466e+38F;
  float best1 = 3.402823466e+38F;
  int best_k0 = 0;
  int best_k1 = 0;

  #pragma unroll 1
  for (int k0 = 0; k0 < kK; k0 += 16) {
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc1;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc2;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc3;
    wmma::fill_fragment(acc0, 0.0F);
    wmma::fill_fragment(acc1, 0.0F);
    wmma::fill_fragment(acc2, 0.0F);
    wmma::fill_fragment(acc3, 0.0F);

    #pragma unroll
    for (int d0 = 0; d0 < kD; d0 += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16,
                     __nv_bfloat16, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16,
                     __nv_bfloat16, wmma::col_major> x;

      const __nv_bfloat16* a_ptr =
          centroids +
          (static_cast<long long>(b) * kK + k0) * kD + d0;
      wmma::load_matrix_sync(a, a_ptr, kD);

      wmma::load_matrix_sync(
          x, token_tile + (warp_token + 0) * kD + d0, kD);
      wmma::mma_sync(acc0, a, x, acc0);
      wmma::load_matrix_sync(
          x, token_tile + (warp_token + 16) * kD + d0, kD);
      wmma::mma_sync(acc1, a, x, acc1);
      wmma::load_matrix_sync(
          x, token_tile + (warp_token + 32) * kD + d0, kD);
      wmma::mma_sync(acc2, a, x, acc2);
      wmma::load_matrix_sync(
          x, token_tile + (warp_token + 48) * kD + d0, kD);
      wmma::mma_sync(acc3, a, x, acc3);
    }

    wmma::store_matrix_sync(
        &score_tile[warp][0][0], acc0, 16, wmma::mem_row_major);
    wmma::store_matrix_sync(
        &score_tile[warp][1][0], acc1, 16, wmma::mem_row_major);
    wmma::store_matrix_sync(
        &score_tile[warp][2][0], acc2, 16, wmma::mem_row_major);
    wmma::store_matrix_sync(
        &score_tile[warp][3][0], acc3, 16, wmma::mem_row_major);
    __syncwarp();

    const int column = lane & 15;
    const int first_tile = lane >> 4;
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
      const int k = k0 + r;
      const float norm = centroid_sq[static_cast<long long>(b) * kK + k];
      const float dot0 = score_tile[warp][first_tile][r * 16 + column];
      const float dot1 = score_tile[warp][first_tile + 2][r * 16 + column];
      const float candidate0 = norm - 2.0F * dot0;
      const float candidate1 = norm - 2.0F * dot1;
      if (candidate0 < best0) {
        best0 = candidate0;
        best_k0 = k;
      }
      if (candidate1 < best1) {
        best1 = candidate1;
        best_k1 = k;
      }
    }
    __syncwarp();
  }

  const long long output_base = static_cast<long long>(b) * kN + block_token;
  assignments[output_base + warp_token + lane] = best_k0;
  assignments[output_base + warp_token + lane + 32] = best_k1;
}
