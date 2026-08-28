#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cmath>

namespace {

constexpr int kTokens = 32768;
constexpr int kBlocks = 8192;
constexpr int kHeadDim = 128;
constexpr int kIndexHeads = 8;
constexpr int kQueryHeads = 32;
constexpr int kKvHeads = 4;
constexpr int kSelectedBlocks = 512;
constexpr int kSelectedTokens = 2048;
constexpr float kScale = 0.08838834764831844055F;

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  return value;
}

__device__ __forceinline__ float block_sum_128(float value) {
  __shared__ float warp_sums[4];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_sum(value);
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();
  float total = threadIdx.x < 4 ? warp_sums[lane] : 0.0F;
  if (warp == 0) {
    total = warp_sum(total);
  }
  if (threadIdx.x == 0) {
    warp_sums[0] = total;
  }
  __syncthreads();
  return warp_sums[0];
}

__device__ __forceinline__ bool better(float left_score, int left_index,
                                       float right_score, int right_index) {
  return left_score > right_score ||
         (left_score == right_score && left_index < right_index);
}

}  // namespace

extern "C" __global__ void qsa_pool_layernorm(
    const __nv_bfloat16* index_k, const float* weight, float* normalized_keys) {
  const int block = blockIdx.x;
  const int dimension = threadIdx.x;
  float pooled = 0.0F;
  #pragma unroll
  for (int token = 0; token < 4; ++token) {
    pooled += __bfloat162float(
        index_k[(block * 4 + token) * kHeadDim + dimension]);
  }
  pooled *= 0.25F;
  const float mean = block_sum_128(pooled) * (1.0F / 128.0F);
  const float centered = pooled - mean;
  const float variance =
      block_sum_128(centered * centered) * (1.0F / 128.0F);
  normalized_keys[block * kHeadDim + dimension] =
      centered * rsqrtf(variance + 1.0e-6F) * weight[dimension];
}

extern "C" __global__ void qsa_score_topk(
    const __nv_bfloat16* index_q, const float* normalized_keys,
    int* block_indices) {
  extern __shared__ unsigned char storage[];
  float* scores = reinterpret_cast<float*>(storage);
  int* indices = reinterpret_cast<int*>(scores + kBlocks);
  __nv_bfloat16* query =
      reinterpret_cast<__nv_bfloat16*>(indices + kBlocks);
  const int query_index = blockIdx.x;
  for (int offset = threadIdx.x; offset < kIndexHeads * kHeadDim;
       offset += blockDim.x) {
    query[offset] = index_q[query_index * kIndexHeads * kHeadDim + offset];
  }
  __syncthreads();

  const int complete_blocks = min((query_index + 1) / 4, kBlocks);
  for (int block = threadIdx.x; block < kBlocks; block += blockDim.x) {
    float score = -CUDART_INF_F;
    int index = INT_MAX;
    if (block < complete_blocks) {
      float sum = 0.0F;
      #pragma unroll
      for (int head = 0; head < kIndexHeads; ++head) {
        float dot = 0.0F;
        #pragma unroll 4
        for (int dimension = 0; dimension < kHeadDim; ++dimension) {
          dot = fmaf(
              __bfloat162float(query[head * kHeadDim + dimension]),
              normalized_keys[block * kHeadDim + dimension], dot);
        }
        sum += fmaxf(dot, 0.0F);
      }
      score = sum * kScale;
      index = block;
    }
    scores[block] = score;
    indices[block] = index;
  }
  __syncthreads();

  // Bitonic order is best score first, with the block id as the deterministic tie break.
  for (int width = 2; width <= kBlocks; width <<= 1) {
    for (int stride = width >> 1; stride > 0; stride >>= 1) {
      for (int left = threadIdx.x; left < kBlocks; left += blockDim.x) {
        const int right = left ^ stride;
        if (right > left) {
          const bool best_first = (left & width) == 0;
          const float left_score = scores[left];
          const float right_score = scores[right];
          const int left_index = indices[left];
          const int right_index = indices[right];
          const bool swap =
              best_first
                  ? better(right_score, right_index, left_score, left_index)
                  : better(left_score, left_index, right_score, right_index);
          if (swap) {
            scores[left] = right_score;
            scores[right] = left_score;
            indices[left] = right_index;
            indices[right] = left_index;
          }
        }
      }
      __syncthreads();
    }
  }
  for (int slot = threadIdx.x; slot < kSelectedBlocks; slot += blockDim.x) {
    const int index = indices[slot];
    block_indices[query_index * kSelectedBlocks + slot] =
        index == INT_MAX ? -1 : index;
  }
}

extern "C" __global__ void qsa_expand(const int* block_indices,
                                       int* token_indices) {
  const int query = blockIdx.x;
  for (int slot = threadIdx.x; slot < kSelectedTokens; slot += blockDim.x) {
    const int block =
        block_indices[query * kSelectedBlocks + slot / 4];
    token_indices[query * kSelectedTokens + slot] =
        block < 0 ? -1 : block * 4 + slot % 4;
  }
}

extern "C" __global__ void qsa_attention(
    const __nv_bfloat16* q, const __nv_bfloat16* k,
    const __nv_bfloat16* v, const int* token_indices,
    __nv_bfloat16* output) {
  const int query_index = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int query_head = kv_head * 8 + warp;
  float query_values[4];
  float accumulator[4] = {0.0F, 0.0F, 0.0F, 0.0F};
  #pragma unroll
  for (int item = 0; item < 4; ++item) {
    const int dimension = lane + item * 32;
    query_values[item] = __bfloat162float(
        q[(query_index * kQueryHeads + query_head) * kHeadDim + dimension]);
  }
  float running_max = -CUDART_INF_F;
  float running_sum = 0.0F;

  for (int slot = 0; slot < kSelectedTokens; ++slot) {
    int token = lane == 0
                    ? token_indices[query_index * kSelectedTokens + slot]
                    : 0;
    token = __shfl_sync(0xffffffffU, token, 0);
    if (token < 0) {
      continue;
    }
    float dot = 0.0F;
    #pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int dimension = lane + item * 32;
      dot = fmaf(
          query_values[item],
          __bfloat162float(
              k[(token * kKvHeads + kv_head) * kHeadDim + dimension]),
          dot);
    }
    dot = warp_sum(dot);
    const float logit = __shfl_sync(0xffffffffU, dot, 0) * kScale;
    const float next_max = fmaxf(running_max, logit);
    const float old_scale =
        isinf(running_max) ? 0.0F : __expf(running_max - next_max);
    const float weight_value = __expf(logit - next_max);
    #pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int dimension = lane + item * 32;
      const float value = __bfloat162float(
          v[(token * kKvHeads + kv_head) * kHeadDim + dimension]);
      accumulator[item] =
          accumulator[item] * old_scale + weight_value * value;
    }
    running_sum = running_sum * old_scale + weight_value;
    running_max = next_max;
  }

  #pragma unroll
  for (int item = 0; item < 4; ++item) {
    const int dimension = lane + item * 32;
    const float value = running_sum > 0.0F
                            ? accumulator[item] / running_sum
                            : 0.0F;
    output[(query_index * kQueryHeads + query_head) * kHeadDim + dimension] =
        __float2bfloat16_rn(value);
  }
}
