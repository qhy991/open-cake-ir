// B300 development probe: route BF16 rows into expert-owned contiguous bins.
// One GPU observes all four source ranks. This is not an EP4 mailbox or a
// replacement for Cake's typed ranked effects and native lowering.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

namespace {
constexpr int R = 4;
constexpr int T = 512;
constexpr int K = 8;
constexpr int E = 128;
constexpr int H = 2048;
constexpr int LOCAL_E = E / R;
constexpr int MAX_ROWS_PER_EXPERT = R * T;
constexpr int ROUTES = R * T * K;

// The returned old value is the unique row reservation within one expert.
// GPU-scope relaxed ordering is sufficient here because a subsequent kernel
// boundary and host synchronization precede all reads. A live cross-device
// tile queue will require system release/acquire publication separately.
__device__ __forceinline__ unsigned reserve_row(unsigned* count) {
  unsigned old;
  asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;"
               : "=r"(old) : "l"(count), "r"(1u) : "memory");
  return old;
}

__global__ void pack_kernel(const uint16_t* hidden, const int32_t* expert_ids,
                            unsigned* counts, int32_t* route_keys,
                            uint16_t* packed, int* error, int owner_rank) {
  const int key = int(blockIdx.x);
  const int expert = expert_ids[key];
  if (expert < 0 || expert >= E) {
    if (threadIdx.x == 0) atomicExch(error, 1);
    return;
  }
  if (expert / LOCAL_E != owner_rank) return;
  __shared__ unsigned row;
  if (threadIdx.x == 0) {
    const int local_expert = expert - owner_rank * LOCAL_E;
    row = reserve_row(&counts[local_expert]);
    if (row < MAX_ROWS_PER_EXPERT)
      route_keys[local_expert * MAX_ROWS_PER_EXPERT + row] = key;
    else
      atomicExch(error, 2);
  }
  __syncthreads();
  if (row >= MAX_ROWS_PER_EXPERT) return;
  const int token = key / K;
  const int local_expert = expert - owner_rank * LOCAL_E;
  const uint16_t* source = hidden + size_t(token) * H;
  uint16_t* destination = packed +
      (size_t(local_expert) * MAX_ROWS_PER_EXPERT + row) * H;
  for (int feature = int(threadIdx.x); feature < H; feature += blockDim.x)
    destination[feature] = source[feature];
}
}  // namespace

extern "C" int cake_weave_bin_pack_launch(
    const void* hidden, const void* expert_ids, void* counts, void* route_keys,
    void* packed, void* error, int owner_rank, void* stream) {
  if (!hidden || !expert_ids || !counts || !route_keys || !packed || !error ||
      owner_rank < 0 || owner_rank >= R)
    return int(cudaErrorInvalidValue);
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return int(status);
  cudaDeviceProp prop;
  status = cudaGetDeviceProperties(&prop, device);
  if (status != cudaSuccess) return int(status);
  if (prop.major != 10 || prop.minor != 3 ||
      (std::strcmp(prop.name, "NVIDIA B300") != 0 &&
       std::strcmp(prop.name, "NVIDIA B300 SXM6 AC") != 0))
    return int(cudaErrorInvalidDevice);
  pack_kernel<<<ROUTES, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint16_t*>(hidden),
      static_cast<const int32_t*>(expert_ids),
      static_cast<unsigned*>(counts), static_cast<int32_t*>(route_keys),
      static_cast<uint16_t*>(packed), static_cast<int*>(error), owner_rank);
  return int(cudaGetLastError());
}
