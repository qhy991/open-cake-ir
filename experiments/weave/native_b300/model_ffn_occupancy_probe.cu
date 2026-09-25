// Query the exact Cake model FFN tensor-core stage's B300 residency.
// Compile once with CAKE_UPGATE and once with CAKE_DOWN, using the retained
// generated kernel source. This does not measure kernel latency.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

#if defined(CAKE_UPGATE) && !defined(CAKE_DOWN)
#include "up_gate/kernel.cu"
#define CAKE_STAGE "up_gate"
#define CAKE_KERNEL cake_weave_model_upgate_kernel
#elif defined(CAKE_DOWN) && !defined(CAKE_UPGATE)
#include "down/kernel.cu"
#define CAKE_STAGE "down"
#define CAKE_KERNEL cake_weave_model_down_kernel
#else
#error "Select exactly one model FFN tensor-core stage"
#endif

static int fail(const char* name, cudaError_t error) {
  std::fprintf(stderr, "%s: %s (%d)\n", name, cudaGetErrorString(error), int(error));
  return 1;
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  int count = 0;
  cudaError_t error = cudaGetDeviceCount(&count);
  if (error != cudaSuccess) return fail("cudaGetDeviceCount", error);
  if (count != 1) return 3;
  int device = -1;
  error = cudaGetDevice(&device);
  if (error != cudaSuccess) return fail("cudaGetDevice", error);
  cudaDeviceProp prop{};
  error = cudaGetDeviceProperties(&prop, device);
  if (error != cudaSuccess) return fail("cudaGetDeviceProperties", error);
  if (prop.major != 10 || prop.minor != 3 ||
      (std::strcmp(prop.name, "NVIDIA B300") != 0 &&
       std::strcmp(prop.name, "NVIDIA B300 SXM6 AC") != 0))
    return 4;
  int cooperative = 0;
  error = cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device);
  if (error != cudaSuccess) return fail("cudaDevAttrCooperativeLaunch", error);
  if (cooperative != 1) return 5;
  constexpr int threads = 192;
  constexpr int dynamic_shared = 49200;
  error = cudaFuncSetAttribute(CAKE_KERNEL,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              dynamic_shared);
  if (error != cudaSuccess) return fail("cudaFuncSetAttribute", error);
  cudaFuncAttributes attributes{};
  error = cudaFuncGetAttributes(&attributes, CAKE_KERNEL);
  if (error != cudaSuccess) return fail("cudaFuncGetAttributes", error);
  int active = 0;
  error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, CAKE_KERNEL, threads, dynamic_shared);
  if (error != cudaSuccess) return fail("cudaOccupancyMaxActiveBlocksPerMultiprocessor", error);
  if (active < 1) return 6;
  FILE* report = std::fopen(argv[1], "wx");
  if (!report) return 7;
  std::fprintf(report,
      "{\"stage\":\"%s\",\"target\":\"sm_103a\","
      "\"device_name\":\"%s\",\"sm_count\":%d,"
      "\"cooperative_launch\":%d,\"threads_per_cta\":%d,"
      "\"dynamic_shared_bytes\":%d,\"registers_per_thread\":%d,"
      "\"static_shared_bytes\":%zu,\"active_blocks_per_sm\":%d,"
      "\"cooperative_grid_cta_upper_bound\":%d}\n",
      CAKE_STAGE, prop.name, prop.multiProcessorCount, cooperative, threads,
      dynamic_shared, attributes.numRegs, attributes.sharedSizeBytes,
      active, active * prop.multiProcessorCount);
  std::fclose(report);
  std::printf("%s: %d active CTA/SM, %d SMs, grid upper bound %d CTAs\n",
              CAKE_STAGE, active, prop.multiProcessorCount,
              active * prop.multiProcessorCount);
  return 0;
}
