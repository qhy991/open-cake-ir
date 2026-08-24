// CAKE_REPRO_LAUNCH_V1 {"abi":"flash_kmeans_assign_v1","block":[1,1,1],"dynamic_shared_memory_bytes":0,"grid":[1,1,1],"kernel_name":"flash_kmeans_assign_candidate","schema_version":1,"target":"sm_100a"}
#include <cuda_bf16.h>
#include <stdint.h>

extern "C" __global__ void flash_kmeans_assign_candidate(
    const __nv_bfloat16* tokens,
    const __nv_bfloat16* centroids,
    const float* centroid_sq,
    int32_t* assignments) {
  // Intentionally empty: the author owns every implementation decision.
}
