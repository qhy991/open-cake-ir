// B300 development probe for a tile worker's spatial, wave and steal roles.
// One cooperative grid reserves the tensor-core worker's 192-thread/49,200-B
// CTA footprint. A CTA allocates TMEM at its first claimed tile and releases
// it once after all waves, matching a persistent worker's resource lifetime.
// There is no FFN arithmetic or cross-device mailbox in this probe.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 1030
#error "Exact sm_103a is required"
#endif

namespace cg = cooperative_groups;
namespace {
constexpr int kThreads = 192;
constexpr int kDynamicShared = 49200;
constexpr int kWaves = 4;
constexpr int kMaxTasks = 64;

__device__ __forceinline__ uint32_t smem_address(const void* pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}
__device__ __forceinline__ int claim_old(int* pointer) {
  int old;
  asm volatile("atom.relaxed.gpu.global.add.s32 %0, [%1], %2;"
               : "=r"(old) : "l"(pointer), "r"(1) : "memory");
  return old;
}

__global__ void tile_schedule_probe(
    int* task_heads, int* task_owner, int* processed, int* dispatched,
    int* stolen, int* steal_permits, const int* task_counts,
    int communication_ctas, int steal_budget) {
  extern __shared__ __align__(1024) unsigned char shared[];
  uint32_t* tensor_address = reinterpret_cast<uint32_t*>(shared + 49192);
  __shared__ int claimed;
  __shared__ int borrowed;
  const int block = int(blockIdx.x);
  const int warp = int(threadIdx.x) / 32;
  cg::grid_group grid = cg::this_grid();
  bool tensor_owned = false;
  for (int wave = 0; wave < kWaves; ++wave) {
    if (threadIdx.x == 0 && block < communication_ctas)
      atomicAdd(&dispatched[wave], 1);
    grid.sync();  // Every source-chunk dispatch precedes its tile claims.
    while (true) {
      if (threadIdx.x == 0) {
        bool comm = block < communication_ctas;
        bool permitted = !comm;
        bool reserved_permit = false;
        if (comm && task_counts[wave] > 0 && steal_budget > 0) {
          int old = claim_old(steal_permits);
          if (old < steal_budget) {
            permitted = true;
            reserved_permit = true;
          } else {
            atomicSub(steal_permits, 1);
          }
        }
        claimed = -1;
        borrowed = 0;
        if (permitted) {
          int old = claim_old(&task_heads[wave]);
          if (old < task_counts[wave]) {
            claimed = old;
            borrowed = int(comm);
          } else if (reserved_permit) {
            atomicSub(steal_permits, 1);
          }
        }
      }
      __syncthreads();
      if (claimed < 0) break;
      if (!tensor_owned) {
        if (warp == 0) {
          asm volatile(
              "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
              :: "r"(smem_address(tensor_address)), "n"(64) : "memory");
        }
        __syncthreads();
        tensor_owned = true;
      }
      if (threadIdx.x == 0) {
        task_owner[wave * kMaxTasks + claimed] = block;
        atomicAdd(&processed[wave], 1);
        if (borrowed) atomicAdd(&stolen[wave], 1);
      }
      __syncthreads();
    }
    grid.sync();  // All tiles from this wave finish before the next wave.
  }
  if (tensor_owned) {
    __syncthreads();
    if (warp == 0) {
      asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                   :: "r"(*tensor_address), "n"(64) : "memory");
      asm volatile(
          "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
          ::: "memory");
    }
    __syncthreads();
  }
}

struct Case {
  const char* name;
  int grid;
  int communication;
  int budget;
  int tasks[kWaves];
};

int check(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) return 0;
  std::fprintf(stderr, "%s: %s (%d)\n", operation,
               cudaGetErrorString(status), int(status));
  return 1;
}

bool write_case(const char* directory, const Case& scenario,
                const int* owners, const int* processed, const int* dispatched,
                const int* stolen, int permit_count) {
  char path[512];
  int length = std::snprintf(path, sizeof(path), "%s/%s.json",
                             directory, scenario.name);
  if (length <= 0 || length >= int(sizeof(path))) return false;
  FILE* file = std::fopen(path, "wx");
  if (!file) return false;
  std::fprintf(file, "{\"name\":\"%s\",\"grid_ctas\":%d,"
             "\"communication_ctas\":%d,\"steal_budget\":%d,"
             "\"tasks\":[", scenario.name, scenario.grid,
             scenario.communication, scenario.budget);
  for (int wave = 0; wave < kWaves; ++wave)
    std::fprintf(file, "%s%d", wave ? "," : "", scenario.tasks[wave]);
  std::fprintf(file, "],\"processed\":[");
  for (int wave = 0; wave < kWaves; ++wave)
    std::fprintf(file, "%s%d", wave ? "," : "", processed[wave]);
  std::fprintf(file, "],\"dispatched\":[");
  for (int wave = 0; wave < kWaves; ++wave)
    std::fprintf(file, "%s%d", wave ? "," : "", dispatched[wave]);
  std::fprintf(file, "],\"stolen\":[");
  for (int wave = 0; wave < kWaves; ++wave)
    std::fprintf(file, "%s%d", wave ? "," : "", stolen[wave]);
  std::fprintf(file, "],\"permit_count\":%d,\"owner_blocks\":[",
               permit_count);
  for (int wave = 0; wave < kWaves; ++wave) {
    std::fprintf(file, "%s[", wave ? "," : "");
    for (int task = 0; task < scenario.tasks[wave]; ++task)
      std::fprintf(file, "%s%d", task ? "," : "",
                   owners[wave * kMaxTasks + task]);
    std::fprintf(file, "]");
  }
  std::fprintf(file, "]}\n");
  return std::fclose(file) == 0;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  int devices = 0;
  if (check(cudaGetDeviceCount(&devices), "cudaGetDeviceCount")) return 3;
  if (devices != 1) return 4;
  int device = -1;
  if (check(cudaGetDevice(&device), "cudaGetDevice")) return 5;
  cudaDeviceProp prop{};
  if (check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties"))
    return 6;
  if (prop.major != 10 || prop.minor != 3 ||
      (std::strcmp(prop.name, "NVIDIA B300") != 0 &&
       std::strcmp(prop.name, "NVIDIA B300 SXM6 AC") != 0)) return 7;
  int cooperative = 0;
  if (check(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch,
                                   device), "cooperative launch")) return 8;
  if (cooperative != 1) return 9;
  if (check(cudaFuncSetAttribute(tile_schedule_probe,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 kDynamicShared), "dynamic shared attribute")) return 10;
  int active = 0;
  if (check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active, tile_schedule_probe, kThreads, kDynamicShared),
            "cooperative occupancy")) return 11;
  if (active < 1) return 12;
  const Case cases[] = {
      {"subcapacity_no_steal", 64, 1, 0, {0,17,15,32}},
      {"fullgrid_no_steal", 148, 1, 0, {0,17,15,32}},
      {"mixed_c74_budget8", 148, 74, 8, {0,17,15,32}},
      {"near_full_c147_budget32", 148, 147, 32, {0,17,15,32}},
      {"all_comm_c148_budget64", 148, 148, 64, {0,17,15,32}},
      {"terminal_only_c74_budget8", 148, 74, 8, {0,0,0,47}},
  };
  int *heads=nullptr, *owners=nullptr, *processed=nullptr, *dispatched=nullptr;
  int *stolen=nullptr, *permits=nullptr, *tasks=nullptr;
  if (check(cudaMalloc(&heads, kWaves*sizeof(int)), "heads malloc") ||
      check(cudaMalloc(&owners, kWaves*kMaxTasks*sizeof(int)), "owners malloc") ||
      check(cudaMalloc(&processed, kWaves*sizeof(int)), "processed malloc") ||
      check(cudaMalloc(&dispatched, kWaves*sizeof(int)), "dispatched malloc") ||
      check(cudaMalloc(&stolen, kWaves*sizeof(int)), "stolen malloc") ||
      check(cudaMalloc(&permits, sizeof(int)), "permits malloc") ||
      check(cudaMalloc(&tasks, kWaves*sizeof(int)), "tasks malloc")) return 13;
  for (const Case& scenario : cases) {
    if (scenario.grid > active * prop.multiProcessorCount ||
        scenario.communication > scenario.grid ||
        scenario.budget < 0 || scenario.budget > kMaxTasks) return 14;
    if (check(cudaMemset(heads, 0, kWaves*sizeof(int)), "heads reset") ||
        check(cudaMemset(owners, 0xff, kWaves*kMaxTasks*sizeof(int)),
              "owners reset") ||
        check(cudaMemset(processed, 0, kWaves*sizeof(int)),
              "processed reset") ||
        check(cudaMemset(dispatched, 0, kWaves*sizeof(int)),
              "dispatched reset") ||
        check(cudaMemset(stolen, 0, kWaves*sizeof(int)), "stolen reset") ||
        check(cudaMemset(permits, 0, sizeof(int)), "permits reset") ||
        check(cudaMemcpy(tasks, scenario.tasks, kWaves*sizeof(int),
                         cudaMemcpyHostToDevice), "task plan copy")) return 15;
    int communication = scenario.communication;
    int budget = scenario.budget;
    void* arguments[] = {&heads, &owners, &processed, &dispatched,
                         &stolen, &permits, &tasks, &communication, &budget};
    if (check(cudaLaunchCooperativeKernel(
                  reinterpret_cast<const void*>(tile_schedule_probe),
                  dim3(scenario.grid), dim3(kThreads), arguments,
                  kDynamicShared, nullptr), "cooperative tile launch") ||
        check(cudaDeviceSynchronize(), "cooperative tile completion")) return 16;
    int host_owners[kWaves*kMaxTasks];
    int host_processed[kWaves], host_dispatched[kWaves], host_stolen[kWaves];
    int host_permits = -1;
    if (check(cudaMemcpy(host_owners, owners, sizeof(host_owners),
                         cudaMemcpyDeviceToHost), "owners read") ||
        check(cudaMemcpy(host_processed, processed, sizeof(host_processed),
                         cudaMemcpyDeviceToHost), "processed read") ||
        check(cudaMemcpy(host_dispatched, dispatched, sizeof(host_dispatched),
                         cudaMemcpyDeviceToHost), "dispatched read") ||
        check(cudaMemcpy(host_stolen, stolen, sizeof(host_stolen),
                         cudaMemcpyDeviceToHost), "stolen read") ||
        check(cudaMemcpy(&host_permits, permits, sizeof(int),
                         cudaMemcpyDeviceToHost), "permits read")) return 17;
    int stolen_total = 0;
    for (int wave = 0; wave < kWaves; ++wave) {
      if (host_processed[wave] != scenario.tasks[wave] ||
          host_dispatched[wave] != scenario.communication) return 18;
      stolen_total += host_stolen[wave];
      for (int task = 0; task < scenario.tasks[wave]; ++task) {
        int owner = host_owners[wave*kMaxTasks + task];
        if (owner < 0 || owner >= scenario.grid) return 19;
      }
    }
    if (stolen_total != host_permits || stolen_total > scenario.budget)
      return 20;
    if (!write_case(argv[1], scenario, host_owners, host_processed,
                    host_dispatched, host_stolen, host_permits)) return 21;
    std::printf("%s: grid=%d c=%d budget=%d stolen=%d processed=%d\n",
                scenario.name, scenario.grid, scenario.communication,
                scenario.budget, stolen_total,
                host_processed[0]+host_processed[1]+
                host_processed[2]+host_processed[3]);
  }
  char summary_path[512];
  int length=std::snprintf(summary_path,sizeof(summary_path),
                           "%s/resource.json",argv[1]);
  if (length<=0 || length>=int(sizeof(summary_path))) return 22;
  FILE* summary=std::fopen(summary_path,"wx");
  if (!summary) return 23;
  std::fprintf(summary,
      "{\"target\":\"sm_103a\",\"device_name\":\"%s\","
      "\"sm_count\":%d,\"active_blocks_per_sm\":%d,"
      "\"cooperative_grid_cta_upper_bound\":%d,"
      "\"threads_per_cta\":%d,\"dynamic_shared_bytes\":%d}\n",
      prop.name,prop.multiProcessorCount,active,
      prop.multiProcessorCount*active,kThreads,kDynamicShared);
  std::fclose(summary);
  cudaFree(tasks); cudaFree(permits); cudaFree(stolen);
  cudaFree(dispatched); cudaFree(processed); cudaFree(owners); cudaFree(heads);
  return 0;
}
