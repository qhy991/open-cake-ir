// B300 development probe for real Cake up/gate TMA/MMA CTA work units.
// One cooperative grid reserves the tensor-core worker's 192-thread/49,200-B
// CTA footprint. A CTA allocates TMEM at its first claimed tile and releases
// it once after all waves. The stage body below follows the retained Cake
// up/gate emission; only its row/column task coordinates and global tensor-map
// pointers are dynamic. There is no cross-device mailbox in this probe.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "up_gate/kernel.cu"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 1030
#error "Exact sm_103a is required"
#endif

namespace cg = cooperative_groups;
namespace {
constexpr int kThreads = 192;
constexpr int kDynamicShared = 49200;
constexpr int kWaves = 4;
constexpr int kMaxTasks = 768;
constexpr int kLogicalTiles = 64;
constexpr int kStageTasksPerTile = 24;
constexpr int kTotalStageTasks = kLogicalTiles * kStageTasksPerTile;
constexpr int kRows = 128;
constexpr int kHidden = 2048;
constexpr int kOutput = 1536;
constexpr int kInputBytes = kLogicalTiles * kRows * kHidden * int(sizeof(uint16_t));
constexpr int kWeightBytes = kOutput * kHidden * int(sizeof(uint16_t));
constexpr int kOutputBytes = kLogicalTiles * kRows * kOutput * int(sizeof(float));

__device__ __forceinline__ uint32_t smem_address(const void* pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}
__device__ __forceinline__ int claim_old(int* pointer) {
  int old;
  asm volatile("atom.relaxed.gpu.global.add.s32 %0, [%1], %2;"
               : "=r"(old) : "l"(pointer), "r"(1) : "memory");
  return old;
}

__device__ __forceinline__ void cake_upgate_stage_work(
    unsigned char* smem, uint32_t* tm1, int warp, float* output,
    const CUtensorMap* map0, const CUtensorMap* map1, int n_tile) {
  uint64_t* bar0 = reinterpret_cast<uint64_t*>(smem + 49152);
  uint64_t* bar1 = reinterpret_cast<uint64_t*>(smem + 49168);
  uint64_t* free0 = reinterpret_cast<uint64_t*>(smem + 49176);
  float b6[64];
  if (threadIdx.x == 0) {
    for (int stage=0; stage<2; ++stage) {
      cake_init(bar0+stage, 2); cake_init(free0+stage, 1);
    }
    cake_init(bar1, 1);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  if (warp == 5) {
    #pragma unroll 1
    for (int it0=0; it0<32; ++it0) {
      const int stage = it0 % 2;
      if ((threadIdx.x & 31) == 0 && it0 >= 2)
        cake_wait(free0+stage, (it0/2-1)&1);
      __syncwarp();
      // CAKE_OP: up_gate.load_a
      if ((threadIdx.x & 31) == 0) {
        cake_expect(bar0+stage, 16384);
        cake_tma2((smem + (stage)*16384), map0,
                  it0 * 64, 0, bar0+stage);
      }
      // CAKE_OP: up_gate.load_b
      if ((threadIdx.x & 31) == 0) {
        cake_expect(bar0+stage, 8192);
        cake_tma2((smem + 32768 + (stage)*8192), map1,
                  it0 * 64, n_tile * 64, bar0+stage);
      }
    }
  }
  if (warp == 4 && (threadIdx.x & 31) == 0) {
    #pragma unroll 1
    for (int it0=0; it0<32; ++it0) {
      const int stage = it0 % 2;
      cake_wait(bar0+stage, (it0/2)&1);
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      // CAKE_OP: up_gate.mma
      #pragma unroll
      for (int atom=0; atom<4; ++atom) {
        cake_mma((*tm1),
                 cake_desc(cake_smem(smem + (stage)*16384) + atom*32,1024,2),
                 cake_desc(cake_smem(smem + 32768 + (stage)*8192)
                           + atom*32,1024,2),
                 135267472u,it0 != 0 || atom != 0);
      }
      cake_commit(free0+stage);
    }
    cake_commit(bar1);
    cake_wait(free0+0,1);
    cake_wait(free0+1,1);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int stage=0; stage<2; ++stage) {
      cake_inval(bar0+stage); cake_inval(free0+stage);
    }
  }
  // CAKE_OP: up_gate.read_acc
  if (warp <= 3) {
    cake_wait(bar1,0);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    #pragma unroll
    for (int col=0; col<64; col+=16) {
      asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                   "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                   : "=f"(b6[col+0]), "=f"(b6[col+1]), "=f"(b6[col+2]),
                     "=f"(b6[col+3]), "=f"(b6[col+4]), "=f"(b6[col+5]),
                     "=f"(b6[col+6]), "=f"(b6[col+7]), "=f"(b6[col+8]),
                     "=f"(b6[col+9]), "=f"(b6[col+10]), "=f"(b6[col+11]),
                     "=f"(b6[col+12]), "=f"(b6[col+13]), "=f"(b6[col+14]),
                     "=f"(b6[col+15])
                   : "r"((*tm1) + ((int(threadIdx.x)/32)*32 << 16) + col)
                   : "memory");
      asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    }
  }
  // CAKE_OP: up_gate.store
  if (warp <= 3) {
    #pragma unroll
    for (int col=0; col<64; ++col)
      output[int(threadIdx.x) * kOutput + n_tile * 64 + col] = b6[col];
  }
  __syncthreads();
  if (threadIdx.x == 0) cake_inval(bar1);
  __syncthreads();
}

__global__ void tile_schedule_probe(
    int* task_heads, int* task_owner, int* processed, int* dispatched,
    int* stolen, int* steal_permits, const int* task_counts,
    const CUtensorMap* maps_a, const CUtensorMap* map_b, float* outputs,
    int communication_ctas, int steal_budget) {
  extern __shared__ __align__(1024) unsigned char shared[];
  uint32_t* tensor_address = reinterpret_cast<uint32_t*>(shared + 49192);
  __shared__ int claimed;
  __shared__ int borrowed;
  const int block = int(blockIdx.x);
  const int warp = int(threadIdx.x) / 32;
  cg::grid_group grid = cg::this_grid();
  bool tensor_owned = false;
  int wave_offset = 0;
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
      int global_task = wave_offset + claimed;
      int logical_tile = global_task / kStageTasksPerTile;
      int n_tile = global_task % kStageTasksPerTile;
      cake_upgate_stage_work(shared,tensor_address,warp,
                             outputs + logical_tile*kRows*kOutput,
                             maps_a + logical_tile,map_b,n_tile);
      if (threadIdx.x == 0) {
        task_owner[wave * kMaxTasks + claimed] = block;
        atomicAdd(&processed[wave], 1);
        if (borrowed) atomicAdd(&stolen[wave], 1);
      }
      __syncthreads();
    }
    grid.sync();  // All tiles from this wave finish before the next wave.
    wave_offset += task_counts[wave];
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

bool read_exact(const char* directory, const char* name,
                std::vector<unsigned char>& bytes) {
  char path[512];
  int length=std::snprintf(path,sizeof(path),"%s/%s",directory,name);
  if (length<=0 || length>=int(sizeof(path))) return false;
  FILE* file=std::fopen(path,"rb");
  if (!file) return false;
  bool okay=std::fread(bytes.data(),1,bytes.size(),file)==bytes.size()
            && std::fgetc(file)==EOF;
  return std::fclose(file)==0 && okay;
}

bool write_actual(const char* directory, const char* name,
                  const std::vector<unsigned char>& bytes) {
  char path[512];
  int length=std::snprintf(path,sizeof(path),"%s/%s.actual.fp32",
                           directory,name);
  if (length<=0 || length>=int(sizeof(path))) return false;
  FILE* file=std::fopen(path,"wbx");
  if (!file) return false;
  bool okay=std::fwrite(bytes.data(),1,bytes.size(),file)==bytes.size();
  return std::fclose(file)==0 && okay;
}

int check(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) return 0;
  std::fprintf(stderr, "%s: %s (%d)\n", operation,
               cudaGetErrorString(status), int(status));
  return 1;
}

bool write_case(const char* directory, const Case& scenario,
                const int* owners, const int* processed, const int* dispatched,
                const int* stolen, int permit_count, int math_mismatches) {
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
  std::fprintf(file, "],\"permit_count\":%d,"
               "\"math_bit_mismatches\":%d,\"output_bytes\":%d,"
               "\"owner_blocks\":[",
               permit_count,math_mismatches,kOutputBytes);
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
      {"subcapacity_no_steal",64,1,0,{0,408,360,768}},
      {"fullgrid_no_steal",148,1,0,{0,408,360,768}},
      {"near_full_c147_budget768",148,147,768,{0,408,360,768}},
      {"all_comm_c148_budget1536",148,148,1536,{0,408,360,768}},
  };
  std::vector<unsigned char> input(kInputBytes), weight(kWeightBytes),
                             expected(kOutputBytes), actual(kOutputBytes);
  if (!read_exact(argv[1],"x_tiles.bf16",input)
      || !read_exact(argv[1],"w_up_gate.bf16",weight)
      || !read_exact(argv[1],"expected_up_gate.fp32",expected)) return 13;
  __nv_bfloat16 *device_input=nullptr, *device_weight=nullptr;
  float* device_output=nullptr;
  CUtensorMap *device_maps_a=nullptr, *device_map_b=nullptr;
  alignas(128) CUtensorMap host_maps_a[kLogicalTiles];
  alignas(128) CUtensorMap host_map_b;
  if (check(cudaMalloc(&device_input,kInputBytes),"input malloc") ||
      check(cudaMalloc(&device_weight,kWeightBytes),"weight malloc") ||
      check(cudaMalloc(&device_output,kOutputBytes),"output malloc") ||
      check(cudaMalloc(&device_maps_a,sizeof(host_maps_a)),"A maps malloc") ||
      check(cudaMalloc(&device_map_b,sizeof(host_map_b)),"B map malloc") ||
      check(cudaMemcpy(device_input,input.data(),kInputBytes,
                       cudaMemcpyHostToDevice),"input H2D") ||
      check(cudaMemcpy(device_weight,weight.data(),kWeightBytes,
                       cudaMemcpyHostToDevice),"weight H2D")) return 14;
  static_assert(sizeof(CUtensorMap)%64==0,"tensor map array alignment");
  cuuint64_t dims_a[2]={kHidden,kRows};
  cuuint64_t dims_b[2]={kHidden,kOutput};
  cuuint64_t strides[1]={kHidden*2};
  cuuint32_t box_a[2]={64,kRows}, box_b[2]={64,64};
  cuuint32_t steps[2]={1,1};
  for (int tile=0; tile<kLogicalTiles; ++tile) {
    CUresult status=cuTensorMapEncodeTiled(
        &host_maps_a[tile],CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,
        device_input+tile*kRows*kHidden,dims_a,strides,box_a,steps,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status!=CUDA_SUCCESS) {
      std::fprintf(stderr,"A tensor map %d failed: %d\n",tile,int(status));
      return 15;
    }
  }
  CUresult status=cuTensorMapEncodeTiled(
      &host_map_b,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,
      device_weight,dims_b,strides,box_b,steps,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (status!=CUDA_SUCCESS) {
    std::fprintf(stderr,"B tensor map failed: %d\n",int(status));
    return 16;
  }
  if (check(cudaMemcpy(device_maps_a,host_maps_a,sizeof(host_maps_a),
                       cudaMemcpyHostToDevice),"A maps H2D") ||
      check(cudaMemcpy(device_map_b,&host_map_b,sizeof(host_map_b),
                       cudaMemcpyHostToDevice),"B map H2D")) return 17;
  int *heads=nullptr, *owners=nullptr, *processed=nullptr, *dispatched=nullptr;
  int *stolen=nullptr, *permits=nullptr, *tasks=nullptr;
  if (check(cudaMalloc(&heads, kWaves*sizeof(int)), "heads malloc") ||
      check(cudaMalloc(&owners, kWaves*kMaxTasks*sizeof(int)), "owners malloc") ||
      check(cudaMalloc(&processed, kWaves*sizeof(int)), "processed malloc") ||
      check(cudaMalloc(&dispatched, kWaves*sizeof(int)), "dispatched malloc") ||
      check(cudaMalloc(&stolen, kWaves*sizeof(int)), "stolen malloc") ||
      check(cudaMalloc(&permits, sizeof(int)), "permits malloc") ||
      check(cudaMalloc(&tasks, kWaves*sizeof(int)), "tasks malloc")) return 18;
  for (const Case& scenario : cases) {
    if (scenario.grid > active * prop.multiProcessorCount ||
        scenario.communication > scenario.grid ||
        scenario.budget < 0 || scenario.budget > kTotalStageTasks) return 19;
    if (check(cudaMemset(heads, 0, kWaves*sizeof(int)), "heads reset") ||
        check(cudaMemset(owners, 0xff, kWaves*kMaxTasks*sizeof(int)),
              "owners reset") ||
        check(cudaMemset(processed, 0, kWaves*sizeof(int)),
              "processed reset") ||
        check(cudaMemset(dispatched, 0, kWaves*sizeof(int)),
              "dispatched reset") ||
        check(cudaMemset(stolen, 0, kWaves*sizeof(int)), "stolen reset") ||
        check(cudaMemset(permits, 0, sizeof(int)), "permits reset") ||
        check(cudaMemset(device_output,0xff,kOutputBytes),"output sentinel") ||
        check(cudaMemcpy(tasks, scenario.tasks, kWaves*sizeof(int),
                         cudaMemcpyHostToDevice), "task plan copy")) return 20;
    int communication = scenario.communication;
    int budget = scenario.budget;
    void* arguments[] = {&heads, &owners, &processed, &dispatched,
                         &stolen, &permits, &tasks,
                         &device_maps_a, &device_map_b, &device_output,
                         &communication, &budget};
    if (check(cudaLaunchCooperativeKernel(
                  reinterpret_cast<const void*>(tile_schedule_probe),
                  dim3(scenario.grid), dim3(kThreads), arguments,
                  kDynamicShared, nullptr), "cooperative tile launch") ||
        check(cudaDeviceSynchronize(), "cooperative tile completion")) return 21;
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
                         cudaMemcpyDeviceToHost), "permits read") ||
        check(cudaMemcpy(actual.data(),device_output,kOutputBytes,
                         cudaMemcpyDeviceToHost),"up/gate output read")) return 22;
    int stolen_total = 0;
    for (int wave = 0; wave < kWaves; ++wave) {
      if (host_processed[wave] != scenario.tasks[wave] ||
          host_dispatched[wave] != scenario.communication) return 23;
      stolen_total += host_stolen[wave];
      for (int task = 0; task < scenario.tasks[wave]; ++task) {
        int owner = host_owners[wave*kMaxTasks + task];
        if (owner < 0 || owner >= scenario.grid) return 24;
      }
    }
    if (stolen_total != host_permits || stolen_total > scenario.budget)
      return 25;
    int mismatches=0;
    for (int index=0; index<kOutputBytes; index+=4)
      mismatches += actual[index]!=expected[index]
                    || actual[index+1]!=expected[index+1]
                    || actual[index+2]!=expected[index+2]
                    || actual[index+3]!=expected[index+3];
    if (!write_actual(argv[1],scenario.name,actual)) return 26;
    if (!write_case(argv[1], scenario, host_owners, host_processed,
                    host_dispatched, host_stolen, host_permits,
                    mismatches)) return 27;
    if (mismatches) return 28;
    std::printf("%s: grid=%d c=%d budget=%d stolen=%d processed=%d\n",
                scenario.name, scenario.grid, scenario.communication,
                scenario.budget, stolen_total,
                host_processed[0]+host_processed[1]+
                host_processed[2]+host_processed[3]);
  }
  char summary_path[512];
  int length=std::snprintf(summary_path,sizeof(summary_path),
                           "%s/resource.json",argv[1]);
  if (length<=0 || length>=int(sizeof(summary_path))) return 29;
  FILE* summary=std::fopen(summary_path,"wx");
  if (!summary) return 30;
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
  cudaFree(device_map_b); cudaFree(device_maps_a);
  cudaFree(device_output); cudaFree(device_weight); cudaFree(device_input);
  return 0;
}
