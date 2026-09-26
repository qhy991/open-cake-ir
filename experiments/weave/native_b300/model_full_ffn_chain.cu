// B300 development probe for one live Cake up/gate -> activation -> down chain.
// One cooperative grid reserves the tensor-core worker's 192-thread/49,200-B
// CTA footprint. A CTA allocates TMEM at its first claimed tile and releases
// it once after all waves. The stage body below follows the retained Cake
// tensor-core emissions; activation uses the Cake FP32-to-BF16 graph.
// Grid barriers separate full waves of dependent stage work; global proxy
// fences separate activation writes from down's TMA reads. No P2P mailbox.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "down/kernel.cu"
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
constexpr int kStages = 3;
constexpr int kMaxTasks = 4096;
constexpr int kLogicalTiles = 64;
constexpr int kUpGateTasksPerTile = 24;
constexpr int kActivationTasksPerTile = 128;
constexpr int kDownTasksPerTile = 32;
constexpr int kTotalStageTasks = kLogicalTiles *
                                 (kUpGateTasksPerTile+kActivationTasksPerTile+
                                  kDownTasksPerTile);
constexpr int kRows = 128;
constexpr int kInputWidth = 2048;
constexpr int kHidden = 768;
constexpr int kOutput = 2048;
constexpr int kUpGateWidth = 2*kHidden;
constexpr int kInputBytes = kLogicalTiles * kRows * kInputWidth * int(sizeof(uint16_t));
constexpr int kUpGateWeightBytes = kUpGateWidth * kInputWidth * int(sizeof(uint16_t));
constexpr int kUpGateBytes = kLogicalTiles * kRows * kUpGateWidth * int(sizeof(float));
constexpr int kActivatedBytes = kLogicalTiles * kRows * kHidden * int(sizeof(uint16_t));
constexpr int kDownWeightBytes = kOutput * kHidden * int(sizeof(uint16_t));
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
      output[int(threadIdx.x) * kUpGateWidth + n_tile * 64 + col] = b6[col];
  }
  __syncthreads();
  if (threadIdx.x == 0) cake_inval(bar1);
  __syncthreads();
}

__device__ __forceinline__ void cake_activation_stage_work(
    const float* up_gate, __nv_bfloat16* activated,
    int logical_tile, int row) {
  if (threadIdx.x < 32) {
    int lane=int(threadIdx.x);
    const float* source=up_gate+(logical_tile*kRows+row)*(2*kHidden);
    __nv_bfloat16* destination=activated+
        (logical_tile*kRows+row)*kHidden;
    for (int feature=lane; feature<kHidden; feature+=32) {
      // CAKE_OP: activation.load_up/load_gate/neg_gate/exp_gate/denom/
      // silu/multiply/round_bf16/store.
      float up=source[feature];
      float gate=source[kHidden+feature];
      float negative=gate*-1.0f;
      float exponential=expf(negative);
      float denominator=exponential+1.0f;
      float silu=gate/denominator;
      float product=up*silu;
      destination[feature]=__float2bfloat16_rn(product);
    }
    // Normal global stores precede the following stage's async TMA reads.
    asm volatile("fence.proxy.async.global;" ::: "memory");
  }
  __syncthreads();
}

__device__ __forceinline__ void cake_down_stage_work(
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
    for (int it0=0; it0<12; ++it0) {
      const int stage = it0 % 2;
      if ((threadIdx.x & 31) == 0 && it0 >= 2)
        cake_wait(free0+stage, (it0/2-1)&1);
      __syncwarp();
      // CAKE_OP: down.load_a
      if ((threadIdx.x & 31) == 0) {
        cake_expect(bar0+stage, 16384);
        cake_tma2((smem + (stage)*16384), map0,
                  it0 * 64, 0, bar0+stage);
      }
      // CAKE_OP: down.load_b
      if ((threadIdx.x & 31) == 0) {
        cake_expect(bar0+stage, 8192);
        cake_tma2((smem + 32768 + (stage)*8192), map1,
                  it0 * 64, n_tile * 64, bar0+stage);
      }
    }
  }
  if (warp == 4 && (threadIdx.x & 31) == 0) {
    #pragma unroll 1
    for (int it0=0; it0<12; ++it0) {
      const int stage = it0 % 2;
      cake_wait(bar0+stage, (it0/2)&1);
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      // CAKE_OP: down.mma
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
  // CAKE_OP: down.read_acc
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
  // CAKE_OP: down.store
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
    float* up_gate, __nv_bfloat16* activated,
    const CUtensorMap* up_maps_a, const CUtensorMap* up_map_b,
    const CUtensorMap* down_maps_a, const CUtensorMap* down_map_b,
    float* outputs,
    int communication_ctas, int steal_budget) {
  extern __shared__ __align__(1024) unsigned char shared[];
  uint32_t* tensor_address = reinterpret_cast<uint32_t*>(shared + 49192);
  __shared__ int claimed;
  __shared__ int borrowed;
  const int block = int(blockIdx.x);
  const int warp = int(threadIdx.x) / 32;
  cg::grid_group grid = cg::this_grid();
  bool tensor_owned = false;
  int logical_offset = 0;
  for (int wave = 0; wave < kWaves; ++wave) {
    if (threadIdx.x == 0 && block < communication_ctas)
      atomicAdd(&dispatched[wave], 1);
    grid.sync();  // Every source-chunk dispatch precedes its tile claims.
    for (int stage=0; stage<kStages; ++stage) {
    const int work_index=stage*kWaves+wave;
    while (true) {
      if (threadIdx.x == 0) {
        bool comm = block < communication_ctas;
        bool permitted = !comm;
        bool reserved_permit = false;
        if (comm && task_counts[work_index] > 0 && steal_budget > 0) {
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
          int old = claim_old(&task_heads[work_index]);
          if (old < task_counts[work_index]) {
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
      int factor=stage==0 ? kUpGateTasksPerTile :
                 stage==1 ? kActivationTasksPerTile : kDownTasksPerTile;
      int logical_tile=logical_offset+claimed/factor;
      int subtile=claimed%factor;
      if (stage==0) {
        cake_upgate_stage_work(shared,tensor_address,warp,
                               up_gate+logical_tile*kRows*kUpGateWidth,
                               up_maps_a+logical_tile,up_map_b,subtile);
        // Each writer publishes its ordinary global stores before the next
        // stage's ordinary global loads on other CTAs.
        __threadfence();
      } else if (stage==1) {
        cake_activation_stage_work(up_gate,activated,logical_tile,subtile);
      } else {
        cake_down_stage_work(shared,tensor_address,warp,
                             outputs+logical_tile*kRows*kOutput,
                             down_maps_a+logical_tile,down_map_b,subtile);
      }
      if (threadIdx.x == 0) {
        task_owner[work_index * kMaxTasks + claimed] = block;
        atomicAdd(&processed[work_index], 1);
        if (borrowed) atomicAdd(&stolen[work_index], 1);
      }
      __syncthreads();
    }
    grid.sync();  // All predecessor stage CTAs finish before the successor.
    }
    logical_offset += task_counts[wave]/kUpGateTasksPerTile;
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
  int tasks[kStages][kWaves];
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

bool write_actual(const char* directory, const char* name, const char* stage,
                  const std::vector<unsigned char>& bytes) {
  char path[512];
  int length=std::snprintf(path,sizeof(path),"%s/%s.%s",
                           directory,name,stage);
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
                const int* stolen, int permit_count,
                int upgate_mismatches, int activation_mismatches,
                int down_mismatches) {
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
  for (int stage=0; stage<kStages; ++stage) {
    std::fprintf(file,"%s[",stage ? "," : "");
    for (int wave=0; wave<kWaves; ++wave)
      std::fprintf(file,"%s%d",wave ? "," : "",scenario.tasks[stage][wave]);
    std::fprintf(file,"]");
  }
  std::fprintf(file, "],\"processed\":[");
  for (int stage=0; stage<kStages; ++stage) {
    std::fprintf(file,"%s[",stage ? "," : "");
    for (int wave=0; wave<kWaves; ++wave)
      std::fprintf(file,"%s%d",wave ? "," : "",
                   processed[stage*kWaves+wave]);
    std::fprintf(file,"]");
  }
  std::fprintf(file, "],\"dispatched\":[");
  for (int wave = 0; wave < kWaves; ++wave)
    std::fprintf(file, "%s%d", wave ? "," : "", dispatched[wave]);
  std::fprintf(file, "],\"stolen\":[");
  for (int stage=0; stage<kStages; ++stage) {
    std::fprintf(file,"%s[",stage ? "," : "");
    for (int wave=0; wave<kWaves; ++wave)
      std::fprintf(file,"%s%d",wave ? "," : "",
                   stolen[stage*kWaves+wave]);
    std::fprintf(file,"]");
  }
  std::fprintf(file, "],\"permit_count\":%d,"
               "\"upgate_bit_mismatches\":%d,"
               "\"activation_bit_mismatches\":%d,"
               "\"down_bit_mismatches\":%d,"
               "\"upgate_bytes\":%d,\"activation_bytes\":%d,"
               "\"down_bytes\":%d,"
               "\"owner_blocks\":[",
               permit_count,upgate_mismatches,activation_mismatches,
               down_mismatches,kUpGateBytes,kActivatedBytes,kOutputBytes);
  for (int stage=0; stage<kStages; ++stage) {
    std::fprintf(file,"%s[",stage ? "," : "");
    for (int wave=0; wave<kWaves; ++wave) {
      std::fprintf(file,"%s[",wave ? "," : "");
      for (int task=0; task<scenario.tasks[stage][wave]; ++task)
        std::fprintf(file,"%s%d",task ? "," : "",
                     owners[(stage*kWaves+wave)*kMaxTasks+task]);
      std::fprintf(file,"]");
    }
    std::fprintf(file,"]");
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
      {"subcapacity_no_steal",64,1,0,
       {{0,408,360,768},{0,2176,1920,4096},{0,544,480,1024}}},
      {"fullgrid_no_steal",148,1,0,
       {{0,408,360,768},{0,2176,1920,4096},{0,544,480,1024}}},
      {"near_full_c147_budget5888",148,147,5888,
       {{0,408,360,768},{0,2176,1920,4096},{0,544,480,1024}}},
      {"all_comm_c148_budget11776",148,148,11776,
       {{0,408,360,768},{0,2176,1920,4096},{0,544,480,1024}}},
  };
  std::vector<unsigned char> input(kInputBytes),
                             up_weight(kUpGateWeightBytes),
                             expected_upgate(kUpGateBytes),
                             actual_upgate(kUpGateBytes),
                             expected_activation(kActivatedBytes),
                             actual_activation(kActivatedBytes),
                             down_weight(kDownWeightBytes),
                             expected_down(kOutputBytes),
                             actual_down(kOutputBytes);
  if (!read_exact(argv[1],"x_tiles.bf16",input)
      || !read_exact(argv[1],"w_up_gate.bf16",up_weight)
      || !read_exact(argv[1],"expected_up_gate.fp32",expected_upgate)
      || !read_exact(argv[1],"expected_activated.bf16",expected_activation)
      || !read_exact(argv[1],"w_down.bf16",down_weight)
      || !read_exact(argv[1],"expected_down.fp32",expected_down)) return 13;
  float* device_up_gate=nullptr;
  __nv_bfloat16 *device_input=nullptr, *device_up_weight=nullptr;
  __nv_bfloat16 *device_activated=nullptr, *device_down_weight=nullptr;
  float* device_output=nullptr;
  CUtensorMap *device_up_maps_a=nullptr, *device_up_map_b=nullptr;
  CUtensorMap *device_down_maps_a=nullptr, *device_down_map_b=nullptr;
  alignas(128) CUtensorMap host_up_maps_a[kLogicalTiles];
  alignas(128) CUtensorMap host_up_map_b;
  alignas(128) CUtensorMap host_down_maps_a[kLogicalTiles];
  alignas(128) CUtensorMap host_down_map_b;
  if (check(cudaMalloc(&device_input,kInputBytes),"input malloc") ||
      check(cudaMalloc(&device_up_weight,kUpGateWeightBytes),
            "up/gate weight malloc") ||
      check(cudaMalloc(&device_up_gate,kUpGateBytes),"up/gate malloc") ||
      check(cudaMalloc(&device_activated,kActivatedBytes),
            "activated malloc") ||
      check(cudaMalloc(&device_down_weight,kDownWeightBytes),
            "down weight malloc") ||
      check(cudaMalloc(&device_output,kOutputBytes),"output malloc") ||
      check(cudaMalloc(&device_up_maps_a,sizeof(host_up_maps_a)),
            "up A maps malloc") ||
      check(cudaMalloc(&device_up_map_b,sizeof(host_up_map_b)),
            "up B map malloc") ||
      check(cudaMalloc(&device_down_maps_a,sizeof(host_down_maps_a)),
            "down A maps malloc") ||
      check(cudaMalloc(&device_down_map_b,sizeof(host_down_map_b)),
            "down B map malloc") ||
      check(cudaMemcpy(device_input,input.data(),kInputBytes,
                       cudaMemcpyHostToDevice),"input H2D") ||
      check(cudaMemcpy(device_up_weight,up_weight.data(),kUpGateWeightBytes,
                       cudaMemcpyHostToDevice),"up weight H2D") ||
      check(cudaMemcpy(device_down_weight,down_weight.data(),kDownWeightBytes,
                       cudaMemcpyHostToDevice),"down weight H2D")) return 14;
  static_assert(sizeof(CUtensorMap)%64==0,"tensor map array alignment");
  cuuint64_t up_dims_a[2]={kInputWidth,kRows};
  cuuint64_t up_dims_b[2]={kInputWidth,kUpGateWidth};
  cuuint64_t up_strides[1]={kInputWidth*2};
  cuuint64_t down_dims_a[2]={kHidden,kRows};
  cuuint64_t down_dims_b[2]={kHidden,kOutput};
  cuuint64_t down_strides[1]={kHidden*2};
  cuuint32_t box_a[2]={64,kRows}, box_b[2]={64,64};
  cuuint32_t steps[2]={1,1};
  for (int tile=0; tile<kLogicalTiles; ++tile) {
    CUresult up_status=cuTensorMapEncodeTiled(
        &host_up_maps_a[tile],CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,
        device_input+tile*kRows*kInputWidth,up_dims_a,up_strides,box_a,steps,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUresult down_status=cuTensorMapEncodeTiled(
        &host_down_maps_a[tile],CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,
        device_activated+tile*kRows*kHidden,down_dims_a,down_strides,box_a,steps,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (up_status!=CUDA_SUCCESS || down_status!=CUDA_SUCCESS) {
      std::fprintf(stderr,"A tensor maps %d failed: %d/%d\n",tile,
                   int(up_status),int(down_status));
      return 15;
    }
  }
  CUresult up_status=cuTensorMapEncodeTiled(
      &host_up_map_b,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,
      device_up_weight,up_dims_b,up_strides,box_b,steps,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  CUresult down_status=cuTensorMapEncodeTiled(
      &host_down_map_b,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,
      device_down_weight,down_dims_b,down_strides,box_b,steps,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (up_status!=CUDA_SUCCESS || down_status!=CUDA_SUCCESS) {
    std::fprintf(stderr,"B tensor maps failed: %d/%d\n",
                 int(up_status),int(down_status));
    return 16;
  }
  if (check(cudaMemcpy(device_up_maps_a,host_up_maps_a,sizeof(host_up_maps_a),
                       cudaMemcpyHostToDevice),"up A maps H2D") ||
      check(cudaMemcpy(device_up_map_b,&host_up_map_b,sizeof(host_up_map_b),
                       cudaMemcpyHostToDevice),"up B map H2D") ||
      check(cudaMemcpy(device_down_maps_a,host_down_maps_a,
                       sizeof(host_down_maps_a),cudaMemcpyHostToDevice),
            "down A maps H2D") ||
      check(cudaMemcpy(device_down_map_b,&host_down_map_b,
                       sizeof(host_down_map_b),cudaMemcpyHostToDevice),
            "down B map H2D")) return 17;
  int *heads=nullptr, *owners=nullptr, *processed=nullptr, *dispatched=nullptr;
  int *stolen=nullptr, *permits=nullptr, *tasks=nullptr;
  if (check(cudaMalloc(&heads, kStages*kWaves*sizeof(int)), "heads malloc") ||
      check(cudaMalloc(&owners, kStages*kWaves*kMaxTasks*sizeof(int)),
            "owners malloc") ||
      check(cudaMalloc(&processed, kStages*kWaves*sizeof(int)),
            "processed malloc") ||
      check(cudaMalloc(&dispatched, kWaves*sizeof(int)), "dispatched malloc") ||
      check(cudaMalloc(&stolen, kStages*kWaves*sizeof(int)), "stolen malloc") ||
      check(cudaMalloc(&permits, sizeof(int)), "permits malloc") ||
      check(cudaMalloc(&tasks, kStages*kWaves*sizeof(int)),
            "tasks malloc")) return 18;
  for (const Case& scenario : cases) {
    if (scenario.grid > active * prop.multiProcessorCount ||
        scenario.communication > scenario.grid ||
        scenario.budget < 0 || scenario.budget > kTotalStageTasks) return 19;
    if (check(cudaMemset(heads, 0, kStages*kWaves*sizeof(int)), "heads reset") ||
        check(cudaMemset(owners, 0xff,
                         kStages*kWaves*kMaxTasks*sizeof(int)),
              "owners reset") ||
        check(cudaMemset(processed, 0, kStages*kWaves*sizeof(int)),
              "processed reset") ||
        check(cudaMemset(dispatched, 0, kWaves*sizeof(int)),
              "dispatched reset") ||
        check(cudaMemset(stolen, 0, kStages*kWaves*sizeof(int)),
              "stolen reset") ||
        check(cudaMemset(permits, 0, sizeof(int)), "permits reset") ||
        check(cudaMemset(device_up_gate,0xff,kUpGateBytes),
              "up/gate sentinel") ||
        check(cudaMemset(device_activated,0xff,kActivatedBytes),
              "activated sentinel") ||
        check(cudaMemset(device_output,0xff,kOutputBytes),"output sentinel") ||
        check(cudaMemcpy(tasks, scenario.tasks, sizeof(scenario.tasks),
                         cudaMemcpyHostToDevice), "task plan copy")) return 20;
    int communication = scenario.communication;
    int budget = scenario.budget;
    void* arguments[] = {&heads, &owners, &processed, &dispatched,
                         &stolen, &permits, &tasks,
                         &device_up_gate, &device_activated,
                         &device_up_maps_a, &device_up_map_b,
                         &device_down_maps_a, &device_down_map_b,
                         &device_output,
                         &communication, &budget};
    if (check(cudaLaunchCooperativeKernel(
                  reinterpret_cast<const void*>(tile_schedule_probe),
                  dim3(scenario.grid), dim3(kThreads), arguments,
                  kDynamicShared, nullptr), "cooperative tile launch") ||
        check(cudaDeviceSynchronize(), "cooperative tile completion")) return 21;
    int host_owners[kStages*kWaves*kMaxTasks];
    int host_processed[kStages*kWaves], host_dispatched[kWaves];
    int host_stolen[kStages*kWaves];
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
        check(cudaMemcpy(actual_upgate.data(),device_up_gate,
                         kUpGateBytes,cudaMemcpyDeviceToHost),
              "up/gate read") ||
        check(cudaMemcpy(actual_activation.data(),device_activated,
                         kActivatedBytes,cudaMemcpyDeviceToHost),
              "activated read") ||
        check(cudaMemcpy(actual_down.data(),device_output,kOutputBytes,
                         cudaMemcpyDeviceToHost),"down output read")) return 22;
    int stolen_total = 0;
    for (int wave = 0; wave < kWaves; ++wave) {
      if (host_dispatched[wave] != scenario.communication) return 23;
      for (int stage=0; stage<kStages; ++stage) {
        int index=stage*kWaves+wave;
        if (host_processed[index] != scenario.tasks[stage][wave]) return 23;
        stolen_total += host_stolen[index];
        for (int task=0; task<scenario.tasks[stage][wave]; ++task) {
          int owner=host_owners[index*kMaxTasks+task];
          if (owner<0 || owner>=scenario.grid) return 24;
        }
      }
    }
    if (stolen_total != host_permits || stolen_total > scenario.budget)
      return 25;
    int upgate_mismatches=0, activation_mismatches=0, down_mismatches=0;
    for (int index=0; index<kUpGateBytes; index+=4)
      upgate_mismatches += actual_upgate[index]!=expected_upgate[index]
                          || actual_upgate[index+1]!=expected_upgate[index+1]
                          || actual_upgate[index+2]!=expected_upgate[index+2]
                          || actual_upgate[index+3]!=expected_upgate[index+3];
    for (int index=0; index<kActivatedBytes; index+=2)
      activation_mismatches +=
          actual_activation[index]!=expected_activation[index]
          || actual_activation[index+1]!=expected_activation[index+1];
    for (int index=0; index<kOutputBytes; index+=4)
      down_mismatches += actual_down[index]!=expected_down[index]
                         || actual_down[index+1]!=expected_down[index+1]
                         || actual_down[index+2]!=expected_down[index+2]
                         || actual_down[index+3]!=expected_down[index+3];
    if (!write_actual(argv[1],scenario.name,"upgate.actual.fp32",
                      actual_upgate)
        || !write_actual(argv[1],scenario.name,"activated.actual.bf16",
                      actual_activation)
        || !write_actual(argv[1],scenario.name,"down.actual.fp32",
                         actual_down)) return 26;
    if (!write_case(argv[1], scenario, host_owners, host_processed,
                    host_dispatched, host_stolen, host_permits,
                    upgate_mismatches,activation_mismatches,
                    down_mismatches)) return 27;
    if (upgate_mismatches || activation_mismatches || down_mismatches)
      return 28;
    std::printf("%s: grid=%d c=%d budget=%d stolen=%d processed=%d\n",
                scenario.name, scenario.grid, scenario.communication,
                scenario.budget, stolen_total,
                host_processed[0]+host_processed[1]+
                host_processed[2]+host_processed[3]+
                host_processed[4]+host_processed[5]+
                host_processed[6]+host_processed[7]+
                host_processed[8]+host_processed[9]+
                host_processed[10]+host_processed[11]);
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
  cudaFree(device_down_map_b); cudaFree(device_down_maps_a);
  cudaFree(device_up_map_b); cudaFree(device_up_maps_a);
  cudaFree(device_output); cudaFree(device_down_weight);
  cudaFree(device_activated); cudaFree(device_up_gate);
  cudaFree(device_up_weight); cudaFree(device_input);
  return 0;
}
