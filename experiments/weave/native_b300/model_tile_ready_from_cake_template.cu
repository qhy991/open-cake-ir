// B300 development probe for tile-ready Cake up/gate -> activation -> down.
// One cooperative grid reserves the tensor-core worker's 192-thread/49,200-B
// CTA footprint. A CTA allocates TMEM at its first claimed tile and releases
// it once after all waves. The stage body below follows the retained Cake
// tensor-core emissions; activation uses the Cake FP32-to-BF16 graph.
// Per-tile completion counters publish successors with GPU-scope release/
// acquire. Grid barriers only delimit source waves. No P2P mailbox.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
@CAKE_HELPERS@
#include <cmath>
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
__device__ __forceinline__ bool reserve_bounded(int* pointer, int limit) {
  int old=atomicAdd(pointer,0);
  while (old<limit) {
    int observed=atomicCAS(pointer,old,old+1);
    if (observed==old) return true;
    old=observed;
  }
  return false;
}
__device__ __forceinline__ int completion_acquire(const int* pointer) {
  int value;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];"
               : "=r"(value) : "l"(pointer) : "memory");
  return value;
}
__device__ __forceinline__ int completion_release_increment(int* pointer) {
  int old;
  asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;"
               : "=r"(old) : "l"(pointer), "r"(1) : "memory");
  return old;
}

@UPGATE_STAGE@
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

@DOWN_STAGE@
__global__ void tile_schedule_probe(
    int* task_heads, int* tile_completed, int* task_owner,
    int* processed, int* dispatched, int* stolen, int* steal_permits,
    const int* task_counts, int* overlap,
    float* up_gate, __nv_bfloat16* activated,
    const CUtensorMap* up_maps_a, const CUtensorMap* up_map_b,
    const CUtensorMap* down_maps_a, const CUtensorMap* down_map_b,
    float* outputs, int communication_ctas, int steal_budget) {
  extern __shared__ __align__(1024) unsigned char shared[];
  uint32_t* tensor_address = reinterpret_cast<uint32_t*>(shared + 49192);
  __shared__ int claimed, claimed_stage, claimed_tile, claimed_subtile;
  __shared__ int borrowed;
  const int block = int(blockIdx.x);
  const int warp = int(threadIdx.x) / 32;
  cg::grid_group grid = cg::this_grid();
  bool tensor_owned = false;
  int logical_offset = 0;
  for (int wave=0; wave<kWaves; ++wave) {
    const int tile_count=task_counts[wave]/kUpGateTasksPerTile;
    const int total_tasks=task_counts[wave]+task_counts[kWaves+wave]+
                          task_counts[2*kWaves+wave];
    if (threadIdx.x==0 && block<communication_ctas)
      atomicAdd(&dispatched[wave],1);
    grid.sync();  // Source wave becomes visible before any tile claim.
    while (true) {
      if (threadIdx.x==0) {
        const bool comm=block<communication_ctas;
        bool permitted=!comm;
        bool reserved=false;
        if (comm && total_tasks>0 && steal_budget>0) {
          reserved=reserve_bounded(steal_permits,steal_budget);
          permitted=reserved;
        }
        claimed=-1; borrowed=0;
        if (permitted) {
          // Prefer a ready successor. Each tile owns its own stage heads;
          // another tile may still be computing a predecessor stage.
          for (int stage=kStages-1; stage>=0 && claimed<0; --stage) {
            const int units=stage==0 ? kUpGateTasksPerTile :
                            stage==1 ? kActivationTasksPerTile :
                                       kDownTasksPerTile;
            const int predecessor=stage==1 ? kUpGateTasksPerTile :
                                  kActivationTasksPerTile;
            for (int attempt=0; attempt<tile_count && claimed<0; ++attempt) {
              const int tile=logical_offset+(block+attempt)%tile_count;
              if (stage>0 &&
                  completion_acquire(&tile_completed[(stage-1)*kLogicalTiles+tile])
                    !=predecessor) continue;
              int* head=&task_heads[stage*kLogicalTiles+tile];
              int old=atomicAdd(head,0);
              while (old<units) {
                int observed=atomicCAS(head,old,old+1);
                if (observed==old) {
                  claimed=(tile-logical_offset)*units+old;
                  claimed_stage=stage;
                  claimed_tile=tile;
                  claimed_subtile=old;
                  borrowed=int(comm);
                  break;
                }
                old=observed;
              }
            }
          }
        }
        if (claimed<0 && reserved) atomicSub(steal_permits,1);
        if (claimed>=0 && claimed_stage>0) {
          const int preceding=(claimed_stage-1)*kWaves+wave;
          if (atomicAdd(&processed[preceding],0)<task_counts[preceding])
            atomicExch(&overlap[claimed_stage-1],1);
        }
      }
      __syncthreads();
      if (claimed<0) {
        if (threadIdx.x==0) {
          int completed_total=0;
          for (int stage=0;stage<kStages;++stage)
            completed_total+=atomicAdd(&processed[stage*kWaves+wave],0);
          claimed=(completed_total==total_tasks) ? -2 : -1;
        }
        __syncthreads();
        if (claimed==-2) break;
        __nanosleep(128);
        continue;
      }
      if (!tensor_owned) {
        if (warp==0) {
          asm volatile(
              "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
              :: "r"(smem_address(tensor_address)), "n"(64) : "memory");
        }
        __syncthreads();
        tensor_owned=true;
      }
      const int stage=claimed_stage;
      const int tile=claimed_tile;
      const int subtile=claimed_subtile;
      if (stage==0) {
        cake_upgate_stage_work(shared,tensor_address,warp,
                               up_gate+tile*kRows*kUpGateWidth,
                               up_maps_a+tile,up_map_b,subtile);
      } else if (stage==1) {
        cake_activation_stage_work(up_gate,activated,tile,subtile);
      } else {
        cake_down_stage_work(shared,tensor_address,warp,
                             outputs+tile*kRows*kOutput,
                             down_maps_a+tile,down_map_b,subtile);
      }
      // The completion RMW represents every writer in this CTA. Activation
      // writers also issue an async-proxy fence before down's TMA read.
      __threadfence();
      __syncthreads();
      if (threadIdx.x==0) {
        const int work_index=stage*kWaves+wave;
        task_owner[work_index*kMaxTasks+claimed]=block;
        completion_release_increment(
            &tile_completed[stage*kLogicalTiles+tile]);
        atomicAdd(&processed[work_index],1);
        if (borrowed) atomicAdd(&stolen[work_index],1);
      }
      __syncthreads();
    }
    grid.sync();  // Only the next source wave waits for the whole grid.
    logical_offset+=tile_count;
  }
  if (tensor_owned) {
    __syncthreads();
    if (warp==0) {
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
                const int* stolen, const int* tile_completed,
                const int* overlap, int permit_count,
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
  std::fprintf(file, "],\"tile_completed\":[");
  for (int stage=0; stage<kStages; ++stage) {
    std::fprintf(file,"%s[",stage ? "," : "");
    for (int tile=0; tile<kLogicalTiles; ++tile)
      std::fprintf(file,"%s%d",tile ? "," : "",
                   tile_completed[stage*kLogicalTiles+tile]);
    std::fprintf(file,"]");
  }
  std::fprintf(file,"],\"stage_overlap\":[%d,%d]}\n",
               overlap[0],overlap[1]);
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
  int *heads=nullptr, *tile_completed=nullptr, *owners=nullptr;
  int *processed=nullptr, *dispatched=nullptr, *overlap=nullptr;
  int *stolen=nullptr, *permits=nullptr, *tasks=nullptr;
  if (check(cudaMalloc(&heads, kStages*kLogicalTiles*sizeof(int)),
            "heads malloc") ||
      check(cudaMalloc(&tile_completed,kStages*kLogicalTiles*sizeof(int)),
            "tile completion malloc") ||
      check(cudaMalloc(&owners, kStages*kWaves*kMaxTasks*sizeof(int)),
            "owners malloc") ||
      check(cudaMalloc(&processed, kStages*kWaves*sizeof(int)),
            "processed malloc") ||
      check(cudaMalloc(&dispatched, kWaves*sizeof(int)), "dispatched malloc") ||
      check(cudaMalloc(&stolen, kStages*kWaves*sizeof(int)), "stolen malloc") ||
      check(cudaMalloc(&permits, sizeof(int)), "permits malloc") ||
      check(cudaMalloc(&overlap,2*sizeof(int)), "overlap malloc") ||
      check(cudaMalloc(&tasks, kStages*kWaves*sizeof(int)),
            "tasks malloc")) return 18;
  for (const Case& scenario : cases) {
    if (scenario.grid > active * prop.multiProcessorCount ||
        scenario.communication > scenario.grid ||
        scenario.budget < 0 || scenario.budget > kTotalStageTasks) return 19;
    if (check(cudaMemset(heads,0,kStages*kLogicalTiles*sizeof(int)),
              "heads reset") ||
        check(cudaMemset(tile_completed,0,
                         kStages*kLogicalTiles*sizeof(int)),
              "tile completion reset") ||
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
        check(cudaMemset(overlap,0,2*sizeof(int)),"overlap reset") ||
        check(cudaMemset(device_up_gate,0xff,kUpGateBytes),
              "up/gate sentinel") ||
        check(cudaMemset(device_activated,0xff,kActivatedBytes),
              "activated sentinel") ||
        check(cudaMemset(device_output,0xff,kOutputBytes),"output sentinel") ||
        check(cudaMemcpy(tasks, scenario.tasks, sizeof(scenario.tasks),
                         cudaMemcpyHostToDevice), "task plan copy")) return 20;
    int communication = scenario.communication;
    int budget = scenario.budget;
    void* arguments[] = {&heads, &tile_completed, &owners,
                         &processed, &dispatched,
                         &stolen, &permits, &tasks, &overlap,
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
    int host_tile_completed[kStages*kLogicalTiles], host_overlap[2];
    int host_permits = -1;
    if (check(cudaMemcpy(host_owners, owners, sizeof(host_owners),
                         cudaMemcpyDeviceToHost), "owners read") ||
        check(cudaMemcpy(host_processed, processed, sizeof(host_processed),
                         cudaMemcpyDeviceToHost), "processed read") ||
        check(cudaMemcpy(host_dispatched, dispatched, sizeof(host_dispatched),
                         cudaMemcpyDeviceToHost), "dispatched read") ||
        check(cudaMemcpy(host_stolen, stolen, sizeof(host_stolen),
                         cudaMemcpyDeviceToHost), "stolen read") ||
        check(cudaMemcpy(host_tile_completed,tile_completed,
                         sizeof(host_tile_completed),cudaMemcpyDeviceToHost),
              "tile completion read") ||
        check(cudaMemcpy(host_overlap,overlap,sizeof(host_overlap),
                         cudaMemcpyDeviceToHost),"overlap read") ||
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
    for (int tile=0;tile<kLogicalTiles;++tile) {
      if (host_tile_completed[tile]!=kUpGateTasksPerTile ||
          host_tile_completed[kLogicalTiles+tile]
            !=kActivationTasksPerTile ||
          host_tile_completed[2*kLogicalTiles+tile]!=kDownTasksPerTile)
        return 25;
    }
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
                    host_dispatched, host_stolen, host_tile_completed,
                    host_overlap,host_permits,
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
  cudaFree(tasks); cudaFree(overlap); cudaFree(permits); cudaFree(stolen);
  cudaFree(dispatched); cudaFree(processed); cudaFree(owners); cudaFree(heads);
  cudaFree(tile_completed);
  cudaFree(device_down_map_b); cudaFree(device_down_maps_a);
  cudaFree(device_up_map_b); cudaFree(device_up_maps_a);
  cudaFree(device_output); cudaFree(device_down_weight);
  cudaFree(device_activated); cudaFree(device_up_gate);
  cudaFree(device_up_weight); cudaFree(device_input);
  return 0;
}
