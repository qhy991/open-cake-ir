// B300 development bridge: Cake tile worker consumes real four-rank peer bins.
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
constexpr int kExperts = 32;
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
@ACTIVATION_STAGE@
@DOWN_STAGE@
__global__ void tile_schedule_probe(
    int* task_heads, int* tile_completed, int* task_owner,
    int* processed, int* dispatched, int* stolen, int* steal_permits,
    const int* task_counts, int* overlap, const int* tile_expert,
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
      const int expert=tile_expert[tile];
      if (expert<0 || expert>=kExperts) asm volatile("trap;");
      if (stage==0) {
        cake_upgate_stage_work(shared,tensor_address,warp,
                               up_gate+tile*kRows*kUpGateWidth,
                               up_maps_a+tile,up_map_b+expert,subtile);
      } else if (stage==1) {
        cake_activation_stage_work(up_gate,activated,tile,subtile);
      } else {
        cake_down_stage_work(shared,tensor_address,warp,
                             outputs+tile*kRows*kOutput,
                             down_maps_a+tile,down_map_b+expert,subtile);
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

bool read_slice(const char* root,const char* name,size_t offset,
                std::vector<unsigned char>& bytes) {
  char path[512];
  int n=std::snprintf(path,sizeof(path),"%s/%s",root,name);
  if (n<1 || n>=int(sizeof(path))) return false;
  FILE* file=std::fopen(path,"rb");
  if (!file) return false;
  bool okay=std::fseek(file,long(offset),SEEK_SET)==0
             && std::fread(bytes.data(),1,bytes.size(),file)==bytes.size();
  return std::fclose(file)==0 && okay;
}

bool write_case(const char* root,const Case& scenario,const int* owners,
                const int* processed,const int* dispatched,const int* stolen,
                const int* completed,const int* overlap,int permits) {
  char path[512];
  int n=std::snprintf(path,sizeof(path),"%s/%s.json",root,scenario.name);
  if (n<1 || n>=int(sizeof(path))) return false;
  FILE* file=std::fopen(path,"wx");
  if (!file) return false;
  std::fprintf(file,"{\"name\":\"%s\",\"grid_ctas\":%d,"
                    "\"communication_ctas\":%d,\"steal_budget\":%d,"
                    "\"permit_count\":%d,\"stage_overlap\":[%d,%d],"
                    "\"upgate_bytes\":%d,\"activation_bytes\":%d,"
                    "\"down_bytes\":%d,\"tasks\":[",
               scenario.name,scenario.grid,scenario.communication,
               scenario.budget,permits,overlap[0],overlap[1],
               kUpGateBytes,kActivatedBytes,kOutputBytes);
  for (int stage=0;stage<kStages;++stage) {
    std::fprintf(file,"%s[",stage ? "," : "");
    for (int wave=0;wave<kWaves;++wave)
      std::fprintf(file,"%s%d",wave ? "," : "",scenario.tasks[stage][wave]);
    std::fprintf(file,"]");
  }
  std::fprintf(file,"],\"processed\":[");
  for (int stage=0;stage<kStages;++stage) {
    std::fprintf(file,"%s[",stage ? "," : "");
    for (int wave=0;wave<kWaves;++wave)
      std::fprintf(file,"%s%d",wave ? "," : "",processed[stage*kWaves+wave]);
    std::fprintf(file,"]");
  }
  std::fprintf(file,"],\"stolen\":[");
  for (int stage=0;stage<kStages;++stage) {
    std::fprintf(file,"%s[",stage ? "," : "");
    for (int wave=0;wave<kWaves;++wave)
      std::fprintf(file,"%s%d",wave ? "," : "",stolen[stage*kWaves+wave]);
    std::fprintf(file,"]");
  }
  std::fprintf(file,"],\"dispatched\":[");
  for (int wave=0;wave<kWaves;++wave)
    std::fprintf(file,"%s%d",wave ? "," : "",dispatched[wave]);
  std::fprintf(file,"],\"tile_completed\":[");
  for (int stage=0;stage<kStages;++stage) {
    std::fprintf(file,"%s[",stage ? "," : "");
    for (int tile=0;tile<kLogicalTiles;++tile)
      std::fprintf(file,"%s%d",tile ? "," : "",
                   completed[stage*kLogicalTiles+tile]);
    std::fprintf(file,"]");
  }
  std::fprintf(file,"],\"owner_blocks\":[");
  for (int stage=0;stage<kStages;++stage) {
    std::fprintf(file,"%s[",stage ? "," : "");
    for (int wave=0;wave<kWaves;++wave) {
      std::fprintf(file,"%s[",wave ? "," : "");
      for (int task=0;task<scenario.tasks[stage][wave];++task)
        std::fprintf(file,"%s%d",task ? "," : "",
                     owners[(stage*kWaves+wave)*kMaxTasks+task]);
      std::fprintf(file,"]");
    }
    std::fprintf(file,"]");
  }
  std::fprintf(file,"]}\n");
  return std::fclose(file)==0;
}
} // namespace

int main(int argc,char** argv) {
  if (argc!=4) return 2;
  int owner=-1,devices=0;
  if (std::sscanf(argv[2],"%d",&owner)!=1 || owner<0 || owner>=4 ||
      check(cudaGetDeviceCount(&devices),"device count") || devices!=4 ||
      check(cudaSetDevice(owner),"select owner device")) return 3;
  cudaDeviceProp prop{};
  if (check(cudaGetDeviceProperties(&prop,owner),"device properties") ||
      prop.major!=10 || prop.minor!=3 ||
      (std::strcmp(prop.name,"NVIDIA B300") &&
       std::strcmp(prop.name,"NVIDIA B300 SXM6 AC"))) return 4;
  int cooperative=0;
  if (check(cudaDeviceGetAttribute(&cooperative,cudaDevAttrCooperativeLaunch,
                                   owner),"cooperative support") || cooperative!=1)
    return 5;
  if (check(cudaFuncSetAttribute(tile_schedule_probe,
            cudaFuncAttributeMaxDynamicSharedMemorySize,kDynamicShared),
            "dynamic shared attribute")) return 6;
  int active=0;
  if (check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active,tile_schedule_probe,kThreads,kDynamicShared),
            "cooperative occupancy") || active<1 ||
      active*prop.multiProcessorCount<148) return 7;
  const Case scenario={"near_full_c147_budget5888",148,147,5888,
      {{0,408,360,768},{0,2176,1920,4096},{0,544,480,1024}}};
  std::vector<unsigned char> input(kInputBytes),
      expert_ids(kLogicalTiles*sizeof(int)),
      up_weight(kExperts*kUpGateWeightBytes),
      down_weight(kExperts*kDownWeightBytes),
      actual_upgate(kUpGateBytes),actual_activation(kActivatedBytes),
      actual_down(kOutputBytes);
  if (!read_exact(argv[1],"x_tiles.bf16",input) ||
      !read_exact(argv[1],"tile_expert.i32",expert_ids) ||
      !read_slice(argv[3],"weights_upgate.bf16",
                  size_t(owner)*up_weight.size(),up_weight) ||
      !read_slice(argv[3],"weights_down.bf16",
                  size_t(owner)*down_weight.size(),down_weight)) return 8;
  for (int tile=0;tile<kLogicalTiles;++tile) {
    int expert=-1;
    std::memcpy(&expert,expert_ids.data()+tile*sizeof(int),sizeof(int));
    if (expert<0 || expert>=kExperts) return 9;
  }
  __nv_bfloat16 *device_input=nullptr,*device_up_weight=nullptr;
  __nv_bfloat16 *device_activated=nullptr,*device_down_weight=nullptr;
  float *device_upgate=nullptr,*device_output=nullptr;
  int* device_experts=nullptr;
  CUtensorMap *up_maps_a=nullptr,*up_map_b=nullptr;
  CUtensorMap *down_maps_a=nullptr,*down_map_b=nullptr;
  alignas(128) CUtensorMap host_up_maps_a[kLogicalTiles],host_up_map_b[kExperts];
  alignas(128) CUtensorMap host_down_maps_a[kLogicalTiles],host_down_map_b[kExperts];
  if (check(cudaMalloc(&device_input,kInputBytes),"input malloc") ||
      check(cudaMalloc(&device_experts,expert_ids.size()),"expert malloc") ||
      check(cudaMalloc(&device_up_weight,up_weight.size()),"up weight malloc") ||
      check(cudaMalloc(&device_down_weight,down_weight.size()),"down weight malloc") ||
      check(cudaMalloc(&device_upgate,kUpGateBytes),"upgate malloc") ||
      check(cudaMalloc(&device_activated,kActivatedBytes),"activation malloc") ||
      check(cudaMalloc(&device_output,kOutputBytes),"down malloc") ||
      check(cudaMalloc(&up_maps_a,sizeof(host_up_maps_a)),"up A maps malloc") ||
      check(cudaMalloc(&up_map_b,sizeof(host_up_map_b)),"up B maps malloc") ||
      check(cudaMalloc(&down_maps_a,sizeof(host_down_maps_a)),"down A maps malloc") ||
      check(cudaMalloc(&down_map_b,sizeof(host_down_map_b)),"down B maps malloc") ||
      check(cudaMemcpy(device_input,input.data(),kInputBytes,cudaMemcpyHostToDevice),
            "input H2D") ||
      check(cudaMemcpy(device_experts,expert_ids.data(),expert_ids.size(),
                       cudaMemcpyHostToDevice),"expert H2D") ||
      check(cudaMemcpy(device_up_weight,up_weight.data(),up_weight.size(),
                       cudaMemcpyHostToDevice),"up weight H2D") ||
      check(cudaMemcpy(device_down_weight,down_weight.data(),down_weight.size(),
                       cudaMemcpyHostToDevice),"down weight H2D")) return 10;
  static_assert(sizeof(CUtensorMap)%64==0,"tensor map array alignment");
  cuuint64_t up_dims_a[2]={kInputWidth,kRows};
  cuuint64_t up_dims_b[2]={kInputWidth,kUpGateWidth};
  cuuint64_t up_stride[1]={kInputWidth*2};
  cuuint64_t down_dims_a[2]={kHidden,kRows};
  cuuint64_t down_dims_b[2]={kHidden,kOutput};
  cuuint64_t down_stride[1]={kHidden*2};
  cuuint32_t box_a[2]={64,kRows},box_b[2]={64,64},steps[2]={1,1};
  for (int tile=0;tile<kLogicalTiles;++tile) {
    CUresult up=cuTensorMapEncodeTiled(&host_up_maps_a[tile],
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,
      device_input+tile*kRows*kInputWidth,up_dims_a,up_stride,box_a,steps,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUresult down=cuTensorMapEncodeTiled(&host_down_maps_a[tile],
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,
      device_activated+tile*kRows*kHidden,down_dims_a,down_stride,box_a,steps,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (up!=CUDA_SUCCESS || down!=CUDA_SUCCESS) return 11;
  }
  for (int expert=0;expert<kExperts;++expert) {
    CUresult up=cuTensorMapEncodeTiled(&host_up_map_b[expert],
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,
      device_up_weight+expert*kUpGateWeightBytes/2,up_dims_b,up_stride,box_b,steps,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUresult down=cuTensorMapEncodeTiled(&host_down_map_b[expert],
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,
      device_down_weight+expert*kDownWeightBytes/2,
      down_dims_b,down_stride,box_b,steps,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (up!=CUDA_SUCCESS || down!=CUDA_SUCCESS) return 12;
  }
  if (check(cudaMemcpy(up_maps_a,host_up_maps_a,sizeof(host_up_maps_a),
                       cudaMemcpyHostToDevice),"up A maps H2D") ||
      check(cudaMemcpy(up_map_b,host_up_map_b,sizeof(host_up_map_b),
                       cudaMemcpyHostToDevice),"up B maps H2D") ||
      check(cudaMemcpy(down_maps_a,host_down_maps_a,sizeof(host_down_maps_a),
                       cudaMemcpyHostToDevice),"down A maps H2D") ||
      check(cudaMemcpy(down_map_b,host_down_map_b,sizeof(host_down_map_b),
                       cudaMemcpyHostToDevice),"down B maps H2D")) return 13;
  int *heads=nullptr,*completed=nullptr,*owners=nullptr,*processed=nullptr;
  int *dispatched=nullptr,*stolen=nullptr,*permits=nullptr,*tasks=nullptr;
  int *overlap=nullptr;
  if (check(cudaMalloc(&heads,kStages*kLogicalTiles*sizeof(int)),"heads malloc") ||
      check(cudaMalloc(&completed,kStages*kLogicalTiles*sizeof(int)),"completed malloc") ||
      check(cudaMalloc(&owners,kStages*kWaves*kMaxTasks*sizeof(int)),"owners malloc") ||
      check(cudaMalloc(&processed,kStages*kWaves*sizeof(int)),"processed malloc") ||
      check(cudaMalloc(&dispatched,kWaves*sizeof(int)),"dispatched malloc") ||
      check(cudaMalloc(&stolen,kStages*kWaves*sizeof(int)),"stolen malloc") ||
      check(cudaMalloc(&permits,sizeof(int)),"permits malloc") ||
      check(cudaMalloc(&tasks,kStages*kWaves*sizeof(int)),"tasks malloc") ||
      check(cudaMalloc(&overlap,2*sizeof(int)),"overlap malloc") ||
      check(cudaMemset(heads,0,kStages*kLogicalTiles*sizeof(int)),"heads reset") ||
      check(cudaMemset(completed,0,kStages*kLogicalTiles*sizeof(int)),
            "completed reset") ||
      check(cudaMemset(owners,0xff,kStages*kWaves*kMaxTasks*sizeof(int)),
            "owners reset") ||
      check(cudaMemset(processed,0,kStages*kWaves*sizeof(int)),"processed reset") ||
      check(cudaMemset(dispatched,0,kWaves*sizeof(int)),"dispatched reset") ||
      check(cudaMemset(stolen,0,kStages*kWaves*sizeof(int)),"stolen reset") ||
      check(cudaMemset(permits,0,sizeof(int)),"permits reset") ||
      check(cudaMemset(overlap,0,2*sizeof(int)),"overlap reset") ||
      check(cudaMemset(device_upgate,0xff,kUpGateBytes),"upgate sentinel") ||
      check(cudaMemset(device_activated,0xff,kActivatedBytes),"activation sentinel") ||
      check(cudaMemset(device_output,0xff,kOutputBytes),"down sentinel") ||
      check(cudaMemcpy(tasks,scenario.tasks,sizeof(scenario.tasks),
                       cudaMemcpyHostToDevice),"tasks copy")) return 14;
  int communication=scenario.communication,budget=scenario.budget;
  void* arguments[]={&heads,&completed,&owners,&processed,&dispatched,
                     &stolen,&permits,&tasks,&overlap,&device_experts,
                     &device_upgate,&device_activated,&up_maps_a,&up_map_b,
                     &down_maps_a,&down_map_b,&device_output,
                     &communication,&budget};
  if (check(cudaLaunchCooperativeKernel(
           reinterpret_cast<const void*>(tile_schedule_probe),
           dim3(scenario.grid),dim3(kThreads),arguments,kDynamicShared,nullptr),
           "worker launch") ||
      check(cudaDeviceSynchronize(),"worker completion")) return 15;
  int host_owners[kStages*kWaves*kMaxTasks];
  int host_processed[kStages*kWaves],host_dispatched[kWaves];
  int host_stolen[kStages*kWaves],host_completed[kStages*kLogicalTiles];
  int host_overlap[2],host_permits=-1;
  if (check(cudaMemcpy(host_owners,owners,sizeof(host_owners),
                       cudaMemcpyDeviceToHost),"owners read") ||
      check(cudaMemcpy(host_processed,processed,sizeof(host_processed),
                       cudaMemcpyDeviceToHost),"processed read") ||
      check(cudaMemcpy(host_dispatched,dispatched,sizeof(host_dispatched),
                       cudaMemcpyDeviceToHost),"dispatched read") ||
      check(cudaMemcpy(host_stolen,stolen,sizeof(host_stolen),
                       cudaMemcpyDeviceToHost),"stolen read") ||
      check(cudaMemcpy(host_completed,completed,sizeof(host_completed),
                       cudaMemcpyDeviceToHost),"completed read") ||
      check(cudaMemcpy(host_overlap,overlap,sizeof(host_overlap),
                       cudaMemcpyDeviceToHost),"overlap read") ||
      check(cudaMemcpy(&host_permits,permits,sizeof(int),
                       cudaMemcpyDeviceToHost),"permits read") ||
      check(cudaMemcpy(actual_upgate.data(),device_upgate,kUpGateBytes,
                       cudaMemcpyDeviceToHost),"upgate read") ||
      check(cudaMemcpy(actual_activation.data(),device_activated,kActivatedBytes,
                       cudaMemcpyDeviceToHost),"activation read") ||
      check(cudaMemcpy(actual_down.data(),device_output,kOutputBytes,
                       cudaMemcpyDeviceToHost),"down read")) return 16;
  int stolen_total=0;
  for (int stage=0;stage<kStages;++stage) {
    int units=stage==0 ? kUpGateTasksPerTile :
              stage==1 ? kActivationTasksPerTile : kDownTasksPerTile;
    for (int tile=0;tile<kLogicalTiles;++tile)
      if (host_completed[stage*kLogicalTiles+tile]!=units) return 17;
    for (int wave=0;wave<kWaves;++wave) {
      int index=stage*kWaves+wave;
      if (host_processed[index]!=scenario.tasks[stage][wave]) return 18;
      stolen_total+=host_stolen[index];
      for (int task=0;task<scenario.tasks[stage][wave];++task) {
        int who=host_owners[index*kMaxTasks+task];
        if (who<0 || who>=scenario.grid) return 19;
      }
    }
  }
  for (int wave=0;wave<kWaves;++wave)
    if (host_dispatched[wave]!=scenario.communication) return 20;
  if (stolen_total!=host_permits || stolen_total!=scenario.budget) return 21;
  if (!write_actual(argv[1],scenario.name,"upgate.actual.fp32",actual_upgate) ||
      !write_actual(argv[1],scenario.name,"activated.actual.bf16",actual_activation) ||
      !write_actual(argv[1],scenario.name,"down.actual.fp32",actual_down) ||
      !write_case(argv[1],scenario,host_owners,host_processed,
                  host_dispatched,host_stolen,host_completed,host_overlap,
                  host_permits)) return 22;
  char path[512];
  int n=std::snprintf(path,sizeof(path),"%s/resource.json",argv[1]);
  if (n<1 || n>=int(sizeof(path))) return 23;
  FILE* file=std::fopen(path,"wx");
  if (!file) return 24;
  std::fprintf(file,"{\"owner_rank\":%d,\"target\":\"sm_103a\","
                    "\"device_name\":\"%s\",\"sm_count\":%d,"
                    "\"active_blocks_per_sm\":%d,\"grid_ctas\":%d,"
                    "\"threads_per_cta\":%d,\"dynamic_shared_bytes\":%d}\n",
               owner,prop.name,prop.multiProcessorCount,active,
               scenario.grid,kThreads,kDynamicShared);
  std::fclose(file);
  std::printf("owner %d: processed=%d stolen=%d\n",owner,kTotalStageTasks,
              stolen_total);
  cudaFree(overlap);cudaFree(tasks);cudaFree(permits);cudaFree(stolen);
  cudaFree(dispatched);cudaFree(processed);cudaFree(owners);cudaFree(completed);
  cudaFree(heads);cudaFree(down_map_b);cudaFree(down_maps_a);
  cudaFree(up_map_b);cudaFree(up_maps_a);cudaFree(device_output);
  cudaFree(device_activated);cudaFree(device_upgate);
  cudaFree(device_down_weight);cudaFree(device_up_weight);
  cudaFree(device_experts);cudaFree(device_input);
  return 0;
}
