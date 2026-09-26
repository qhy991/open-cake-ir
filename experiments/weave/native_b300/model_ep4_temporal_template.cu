// B300 temporal development: each source wave publishes expert tiles while
// a 96-CTA Cake FFN grid can execute earlier waves on the same GPU.
// One cooperative grid reserves the tensor-core worker's 192-thread/49,200-B
// CTA footprint. A CTA allocates TMEM at its first claimed tile and releases
// it once after all waves. The stage body below follows the retained Cake
// tensor-core emissions; activation uses the Cake FP32-to-BF16 graph.
// Per-tile completion counters publish successors with GPU-scope release/
// acquire. Grid barriers only delimit source waves within the FFN worker.
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
#include "combine/kernel.cu"

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
__device__ __forceinline__ int worker_wave_acquire(const int* pointer) {
  int value;
  asm volatile("ld.acquire.sys.global.s32 %0, [%1];"
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
    const int* task_counts, const int* wave_ready, int* overlap,
    const int* tile_expert,
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
    if (threadIdx.x==0)
      while (worker_wave_acquire(&wave_ready[wave])==0)
        __nanosleep(64);
    __syncthreads();
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

constexpr int R=4,T=512,K=8,E=128,H=2048;
constexpr int LOCAL_E=E/R,MAX_ROWS=R*T,ROUTES=R*T*K,LOCAL_ROUTES=T*K;
constexpr size_t HIDDEN_BYTES=size_t(R)*T*H*sizeof(uint16_t);
constexpr size_t IDS_BYTES=size_t(ROUTES)*sizeof(int);
constexpr size_t TILE_BYTES=size_t(kLogicalTiles)*kRows*H*sizeof(uint16_t);
constexpr size_t TILE_KEY_BYTES=size_t(kLogicalTiles)*kRows*sizeof(int);
constexpr size_t ROUTE_CONTRIBUTION_BYTES=size_t(ROUTES)*H*sizeof(float);
constexpr size_t FINAL_BYTES=size_t(R)*T*H*sizeof(uint16_t);

struct Bin {
  int count[LOCAL_E];
  int source_done[R];
  int source_wave_done[R][kWaves];
  int wave_consumed[kWaves];
  int ready[LOCAL_E*MAX_ROWS];
  int keys[LOCAL_E*MAX_ROWS];
  int route_location[ROUTES];
  int route_ready[ROUTES];
  uint32_t sorted_order[LOCAL_E*256];
  int early_wave[LOCAL_E];
  int early_count[LOCAL_E];
  int wave_counts[kWaves];
  __nv_bfloat16 rows[size_t(LOCAL_E)*MAX_ROWS*H];
  int error;
};
struct BinParams {
  Bin* bins[R];
  const __nv_bfloat16* hidden;
  const int* ids;
  int* tile_keys;
  int* tile_experts;
  __nv_bfloat16* tile_rows;
  int rank;
};
struct ReturnParams {
  float* contributions[R];
  int* ready[R];
  const float* down;
  const int* tile_keys;
  int rank;
};
__device__ __forceinline__ int system_reserve(int* pointer) {
  int old;
  asm volatile("atom.relaxed.sys.global.add.s32 %0, [%1], %2;"
               : "=r"(old) : "l"(pointer), "r"(1) : "memory");
  return old;
}
__device__ __forceinline__ void system_publish(int* pointer) {
  asm volatile("st.release.sys.global.s32 [%0], %1;"
               :: "l"(pointer), "r"(1) : "memory");
}
__device__ __forceinline__ int system_acquire(const int* pointer) {
  int value;
  asm volatile("ld.acquire.sys.global.s32 %0, [%1];"
               : "=r"(value) : "l"(pointer) : "memory");
  return value;
}
__device__ __forceinline__ uint32_t route_order(int key) {
  int source=key/(T*K),token=(key/K)%T,route=key%K;
  return uint32_t((token/128)*(R*128*K)+source*(128*K)
                  +(token%128)*K+route);
}
__device__ __forceinline__ int original_route_key(uint32_t ordered) {
  int wave=int(ordered)/(R*128*K);
  int remainder=int(ordered)%(R*128*K);
  int source=remainder/(128*K);
  int token=(remainder%(128*K))/K;
  int route=remainder%K;
  return (source*T+wave*128+token)*K+route;
}
__global__ void dispatch_routes(const BinParams* params) {
  int token=int(blockIdx.x),route=int(blockIdx.y);
  int expert=params->ids[token*K+route];
  if (expert<0 || expert>=E) {
    if (threadIdx.x==0) atomicCAS(&params->bins[params->rank]->error,0,1);
    return;
  }
  int owner=expert/LOCAL_E,local_expert=expert%LOCAL_E;
  Bin* remote=params->bins[owner];
  __shared__ int slot;
  if (threadIdx.x==0) slot=system_reserve(&remote->count[local_expert]);
  __syncthreads();
  if (slot<0 || slot>=MAX_ROWS) {
    if (threadIdx.x==0) atomicCAS(&params->bins[params->rank]->error,0,2);
    return;
  }
  int index=local_expert*MAX_ROWS+slot;
  int key=(params->rank*T+token)*K+route;
  const uint16_t* input=reinterpret_cast<const uint16_t*>(params->hidden)
      +size_t(token)*H;
  uint16_t* output=reinterpret_cast<uint16_t*>(remote->rows)
      +size_t(index)*H;
  for (int feature=int(threadIdx.x);feature<H;feature+=int(blockDim.x))
    output[feature]=input[feature];
  if (threadIdx.x==0) {
    remote->keys[index]=key;
    remote->route_location[key]=index;
  }
  __syncthreads();
  if (threadIdx.x==0) {
    system_publish(&remote->ready[index]);
    system_publish(&remote->route_ready[key]);
  }
}
__global__ void mark_source_done(const BinParams* params) {
  if (threadIdx.x!=0) return;
  for (int owner=0;owner<R;++owner)
    system_publish(&params->bins[owner]->source_done[params->rank]);
}
__global__ void derive_expert_order(const BinParams* params) {
  int expert=int(blockIdx.x),lane=int(threadIdx.x);
  Bin* local=params->bins[params->rank];
  if (lane==0)
    for (int source=0;source<R;++source)
      while (system_acquire(&local->source_done[source])==0)
        __nanosleep(64);
  __syncthreads();
  int count=local->count[expert];
  if (count<1 || count>256) {
    if (lane==0) atomicCAS(&local->error,0,4);
    return;
  }
  __shared__ uint32_t order[256];
  order[lane]=lane<count
      ? route_order(local->keys[expert*MAX_ROWS+lane]) : 0xffffffffu;
  __syncthreads();
  for (int width=2;width<=256;width<<=1) {
    for (int span=width>>1;span>0;span>>=1) {
      uint32_t self=order[lane],partner=order[lane^span];
      bool minimum=((lane&width)==0)==((lane&span)==0);
      bool take_partner=minimum ? self>partner : self<partner;
      __syncthreads();
      order[lane]=take_partner ? partner : self;
      __syncthreads();
    }
  }
  if (lane<count) local->sorted_order[expert*256+lane]=order[lane];
  __syncthreads();
  if (lane==0) {
    int cumulative[4]{};
    for (int index=0;index<count;++index) {
      int wave=int(order[index])/(R*128*K);
      if (wave<0 || wave>=kWaves) {
        atomicCAS(&local->error,0,5);return;
      }
      ++cumulative[wave];
    }
    int through1=cumulative[0]+cumulative[1];
    int through2=through1+cumulative[2];
    int early=through1>=64 ? 1 : (through2>=64 ? 2 : -1);
    int split=early==1 ? through1 : through2;
    if (cumulative[0]>=64 || early<0 || split<64 || split>=kRows ||
        count-split<1 || count-split>kRows) {
      atomicCAS(&local->error,0,6);return;
    }
    local->early_wave[expert]=early;
    local->early_count[expert]=split;
  }
}
__global__ void assign_tile_plan(const BinParams* params) {
  int expert=int(threadIdx.x);
  Bin* local=params->bins[params->rank];
  if (expert>=LOCAL_E) return;
  int wave=local->early_wave[expert];
  int wave1=0,prior=0;
  for (int other=0;other<LOCAL_E;++other) {
    int other_wave=local->early_wave[other];
    wave1+=other_wave==1;
    if (other<expert && other_wave==wave) ++prior;
  }
  if (wave!=1 && wave!=2) {
    atomicCAS(&local->error,0,7);return;
  }
  int first=(wave==1 ? 0 : wave1)+prior;
  int terminal=LOCAL_E+expert;
  params->tile_experts[first]=expert;
  params->tile_experts[terminal]=expert;
  int split=local->early_count[expert],count=local->count[expert];
  for (int row=0;row<kRows;++row) {
    params->tile_keys[first*kRows+row]=row<split
        ? original_route_key(local->sorted_order[expert*256+row]) : -1;
    params->tile_keys[terminal*kRows+row]=row<count-split
        ? original_route_key(local->sorted_order[expert*256+split+row]) : -1;
  }
  if (expert==0) {
    local->wave_counts[0]=0;
    local->wave_counts[1]=wave1;
    local->wave_counts[2]=LOCAL_E-wave1;
    local->wave_counts[3]=LOCAL_E;
  }
}
__global__ void expand_stage_tasks(const BinParams* params,int* tasks) {
  int index=int(threadIdx.x);
  if (index>=kStages*kWaves) return;
  int stage=index/kWaves,wave=index%kWaves;
  int factor=stage==0 ? kUpGateTasksPerTile :
             stage==1 ? kActivationTasksPerTile : kDownTasksPerTile;
  tasks[index]=params->bins[params->rank]->wave_counts[wave]*factor;
}
__global__ void dispatch_source_wave(const BinParams* params,int wave) {
  int token=wave*128+int(blockIdx.x),route=int(blockIdx.y);
  int expert=params->ids[token*K+route];
  if (expert<0 || expert>=E) {
    if (threadIdx.x==0) atomicCAS(&params->bins[params->rank]->error,0,11);
    return;
  }
  int owner=expert/LOCAL_E,local_expert=expert%LOCAL_E;
  Bin* remote=params->bins[owner];
  __shared__ int slot;
  if (threadIdx.x==0) slot=system_reserve(&remote->count[local_expert]);
  __syncthreads();
  if (slot<0 || slot>=MAX_ROWS) {
    if (threadIdx.x==0) atomicCAS(&params->bins[params->rank]->error,0,12);
    return;
  }
  int index=local_expert*MAX_ROWS+slot;
  int key=(params->rank*T+token)*K+route;
  const uint16_t* input=reinterpret_cast<const uint16_t*>(params->hidden)
      +size_t(token)*H;
  uint16_t* output=reinterpret_cast<uint16_t*>(remote->rows)
      +size_t(index)*H;
  for (int feature=int(threadIdx.x);feature<H;feature+=int(blockDim.x))
    output[feature]=input[feature];
  if (threadIdx.x==0) {
    remote->keys[index]=key;
    remote->route_location[key]=index;
  }
  __syncthreads();
  if (threadIdx.x==0) {
    system_publish(&remote->ready[index]);
    system_publish(&remote->route_ready[key]);
  }
}
__global__ void mark_source_wave_done(const BinParams* params,int wave) {
  if (threadIdx.x!=0) return;
  for (int owner=0;owner<R;++owner)
    system_publish(&params->bins[owner]->source_wave_done[params->rank][wave]);
}
__device__ __forceinline__ void sort_expert_snapshot(
    Bin* local,int expert,int lane,int count,uint32_t* order) {
  order[lane]=lane<count
      ? route_order(local->keys[expert*MAX_ROWS+lane]) : 0xffffffffu;
  __syncthreads();
  for (int width=2;width<=256;width<<=1) {
    for (int span=width>>1;span>0;span>>=1) {
      uint32_t self=order[lane],partner=order[lane^span];
      bool minimum=((lane&width)==0)==((lane&span)==0);
      bool take_partner=minimum ? self>partner : self<partner;
      __syncthreads();
      order[lane]=take_partner ? partner : self;
      __syncthreads();
    }
  }
}
__global__ void derive_early_wave(const BinParams* params,int wave) {
  int expert=int(blockIdx.x),lane=int(threadIdx.x);
  Bin* local=params->bins[params->rank];
  if (lane==0)
    for (int source=0;source<R;++source)
      while (system_acquire(&local->source_wave_done[source][wave])==0)
        __nanosleep(64);
  __syncthreads();
  if (local->early_wave[expert]!=0) return;
  int count=local->count[expert];
  if (count<1 || count>256) {
    if (lane==0) atomicCAS(&local->error,0,13);
    return;
  }
  __shared__ uint32_t order[256];
  sort_expert_snapshot(local,expert,lane,count,order);
  if (lane==0) {
    if (order[count-1]/(R*128*K)>uint32_t(wave)) {
      atomicCAS(&local->error,0,14);return;
    }
    if (count>=64) {
      if (count>=kRows) {
        atomicCAS(&local->error,0,15);return;
      }
      local->early_wave[expert]=wave;
      local->early_count[expert]=count;
    } else if (wave==2) {
      atomicCAS(&local->error,0,16);return;
    }
  }
  __syncthreads();
  if (local->early_wave[expert]==wave && lane<count)
    local->sorted_order[expert*256+lane]=order[lane];
}
__global__ void assign_early_wave(const BinParams* params,int wave) {
  int expert=int(threadIdx.x);
  Bin* local=params->bins[params->rank];
  if (expert>=LOCAL_E) return;
  int first_wave1=0;
  for (int other=0;other<LOCAL_E;++other)
    first_wave1+=local->early_wave[other]==1;
  int early=local->early_wave[expert];
  if (early==wave) {
    int prior=0;
    for (int other=0;other<expert;++other)
      prior+=local->early_wave[other]==wave;
    int slot=(wave==1 ? 0 : first_wave1)+prior;
    params->tile_experts[slot]=expert;
    int count=local->early_count[expert];
    for (int row=0;row<kRows;++row)
      params->tile_keys[slot*kRows+row]=row<count
          ? original_route_key(local->sorted_order[expert*256+row]) : -1;
  }
  if (expert==0) {
    if (wave==1) local->wave_counts[1]=first_wave1;
    else local->wave_counts[2]=LOCAL_E-first_wave1;
  }
}
__global__ void derive_terminal_order(const BinParams* params) {
  int expert=int(blockIdx.x),lane=int(threadIdx.x);
  Bin* local=params->bins[params->rank];
  if (lane==0)
    for (int source=0;source<R;++source)
      while (system_acquire(&local->source_wave_done[source][3])==0)
        __nanosleep(64);
  __syncthreads();
  int count=local->count[expert],split=local->early_count[expert];
  if (count<1 || count>256 || split<64 || split>=kRows ||
      count-split<1 || count-split>kRows ||
      (local->early_wave[expert]!=1 && local->early_wave[expert]!=2)) {
    if (lane==0) atomicCAS(&local->error,0,17);
    return;
  }
  __shared__ uint32_t order[256];
  sort_expert_snapshot(local,expert,lane,count,order);
  if (lane<count) local->sorted_order[expert*256+lane]=order[lane];
}
__global__ void assign_terminal_tiles(const BinParams* params) {
  int expert=int(threadIdx.x);
  Bin* local=params->bins[params->rank];
  if (expert>=LOCAL_E) return;
  int slot=LOCAL_E+expert;
  params->tile_experts[slot]=expert;
  int split=local->early_count[expert],count=local->count[expert];
  for (int row=0;row<kRows;++row)
    params->tile_keys[slot*kRows+row]=row<count-split
        ? original_route_key(local->sorted_order[expert*256+split+row]) : -1;
  if (expert==0) local->wave_counts[3]=LOCAL_E;
}
__global__ void expand_tasks_for_wave(const BinParams* params,int* tasks,
                                      int wave) {
  int stage=int(threadIdx.x);
  if (stage>=kStages) return;
  int factor=stage==0 ? kUpGateTasksPerTile :
             stage==1 ? kActivationTasksPerTile : kDownTasksPerTile;
  tasks[stage*kWaves+wave]=
      params->bins[params->rank]->wave_counts[wave]*factor;
}
__global__ void gather_wave_tiles(const BinParams* params,int wave) {
  Bin* local=params->bins[params->rank];
  if (threadIdx.x==0)
    for (int source=0;source<R;++source)
      while (system_acquire(&local->source_wave_done[source][wave])==0)
        __nanosleep(64);
  __syncthreads();
  int count=local->wave_counts[wave];
  if (int(blockIdx.x)>=count) return;
  int offset=wave==1 ? 0 : wave==2 ? local->wave_counts[1] : LOCAL_E;
  int tile=offset+int(blockIdx.x),row=int(blockIdx.y);
  int key=params->tile_keys[tile*kRows+row];
  __nv_bfloat16* output=params->tile_rows+(size_t(tile)*kRows+row)*H;
  if (key<0) {
    for (int feature=int(threadIdx.x);feature<H;feature+=int(blockDim.x))
      output[feature]=__float2bfloat16_rn(0.0f);
    return;
  }
  if (key>=ROUTES) asm volatile("trap;");
  if (threadIdx.x==0)
    while (system_acquire(&local->route_ready[key])==0) __nanosleep(64);
  __syncthreads();
  int location=local->route_location[key];
  int expert=params->tile_experts[tile];
  if (location<0 || location>=LOCAL_E*MAX_ROWS ||
      location/MAX_ROWS!=expert) {
    if (threadIdx.x==0) atomicCAS(&local->error,0,18);
    return;
  }
  const __nv_bfloat16* input=local->rows+size_t(location)*H;
  for (int feature=int(threadIdx.x);feature<H;feature+=int(blockDim.x))
    output[feature]=input[feature];
}
__global__ void publish_wave_ready(const BinParams* params,int wave) {
  if (threadIdx.x!=0) return;
  Bin* local=params->bins[params->rank];
  for (int source=0;source<R;++source)
    while (system_acquire(&local->source_wave_done[source][wave])==0)
      __nanosleep(64);
  system_publish(&local->wave_consumed[wave]);
}
__global__ void wait_all_destinations(const BinParams* params,int wave) {
  if (threadIdx.x!=0) return;
  for (int owner=0;owner<R;++owner)
    while (system_acquire(&params->bins[owner]->wave_consumed[wave])==0)
      __nanosleep(64);
}
__global__ void gather_tiles(const BinParams* params) {
  int tile=int(blockIdx.x),row=int(blockIdx.y);
  int key=params->tile_keys[tile*kRows+row];
  __nv_bfloat16* output=params->tile_rows+(size_t(tile)*kRows+row)*H;
  if (key<0) {
    for (int feature=int(threadIdx.x);feature<H;feature+=int(blockDim.x))
      output[feature]=__float2bfloat16_rn(0.0f);
    return;
  }
  if (key>=ROUTES) asm volatile("trap;");
  Bin* local=params->bins[params->rank];
  if (threadIdx.x==0)
    while (system_acquire(&local->route_ready[key])==0) __nanosleep(64);
  __syncthreads();
  int location=local->route_location[key];
  int expert=params->tile_experts[tile];
  if (location<0 || location>=LOCAL_E*MAX_ROWS ||
      location/MAX_ROWS!=expert) {
    if (threadIdx.x==0) atomicCAS(&local->error,0,3);
    return;
  }
  const __nv_bfloat16* input=local->rows+size_t(location)*H;
  for (int feature=int(threadIdx.x);feature<H;feature+=int(blockDim.x))
    output[feature]=input[feature];
}
__global__ void scatter_returns(const ReturnParams* params) {
  int tile=int(blockIdx.x),row=int(blockIdx.y);
  int key=params->tile_keys[tile*kRows+row];
  if (key<0) return;
  if (key>=ROUTES) asm volatile("trap;");
  int source=key/LOCAL_ROUTES,slot=key%LOCAL_ROUTES;
  const float* input=params->down+(size_t(tile)*kRows+row)*H;
  float* output=params->contributions[source]+size_t(slot)*H;
  for (int feature=int(threadIdx.x);feature<H;feature+=int(blockDim.x))
    output[feature]=input[feature];
  __syncthreads();
  if (threadIdx.x==0) system_publish(params->ready[source]+slot);
}
__global__ void wait_returns(const ReturnParams* params) {
  int slot=int(blockIdx.x)*int(blockDim.x)+int(threadIdx.x);
  if (slot>=LOCAL_ROUTES) return;
  while (system_acquire(params->ready[params->rank]+slot)==0)
    __nanosleep(64);
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
bool read_owner(const char* root,int rank,const char* name,
                std::vector<unsigned char>& bytes) {
  char path[512];
  int n=std::snprintf(path,sizeof(path),"%s/rank%d/%s",root,rank,name);
  if (n<1 || n>=int(sizeof(path))) return false;
  FILE* file=std::fopen(path,"rb");
  if (!file) return false;
  bool okay=std::fread(bytes.data(),1,bytes.size(),file)==bytes.size()
            && std::fgetc(file)==EOF;
  return std::fclose(file)==0 && okay;
}
bool write_exact(const char* root,const char* name,const void* bytes,
                 size_t extent) {
  char path[512];
  int n=std::snprintf(path,sizeof(path),"%s/%s",root,name);
  if (n<1 || n>=int(sizeof(path))) return false;
  FILE* file=std::fopen(path,"wbx");
  if (!file) return false;
  bool okay=std::fwrite(bytes,1,extent,file)==extent;
  return std::fclose(file)==0 && okay;
}
struct RankState {
  Bin* bin{};
  __nv_bfloat16 *hidden{},*tile_input{},*up_weight{},*down_weight{};
  __nv_bfloat16 *activated{},*final_output{};
  int *ids{},*tile_keys{},*tile_experts{},*heads{},*completed{},*owners{};
  int *processed{},*dispatched{},*stolen{},*permits{},*tasks{},*overlap{};
  int* return_ready{};
  float *upgate{},*down{},*contributions{},*route_weights{};
  CUtensorMap *up_maps_a{},*up_map_b{},*down_maps_a{},*down_map_b{};
  BinParams* bin_params{};
  ReturnParams* return_params{};
};
} // namespace

int main(int argc,char** argv) {
  if (argc!=6) return 2;
  int communication_control=-1,steal_control=-1;
  if (std::sscanf(argv[4],"%d",&communication_control)!=1 ||
      std::sscanf(argv[5],"%d",&steal_control)!=1 ||
      communication_control<1 || communication_control>96 ||
      steal_control<0 || steal_control>kTotalStageTasks)
    return 2;
  int devices=0;
  if (check(cudaGetDeviceCount(&devices),"device count") || devices!=R)
    return 3;
  std::vector<unsigned char> hidden(HIDDEN_BYTES),ids(IDS_BYTES);
  std::vector<unsigned char> route_weights(ROUTES*sizeof(float));
  if (!read_exact(argv[2],"hidden.bf16",hidden) ||
      !read_exact(argv[2],"expert_ids.i32",ids) ||
      !read_exact(argv[3],"route_weights.fp32",route_weights)) return 4;
  int valid_by_owner[R]{},expert_counts[E]{};
  for (int token=0;token<R*T;++token) {
    bool distinct[E]{};
    for (int route=0;route<K;++route) {
      int expert=-1;
      std::memcpy(&expert,ids.data()+size_t(token*K+route)*sizeof(int),
                  sizeof(int));
      if (expert<0 || expert>=E || distinct[expert]) return 5;
      distinct[expert]=true;
      ++expert_counts[expert];
      ++valid_by_owner[expert/LOCAL_E];
    }
  }
  for (int expert=0;expert<E;++expert)
    if (expert_counts[expert]<64 || expert_counts[expert]>256) return 6;
  const int expected_owner_rows[R]={4039,4196,4016,4133};
  for (int rank=0;rank<R;++rank)
    if (valid_by_owner[rank]!=expected_owner_rows[rank]) return 7;
  Case scenario[R]{};
  int wave_counts[R][kWaves]{};
  for (int rank=0;rank<R;++rank) {
    scenario[rank].name="controlled_ep4";
    scenario[rank].grid=96;
    scenario[rank].communication=communication_control;
    scenario[rank].budget=steal_control;
  }
  RankState state[R]{};
  int sm_counts[R]{},occupancy[R]{};
  char device_names[R][256]{};
  for (int rank=0;rank<R;++rank) {
    if (check(cudaSetDevice(rank),"select rank")) return 11;
    cudaDeviceProp prop{};
    if (check(cudaGetDeviceProperties(&prop,rank),"device properties") ||
        prop.major!=10 || prop.minor!=3 ||
        (std::strcmp(prop.name,"NVIDIA B300") &&
         std::strcmp(prop.name,"NVIDIA B300 SXM6 AC"))) return 12;
    std::snprintf(device_names[rank],sizeof(device_names[rank]),"%s",prop.name);
    sm_counts[rank]=prop.multiProcessorCount;
    int cooperative=0;
    if (check(cudaDeviceGetAttribute(&cooperative,cudaDevAttrCooperativeLaunch,
                                     rank),"cooperative support") || cooperative!=1)
      return 13;
    if (check(cudaFuncSetAttribute(tile_schedule_probe,
            cudaFuncAttributeMaxDynamicSharedMemorySize,kDynamicShared),
            "worker dynamic shared attribute") ||
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &occupancy[rank],tile_schedule_probe,kThreads,kDynamicShared),
            "worker occupancy") ||
        occupancy[rank]<1 || occupancy[rank]*sm_counts[rank]<96) return 14;
    for (int peer=0;peer<R;++peer) {
      if (peer==rank) continue;
      int access=0,atomic=0;
      if (check(cudaDeviceCanAccessPeer(&access,rank,peer),"peer access") ||
          check(cudaDeviceGetP2PAttribute(&atomic,
            cudaDevP2PAttrNativeAtomicSupported,rank,peer),
            "peer native atomic") || access!=1 || atomic!=1) return 15;
      cudaError_t status=cudaDeviceEnablePeerAccess(peer,0);
      if (status!=cudaSuccess && status!=cudaErrorPeerAccessAlreadyEnabled) {
        check(status,"enable peer access");return 16;
      }
    }
    RankState& s=state[rank];
    if (check(cudaMalloc(&s.bin,sizeof(Bin)),"bin malloc") ||
        check(cudaMalloc(&s.hidden,T*H*2),"hidden malloc") ||
        check(cudaMalloc(&s.ids,T*K*4),"ids malloc") ||
        check(cudaMalloc(&s.tile_keys,TILE_KEY_BYTES),"tile keys malloc") ||
        check(cudaMalloc(&s.tile_experts,kLogicalTiles*4),
              "tile experts malloc") ||
        check(cudaMalloc(&s.tile_input,TILE_BYTES),"tile input malloc") ||
        check(cudaMalloc(&s.up_weight,kExperts*kUpGateWeightBytes),
              "up weight malloc") ||
        check(cudaMalloc(&s.down_weight,kExperts*kDownWeightBytes),
              "down weight malloc") ||
        check(cudaMalloc(&s.upgate,kUpGateBytes),"upgate malloc") ||
        check(cudaMalloc(&s.activated,kActivatedBytes),"activated malloc") ||
        check(cudaMalloc(&s.down,kOutputBytes),"down malloc") ||
        check(cudaMalloc(&s.contributions,LOCAL_ROUTES*H*4),
              "return contribution malloc") ||
        check(cudaMalloc(&s.return_ready,LOCAL_ROUTES*4),
              "return ready malloc") ||
        check(cudaMalloc(&s.route_weights,LOCAL_ROUTES*4),
              "route weights malloc") ||
        check(cudaMalloc(&s.final_output,T*H*2),"final output malloc") ||
        check(cudaMalloc(&s.heads,kStages*kLogicalTiles*4),"heads malloc") ||
        check(cudaMalloc(&s.completed,kStages*kLogicalTiles*4),
              "completed malloc") ||
        check(cudaMalloc(&s.owners,kStages*kWaves*kMaxTasks*4),
              "owner IDs malloc") ||
        check(cudaMalloc(&s.processed,kStages*kWaves*4),
              "processed malloc") ||
        check(cudaMalloc(&s.dispatched,kWaves*4),"dispatched malloc") ||
        check(cudaMalloc(&s.stolen,kStages*kWaves*4),"stolen malloc") ||
        check(cudaMalloc(&s.permits,4),"permits malloc") ||
        check(cudaMalloc(&s.tasks,kStages*kWaves*4),"tasks malloc") ||
        check(cudaMalloc(&s.overlap,2*4),"overlap malloc") ||
        check(cudaMalloc(&s.up_maps_a,kLogicalTiles*sizeof(CUtensorMap)),
              "up A maps malloc") ||
        check(cudaMalloc(&s.up_map_b,kExperts*sizeof(CUtensorMap)),
              "up B maps malloc") ||
        check(cudaMalloc(&s.down_maps_a,kLogicalTiles*sizeof(CUtensorMap)),
              "down A maps malloc") ||
        check(cudaMalloc(&s.down_map_b,kExperts*sizeof(CUtensorMap)),
              "down B maps malloc") ||
        check(cudaMalloc(&s.bin_params,sizeof(BinParams)),
              "bin params malloc") ||
        check(cudaMalloc(&s.return_params,sizeof(ReturnParams)),
              "return params malloc")) return 17;
    std::vector<unsigned char> up_weights(kExperts*kUpGateWeightBytes);
    std::vector<unsigned char> down_weights(kExperts*kDownWeightBytes);
    if (!read_slice(argv[3],"weights_upgate.bf16",
                    size_t(rank)*up_weights.size(),up_weights) ||
        !read_slice(argv[3],"weights_down.bf16",
                    size_t(rank)*down_weights.size(),down_weights))
      return 18;
    if (check(cudaMemset(s.bin,0,sizeof(Bin)),"bin reset") ||
        check(cudaMemset(&s.bin->keys,0xff,sizeof(Bin::keys)),
              "bin key sentinel") ||
        check(cudaMemset(&s.bin->route_location,0xff,
                         sizeof(Bin::route_location)),"location sentinel") ||
        check(cudaMemset(s.tile_keys,0xff,TILE_KEY_BYTES),
              "tile key sentinel") ||
        check(cudaMemset(s.tile_experts,0xff,kLogicalTiles*4),
              "tile expert sentinel") ||
        check(cudaMemset(s.tasks,0,kStages*kWaves*4),
              "stage task reset") ||
        check(cudaMemset(s.tile_input,0xff,TILE_BYTES),"tile sentinel") ||
        check(cudaMemset(s.upgate,0xff,kUpGateBytes),"upgate sentinel") ||
        check(cudaMemset(s.activated,0xff,kActivatedBytes),
              "activation sentinel") ||
        check(cudaMemset(s.down,0xff,kOutputBytes),"down sentinel") ||
        check(cudaMemset(s.contributions,0xff,LOCAL_ROUTES*H*4),
              "contribution sentinel") ||
        check(cudaMemset(s.return_ready,0,LOCAL_ROUTES*4),
              "return ready reset") ||
        check(cudaMemset(s.final_output,0xff,T*H*2),"final sentinel") ||
        check(cudaMemset(s.heads,0,kStages*kLogicalTiles*4),
              "heads reset") ||
        check(cudaMemset(s.completed,0,kStages*kLogicalTiles*4),
              "completed reset") ||
        check(cudaMemset(s.owners,0xff,kStages*kWaves*kMaxTasks*4),
              "owner IDs reset") ||
        check(cudaMemset(s.processed,0,kStages*kWaves*4),
              "processed reset") ||
        check(cudaMemset(s.dispatched,0,kWaves*4),
              "dispatched reset") ||
        check(cudaMemset(s.stolen,0,kStages*kWaves*4),"stolen reset") ||
        check(cudaMemset(s.permits,0,4),"permits reset") ||
        check(cudaMemset(s.overlap,0,2*4),"overlap reset") ||
        check(cudaMemcpy(s.hidden,hidden.data()+size_t(rank)*T*H*2,
                         T*H*2,cudaMemcpyHostToDevice),"hidden H2D") ||
        check(cudaMemcpy(s.ids,ids.data()+size_t(rank)*T*K*4,
                         T*K*4,cudaMemcpyHostToDevice),"ids H2D") ||
        check(cudaMemcpy(s.up_weight,up_weights.data(),up_weights.size(),
                         cudaMemcpyHostToDevice),"up weights H2D") ||
        check(cudaMemcpy(s.down_weight,down_weights.data(),
                         down_weights.size(),cudaMemcpyHostToDevice),
              "down weights H2D") ||
        check(cudaMemcpy(s.route_weights,
                         route_weights.data()+size_t(rank)*LOCAL_ROUTES*4,
                         LOCAL_ROUTES*4,cudaMemcpyHostToDevice),
              "route weights H2D"))
      return 19;
  }
  static_assert(sizeof(CUtensorMap)%64==0,"tensor map array alignment");
  cuuint64_t up_dims_a[2]={kInputWidth,kRows};
  cuuint64_t up_dims_b[2]={kInputWidth,kUpGateWidth};
  cuuint64_t up_stride[1]={kInputWidth*2};
  cuuint64_t down_dims_a[2]={kHidden,kRows};
  cuuint64_t down_dims_b[2]={kHidden,kOutput};
  cuuint64_t down_stride[1]={kHidden*2};
  cuuint32_t box_a[2]={64,kRows},box_b[2]={64,64},steps[2]={1,1};
  for (int rank=0;rank<R;++rank) {
    if (check(cudaSetDevice(rank),"select tensor map rank")) return 20;
    RankState& s=state[rank];
    alignas(128) CUtensorMap up_a[kLogicalTiles],up_b[kExperts];
    alignas(128) CUtensorMap down_a[kLogicalTiles],down_b[kExperts];
    for (int tile=0;tile<kLogicalTiles;++tile) {
      CUresult up=cuTensorMapEncodeTiled(&up_a[tile],
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,
        s.tile_input+tile*kRows*kInputWidth,
        up_dims_a,up_stride,box_a,steps,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      CUresult down=cuTensorMapEncodeTiled(&down_a[tile],
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,
        s.activated+tile*kRows*kHidden,
        down_dims_a,down_stride,box_a,steps,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (up!=CUDA_SUCCESS || down!=CUDA_SUCCESS) return 21;
    }
    for (int expert=0;expert<kExperts;++expert) {
      CUresult up=cuTensorMapEncodeTiled(&up_b[expert],
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,
        s.up_weight+expert*kUpGateWeightBytes/2,
        up_dims_b,up_stride,box_b,steps,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      CUresult down=cuTensorMapEncodeTiled(&down_b[expert],
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,
        s.down_weight+expert*kDownWeightBytes/2,
        down_dims_b,down_stride,box_b,steps,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (up!=CUDA_SUCCESS || down!=CUDA_SUCCESS) return 22;
    }
    if (check(cudaMemcpy(s.up_maps_a,up_a,sizeof(up_a),
                         cudaMemcpyHostToDevice),"up A maps H2D") ||
        check(cudaMemcpy(s.up_map_b,up_b,sizeof(up_b),
                         cudaMemcpyHostToDevice),"up B maps H2D") ||
        check(cudaMemcpy(s.down_maps_a,down_a,sizeof(down_a),
                         cudaMemcpyHostToDevice),"down A maps H2D") ||
        check(cudaMemcpy(s.down_map_b,down_b,sizeof(down_b),
                         cudaMemcpyHostToDevice),"down B maps H2D")) return 23;
  }
  for (int rank=0;rank<R;++rank) {
    if (check(cudaSetDevice(rank),"select params rank")) return 24;
    BinParams bin{};
    ReturnParams returned{};
    for (int peer=0;peer<R;++peer) {
      bin.bins[peer]=state[peer].bin;
      returned.contributions[peer]=state[peer].contributions;
      returned.ready[peer]=state[peer].return_ready;
    }
    bin.hidden=state[rank].hidden;
    bin.ids=state[rank].ids;
    bin.tile_keys=state[rank].tile_keys;
    bin.tile_experts=state[rank].tile_experts;
    bin.tile_rows=state[rank].tile_input;
    bin.rank=rank;
    returned.down=state[rank].down;
    returned.tile_keys=state[rank].tile_keys;
    returned.rank=rank;
    if (check(cudaMemcpy(state[rank].bin_params,&bin,sizeof(bin),
                         cudaMemcpyHostToDevice),"bin params H2D") ||
        check(cudaMemcpy(state[rank].return_params,&returned,
                         sizeof(returned),cudaMemcpyHostToDevice),
              "return params H2D")) return 25;
  }
  // The cooperative worker waits for each wave on the default stream. A
  // nonblocking stream on each rank publishes source chunks, tile manifests,
  // gathered rows and task counts. No host-side synchronization divides them.
  cudaStream_t communication_stream[R]{};
  for (int rank=0;rank<R;++rank) {
    if (check(cudaSetDevice(rank),"select communication stream rank") ||
        check(cudaStreamCreateWithFlags(&communication_stream[rank],
              cudaStreamNonBlocking),"create communication stream")) return 26;
  }
  int communication=communication_control,budget=steal_control;
  for (int rank=0;rank<R;++rank) {
    if (check(cudaSetDevice(rank),"select Cake worker rank")) return 30;
    RankState& s=state[rank];
    int* wave_ready=&s.bin->wave_consumed[0];
    void* args[]={&s.heads,&s.completed,&s.owners,&s.processed,
                  &s.dispatched,&s.stolen,&s.permits,&s.tasks,&wave_ready,&s.overlap,
                  &s.tile_experts,&s.upgate,&s.activated,
                  &s.up_maps_a,&s.up_map_b,&s.down_maps_a,&s.down_map_b,
                  &s.down,&communication,&budget};
    if (check(cudaLaunchCooperativeKernel(
             reinterpret_cast<const void*>(tile_schedule_probe),
             dim3(scenario[rank].grid),dim3(kThreads),args,
             kDynamicShared,nullptr),
             "Cake worker launch")) return 31;
  }
  for (int wave=0;wave<kWaves;++wave) {
    for (int rank=0;rank<R;++rank) {
      if (check(cudaSetDevice(rank),"select source wave rank")) return 27;
      dispatch_source_wave<<<dim3(128,K),256,0,communication_stream[rank]>>>(
          state[rank].bin_params,wave);
      if (check(cudaGetLastError(),"source wave dispatch launch")) return 27;
      mark_source_wave_done<<<1,1,0,communication_stream[rank]>>>(
          state[rank].bin_params,wave);
      if (check(cudaGetLastError(),"source wave publish launch")) return 28;
    }
    for (int rank=0;rank<R;++rank) {
      if (check(cudaSetDevice(rank),"select destination wave rank")) return 29;
      RankState& s=state[rank];
      cudaStream_t stream=communication_stream[rank];
      if (wave==1 || wave==2) {
        derive_early_wave<<<LOCAL_E,256,0,stream>>>(s.bin_params,wave);
        if (check(cudaGetLastError(),"early route sort launch")) return 30;
        assign_early_wave<<<1,LOCAL_E,0,stream>>>(s.bin_params,wave);
        if (check(cudaGetLastError(),"early manifest launch")) return 30;
      } else if (wave==3) {
        derive_terminal_order<<<LOCAL_E,256,0,stream>>>(s.bin_params);
        if (check(cudaGetLastError(),"terminal route sort launch")) return 30;
        assign_terminal_tiles<<<1,LOCAL_E,0,stream>>>(s.bin_params);
        if (check(cudaGetLastError(),"terminal manifest launch")) return 30;
      }
      expand_tasks_for_wave<<<1,kStages,0,stream>>>(s.bin_params,s.tasks,wave);
      if (check(cudaGetLastError(),"wave task expansion launch")) return 30;
      if (wave>0) {
        gather_wave_tiles<<<dim3(LOCAL_E,kRows),256,0,stream>>>(s.bin_params,wave);
        if (check(cudaGetLastError(),"wave gather launch")) return 30;
      }
      publish_wave_ready<<<1,1,0,stream>>>(s.bin_params,wave);
      if (check(cudaGetLastError(),"wave ready publish launch")) return 30;
    }
    // A source cannot begin the next wave until every destination has
    // snapshotted its current expert counts and gathered published tiles.
    for (int rank=0;rank<R;++rank) {
      if (check(cudaSetDevice(rank),"select wave fence rank")) return 30;
      wait_all_destinations<<<1,1,0,communication_stream[rank]>>>(
          state[rank].bin_params,wave);
      if (check(cudaGetLastError(),"wave fence launch")) return 30;
    }
  }
  for (int rank=0;rank<R;++rank) {
    if (check(cudaSetDevice(rank),"select return rank")) return 32;
    scatter_returns<<<dim3(kLogicalTiles,kRows),256>>>(
        state[rank].return_params);
    if (check(cudaGetLastError(),"return launch")) return 33;
  }
  for (int rank=0;rank<R;++rank) {
    if (check(cudaSetDevice(rank),"select combine rank")) return 34;
    wait_returns<<<(LOCAL_ROUTES+255)/256,256>>>(
        state[rank].return_params);
    if (check(cudaGetLastError(),"return acquire launch")) return 35;
    cake_weave_rank512_combine_kernel<<<dim3(T,8),256>>>(
        state[rank].contributions,state[rank].route_weights,
        state[rank].final_output);
    if (check(cudaGetLastError(),"Cake combine launch")) return 36;
  }
  for (int rank=0;rank<R;++rank) {
    if (check(cudaSetDevice(rank),"select completion rank") ||
        check(cudaDeviceSynchronize(),"full chain completion")) return 37;
  }
  std::vector<unsigned char> observed_hidden(HIDDEN_BYTES),observed_ids(IDS_BYTES);
  std::vector<unsigned char> route_contributions(ROUTE_CONTRIBUTION_BYTES);
  std::vector<unsigned char> final_output(FINAL_BYTES);
  std::vector<int> return_flags(ROUTES);
  int stolen_by_rank[R]{},bin_rows_by_rank[R]{},bin_error[R]{};
  int overlap_by_rank[R][2]{};
  for (int rank=0;rank<R;++rank) {
    if (check(cudaSetDevice(rank),"select result rank")) return 38;
    RankState& s=state[rank];
    int counts[LOCAL_E],completed[kStages*kLogicalTiles];
    int owner_ids[kStages*kWaves*kMaxTasks];
    int processed[kStages*kWaves],dispatched[kWaves];
    int stolen[kStages*kWaves],permits=-1;
    if (check(cudaMemcpy(counts,s.bin->count,sizeof(counts),
                         cudaMemcpyDeviceToHost),"bin counts read") ||
        check(cudaMemcpy(&bin_error[rank],&s.bin->error,sizeof(int),
                         cudaMemcpyDeviceToHost),"bin error read") ||
        check(cudaMemcpy(wave_counts[rank],s.bin->wave_counts,
                         sizeof(wave_counts[rank]),cudaMemcpyDeviceToHost),
              "GPU tile-wave counts read") ||
        check(cudaMemcpy(completed,s.completed,sizeof(completed),
                         cudaMemcpyDeviceToHost),"stage completion read") ||
        check(cudaMemcpy(owner_ids,s.owners,sizeof(owner_ids),
                         cudaMemcpyDeviceToHost),"task owners read") ||
        check(cudaMemcpy(processed,s.processed,sizeof(processed),
                         cudaMemcpyDeviceToHost),"processed read") ||
        check(cudaMemcpy(dispatched,s.dispatched,sizeof(dispatched),
                         cudaMemcpyDeviceToHost),"dispatched read") ||
        check(cudaMemcpy(stolen,s.stolen,sizeof(stolen),
                         cudaMemcpyDeviceToHost),"stolen read") ||
        check(cudaMemcpy(&permits,s.permits,sizeof(int),
                         cudaMemcpyDeviceToHost),"permits read") ||
        check(cudaMemcpy(overlap_by_rank[rank],s.overlap,2*sizeof(int),
                         cudaMemcpyDeviceToHost),"overlap read") ||
        check(cudaMemcpy(observed_hidden.data()+size_t(rank)*T*H*2,
                         s.hidden,T*H*2,cudaMemcpyDeviceToHost),
              "hidden read") ||
        check(cudaMemcpy(observed_ids.data()+size_t(rank)*T*K*4,
                         s.ids,T*K*4,cudaMemcpyDeviceToHost),"ids read") ||
        check(cudaMemcpy(route_contributions.data()+
                         size_t(rank)*LOCAL_ROUTES*H*4,
                         s.contributions,LOCAL_ROUTES*H*4,
                         cudaMemcpyDeviceToHost),"returned contributions read") ||
        check(cudaMemcpy(final_output.data()+size_t(rank)*T*H*2,
                         s.final_output,T*H*2,cudaMemcpyDeviceToHost),
              "final output read") ||
        check(cudaMemcpy(return_flags.data()+size_t(rank)*LOCAL_ROUTES,
                         s.return_ready,LOCAL_ROUTES*4,
                         cudaMemcpyDeviceToHost),"return flags read")) return 39;
    if (bin_error[rank]!=0 || overlap_by_rank[rank][0]!=1 ||
        overlap_by_rank[rank][1]!=1) return 40;
    for (int expert=0;expert<LOCAL_E;++expert) {
      if (counts[expert]<0 || counts[expert]>MAX_ROWS) return 41;
      bin_rows_by_rank[rank]+=counts[expert];
    }
    if (bin_rows_by_rank[rank]!=valid_by_owner[rank]) return 42;
    int planned_tiles=0;
    for (int wave=0;wave<kWaves;++wave) {
      if (wave_counts[rank][wave]<0 ||
          wave_counts[rank][wave]>kLogicalTiles) return 42;
      planned_tiles+=wave_counts[rank][wave];
      scenario[rank].tasks[0][wave]=
          wave_counts[rank][wave]*kUpGateTasksPerTile;
      scenario[rank].tasks[1][wave]=
          wave_counts[rank][wave]*kActivationTasksPerTile;
      scenario[rank].tasks[2][wave]=
          wave_counts[rank][wave]*kDownTasksPerTile;
    }
    if (planned_tiles!=kLogicalTiles || wave_counts[rank][0]!=0 ||
        wave_counts[rank][3]!=kExperts) return 42;
    for (int stage=0;stage<kStages;++stage) {
      int units=stage==0 ? kUpGateTasksPerTile :
                stage==1 ? kActivationTasksPerTile : kDownTasksPerTile;
      for (int tile=0;tile<kLogicalTiles;++tile)
        if (completed[stage*kLogicalTiles+tile]!=units) return 43;
      for (int wave=0;wave<kWaves;++wave) {
        int index=stage*kWaves+wave;
        if (processed[index]!=scenario[rank].tasks[stage][wave]) return 44;
        stolen_by_rank[rank]+=stolen[index];
        for (int task=0;task<scenario[rank].tasks[stage][wave];++task) {
          int who=owner_ids[index*kMaxTasks+task];
          if (who<0 || who>=scenario[rank].grid) return 45;
        }
      }
    }
    for (int wave=0;wave<kWaves;++wave)
      if (dispatched[wave]!=scenario[rank].communication) return 46;
    if (stolen_by_rank[rank]<0 ||
        stolen_by_rank[rank]>scenario[rank].budget ||
        permits!=stolen_by_rank[rank] ||
        (scenario[rank].communication==148 &&
         stolen_by_rank[rank]!=kTotalStageTasks))
      return 47;
    std::vector<unsigned char> tile_input(TILE_BYTES),down(kOutputBytes);
    std::vector<unsigned char> manifest_keys(TILE_KEY_BYTES),
                               manifest_experts(kLogicalTiles*sizeof(int));
    if (check(cudaMemcpy(tile_input.data(),s.tile_input,TILE_BYTES,
                         cudaMemcpyDeviceToHost),"tile input read") ||
        check(cudaMemcpy(manifest_keys.data(),s.tile_keys,TILE_KEY_BYTES,
                         cudaMemcpyDeviceToHost),"GPU tile route keys read") ||
        check(cudaMemcpy(manifest_experts.data(),s.tile_experts,
                         manifest_experts.size(),cudaMemcpyDeviceToHost),
              "GPU tile experts read") ||
        check(cudaMemcpy(down.data(),s.down,kOutputBytes,
                         cudaMemcpyDeviceToHost),"down read")) return 48;
    char name[128];
    std::snprintf(name,sizeof(name),"device_outputs/rank%d/tile_input.bf16",rank);
    if (!write_exact(argv[1],name,tile_input.data(),tile_input.size())) return 49;
    std::snprintf(name,sizeof(name),"device_outputs/rank%d/tile_route_keys.i32",rank);
    if (!write_exact(argv[1],name,manifest_keys.data(),
                     manifest_keys.size())) return 49;
    std::snprintf(name,sizeof(name),"device_outputs/rank%d/tile_expert.i32",rank);
    if (!write_exact(argv[1],name,manifest_experts.data(),
                     manifest_experts.size())) return 49;
    std::snprintf(name,sizeof(name),"device_outputs/rank%d/wave_counts.i32",rank);
    if (!write_exact(argv[1],name,wave_counts[rank],
                     sizeof(wave_counts[rank]))) return 49;
    std::snprintf(name,sizeof(name),"device_outputs/rank%d/down.fp32",rank);
    if (!write_exact(argv[1],name,down.data(),down.size())) return 50;
  }
  for (int flag:return_flags) if (flag!=1) return 51;
  if (!write_exact(argv[1],"device_outputs/contributions.fp32",
                   route_contributions.data(),route_contributions.size()) ||
      !write_exact(argv[1],"device_outputs/output.bf16",
                   final_output.data(),final_output.size()) ||
      !write_exact(argv[1],"device_outputs/observed_hidden.bf16",
                   observed_hidden.data(),observed_hidden.size()) ||
      !write_exact(argv[1],"device_outputs/observed_expert_ids.i32",
                   observed_ids.data(),observed_ids.size()) ||
      !write_exact(argv[1],"device_outputs/return_ready.i32",
                   return_flags.data(),return_flags.size()*sizeof(int)))
    return 52;
  char path[512];
  int n=std::snprintf(path,sizeof(path),"%s/device_report.json",argv[1]);
  if (n<1 || n>=int(sizeof(path))) return 53;
  FILE* report=std::fopen(path,"wx");
  if (!report) return 54;
  std::fprintf(report,"{\"target\":\"sm_103a\",\"ranks\":4,"
                      "\"communication_ctas\":%d,\"steal_budget\":%d,"
                      "\"valid_routes_by_owner\":[%d,%d,%d,%d],"
                      "\"bin_rows_by_owner\":[%d,%d,%d,%d],"
                      "\"stolen_by_owner\":[%d,%d,%d,%d],"
                      "\"sm_counts\":[%d,%d,%d,%d],"
                      "\"worker_grid_ctas\":[%d,%d,%d,%d],"
                      "\"active_blocks_per_sm\":[%d,%d,%d,%d],"
                      "\"overlap_flags\":[[%d,%d],[%d,%d],[%d,%d],[%d,%d]],"
                      "\"no_interphase_host_sync\":true,"
                      "\"device_names\":[\"%s\",\"%s\",\"%s\",\"%s\"],"
                      "\"tile_waves_by_owner\":[",
              communication_control,steal_control,
              valid_by_owner[0],valid_by_owner[1],valid_by_owner[2],
              valid_by_owner[3],bin_rows_by_rank[0],bin_rows_by_rank[1],
              bin_rows_by_rank[2],bin_rows_by_rank[3],
              stolen_by_rank[0],stolen_by_rank[1],stolen_by_rank[2],
              stolen_by_rank[3],sm_counts[0],sm_counts[1],sm_counts[2],
              sm_counts[3],scenario[0].grid,scenario[1].grid,
              scenario[2].grid,scenario[3].grid,
              occupancy[0],occupancy[1],occupancy[2],occupancy[3],
              overlap_by_rank[0][0],overlap_by_rank[0][1],
              overlap_by_rank[1][0],overlap_by_rank[1][1],
              overlap_by_rank[2][0],overlap_by_rank[2][1],
              overlap_by_rank[3][0],overlap_by_rank[3][1],
              device_names[0],device_names[1],device_names[2],device_names[3]);
  for (int rank=0;rank<R;++rank) {
    std::fprintf(report,"%s[",rank ? "," : "");
    for (int wave=0;wave<kWaves;++wave)
      std::fprintf(report,"%s%d",wave ? "," : "",wave_counts[rank][wave]);
    std::fprintf(report,"]");
  }
  std::fprintf(report,"]}\n");
  if (std::fclose(report)) return 55;
  for (int rank=0;rank<R;++rank) {
    cudaSetDevice(rank);
    cudaStreamDestroy(communication_stream[rank]);
    RankState& s=state[rank];
    cudaFree(s.return_params);cudaFree(s.bin_params);
    cudaFree(s.down_map_b);cudaFree(s.down_maps_a);
    cudaFree(s.up_map_b);cudaFree(s.up_maps_a);
    cudaFree(s.overlap);cudaFree(s.tasks);cudaFree(s.permits);
    cudaFree(s.stolen);cudaFree(s.dispatched);cudaFree(s.processed);
    cudaFree(s.owners);cudaFree(s.completed);cudaFree(s.heads);
    cudaFree(s.final_output);cudaFree(s.route_weights);
    cudaFree(s.return_ready);cudaFree(s.contributions);
    cudaFree(s.down);cudaFree(s.activated);cudaFree(s.upgate);
    cudaFree(s.down_weight);cudaFree(s.up_weight);
    cudaFree(s.tile_input);cudaFree(s.tile_experts);cudaFree(s.tile_keys);
    cudaFree(s.ids);cudaFree(s.hidden);cudaFree(s.bin);
  }
  std::printf("full four-rank chain: 16384 routes, 64 tiles/rank, 11776 stage tasks/rank\n");
  return 0;
}
