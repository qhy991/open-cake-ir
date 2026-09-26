// B300 source-event development: each source token sends one BF16 payload
// per remote destination rank, and expert-bin rows reference that payload.
// A source completion publishes every newly full tile; the wave-end event
// flushes thresholded partial tiles.
// Each finite cooperative Cake worker uses the 192-thread/49,200-B CTA
// footprint, allocating TMEM at its first task and releasing it on exit.
// The stage body below follows the retained Cake
// tensor-core emissions; activation uses the Cake FP32-to-BF16 graph.
// Per-tile completion counters publish successors with GPU-scope release/
// acquire. Grid barriers delimit stage-task completion within one event.
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
#include <chrono>
#include <new>
#include <thread>
#include <vector>
@COMBINE_SOURCE@

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != @CUDA_ARCH@
#error "Exact @TARGET_ID@ is required"
#endif

namespace cg = cooperative_groups;
namespace {
constexpr int kThreads = @THREADS@;
constexpr int kDynamicShared = @SHARED_BYTES@;
constexpr int kWaves = 4;
constexpr int kSourceRanks = 4;
constexpr int kEvents = kWaves * (kSourceRanks+1);
constexpr int kStages = 3;
constexpr int kLogicalTiles = @TILE_CAPACITY@;
constexpr int kExpectedTiles = 64;  // retained route-set oracle only
constexpr int kExperts = 32;
constexpr int kUpGateTasksPerTile = @UPGATE_UNITS@;
constexpr int kActivationTasksPerTile = @ACTIVATION_UNITS@;
constexpr int kDownTasksPerTile = @DOWN_UNITS@;
constexpr int kMaxTasks = kLogicalTiles*kActivationTasksPerTile;
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
    float* outputs, int communication_ctas, int steal_budget,
    int selected_wave) {
  extern __shared__ __align__(1024) unsigned char shared[];
  uint32_t* tensor_address = reinterpret_cast<uint32_t*>(shared + 49192);
  __shared__ int claimed, claimed_stage, claimed_tile, claimed_subtile;
  __shared__ int borrowed;
  const int block = int(blockIdx.x);
  const int warp = int(threadIdx.x) / 32;
  cg::grid_group grid = cg::this_grid();
  bool tensor_owned = false;
  int logical_offset=0;
  for (int earlier=0;earlier<selected_wave;++earlier)
    logical_offset+=task_counts[earlier]/kUpGateTasksPerTile;
  for (int wave=selected_wave; wave<=selected_wave; ++wave) {
    // CAKE_EFFECT: tile.acquire
    // CAKE_EFFECT: task.acquire
    if (threadIdx.x==0)
      while (worker_wave_acquire(&wave_ready[wave])==0)
        __nanosleep(64);
    __syncthreads();
    const int tile_count=task_counts[wave]/kUpGateTasksPerTile;
    const int total_tasks=task_counts[wave]+task_counts[kEvents+wave]+
                          task_counts[2*kEvents+wave];
    if (threadIdx.x==0 && block<communication_ctas)
      atomicAdd(&dispatched[wave],1);
    grid.sync();  // Source wave becomes visible before any tile claim.
    while (true) {
      if (threadIdx.x==0) {
        const bool comm=block<communication_ctas;
        bool permitted=!comm;
        bool reserved=false;
        if (comm && total_tasks>0 && steal_budget>0) {
          // CAKE_EFFECT: steal.permit
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
                  // CAKE_EFFECT: task.predecessor.acquire
                  completion_acquire(&tile_completed[(stage-1)*kLogicalTiles+tile])
                    !=predecessor) continue;
              int* head=&task_heads[stage*kLogicalTiles+tile];
              int old=atomicAdd(head,0);
              while (old<units) {
                // CAKE_EFFECT: task.reserve
                // CAKE_EFFECT: task.claim
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
          const int preceding=(claimed_stage-1)*kEvents+wave;
          if (atomicAdd(&processed[preceding],0)<task_counts[preceding])
            atomicExch(&overlap[claimed_stage-1],1);
        }
      }
      __syncthreads();
      if (claimed<0) {
        if (threadIdx.x==0) {
          int completed_total=0;
          for (int stage=0;stage<kStages;++stage)
            completed_total+=atomicAdd(&processed[stage*kEvents+wave],0);
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
        const int work_index=stage*kEvents+wave;
        task_owner[work_index*kMaxTasks+claimed]=block;
        // CAKE_EFFECT: task.predecessor.publish
        completion_release_increment(
            &tile_completed[stage*kLogicalTiles+tile]);
        atomicAdd(&processed[work_index],1);
        // CAKE_EFFECT: steal.account
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

int check(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) return 0;
  std::fprintf(stderr, "%s: %s (%d)\n", operation,
               cudaGetErrorString(status), int(status));
  return 1;
}

constexpr int R=4,T=512,K=8,E=128,H=2048;
static_assert(R==kSourceRanks,"source-event count follows the EP world size");
constexpr int LOCAL_E=E/R,MAX_ROWS=R*T,ROUTES=R*T*K,LOCAL_ROUTES=T*K;
constexpr int PAYLOAD_CAP=(R-1)*T;
constexpr size_t HIDDEN_BYTES=size_t(R)*T*H*sizeof(uint16_t);
constexpr size_t IDS_BYTES=size_t(ROUTES)*sizeof(int);
constexpr size_t TILE_BYTES=size_t(kLogicalTiles)*kRows*H*sizeof(uint16_t);
constexpr size_t TILE_KEY_BYTES=size_t(kLogicalTiles)*kRows*sizeof(int);
constexpr size_t ROUTE_CONTRIBUTION_BYTES=size_t(ROUTES)*H*sizeof(float);
constexpr size_t FINAL_BYTES=size_t(R)*T*H*sizeof(uint16_t);

struct Bin {
  int count[LOCAL_E];
  int payload_count;
  int payload_ready[PAYLOAD_CAP];
  int payload_key[PAYLOAD_CAP];
  int source_wave_done[R][kWaves];
  int wave_consumed[kEvents];
  int ready[LOCAL_E*MAX_ROWS];
  int keys[LOCAL_E*MAX_ROWS];
  int row_payload_slot[LOCAL_E*MAX_ROWS];
  int route_location[ROUTES];
  int route_ready[ROUTES];
  uint32_t sorted_order[LOCAL_E*MAX_ROWS];
  int consumed_rows[LOCAL_E];
  int wave_row_start[LOCAL_E];
  int wave_full_tiles[LOCAL_E];
  int wave_partial_rows[LOCAL_E];
  int wave_tile_counts[LOCAL_E];
  int wave_counts[kEvents];
  __nv_bfloat16 payload[size_t(PAYLOAD_CAP)*H];
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
__device__ __forceinline__ uint32_t route_order(int key,int chunk_tokens) {
  int source=key/(T*K),token=(key/K)%T,route=key%K;
  return uint32_t((token/chunk_tokens)*(R*chunk_tokens*K)
                  +source*(chunk_tokens*K)
                  +(token%chunk_tokens)*K+route);
}
__device__ __forceinline__ int original_route_key(
    uint32_t ordered,int chunk_tokens) {
  int wave=int(ordered)/(R*chunk_tokens*K);
  int remainder=int(ordered)%(R*chunk_tokens*K);
  int source=remainder/(chunk_tokens*K);
  int token=(remainder%(chunk_tokens*K))/K;
  int route=remainder%K;
  return (source*T+wave*chunk_tokens+token)*K+route;
}
__global__ void dispatch_source_wave(
    const BinParams* params,int wave,int chunk_tokens) {
  int token=wave*chunk_tokens+int(blockIdx.x);
  __shared__ int owner_slot[R],route_expert[K],route_owner[K],valid;
  if (threadIdx.x==0) {
    valid=1;
    for (int owner=0;owner<R;++owner) owner_slot[owner]=-2;
    for (int route=0;route<K;++route) {
      int expert=params->ids[token*K+route];
      if (expert<0 || expert>=E) {
        atomicCAS(&params->bins[params->rank]->error,0,11);
        valid=0;break;
      }
      route_expert[route]=expert;
      route_owner[route]=expert/LOCAL_E;
      owner_slot[route_owner[route]]=-1;
    }
    if (valid)
      for (int owner=0;owner<R;++owner) {
        if (owner==params->rank || owner_slot[owner]!=-1) continue;
        Bin* remote=params->bins[owner];
        // CAKE_EFFECT: payload.reserve
        int slot=system_reserve(&remote->payload_count);
        if (slot<0 || slot>=PAYLOAD_CAP) {
          atomicCAS(&params->bins[params->rank]->error,0,12);
          valid=0;break;
        }
        owner_slot[owner]=slot;
        remote->payload_key[slot]=params->rank*T+token;
      }
  }
  __syncthreads();
  if (!valid) return;
  const __nv_bfloat16* input=params->hidden+size_t(token)*H;
  for (int owner=0;owner<R;++owner) {
    int slot=owner_slot[owner];
    if (slot<0) continue;
    Bin* remote=params->bins[owner];
    __nv_bfloat16* output=remote->payload+size_t(slot)*H;
    for (int feature=int(threadIdx.x);feature<H;feature+=int(blockDim.x))
      output[feature]=input[feature];
    __syncthreads();
    // CAKE_EFFECT: payload.publish
    if (threadIdx.x==0) system_publish(&remote->payload_ready[slot]);
    __syncthreads();
  }
  if (threadIdx.x==0) {
    for (int route=0;route<K;++route) {
      int expert=route_expert[route],owner=route_owner[route];
      Bin* remote=params->bins[owner];
      // CAKE_EFFECT: bin.reserve
      int slot=system_reserve(&remote->count[expert%LOCAL_E]);
      if (slot<0 || slot>=MAX_ROWS) {
        atomicCAS(&params->bins[params->rank]->error,0,13);
        continue;
      }
      int index=(expert%LOCAL_E)*MAX_ROWS+slot;
      int key=(params->rank*T+token)*K+route;
      remote->keys[index]=key;
      remote->route_location[key]=index;
      remote->row_payload_slot[index]=owner_slot[owner];
      // CAKE_EFFECT: bin.publish
      system_publish(&remote->ready[index]);
      system_publish(&remote->route_ready[key]);
    }
  }
}
__global__ void mark_source_wave_done(const BinParams* params,int wave) {
  if (threadIdx.x!=0) return;
  // CAKE_EFFECT: source.complete
  for (int owner=0;owner<R;++owner)
    system_publish(&params->bins[owner]->source_wave_done[params->rank][wave]);
}
__device__ __forceinline__ void sort_expert_snapshot(
    Bin* local,int expert,int lane,int count,uint32_t* order,
    int chunk_tokens) {
  for (int index=lane;index<MAX_ROWS;index+=int(blockDim.x))
    order[index]=index<count
        ? route_order(local->keys[expert*MAX_ROWS+index],chunk_tokens)
        : 0xffffffffu;
  __syncthreads();
  for (int width=2;width<=MAX_ROWS;width<<=1) {
    for (int span=width>>1;span>0;span>>=1) {
      for (int index=lane;index<MAX_ROWS;index+=int(blockDim.x)) {
        int peer=index^span;
        if (peer>index) {
          uint32_t left=order[index],right=order[peer];
          bool ascending=(index&width)==0;
          if ((ascending && left>right) || (!ascending && left<right)) {
            order[index]=right;
            order[peer]=left;
          }
        }
      }
      __syncthreads();
    }
  }
}
__global__ void derive_wave_order(const BinParams* params,int wave,
                                  int source,int chunks,int chunk_tokens) {
  int expert=int(blockIdx.x),lane=int(threadIdx.x);
  Bin* local=params->bins[params->rank];
  if (lane==0) {
    int needed=source<R ? source+1 : R;
    for (int peer=source<R ? source : 0;peer<needed;++peer)
      while (system_acquire(&local->source_wave_done[peer][wave])==0)
        __nanosleep(64);
  }
  __syncthreads();
  int count=local->count[expert];
  int consumed=local->consumed_rows[expert];
  if (count<consumed || count>MAX_ROWS) {
    if (lane==0) atomicCAS(&local->error,0,13);
    return;
  }
  __shared__ uint32_t order[MAX_ROWS];
  sort_expert_snapshot(local,expert,lane,count,order,chunk_tokens);
  if (lane==0) {
    if (count>0 && order[count-1]/(R*chunk_tokens*K)>uint32_t(wave)) {
      atomicCAS(&local->error,0,14);return;
    }
    int available=count-consumed;
    int full=available/kRows;
    int tail=available%kRows;
    int partial=(source==R && (wave==chunks-1 || tail>=64)) ? tail : 0;
    local->wave_row_start[expert]=consumed;
    local->wave_full_tiles[expert]=full;
    local->wave_partial_rows[expert]=partial;
    local->wave_tile_counts[expert]=full+(partial>0);
    local->consumed_rows[expert]=consumed+full*kRows+partial;
  }
  __syncthreads();
  for (int index=consumed+lane;index<count;index+=int(blockDim.x))
    local->sorted_order[expert*MAX_ROWS+index]=order[index];
}
__global__ void assign_wave_tiles(const BinParams* params,int event,
                                  int chunk_tokens) {
  int expert=int(threadIdx.x);
  Bin* local=params->bins[params->rank];
  if (expert>=LOCAL_E) return;
  int base=0,prior=0,total=0;
  for (int earlier=0;earlier<event;++earlier)
    base+=local->wave_counts[earlier];
  for (int other=0;other<LOCAL_E;++other) {
    int count=local->wave_tile_counts[other];
    if (other<expert) prior+=count;
    total+=count;
  }
  if (base+total>kLogicalTiles) {
    if (expert==0) atomicCAS(&local->error,0,15);
    return;
  }
  int full=local->wave_full_tiles[expert];
  int partial=local->wave_partial_rows[expert];
  int start=local->wave_row_start[expert];
  for (int piece=0;piece<local->wave_tile_counts[expert];++piece) {
    int slot=base+prior+piece;
    int valid=piece<full ? kRows : partial;
    // CAKE_EFFECT: tile.construct
    params->tile_experts[slot]=expert;
    for (int row=0;row<kRows;++row)
      params->tile_keys[slot*kRows+row]=row<valid
          ? original_route_key(local->sorted_order[
              expert*MAX_ROWS+start+piece*kRows+row],chunk_tokens) : -1;
  }
  if (expert==0) local->wave_counts[event]=total;
}
__global__ void expand_tasks_for_wave(const BinParams* params,int* tasks,
                                      int event) {
  int stage=int(threadIdx.x);
  if (stage>=kStages) return;
  int factor=stage==0 ? kUpGateTasksPerTile :
             stage==1 ? kActivationTasksPerTile : kDownTasksPerTile;
  tasks[stage*kEvents+event]=
      params->bins[params->rank]->wave_counts[event]*factor;
}
__global__ void gather_wave_tiles(const BinParams* params,int event,
                                  int wave,int source) {
  Bin* local=params->bins[params->rank];
  if (threadIdx.x==0) {
    int needed=source<R ? source+1 : R;
    for (int peer=source<R ? source : 0;peer<needed;++peer)
      while (system_acquire(&local->source_wave_done[peer][wave])==0)
        __nanosleep(64);
  }
  __syncthreads();
  int count=local->wave_counts[event];
  int offset=0;
  for (int earlier=0;earlier<event;++earlier)
    offset+=local->wave_counts[earlier];
  for (int logical=int(blockIdx.x);logical<count;logical+=LOCAL_E) {
    int tile=offset+logical,row=int(blockIdx.y);
    int key=params->tile_keys[tile*kRows+row];
    __nv_bfloat16* output=params->tile_rows+(size_t(tile)*kRows+row)*H;
    if (key<0) {
      for (int feature=int(threadIdx.x);feature<H;feature+=int(blockDim.x))
        output[feature]=__float2bfloat16_rn(0.0f);
      continue;
    }
    if (key>=ROUTES) asm volatile("trap;");
    // CAKE_EFFECT: bin.acquire
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
    int source_rank=key/(T*K),token=(key/K)%T;
    int payload_slot=local->row_payload_slot[location];
    const __nv_bfloat16* input=nullptr;
    if (source_rank==params->rank) {
      if (payload_slot!=-1) {
        if (threadIdx.x==0) atomicCAS(&local->error,0,19);
        return;
      }
      input=params->hidden+size_t(token)*H;
    } else {
      if (payload_slot<0 || payload_slot>=PAYLOAD_CAP) {
        if (threadIdx.x==0) atomicCAS(&local->error,0,20);
        return;
      }
      // CAKE_EFFECT: payload.acquire
      if (threadIdx.x==0)
        while (system_acquire(&local->payload_ready[payload_slot])==0)
          __nanosleep(64);
      __syncthreads();
      if (local->payload_key[payload_slot]!=source_rank*T+token) {
        if (threadIdx.x==0) atomicCAS(&local->error,0,21);
        return;
      }
      input=local->payload+size_t(payload_slot)*H;
    }
    for (int feature=int(threadIdx.x);feature<H;feature+=int(blockDim.x))
      output[feature]=input[feature];
  }
}
__global__ void publish_wave_ready(const BinParams* params,int event,
                                   int wave,int source) {
  if (threadIdx.x!=0) return;
  Bin* local=params->bins[params->rank];
  int needed=source<R ? source+1 : R;
  for (int peer=source<R ? source : 0;peer<needed;++peer)
    while (system_acquire(&local->source_wave_done[peer][wave])==0)
      __nanosleep(64);
  // CAKE_EFFECT: tile.publish
  // CAKE_EFFECT: task.publish
  system_publish(&local->wave_consumed[event]);
}
__global__ void wait_all_destinations(const BinParams* params,int event) {
  if (threadIdx.x!=0) return;
  for (int owner=0;owner<R;++owner)
    while (system_acquire(&params->bins[owner]->wave_consumed[event])==0)
      __nanosleep(64);
}
__global__ void scatter_returns(const ReturnParams* params) {
  int tile=int(blockIdx.x),row=int(blockIdx.y);
  int key=params->tile_keys[tile*kRows+row];
  if (key<0) return;
  if (key>=ROUTES) asm volatile("trap;");
  // CAKE_EFFECT: return.index
  int source=key/LOCAL_ROUTES,slot=key%LOCAL_ROUTES;
  const float* input=params->down+(size_t(tile)*kRows+row)*H;
  float* output=params->contributions[source]+size_t(slot)*H;
  for (int feature=int(threadIdx.x);feature<H;feature+=int(blockDim.x))
    output[feature]=input[feature];
  __syncthreads();
  // CAKE_EFFECT: return.publish
  if (threadIdx.x==0) system_publish(params->ready[source]+slot);
}
__global__ void wait_returns(const ReturnParams* params) {
  int slot=int(blockIdx.x)*int(blockDim.x)+int(threadIdx.x);
  if (slot>=LOCAL_ROUTES) return;
  // CAKE_EFFECT: return.acquire
  while (system_acquire(params->ready[params->rank]+slot)==0)
    __nanosleep(64);
}
