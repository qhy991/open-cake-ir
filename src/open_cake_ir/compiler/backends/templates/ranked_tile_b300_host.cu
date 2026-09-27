// Tensor-pointer ABI for the exact B300 ranked-tile source-event protocol.
// Input and output allocations belong to the caller. This wrapper owns only
// rank-local scratch, peer bins, descriptors and one compute stream.
struct RankedTileRankState {
  int rank=-1;
  const __nv_bfloat16 *hidden{},*up_weight{},*down_weight{};
  const int* ids{};
  const float* route_weights{};
  __nv_bfloat16* output{};
  Bin* bin{};
  int *tile_keys{},*tile_experts{},*heads{},*completed{},*owners{};
  int *processed{},*dispatched{},*stolen{},*permits{},*tasks{},*overlap{};
  int* return_ready{};
  float *upgate{},*down{},*contributions{};
  __nv_bfloat16 *tile_input{},*activated{};
  CUtensorMap *up_maps_a{},*up_map_b{},*down_maps_a{},*down_map_b{};
  BinParams* bin_params{};
  ReturnParams* return_params{};
  cudaStream_t compute{};
};
struct RankedTileHostState {
  RankedTileRankState ranks[R];
  bool poisoned=false;
  int completed_launches=0;
};

void ranked_tile_release(RankedTileHostState* state) {
  if (!state) return;
  for (int rank=0;rank<R;++rank) {
    RankedTileRankState& s=state->ranks[rank];
    if (s.rank!=rank) continue;
    cudaSetDevice(rank);
    cudaDeviceSynchronize();
    if (s.compute) cudaStreamDestroy(s.compute);
    void* allocated[] = {
      s.return_params,s.bin_params,s.down_map_b,s.down_maps_a,
      s.up_map_b,s.up_maps_a,s.overlap,s.tasks,s.permits,s.stolen,
      s.dispatched,s.processed,s.owners,s.completed,s.heads,
      s.return_ready,s.contributions,s.down,s.activated,s.upgate,
      s.tile_input,s.tile_experts,s.tile_keys,s.bin,
    };
    for (void* pointer:allocated) if (pointer) cudaFree(pointer);
  }
  delete state;
}

cudaError_t ranked_tile_check_pointer(const void* pointer,int rank) {
  if (!pointer) return cudaErrorInvalidValue;
  cudaPointerAttributes attributes{};
  cudaError_t error=cudaPointerGetAttributes(&attributes,pointer);
  if (error!=cudaSuccess) return error;
  return attributes.type==cudaMemoryTypeDevice && attributes.device==rank
      ? cudaSuccess : cudaErrorInvalidDevicePointer;
}

cudaError_t ranked_tile_allocate(RankedTileRankState& s) {
  cudaError_t error;
#define CAKE_ALLOC(pointer, bytes) \
  do { error=cudaMalloc(&(pointer),(bytes)); if (error!=cudaSuccess) return error; } while (0)
  CAKE_ALLOC(s.bin,sizeof(Bin));
  CAKE_ALLOC(s.tile_keys,TILE_KEY_BYTES);
  CAKE_ALLOC(s.tile_experts,kLogicalTiles*sizeof(int));
  CAKE_ALLOC(s.tile_input,TILE_BYTES);
  CAKE_ALLOC(s.upgate,kUpGateBytes);
  CAKE_ALLOC(s.activated,kActivatedBytes);
  CAKE_ALLOC(s.down,kOutputBytes);
  CAKE_ALLOC(s.contributions,LOCAL_ROUTES*H*sizeof(float));
  CAKE_ALLOC(s.return_ready,LOCAL_ROUTES*sizeof(int));
  CAKE_ALLOC(s.heads,kStages*kLogicalTiles*sizeof(int));
  CAKE_ALLOC(s.completed,kStages*kLogicalTiles*sizeof(int));
  CAKE_ALLOC(s.owners,size_t(kStages)*kEvents*kMaxTasks*sizeof(int));
  CAKE_ALLOC(s.processed,kStages*kEvents*sizeof(int));
  CAKE_ALLOC(s.dispatched,kEvents*sizeof(int));
  CAKE_ALLOC(s.stolen,kStages*kEvents*sizeof(int));
  CAKE_ALLOC(s.permits,sizeof(int));
  CAKE_ALLOC(s.tasks,kStages*kEvents*sizeof(int));
  CAKE_ALLOC(s.overlap,(kEvents+2)*sizeof(int));
  CAKE_ALLOC(s.up_maps_a,kLogicalTiles*sizeof(CUtensorMap));
  CAKE_ALLOC(s.up_map_b,kExperts*sizeof(CUtensorMap));
  CAKE_ALLOC(s.down_maps_a,kLogicalTiles*sizeof(CUtensorMap));
  CAKE_ALLOC(s.down_map_b,kExperts*sizeof(CUtensorMap));
  CAKE_ALLOC(s.bin_params,sizeof(BinParams));
  CAKE_ALLOC(s.return_params,sizeof(ReturnParams));
#undef CAKE_ALLOC
  error=cudaStreamCreateWithFlags(&s.compute,cudaStreamNonBlocking);
  if (error!=cudaSuccess) return error;
  return cudaSuccess;
}

cudaError_t ranked_tile_bind(RankedTileHostState& state,int rank) {
  RankedTileRankState& s=state.ranks[rank];
  BinParams bin{};
  ReturnParams returned{};
  for (int peer=0;peer<R;++peer) {
    bin.bins[peer]=state.ranks[peer].bin;
    returned.contributions[peer]=state.ranks[peer].contributions;
    returned.ready[peer]=state.ranks[peer].return_ready;
  }
  bin.hidden=s.hidden;bin.ids=s.ids;
  bin.tile_keys=s.tile_keys;bin.tile_experts=s.tile_experts;
  bin.tile_rows=s.tile_input;bin.rank=rank;
  returned.down=s.down;returned.tile_keys=s.tile_keys;returned.rank=rank;
  cudaError_t error=cudaMemcpy(s.bin_params,&bin,sizeof(bin),cudaMemcpyHostToDevice);
  if (error!=cudaSuccess) return error;
  error=cudaMemcpy(s.return_params,&returned,sizeof(returned),cudaMemcpyHostToDevice);
  if (error!=cudaSuccess) return error;
  static_assert(sizeof(CUtensorMap)%64==0,"tensor maps need hardware alignment");
  cuuint64_t up_dims_a[2]={kInputWidth,kRows};
  cuuint64_t up_dims_b[2]={kInputWidth,kUpGateWidth};
  cuuint64_t up_stride[1]={kInputWidth*2};
  cuuint64_t down_dims_a[2]={kHidden,kRows};
  cuuint64_t down_dims_b[2]={kHidden,kOutput};
  cuuint64_t down_stride[1]={kHidden*2};
  cuuint32_t box_a[2]={64,kRows},box_b[2]={64,64},steps[2]={1,1};
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
    if (up!=CUDA_SUCCESS || down!=CUDA_SUCCESS) return cudaErrorInvalidValue;
  }
  for (int expert=0;expert<kExperts;++expert) {
    // The Driver tensor-map API takes a mutable address even for operands
    // whose Cake buffers are INPUT and are only read by TMA.
    CUresult up=cuTensorMapEncodeTiled(&up_b[expert],
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,
      const_cast<__nv_bfloat16*>(s.up_weight+expert*kUpGateWeightBytes/2),
      up_dims_b,up_stride,box_b,steps,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUresult down=cuTensorMapEncodeTiled(&down_b[expert],
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,
      const_cast<__nv_bfloat16*>(s.down_weight+expert*kDownWeightBytes/2),
      down_dims_b,down_stride,box_b,steps,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (up!=CUDA_SUCCESS || down!=CUDA_SUCCESS) return cudaErrorInvalidValue;
  }
  error=cudaMemcpy(s.up_maps_a,up_a,sizeof(up_a),cudaMemcpyHostToDevice);
  if (error!=cudaSuccess) return error;
  error=cudaMemcpy(s.up_map_b,up_b,sizeof(up_b),cudaMemcpyHostToDevice);
  if (error!=cudaSuccess) return error;
  error=cudaMemcpy(s.down_maps_a,down_a,sizeof(down_a),cudaMemcpyHostToDevice);
  if (error!=cudaSuccess) return error;
  return cudaMemcpy(s.down_map_b,down_b,sizeof(down_b),cudaMemcpyHostToDevice);
}

cudaError_t ranked_tile_validate_routes(const RankedTileRankState& s) {
  // CAKE_EFFECT: input_domain.admit
  int host_ids[T*K];
  cudaError_t error=cudaMemcpy(host_ids,s.ids,sizeof(host_ids),
                               cudaMemcpyDeviceToHost);
  if (error!=cudaSuccess) return error;
  for (int token=0;token<T;++token) {
    for (int route=0;route<K;++route) {
      int expert=host_ids[token*K+route];
      if (expert<0 || expert>=E) return cudaErrorInvalidValue;
      for (int prior=0;prior<route;++prior)
        if (host_ids[token*K+prior]==expert) return cudaErrorInvalidValue;
    }
  }
  return cudaSuccess;
}

cudaError_t ranked_tile_reset(RankedTileRankState& s) {
  // CAKE_EFFECT: state.reset
  cudaError_t error;
#define CAKE_CLEAR(pointer, bytes, value) \
  do { error=cudaMemsetAsync((pointer),(value),(bytes),s.compute); \
       if (error!=cudaSuccess) return error; } while (0)
  CAKE_CLEAR(s.bin,sizeof(Bin),0);
  CAKE_CLEAR(&s.bin->keys,sizeof(Bin::keys),0xff);
  CAKE_CLEAR(&s.bin->payload_key,sizeof(Bin::payload_key),0xff);
  CAKE_CLEAR(&s.bin->row_payload_slot,sizeof(Bin::row_payload_slot),0xff);
  CAKE_CLEAR(&s.bin->route_location,sizeof(Bin::route_location),0xff);
  CAKE_CLEAR(s.tile_keys,TILE_KEY_BYTES,0xff);
  CAKE_CLEAR(s.tile_experts,kLogicalTiles*sizeof(int),0xff);
  CAKE_CLEAR(s.tasks,kStages*kEvents*sizeof(int),0);
  CAKE_CLEAR(s.tile_input,TILE_BYTES,0xff);
  CAKE_CLEAR(s.upgate,kUpGateBytes,0xff);
  CAKE_CLEAR(s.activated,kActivatedBytes,0xff);
  CAKE_CLEAR(s.down,kOutputBytes,0xff);
  CAKE_CLEAR(s.contributions,LOCAL_ROUTES*H*sizeof(float),0xff);
  CAKE_CLEAR(s.return_ready,LOCAL_ROUTES*sizeof(int),0);
  CAKE_CLEAR(s.output,T*H*sizeof(uint16_t),0xff);
  CAKE_CLEAR(s.heads,kStages*kLogicalTiles*sizeof(int),0);
  CAKE_CLEAR(s.completed,kStages*kLogicalTiles*sizeof(int),0);
  CAKE_CLEAR(s.owners,size_t(kStages)*kEvents*kMaxTasks*sizeof(int),0xff);
  CAKE_CLEAR(s.processed,kStages*kEvents*sizeof(int),0);
  CAKE_CLEAR(s.dispatched,kEvents*sizeof(int),0);
  CAKE_CLEAR(s.stolen,kStages*kEvents*sizeof(int),0);
  CAKE_CLEAR(s.permits,sizeof(int),0);
  CAKE_CLEAR(s.overlap,(kEvents+2)*sizeof(int),0);
#undef CAKE_CLEAR
  return cudaSuccess;
}
} // namespace

extern "C" int @ENTRY@_abi_version() { return 4; }
extern "C" int @ENTRY@_ranks() { return R; }
extern "C" int @ENTRY@_source_events() { return kEvents; }
extern "C" size_t @ENTRY@_bin_bytes() { return sizeof(Bin); }
extern "C" size_t @ENTRY@_output_bytes() {
  return size_t(T)*H*sizeof(__nv_bfloat16);
}

extern "C" int @ENTRY@_create(
    const void** hidden,const void** ids,const void** route_weights,
    const void** up_weight,const void** down_weight,void** output,
    void** result) {
  if (!hidden || !ids || !route_weights || !up_weight || !down_weight ||
      !output || !result) return int(cudaErrorInvalidValue);
  *result=nullptr;
  int devices=0;
  cudaError_t error=cudaGetDeviceCount(&devices);
  if (error!=cudaSuccess || devices!=R) return int(cudaErrorInvalidDevice);
  auto* state=new (std::nothrow) RankedTileHostState{};
  if (!state) return int(cudaErrorMemoryAllocation);
  for (int rank=0;rank<R;++rank) {
    RankedTileRankState& s=state->ranks[rank];
    s.rank=rank;
    error=cudaSetDevice(rank);
    if (error!=cudaSuccess) break;
    cudaDeviceProp prop{};
    error=cudaGetDeviceProperties(&prop,rank);
    if (error!=cudaSuccess) break;
    if (prop.major!=@CC_MAJOR@ || prop.minor!=@CC_MINOR@ ||
        prop.multiProcessorCount!=@SMS@ || (@DEVICE_NAME_CHECK@)) {
      error=cudaErrorInvalidDevice;break;
    }
    int cooperative=0,resident=0;
    error=cudaDeviceGetAttribute(&cooperative,cudaDevAttrCooperativeLaunch,rank);
    if (error!=cudaSuccess || cooperative!=1) {
      error=cudaErrorNotSupported;break;
    }
    error=cudaFuncSetAttribute(tile_schedule_probe,
        cudaFuncAttributeMaxDynamicSharedMemorySize,kDynamicShared);
    if (error!=cudaSuccess) break;
    error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident,tile_schedule_probe,kThreads,kDynamicShared);
    if (error!=cudaSuccess || resident*prop.multiProcessorCount<96) {
      error=cudaErrorCooperativeLaunchTooLarge;break;
    }
    const void* local[]={hidden[rank],ids[rank],route_weights[rank],
                         up_weight[rank],down_weight[rank],output[rank]};
    for (const void* pointer:local) {
      error=ranked_tile_check_pointer(pointer,rank);
      if (error!=cudaSuccess) break;
    }
    if (error!=cudaSuccess) break;
    for (int peer=0;peer<R;++peer) {
      if (peer==rank) continue;
      int access=0,atomic=0;
      // CAKE_EFFECT: peer_pair.admit
      error=cudaDeviceCanAccessPeer(&access,rank,peer);
      if (error!=cudaSuccess || access!=1) {
        error=cudaErrorNotSupported;break;
      }
      error=cudaDeviceGetP2PAttribute(&atomic,
          cudaDevP2PAttrNativeAtomicSupported,rank,peer);
      if (error!=cudaSuccess || atomic!=1) {
        error=cudaErrorNotSupported;break;
      }
      error=cudaDeviceEnablePeerAccess(peer,0);
      if (error==cudaErrorPeerAccessAlreadyEnabled) {
        // Reusing a process is valid, but the handled error remains in this
        // host thread's CUDA error slot until cudaGetLastError clears it.
        // Otherwise the next kernel-launch check mistakes 704 for its launch.
        cudaError_t pending=cudaGetLastError();
        error=(pending==cudaSuccess ||
               pending==cudaErrorPeerAccessAlreadyEnabled)
            ? cudaSuccess : pending;
      }
      if (error!=cudaSuccess) break;
    }
    if (error!=cudaSuccess) break;
    s.hidden=static_cast<const __nv_bfloat16*>(hidden[rank]);
    s.ids=static_cast<const int*>(ids[rank]);
    s.route_weights=static_cast<const float*>(route_weights[rank]);
    s.up_weight=static_cast<const __nv_bfloat16*>(up_weight[rank]);
    s.down_weight=static_cast<const __nv_bfloat16*>(down_weight[rank]);
    s.output=static_cast<__nv_bfloat16*>(output[rank]);
    error=ranked_tile_allocate(s);
    if (error!=cudaSuccess) break;
  }
  if (error==cudaSuccess)
    for (int rank=0;rank<R;++rank) {
      error=cudaSetDevice(rank);
      if (error!=cudaSuccess) break;
      error=ranked_tile_bind(*state,rank);
      if (error!=cudaSuccess) break;
    }
  if (error!=cudaSuccess) {
    ranked_tile_release(state);
    return int(error);
  }
  *result=state;
  return 0;
}

extern "C" int @ENTRY@_destroy(void* opaque) {
  if (!opaque) return int(cudaErrorInvalidValue);
  auto* state=static_cast<RankedTileHostState*>(opaque);
  // A failed partial multi-rank enqueue can leave device waiters live.
  // The isolated evaluator process must exit and let CUDA tear down those
  // contexts; freeing the buffers underneath the waiters is unsafe.
  if (state->poisoned) return int(cudaErrorNotReady);
  ranked_tile_release(state);
  return 0;
}

extern "C" int @ENTRY@_stolen(void* opaque,int rank,int* result) {
  auto* state=static_cast<RankedTileHostState*>(opaque);
  if (!state || state->poisoned || state->completed_launches<1 ||
      rank<0 || rank>=R || !result) return int(cudaErrorInvalidValue);
  cudaError_t error=cudaSetDevice(rank);
  if (error!=cudaSuccess) return int(error);
  int observed[kStages*kEvents];
  error=cudaMemcpy(observed,state->ranks[rank].stolen,sizeof(observed),
                   cudaMemcpyDeviceToHost);
  if (error!=cudaSuccess) return int(error);
  int total=0;
  for (int count:observed) {
    if (count<0) return int(cudaErrorUnknown);
    total+=count;
  }
  *result=total;
  return 0;
}

extern "C" int @ENTRY@_payloads(void* opaque,int rank,int* result) {
  auto* state=static_cast<RankedTileHostState*>(opaque);
  if (!state || state->poisoned || state->completed_launches<1 ||
      rank<0 || rank>=R || !result) return int(cudaErrorInvalidValue);
  cudaError_t error=cudaSetDevice(rank);
  if (error!=cudaSuccess) return int(error);
  error=cudaMemcpy(result,&state->ranks[rank].bin->payload_count,
                   sizeof(int),cudaMemcpyDeviceToHost);
  if (error!=cudaSuccess) return int(error);
  return *result>=0 && *result<=PAYLOAD_CAP ? 0 : int(cudaErrorUnknown);
}

extern "C" int @ENTRY@_tile_counts(void* opaque,int rank,int* result,
                                    int capacity) {
  auto* state=static_cast<RankedTileHostState*>(opaque);
  if (!state || state->poisoned || state->completed_launches<1 ||
      rank<0 || rank>=R || !result || capacity!=kEvents)
    return int(cudaErrorInvalidValue);
  cudaError_t error=cudaSetDevice(rank);
  if (error!=cudaSuccess) return int(error);
  error=cudaMemcpy(result,state->ranks[rank].bin->wave_counts,
                   kEvents*sizeof(int),cudaMemcpyDeviceToHost);
  if (error!=cudaSuccess) return int(error);
  int total=0;
  for (int event=0;event<kEvents;++event) {
    if (result[event]<0 || result[event]>kLogicalTiles)
      return int(cudaErrorUnknown);
    total+=result[event];
  }
  return total<=kLogicalTiles ? 0 : int(cudaErrorUnknown);
}

extern "C" int @ENTRY@_launch(void* opaque,const int* communication_ctas,
                               const int* steal_budgets,
                               const int* chunks_by_rank) {
  auto* state=static_cast<RankedTileHostState*>(opaque);
  if (!state || state->poisoned || !communication_ctas || !steal_budgets ||
      !chunks_by_rank)
    return int(cudaErrorInvalidValue);
  // CAKE_EFFECT: controls.admit
  const int chunks=chunks_by_rank[0];
  if (chunks!=1 && chunks!=2 && chunks!=4)
    return int(cudaErrorInvalidValue);
  for (int rank=0;rank<R;++rank)
    if (communication_ctas[rank]<1 || communication_ctas[rank]>=96 ||
        steal_budgets[rank]<0 || steal_budgets[rank]>kTotalStageTasks ||
        chunks_by_rank[rank]!=chunks)
      return int(cudaErrorInvalidValue);
  int chunk_tokens=T/chunks;
  int active_chunks=chunks;
  cudaError_t error=cudaSuccess;
#define CAKE_RUN(call) \
  do { error=(call); if (error!=cudaSuccess) { state->poisoned=true; return int(error); } } while (0)
  // Complete admission and reset before any rank starts publishing remotely.
  for (int rank=0;rank<R;++rank) {
    CAKE_RUN(cudaSetDevice(rank));
    CAKE_RUN(ranked_tile_validate_routes(state->ranks[rank]));
  }
  for (int rank=0;rank<R;++rank) {
    CAKE_RUN(cudaSetDevice(rank));
    CAKE_RUN(ranked_tile_reset(state->ranks[rank]));
  }
  // The rank streams are nonblocking: a default-stream memset cannot order
  // their launches, and a source rank may write into a peer's Bin. Complete
  // every peer reset before launching any cooperative grid.
  for (int rank=0;rank<R;++rank) {
    CAKE_RUN(cudaSetDevice(rank));
    CAKE_RUN(cudaStreamSynchronize(state->ranks[rank].compute));
  }
  for (int rank=0;rank<R;++rank) {
    CAKE_RUN(cudaSetDevice(rank));
    RankedTileRankState& s=state->ranks[rank];
    int* event_ready=&s.bin->wave_consumed[0];
    int first_event=0;
    int communication=communication_ctas[rank];
    int budget=steal_budgets[rank];
    void* args[]={&s.heads,&s.completed,&s.owners,&s.processed,
                  &s.dispatched,&s.stolen,&s.permits,&s.tasks,
                  &event_ready,&s.overlap,&s.tile_experts,&s.upgate,
                  &s.activated,&s.up_maps_a,&s.up_map_b,&s.down_maps_a,
                  &s.down_map_b,&s.down,&communication,&budget,
                  &s.bin_params,&chunk_tokens,&active_chunks,
                  &first_event};
    // CAKE_EFFECT: launch.rank
    CAKE_RUN(cudaLaunchCooperativeKernel(
        reinterpret_cast<const void*>(tile_schedule_probe),
        dim3(96),dim3(kThreads),args,kDynamicShared,s.compute));
  }
  for (int rank=0;rank<R;++rank) {
    CAKE_RUN(cudaSetDevice(rank));
    RankedTileRankState& s=state->ranks[rank];
    scatter_returns<<<dim3(kLogicalTiles,kRows),256,0,s.compute>>>(
        s.return_params);
    CAKE_RUN(cudaGetLastError());
  }
  for (int rank=0;rank<R;++rank) {
    CAKE_RUN(cudaSetDevice(rank));
    RankedTileRankState& s=state->ranks[rank];
    wait_returns<<<(LOCAL_ROUTES+255)/256,256,0,s.compute>>>(s.return_params);
    CAKE_RUN(cudaGetLastError());
    // The current generated combine kernel spells every global argument as
    // mutable; its Schedule marks route_weights INPUT and emission only reads it.
    cake_weave_rank512_combine_kernel<<<dim3(T,8),256,0,s.compute>>>(
        s.contributions,const_cast<float*>(s.route_weights),s.output);
    CAKE_RUN(cudaGetLastError());
  }
  for (int rank=0;rank<R;++rank) {
    CAKE_RUN(cudaSetDevice(rank));
    CAKE_RUN(cudaDeviceSynchronize());
  }
  for (int rank=0;rank<R;++rank) {
    CAKE_RUN(cudaSetDevice(rank));
    RankedTileRankState& s=state->ranks[rank];
    int device_error=0;
    // CAKE_EFFECT: status.acquire
    CAKE_RUN(cudaMemcpy(&device_error,&s.bin->error,sizeof(int),
                        cudaMemcpyDeviceToHost));
    if (device_error!=0) {
      if (device_error==18) {
        RouteMismatch detail{};
        CAKE_RUN(cudaMemcpy(&detail,&s.bin->route_mismatch,sizeof(detail),
                            cudaMemcpyDeviceToHost));
        std::fprintf(stderr,
            "ranked tile route mismatch rank=%d event=%d block=%d key=%d "
            "location=%d expected_expert=%d location_key=%d\n",
            rank,detail.event,detail.block,detail.key,detail.location,
            detail.expected_expert,detail.location_key);
      }
      return 1000+device_error;
    }
    int completed[kStages*kEvents],required[kStages*kEvents];
    CAKE_RUN(cudaMemcpy(completed,s.processed,sizeof(completed),
                        cudaMemcpyDeviceToHost));
    CAKE_RUN(cudaMemcpy(required,s.tasks,sizeof(required),
                        cudaMemcpyDeviceToHost));
    for (int index=0;index<kStages*kEvents;++index)
      if (completed[index]!=required[index]) return 1100+index;
    int stolen[kStages*kEvents],permits=-1;
    CAKE_RUN(cudaMemcpy(stolen,s.stolen,sizeof(stolen),
                        cudaMemcpyDeviceToHost));
    CAKE_RUN(cudaMemcpy(&permits,s.permits,sizeof(int),
                        cudaMemcpyDeviceToHost));
    int borrowed=0;
    for (int count:stolen) {
      if (count<0) return 1300;
      borrowed+=count;
    }
    if (borrowed>steal_budgets[rank] || permits!=borrowed) return 1301;
    int flags[LOCAL_ROUTES];
    CAKE_RUN(cudaMemcpy(flags,s.return_ready,sizeof(flags),
                        cudaMemcpyDeviceToHost));
    for (int slot=0;slot<LOCAL_ROUTES;++slot)
      if (flags[slot]!=1) return 1200+slot;
    const char* phase_trace=std::getenv("CAKE_WEAVE_PHASE_TRACE");
    if (phase_trace && phase_trace[0]=='1') {
      unsigned long long phases[kEvents][kPhasePoints];
      int event_overlap[kEvents+2];
      CAKE_RUN(cudaMemcpy(phases,s.bin->phase_cycles,sizeof(phases),
                          cudaMemcpyDeviceToHost));
      CAKE_RUN(cudaMemcpy(event_overlap,s.overlap,sizeof(event_overlap),
                          cudaMemcpyDeviceToHost));
      for (int event=0;event<chunks*(R+1);++event) {
        std::fprintf(stderr,"CAKE_PHASE launch=%d rank=%d event=%d cycles=",
                     state->completed_launches,rank,event);
        for (int point=0;point<kPhasePoints-1;++point)
          std::fprintf(stderr,"%s%llu",point==0 ? "" : ",",
                       phases[event][point+1]-phases[event][point]);
        std::fprintf(stderr,"\n");
        std::fprintf(stderr,
            "CAKE_EVENT_OVERLAP launch=%d rank=%d event=%d progressed=%d\n",
            state->completed_launches,rank,event,event_overlap[2+event]);
      }
    }
  }
#undef CAKE_RUN
  ++state->completed_launches;
  return 0;
}
