@CAKE_RANKED_TILE_DEVICE@bool read_slice(const char* root,const char* name,size_t offset,
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
  bool plan_only=argc==7 && std::strcmp(argv[6],"plan-only")==0;
  bool general=argc==7 && std::strcmp(argv[6],"general")==0;
  if (argc!=6 && !plan_only && !general) return 2;
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
    if (!plan_only && !general &&
        (expert_counts[expert]<64 || expert_counts[expert]>256))
      return 6;
  const int expected_owner_rows[R]={4039,4196,4016,4133};
  for (int rank=0;rank<R;++rank)
    if (!plan_only && !general &&
        valid_by_owner[rank]!=expected_owner_rows[rank]) return 7;
  Case scenario[R]{};
  int wave_counts[R][kEvents]{};
  int wave_summary[R][kWaves]{};
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
        check(cudaMalloc(&s.owners,kStages*kEvents*kMaxTasks*4),
              "owner IDs malloc") ||
        check(cudaMalloc(&s.processed,kStages*kEvents*4),
              "processed malloc") ||
        check(cudaMalloc(&s.dispatched,kEvents*4),"dispatched malloc") ||
        check(cudaMalloc(&s.stolen,kStages*kEvents*4),"stolen malloc") ||
        check(cudaMalloc(&s.permits,4),"permits malloc") ||
        check(cudaMalloc(&s.tasks,kStages*kEvents*4),"tasks malloc") ||
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
        check(cudaMemset(s.tasks,0,kStages*kEvents*4),
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
        check(cudaMemset(s.owners,0xff,kStages*kEvents*kMaxTasks*4),
              "owner IDs reset") ||
        check(cudaMemset(s.processed,0,kStages*kEvents*4),
              "processed reset") ||
        check(cudaMemset(s.dispatched,0,kEvents*4),
              "dispatched reset") ||
        check(cudaMemset(s.stolen,0,kStages*kEvents*4),"stolen reset") ||
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
  // Each source-completion event can publish full tiles. Only the fifth
  // event of a wave flushes thresholded partial tiles. Each Cake worker
  // launch is finite, so communication never depends on a blocked grid.
  cudaStream_t communication_stream[R]{},compute_stream[R]{};
  cudaEvent_t ready_events[R][kEvents]{},fence_events[R][kEvents]{};
  for (int rank=0;rank<R;++rank) {
    if (check(cudaSetDevice(rank),"select communication stream rank") ||
        check(cudaStreamCreateWithFlags(&communication_stream[rank],
              cudaStreamNonBlocking),"create communication stream") ||
        check(cudaStreamCreateWithFlags(&compute_stream[rank],
              cudaStreamNonBlocking),"create compute stream")) return 26;
    for (int event=0;event<kEvents;++event)
      if (check(cudaEventCreateWithFlags(&ready_events[rank][event],
                cudaEventDisableTiming),"ready event") ||
          check(cudaEventCreateWithFlags(&fence_events[rank][event],
                cudaEventDisableTiming),"fence event")) return 26;
  }
  int communication=communication_control,budget=steal_control;
  for (int wave=0;wave<kWaves;++wave) {
    for (int source=0;source<=R;++source) {
      int event=wave*(R+1)+source;
      if (source<R) {
        if (check(cudaSetDevice(source),"select source rank")) return 27;
        dispatch_source_wave<<<dim3(128,K),256,0,
                               communication_stream[source]>>>(
            state[source].bin_params,wave);
        if (check(cudaGetLastError(),"source dispatch launch")) return 27;
        mark_source_wave_done<<<1,1,0,communication_stream[source]>>>(
            state[source].bin_params,wave);
        if (check(cudaGetLastError(),"source completion publish")) return 28;
      }
      for (int rank=0;rank<R;++rank) {
        if (check(cudaSetDevice(rank),"select destination event rank")) return 29;
        RankState& s=state[rank];
        cudaStream_t stream=communication_stream[rank];
        derive_wave_order<<<LOCAL_E,256,0,stream>>>(s.bin_params,wave,source);
        if (check(cudaGetLastError(),"event expert sort launch")) return 30;
        assign_wave_tiles<<<1,LOCAL_E,0,stream>>>(s.bin_params,event);
        if (check(cudaGetLastError(),"event manifest launch")) return 30;
        expand_tasks_for_wave<<<1,kStages,0,stream>>>(s.bin_params,s.tasks,event);
        if (check(cudaGetLastError(),"event tasks launch")) return 30;
        gather_wave_tiles<<<dim3(LOCAL_E,kRows),256,0,stream>>>(
            s.bin_params,event,wave,source);
        if (check(cudaGetLastError(),"event gather launch")) return 30;
        publish_wave_ready<<<1,1,0,stream>>>(s.bin_params,event,wave,source);
        if (check(cudaGetLastError(),"event ready publish")) return 30;
        if (check(cudaEventRecord(ready_events[rank][event],stream),
                  "ready event record")) return 30;
      }
      // No source may change bin counts until every destination has taken
      // this event's snapshot and gathered its newly published tiles.
      for (int rank=0;rank<R;++rank) {
        if (check(cudaSetDevice(rank),"select event fence rank")) return 30;
        wait_all_destinations<<<1,1,0,communication_stream[rank]>>>(
            state[rank].bin_params,event);
        if (check(cudaGetLastError(),"event fence launch")) return 30;
        if (check(cudaEventRecord(fence_events[rank][event],
                                  communication_stream[rank]),
                  "fence event record")) return 30;
      }
      for (int rank=0;rank<R;++rank) {
        if (check(cudaSetDevice(rank),"select Cake event rank")) return 31;
        if (plan_only) continue;
        RankState& s=state[rank];
        if (check(cudaStreamWaitEvent(compute_stream[rank],
                                      ready_events[rank][event]),
                  "wait for published Cake event")) return 31;
        int* wave_ready=&s.bin->wave_consumed[0];
        int selected_event=event;
        void* args[]={&s.heads,&s.completed,&s.owners,&s.processed,
                      &s.dispatched,&s.stolen,&s.permits,&s.tasks,
                      &wave_ready,&s.overlap,&s.tile_experts,&s.upgate,
                      &s.activated,&s.up_maps_a,&s.up_map_b,&s.down_maps_a,
                      &s.down_map_b,&s.down,&communication,&budget,
                      &selected_event};
        if (check(cudaLaunchCooperativeKernel(
                 reinterpret_cast<const void*>(tile_schedule_probe),
                 dim3(scenario[rank].grid),dim3(kThreads),args,
                 kDynamicShared,compute_stream[rank]),
                 "Cake event worker launch")) return 31;
      }
    }
  }
  for (int rank=0;rank<R && !plan_only;++rank) {
    if (check(cudaSetDevice(rank),"select return rank")) return 32;
    scatter_returns<<<dim3(kLogicalTiles,kRows),256,0,compute_stream[rank]>>>(
        state[rank].return_params);
    if (check(cudaGetLastError(),"return launch")) return 33;
  }
  for (int rank=0;rank<R && !plan_only;++rank) {
    if (check(cudaSetDevice(rank),"select combine rank")) return 34;
    wait_returns<<<(LOCAL_ROUTES+255)/256,256,0,compute_stream[rank]>>>(
        state[rank].return_params);
    if (check(cudaGetLastError(),"return acquire launch")) return 35;
    cake_weave_rank512_combine_kernel<<<dim3(T,8),256,0,compute_stream[rank]>>>(
        state[rank].contributions,state[rank].route_weights,
        state[rank].final_output);
    if (check(cudaGetLastError(),"Cake combine launch")) return 36;
  }
  bool published_all=false;
  for (int second=0;second<20 && !published_all;++second) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    published_all=true;
    for (int rank=0;rank<R;++rank) {
      if (check(cudaSetDevice(rank),"select wave event rank")) return 56;
      int ready=-1,fenced=-1;
      for (int event=0;event<kEvents;++event) {
        cudaError_t ready_status=cudaEventQuery(ready_events[rank][event]);
        cudaError_t fence_status=cudaEventQuery(fence_events[rank][event]);
        if (ready_status!=cudaSuccess && ready_status!=cudaErrorNotReady)
          check(ready_status,"ready event query");
        if (fence_status!=cudaSuccess && fence_status!=cudaErrorNotReady)
          check(fence_status,"fence event query");
        if (ready_status==cudaSuccess)
          ready=event;
        if (fence_status==cudaSuccess)
          fenced=event;
      }
      if (fenced!=kEvents-1) published_all=false;
      std::fprintf(stderr,"event_progress second=%d rank=%d ready=%d fence=%d\n",
                   second+1,rank,ready,fenced);
    }
  }
  if (!published_all) return 56;
  for (int rank=0;rank<R;++rank) {
    if (check(cudaSetDevice(rank),"select completion rank") ||
        check(cudaDeviceSynchronize(),"full chain completion")) return 37;
  }
  if (plan_only) {
    int logical_tiles[R]{},observed_rows[R]{};
    std::vector<unsigned char> seen(ROUTES);
    for (int rank=0;rank<R;++rank) {
      if (check(cudaSetDevice(rank),"select plan result rank")) return 57;
      Bin* bin=state[rank].bin;
      int counts[LOCAL_E],error=0;
      if (check(cudaMemcpy(&error,&bin->error,4,cudaMemcpyDeviceToHost),
                "plan error read") ||
          check(cudaMemcpy(counts,bin->count,sizeof(counts),
                           cudaMemcpyDeviceToHost),"plan bin counts read") ||
          check(cudaMemcpy(wave_counts[rank],bin->wave_counts,
                           sizeof(wave_counts[rank]),cudaMemcpyDeviceToHost),
                "plan wave counts read")) return 57;
      if (error!=0) {
        std::fprintf(stderr,"rank %d planner error %d\n",rank,error);
        return 58;
      }
      for (int expert=0;expert<LOCAL_E;++expert) {
        if (counts[expert]<0 || counts[expert]>MAX_ROWS) return 58;
        observed_rows[rank]+=counts[expert];
      }
      if (observed_rows[rank]!=valid_by_owner[rank]) return 58;
      for (int event=0;event<kEvents;++event) {
        if (wave_counts[rank][event]<0 ||
            logical_tiles[rank]+wave_counts[rank][event]>kLogicalTiles)
          return 58;
        logical_tiles[rank]+=wave_counts[rank][event];
        wave_summary[rank][event/(R+1)]+=wave_counts[rank][event];
      }
      std::vector<int> keys(size_t(logical_tiles[rank])*kRows);
      std::vector<int> experts(logical_tiles[rank]);
      if (logical_tiles[rank]>0 &&
          (check(cudaMemcpy(keys.data(),state[rank].tile_keys,
                            keys.size()*sizeof(int),cudaMemcpyDeviceToHost),
                 "plan tile keys read") ||
           check(cudaMemcpy(experts.data(),state[rank].tile_experts,
                            experts.size()*sizeof(int),cudaMemcpyDeviceToHost),
                 "plan tile experts read"))) return 59;
      int covered=0;
      for (int tile=0;tile<logical_tiles[rank];++tile) {
        int expert=experts[tile];
        if (expert<0 || expert>=LOCAL_E) return 59;
        for (int row=0;row<kRows;++row) {
          int key=keys[tile*kRows+row];
          if (key<0) continue;
          int assigned=-1;
          if (key>=ROUTES || seen[key]) return 59;
          std::memcpy(&assigned,ids.data()+size_t(key)*sizeof(int),4);
          if (assigned!=rank*LOCAL_E+expert) return 59;
          seen[key]=1;
          ++covered;
        }
      }
      if (covered!=valid_by_owner[rank]) return 59;
      char name[128];
      std::snprintf(name,sizeof(name),
                    "device_outputs/rank%d/tile_route_keys.i32",rank);
      if (!write_exact(argv[1],name,keys.data(),keys.size()*sizeof(int)))
        return 60;
      std::snprintf(name,sizeof(name),"device_outputs/rank%d/tile_expert.i32",rank);
      if (!write_exact(argv[1],name,experts.data(),experts.size()*sizeof(int)))
        return 60;
      std::snprintf(name,sizeof(name),"device_outputs/rank%d/wave_counts.i32",rank);
      if (!write_exact(argv[1],name,wave_summary[rank],
                       sizeof(wave_summary[rank]))) return 60;
      std::snprintf(name,sizeof(name),"device_outputs/rank%d/event_counts.i32",rank);
      if (!write_exact(argv[1],name,wave_counts[rank],
                       sizeof(wave_counts[rank]))) return 60;
    }
    for (unsigned char flag:seen) if (flag!=1) return 61;
    char path[512];
    int n=std::snprintf(path,sizeof(path),"%s/plan_device_report.json",argv[1]);
    if (n<1 || n>=int(sizeof(path))) return 62;
    FILE* report=std::fopen(path,"wx");
    if (!report) return 62;
    std::fprintf(report,"{\"target\":\"sm_103a\",\"safe_tile_capacity\":255,"
                        "\"routes\":%d,\"owner_routes\":[%d,%d,%d,%d],"
                        "\"logical_tiles_by_owner\":[%d,%d,%d,%d],"
                        "\"tile_waves_by_owner\":[",ROUTES,
                 observed_rows[0],observed_rows[1],observed_rows[2],observed_rows[3],
                 logical_tiles[0],logical_tiles[1],logical_tiles[2],logical_tiles[3]);
    for (int rank=0;rank<R;++rank) {
      std::fprintf(report,"%s[",rank ? "," : "");
      for (int wave=0;wave<kWaves;++wave)
        std::fprintf(report,"%s%d",wave ? "," : "",wave_summary[rank][wave]);
      std::fprintf(report,"]");
    }
    std::fprintf(report,"],\"tile_events_by_owner\":[");
    for (int rank=0;rank<R;++rank) {
      std::fprintf(report,"%s[",rank ? "," : "");
      for (int event=0;event<kEvents;++event)
        std::fprintf(report,"%s%d",event ? "," : "",wave_counts[rank][event]);
      std::fprintf(report,"]");
    }
    std::fprintf(report,"]}\n");
    if (std::fclose(report)) return 62;
    std::fprintf(stderr,"capacity planner: %d routes, tiles [%d,%d,%d,%d]\n",
                 ROUTES,logical_tiles[0],logical_tiles[1],
                 logical_tiles[2],logical_tiles[3]);
    return 0;
  }
  std::vector<unsigned char> observed_hidden(HIDDEN_BYTES),observed_ids(IDS_BYTES);
  std::vector<unsigned char> route_contributions(ROUTE_CONTRIBUTION_BYTES);
  std::vector<unsigned char> final_output(FINAL_BYTES);
  std::vector<int> return_flags(ROUTES);
  int stolen_by_rank[R]{},bin_rows_by_rank[R]{},bin_error[R]{};
  int planned_tiles_by_rank[R]{};
  int overlap_by_rank[R][2]{};
  for (int rank=0;rank<R;++rank) {
    if (check(cudaSetDevice(rank),"select result rank")) return 38;
    RankState& s=state[rank];
    int counts[LOCAL_E],completed[kStages*kLogicalTiles];
    std::vector<int> owner_ids(size_t(kStages)*kEvents*kMaxTasks);
    int processed[kStages*kEvents],dispatched[kEvents];
    int stolen[kStages*kEvents],permits=-1;
    if (check(cudaMemcpy(counts,s.bin->count,sizeof(counts),
                         cudaMemcpyDeviceToHost),"bin counts read") ||
        check(cudaMemcpy(&bin_error[rank],&s.bin->error,sizeof(int),
                         cudaMemcpyDeviceToHost),"bin error read") ||
        check(cudaMemcpy(wave_counts[rank],s.bin->wave_counts,
                         sizeof(wave_counts[rank]),cudaMemcpyDeviceToHost),
              "GPU tile-wave counts read") ||
        check(cudaMemcpy(completed,s.completed,sizeof(completed),
                         cudaMemcpyDeviceToHost),"stage completion read") ||
        check(cudaMemcpy(owner_ids.data(),s.owners,
                         owner_ids.size()*sizeof(int),
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
    if (bin_error[rank]!=0 ||
        (!general && (overlap_by_rank[rank][0]!=1 ||
                      overlap_by_rank[rank][1]!=1))) return 40;
    for (int expert=0;expert<LOCAL_E;++expert) {
      if (counts[expert]<0 || counts[expert]>MAX_ROWS) return 41;
      bin_rows_by_rank[rank]+=counts[expert];
    }
    if (bin_rows_by_rank[rank]!=valid_by_owner[rank]) return 42;
    int planned_tiles=0;
    for (int wave=0;wave<kEvents;++wave) {
      if (wave_counts[rank][wave]<0 ||
          wave_counts[rank][wave]>kLogicalTiles) return 42;
      planned_tiles+=wave_counts[rank][wave];
      wave_summary[rank][wave/(R+1)]+=wave_counts[rank][wave];
      scenario[rank].tasks[0][wave]=
          wave_counts[rank][wave]*kUpGateTasksPerTile;
      scenario[rank].tasks[1][wave]=
          wave_counts[rank][wave]*kActivationTasksPerTile;
      scenario[rank].tasks[2][wave]=
          wave_counts[rank][wave]*kDownTasksPerTile;
    }
    if ((!general && planned_tiles!=kExpectedTiles) ||
        planned_tiles>kLogicalTiles) return 42;
    planned_tiles_by_rank[rank]=planned_tiles;
    for (int stage=0;stage<kStages;++stage) {
      int units=stage==0 ? kUpGateTasksPerTile :
                stage==1 ? kActivationTasksPerTile : kDownTasksPerTile;
      for (int tile=0;tile<planned_tiles;++tile)
        if (completed[stage*kLogicalTiles+tile]!=units) return 43;
      for (int wave=0;wave<kEvents;++wave) {
        int index=stage*kEvents+wave;
        if (processed[index]!=scenario[rank].tasks[stage][wave]) return 44;
        stolen_by_rank[rank]+=stolen[index];
        for (int task=0;task<scenario[rank].tasks[stage][wave];++task) {
          int who=owner_ids[index*kMaxTasks+task];
          if (who<0 || who>=scenario[rank].grid) return 45;
        }
      }
    }
    for (int wave=0;wave<kEvents;++wave)
      if (dispatched[wave]!=scenario[rank].communication) return 46;
    if (stolen_by_rank[rank]<0 ||
        stolen_by_rank[rank]>scenario[rank].budget ||
        permits!=stolen_by_rank[rank] ||
        (scenario[rank].communication==148 &&
         stolen_by_rank[rank]!=kTotalStageTasks))
      return 47;
    size_t tile_input_bytes=size_t(planned_tiles)*kRows*H*sizeof(uint16_t);
    size_t down_bytes=size_t(planned_tiles)*kRows*H*sizeof(float);
    size_t tile_key_bytes=size_t(planned_tiles)*kRows*sizeof(int);
    std::vector<unsigned char> tile_input(tile_input_bytes),down(down_bytes);
    std::vector<unsigned char> manifest_keys(tile_key_bytes),
                               manifest_experts(size_t(planned_tiles)*sizeof(int));
    if (planned_tiles>0 &&
        (check(cudaMemcpy(tile_input.data(),s.tile_input,tile_input_bytes,
                         cudaMemcpyDeviceToHost),"tile input read") ||
        check(cudaMemcpy(manifest_keys.data(),s.tile_keys,tile_key_bytes,
                         cudaMemcpyDeviceToHost),"GPU tile route keys read") ||
        check(cudaMemcpy(manifest_experts.data(),s.tile_experts,
                         manifest_experts.size(),cudaMemcpyDeviceToHost),
              "GPU tile experts read") ||
        check(cudaMemcpy(down.data(),s.down,down_bytes,
                         cudaMemcpyDeviceToHost),"down read"))) return 48;
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
    if (!write_exact(argv[1],name,wave_summary[rank],
                     sizeof(wave_summary[rank]))) return 49;
    std::snprintf(name,sizeof(name),"device_outputs/rank%d/event_counts.i32",rank);
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
                      "\"safe_tile_capacity\":255,"
                      "\"logical_tiles_by_owner\":[%d,%d,%d,%d],"
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
              planned_tiles_by_rank[0],planned_tiles_by_rank[1],
              planned_tiles_by_rank[2],planned_tiles_by_rank[3],
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
      std::fprintf(report,"%s%d",wave ? "," : "",wave_summary[rank][wave]);
    std::fprintf(report,"]");
  }
  std::fprintf(report,"],\"tile_events_by_owner\":[");
  for (int rank=0;rank<R;++rank) {
    std::fprintf(report,"%s[",rank ? "," : "");
    for (int event=0;event<kEvents;++event)
      std::fprintf(report,"%s%d",event ? "," : "",wave_counts[rank][event]);
    std::fprintf(report,"]");
  }
  std::fprintf(report,"]}\n");
  if (std::fclose(report)) return 55;
  for (int rank=0;rank<R;++rank) {
    cudaSetDevice(rank);
    cudaStreamDestroy(communication_stream[rank]);
    cudaStreamDestroy(compute_stream[rank]);
    for (int wave=0;wave<kEvents;++wave) {
      cudaEventDestroy(ready_events[rank][wave]);
      cudaEventDestroy(fence_events[rank][wave]);
    }
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
  constexpr int stage_units=kUpGateTasksPerTile+kActivationTasksPerTile+
                            kDownTasksPerTile;
  std::printf("full four-rank chain: %d routes, tiles/rank [%d,%d,%d,%d], "
              "stage tasks/rank [%d,%d,%d,%d]\n",ROUTES,
              planned_tiles_by_rank[0],planned_tiles_by_rank[1],
              planned_tiles_by_rank[2],planned_tiles_by_rank[3],
              planned_tiles_by_rank[0]*stage_units,
              planned_tiles_by_rank[1]*stage_units,
              planned_tiles_by_rank[2]*stage_units,
              planned_tiles_by_rank[3]*stage_units);
  return 0;
}
