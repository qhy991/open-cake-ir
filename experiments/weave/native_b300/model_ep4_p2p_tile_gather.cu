// B300 development probe: four-rank peer bins publish route-keyed BF16 rows;
// the destination GPU acquires them and materializes model M128 tiles from
// the admitted threshold-64 plan. Tile math and return follow separately.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 1030
#error "Exact sm_103a is required"
#endif

namespace {
constexpr int R=4, T=512, K=8, E=128, H=2048, TILES=64, ROWS=128;
constexpr int LOCAL_E=E/R, MAX_ROWS=R*T;
constexpr int ROUTES=R*T*K;
constexpr size_t HIDDEN_BYTES=size_t(R)*T*H*sizeof(uint16_t);
constexpr size_t IDS_BYTES=size_t(ROUTES)*sizeof(int);
constexpr size_t TILE_BYTES=size_t(TILES)*ROWS*H*sizeof(uint16_t);
constexpr size_t TILE_KEY_BYTES=size_t(TILES)*ROWS*sizeof(int);

struct Bin {
  int count[LOCAL_E];
  int ready[LOCAL_E*MAX_ROWS];
  int keys[LOCAL_E*MAX_ROWS];
  int route_location[ROUTES];
  int route_ready[ROUTES];
  __nv_bfloat16 rows[size_t(LOCAL_E)*MAX_ROWS*H];
  int error;
};
struct Params {
  Bin* bins[R];
  const __nv_bfloat16* hidden;
  const int* ids;
  const int* tile_keys;
  const int* tile_experts;
  __nv_bfloat16* tile_rows;
  int rank;
};

__device__ __forceinline__ int reserve_remote(int* pointer) {
  int old;
  asm volatile("atom.relaxed.sys.global.add.s32 %0, [%1], %2;"
               : "=r"(old) : "l"(pointer), "r"(1) : "memory");
  return old;
}
__device__ __forceinline__ void publish_remote(int* pointer) {
  asm volatile("st.release.sys.global.s32 [%0], %1;"
               :: "l"(pointer), "r"(1) : "memory");
}
__device__ __forceinline__ int acquire_route(const int* pointer) {
  int value;
  asm volatile("ld.acquire.sys.global.s32 %0, [%1];"
               : "=r"(value) : "l"(pointer) : "memory");
  return value;
}
__global__ void dispatch(const Params* params) {
  const int token=int(blockIdx.x), route=int(blockIdx.y);
  const int expert=params->ids[token*K+route];
  if (expert<0 || expert>=E) {
    if (threadIdx.x==0) atomicCAS(&params->bins[params->rank]->error,0,1);
    return;
  }
  const int owner=expert/LOCAL_E;
  const int local=expert%LOCAL_E;
  Bin* destination=params->bins[owner];
  __shared__ int slot;
  if (threadIdx.x==0) slot=reserve_remote(&destination->count[local]);
  __syncthreads();
  if (slot<0 || slot>=MAX_ROWS) {
    if (threadIdx.x==0) atomicCAS(&params->bins[params->rank]->error,0,2);
    return;
  }
  const int index=local*MAX_ROWS+slot;
  const uint16_t* source=reinterpret_cast<const uint16_t*>(params->hidden)
      +size_t(token)*H;
  uint16_t* target=reinterpret_cast<uint16_t*>(destination->rows)
      +size_t(index)*H;
  for (int feature=int(threadIdx.x);feature<H;feature+=int(blockDim.x))
    target[feature]=source[feature];
  if (threadIdx.x==0)
    destination->keys[index]=(params->rank*T+token)*K+route;
  int key=(params->rank*T+token)*K+route;
  if (threadIdx.x==0) destination->route_location[key]=index;
  __syncthreads();
  if (threadIdx.x==0) {
    publish_remote(&destination->ready[index]);
    publish_remote(&destination->route_ready[key]);
  }
}
__global__ void gather_tile(const Params* params) {
  int tile=int(blockIdx.x),row=int(blockIdx.y);
  int key=params->tile_keys[tile*ROWS+row];
  __nv_bfloat16* output=params->tile_rows+(size_t(tile)*ROWS+row)*H;
  if (key<0) {
    for (int feature=int(threadIdx.x);feature<H;feature+=int(blockDim.x))
      output[feature]=__nv_bfloat16(0.0f);
    return;
  }
  if (key>=ROUTES) asm volatile("trap;");
  Bin* local=params->bins[params->rank];
  if (threadIdx.x==0)
    while (acquire_route(&local->route_ready[key])==0) __nanosleep(64);
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

int check(cudaError_t status,const char* operation) {
  if (status==cudaSuccess) return 0;
  std::fprintf(stderr,"%s: %s (%d)\n",operation,cudaGetErrorString(status),int(status));
  return 1;
}
bool read_exact(const char* root,const char* name,std::vector<unsigned char>& out) {
  char path[512];
  int length=std::snprintf(path,sizeof(path),"%s/%s",root,name);
  if (length<1 || length>=int(sizeof(path))) return false;
  FILE* file=std::fopen(path,"rb");
  if (!file) return false;
  bool okay=std::fread(out.data(),1,out.size(),file)==out.size()
            && std::fgetc(file)==EOF;
  return std::fclose(file)==0 && okay;
}
bool read_owner(const char* root,int owner,const char* name,
                std::vector<unsigned char>& out) {
  char path[512];
  int length=std::snprintf(path,sizeof(path),"%s/rank%d/%s",root,owner,name);
  if (length<1 || length>=int(sizeof(path))) return false;
  FILE* file=std::fopen(path,"rb");
  if (!file) return false;
  bool okay=std::fread(out.data(),1,out.size(),file)==out.size()
            && std::fgetc(file)==EOF;
  return std::fclose(file)==0 && okay;
}
bool write_exact(const char* root,const char* name,const void* bytes,size_t size) {
  char path[512];
  int length=std::snprintf(path,sizeof(path),"%s/%s",root,name);
  if (length<1 || length>=int(sizeof(path))) return false;
  FILE* file=std::fopen(path,"wbx");
  if (!file) return false;
  bool okay=std::fwrite(bytes,1,size,file)==size;
  return std::fclose(file)==0 && okay;
}
bool owner_path(const char* root,int owner,const char* name,char* out,size_t cap) {
  int length=std::snprintf(out,cap,"%s/device_outputs/owner%d/%s",root,owner,name);
  return length>0 && length<int(cap);
}
bool write_owner(const char* root,int owner,const char* name,
                 const void* bytes,size_t size) {
  char path[512];
  if (!owner_path(root,owner,name,path,sizeof(path))) return false;
  FILE* file=std::fopen(path,"wbx");
  if (!file) return false;
  bool okay=std::fwrite(bytes,1,size,file)==size;
  return std::fclose(file)==0 && okay;
}
} // namespace

int main(int argc,char** argv) {
  if (argc!=3) return 2;
  int devices=0;
  if (check(cudaGetDeviceCount(&devices),"device count") || devices!=R) return 3;
  std::vector<unsigned char> hidden(HIDDEN_BYTES), ids(IDS_BYTES);
  if (!read_exact(argv[1],"hidden.bf16",hidden)
      || !read_exact(argv[1],"expert_ids.i32",ids)) return 4;
  for (int token=0;token<R*T;++token) {
    bool seen[E]{};
    for (int route=0;route<K;++route) {
      int expert=-1;
      std::memcpy(&expert,ids.data()+size_t(token*K+route)*sizeof(int),sizeof(int));
      if (expert<0 || expert>=E || seen[expert]) return 5;
      seen[expert]=true;
    }
  }
  std::vector<unsigned char> plan_keys[R],plan_experts[R],tile_outputs[R];
  int seen[ROUTES]{};
  for (int owner=0;owner<R;++owner) {
    plan_keys[owner].resize(TILE_KEY_BYTES);
    plan_experts[owner].resize(TILES*sizeof(int));
    tile_outputs[owner].resize(TILE_BYTES);
    if (!read_owner(argv[2],owner,"tile_route_keys.i32",plan_keys[owner]) ||
        !read_owner(argv[2],owner,"tile_expert.i32",plan_experts[owner]))
      return 6;
    for (int tile=0;tile<TILES;++tile) {
      int local_expert=-1;
      std::memcpy(&local_expert,
                  plan_experts[owner].data()+size_t(tile)*sizeof(int),
                  sizeof(int));
      if (local_expert<0 || local_expert>=LOCAL_E) return 6;
      for (int row=0;row<ROWS;++row) {
        int key=-1,global_expert=-1;
        std::memcpy(&key,plan_keys[owner].data()+
                   size_t(tile*ROWS+row)*sizeof(int),sizeof(int));
        if (key<0) continue;
        if (key>=ROUTES || seen[key]++) return 6;
        std::memcpy(&global_expert,
                    ids.data()+size_t(key)*sizeof(int),sizeof(int));
        if (global_expert/LOCAL_E!=owner ||
            global_expert%LOCAL_E!=local_expert) return 6;
      }
    }
  }
  for (int key=0;key<ROUTES;++key) if (seen[key]!=1) return 6;
  Bin* bins[R]{};
  __nv_bfloat16* gpu_hidden[R]{};
  int* gpu_ids[R]{};
  int* gpu_tile_keys[R]{};
  int* gpu_tile_experts[R]{};
  __nv_bfloat16* gpu_tile_rows[R]{};
  Params* gpu_params[R]{};
  for (int rank=0;rank<R;++rank) {
    if (check(cudaSetDevice(rank),"select device")) return 6;
    cudaDeviceProp prop{};
    if (check(cudaGetDeviceProperties(&prop,rank),"device properties") ||
        prop.major!=10 || prop.minor!=3 ||
        (std::strcmp(prop.name,"NVIDIA B300") &&
         std::strcmp(prop.name,"NVIDIA B300 SXM6 AC"))) return 7;
    for (int peer=0;peer<R;++peer) {
      if (peer==rank) continue;
      int access=0,atomic=0;
      if (check(cudaDeviceCanAccessPeer(&access,rank,peer),"peer access") ||
          check(cudaDeviceGetP2PAttribute(&atomic,cudaDevP2PAttrNativeAtomicSupported,
                                          rank,peer),"native peer atomic") ||
          access!=1 || atomic!=1) return 8;
      cudaError_t status=cudaDeviceEnablePeerAccess(peer,0);
      if (status!=cudaSuccess && status!=cudaErrorPeerAccessAlreadyEnabled) {
        check(status,"enable peer access");return 9;
      }
    }
    if (check(cudaMalloc(&bins[rank],sizeof(Bin)),"bin malloc") ||
        check(cudaMalloc(&gpu_hidden[rank],T*H*sizeof(uint16_t)),"hidden malloc") ||
        check(cudaMalloc(&gpu_ids[rank],T*K*sizeof(int)),"ids malloc") ||
        check(cudaMalloc(&gpu_tile_keys[rank],TILE_KEY_BYTES),
              "tile keys malloc") ||
        check(cudaMalloc(&gpu_tile_experts[rank],TILES*sizeof(int)),
              "tile experts malloc") ||
        check(cudaMalloc(&gpu_tile_rows[rank],TILE_BYTES),
              "tile rows malloc") ||
        check(cudaMalloc(&gpu_params[rank],sizeof(Params)),"params malloc") ||
        check(cudaMemset(bins[rank],0,sizeof(Bin)),"bin reset") ||
        check(cudaMemset(&bins[rank]->route_location,0xff,
                         sizeof(Bin::route_location)),"location sentinel") ||
        check(cudaMemset(&bins[rank]->keys,0xff,sizeof(Bin::keys)),
              "key sentinel") ||
        check(cudaMemset(gpu_tile_rows[rank],0xff,TILE_BYTES),
              "tile row sentinel") ||
        check(cudaMemcpy(gpu_hidden[rank],hidden.data()+size_t(rank)*T*H*2,
                         T*H*2,cudaMemcpyHostToDevice),"hidden copy") ||
        check(cudaMemcpy(gpu_ids[rank],ids.data()+size_t(rank)*T*K*4,
                         T*K*4,cudaMemcpyHostToDevice),"ids copy") ||
        check(cudaMemcpy(gpu_tile_keys[rank],plan_keys[rank].data(),
                         TILE_KEY_BYTES,cudaMemcpyHostToDevice),
              "tile keys copy") ||
        check(cudaMemcpy(gpu_tile_experts[rank],plan_experts[rank].data(),
                         TILES*sizeof(int),cudaMemcpyHostToDevice),
              "tile experts copy")) return 10;
  }
  for (int rank=0;rank<R;++rank) {
    if (check(cudaSetDevice(rank),"select owner")) return 11;
    Params params{};
    for (int peer=0;peer<R;++peer) params.bins[peer]=bins[peer];
    params.hidden=gpu_hidden[rank];params.ids=gpu_ids[rank];params.rank=rank;
    params.tile_keys=gpu_tile_keys[rank];
    params.tile_experts=gpu_tile_experts[rank];
    params.tile_rows=gpu_tile_rows[rank];
    if (check(cudaMemcpy(gpu_params[rank],&params,sizeof(params),
                         cudaMemcpyHostToDevice),"params copy")) return 12;
  }
  for (int rank=0;rank<R;++rank) {
    if (check(cudaSetDevice(rank),"select launch device")) return 13;
    dispatch<<<dim3(T,K),256>>>(gpu_params[rank]);
    if (check(cudaGetLastError(),"dispatch launch")) return 14;
  }
  for (int rank=0;rank<R;++rank) {
    if (check(cudaSetDevice(rank),"select tile gather device")) return 15;
    gather_tile<<<dim3(TILES,ROWS),256>>>(gpu_params[rank]);
    if (check(cudaGetLastError(),"tile gather launch")) return 15;
  }
  for (int rank=0;rank<R;++rank) {
    if (check(cudaSetDevice(rank),"select completion device") ||
        check(cudaDeviceSynchronize(),"rank dispatch completion")) return 15;
  }
  std::vector<unsigned char> observed_hidden(HIDDEN_BYTES), observed_ids(IDS_BYTES);
  for (int rank=0;rank<R;++rank) {
    if (check(cudaSetDevice(rank),"select input read device") ||
        check(cudaMemcpy(observed_hidden.data()+size_t(rank)*T*H*2,gpu_hidden[rank],
                         T*H*2,cudaMemcpyDeviceToHost),"hidden read") ||
        check(cudaMemcpy(observed_ids.data()+size_t(rank)*T*K*4,gpu_ids[rank],
                         T*K*4,cudaMemcpyDeviceToHost),"ids read") ||
        check(cudaMemcpy(tile_outputs[rank].data(),gpu_tile_rows[rank],
                         TILE_BYTES,cudaMemcpyDeviceToHost),
              "tile rows read")) return 16;
  }
  if (!write_exact(argv[1],"device_outputs/observed_hidden.bf16",
                   observed_hidden.data(),observed_hidden.size()) ||
      !write_exact(argv[1],"device_outputs/observed_expert_ids.i32",
                   observed_ids.data(),observed_ids.size())) return 17;
  for (int owner=0;owner<R;++owner) {
    if (check(cudaSetDevice(owner),"select bin read device")) return 18;
    int counts[LOCAL_E], keys[LOCAL_E*MAX_ROWS], error=-1;
    if (check(cudaMemcpy(counts,bins[owner]->count,sizeof(counts),
                         cudaMemcpyDeviceToHost),"counts read") ||
        check(cudaMemcpy(keys,bins[owner]->keys,sizeof(keys),
                         cudaMemcpyDeviceToHost),"keys read") ||
        check(cudaMemcpy(&error,&bins[owner]->error,sizeof(int),
                         cudaMemcpyDeviceToHost),"error read")) return 19;
    int used=0;
    for (int expert=0;expert<LOCAL_E;++expert) {
      if (counts[expert]<0 || counts[expert]>MAX_ROWS) return 20;
      used+=counts[expert];
      std::vector<int> ready(counts[expert]);
      if (counts[expert] && check(cudaMemcpy(ready.data(),
          bins[owner]->ready+expert*MAX_ROWS,counts[expert]*sizeof(int),
          cudaMemcpyDeviceToHost),"ready read")) return 21;
      for (int flag:ready) if (flag!=1) return 22;
    }
    if (error!=0 || !write_owner(argv[1],owner,"counts.u32",
                                 counts,sizeof(counts)) ||
        !write_owner(argv[1],owner,"keys.i32",keys,sizeof(keys))) return 23;
    if (!write_owner(argv[1],owner,"tiles.bf16",
                     tile_outputs[owner].data(),TILE_BYTES)) return 23;
    char path[512];
    if (!owner_path(argv[1],owner,"rows.bf16",path,sizeof(path))) return 24;
    FILE* file=std::fopen(path,"wbx");
    if (!file) return 25;
    for (int expert=0;expert<LOCAL_E;++expert) {
      std::vector<uint16_t> rows(size_t(counts[expert])*H);
      if (counts[expert] && check(cudaMemcpy(rows.data(),
          bins[owner]->rows+size_t(expert)*MAX_ROWS*H,rows.size()*2,
          cudaMemcpyDeviceToHost),"rows read")) return 26;
      if (std::fwrite(rows.data(),2,rows.size(),file)!=rows.size()) return 27;
    }
    if (std::fclose(file)) return 28;
    char report[128];
    int length=std::snprintf(report,sizeof(report),
       "{\"owner_rank\":%d,\"error_flag\":%d,\"used_rows\":%d}\n",
       owner,error,used);
    if (length<=0 || length>=int(sizeof(report)) ||
        !write_owner(argv[1],owner,"device.json",report,size_t(length))) return 29;
    std::printf("owner %d: routes=%d error=%d\n",owner,used,error);
  }
  for (int rank=0;rank<R;++rank) {
    cudaSetDevice(rank);cudaFree(gpu_params[rank]);cudaFree(gpu_ids[rank]);
    cudaFree(gpu_tile_rows[rank]);cudaFree(gpu_tile_experts[rank]);
    cudaFree(gpu_tile_keys[rank]);cudaFree(gpu_hidden[rank]);
    cudaFree(bins[rank]);
  }
  return 0;
}
