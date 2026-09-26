// B300 development probe: four source ranks publish model BF16 route rows
// into expert-owned peer bins. Cake's one-GPU bin Schedule owns the row/key
// meaning; this protocol probe tests native system-scope peer reservation and
// release/acquire for the distributed transport, before tile publication.
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
constexpr int R=4, T=512, K=8, E=128, H=2048;
constexpr int LOCAL_E=E/R, MAX_ROWS=R*T;
constexpr int ROUTES=R*T*K;
constexpr size_t HIDDEN_BYTES=size_t(R)*T*H*sizeof(uint16_t);
constexpr size_t IDS_BYTES=size_t(ROUTES)*sizeof(int);

struct Bin {
  int count[LOCAL_E];
  int ready[LOCAL_E*MAX_ROWS];
  int keys[LOCAL_E*MAX_ROWS];
  __nv_bfloat16 rows[size_t(LOCAL_E)*MAX_ROWS*H];
  int error;
};
struct Params {
  Bin* bins[R];
  const __nv_bfloat16* hidden;
  const int* ids;
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
  __syncthreads();
  if (threadIdx.x==0) publish_remote(&destination->ready[index]);
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
  if (argc!=2) return 2;
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
  Bin* bins[R]{};
  __nv_bfloat16* gpu_hidden[R]{};
  int* gpu_ids[R]{};
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
        check(cudaMalloc(&gpu_params[rank],sizeof(Params)),"params malloc") ||
        check(cudaMemset(bins[rank],0,sizeof(Bin)),"bin reset") ||
        check(cudaMemset(&bins[rank]->keys,0xff,sizeof(Bin::keys)),
              "key sentinel") ||
        check(cudaMemcpy(gpu_hidden[rank],hidden.data()+size_t(rank)*T*H*2,
                         T*H*2,cudaMemcpyHostToDevice),"hidden copy") ||
        check(cudaMemcpy(gpu_ids[rank],ids.data()+size_t(rank)*T*K*4,
                         T*K*4,cudaMemcpyHostToDevice),"ids copy")) return 10;
  }
  for (int rank=0;rank<R;++rank) {
    if (check(cudaSetDevice(rank),"select owner")) return 11;
    Params params{};
    for (int peer=0;peer<R;++peer) params.bins[peer]=bins[peer];
    params.hidden=gpu_hidden[rank];params.ids=gpu_ids[rank];params.rank=rank;
    if (check(cudaMemcpy(gpu_params[rank],&params,sizeof(params),
                         cudaMemcpyHostToDevice),"params copy")) return 12;
  }
  for (int rank=0;rank<R;++rank) {
    if (check(cudaSetDevice(rank),"select launch device")) return 13;
    dispatch<<<dim3(T,K),256>>>(gpu_params[rank]);
    if (check(cudaGetLastError(),"dispatch launch")) return 14;
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
                         T*K*4,cudaMemcpyDeviceToHost),"ids read")) return 16;
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
    cudaFree(gpu_hidden[rank]);cudaFree(bins[rank]);
  }
  return 0;
}
