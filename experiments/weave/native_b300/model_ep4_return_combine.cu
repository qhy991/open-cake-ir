// B300 development probe: deterministic-key P2P FP32 return, system release/
// acquire, then the Cake-lowered rank-local T512 weighted-combine kernel.
// Down outputs come from the separately validated four-rank Cake FFN bridge.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "combine/kernel.cu"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 1030
#error "Exact sm_103a is required"
#endif

namespace {
constexpr int R=4,T=512,K=8,H=2048,TILES=64,ROWS=128;
constexpr int ROUTES=R*T*K,LOCAL_ROUTES=T*K;
constexpr size_t DOWN_BYTES=size_t(TILES)*ROWS*H*sizeof(float);
constexpr size_t KEY_BYTES=size_t(TILES)*ROWS*sizeof(int);
constexpr size_t CONTRIBUTION_BYTES=size_t(ROUTES)*H*sizeof(float);
constexpr size_t OUTPUT_BYTES=size_t(R)*T*H*sizeof(uint16_t);
struct Params {
  float* contribution[R];
  int* ready[R];
  const float* down;
  const int* route_key;
  int rank;
};

__device__ __forceinline__ void release_ready(int* pointer) {
  asm volatile("st.release.sys.global.s32 [%0], %1;"
               :: "l"(pointer), "r"(1) : "memory");
}
__device__ __forceinline__ int acquire_ready(const int* pointer) {
  int result;
  asm volatile("ld.acquire.sys.global.s32 %0, [%1];"
               : "=r"(result) : "l"(pointer) : "memory");
  return result;
}
__global__ void return_scatter(const Params* params) {
  int tile=int(blockIdx.x),row=int(blockIdx.y);
  int key=params->route_key[tile*ROWS+row];
  if (key<0) return;
  if (key>=ROUTES) asm volatile("trap;");
  int source=key/LOCAL_ROUTES,slot=key%LOCAL_ROUTES;
  const float* input=params->down+(size_t(tile)*ROWS+row)*H;
  float* output=params->contribution[source]+size_t(slot)*H;
  for (int feature=int(threadIdx.x);feature<H;feature+=int(blockDim.x))
    output[feature]=input[feature];
  __syncthreads();
  if (threadIdx.x==0) release_ready(params->ready[source]+slot);
}
__global__ void wait_ready(const Params* params) {
  int slot=int(blockIdx.x)*int(blockDim.x)+int(threadIdx.x);
  if (slot>=LOCAL_ROUTES) return;
  while (acquire_ready(params->ready[params->rank]+slot)==0)
    __nanosleep(64);
}
int check(cudaError_t status,const char* operation) {
  if (status==cudaSuccess) return 0;
  std::fprintf(stderr,"%s: %s (%d)\n",operation,cudaGetErrorString(status),int(status));
  return 1;
}
bool read_exact(const char* root,const char* name,
                std::vector<unsigned char>& bytes) {
  char path[512];
  int n=std::snprintf(path,sizeof(path),"%s/%s",root,name);
  if (n<1 || n>=int(sizeof(path))) return false;
  FILE* file=std::fopen(path,"rb");
  if (!file) return false;
  bool okay=std::fread(bytes.data(),1,bytes.size(),file)==bytes.size()
            && std::fgetc(file)==EOF;
  return std::fclose(file)==0 && okay;
}
bool read_owner(const char* root,int owner,const char* name,
                std::vector<unsigned char>& bytes) {
  char path[512];
  int n=std::snprintf(path,sizeof(path),"%s/rank%d/%s",root,owner,name);
  if (n<1 || n>=int(sizeof(path))) return false;
  FILE* file=std::fopen(path,"rb");
  if (!file) return false;
  bool okay=std::fread(bytes.data(),1,bytes.size(),file)==bytes.size()
            && std::fgetc(file)==EOF;
  return std::fclose(file)==0 && okay;
}
bool write_exact(const char* root,const char* name,
                 const void* bytes,size_t extent) {
  char path[512];
  int n=std::snprintf(path,sizeof(path),"%s/%s",root,name);
  if (n<1 || n>=int(sizeof(path))) return false;
  FILE* file=std::fopen(path,"wbx");
  if (!file) return false;
  bool okay=std::fwrite(bytes,1,extent,file)==extent;
  return std::fclose(file)==0 && okay;
}
} // namespace

int main(int argc,char** argv) {
  if (argc!=4) return 2;
  int devices=0;
  if (check(cudaGetDeviceCount(&devices),"device count") || devices!=R)
    return 3;
  std::vector<unsigned char> owner_down[R],owner_keys[R];
  int seen[ROUTES]{};
  int valid_by_owner[R]{};
  for (int owner=0;owner<R;++owner) {
    owner_down[owner].resize(DOWN_BYTES);
    owner_keys[owner].resize(KEY_BYTES);
    if (!read_owner(argv[2],owner,
        "near_full_c147_budget5888.down.actual.fp32",owner_down[owner]) ||
        !read_owner(argv[2],owner,"tile_route_keys.i32",owner_keys[owner]))
      return 4;
    for (int index=0;index<TILES*ROWS;++index) {
      int key=-1;
      std::memcpy(&key,owner_keys[owner].data()+size_t(index)*sizeof(int),
                  sizeof(int));
      if (key<0) continue;
      if (key>=ROUTES || seen[key]++) return 5;
      ++valid_by_owner[owner];
    }
  }
  for (int key=0;key<ROUTES;++key) if (seen[key]!=1) return 6;
  std::vector<unsigned char> weights(ROUTES*sizeof(float));
  if (!read_exact(argv[3],"route_weights.fp32",weights)) return 7;
  float* contributions[R]{};
  float* gpu_down[R]{};
  float* gpu_weights[R]{};
  int* ready[R]{};
  int* gpu_keys[R]{};
  __nv_bfloat16* gpu_output[R]{};
  Params* gpu_params[R]{};
  char device_names[R][128]{};
  int sm_counts[R]{};
  for (int rank=0;rank<R;++rank) {
    if (check(cudaSetDevice(rank),"select rank")) return 8;
    cudaDeviceProp prop{};
    if (check(cudaGetDeviceProperties(&prop,rank),"device properties") ||
        prop.major!=10 || prop.minor!=3 ||
        (std::strcmp(prop.name,"NVIDIA B300") &&
         std::strcmp(prop.name,"NVIDIA B300 SXM6 AC"))) return 9;
    std::snprintf(device_names[rank],sizeof(device_names[rank]),"%s",prop.name);
    sm_counts[rank]=prop.multiProcessorCount;
    for (int peer=0;peer<R;++peer) {
      if (peer==rank) continue;
      int access=0,atomic=0;
      if (check(cudaDeviceCanAccessPeer(&access,rank,peer),"peer access") ||
          check(cudaDeviceGetP2PAttribute(&atomic,cudaDevP2PAttrNativeAtomicSupported,
                                          rank,peer),"native peer atomic") ||
          access!=1 || atomic!=1) return 10;
      cudaError_t status=cudaDeviceEnablePeerAccess(peer,0);
      if (status!=cudaSuccess && status!=cudaErrorPeerAccessAlreadyEnabled) {
        check(status,"enable peer access");return 11;
      }
    }
    if (check(cudaMalloc(&contributions[rank],LOCAL_ROUTES*H*sizeof(float)),
              "contribution malloc") ||
        check(cudaMalloc(&ready[rank],LOCAL_ROUTES*sizeof(int)),"ready malloc") ||
        check(cudaMalloc(&gpu_output[rank],T*H*sizeof(uint16_t)),
              "output malloc") ||
        check(cudaMalloc(&gpu_weights[rank],LOCAL_ROUTES*sizeof(float)),
              "weights malloc") ||
        check(cudaMalloc(&gpu_down[rank],DOWN_BYTES),"down input malloc") ||
        check(cudaMalloc(&gpu_keys[rank],KEY_BYTES),"keys malloc") ||
        check(cudaMalloc(&gpu_params[rank],sizeof(Params)),"params malloc") ||
        check(cudaMemset(contributions[rank],0xff,
                         LOCAL_ROUTES*H*sizeof(float)),"contribution sentinel") ||
        check(cudaMemset(ready[rank],0,LOCAL_ROUTES*sizeof(int)),"ready reset") ||
        check(cudaMemset(gpu_output[rank],0xff,T*H*sizeof(uint16_t)),
              "output sentinel") ||
        check(cudaMemcpy(gpu_weights[rank],
            weights.data()+size_t(rank)*LOCAL_ROUTES*sizeof(float),
            LOCAL_ROUTES*sizeof(float),cudaMemcpyHostToDevice),
            "weights H2D") ||
        check(cudaMemcpy(gpu_down[rank],owner_down[rank].data(),DOWN_BYTES,
                         cudaMemcpyHostToDevice),"down H2D") ||
        check(cudaMemcpy(gpu_keys[rank],owner_keys[rank].data(),KEY_BYTES,
                         cudaMemcpyHostToDevice),"keys H2D")) return 12;
  }
  for (int rank=0;rank<R;++rank) {
    if (check(cudaSetDevice(rank),"select params rank")) return 13;
    Params params{};
    for (int peer=0;peer<R;++peer) {
      params.contribution[peer]=contributions[peer];
      params.ready[peer]=ready[peer];
    }
    params.down=gpu_down[rank];params.route_key=gpu_keys[rank];params.rank=rank;
    if (check(cudaMemcpy(gpu_params[rank],&params,sizeof(params),
                         cudaMemcpyHostToDevice),"params H2D")) return 14;
  }
  for (int rank=0;rank<R;++rank) {
    if (check(cudaSetDevice(rank),"select scatter rank")) return 15;
    return_scatter<<<dim3(TILES,ROWS),256>>>(gpu_params[rank]);
    if (check(cudaGetLastError(),"scatter launch")) return 16;
  }
  for (int rank=0;rank<R;++rank) {
    if (check(cudaSetDevice(rank),"select acquire rank")) return 17;
    wait_ready<<<(LOCAL_ROUTES+255)/256,256>>>(gpu_params[rank]);
    if (check(cudaGetLastError(),"acquire launch")) return 18;
    cake_weave_rank512_combine_kernel<<<dim3(T,8),256>>>(
        contributions[rank],gpu_weights[rank],gpu_output[rank]);
    if (check(cudaGetLastError(),"Cake combine launch")) return 19;
  }
  for (int rank=0;rank<R;++rank) {
    if (check(cudaSetDevice(rank),"select completion rank") ||
        check(cudaDeviceSynchronize(),"return and combine completion"))
      return 20;
  }
  std::vector<unsigned char> actual_contrib(CONTRIBUTION_BYTES);
  std::vector<unsigned char> actual_output(OUTPUT_BYTES);
  std::vector<int> actual_ready(ROUTES);
  std::vector<unsigned char> observed_keys(R*KEY_BYTES);
  for (int rank=0;rank<R;++rank) {
    if (check(cudaSetDevice(rank),"select readback rank") ||
        check(cudaMemcpy(actual_contrib.data()+
                         size_t(rank)*LOCAL_ROUTES*H*sizeof(float),
                         contributions[rank],LOCAL_ROUTES*H*sizeof(float),
                         cudaMemcpyDeviceToHost),"contribution D2H") ||
        check(cudaMemcpy(actual_output.data()+size_t(rank)*T*H*2,
                         gpu_output[rank],T*H*2,cudaMemcpyDeviceToHost),
              "output D2H") ||
        check(cudaMemcpy(actual_ready.data()+size_t(rank)*LOCAL_ROUTES,
                         ready[rank],LOCAL_ROUTES*sizeof(int),
                         cudaMemcpyDeviceToHost),"ready D2H") ||
        check(cudaMemcpy(observed_keys.data()+size_t(rank)*KEY_BYTES,
                         gpu_keys[rank],KEY_BYTES,cudaMemcpyDeviceToHost),
              "keys D2H")) return 21;
  }
  for (int flag:actual_ready) if (flag!=1) return 22;
  if (!write_exact(argv[1],"contributions.fp32",actual_contrib.data(),
                   actual_contrib.size()) ||
      !write_exact(argv[1],"output.bf16",actual_output.data(),
                   actual_output.size()) ||
      !write_exact(argv[1],"ready.i32",actual_ready.data(),
                   actual_ready.size()*sizeof(int)) ||
      !write_exact(argv[1],"observed_keys.i32",observed_keys.data(),
                   observed_keys.size())) return 23;
  char path[512];
  int n=std::snprintf(path,sizeof(path),"%s/device_report.json",argv[1]);
  if (n<1 || n>=int(sizeof(path))) return 24;
  FILE* file=std::fopen(path,"wx");
  if (!file) return 25;
  std::fprintf(file,"{\"target\":\"sm_103a\",\"ranks\":4,"
                    "\"valid_routes_by_owner\":[%d,%d,%d,%d],"
                    "\"sm_counts\":[%d,%d,%d,%d],"
                    "\"device_names\":[\"%s\",\"%s\",\"%s\",\"%s\"]}\n",
               valid_by_owner[0],valid_by_owner[1],valid_by_owner[2],
               valid_by_owner[3],sm_counts[0],sm_counts[1],sm_counts[2],
               sm_counts[3],device_names[0],device_names[1],
               device_names[2],device_names[3]);
  if (std::fclose(file)) return 26;
  for (int rank=0;rank<R;++rank) {
    cudaSetDevice(rank);cudaFree(gpu_params[rank]);cudaFree(gpu_keys[rank]);
    cudaFree(gpu_down[rank]);cudaFree(gpu_weights[rank]);
    cudaFree(gpu_output[rank]);cudaFree(ready[rank]);
    cudaFree(contributions[rank]);
  }
  std::printf("returned %d route contributions to four source ranks\n",ROUTES);
  return 0;
}
