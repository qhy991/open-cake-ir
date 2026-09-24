// Four-GPU BF16 EP tail-case development successor, not Cake Program lowering.
// The previously measured T=8 source remains ep4_mailbox.cu unchanged.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "chunk_math.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {
constexpr int R = 4;
constexpr int T = 7;
constexpr int K = 2;
constexpr int E = 8;
constexpr int H = 16;
constexpr int I = 32;
constexpr int CAP = R * T * K;
constexpr int SMS = 148;

struct Mailbox {
    int tail, head, dispatch_sources_done, dispatch_cursor;
    int dispatch_completed, comm_done, combine_cursor, steal_permits;
    int stolen, compute_completed, status;
    int ready[CAP];
    int source[CAP], token[CAP], route_slot[CAP], expert[CAP];
    __nv_bfloat16 payload[CAP * H];
    float contributions[T * K * H];
    int contribution_ready[T * K];
    int chunk_completed[T];
    __nv_bfloat16 output[T * H];
};

struct Params {
    Mailbox* mailboxes[R];
    const __nv_bfloat16* hidden;
    const int* expert_ids;
    const float* route_weights;
    const __nv_bfloat16* w_up_gate;
    const __nv_bfloat16* w_down;
    int rank, comm_ctas, chunks, steal_budget;
    int source_chunks[R];
};

__device__ __forceinline__ int sys_add_relaxed(int* pointer, int value) {
    int old;
    asm volatile("atom.relaxed.sys.global.add.s32 %0, [%1], %2;"
                 : "=r"(old)
                 : "l"(reinterpret_cast<unsigned long long>(pointer)), "r"(value)
                 : "memory");
    return old;
}

__device__ __forceinline__ int sys_add_acq_rel(int* pointer, int value) {
    int old;
    asm volatile("atom.acq_rel.sys.global.add.s32 %0, [%1], %2;"
                 : "=r"(old)
                 : "l"(reinterpret_cast<unsigned long long>(pointer)), "r"(value)
                 : "memory");
    return old;
}

__device__ __forceinline__ int sys_add_release(int* pointer, int value) {
    int old;
    asm volatile("atom.release.sys.global.add.s32 %0, [%1], %2;"
                 : "=r"(old)
                 : "l"(reinterpret_cast<unsigned long long>(pointer)), "r"(value)
                 : "memory");
    return old;
}

__device__ __forceinline__ int sys_load_acquire(const int* pointer) {
    int value;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];"
                 : "=r"(value)
                 : "l"(reinterpret_cast<unsigned long long>(pointer))
                 : "memory");
    return value;
}

__device__ __forceinline__ int flag_load_acquire(const int* pointer) {
    return sys_load_acquire(pointer);
}

__device__ __forceinline__ void flag_publish(int* pointer) {
    asm volatile("st.release.sys.global.s32 [%0], %1;" ::
                 "l"(reinterpret_cast<unsigned long long>(pointer)), "r"(1)
                 : "memory");
}

__device__ __forceinline__ int warp_claim(int* cursor, int lane) {
    int task = lane == 0 ? atomicAdd(cursor, 1) : 0;
    return __shfl_sync(0xffffffffu, task, 0);
}

__device__ __forceinline__ void dispatch_one(const Params* params, int task,
                                              int lane) {
    const int token = task / K;
    const int route = task % K;
    const int expert = params->expert_ids[task];
    const int destination = expert / (E / R);
    Mailbox* remote = params->mailboxes[destination];
    int slot = lane == 0 ? sys_add_relaxed(&remote->tail, 1) : 0;
    slot = __shfl_sync(0xffffffffu, slot, 0);
    if (lane == 0) {
        remote->source[slot] = params->rank;
        remote->token[slot] = token;
        remote->route_slot[slot] = route;
        remote->expert[slot] = expert;
    }
    if (lane < H)
        remote->payload[slot * H + lane] = params->hidden[token * H + lane];
    __syncwarp();
    if (lane == 0) {
        flag_publish(&remote->ready[slot]);
        const int old = sys_add_acq_rel(&params->mailboxes[params->rank]->dispatch_completed, 1);
        if (old == T * K - 1) {
            for (int peer = 0; peer < R; ++peer)
                sys_add_release(&params->mailboxes[peer]->dispatch_sources_done, 1);
        }
    }
    __syncwarp();
}

__device__ __forceinline__ int try_claim(Mailbox* local) {
    while (true) {
        const int head = atomicAdd(&local->head, 0);
        const int tail = sys_load_acquire(&local->tail);
        if (head >= tail) return -1;
        if (atomicCAS(&local->head, head, head + 1) == head) return head;
    }
}

__device__ __forceinline__ bool no_more_inbound(Mailbox* local) {
    if (sys_load_acquire(&local->dispatch_sources_done) != R) return false;
    const int tail = sys_load_acquire(&local->tail);
    return atomicAdd(&local->head, 0) >= tail;
}

__device__ __forceinline__ void compute_one(const Params* params, int slot,
                                             int lane) {
    Mailbox* local = params->mailboxes[params->rank];
    while (flag_load_acquire(&local->ready[slot]) == 0) __nanosleep(64);
    const int expert = local->expert[slot];
    const int local_expert = expert - params->rank * (E / R);
    const int source = local->source[slot];
    const int token = local->token[slot];
    const int route = local->route_slot[slot];
    __shared__ float activated[I];
    if (lane < I) {
        float x1 = 0.0f, x2 = 0.0f;
        for (int h = 0; h < H; ++h) {
            const float x = __bfloat162float(local->payload[slot * H + h]);
            const int base = local_expert * (2 * I * H);
            x1 = fmaf(x, __bfloat162float(params->w_up_gate[base + lane * H + h]), x1);
            x2 = fmaf(x, __bfloat162float(params->w_up_gate[
                base + (I + lane) * H + h]), x2);
        }
        activated[lane] = x1 * (x2 / (1.0f + expf(-x2)));
    }
    __syncthreads();
    Mailbox* origin = params->mailboxes[source];
    const int contribution = token * K + route;
    if (lane < H) {
        float value = 0.0f;
        for (int i = 0; i < I; ++i) {
            const int index = local_expert * (H * I) + lane * I + i;
            value = fmaf(activated[i], __bfloat162float(params->w_down[index]), value);
        }
        origin->contributions[contribution * H + lane] = value;
    }
    __syncthreads();
    if (lane == 0) {
        flag_publish(&origin->contribution_ready[contribution]);
        sys_add_relaxed(&origin->chunk_completed[
            cake_weave::chunk_for_token<T>(token, params->source_chunks[source])], 1);
        atomicAdd(&local->compute_completed, 1);
    }
    __syncthreads();
}

__device__ __forceinline__ void combine_one(const Params* params, int token,
                                             int lane) {
    Mailbox* local = params->mailboxes[params->rank];
    const int chunk = cake_weave::chunk_for_token<T>(token, params->chunks);
    while (sys_load_acquire(&local->chunk_completed[chunk]) <
           cake_weave::chunk_size<T>(chunk, params->chunks) * K) __nanosleep(64);
    for (int route = 0; route < K; ++route)
        while (flag_load_acquire(&local->contribution_ready[token * K + route]) == 0)
            __nanosleep(64);
    if (lane < H) {
        float value = 0.0f;
        for (int route = 0; route < K; ++route) {
            const int index = (token * K + route) * H + lane;
            value = fmaf(local->contributions[index],
                         params->route_weights[token * K + route], value);
        }
        local->output[token * H + lane] = __float2bfloat16_rn(value);
    }
    __syncwarp();
}

__global__ void ep4_mailbox(Params* params) {
    const int lane = int(threadIdx.x);
    Mailbox* local = params->mailboxes[params->rank];
    if (params->comm_ctas <= 0 || params->comm_ctas >= SMS ||
        params->chunks <= 0 || params->chunks > T ||
        params->steal_budget < 0 || params->steal_budget > CAP) {
        if (blockIdx.x == 0 && lane == 0) local->status = -1;
        return;
    }
    const bool communication = int(blockIdx.x) < params->comm_ctas;
    if (communication) {
        while (true) {
            const int task = warp_claim(&local->dispatch_cursor, lane);
            if (task >= T * K) break;
            dispatch_one(params, task, lane);
        }
        if (lane == 0)
            while (sys_load_acquire(&local->dispatch_completed) < T * K)
                __nanosleep(64);
        __syncwarp();
        while (true) {
            int stop = 0;
            if (lane == 0) {
                stop = sys_load_acquire(&local->chunk_completed[0]) >=
                       cake_weave::chunk_size<T>(0, params->chunks) * K ||
                       atomicAdd(&local->steal_permits, 1) >= params->steal_budget;
            }
            stop = __shfl_sync(0xffffffffu, stop, 0);
            if (stop) break;
            int slot = lane == 0 ? try_claim(local) : 0;
            slot = __shfl_sync(0xffffffffu, slot, 0);
            if (slot >= 0) {
                compute_one(params, slot, lane);
                if (lane == 0) atomicAdd(&local->stolen, 1);
            } else if (no_more_inbound(local)) break;
        }
        if (lane == 0) atomicAdd(&local->comm_done, 1);
        __syncwarp();
    } else {
        while (true) {
            int slot = lane == 0 ? try_claim(local) : 0;
            slot = __shfl_sync(0xffffffffu, slot, 0);
            if (slot >= 0) {
                compute_one(params, slot, lane);
            } else if (no_more_inbound(local)) break;
            else __nanosleep(64);
        }
    }
    if (lane == 0)
        while (atomicAdd(&local->comm_done, 0) < params->comm_ctas)
            __nanosleep(64);
    __syncwarp();
    while (true) {
        const int token = warp_claim(&local->combine_cursor, lane);
        if (token >= T) break;
        combine_one(params, token, lane);
    }
}
}  // namespace

extern "C" size_t weave_ep4_mailbox_bytes() { return sizeof(Mailbox); }
extern "C" int weave_ep4_tokens() { return T; }
extern "C" size_t weave_ep4_output_offset() { return offsetof(Mailbox, output); }
extern "C" size_t weave_ep4_output_bytes() { return T * H * sizeof(__nv_bfloat16); }
extern "C" size_t weave_ep4_status_offset() { return offsetof(Mailbox, status); }
extern "C" size_t weave_ep4_stolen_offset() { return offsetof(Mailbox, stolen); }
extern "C" size_t weave_ep4_compute_completed_offset() {
    return offsetof(Mailbox, compute_completed);
}
extern "C" size_t weave_ep4_dispatch_done_offset() {
    return offsetof(Mailbox, dispatch_sources_done);
}

extern "C" int weave_ep4_prepare_rank(int rank, void** mailboxes,
    const void* hidden, const void* expert_ids, const void* route_weights,
    const void* w_up_gate, const void* w_down,
    int comm_ctas, int chunks, int steal_budget,
    const int* source_chunks, void** result) {
    if (rank < 0 || rank >= R || !mailboxes || !hidden || !expert_ids ||
        !route_weights || !w_up_gate || !w_down || !source_chunks || !result)
        return int(cudaErrorInvalidValue);
    if (chunks < 1 || chunks > T || source_chunks[rank] != chunks)
        return int(cudaErrorInvalidValue);
    Params host{};
    for (int peer = 0; peer < R; ++peer) {
        if (!mailboxes[peer] || source_chunks[peer] < 1 || source_chunks[peer] > T)
            return int(cudaErrorInvalidValue);
        host.mailboxes[peer] = static_cast<Mailbox*>(mailboxes[peer]);
        host.source_chunks[peer] = source_chunks[peer];
    }
    host.hidden = static_cast<const __nv_bfloat16*>(hidden);
    host.expert_ids = static_cast<const int*>(expert_ids);
    host.route_weights = static_cast<const float*>(route_weights);
    host.w_up_gate = static_cast<const __nv_bfloat16*>(w_up_gate);
    host.w_down = static_cast<const __nv_bfloat16*>(w_down);
    host.rank = rank; host.comm_ctas = comm_ctas;
    host.chunks = chunks; host.steal_budget = steal_budget;
    *result = nullptr;
    void* pointer = nullptr;
    cudaError_t error = cudaMalloc(&pointer, sizeof(Params));
    if (error != cudaSuccess) return int(error);
    error = cudaMemcpy(pointer, &host, sizeof(Params), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) { cudaFree(pointer); return int(error); }
    *result = pointer;
    return 0;
}

extern "C" int weave_ep4_launch(void** parameters) {
    if (!parameters) return int(cudaErrorInvalidValue);
    int count = 0;
    cudaError_t error = cudaGetDeviceCount(&count);
    if (error != cudaSuccess || count != R) return int(cudaErrorInvalidDevice);
    for (int rank = 0; rank < R; ++rank) {
        error = cudaSetDevice(rank);
        if (error != cudaSuccess) return int(error);
        cudaDeviceProp prop{};
        error = cudaGetDeviceProperties(&prop, rank);
        if (error != cudaSuccess) return int(error);
        if ((std::strcmp(prop.name, "NVIDIA B300") != 0 &&
             std::strcmp(prop.name, "NVIDIA B300 SXM6 AC") != 0) ||
            prop.major != 10 || prop.minor != 3 ||
            prop.multiProcessorCount != SMS) return int(cudaErrorInvalidDevice);
        int cooperative = 0;
        error = cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, rank);
        if (error != cudaSuccess || cooperative != 1) return int(cudaErrorNotSupported);
        int resident = 0;
        error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &resident, ep4_mailbox, 32, 0);
        if (error != cudaSuccess || resident < 1) return int(cudaErrorCooperativeLaunchTooLarge);
        for (int peer = 0; peer < R; ++peer) {
            if (peer == rank) continue;
            int access = 0, native_atomic = 0;
            error = cudaDeviceCanAccessPeer(&access, rank, peer);
            if (error != cudaSuccess || access != 1) return int(cudaErrorNotSupported);
            error = cudaDeviceGetP2PAttribute(&native_atomic,
                cudaDevP2PAttrNativeAtomicSupported, rank, peer);
            if (error != cudaSuccess || native_atomic != 1)
                return int(cudaErrorNotSupported);
            error = cudaDeviceEnablePeerAccess(peer, 0);
            if (error != cudaSuccess && error != cudaErrorPeerAccessAlreadyEnabled)
                return int(error);
        }
    }
    for (int rank = 0; rank < R; ++rank) {
        error = cudaSetDevice(rank);
        if (error != cudaSuccess) return int(error);
        void* args[] = {&parameters[rank]};
        error = cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(ep4_mailbox),
            dim3(SMS), dim3(32), args, 0, nullptr);
        if (error != cudaSuccess) return int(error);
    }
    return 0;
}
