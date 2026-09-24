// B300 scheduling mechanism probe. This is a direct CUDA reference, not Cake lowering.
// No cross-GPU communication or MoE numerical claim follows from this program.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

constexpr unsigned kWidth = 16;
constexpr unsigned kWarp = 32;
constexpr int kTargetSms = 148;

enum Counter : unsigned {
    kDispatchCursor = 0,
    kDispatchCompleted = 1,
    kComputeCursor = 2,
    kCombineCursor = 3,
    kStealPermit = 4,
    kCommPhaseDone = 5,
    kComputeCompleted = 6,
};
enum Statistic : unsigned {
    kStolen = 0,
    kComputedByCompute = 1,
    kCombinedByCompute = 2,
    kFirstCombineComputeDone = 3,
};

__device__ __forceinline__ unsigned claim(unsigned* pointer) {
    unsigned old;
    asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;"
                 : "=r"(old)
                 : "l"(reinterpret_cast<unsigned long long>(pointer)), "r"(1u)
                 : "memory");
    return old;
}

__device__ __forceinline__ unsigned load_relaxed(const unsigned* pointer) {
    unsigned value;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];"
                 : "=r"(value)
                 : "l"(reinterpret_cast<unsigned long long>(pointer))
                 : "memory");
    return value;
}

__device__ __forceinline__ unsigned load_acquire(const unsigned* pointer) {
    unsigned value;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                 : "=r"(value)
                 : "l"(reinterpret_cast<unsigned long long>(pointer))
                 : "memory");
    return value;
}

__device__ __forceinline__ void publish(unsigned* pointer) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;"
                 :: "l"(reinterpret_cast<unsigned long long>(pointer)), "r"(1u)
                 : "memory");
}

__device__ __forceinline__ void wait_for(const unsigned* pointer) {
    while (load_acquire(pointer) == 0) {
        __nanosleep(64);
    }
}

__device__ __forceinline__ unsigned claim_for_warp(unsigned* pointer, unsigned lane) {
    unsigned tile = lane == 0 ? claim(pointer) : 0;
    return __shfl_sync(0xffffffffu, tile, 0);
}

__device__ __forceinline__ void dispatch_tile(
    unsigned tile, unsigned lane, const float* input, float* staged,
    unsigned* dispatch_ready, unsigned* counters, unsigned* dispatch_executions) {
    if (lane < kWidth) {
        staged[tile * kWidth + lane] = input[tile * kWidth + lane];
    }
    __syncwarp();
    if (lane == 0) {
        publish(dispatch_ready + tile);
        claim(dispatch_executions + tile);
        claim(counters + kDispatchCompleted);
    }
    __syncwarp();
}

__device__ __forceinline__ void compute_tile(
    unsigned tile, unsigned lane, const float* weights, const float* staged,
    float* computed, const unsigned* dispatch_ready, unsigned* compute_ready,
    unsigned* chunk_done, unsigned* counters, unsigned* compute_executions,
    unsigned* statistics,
    unsigned tiles_per_chunk, int repeats, bool stolen) {
    wait_for(dispatch_ready + tile);
    if (lane < kWidth) {
        float value = 0.0f;
        for (int repeat = 0; repeat < repeats; ++repeat) {
            for (unsigned k = 0; k < kWidth; ++k) {
                value = fmaf(staged[tile * kWidth + k], weights[k * kWidth + lane], value);
            }
        }
        computed[tile * kWidth + lane] = value;
    }
    __syncwarp();
    if (lane == 0) {
        publish(compute_ready + tile);
        claim(compute_executions + tile);
        claim(chunk_done + tile / tiles_per_chunk);
        claim(counters + kComputeCompleted);
        claim(statistics + (stolen ? kStolen : kComputedByCompute));
    }
    __syncwarp();
}

__device__ __forceinline__ void combine_tile(
    unsigned tile, unsigned lane, const float* computed, float* output,
    const unsigned* compute_ready, const unsigned* chunk_done,
    const unsigned* counters, unsigned* combine_executions,
    unsigned* statistics,
    unsigned tiles_per_chunk, bool compute_worker) {
    const unsigned chunk = tile / tiles_per_chunk;
    while (load_relaxed(chunk_done + chunk) < tiles_per_chunk) {
        __nanosleep(64);
    }
    wait_for(compute_ready + tile);
    if (lane == 0) {
        atomicCAS(statistics + kFirstCombineComputeDone, 0xffffffffu,
                  load_relaxed(counters + kComputeCompleted));
    }
    if (lane < kWidth) {
        output[tile * kWidth + lane] = computed[tile * kWidth + lane];
    }
    __syncwarp();
    if (lane == 0) {
        claim(combine_executions + tile);
        if (compute_worker) {
            claim(statistics + kCombinedByCompute);
        }
    }
    __syncwarp();
}

__global__ void worker_pipeline(
    const float* input, const float* weights, float* staged, float* computed,
    float* output, unsigned* dispatch_ready, unsigned* compute_ready,
    unsigned* chunk_done, unsigned* counters, unsigned* statistics,
    unsigned* dispatch_executions, unsigned* compute_executions,
    unsigned* combine_executions, const int* runtime_plan, int* status,
    unsigned* role_trace, int tile_count, int repeats) {
    const unsigned lane = threadIdx.x;
    const unsigned c = static_cast<unsigned>(runtime_plan[0]);
    const unsigned chunks = static_cast<unsigned>(runtime_plan[1]);
    const unsigned max_steal = static_cast<unsigned>(runtime_plan[2]);
    const unsigned tiles = static_cast<unsigned>(tile_count);
    if (c == 0 || c >= gridDim.x || chunks == 0 || chunks > 8 ||
        runtime_plan[2] < 0 ||
        tiles == 0 || tiles % chunks != 0 || repeats <= 0) {
        if (blockIdx.x == 0 && lane == 0) {
            *status = -1;
        }
        return;
    }
    const unsigned tiles_per_chunk = tiles / chunks;
    const bool communication_worker = blockIdx.x < c;
    if (lane == 0) {
        role_trace[blockIdx.x] = communication_worker ? 1u : 0u;
    }

    if (communication_worker) {
        while (true) {
            const unsigned tile = claim_for_warp(counters + kDispatchCursor, lane);
            if (tile >= tiles) break;
            dispatch_tile(tile, lane, input, staged, dispatch_ready, counters,
                          dispatch_executions);
        }
        if (lane == 0) {
            while (load_relaxed(counters + kDispatchCompleted) < tiles) {
                __nanosleep(64);
            }
        }
        __syncwarp();
        while (true) {
            unsigned stop = 0;
            if (lane == 0) {
                stop = load_relaxed(chunk_done) >= tiles_per_chunk ||
                       claim(counters + kStealPermit) >= max_steal;
            }
            stop = __shfl_sync(0xffffffffu, stop, 0);
            if (stop) break;
            const unsigned tile = claim_for_warp(counters + kComputeCursor, lane);
            if (tile >= tiles) break;
            compute_tile(tile, lane, weights, staged, computed, dispatch_ready,
                         compute_ready, chunk_done, counters,
                         compute_executions, statistics,
                         tiles_per_chunk, repeats, true);
        }
        if (lane == 0) {
            claim(counters + kCommPhaseDone);
        }
        __syncwarp();
    } else {
        while (true) {
            const unsigned tile = claim_for_warp(counters + kComputeCursor, lane);
            if (tile >= tiles) break;
            compute_tile(tile, lane, weights, staged, computed, dispatch_ready,
                         compute_ready, chunk_done, counters,
                         compute_executions, statistics,
                         tiles_per_chunk, repeats, false);
        }
    }

    if (lane == 0) {
        while (load_relaxed(counters + kCommPhaseDone) < c) {
            __nanosleep(64);
        }
    }
    __syncwarp();
    while (true) {
        const unsigned tile = claim_for_warp(counters + kCombineCursor, lane);
        if (tile >= tiles) break;
        combine_tile(tile, lane, computed, output, compute_ready, chunk_done,
                     counters, combine_executions, statistics, tiles_per_chunk,
                     !communication_worker);
    }
}

}  // namespace

// Buffer order is part of the standalone test ABI, not a Cake Program contract.
extern "C" int weave_worker_launch(void** buffers, int tile_count, int repeats,
                                     void* stream) {
    if (!buffers || tile_count <= 0 || repeats <= 0) return int(cudaErrorInvalidValue);
    for (int i = 0; i < 16; ++i) {
        if (!buffers[i]) return int(cudaErrorInvalidValue);
    }
    int device = -1;
    cudaError_t result = cudaGetDevice(&device);
    if (result != cudaSuccess) return int(result);
    cudaDeviceProp properties{};
    result = cudaGetDeviceProperties(&properties, device);
    if (result != cudaSuccess) return int(result);
    if (std::strcmp(properties.name, "NVIDIA B300") != 0 ||
        properties.major != 10 || properties.minor != 3 ||
        properties.multiProcessorCount != kTargetSms) return int(cudaErrorInvalidDevice);
    int cooperative = 0;
    result = cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device);
    if (result != cudaSuccess || cooperative != 1) return int(cudaErrorNotSupported);
    int resident_blocks = 0;
    result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident_blocks, worker_pipeline, kWarp, 0);
    if (result != cudaSuccess || resident_blocks < 1) return int(cudaErrorCooperativeLaunchTooLarge);
    void* arguments[] = {&buffers[0], &buffers[1], &buffers[2], &buffers[3],
                         &buffers[4], &buffers[5], &buffers[6], &buffers[7],
                         &buffers[8], &buffers[9], &buffers[10], &buffers[11],
                         &buffers[12], &buffers[13], &buffers[14],
                         &buffers[15],
                         &tile_count, &repeats};
    result = cudaLaunchCooperativeKernel(
        reinterpret_cast<const void*>(worker_pipeline), dim3(kTargetSms),
        dim3(kWarp), arguments, 0, reinterpret_cast<cudaStream_t>(stream));
    return int(result);
}
