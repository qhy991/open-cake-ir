#include <metal_stdlib>

using namespace metal;

constant uint kPartialCapacity = 32;
constant uint kBroadcastSlot = 32;
constant float kEpoch0DirtySentinel = -12345.25f;
constant float kEpoch1DirtySentinel = 9876.5f;

// This kernel is deliberately a finite engineering probe, not a Compiler lowering.
// The host admits it only when the selected pipeline reports a 32-thread SIMDgroup.
kernel void hierarchical_reduction_w32_probe(
    device const float *epoch0Input [[buffer(0)]],
    device const float *epoch1Input [[buffer(1)]],
    device float *broadcastOutputs [[buffer(2)]],
    device float *ownerResults [[buffer(3)]],
    device float *dirtySnapshots [[buffer(4)]],
    threadgroup float *scratch [[threadgroup(0)]],
    uint threadIndex [[thread_index_in_threadgroup]],
    uint laneIndex [[thread_index_in_simdgroup]],
    uint simdgroupIndex [[simdgroup_index_in_threadgroup]],
    uint simdgroupCount [[simdgroups_per_threadgroup]],
    uint3 threadsPerThreadgroup [[threads_per_threadgroup]])
{
    // Dirty every partial slot, including slots outside the live SIMDgroup count.
    // This makes a missing validity predicate observable instead of relying on
    // unspecified initial threadgroup-memory contents.
    if (threadIndex < kPartialCapacity) {
        scratch[threadIndex] = kEpoch0DirtySentinel;
    }
    if (threadIndex == 0) {
        scratch[kBroadcastSlot] = kEpoch0DirtySentinel;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (threadIndex == 0) {
        dirtySnapshots[0] = scratch[0];
        dirtySnapshots[1] = scratch[kBroadcastSlot];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float partial = simd_sum(epoch0Input[threadIndex]);
    if (laneIndex == 0) {
        scratch[simdgroupIndex] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroupIndex == 0) {
        float finalInput = laneIndex < simdgroupCount ? scratch[laneIndex] : 0.0f;
        float finalSum = simd_sum(finalInput);
        if (laneIndex == 0) {
            scratch[kBroadcastSlot] = finalSum;
            ownerResults[0] = finalSum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    broadcastOutputs[threadIndex] = scratch[kBroadcastSlot];

    // Every live thread has consumed the epoch-0 scalar before any slot is reused.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (threadIndex < kPartialCapacity) {
        scratch[threadIndex] = kEpoch1DirtySentinel;
    }
    if (threadIndex == 0) {
        scratch[kBroadcastSlot] = kEpoch1DirtySentinel;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (threadIndex == 0) {
        dirtySnapshots[2] = scratch[0];
        dirtySnapshots[3] = scratch[kBroadcastSlot];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    partial = simd_sum(epoch1Input[threadIndex]);
    if (laneIndex == 0) {
        scratch[simdgroupIndex] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroupIndex == 0) {
        float finalInput = laneIndex < simdgroupCount ? scratch[laneIndex] : 0.0f;
        float finalSum = simd_sum(finalInput);
        if (laneIndex == 0) {
            scratch[kBroadcastSlot] = finalSum;
            ownerResults[1] = finalSum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    broadcastOutputs[threadsPerThreadgroup.x + threadIndex] = scratch[kBroadcastSlot];
}
