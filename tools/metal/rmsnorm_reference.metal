// Handwritten mathematical references, authored under known-kernel reproduction.
// Consumer substitutes __COLUMNS__ from the fixed weighted RMSNorm contract.
// These are never Compiler-generated Lowerings or an old-Compiler RMSNorm baseline.
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)
constant uint columns = __COLUMNS__;

kernel void reference_serial(device const float* x [[buffer(0)]],
                             device const float* weight [[buffer(1)]],
                             device float* out [[buffer(2)]],
                             uint row [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_threadgroup]]) {
    if (lane != 0) return;
    float total = 0.0f;
    for (uint c = 0; c < columns; ++c) {
        float value = x[ulong(row) * columns + c];
        total += value * value;
    }
    float inverse = precise::rsqrt(total / float(columns) + 0.00001f);
    for (uint c = 0; c < columns; ++c)
        out[ulong(row) * columns + c] = (x[ulong(row) * columns + c] * inverse) * weight[c];
}

kernel void reference_simd(device const float* x [[buffer(0)]],
                           device const float* weight [[buffer(1)]],
                           device float* out [[buffer(2)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_threadgroup]]) {
    float partial = 0.0f;
    for (uint c = lane; c < columns; c += 32) {
        float value = x[ulong(row) * columns + c];
        partial += value * value;
    }
    float total = simd_sum(partial);
    float inverse = precise::rsqrt(total / float(columns) + 0.00001f);
    for (uint c = lane; c < columns; c += 32)
        out[ulong(row) * columns + c] = (x[ulong(row) * columns + c] * inverse) * weight[c];
}
