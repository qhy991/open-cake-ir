#include <metal_stdlib>

using namespace metal;

// Capability probe only: one SIMD-group multiplies two row-major 8x8 BF16
// matrices and writes the FP32 accumulator. The host supplies an identity lhs
// and a row-major rhs containing 1...64, so the output must reproduce the rhs.
kernel void open_cake_bfloat_simdgroup_mma_probe(
    device const bfloat *lhs [[buffer(0)]],
    device const bfloat *rhs [[buffer(1)]],
    device float *output [[buffer(2)]]) {
  simdgroup_bfloat8x8 lhs_matrix;
  simdgroup_bfloat8x8 rhs_matrix;
  simdgroup_float8x8 accumulator =
      make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  simdgroup_load(lhs_matrix, lhs, 8);
  simdgroup_load(rhs_matrix, rhs, 8);
  simdgroup_multiply_accumulate(
      accumulator, lhs_matrix, rhs_matrix, accumulator);
  simdgroup_store(accumulator, output, 8);
}
