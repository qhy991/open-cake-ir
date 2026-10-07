# Rank-two register transpose

`lm.transpose(value)` exchanges the two axes of one resident register value. It maps
`result[j,i] = value[i,j]`, preserves FP32/FP16/BF16/INT32 dtype, and performs no arithmetic
conversion. The canonical operation is `transpose` with empty parameters. The result
shape is the reversed input shape; explicit result buffers obey the same rule.

This is a value permutation, not a global-memory stride change, shared-memory layout
algebra, reinterpretation, or promise of free lane exchange. Loads and stores retain
explicit AccessMaps and masks. The operation reads one register value and writes one
fresh SSA register result. Existing type, single-writer, role, lifetime, input-effect and
loop-escape checks continue to apply. FP8 and other ranks are outside this first slice.

## Why an explicit value operation

The existing MMA contract is `A[M,K] * B[N,K]`. A loaded global B[K,N] tile has the other
orientation. A square tile can hide a mistaken subscript; changing the MMA carry rule
without actually permuting values would implement a different matrix product.
`broadcast_in_dim` inserts/expands dimensions and requires ordered dimension positions;
it does not represent a permutation. The transpose produces the required B[N,K] value
while preserving original global B[K,N] indexing and the existing MMA precision contract.

## Analyses and lowering

The shared value-typing owner derives reversed shape and unchanged dtype. The Python
frontend infers that result or checks an explicit buffer. Work accounting records zero
floating-point arithmetic; it does not model physical lane exchange or latency.
Both contraction-loop provenance and argmin-domain tracing follow the same transpose
axis swap through unique, well-typed cast/transpose chains. Cycles, ambiguous writers,
wrong shapes/storage/dtypes provide no transpose proof. A K-loop remains an accumulation;
an output-axis loop does not become a contraction merely because a tile was transposed.

The proposed gfx938 Target admits this value operation; other Targets remain unchanged.
Triton emits `tl.trans(value)` without a cast, outside or inside already-supported loops.
The existing nested-MMA slice still requires directly loaded operands and refuses cast
or transposed operands. This work does not widen that separate emission domain, change
FP32 to TF32, or promise MMAC selection for shapes such as M=1.

## Acceptance scope

Software cases cover non-square copies with masked tails, double transpose, dtype
preservation, transposed B[K,N] with K-tail accumulation, and argmin-domain coordinate
tracing using a CPU semantic fixture. The Corpus has one tail-copy case and one
non-square K-loop MMA case. Device qualification must include bit-preserving copies
and original-contract contraction cases on the exact DTK route before hardware
promotion. Component correctness does not establish a faster GEMM, attention result or
same-budget authoring improvement.
