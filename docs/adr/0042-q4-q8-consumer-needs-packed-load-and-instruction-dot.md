# ADR 0042: Q4×Q8 consumer needs packed Load and instruction-bound Dot

Status: proposed design only. Implementation is gated on Kernel A producing all 576
Q8_1 bytes correctly on the exact gfx1151 Executor; no consumer or performance claim
exists yet.

## Context

ADR 0041 implements the live Q8_1 producer required by the frozen
`llama-q4_0-q8_1-mmvq-f32-conformance-v1` Workload. The next kernel consumes one raw
Q4_0 record and record zero of that Q8_1 workspace. The Compiler already owns both
packed ABIs, little-endian fields, Q4 low-then-high nibble order, scalar dtypes,
register reshape, FP32 arithmetic/reduction and Store. Adding an opaque MMVQ operation
would duplicate those facts and hide the correction that the Workload must verify.

At `llama.cpp@0a5ac49bce4c42893368585edf3ffb41a38f0108`, Q4_0 uses `VDR=2`.
For K32, two contributing lanes each compute one 16-value partial using four dot4
atoms, then apply:

```text
fp32(d4) * (fp32(partial_sumi) * fp32(d8) - 4 * fp32(s8))
```

The two partials are added in FP32. On the RDNA path, `ggml_cuda_dp4a` calls
`__builtin_amdgcn_sudot4(true, a, true, b, c, false)`. LLVM source suggests a GFX11
`v_dot4_i32_iu8` realization with signedness modifiers, but only a retained gfx1151
HSACO disassembly can establish the actual emitted instruction.

## Decision

### 1. Extend existing Load for typed packed fields

Do not add a `decode_q4_0` OperationKind. A Load whose global source has a
`packed_block` relation has two edge-derived forms:

- one UINT8 write: the existing raw byte load;
- registry-ordered typed writes: mechanical field extraction.

The first typed forms are:

- Q4_0 → FP16 `d` and UINT8 logical `qs[32]`;
- Q8_1 → FP16 `d`, FP16 `s` and INT8 `qs[32]`.

Q4 extraction maps physical low nibble `j` to logical `j` and high nibble `j` to
logical `16+j`. It does not subtract eight, dequantize, correct or dot. AccessMap owns
only the record prefix; the packed registry continues to own the internal byte axis.
The rank-one Q4 input therefore needs the typed packed Load spelling of an empty record
prefix, while Q8 explicitly selects workspace record zero.

### 2. Add one generic row-wise Dot

Add a binary `dot`, not `q4_dot`, `dp4a` or `mmvq`:

```text
reads: UINT8 and INT8 register tensors with identical shape
writes: INT32 tensor with one axis collapsed
parameters:
  axis: contiguous last axis
  accumulator: int32
  instruction.contract: explicit atom, operand signedness and clamp policy
```

The admitted last-axis extent is a multiple of four. The first gfx1151 contract must
name the exact unsigned-Q4/signed-Q8, INT32, no-clamp physical atom. A contract without
a backend body is lowering-blocking; scalar multiply-add is not a fallback for a
Schedule that declares the AMD atom.

### 3. Add only two widening Cast relations

- FP16 → FP32;
- INT32 → FP32.

Both are shape-preserving register conversions with no rounding-policy choice. Existing
FP32 primitives express the complete scale and correction arithmetic.

### 4. Compose K32 visibly

```text
q4 logical[32] -> reshape [4,8]
q8 logical[32] -> reshape [4,8]
dot axis=1 -> int32[4]
cast -> fp32[4]
reshape [2,2]
reduce axis=0 -> fp32[2]       # low+high plane for each 16-value partial
partial * d8 - 4*s8
partial * d4
reduce axis=0 -> fp32[1]
store
```

This uses no new permute, concatenate, bitwise, INT32 reduction or partial-specific
operation. Workload tests—not Compiler operator recognition—own the factor four, field
sources, partial grouping and final tolerance.

## Verifier and Target boundary

- Typed packed Load must match registry field count/order/dtype/logical extent, read
  global UINT8 packed storage and write registers through the exact record prefix.
- Dot inputs must be register UINT8×INT8, same shape, last-axis extent divisible by four;
  output is the collapsed INT32 shape.
- The instruction contract names signedness, INT32 accumulator and `clamp=false`.
- gfx1151 alone may gain the draft dot kind/contract. sm_100a is not widened.
- Target publication requires actual gfx1151 compile/disassembly; upstream and LLVM
  source establish a hypothesis, not the local toolchain's hardware fact.

## Deliberate exclusions

Do not add an operator-named Q4/MMVQ kind, public Q8 input, fused producer/consumer
mode, llama.cpp `VDR` or warp-table lookup, hidden `-4*s` correction, packed scalar
dtype, silent scalar dot fallback or wave search. The Workload/Evaluation layer retains
the two-kernel ordering, intermediate custody and combined result.

## Next gate

After Kernel A hardware correctness, implement in order: typed packed Load, widening
Cast, generic Dot parser/verifier, draft gfx1151 instruction contract and emitter, then
one K32 consumer Schedule plus isolated nibble/order/record/correction negatives. Run a
new full Corpus Gate and obtain independent approval. Only then compile on gfx1151,
retain AMDGCN/HSACO, prove the dot atom survived, and run the two-kernel seven-case
correctness boundary with two launches and zero fallback. Performance work begins only
after that evidence passes.
