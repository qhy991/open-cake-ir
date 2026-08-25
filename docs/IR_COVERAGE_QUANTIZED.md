# DeepGEMM: quantization is not a dtype

Surveyed against the typed IR. Thirty-one axes across FP8 and FP4 block-wise GEMM,
grouped and masked variants, and a fused MoE megakernel. Paths under `DeepGEMM-upstream/`.

The IR has `fp8_e4m3` in its dtype set. That is the whole of its quantization support, and
this survey is mostly about why that is not the interesting part.

## A scale tensor is a relation, not a buffer

Three things are missing and they compound.

**There is no "scales of" relation.** `MmaParameters` has one accumulator and no scale
operand; `Buffer` has a mode of input, output or scratch and no way to say a buffer holds
another buffer's scales.

**The layout relationship is non-trivial and cannot be stated.** The operand is K-major and
its scale tensor is MN-major -- transposed relative to what it scales. The scale's MN extent
is `get_tma_aligned_size(mn, element_size)`, which is *padded*, and its K extent is `k`
divided by both the granularity and four, because UE8M0 bytes are packed four to an int32.
Writing those as plain integers would lose the relation, so the IR could neither verify the
scale shape nor re-derive it after a tile change.

**And the padded stride is a function of a dynamic dimension.** For every m-grouped and
masked path, `mn` is not known at codegen time. `Buffer.shape` is a static integer list.

## Granularity is a kernel selector, not a parameter

The recipe is a triple: granularity along MN for each operand, and along K. `gran_mn == 1`
means per-token, `128` means per-block. That triple picks among three kernel families --
one where scales are MMA operands, one where they are applied by hand in the mainloop, and
one with no scales at all -- and the choice then feeds back into stage counting and into
which block-N candidates are legal.

The tile is not free either: `block_k = 128 / element_size`, because 128 is the scale block
size, and the kernel asserts it.

## The scales are applied three different ways in one repository

**As MMA operands.** On SM100 the scales get their own TMA descriptors, land in shared
memory, are shuffled in place by a warp that exists *only* for this, are copied into tensor
memory, and are then named in the instruction descriptor. There is no rescale step; the
accumulator comes out already correct.

**In the mainloop, in registers, between two accumulators.** On SM90 the raw output and a
running rescaled total are separate register arrays, and each k-block multiplies one into
the other. `ElementwiseOp{mul}` with `broadcast_axis` describes that arithmetic exactly and
says nothing about its placement, which is the part that matters.

**By the epilogue, for the next GEMM.** In the fused MoE kernel the L1 epilogue computes
SwiGLU, reduces an absolute maximum across warp pairs through shared memory, derives a
scale factor, and writes it as a raw UE8M0 byte directly into the L2 GEMM's scale buffer.
One kernel's epilogue produces another kernel's operand metadata.

## The scale pipeline runs at a different rate than the data pipeline

When the K granularity is 128 rather than 32, the scale TMA fires on one k-block in four,
and so does the copy into tensor memory, and the MMA selects which of four packed bytes to
use. A sub-rate producer nested inside a parent pipeline, sharing its barrier and
conditionally contributing to its transaction-byte count. `Pipeline(name, stages)` has no
form for this.

## The host-sync boundary has a name in the source

This repository already found that MoE work is shaped by problem sizes being device-resident
during graph replay. DeepGEMM names the variable:

```cpp
const auto has_synced_ks = ks_cpu.has_value() and not ks_cpu.value().empty();
```

Both sides are deliberate. The masked path chooses its config from a caller-supplied
*expected* M, never touching the device value. The k-grouped path on one architecture
*requires* a host vector of per-group K and asserts it. And a third layout exists
specifically to remove that requirement, at the cost of over-allocating the packed scale
buffer to an upper bound because the true size is device-resident.

Nothing in the IR distinguishes an extent that is a device tensor from one that is a host
integer, which is precisely the distinction that decides whether a synchronisation is
needed.

## What does fit

Persistent grid with swizzled traversal, warp-role specialisation, the barrier topology,
CTA occupancy, per-operand swizzle, and the epilogue store atom all map directly. That is
six of thirty-one, and they are the parts a non-quantized GEMM would also need.

Worth noting that the barrier topology fits *including* a barrier that exists only because
scales need an extra transpose hop, and that the warp-role vocabulary fits *including* a
warp whose entire job is reshuffling scale bytes. The structure survives; the reason for
the structure is what has no name.

## Two more that are simply absent

The MMA instruction's N field is patched at runtime from a device-derived effective M, so
`MmaInstruction.shape` being static is wrong for the padded m-grouped path. And a tensor-core
duty-cycle throttle inserts a timed spin after every k-block, deliberately stalling to avoid
clock throttling -- an intent with no vocabulary anywhere in the IR.
