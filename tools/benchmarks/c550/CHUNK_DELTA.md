# Chunk gated-delta starter

Tracking issue: [#419](https://github.com/qhy991/open-cake-ir/issues/419).
Draft PR: [#425](https://github.com/qhy991/open-cake-ir/pull/425).

This source implements the original L2/060 chunk equations with existing CAKE
primitives. It is an untuned starting design. Native compilation, original-device
correctness and runtime admission have not been established.

## Original contract

- Query/key are BF16 `[B,4,S,128]`; value is BF16 `[B,16,S,128]`; g and beta are
  BF16 `[B,16,S]`.
- Scale is the original fixed scalar `0.08838834764831845`. It is checked by the
  Workload binding and original-call wrapper, with no extra scalar tensor.
  Its provenance is `literal_input`, from the original Workload's scalar value.
- Output is BF16 `[B,S,16,128]`. There is no public initial/final-state tensor.
- Value head h reads query/key head `h % 4`, following the original repeated
  head sequence.
- Normalization keeps the BF16 square, sum result, epsilon addition, inverse
  result and product boundaries before FP32 conversion. Query scaling then
  occurs in FP32.
- Chunk size remains 64. Sequence padding is zero; only original rows are returned.
- Intra-chunk decay uses cumulative g differences. The transformed keys,
  inter-chunk attention and state updates use raw g exactly where the reference
  does. The source does not substitute the usual token delta-rule recurrence.

## State and stage ownership

The design comparison considered reference-shaped triangular updates and an
algebraic inverse/forward-substitution alternative. The latter was rejected for
this baseline because it would change more numerical association and require
separate justification. The selected source keeps the original 63 row updates.

1. Normalize q/k into FP32 after the explicit BF16 boundaries.
2. Compute padded prefix g using masked reductions. No scan capability is added.
3. Build the resident strict triangular matrix and apply its 63 static SSA row
   updates, then identity. Recompute it per transformed output column, producing
   transposed U and W with one writer per element.
4. For each chunk, write corrected values and chunk outputs. A fresh state tensor
   is produced for the next chunk. No state tensor is overwritten in place.
5. Pack the output, crop padding, restore original axis order and cast to BF16.

The first state is explicitly zero. Its zero products are represented without
allocating a state input, retaining nonfinite propagation in those expressions.
The final state update is dead after the only public output; it is omitted and
does not create an unused private tensor or a new public output.

The public helper interface is `source_for(B,S,scale=...)`,
`program_for(B,S,scale=...)` and `source_for_workload(workload,case_id)`.
All state/layout choices remain private to this one source module.

## Software evidence

At fixed `d60e6d57`, all 16 original shapes construct and all **438 leaf stages**
assess and lower. Four contracts pass without skips. They run actual emitted
small arithmetic against an independent array implementation and cover chunk
boundaries, tails, signed beta, zero initial state, causal limits, repeated-head
mapping, and recurrent dependencies. Wrong group mapping, causal masking,
correction sign and raw/cumulative-g substitution each fail the controls.

Small controls use smaller dimensions/chunks and nominal FP32 arithmetic with
explicit BF16 rounding. They do not simulate native reciprocal-square-root,
exponential or reduction behavior. Original shapes are separately checked by
source lowering; no device result is inferred.

An additional synthetic CPU PyTorch check in the existing MetaX installation
matched BF16 square, sum and epsilon addition. The first difference occurred at
rsqrt: for BF16 0.30078125, direct CPU rsqrt returned 1.828125 while FP32 rsqrt
then BF16 returned 1.8203125; for 2.75, they returned 0.60546875 and 0.6015625.
The direct CPU values matched a BF16 sqrt-then-reciprocal control. This does not
identify the C550 GPU implementation. The observations are retained privately;
the original Bench comparator and tolerances remain unchanged. The device gate
must check the relevant boundaries.

## Cost and readiness limits

Stage count is `3*ceil(S/64)+3`, reaching **99** at S2048. The largest generated
source is **325,160 bytes**. Recomputing the triangular matrix per output column
trades extra work for fewer global interfaces and a simple state graph. This
cost is a property of this chosen baseline, not a lower bound on every CAKE
implementation.

The current build owner calls `build_stage` for every stage. Each call invokes
the isolated compiler in a fresh jail; `/tmp/triton-cache` is on that jail's new
tmpfs. There is no cross-candidate stage cache on this path. Two generic builds
of the longest Program require **198 compiler entries**, exceeding the current
192-entry Run budget. The source task changes neither that budget nor frozen
tools. Partial progress or one accepted candidate must not be reported as
multiple completed optimization iterations.

The largest declared two-arm/fresh-set memory lower bound is **6.854 GiB**, using
`2I + 18O + 18S` for identical starters. It includes distinct retained states and
all Program intermediates. It excludes backend private memory, context and
allocator costs; memory remains unqualified. Register pressure in the resident
64x64 transform also requires native inspection.

Generated Programs, projected source, compile requirements, original workload
identities and CPU observations remain outside Git. No raw reference/data,
Compiler change, target expansion, GPU/provider call or performance result is
included. No promotion is made from these source/software observations.
