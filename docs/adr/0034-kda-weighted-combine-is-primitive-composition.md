# ADR 0034: KDA weighted combine is primitive composition

Status: proposed, 2026-08-25.

## Outcome and non-goals

Express the arithmetic body of KDA MoE v1's non-fused weighted-combine kernel by composing the existing
runtime-indexed `AccessMap`, `elementwise(mul)`, `reduce(sum)` and ordinary `store`.
Add no `moe`, `combine`, `scatter`, `cast`, program-DAG or KDA-version construct.

This slice does not perform route selection, atomic group-slot reservation, dispatch,
either grouped GEMM, fused atomic scatter, the combine kernel's persistent-workspace
reset side effect, CLC, PDL or graph capture. It therefore does not make any complete KDA
version expressible and authorizes no performance claim.

## Invariants and owners

- `AccessMap` remains the sole owner of the expert and row coordinates. Invalid runtime
  indices retain the existing masked-zero load semantics.
- Buffer dtypes own the numeric types. A binary elementwise operation preserves a common
  dtype; FP32 combined with BF16 or FP16 produces FP32. Other mixed pairs are refused.
- `reduce(sum)` and `reduce(max)` produce FP32, matching the existing Triton body that
  explicitly accumulates in FP32.
- `store` preserves dtype or performs the one final narrowing used here, FP32 to BF16 or
  FP16. The destination Buffer names that rounding point; a separate cast would give the
  same fact two spellings.
- The verifier checks these relations before lowering. Backend type promotion or store
  conversion is an implementation of the declared relation, never its authority.

## Smallest complete slice

One token program loads eight expert ids, row ids and FP32 route weights. It uses the two
INT32 tiles as zipped indices into BF16 expert rows, multiplies the selected `[8, 16]`
tile by the weights along the route axis, sums that axis in FP32, and stores one BF16
`[16]` output row. Deterministic inputs include valid routes and KDA's `-1` sentinel.

A contract mutation declares the weighted intermediate as BF16 even though BF16 times
FP32 produces FP32. It must receive a localized dtype Finding and remain unlowerable.
The mutation is derived from the positive Schedule in its focused test rather than copied
into a second mostly identical JSON file. The existing Flash-KMeans dtype falsifier gains
the same newly modeled Finding rather than silently escaping it.

## Acceptance evidence

1. schema and typed parsing accept the existing vocabulary without a new spelling;
2. verifier tests cover the mixed arithmetic, reduction and store rules plus unsupported
   mixed types;
3. the positive case and existing dtype falsifier pass the reviewed Corpus Gate
   dispositions, while the focused KDA mutation fails for the intended reason;
4. generated source visibly performs zipped masked loads, route-axis weighting, FP32 sum
   and BF16 output storage;
5. one brokered B200 correctness run matches an independent gather-weight-sum oracle,
   with no timing and no retry.

The first frozen attempt reached broker allocation but stopped before compilation because
the worker's unqualified `python3` could not import Torch. The first successor executed
and computed correctness, but the instrument then failed on its late Cutlass metadata
import before writing the result. Both failures are retained and neither supports a pass.
The metadata import now happens before any compiler or GPU work, and a fully preflighted
second successor is frozen; item 5 and this ADR's acceptance remain pending its result.

Failure at any step is retained as the missing primitive or backend boundary. It is not
repaired by adding a workload-named operation or widening a numerical tolerance.

## References

- CAKE paper sections 2--4: typed schedules, pre-compile verification and primitive-first
  agent feedback.
- local KDA MoE v1 `_combine_kernel` under
  `.judge/kernel_versions/moe/v-01/kernel.py`.
- [`ADR 0020`](0020-kda-deltas-are-an-external-expressibility-corpus.md) and
  [`ADR 0026`](0026-runtime-indexed-loads-are-access-map-composition.md).
