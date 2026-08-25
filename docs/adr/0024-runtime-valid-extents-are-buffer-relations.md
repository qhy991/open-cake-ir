# ADR 0024: runtime-valid extents are buffer relations

Status: accepted in Compiler v19, 2026-08-25.

## Outcome and non-goals

A global padded Buffer may declare that one dimension has a runtime-valid prefix whose
lengths come from one device-resident INT32 input Buffer.  The relation also names the
data axes that index those lengths.  Every global access derives the corresponding mask;
the Schedule does not repeat the predicate on each load or store.

This is not grouped GEMM, a CLC scheduler, a work-tile table, shape dispatch, dynamic
allocation, an atomic append counter, gather/scatter, a program DAG or a KDA/MoE mode.
Those mechanisms have separate effects and legality rules.  In particular this relation
does not make any complete KDA version expressible.

## Evidence and constraints

KDA MoE v1 stores routed activations as padded tensors with shape
`[local_expert, capacity, feature]`.  The device-resident `counts[local_expert]` Buffer is
the authority for how many rows of each expert are valid.  Both grouped GEMMs consult it;
the final epilogue also refuses rows beyond it.  Later KDA versions tune capacity and
filter invalid tiles, but they retain the same valid-row fact.

The paper treats shape, ownership and lifetime as declared resource facts, and asks the
compiler to derive mechanical metadata and verify producer/consumer consistency.  Making
every operation restate `row < counts[expert]` would violate that ownership model.  A
host-side length list would also misrepresent KDA: the count stays device-resident so
captured execution does not synchronize merely to discover work.

## Minimal primitive and SSOT

One optional `Buffer.valid_extent` relation owns the fact:

```json
{
  "dimension": 1,
  "buffer": "counts",
  "indexed_by": [0]
}
```

- `dimension` is the padded data axis whose valid region is the prefix `[0, length)`;
- `buffer` is the device-resident INT32 input containing those lengths;
- `indexed_by[i]` is the data axis used to index extent-buffer axis `i`.

For data shape `[4, 8, 16]`, the relation above requires `counts.shape == [4]`.
`dimension` cannot appear in `indexed_by`; indexed axes are unique and in range.  The
extent Buffer is global input state and cannot refer to itself.  Runtime values belong to
the Workload input contract; generated address masks remain memory-safe for negative or
over-capacity values because the ordinary static bounds still apply.

The existing AccessMap remains the single coordinate authority.  When an access addresses
the related Buffer, lowering reuses the AccessMap expressions for `indexed_by`, loads the
extent, and conjoins `coordinate[dimension] < extent`.  There is no second boundary enum
or authored predicate.  A masked load materializes a dense tile with zero in invalid
positions, so the on-chip destination does not carry the relation.

## Initial lowering domain

The first Triton slice supports one extent axis indexed by one scalar program axis.  This
matches KDA's one expert id selecting one count while vector lanes address padded rows and
features.  The typed relation can describe more indexing axes, but an access outside this
initial backend domain receives a localized lowering-blocking Finding rather than an
invented address calculation.

The standalone profile loads one `[8,16]` tile for each of four groups from a padded
`[4,8,16]` FP32 input.  Counts `[0,3,8,5]` make empty, partial and full groups observable.
The load zero-fills invalid rows and an ordinary dense store writes the complete output.
This isolates extent semantics from grouped GEMM arithmetic and output initialization.

## Failure and compatibility semantics

- unknown/self extent Buffer, non-INT32 or non-global-input extents, invalid axes and
  derived extent-shape mismatch are data-consistency Findings;
- attaching the relation to a non-global data Buffer is a data-consistency Finding;
- an access whose extent indexing is outside the declared Triton subset is a hardware-
  conformance Finding and cannot reach emission;
- a global Buffer without `valid_extent` keeps its existing dense meaning and generated
  bytes;
- readers never infer raggedness from names, counts-shaped tensors or padded zeros.

Static verification proves the declared relation and the generated predicate, not the
runtime distribution of length values or grouped-kernel performance.  Correctness remains
an external B200 comparison.

## Acceptance evidence

1. schema and typed parser accept one spelling and reject structurally repeated axes;
2. axis, dtype, storage, derived-shape and backend-domain failures are independently
   triggerable;
3. a positive and relation-drift Corpus pair match reviewed expectations;
4. all previous lowering digests remain unchanged;
5. generated source visibly loads `counts[group]` and combines its predicate with the
   row mask;
6. brokered B200 execution matches an independent zero-padding oracle over empty,
   partial and full groups;
7. complete KDA coverage remains 0/57 and persistent/CLC scheduling remains missing.

The released Gate matched 27/27 cases over 43 Revision-bound sources; dry-run review
showed that only the two new Corpus expectations changed. The released generated source
compiled and launched on a brokered NVIDIA B200 and matched 512/512 oracle elements with
zero deviation under the unchanged `1e-5` correctness boundary. The immutable observation
is `inventory/VALID_EXTENT_OBSERVATION_20260825.json`; no performance was measured.

## References

- [CAKE: Compiler-Agent Co-Design for Frontier Kernel Evolution](https://arxiv.org/html/2608.12629v1#S2)
- local KDA MoE v1 `_route_kernel` and grouped-GEMM `gidx_mapping` under
  `.judge/kernel_versions/moe/v-01`
- [`ADR 0020`](0020-kda-deltas-are-an-external-expressibility-corpus.md)
- [`IR_COVERAGE.md`](../IR_COVERAGE.md)
