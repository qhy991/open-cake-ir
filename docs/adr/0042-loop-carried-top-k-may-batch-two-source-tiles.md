# ADR 0042: Loop-carried top-k may batch two source tiles

## Status

Proposed. It authorizes implementation and Corpus Gate preparation, not expectation
adoption, a Compiler Revision, provider run, GPU campaign, or Candidate promotion. The
existing external repository-owner gates remain unchanged.

## Irreducible goal

Reduce how often an FP32 loop-carried `top_k` performs its resident merge when two
consecutive source tiles fit under the same merge width. QSA score/top-k supplies the
first concrete case: source extent 128 and k=512 already produce a 1024-key merge, and
batching two source tiles keeps that width while halving the number of full-pair merges.
This is a falsifiable scheduling hypothesis, not a QSA-specific semantic operation or a
performance claim.

## Decision and canonical owner

The existing `top_k` operation gains one optional parameter,
`source_tiles_per_merge`. Omission means the existing one-source-tile cadence. The only
serialized value admitted by this Revision is `2`; writing `1` is a non-canonical second
spelling and any other value is unsupported. The field belongs to `TopKParameters`, not
`TileLoop.range_options`, because it changes when one selection operation updates its
own carried state rather than how every operation in the loop executes.

The value two is legal only for an FP32 `top_k` with `across_loop: true`. The operation
still returns the deterministic top-k over every valid source position in the loop:
descending FP32 values, canonical equality of positive and negative zero, lowest global
source position on ties, and the existing `reject_input` NaN policy. A first source tile
is held pending and the next tile is merged with it and the carried top-k state. An odd
last tile is flushed once; a partial last tile uses `global_position < effective_stop`;
a zero-trip loop retains the existing negative-infinity value and negative-one index
sentinels. Pair-local positions never become results.

Delayed intermediate results are deliberately unobservable. Either top-k output being
read by another operation inside the same tile loop is a schedule-semantics error; both
outputs may be consumed after finalization outside the loop. The operation's public
effects remain one source read and the same ordered value/index writes.

## Performance transparency and analysis

The pending representation is derived lowering state, not another IR Buffer. Requiring
an author to name a UINT64 key tile, destination, or placement would expose one backend
algorithm and add a second spelling of the same top-k semantics. Hiding the state from
analysis would be equally invalid. The static profile must therefore derive and report
the cadence, two-tile source batch extent, merge width, merge-count formula, and pending
elements from this field. The current lowering carries one packed UINT64 composite key
per source element, so the derived logical pending state is `8 * source_extent` bytes:
1024 bytes for the QSA tile-128 case. Physical register and shared-memory placement remain
unknown until compilation or profiling supplies them.

For source extent `n`, the current SM100a Triton merge width is
`2 * max(k, 2 * n)`. Its first slice admits only `loop_unroll_factor=1`,
`warp_specialize=false`, and `flatten=false`. Unsupported combinations fail closed; the
Compiler does not erase the declared option or fall back to per-tile merging.

## P1-P8 and failure semantics

- P1/P3: one optional field extends the canonical `top_k`; no new sort operation,
  explicit pending destination, layout algebra, or arbitrary batching factor is added.
- P2/P5: merge cadence and derived pending extent are visible to profile and cost
  analysis even though key packing stays backend-owned.
- P4: schema, typed parsing, loop placement, dtype, result-observability, source/tile,
  power-of-two, and backend loop-option rules fail before emission.
- P6/P7: the same change updates focused contracts and every analysis that consumes
  top-k merge structure, then passes a separately reviewed full Corpus Gate.
- P8: the SM100a lowering documents pair formation, odd-tail flush, partial-tail mask,
  and exact supported loop controls. On-device evidence remains the performance ground
  truth.

`TOP_K_MERGE_CADENCE_REQUIRES_ACROSS_LOOP` rejects a cadence with no carried state.
`TOP_K_MERGE_CADENCE_OUTPUT_READ_IN_LOOP` rejects an observer before a pair is complete.
`TRITON_TOP_K_TWO_TILE_CONTROL_FLOW_UNSUPPORTED` rejects a loop option the current
backend cannot preserve. Existing top-k findings continue to own loop placement,
source/tile agreement, dtype, result shapes, spaces, power-of-two extents, and the
deliberate INT32 carried-state exclusion. Toolchain, broker, or profiler failures remain
unknown; oracle mismatch is invalid; a correct but immaterial result is a negative result.

## Acceptance evidence

1. Focused parser/schema tests prove omission maps to one, explicit two is admitted, and
   explicit one or any other value is refused.
2. Focused verifier tests prove the across-loop, in-loop-reader, INT32, and Triton
   range-option boundaries.
3. One generic positive Corpus Schedule covers two-tile FP32 carried selection with an
   odd final trip; one generic negative Schedule exposes an in-loop result reader.
   Expectations are adopted only in a separate repository-owner-reviewed act.
4. The full Corpus Gate matches and an external repository-owner approval binds the exact
   successor Gate before release.
5. After release, exact-toolchain compilation and fixed-workload B200 correctness must
   cover zero, odd, even, and partial tails. CUPTI evidence must show that the first tile
   does not execute the expensive merge; a source-level conditional that still executes
   it is a failed hypothesis, not an optimization.

No result from this change is a checkpoint reproduction or end-to-end serving result.
