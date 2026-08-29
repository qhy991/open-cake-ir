# ADR 0044: Loop-carried top-k canonical lowering may use exact half-selection

## Status

Proposed. This records a generic lowering successor and its evidence gates. It does not
authorize Corpus expectation adoption, a Compiler Revision, provider run, GPU campaign,
or Candidate promotion. Those transitions remain externally reviewed and content-bound.

## Irreducible goal

Reduce exact ordered-selection work in an existing FP32 loop-carried `top_k` when its
merge consists of two equal `k`-key halves. The motivating QSA geometry has `k=512` and
merge width `1024`; earlier merge2 evidence established that selection work is material.
The IR already says everything needed for the optimization, so no operation, field,
algorithm flag, QSA branch, unordered shortcut, or layout vocabulary is added.

## Decision and correctness proof

The carried composite-key state is descending by induction: initialization is all invalid
zero keys and every completed merge returns a descending prefix. The source half is
sorted ascending. Concatenating a descending half with an ascending half forms a bitonic
valley; `tl.bitonic_merge(..., descending=True)` produces the exact descending union.
The first `k` keys are therefore the same ordered top-k returned by the existing lowering.

The first half is extracted using public tensor-shape operations only: reshape the
`2k` result to `[2,k]`, transpose to `[k,2]`, split the last dimension, and retain the
first result. This does not expose or declare a backend layout.

Composite-key construction and decoding are unchanged. Consequently:

- FP32 values remain descending and equal values retain lowest-global-position ties;
- positive and negative zero remain canonical equals before the index tie break;
- every valid key, including negative infinity, remains greater than invalid key zero;
- merge1, merge2 even pairs, odd flush, partial tails, zero trips, sentinels, and final
  `INT_MAX` to `-1` conversion keep their existing semantics;
- `reject_input` remains the only admitted NaN policy.

The selector is derived only when `merge_width == 2*k` and the structural threshold below
is met. Every other geometry retains canonical `tl.topk`; there is no author-visible
choice or compatibility alias.

## Toolchain-specific structural model

The work model belongs to Triton v3.7.1 `standard.py`, not to the public semantic API.
Executor v39 remains the runtime authority for `triton==3.7.1`; the SM100a Target records
the exact-tag provenance but does not duplicate an independently mutable toolchain
version. A later Executor/toolchain requires fresh compiled evidence and may invalidate
the model's usefulness without changing `top_k` semantics.

Let `m=log2(k)`. Count input lanes participating in compare/select work: both lanes of a
compare-exchange and both inputs of the pairwise top-k reduction. For v3.7.1
`topk(2k,k)`, the first bitonic rounds cost `k*m*(m+1)`, the pairwise reduction costs
`2k`, and the final retained-half merge costs `k*m`, for
`k*(m^2+2m+2)`. Sorting one source half and merging the bitonic `2k` union costs
`k*(m+1)*(m+4)/2`.

At `k=512`, this model changes from `51,712` to `33,280` comparison-lane units, a
`35.6436%` reduction. At `k=256` it reduces `34.1463%`; at `k=128` only `32.3077%`.
The lowering therefore requires at least one-third structural reduction rather than
hard-coding a particular `k`. These counts are not cycles, instructions, latency, MFU,
BWU, or a performance prediction.

## Resource and failure boundaries

The Schedule adds no allocation, carried state, pending state, or merge width. That proves
only a zero delta in declared shared-memory allocation. `tl.sort` and
`tl.bitonic_merge` own implicit storage, synchronization, registers, spills, and physical
shared memory; profile must report those facts as unknown until matched compiled
artifacts exist.

Before correctness or timing, exact Executor-v39 compilation must prove that the public
namespace export, shape manipulation, and kernel resource limits are accepted. A
compile/toolchain failure is unknown evidence, an oracle mismatch is invalid, and a
correct result below the existing five-percent Program boundary is a valid negative.
Profiler duration is never Program latency.

## P1-P8 and acceptance evidence

- P1/P3: one existing `top_k` and one derived canonical lowering; no syntax or synonym.
- P2/P5: profile exposes the exact geometry, versioned frontend model, selected algorithm,
  comparison-lane counts, and explicit abstention from compiled resources and timing.
- P4: existing typing, loop placement, result observability, tie, sentinel, and tail
  legality remain the sole semantic owners.
- P6/P7: randomized reference tests cover repeated zero/even/odd/partial merges and key
  boundaries; generic Corpus positives cover both merge1 and merge2 optimized paths.
- P8: exact-tag Triton provenance plus Executor-v39 compilation and B200 evidence ground
  the lowering without inspecting or importing Direct CUDA, PTX, or SASS mechanisms.

Acceptance requires focused tests, separately reviewed Corpus expectations, a complete
matching Corpus Gate, repository-owner approval bound to that exact Gate, exact-toolchain
compile with matched resource metadata, unchanged external-oracle correctness, and fixed
complete-Program CUPTI timing against a fresh IEEE control and Direct black-box baseline.

For the current QSA frontier, score/top-k must reach approximately `69.288 ms` before any
composition with the previously measured attention-stage3 control is warranted; it must
reach `66.237 ms` to cross the five-percent Program boundary by itself. No result is a
checkpoint reproduction or end-to-end serving result.
