# ADR 0045: Triton lowering admits multiple explicit MMA DAG nodes

## Status

Proposed. This removes one backend restriction over existing typed IR. It does not
authorize Corpus expectation adoption, a Compiler Revision, a QSA candidate, GPU work,
or promotion; those remain separately gated.

## Irreducible goal

Lower a Schedule that explicitly composes more than one independent `mma` operation.
The first measured use may split one IEEE FP32 K128 contraction into two K64 partial
contractions and add their FP32 results before the required per-head ReLU. The vocabulary
already expresses each operation, buffer, access subrange, dependency and add. A new
`multi_mma`, `split_k`, mode, count parameter, partial-dot operation, Target contract, or
QSA branch would duplicate facts already owned by the DAG.

## Decision and canonical owners

Remove the Triton emitter's historical global `mma <= 1` precondition. Every `mma`
continues to use the existing per-operation owners:

- typed IR owns its reads, unique write, FP32 accumulator, instruction and tile;
- common verifier owns operand/result dtypes, Target admission, placement and writers;
- Triton preflight owns backend instruction-family support at that operation's path;
- declaration order plus `depends_on` owns when each partial becomes available;
- the explicit consumer operation owns combination and reassociation;
- work, liveness and profile walk every operation rather than a backend summary flag.

The emitter must not infer that two writes target one hidden accumulator. Multiple writers
remain illegal. A contraction accumulated across a declared tile loop continues to use
the existing access-derived `+=` rule independently for each `mma`; an independent partial
uses assignment.

## K64 candidate semantic boundary

The motivating Schedule candidate is explicit:

```text
partial_lo = IEEE_FP32_DOT(q[..., 0:64],   k[..., 0:64])
partial_hi = IEEE_FP32_DOT(q[..., 64:128], k[..., 64:128])
head_score = FP32_ADD(partial_lo, partial_hi)
positive   = RELU(head_score)
score      = SUM_OVER_INDEX_HEADS(positive)
```

The two K intervals must be disjoint and cover `[0,128)`, query and key halves must match,
both writes must be FP32 register scratch, and the add must dominate the one per-head
ReLU. The real-number contraction is unchanged, but two K64 reductions plus an FP32 add
need not be bitwise identical to one K128 reduction. Only the unchanged external oracle
and tolerance may decide correctness; near-tie selection changes cannot be waived as
mathematical equivalence.

The candidate's public inputs, output, dtypes, shapes, merge2 selection, causal stop, tie
break, sentinel and Program ABI remain unchanged. No new global materialization, state,
atomic, barrier or side effect is introduced.

## Analysis, resource and failure boundaries

For a generic two-K64 `[16,16]` example, work owns two contraction counts plus one explicit
FP32 add. The two MMA FLOP counts sum to one K128 contraction; total FLOPs are larger by
the add. Repeated identical instruction contracts remain visible twice while their set has
one `contended_contract`. Profile reports two textual `tl.dot` calls. Logical register
pressure is only a deterministic liveness feature, not a physical-register prediction.

The QSA candidate's preliminary logical feature changes from `284` to `168`, but compiled
registers, implicit shared memory, spills and occupancy remain unknown until exact
Executor-v39 compilation. The current score kernel uses `143,360` dynamic shared bytes and
one shared-memory-bound CTA/SM. Two-CTA residency from shared memory alone would require at
most `116,736` bytes per CTA; this is a falsification threshold, not a prediction.

A toolchain or resource failure is unknown, an external-oracle mismatch is invalid, and a
correct result below the fixed complete-Program materiality boundary is a valid negative.
Compiled-resource improvement without complete-Program improvement is not promotion.

## P1-P8 and acceptance evidence

- P1/P3: repeat existing orthogonal `mma` nodes and compose through an existing explicit
  add; no syntax, synonym, implicit accumulator or count mode.
- P2/P5: each contract, tile, access slice, partial, work count, liveness interval and
  emitted dot remains inspectable; compiled physical resources remain abstentions.
- P4: the second and later MMA receive the same Target, dtype, placement, unique-writer
  and backend-family checks and localized Findings as the first.
- P6: focused tests cover two emitted dots, explicit-add order, work/profile counts,
  second-operation failures and frozen one-MMA identity; generic Corpus positive and
  deliberate-negative cases exercise the successor.
- P7: preflight, constructor, emitter, work/liveness consumers, profile diagnostics and
  frozen Program witness agree on the same operation DAG.
- P8: all contractions retain the existing SM100a `triton.dot.fp32_ieee` Target contract;
  Executor v39 exact compile, resource metadata, external oracle and B200 timing ground
  the candidate.

The frozen QSA Program-v2 bytes and its five canonical one-MMA lowering identities must
remain exact. If evaluation would require changing that Program, the evaluator, Workload,
oracle, tolerance or Executor, work stops for new authorization.

For the current `127.824867 ms` frontier, a candidate must reach `121.433624 ms` for a
standalone five-percent promotion. If only score changes, score/top-k must reach
approximately `52.088054 ms`; `55.139770 ms` is the maximum that can justify later
composition with the already measured attention-stage3 control. No result is a checkpoint
or end-to-end serving claim.
