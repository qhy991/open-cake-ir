# ADR 0046: Dynamic loop stop owns whole-grid work

## Status

Proposed for the user-authorized Compiler successor. Promotion still requires the exact
full Corpus Gate and its repository-owner approval. This changes no Workload, Program,
evaluator, oracle, tolerance, timing boundary, materiality threshold, Target, Executor,
IR syntax, legality, or lowering.

## Irreducible goal

Make declared work and utilization evidence agree with the loop domain the backend
executes. A `TileLoop.stop` derived from a scalar program coordinate may execute fewer
iterations than the static Buffer extent. Charging every coordinate the static maximum
is an overcount, not a conservative lower bound, and cannot support MFU or roofline
claims.

## Decision and canonical owners

`work.py` is the single owner of loop-trip domains. For one query-derived stop it applies
the existing lowering semantics in this order:

```text
program coordinate -> add -> floor_div -> clamp to [0, loop extent]
                   -> ceil_div by loop tile
```

The result carries one trip count per scalar program coordinate plus the multiplicity of
all other program axes. Whole-grid operation repetition composes that distribution with
any enclosing static loop trips. An operation outside a loop executes once per program
tile. One operation chain containing more than one dynamic stop remains unmodeled and is
reported as `unknown`; it never falls back to the static maximum.

`analysis.top_k_merge_structure` consumes this same owner for source-tile, grouped-merge,
and tail-flush cadence. `WorkBound` and the profile envelope expose per-operation
whole-grid repetitions and missing reasons. Arithmetic with unknown repetition is omitted
and named in `uncounted_arithmetic`, so the remaining count is a sound lower bound.

The utilization owner also preserves error direction. Counted arithmetic is exact or a
lower bound and may contribute a roofline time floor only against a
`device_specification` ceiling. Compulsory bytes additionally require an exact count. A
microbenchmark reference or partial-addressing byte count remains a comparative or
upper-bound ratio; neither can become a time floor or a `refuted` result merely by
exceeding one. A stopped global loop buffer is exact only when the whole-Program union of
emitted padded tiles reaches its full static extent.

## QSA correction

For the frozen T=32768 score/top-k Schedule, `stop(q)=floor((q+1)/4)` after clamping.

| Score tile | Whole-grid loop executions | MMA FLOPs | Counted FLOPs |
|---:|---:|---:|---:|
| 128 | 1,064,768 | 279,122,542,592 | 281,303,187,456 |
| 256 | 540,576 | 283,417,509,888 | 285,631,709,184 |

The tile256 predecessor charged 1,048,576 loop executions and 554,050,781,184 counted
FLOPs, overcounting the corrected work by about 1.94 times. `cast_query` remains
uncounted, so the corrected arithmetic is still a lower bound rather than exact work.
These counts are Compiler analysis, not latency, MFU, BWU, or a GPU result.

## P1-P8 and evidence boundary

- P1/P3: authors keep the one existing `LoopStop`; work and top-k cadence share one
  generic trip-domain owner, with no QSA branch or second stop spelling.
- P2/P5: every operation exposes an exact whole-grid repetition or an explicit missing
  reason; known work remains usable when a different operation abstains.
- P4: the modeled domain requires the existing scalar program axis and static extents;
  unsupported dynamic composition is `unknown`, not invalid and not guessed.
- P6: focused contracts cover tile128, tile256, clamp, zero and partial prefixes, another
  program-axis multiplicity, static nesting, multi-dynamic abstention, profile projection,
  top-k/work equality, and utilization error direction.
- P7: work, analysis, profile and utilization change together. The existing canonical QSA
  and generic dynamic-stop Corpus cases must retain their Schedule, Findings, lowering,
  manifest expectations, and exact full-Gate match.
- P8: the SM100a lowering and emitted loop stop are unchanged. The frozen QSA Program-v2
  must load all five byte-identical node lowerings under the successor.

Historical R10-R27 evidence remains append-only. The released SM100a Target still declares
`peak: null`, attention traffic is not an exact measured Program byte count, and the
Program mixes IEEE FP32 and BF16 contraction contracts. Therefore official complete-
Program MFU, BWU and roofline efficiency remain unknown; the 70% objective is not met by
this correction.
