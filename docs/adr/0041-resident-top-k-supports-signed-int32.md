# ADR 0041: Resident top-k supports signed INT32 values

## Status

Proposed. It authorizes implementation and Corpus Gate preparation, not a Compiler
Revision, provider run, GPU campaign, or Candidate promotion. Release remains behind the
external repository-owner approval gate.

## Irreducible goal

Express a deterministic reordering of selected block ids before QSA token expansion so
the selected-attention gather can consume position-local ids. The selected set and final
QSA semantics are unchanged; only an order that the Workload treats as irrelevant changes.

## Existing vocabulary and missing fact

The canonical `top_k` operation already expresses descending value order, lowest-source-
position tie handling, resident values, and paired value/position results. It currently
admits only FP32 scores. The canonical `cast` deliberately converts only among floating
types, so using FP32 as an adapter for INT32 ids would add ambiguous integer conversion
semantics solely to reach an operation that already has the required order contract.

## Decision

- Extend resident rank-one `top_k` to accept FP32 or signed INT32 source values.
- Values preserve the source dtype; source positions remain INT32.
- INT32 order is ordinary signed descending order. Equal values retain the existing
  lowest-source-position tie break.
- `nan_policy` remains the one canonical top-k syntax and is vacuous for INT32.
- The Triton lowering maps signed INT32 to monotonic unsigned order with a sign-bit XOR,
  packs that order and the existing tie key into one UINT64, applies `tl.topk`, then
  decodes the original INT32 values and source positions.
- INT32 `across_loop` remains explicitly unsupported. The current need is a complete
  resident reorder; extending carried initialization/finalization without a second use
  case would widen the contract speculatively.

No QSA-named Compiler branch, sort synonym, integer cast mode, or layout abstraction is
added.

## Failure semantics

- Unsupported source dtype remains `TOP_K_VALUE_DTYPE`.
- INT32 with `across_loop: true` is lowering-blocked by
  `TOP_K_INT32_ACROSS_LOOP_UNLOWERABLE` at the operation parameters.
- Shape, power-of-two, register-space, result-dtype, and k constraints remain unchanged.
- Candidate correctness and performance remain external GPU evidence.

## Acceptance evidence

1. One accepted Corpus Schedule protects signed INT32 values including negative sentinel
   ordering and UINT64 key/decode emission.
2. One failure Schedule protects the deliberate INT32 across-loop exclusion.
3. All pre-existing FP32 top-k lowering bytes remain unchanged.
4. The full Corpus Gate matches only after separately reviewed expectation adoption.
5. A repository-owner approval binds the exact successor Gate before release.
