# ADR 0035: atomic slot reservation is state plus RMW

Status: accepted, 2026-08-25.

## Outcome and non-goals

Express KDA MoE v1's unique local-expert slot reservation with two orthogonal
primitives: a caller-owned global `state` Buffer and one `atomic_rmw` operation. The
operation atomically adds an INT32 scalar and returns the old value. Add no `route`,
`dispatch`, `slot`, `counter`, `scatter`, `moe` or KDA-version spelling.

This slice does not form routing groups, select experts, write runtime-indexed
destinations, copy dispatched rows, quantize, perform either grouped GEMM, combine
outputs, reset persistent workspace, compose kernels, use PDL or make a performance
claim. Complete KDA-version coverage therefore remains 0/57.

## Constraints, invariants and owners

- `state` is a Buffer mode, not an operation flag. It names global memory supplied by
  the caller whose pre-launch value is readable and whose post-launch value may differ.
  A state Buffer must be both read and written; a read-only or write-only use must use
  the existing `input` or `output` mode instead.
- `AccessMap` remains the sole address owner. The first admitted atomic target is a
  rank-one global state Buffer indexed by one rank-one register INT32 Buffer.
- `atomic_rmw` is one canonical read-modify-write form. Its admitted operation is INT32
  `add`; its scalar is in signed INT32 range, while `order: relaxed` and `scope: device`
  are visible because they are correctness and hardware commitments. The scalar update
  is explicit and the result is always the value that preceded this operation.
- The operation reads the target followed by its AccessMap-owned runtime index Buffers.
  It writes the same target followed by one register result. This makes both the memory
  effect and returned data dependency visible without a second target field.
- A masked runtime coordinate performs no memory effect and yields zero in the result.
  KDA still carries its `-1` expert sentinel separately, so the unused position value is
  not mistaken for a valid route.
- The SM100 Triton lowering maps `device` to Triton's GPU scope and requires the Target
  contract `triton.atomic_add.i32.relaxed.gpu`. Other dtypes, operations, orders, scopes,
  target modes and result placements fail before lowering.

## Dataflow and state transition

The smallest complete slice launches eight token programs over eight route slots:

```text
global expert ids -> ordinary load -> register expert-id tile ----+
                                                                |
caller state counts[expert] -------------------------------------+-> atomic_rmw(+1)
                                                                      |        |
                                                        mutated counts state   old positions
                                                                               |
                                                                         ordinary store
```

The deterministic observation begins from non-zero counters, contains repeated expert
ids across programs and includes `-1`. Because relaxed atomics do not define which
contender receives which old value, correctness is the invariant rather than an invented
lane order: for each expert, returned positions are exactly a permutation of the
pre-launch half-open counter range; invalid routes return zero; final counters equal the
initial value plus the number of valid reservations.

## Failure, rollback and compatibility

Focused mutations must localize wrong state mode, target/result dtype, result placement,
read/write order and AccessMap shape before lowering. A Target without the exact atomic
contract remains accepted only if another backend can implement the semantics; the
current Triton route is lowering-ineligible.

Existing Schedules contain neither `state` nor `atomic_rmw`. Their typed forms and
generated sources must remain unchanged. Rollback removes the Buffer mode, operation,
verifier and emitter body together; there is no compatibility alias or alternate
reservation spelling.

## Acceptance evidence

1. schema and typed parsing expose exactly `state` and `atomic_rmw`, with no use-case
   vocabulary;
2. verifier tests cover effects, types, addressing, ordering, scope and Target support;
3. a positive Corpus case passes the reviewed full Gate while mutations fail for the
   intended localized reasons;
4. generated source contains one masked relaxed GPU-scope atomic add and returns its old
   values;
5. the frozen zero-retry plan produced
   [`ATOMIC_RESERVATION_B200_OBSERVATION_20260825.json`](../../inventory/ATOMIC_RESERVATION_B200_OBSERVATION_20260825.json):
   one brokered B200 compile and launch satisfied all 64 contention outputs from non-zero
   initial state, with no mismatch and no timing.

Passing this slice proves unique slot ownership only. Runtime-indexed store conflict
semantics remain the next prerequisite before route/dispatch can be claimed.

## References

- CAKE paper sections 2--4 and Appendix B.1: typed, performance-transparent,
  hardware-grounded primitives with localized pre-compile diagnostics.
- KDA MoE v-01 `kernel.py:374-380`: relaxed GPU-scope atomic additions to per-expert
  counts return the row position used by dispatch.
- [Triton `atomic_add` API](https://triton-lang.org/main/python-api/generated/triton.language.atomic_add.html):
  returned-old-value, `relaxed` semantics and `gpu` scope contracts.
- [`ADR 0020`](0020-kda-deltas-are-an-external-expressibility-corpus.md) and
  [`ADR 0026`](0026-runtime-indexed-loads-are-access-map-composition.md).
