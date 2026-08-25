# ADR 0036: atomic reservation proves indexed-store ownership

Status: proposed, 2026-08-25.

## Outcome and non-goals

Admit an ordinary runtime-indexed `store` only when its destination coordinates
mechanically inherit exclusive ownership from the existing `atomic_rmw` result. Add no
conflict flag, uniqueness assertion, scatter operation, route operation or proof
framework.

This slice models KDA MoE v1's write to an atomically reserved expert row. It does not
admit an ordinary write merely because an author claims destinations are unique, atomic
reduction scatter as introduced by KDA v43, or the degree-one conditional direct store
introduced by v49. It also does not form routes, copy a complete hidden row, compose
kernels or make any performance claim. Complete KDA-version coverage remains 0/57.

## Constraints, invariants and owners

- `AccessMap` remains the sole address owner. `store` keeps its existing replacement
  effect; there is no second conflict-semantics field.
- An indexed store reads its value first, followed by the first-use ordered runtime
  index Buffers owned by its AccessMap. Its value tile has the AccessMap's zipped runtime
  domain once, followed by each independent tile or full-dimension domain in access
  order.
- The first admitted ownership proof has exactly two runtime coordinates. The first is
  the sole INT32 index of one rank-one state target; the second is the returned-old-value
  Buffer of the `atomic_rmw` that increments that target by one.
- The store reads the atomic result, and the general operation dataflow rules require
  that producer to execute earlier. Because the result is register-resident, the
  ownership rule independently requires reservation and store to use the same role; a
  barrier cannot transfer a register between roles. The atomic target index and result
  must be the exact Buffers named by the store AccessMap; aliases, transformed
  coordinates and handwritten assertions are not proofs. A second store-specific
  dependency-order rule would duplicate the general authority.
- For every successful atomic update at state coordinate `e`, increment-by-one returns
  a distinct prior value `p`. Therefore every valid `(e, p)` store coordinate has one
  owner across all lanes and programs. The state and destination reservation axes have
  the same extent, so their lower/upper-bound masks agree; the position bound masks
  out-of-capacity positions before the store. The static launch domain admits at most
  2^32 updates, before a returned INT32 bit pattern could repeat after wrap.
- The existing one-writer Buffer invariant proves no second operation mutates the state
  target. The first admitted subset executes both the atomic and store once per program,
  outside tile loops; this keeps the static wrap bound and ownership transition exact
  instead of borrowing a current backend's loop-position refusal.
- Other indexed-store forms fail before lowering. Atomic add of zero or an unrelated
  scalar remains a valid RMW operation, but it cannot prove store ownership.
- The Schedule does not constrain the caller's initial state values. An initial counter
  near signed INT32 wrap can therefore make positions negative or out of capacity; the
  store mask drops them without creating a conflicting write. Exact dispatch
  completeness needs a separate runtime extent relation. The correctness observation's
  pre-launch oracle owns its narrower all-reservations-fit input condition.

## Smallest complete vertical slice

One program loads a route's expert ids and payload values, reserves one position per
valid expert, then stores the payload at `[expert, old_count]`:

```text
expert ids -> load --------------------+----------------------+
                                        |                      |
state counts[expert] -> atomic_rmw(+1) -+-> old position ------+-> AccessMap
payloads -> load ----------------------------------------------+-> store payload
                                                               [expert, position]
```

Repeated expert ids across programs exercise contention, `-1` exercises the coordinate
lower bound, and non-zero initial counters leave untouched output prefixes that expose
stray writes. The generated-source test covers the position upper bound. The observation
oracle declares a narrower exact-set domain before launch: every reserved interval must
fit the output capacity. Within that domain it checks the set of payloads written into
each interval and the final counters without inventing an ordering among relaxed atomic
contenders.

## Failure, rollback and compatibility

Focused mutations must independently refuse reversed or unrelated coordinates,
mismatched mask domains, read-before-write dataflow or a cross-role race, a non-unit
atomic increment, loop repetition, a launch large enough to wrap, incorrect read
ordering and a value shape that does not match the addressed domain. The existing
arbitrary indexed-store negative remains rejected.

Existing direct stores and indexed loads retain their typed form and generated source.
Rollback removes only the verifier admission, its lowering evidence and the new Corpus
case; no schema vocabulary or compatibility path is involved.

## Acceptance evidence

1. schema and typed parsing add no vocabulary;
2. verifier tests prove the exact reservation-derived ownership relation and localize
   each unsupported near miss;
3. generated Triton source uses the existing zipped indexed address and mask;
4. the positive case passes the reviewed full Corpus Gate while the pre-existing
   arbitrary indexed store remains failure-capable;
5. one frozen, zero-retry brokered B200 correctness observation matches an independent
   set-valued oracle, with no timing claim.

The first frozen attempt (`gpuq-e988eaf8bb5c`) stopped at broker worker cwd admission:
a mode-0700 temporary parent made the checkout unreadable to the worker. It performed no
Python observer, compilation or GPU work and was not retried. Items 1--4 are satisfied;
item 5 remains missing, so this ADR remains proposed.

## KDA evidence and distinction

- v1 obtains `pos` from a relaxed GPU-scope atomic increment of `counts[local]`, then
  writes route metadata and dispatched data at `[local, pos, ...]`. The returned value
  is the ownership proof.
- v43 changes the G2 finalize path to weighted BF16 atomic addition because different
  experts can contribute to the same token output. That is reduction semantics, not an
  ordinary store.
- v49 encodes each token's local route degree during routing, skips output clearing only
  at degree one, and selects an ordinary vector store only under that runtime predicate;
  degree two or more retains BF16 atomics. That proof is not represented by this slice.

## References

- CAKE paper sections 2--4 and Appendix B.1: typed, analyzable primitives and localized
  pre-compile diagnostics.
- KDA MoE v1 routing body and the focused v42--v43 and v48--v49 source diffs.
- [`ADR 0026`](0026-runtime-indexed-loads-are-access-map-composition.md) and
  [`ADR 0035`](0035-atomic-slot-reservation-is-state-plus-rmw.md).
