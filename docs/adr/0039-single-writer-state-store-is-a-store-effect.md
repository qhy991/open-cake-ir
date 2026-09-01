# ADR 0039: a single-writer state update is a proven store effect

Status: proposed successor Compiler change, 2026-09-01.

## Decision

Allow the existing `store` operation to target a global `state` Buffer only when the
Verifier proves that the Schedule gives every launched program a disjoint destination
region. `Buffer.mode` continues to own whether memory is caller-owned mutable state;
`AccessMap` continues to own coordinates. No new operation kind, ownership flag, effect
string, pointer expression or Workload assertion is added.

The first admitted form is deliberately static and direct:

1. the Schedule uses `program_map`, not legacy `grid`;
2. the state store executes outside every `TileLoop`;
3. every Program axis appears exactly once in the store AccessMap;
4. that axis is owned by the same state Buffer and the same destination dimension;
5. scalar `program` coordinates require tile 1; `program_tile` uses the declared tile;
6. every remaining destination dimension is covered once by its full `dimension`;
7. no `buffer` or `loop_tile` index participates;
8. existing rules still require the state to be read and written exactly once, preserve
   dtype, obey operation order and lower through the exact Target route.

`outputs` may be empty only for a Schedule that updates caller-owned state. The generated
host wrapper returns an empty output tuple while the caller observes the in-place state
effect. A Schedule with neither an output nor state is rejected; inventing a replacement
output for an in-place parent would change its ABI.

This proves a one-to-one partition from program coordinates and resident element lanes to
state coordinates. The state Buffer itself is the ownership authority; an equally shaped
input cannot stand in for it.

## Why this is an IR boundary

The current IR already has caller-owned state, direct AccessMaps and a Triton `tl.store`
body, but the Verifier rejects every ordinary store to state as `OP_STORE_DESTINATION`.
The only admitted state effect is a narrow INT32 atomic add. Independent AKA-derived
contracts repeatedly require a computed FP32 value to overwrite a caller-owned element
under single-writer ownership. Encoding those as atomics changes semantics; copying state
to an output changes the public ABI and lifecycle owner.

The missing concept is therefore not a workload-sized state-update operation. It is the
legality proof for an existing store effect.

The provisional AKA review supplies independent pressure, not acceptance evidence:

- `case-000524` isolates a fixed-instance ownership-checked non-atomic state update;
- `case-000517`, `case-000539`, `case-000540` and `case-000568` describe computed
  single-writer FP32 state stores from different operator/source paths;
- `case-000367` describes the same effect for a contiguous elementwise state update;
- `case-000180` still requires an indirect predicated state store and remains outside this
  proposal.

All are `semantic_binding=reviewer_claimed` and `gpu_test=not_run`. They justify testing
this narrow verifier rule, not a correctness or coverage claim.

## Non-goals

- no runtime or symbolic Buffer shape, grid or loop bound;
- no unchecked gather/scatter or runtime-indexed state store;
- no `unique`, `collision_free` or user-asserted ownership flag;
- no cross-CTA reduction or ordering guarantee;
- no atomic replacement, compare-and-swap or memory-order extension;
- no aliasing between state and another global Buffer;
- no store inside a tile loop;
- no partial state slice, broadcasted destination or arbitrary pointer arithmetic;
- no Program IR, Portfolio rule or Workload precondition in the Compiler.

Reservation-owned indexed stores remain the sole admitted indirect ordinary-store form.

## Verifier and lowering

The Verifier localizes an unproved state write to the store AccessMap. It checks the
ProgramMap axis owner, destination dimension, coordinate kind, tile relation, full
dimension coverage, loop placement and unused launch axes. A failure blocks acceptance as
program safety; it is not a backend gap.

The Triton emitter already derives the pointer and mask from the AccessMap and emits
`tl.store`. The host wrapper already passes state as caller-owned memory. The successor
must demonstrate that the generated source writes the state pointer rather than allocating
or returning a replacement tensor.

## Acceptance evidence

The proposal requires:

- one fixed-shape FP32 `state = state + update` Corpus case;
- a near miss whose Program axis is not consumed by the state store;
- a near miss whose Program axis is owned by a different Buffer;
- exact localized Findings for both near misses;
- deterministic Triton lowering for the positive case;
- all pre-existing Corpus cases unchanged;
- full Compiler contract tests and Corpus Gate;
- external human approval before any released successor lock;
- independent complete-output GPU correctness before a correctness or coverage claim.

This automation may prepare the successor draft and Gate. It must not create or modify
`compiler/release-approval.json`.
