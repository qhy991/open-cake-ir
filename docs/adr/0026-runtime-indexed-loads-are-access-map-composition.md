# ADR 0026: runtime-indexed loads are AccessMap composition

Status: accepted, 2026-08-25.

## Outcome and non-goals

Describe the first data-dependent addressing step shared by KDA MoE v1 routing and
combine: a global load whose coordinates include INT32 values produced earlier in the
same kernel.  The addition is one `AccessIndexKind.BUFFER` spelling on the existing
`AccessMap`; it is not a `gather`, `routing`, `moe`, expert or KDA-version operation.

This slice does not add atomic slot allocation, indexed stores, scatter conflict
semantics, group Top-K, multi-kernel program composition, CLC, PDL or a performance
claim.  Complete KDA-version coverage therefore remains 0/57.

## Constraints and invariants

- `AccessMap` remains the sole owner of every global coordinate.  A `load` does not
  acquire a second indexing parameter.
- A `buffer` index names a rank-one, register-resident INT32 Buffer read by that
  operation.  Runtime coordinates are data dependencies, not hidden emitter inputs.
- All `buffer` indices in one AccessMap have the same shape and are zipped elementwise.
  Two `[k]` index Buffers address `k` coordinate tuples, not a `k x k` Cartesian product.
- Ordinary program, tile, loop and full-dimension coordinates keep their existing
  meanings and form independent result axes.
- `mask_tiled_axes` also bounds every runtime coordinate on both sides.  An out-of-range
  indexed load contributes zero.  There is no unchecked or wraparound spelling.
- The first lowering subset is a direct global load.  TMA and indexed writes are
  refused because their transfer and conflict contracts have not been defined.

The emitted result shape is derived from those facts: the common runtime-index domain is
included once, followed in access order by every other vector domain.  The staged Buffer
must have exactly that shape and preserve the loaded dtype.

## Dataflow and state transitions

The smallest complete KDA-derived path is:

```text
global expert ids  -> ordinary load -> register expert ids --+
global row ids     -> ordinary load -> register row ids ------+-> indexed load
global expert rows --------------------------------------------+       |
                                                                    register tile
                                                                          |
                                                                    ordinary store
```

The positive Corpus Schedule uses eight token programs and eight routed slots.  Its
indexed load addresses `[expert_id[slot], row_id[slot], hidden]`; the two address Buffers
are zipped, and the hidden dimension crosses that shared slot domain.  Deterministic
inputs include the `-1` sentinel used by KDA, so the independent oracle observes both
valid rows and the required zero result for an invalid route.

## Failure, rollback and compatibility

The companion Corpus case changes one runtime index to FP32 and must fail with a
localized index-dtype Finding before lowering.  Further contract probes cover an unknown
index Buffer, non-register placement, an undeclared read, unequal index domains, a
non-load user and TMA movement.  Missing coverage is reported as unlowerable; it is not
projected as a passed static check.

Existing AccessMaps contain no `buffer` coordinate.  Their verifier path and emitted
source remain byte-identical, so every prior lowering digest is expected to stay fixed.
Rollback removes the new enum member, verifier branch and Triton address branch together;
there is no compatibility adapter or second writable representation.

## Acceptance evidence

1. The positive and dtype-drift Schedules are separately reviewed before expectations
   are adopted.
2. All earlier Corpus findings and lowering digests stay unchanged.
3. Parser, schema, verifier and emitted-source contract tests cover the new coordinate.
4. The released generated source compiles and launches through the broker on B200.
5. An independent oracle matches every valid gathered value and every masked sentinel;
   no timing is taken.

Passing this slice proves runtime-indexed load composition only.  The next baseline
prerequisite is relaxed device-scope atomic fetch-add for unique group-slot ownership;
only after that composes with indexed stores can the route/dispatch kernel be claimed.
