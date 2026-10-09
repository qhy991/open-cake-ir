# Static output store partitions

An ordinary global output can have several stores when the verifier proves that
those stores form an exact partition. The proof uses the existing concrete
`AccessMap` coordinates; no new IR operation or layout algebra is introduced.

The admitted subset requires:

- One role writes each output, using ordinary `store` operations.
- The output has independent global storage, no allocation alias, no dynamic
  valid extent, and no reads inside the Schedule.
- The Schedule has no tile loop, mutable state, atomic operation or persistent walk.
- Each store consumes every program axis exactly once, at the same output
  dimension. Each axis covers that dimension exactly. Tiled program coordinates
  require a divisible dimension; masked program tails remain outside this proof.
- Other coordinates are static dimension intervals in their own dimensions.
  Rectangles stay in bounds, are pairwise disjoint, and their total volume equals
  the complete per-program output domain.

The proof is target-neutral. Backend admission and per-store value typing still
apply. `program_safety` checks every producer when examining ordering and
cross-role reads. The frontend keeps its serial effect dependency chain: a later
store depends transitively on all prior stores to the same output. Native emitters
visit each store independently. This extension gives no permission to overwrite a
register value or to read a partially written output.

The retained C550 examples are `fib_rmsnorm_h1536` event 5,
`add_rmsnorm_bf16` event 126, and `softmax_backward` event 141. Their proposed
intervals respectively split 1536 into 1024+512, 2560 into 2048+512, and 1024 into
512+512. Admission enables their independent evaluation; it does not establish
numerical correctness or a performance benefit on a device.

## Design principle check

P1 retains ordinary slicing and stores. P2 keeps each slice, store and program
mapping explicit. P3 introduces no alternate syntax. P4 retains existing value
and address typing. P5 adds a bounded concrete coverage proof. P6 adds positive,
negative and emitted-effect tests and runs the unchanged Corpus Gate. P7 updates
multi-producer ordering analysis with admission. P8 preserves the declared global
store effect and leaves backend/device qualification separate.
