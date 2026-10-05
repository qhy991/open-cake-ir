# Explicit squared-difference candidates

## Proposal and evidence boundary

C550's qualified N partition (F-2026-10-03-003) and single-stage partial unroll
(F-2026-10-04-005) motivate a joint candidate constructor. The existing K and N
passes each admit only the original loop-free graph, so applying one invalidates
the other's input domain. This is a pass-composition limitation, not evidence that
the IR cannot express the combination. Retained manually composed candidates
already lower; measured combinations have not beaten the strong N-partition seed.

Before implementation, the proposal satisfies P1–P8 as follows: reuse the existing
Schedule editing and Program rewrite API (P1); retain explicit tiles and range
options, and report their structural effects (P2); use the existing load/subtract/
square/sum/store form, with one shared builder (P3); keep construction, verifier
and backend admission (P4); preserve access ownership and reduction carry (P5);
check the unchanged Corpus and guarded counterexamples (P6); change no primitive,
type or analysis rule (P7); keep exact Target/toolchain admission and no inferred
hardware facts (P8).

## Callable contract

`Compiler.rewrite_program(program, "specialize_squared_difference", parameters)`
requires `stage`, `output_tile`, `k_tile`, `loop_unroll_factor`, `schedule_id` and
`entry_point`. The selected stage must be the original ordinary FP32 row/centroid
subtraction, square, sum and store. Existing loops, synchronization, allocations,
residency or additional operations are refused by their location. The public
Program tensors and bindings remain unchanged.

A tile equal to its full extent leaves that dimension unchanged. Smaller tiles
must be positive powers of two. An actual K loop is single-stage; its explicit
unroll factor must divide the fixed ceiling trip count. Without a K loop the
factor must be one. Tail accesses retain masks. Unit output partitions use a scalar program coordinate,
rank-one differences and an axis-zero sum; the reduced scalar remains `[1]`.
Only an input refusal for the replaced non-power-of-two arange extent can be
repaired through the Program API; all other input refusals remain blocking. The
fixed contraction Workload still owns its admitted shapes. FP32 subtraction precedes square;
K tiling changes reduction grouping, so external-oracle evaluation remains
mandatory. Backend qualification is not expanded by this constructor.

The old `tile_squared_difference` and `tile_squared_difference_outputs` entry
points retain their input and parameter contracts and share the construction code.
The new callable is available only when a Run grants it through the existing
transformation-access owner. Declaring it does not change frozen grants or arms.

## Feedback and selection

The result message reports the intermediate tile, logical elements per tile,
static CTA count, row-input load multiplicity and K trip/unroll counts. Logical
size is not a physical register count. Repeated input loads are not a prediction
of device-memory traffic because cache behavior is unmeasured. Result refusals
retain each blocking Finding's code, category, path and message.

No rewrite runs implicitly and no speedup is promised. Structural ranking remains
retired (ADR 0071). Actual compiled allocations stay with the existing toolchain
feedback; selection can use only an explicitly bound EmpiricalCostModel in its
qualified context. C550's unmodeled facts and timing limitations remain reported.
Promotion here is of a guarded candidate-construction mechanism. Device performance
qualification of the joint mechanism is pending a successor experiment; no
combined default or new IR primitive is promoted.
