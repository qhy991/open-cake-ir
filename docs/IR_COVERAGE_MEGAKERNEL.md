# Megakernels and Kimi Delta Attention: three assumptions, all load-bearing

Surveyed against the typed IR. `Megakernels/` fuses a whole transformer layer into one
persistent launch; `KDA-Pilot/` carries the CuTe-DSL Kimi Delta Attention decode kernel,
which is one of the paper's own benchmarks.

Twenty-six axes. One is fully expressible.

## Nothing crosses the CTA

`Barrier` is CTA-local by construction. The megakernel's entire dependency graph --
QKV to attention to output projection to up-gate to down-projection to the next layer, all
inside one launch -- runs on global atomic counters. A down-projection instruction on one
SM spins on `while (*(volatile int*)&Bar[...] < EXPECTED) __nanosleep(...)` until the
up-gate instructions across every SM have drained their stores and bumped the counter.

Five things are missing at once, and they are not independent:

* a mechanism value for atomic-counter-in-global with a spin and a backoff;
* `producers` and `consumers` that name Operations on *other CTAs*, not Roles in this one;
* a barrier whose identity is an index into a global buffer -- here `(layer, opcode,
  head_slot)`, fine-grained down to per-KV-head and per-reduction-column;
* a release-ordering attribute, since correctness depends on the store drain preceding the
  atomic;
* a host-side reset lifecycle, because the counters are refilled per token.

There is no device-side grid barrier anywhere in this design. Cooperative launch,
`grid.sync`, and cooperative groups appear zero times; the only true barriers are CTA-wide
at kernel entry and exit. The global counter *is* the synchronisation model.

## Nothing crosses the operator

Three resources have lifetimes that span sub-operations, and the IR partitions all three
statically per kernel.

**Shared memory is a pool, not a partition.** Thirteen sixteen-kilobyte pages, and each
instruction receives a logical-to-physical permutation chosen so that the pages it releases
first are exactly the ones the *next* instruction needs first. A page is handed forward the
moment its sixteen consumer warps arrive, strictly before the current instruction retires.
`Buffer.byte_offset` is precisely the thing this design refuses to have.

**Tensor memory is allocated once for the whole kernel** and passed from operator to
operator through a semaphore whose phase is the instruction index.

**The instruction pipeline stages the operator descriptor itself** -- a two-deep ring of
`{instruction, timings, page permutation, thirty-two semaphores, scratch}`. `Pipeline`
attaches `stages` to a data buffer inside one operator's loop; there is no noun here to
attach it to.

## Nothing is dynamic

The op stream is a device tensor of 128-byte instructions, and the persistent loop's trip
count is `instructions.rows()` -- a runtime extent. Opcodes are dispatched by a compile-time
chain that every one of the five role warps re-enters per instruction. Semaphore counts are
computed per instruction from its operands. The number of consumer warps that do work is a
function of a `reduction_elements` field, and the downstream barrier's expected arrival
count is another field.

The schedule itself is built in Python at model-load time -- a DAG over every layer,
list-scheduled onto SMs by a cost model with five assignment policies, frozen into an int32
tensor, then interpreted by the GPU. Its *shape* depends on runtime values: the number of
attention partitions is chosen from the prompt length, and choosing more than one
partition **creates an operator** that does not otherwise exist and re-points which barrier
slot the partial attention signals.

Work is addressed by `%smid`, not `blockIdx`.

## Kimi Delta Attention

The paper's own benchmark, and the IR owns exactly one of its axes.

**Carried recurrent state.** The K-by-V state matrix per request and head replaces the KV
cache entirely: loaded, mutated, and stored back every decode step, through a runtime
gather into a slot pool whose size is a runtime value. `Buffer.mode` has `input`, `output`
and `scratch` but no in-out; `AccessMap.indices` are static, so a leading index gathered
from another buffer has no form.

**The decay gate needs arithmetic the vocabulary does not have.** A per-channel decay
vector is `exp(-exp(A_log) * softplus(a + bias))` with a threshold branch to a linear tail,
and the correction term is a sigmoid. `ElementwiseOp` has square, rsqrt, add, sub and mul.
No exp, no log, no sigmoid, no softplus, no piecewise branch, and no warp broadcast.

**The recurrence is two passes fused into one.** The output must be read from the
*post-update* state, so one loop simultaneously writes the new state back to shared memory
and reduces it against the query. `reduce_sum` names a reduction over an axis; it does not
name an operation that mutates a buffer and reduces from it in the same pass, over a
strided lane mapping, with a shuffle-broadcast correction.

**The one axis that fits.** V-tile blocking of the state with double buffering maps exactly
onto `ProgramAxis`, `TileLoop` and `Buffer.stages`. That is the whole list.

Also worth noting: at small batch there are too few request-head pairs to fill the device,
so one state matrix is split across eight CTAs by V-tile range -- legal only because
distinct V columns of the state are independent. That is a second Schedule, chosen by a
runtime scalar, differing in warp count, grid, tile and decomposition.

## What this says that the other surveys did not

FlashInfer showed that a schedule can be a runtime value. This shows something stronger:
the *program* can be a runtime value. A megakernel is an interpreter, and its schedule is
its input.

That is not a gap in the vocabulary. A Schedule here describes one kernel with a fixed,
statically ordered operation list; a megakernel is one kernel whose operation list arrives
as data and whose operators hand each other on-chip resources. The distance is not a field.

The honest reading is that this family is out of scope for a Schedule and would need
something above it -- which is consistent with where the paper puts fused graph kernels in
its corpus, and consistent with what this repository's own architecture already says about
the Compiler owning one kernel. Recording it matters anyway, because "the corpus has no
fused graph kernels" and "the IR cannot express fused graph kernels" are different
statements, and only the second is actionable.
