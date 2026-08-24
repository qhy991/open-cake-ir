# What four libraries agree the IR is missing

The paper derives Cake IR bottom-up from CUTLASS, FlashInfer, FlashAttention-4,
TensorRT-LLM, DeepGEMM, Alpha-MoE and TileLang. This repository skipped that step and grew
its vocabulary one operator at a time. These are the surveys that step would have produced,
run against four of those sources plus kernel work carried out on this machine.

| Source | Axes | Fully expressible | Detail |
| --- | ---: | ---: | --- |
| Local kernel work (MoE sweep, PTX phases) | 15 | 5 | `IR_COVERAGE.md` |
| CUTLASS 4.5.2 SM100 | 36 | 8 | `IR_COVERAGE_CUTLASS.md` |
| FlashInfer attention | 26 | 2 | `IR_COVERAGE_ATTENTION.md` |
| DeepGEMM quantized and grouped | 31 | 6 | `IR_COVERAGE_QUANTIZED.md` |
| Megakernels and Kimi Delta Attention | 26 | 1 | `IR_COVERAGE_MEGAKERNEL.md` |

The counts are not the finding. The finding is that the misses fall into three groups that
need three different answers, and only one of them is a vocabulary problem.

## Group one: the vocabulary is thin, and that is fixable

CUTLASS and DeepGEMM mostly ask for *more of the same kind of thing*. A fifth swizzle mode.
Load movement declared per operand rather than per load. A register budget that belongs to a
Role instead of to the kernel -- every one of the four sources says this independently, and
three of them ship kernels that deallocate to a small budget in one role so another can have
more. A cluster shape between the grid and the CTA. Dtypes the IR does not list.

These are additions to a vocabulary that basically works, and the evidence for each is a
concrete kernel that cannot otherwise be written down.

Three are structural rather than additive but still local:

* **A pipeline's kind is semantic.** Nine pipeline classes appear in one CUTLASS kernel
  family and they differ in who issues the consumer release -- a warp, or the asynchronous
  MMA unit. Two barrier mechanisms cannot say that.
* **Stage count is a residual**, computed from shared-memory capacity minus every other
  pipeline's barrier storage. The IR treats stages and allocations as independent facts.
* **Aliasing couples allocation to traversal.** Two accumulator stages sharing physical
  columns forces both a different stage index and a reversed subtile order.

## Group two: three resources have lifetimes the IR does not model

Tensor memory is the clearest case and all three GPU-side sources hit it. It is not a size;
it is a resource with an allocation lock whose release ordering lets the next CTA rasterize,
a hardware legality map from warp index to subpartition, and -- in the megakernel -- a single
allocation handed from one operator to the next through a phase-keyed semaphore. In
DeepGEMM its columns are contended between accumulators and scales, and that contention is
what prunes the tile search.

Shared memory is the same story one level down. The megakernel refuses to have a
`byte_offset` at all: thirteen pages, and each operator receives a permutation chosen so the
pages it releases first are the ones the next operator needs first.

And a scale tensor is a *relation*, not a buffer. Its shape is a padded function of the
operand's, its major order is transposed against the operand's, and its granularity triple
selects among three kernel families. Writing the numbers down as integers loses exactly the
part that would let a verifier check them after a tile change.

## Group three: the schedule is a runtime value, and that is not a field

This is where the surveys converge hardest, and it is the same finding the local MoE work
produced before any of them ran.

* FlashInfer decides which tiles a CTA processes with a device array from a cost model --
  sometimes bin-packed by a *GPU kernel* -- or an occupancy measurement, or a per-SM atomic
  counter that lets a CTA choose at runtime which kernel body it is.
* DeepGEMM re-reads per-group extents from a device pointer inside the block loop, and
  names its host-sync boundary `has_synced_ks`. A whole alternative layout exists to remove
  that sync, at the cost of over-allocating to a bound.
* The megakernel's operation list *is* a device tensor, and the persistent loop's trip count
  is its row count. Choosing more attention partitions creates an operator that otherwise
  does not exist.
* Kimi Delta Attention gathers its carried state's leading index from another buffer, into a
  pool whose size is a runtime value.

CUTLASS shows what this costs a type system that tries to absorb it: `is_integral` is
specialised to `true_type` for a struct that carries a pointer, so a ragged extent can
inhabit a shape tuple.

Under this heading the IR's assumptions are load-bearing, not incidental. Every buffer shape
is a static integer list; every loop bound derives from a declared dimension; `grid` is a
static triple; one Schedule is one kernel with a fixed, statically ordered operation list.

## What to do about each

**Group one is work, and it should be ordered by evidence.** A register budget on `Role`
is asked for by all four sources and is small. *Done*: `Role.registers_per_thread` states
the split, and because `setmaxnreg` redistributes a launch-time allocation rather than
creating registers, the verifier holds the roles' total to the CTA allocation the residency
commitment declares, refuses a split that does not span whole warpgroups, and refuses a
partial one. The budget reaches the CuTe-DSL backend as `setmaxregister_increase` and
`_decrease`. It is now hardware-verified, and the verification produced a result worth keeping.

Installing CuTe-DSL closed a larger gap first: the warp-specialised Blackwell backend --
tcgen05 MMA, tensor memory, mbarriers, TMA -- had never been run. It compiles and matches
a float32 reference exactly, so both backends now have hardware evidence rather than one.

The register split itself compiles and runs. But the warpgroup-alignment rule was tested
by emitting a Schedule past the verifier with three *different* budgets issued from one
warpgroup, and that compiled and produced correct results too. That is not evidence the
split is legal: `setmaxnreg.sync.aligned` executed by part of a warpgroup is undefined, and
undefined behaviour routinely looks correct. What the run establishes is that the toolchain
accepts the violation silently. A gate that only restates what the compiler already
enforces adds nothing; this one catches what the compiler lets through, which is the
argument for keeping it blocking. Per-operand load movement and the missing
swizzle mode are each one kernel away from being needed. Pipeline kind and the carveout
dependency are real design, not fields.

**Group two needs a decision about what a resource is** before any field is added. The
current model -- a static byte partition per kernel -- is what makes all three cases
inexpressible, and adding a lock attribute to `Allocation` would not change that.

One half of the tensor-memory case turned out to be a live defect in this repository
rather than a gap against CUTLASS, and it has been closed. The CuTe-DSL backend already
emits the whole allocation protocol -- `allocate`, `wait_for_alloc`, then
`relinquish_alloc_permit` and `free` -- and it chose the warp that issues them by taking
the role of the first operation, in declaration order, whose reads touched tensor memory.
The verifier admits a Schedule where two roles read tensor memory (probed: it assesses
clean, and only the backend's own incompleteness stopped the lowering), so that inference
was ambiguous by more than accident: reordering two operations would have moved a
`tcgen05.alloc` to a different warp and its matching release with it.

`Allocation.allocating_role` now carries it, required for tensor memory and refused
elsewhere, and the corpus Schedule declares the epilogue -- what the inference picked, so
the emitted source is byte-identical and the hardware evidence taken on it still holds.
Naming `mma` instead moves the instruction, which is the point: an author can now change
a decision that used to belong to the backend.

That does not settle what a resource is. It settles who takes one out. Release *timing* --
CUTLASS releases inside the persistent loop so the next CTA can rasterize, this backend at
the end of the role body -- is still the backend's, and so is everything about the shared
memory page pool and the scale relation.

**Group three is not a vocabulary question.** There are two coherent answers and this
survey does not settle which:

1. A Schedule stays one statically-shaped kernel, and ragged extents, device-computed work
   lists and split-combine decomposition live above it -- which is where the paper already
   puts dispatch and portfolio concerns, and consistent with what this repository's own
   architecture says about the Compiler owning one kernel. Attention then becomes a family
   of static Schedules plus a scheduler that is not a Schedule.
2. A runtime extent enters the type system -- a declared bound with a device-resident actual
   -- and the verifier reasons about the bound while the loop reads the actual. This is a
   change to the type system, not the vocabulary, and it means re-examining every gate that
   is shape arithmetic, which is most of them.

The megakernel family is out of scope for either. A megakernel is an interpreter and its
schedule is its input; the distance from a Schedule is not a field.

## What a second operator actually cost

Softmax was admitted to test the claim the profile registry was reshaped to make: that an
operator is one row plus the Schedules that claim it. The bill, end to end:

* **Vocabulary: three additions and one merge.** `exp` and `div` joined `ElementwiseOp`.
  A max fold was needed, and that was the decision -- a `reduce_max` kind beside
  `reduce_sum` would have been two spellings of collapsing an axis, so `reduce_sum` became
  `reduce` with an `op`, which is the shape `elementwise` already had. `reduce_argmin`
  stayed separate because it returns an index, and that is what makes its tie-break and
  NaN policy observable at all.
* **Backend: one change, and it was a removal.** The Triton emitter required exactly one
  tile loop. Softmax is two passes over a row, so with a loop the verifier refused it --
  `BUFFER_ESCAPES_LOOP`, correctly: `weights` computed inside a loop holds only the last
  iteration. At this shape the row is already resident and there is nothing to iterate, so
  the Schedule declares no loop and the emitter no longer insists on one. The loop it used
  to emit had a trip count of one and an unused iterator.
* **Everything else: nothing.** One profile row, one conformance function, two corpus
  cases. No new finding code, no emitter structure, no scheduler.

**And a third operator cost nothing.** LayerNorm was admitted next, chosen because on
paper it needed no vocabulary: two folds, `mul`, `sub`, `square`, `add`, `rsqrt` and two
broadcasts, every one of them already there for the two before it. It landed that way --
one profile row, one conformance rule, two corpus cases, one oracle, and not a single
change to the IR or to either backend. Fifteen operations and eighteen buffers, correct on
a B200 against `torch.nn.functional.layer_norm` to 1.9e-06.

That is the claim under test, and softmax could not make it: softmax had to decide what a
reduction is. A vocabulary that stops needing additions is the only evidence that the
distillation converged, and one operator is the beginning of it rather than the proof.

The gate earned its keep twice here. It refused the two-pass-in-a-loop Schedule that would
have computed a softmax over stale maxima, and the profile rule refused a drift case with
a mismatched output shape. Both were predicted before running and both fired exactly there.

What this does **not** show is that attention follows. This softmax holds a whole row in
registers; a long one needs online rescaling, which is a different Schedule and probably a
different reduction vocabulary. ADR 0006 still stands.

## The finding in the other direction

`reduce_argmin`, with its tie-break and NaN-policy vocabulary, appears in none of the four
libraries. It is in the IR because Flash-KMeans needs it, which is legitimate -- an IR
serves its corpus. It is worth recording that one operation is exercised by one operator
while the gaps above are exercised by whole libraries, because that ratio is what a survey
before the fact would have shown and a corpus of three families cannot.

## Corpus coverage, restated

Ten cases across three families, against the paper's roughly four hundred across
twenty-eight. Attention, MoE, quantized GEMM and fused graph kernels -- the four families
these surveys are about -- have no local representation at all.
