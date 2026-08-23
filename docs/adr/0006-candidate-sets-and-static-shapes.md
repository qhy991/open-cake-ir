# ADR 0006: candidate sets, and a Schedule stays statically shaped

Two decisions the surveys left open. Both are settled against the paper's stated design
rather than against convenience, and both have consequences worth writing down.

## Candidate sets: adopt

The paper's loop generates *structurally distinct* candidates and then filters them with IR
construction checks, verifier gates and cost-model ranking before spending GPU time. One
candidate per Turn leaves the ranking stage with nothing to rank, so the current shape is
not a conservative version of the loop; it is a loop missing a stage.

The hesitation was that allowing a set changes what a Turn is, and a Turn is part of a
Study Contract's treatment. That hesitation does not survive inspection. How many
candidates a Turn may write is a property of the authoring protocol, and it is granted
identically to both arms. What differs between the arms is what happens to those candidates
before they reach a GPU -- the Cake arm has a verifier and a ranking, the direct CUDA arm
has a compiler -- and that asymmetry *is* the treatment the study exists to measure.
Allowing sets sharpens the contrast rather than confounding it.

Frozen Study Contracts keep their bindings. The change arrives as a successor.

## Runtime extents: refuse, and put the ragged part above the Schedule

FlashInfer, DeepGEMM and the megakernels all decide their work at runtime -- a device array
from a cost model, an occupancy measurement, a per-SM atomic counter, an operation list
that is itself a tensor. Two answers were coherent: keep a Schedule statically shaped and
put that layer above it, or admit a runtime extent into the type system as a declared bound
with a device-resident actual.

**P4 decides it.** The principle is to use typing rules to reject ill-typed programs *during
construction*. A runtime extent turns every shape gate from something the verifier proves
into something it bounds, and most of the gates are shape arithmetic. That is not a small
weakening of one rule; it is a change in what the verifier is.

Two further points support the same reading. The `is_integral<VariableLength>`
specialisation that made ragged extents inhabit a shape tuple is CUTLASS's, not Cake's --
the paper does not claim its IR absorbed them. And the paper places generalisation in a
later stage gated behind strong per-shape seeds, which is where a scheduler that is not a
Schedule belongs.

### The consequence, stated plainly

Attention becomes expressible only as a family of statically shaped Schedules plus a
scheduler that is not a Schedule, and that scheduler does not exist here. Claiming coverage
of attention therefore means building that layer, not adding fields. Anyone reading the
corpus should know that the absence is deliberate rather than pending.

Megakernels stay out of scope under either answer. A megakernel is an interpreter whose
schedule is its input.

## What follows

Build the ranking primitive at the Compiler level first, where it is testable without
touching a contract, then change the Lab to produce sets. The other order leaves the
ranking with nothing to rank and churns contracts first.

On what a cost model may be: it will not invent a time. A Target declares no clock and no
bandwidth, and a predicted time derived from neither would be exactly the invented number
the analysis has refused to produce from the start. DeepGEMM is the precedent that this is
not a cop-out -- its own layout comparison ranks on wave count and last-wave utilisation,
and its `num_cycles` field is hardwired to zero behind a TODO. Measurement on this machine
supports the same choice: across two kernels binding on two different resources, the
predicted binding resource was correct both times while the magnitudes were loose. Rank on
what is measured to be right.
