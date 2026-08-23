# FlashInfer attention: where the static-shape assumption ends

Surveyed against the typed IR. Four kernel families that do not share a scheduling
vocabulary -- FA2 on SM80, FA3 on SM90, a cooperative persistent runner, and a CUTLASS
SM100 kernel with sixteen warps in six roles. Paths are under `flashinfer-d006-state-tma/`.

Attention is the family this corpus has no coverage of at all, and it turns out to be the
family that breaks the IR's central assumption rather than merely extending it.

## The finding

The IR states that every buffer shape is a static integer list and every loop bound derives
from a declared buffer dimension. In attention, **the schedule is a runtime value.**

Concretely, the thing that decides which tiles a CTA processes is, depending on the kernel:

* a device array produced by a host-side cost model that bin-packs work onto CTAs
  longest-first (`scheduler.cuh:913-948`);
* the same bin-packing **run as a GPU kernel**, emitting `work_indptr`, `qo_tile_indices`,
  `head_indices` and `batch_indices` into device buffers (`blackwell/plan.cuh:64-140`) --
  so the attention kernel's schedule is not known when the plan kernel launches;
* an occupancy *measurement*: `cudaOccupancyMaxActiveBlocksPerMultiprocessor` is queried
  and the answer decides whether to split the KV axis at all (`scheduler.cuh:176-206`);
* per-SM atomic counters, with each CTA reading `%smid`, claiming a slot, and thereby
  deciding **which kernel it is** -- prefill or decode -- with work stealing when one side
  drains (`pod.cuh:62-123`).

And the extents are ragged. A request's KV length is `indptr[i+1] - indptr[i]` pages with a
partially valid last page (`page.cuh:38-57`); the address function is a two-level gather
through a page table with a runtime divisor, not an affine map.

The clearest single piece of evidence is that CUTLASS had to lie to its own type system to
support this. `fmha_fusion.hpp:157-202` declares a pointer-carrying struct to be an
integral type:

```cpp
struct VariableLength { int* segment_offsets; ... };
template <> struct is_integral<VariableLength> : true_type {};
```

so that a ragged extent can inhabit a shape tuple. A static-shape tuple system does not
survive contact with ragged KV, and that is what the IR is.

## Where it shows up

Nine axes are hard NOs, and they are the top of the centrality ranking rather than the tail:

| Axis | Why it does not fit |
| --- | --- |
| Device-metadata work partitioning | `ProgramMap.persistent`/`.traversal` picks a static walk; here the CTA-to-tile map is an array, sometimes computed on the GPU |
| Ragged sequence length | Contradicts static shapes and derived loop bounds directly |
| Split-KV plus LSE combine | Split factor comes from an occupancy query or a per-request device pointer; no partial-result-plus-combine construct |
| Mask-driven work pruning | Causal, sliding-window and custom masks change *trip counts* and split one loop into masked, unmasked and window regions |
| Grid-wide sync over two tile sizes | One launch, three phases, `grid.sync()`, one shared block reinterpreted as three storage types |
| Runtime CTA-to-operator assignment | A CTA decides at runtime which kernel body it runs |
| Occupancy feedback | `residency` is a declaration this repo checks; here occupancy is measured and the measurement changes the algorithm |
| Empty-work fast path | Producer and consumer must take the same branch or the pipeline deadlocks -- a coupling with no vocabulary |
| Programmatic dependent launch | Inter-kernel overlap points placed inside the kernel body |

## What partially fits, and what the gap is inside it

**Online softmax** is the operation the IR is closest to expressing and still misses. The
rescale is a three-method protocol, not one operation: `update` computes a row max and the
scale `exp2((max_prev - max_cur) * scale)`, `rescale_o` applies that scale to the *other*
accumulator broadcast along N, and `finalize` does a quad all-reduce and then overwrites
the row sum with the log-sum-exp. `ElementwiseOp{mul}` with `broadcast_axis` covers
`rescale_o` exactly. What is missing is the loop-carried `{m, d}` state across tiles,
`reduce_max`, `exp2`, and -- notably -- a reduction whose *scope changes*: the in-loop sum
is warp-local by construction and the cross-lane part is deferred to the finalize step.
`ReductionScope` has one member, `cta`.

The state also lives in four different places across this one repository: per-thread
registers, CuTe fragments, shared memory, and TMEM -- where the softmax statistics are
*aliased onto the QK accumulator's columns* (`V0 = S0`) to act as a mailbox from the
Softmax roles to the Correction role.

**Warp roles** are named correctly by `Role(name, warps)`, but SM100 attention has six of
them and two are indexed peers of the same role -- `Softmax0` and `Softmax1` ping-pong over
alternating tiles under an ordered-sequence barrier, which is neither producer/consumer nor
a full-CTA barrier. One role, `Empty`, exists to hold registers at 24 so the others can have
more. Register budgets here are a table over role by tile by data-movement mechanism, not
the single number `residency.registers_per_thread` holds.

**GQA head mapping** has two incompatible forms in the same repository: FA2 fuses the head
into the Q tile axis as `token * group_size` and unpacks with a fast divide, while FA3
explicitly does not, and maps heads to `blockIdx.y`. A `ProgramAxis` can name a head axis;
it cannot say that the axis is a fused product of two, nor that two kernels for the same
mathematics choose opposite mappings.

## What this changes about the ordering

The gaps CUTLASS exposes are about *what a kernel commits to* and are additions to a
vocabulary that basically works. The gaps here are about *what a schedule is*, and they are
not additions.

There are two coherent responses and they are not the same project:

1. Accept that a Schedule describes one statically-shaped kernel, and put ragged extents,
   device-computed work lists and split-combine decomposition in a layer above it -- which
   is where the paper puts portfolio and dispatch concerns anyway. Attention then becomes
   expressible as a family of statically-shaped Schedules plus a scheduler that is not a
   Schedule.
2. Admit a runtime extent into the IR -- a declared bound with a device-resident actual --
   and let the verifier reason about the bound while the loop reads the actual. This is
   what `VariableLength` is, and it is a change to the type system rather than to the
   vocabulary.

The second is the larger claim and would need the verifier's gates re-examined one by one,
because most of them are shape arithmetic. Nothing here settles which is right; what the
survey settles is that adding fields will not reach this family.
