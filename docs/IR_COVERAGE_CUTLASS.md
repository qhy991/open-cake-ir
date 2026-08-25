# CUTLASS 4.5.2, SM100: what its kernels ask a schedule to say

Surveyed against the typed IR. Thirty-six distinct schedule-level axes, from the SM100
collective builders, kernel schedules, tile schedulers, pipelines and epilogue builders.
Paths are under `cutlass-v4.5.2/include/cutlass/`.

## What the IR already says

Eight axes map directly, and they are not the peripheral ones:

* **Warp-role partitioning** -- every SM100 kernel is a five-or-six-way `WarpCategory`
  switch (`MMA`, `Sched`, `MainloopLoad`, `EpilogueLoad`, `Epilogue`, and `TensorMapUpdate`
  for pointer-array kernels). `MaxThreadsPerBlock` and every pipeline's producer/consumer
  assignment derive from it. `Role(name, warps)` is exactly this.
* **1SM versus 2SM UMMA** -- `MmaInstruction.cta_group`. This one bifurcates the whole
  builder tree, so it is worth noting that the IR carries it as an ordinary field.
* **Pipeline depth**, four separate times: mainloop, accumulator, scheduler, and the
  epilogue's C and D paths. All are `Pipeline(name, stages)`.
* **Operand major mode and operand source** -- `MmaInstruction.operand_major{k,mn}` and
  `operand_source{shared,tensor}` match the `_SS` and TMEM-source atom families exactly.
* **Raster order** -- `ProgramMap.traversal`, added this week; `{Heuristic, AlongM, AlongN}`
  maps onto it.
* **Epilogue subtile** -- `EpilogueParameters.subtile` with `TileLoop.tile`.

## The five gaps that matter

**1. Tensor memory is a resource with a protocol, not just a size.**
`Allocation.tensor_columns` holds the 512-column capacity. What it does not hold is the
allocation *lock*: the MMA warp holds it across the persistent loop and must
`release_allocation_lock()` before deallocating so the next CTA can rasterize, and on 2SM
the deallocation is an asymmetric leader/follower arrive-wait-arrive across the peer CTA.

*Partly closed.* `Allocation.allocating_role` now names the role that issues the
allocation and its release, which this repository's own CuTe-DSL backend had been
inferring from operation declaration order. What is still the backend's is *when* the
release happens -- it lands at the end of the role body, and CUTLASS's whole point is that
its position inside the persistent loop is a decision. The 2SM asymmetry is untouched.
Nor does it hold the warp-to-subpartition legality map -- `Shape<_2,_2>` or `Shape<_4,_1>`
decides which of four epilogue warps may touch which TMEM subpartition, and that is a
hardware rule, not a preference.

**2. A pipeline's kind is a semantic axis, not only its depth.**
Nine distinct pipeline classes appear in one kernel family, and they differ in *who issues
the consumer release* -- a warp, or the asynchronous UMMA unit itself. `Barrier.mechanism`
with two values is too coarse for that. Two more mechanisms are missing outright: an
ordered-sequence barrier that phases mainloop-prologue loads ahead of epilogue-C loads, and
a raw cluster barrier. Each pipeline also carries a cluster arrival mask derived from the
cluster shape and the block's position in it, which has no IR analogue at all.

**3. Stage count is a residual, and the IR treats it as independent.**
`StageCountAutoCarveout` computes mainloop depth as `(smem capacity - every other
pipeline's barrier storage - CLC response storage - tensormap storage) / per-stage bytes`.
The IR lets a Schedule state `Buffer.stages` and `Allocation.size_bytes` separately, with
nothing saying the first is a function of the second minus everything else's overhead.

**4. Two buffer-aliasing modes couple allocation to traversal.**
`IsOverlappingAccum` makes two accumulator stages share physical TMEM columns, which forces
stage selection by `phase() ^ 1` instead of `index()` *and* reverses the epilogue's subtile
traversal direction. `ReuseSmemC` aliases the store buffer onto the load buffer and thereby
serialises store completion against the next load. Both are aliasing decisions with
consequences for loop order, and neither is a `stages` integer.

**5. Where work comes from is separate from how work is mapped.**
`ProgramMap.persistent` says a fixed CTA count walks the tiles. It does not say whether the
next tile id is computed from a grid-stride or *fetched from a hardware queue* -- Cluster
Launch Control writes an opaque sixteen-byte response into shared memory through an
mbarrier. That distinction is why the `Sched` warp, the CLC pipeline and the throttle
pipeline exist at all, so it is not a detail below the schedule.

## Smaller, cheap, and evidenced

* `Buffer.swizzle` has four modes; a fifth, `SW128_32B`, is *mandatory* for MN-major TF32.
* Load movement is per-load in the IR but per-operand in reality: mixed kernels use TMA for
  A and `cp.async` for B in the same mainloop.
* `ProgramMap.traversal` needs an integer parameter -- the L2 rasterization swizzle width,
  with an explicitly unswizzled residual region.
* A register budget is per-role here, not per-kernel: transform kernels dynamically
  deallocate to 48 registers and reallocate to 256 per warp category. `residency` is
  kernel-wide.
* `DType` is missing tf32, e5m2, the FP6 and FP4 families, int8, complex and sparse
  elements.

## A finding in the other direction

`reduce_argmin`, with its tie-break and NaN-policy vocabulary, appears nowhere in the SM100
path. It exists in this IR because Flash-KMeans needs it. That is legitimate -- the IR
serves its corpus -- but it is worth recording that one of the operations is exercised by a
single operator while the gaps above are exercised by the whole library.

## Not in scope, deliberately

Two axes are genuinely outside one Schedule and should stay there: programmatic dependent
launch, which sequences *kernels*, and the split-K/stream-K decomposition with its
deterministic-or-not workspace fixup, which changes the reduction dataflow across CTAs
rather than the map within one.
