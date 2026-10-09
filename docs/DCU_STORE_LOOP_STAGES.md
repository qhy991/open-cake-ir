# Explicit DCU store-loop stage selection

`Compiler.specialize_triton_store_loop(schedule, loop_name=..., num_stages=...,
schedule_id=..., entry_point=...)` changes the existing pipeline depth of one named
store-bearing TileLoop. The complete-Program action has the same name plus `stage`.
The existing transformation catalog is the callable authority; no separate task runner
or implicit optimizer is introduced.

The qualified input domain is gfx938/Triton, one pure role, ordinary input/output/register
storage and fixed unnested regions. The selected region contains ordinary input loads,
register arithmetic/casts/comparisons/selections and owned output stores. It contains no
reduction, MMA, state, synchronization or mutable output loads. Other sibling regions may
compute reductions. Existing canonical assessment owns typing, loop carry and store
coverage. Persistent maps, explicit allocation/residency commitments and dynamic stops
remain outside this action.

The caller selects a positive depth no larger than the selected loop's fixed trip count.
The pass neither clamps values nor selects a preferred depth. An unchanged depth returns
`unchanged`. The result retains buffers, arithmetic operations, accesses, Workload metadata,
Program bindings, outputs and every other loop option. Reassessment and lowering must both
succeed. No numeric reassociation, new instruction or memory-layout guarantee is inferred.

The development observation motivating this action is region-specific: a retained RMSNorm
loop changes device diagnostics when `num_stages` changes on its output loop, while the same
change on its sum loop has no corresponding effect. Other shapes and starters can be
faster. Use this action to construct a small number of explicit candidates, inspect each
compiled allocation and evaluate against the original oracle; do not treat higher depth
as better or profiled duration as unprofiled end-to-end performance.

F-2026-10-07-004 owns the original evidence. Initial software coverage includes a masked
117-column tail, complete-Program ABI preservation, unchanged sum-region options and
negative selection/commitment/ownership cases. Device qualification remains separate.
