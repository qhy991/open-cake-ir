# B300 Cake authoring notes from retained failures

These notes apply to the fixed B300 rewrite tasks. The Workload still owns shapes,
precision, oracle and tolerances; the external reference is not a correctness authority.

## Masked row tails already have a representation

For a row of width 7168, declare a column program coordinate with tile=8192 and use
that coordinate in the load and store. For width 1536, use tile=2048. The tensor's
actual shape remains unchanged; the emitted access uses its bounds mask. Normalize
by the actual width (7168 or 1536), not by the padded tile width. Loaded masked lanes
are zero, so a sum of squares has the right neutral value.

A direct slice `x[row, :]` still requests a 7168- or 1536-element arange and is refused
by Triton. A one-trip `lm.range` is not the spelling for a masked tile. Use an explicit
`lm.program(x, axis=1, dimension=1, tile=8192)` column coordinate alongside the row
coordinate. One store owns the output; do not split it into multiple store writers.
This observation is specific to masked sum-of-squares rows; do not assume zero is a
valid neutral element for every other reduction or operation.

## Broadcast axes name retained dimensions

For a tile `[R,C]`, a vector `[C]` spans axis=1; a row reduction `[R]` spans axis=0.
The axis is not the dimension being inserted. Thus:

```python
normalized = values * lm.broadcast(inverse_per_row, axis=0)
weighted = normalized * lm.broadcast(weights, axis=1)
```

A named broadcast
marker is also allowed on the successor frontend and erases into the same canonical
operation. Broadcast remains an explicit operand relation; it is not a materialized
splat, reshape, arbitrary Python alias or standalone value to store.

## Reductions and real loops

For a single row tile, use `across_loop=False`. For a reduction that carries over a real
K loop, omit `across_loop` to select the existing default; explicit `True` is not the
canonical serialized spelling. Keep the reduction axis on K, and fix the actual owning
refusal before trying another candidate.

## Preserve the structure being compared

Compare at least two complete work partitions: for example one row per program and
a small row pack, with correct row-tail masking and retained-axis broadcasts. Group
count tuning alone is not a structural alternative. A row pack can lose occupancy or
increase register pressure; do not presume it is faster. Keep the fixed original
starter as the measured baseline and retain null or negative results.

If construction or assessment refuses a candidate, fix the named semantic/shape
contract. Do not remove a verifier guard or invent an unsupported method. Record an
actual missing primitive only after an existing complete representation was ruled out.
