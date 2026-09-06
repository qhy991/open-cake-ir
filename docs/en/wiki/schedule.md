# Read a Schedule: data, workers, operations, and addresses

[中文原文](../../wiki/schedule.md) · [English home](../README.md)

A Schedule answers how to compute. We use the same complete [FMA plan](../../../corpus/schedules/fma-b8-smoke.json) as the [tutorial](../GETTING_STARTED.md).

[Guide index](README.md) · [Primitives](primitives.md)

## 1. The table of numbers

The a, b, c, and y arrays each have `shape: [8,128]`. `a[2,5]` means the third row and sixth value because indices start at zero. Each output is `y[i,j]=a[i,j]*b[i,j]+c[i,j]`.

A tensor is an array with specified dimensions and order. Dtype describes number storage: FP32 uses four bytes per value, BF16 two. Floating-point storage can only approximate some numbers, so storage choice affects precision.

## 2. Buffer: where values live

| Buffer | Storage | Purpose |
| --- | --- | --- |
| a, b, c | global GPU memory | Inputs, unchanged after the call |
| a_tile, b_tile, c_tile | registers | Temporary row values |
| y_tile | registers | Temporary answers |
| y | global GPU memory | Output |

A logical 128-value tile does not mean each thread owns 128 physical registers. Actual allocation belongs to compiled evidence. Shared memory supports cooperating threads; `tensor` names target-specific storage. These spaces cannot be exchanged just by renaming them.

Modes are input, output, scratch, and caller-owned mutable state. State survives the call and may be used again.

## 3. ProgramMap: who handles a row

The batch program axis assigns one row at a time from dimension zero of a, giving eight row jobs. The compute role uses four NVIDIA warps of 32 threads each. A row job, a warp, and one number are different units. Larger plans use tiles and tile_loops; this example needs no tile loop.

## 4. Operations and dependencies

Three loads feed FMA, then store_y writes the result. Each operation has a unique id, a kind, and explicit reads and writes. `depends_on` names required earlier operations, not a guessed machine instruction count. `parameters.op` names arithmetic; `instruction.contract` fixes its numerical realization.

## 5. AccessMap: which position

For load_a, the first coordinate comes from batch and the second walks dimension one. `mask_tiled_axes` checks tiled boundaries. b, c, and y use the same row coordinate. A correct formula attached to the wrong row still produces the wrong program. AccessMap is the sole coordinate owner.

## 6. Remaining fields

`outputs` exposes y. `target` names sm_100a. `lowering.backend` chooses Triton and `entry_point` names the generated function. `residency` declares resource commitments rather than speed. Allocations, pipelines, and barriers are empty here and describe storage and collaboration in larger plans.

Copy a complete example outside the checkout, change one decision, and reassess. Check Workload permission before changing mathematical inputs or outputs; check primitive composition before adding vocabulary. Exact fields belong to the [Authoring Contract](../../../compiler/AUTHORING_CONTRACT.md) and the selected Compiler.
