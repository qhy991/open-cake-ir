# IR primitives: the building blocks of a plan

[中文原文](../../wiki/primitives.md) · [English home](../README.md)

These are operations a Schedule can use, not a promise that every GPU operator is supported. Types, shapes, addresses, targets, and backends still constrain complete plans.

[Guide index](README.md) · [Operators](operators.md) · [Operation definitions](../../../src/open_cake_ir/compiler/ir/operations.py)

## load

Read values into a temporary Buffer. AccessMap supplies coordinates, including admitted INT32 runtime indices. Multiple index arrays pair position by position rather than forming every combination. Global loads and TMA transfers have different requirements; reuse is an intent, not a speed guarantee.

The [indexed-load example](../../../corpus/schedules/indexed-gather-b8-smoke.json) zero-fills masked invalid positions. This is not permission for every access to go out of bounds. Result shape must match the accessed domain: a scalar address produces canonical `[1]`, not an implicit 128-value copy. See the [repair](../ACCESS_DOMAIN_REPAIR_20260906.md).

## store

Write a value to the AccessMap destination with the declared type and rounding. Writing caller state needs a proof of disjoint ownership, or the admitted reservation-derived indexed-store proof. A claim that indices are unique is insufficient. See [state update](../../../corpus/schedules/state-store-b8-smoke.json) and [reservation-owned store](../../../corpus/schedules/reservation-owned-store-b8-smoke.json). State-only calls may return an empty tuple.

## elementwise

Compute at corresponding positions, such as `[1,2]+[3,4]=[4,6]`.

| `op` | Meaning | Example |
| --- | --- | --- |
| `square` | Square | 3 → 9 |
| `abs` | Absolute value | -3 → 3 |
| `round` | Round with the declared rule | -2.5 → -3, ties away from zero |
| `divide_no_nan` | Divide, returning zero for a zero denominator | 6/3 → 2; 6/0 → 0 |
| `rsqrt` | Reciprocal square root | 4 → 1/2 |
| `exp` | Exponential | 0 → 1 |
| `relu` | Replace negatives with zero | [-2,3] → [0,3] |
| `tanh` | Map toward the interval -1 to 1 | 0 → 0 |
| `add` | Add | 2+3 → 5 |
| `sub` | Subtract | 2-3 → -1 |
| `mul` | Multiply | 2*3 → 6 |
| `div` | Divide | 6/3 → 2 |
| `fma` | Fused multiply-add | 2*3+4 → 10 |

FMA rounds `a*b+c` once rather than separately rounding the product. It requires three same-shaped FP32 register inputs and `ptx.fma.rn.f32`, without scalar or broadcast shortcuts. Tanh also requires an explicit instruction contract. See [FMA](../../../corpus/schedules/fma-b8-smoke.json), [nested FMA](../../../corpus/schedules/fma-chain-b8-smoke.json), and [ReLU](../../../corpus/schedules/relu-b8-smoke.json).

Round requires `rounding=nearest_away_from_zero`. The quantization composition operations
retain exact-target, FP32 and shape constraints; their names do not bypass assessment.

## cast

Convert numerical storage type. More bits do not recover previously lost precision; fewer bits may round again. This does not reshape a tensor or move it to another GPU. See the [cast plan](../../../corpus/schedules/cast-b8-smoke.json); allowed pairs depend on the Compiler and backend.

The quantization producer uses this same cast operation. FP32-to-FP16 explicitly uses
`rounding=nearest_even, overflow=ieee`; FP32-to-INT8 uses
`rounding=toward_zero, overflow=forbid`. The rounding and overflow policies appear together.

## reshape

Regroup the same register-held values, for example `[4]` into `[2,2]`. Element count,
order and dtype stay unchanged. This performs no global-memory copy and introduces no
layout algebra. The verifier checks input/result and target constraints; the Q8
producer uses it to group values into blocks of 32.

## mma

Matrix multiply-accumulate: `[2,3]` and `[4,5]` give 23 by paired products and a sum. In this project's rank-two contraction, the final axis of both inputs is K. Types, accumulator, tile, and instruction are explicit. FP32 storage with TF32 multiplication must be requested; multiple explicit MMA nodes compose through dependencies. See [GEMM+bias](../../../corpus/schedules/gemm-bias-b1-smoke.json) and [TF32](../../../corpus/schedules/fp32-tf32-mma-b1-smoke.json).

## epilogue

A fixed post-contraction formula: `centroid_sq_minus_two_dot` for clustering or `bias_add_bf16_round` for bias and BF16 rounding. It is not an arbitrary function body. See [clustering](../../../corpus/schedules/flash-kmeans-assignment-full.json) and the [fixed linear asset](../../../corpus/schedules/tinygemm2-stage4-split-k.json).

## reduce

Collapse one axis using sum or max: `[2,5,1]` gives 8 or 5. Axis selects the dimension; scope selects cooperating threads. See [softmax](../../../corpus/schedules/softmax-b8-smoke.json).

The ordinary algorithm is `backend`. Explicit `xor_tree_32` requires a resident FP32
last axis of 32 and combines values in XOR-offset order 16, 8, 4, 2, 1 outside TileLoops.
That order is part of the numerical contract and cannot be exchanged for another tree.

## reduce_argmin

Return the index of the smallest FP32 value as INT32. `[4,2,8]` returns position 1. A Workload separately defines acceptable equal-distance outcomes. See [nearest centroid](../../../corpus/schedules/flash-kmeans-b32-smoke-v2.json).

## top_k

Return the largest k values and original positions in descending value order, smaller position first on ties. `[4,9,9,1]` with k=2 gives `[9,9]` and `[1,2]`. Resident inputs include FP32 and signed INT32; floating input has an explicit NaN policy. Streaming selection also carries state and has its own limits. See [resident](../../../corpus/schedules/top-k-b8-smoke.json) and [streaming](../../../corpus/schedules/top-k-streaming-b8-smoke.json).

## scan

Keep the running sums: `[2,5,1]` becomes `[2,7,8]` forward or `[8,6,1]` reverse. The admitted operator is sum, with forward/reverse direction. It cannot be replaced by one reduction result. See [forward](../../../corpus/schedules/chunk-cumsum-b8-smoke.json) and [reverse](../../../corpus/schedules/chunk-cumsum-reverse-b8-smoke.json).

## index_expand

Turn block ids into element ids. With scale=4 and extent=4, block 2 produces `[8,9,10,11]`; a later load fetches data. Invalid blocks retain the declared sentinel, such as -1. See [index expansion](../../../corpus/schedules/index-expand-b8-smoke.json).

## online_softmax

Process score/value tiles while maintaining the running maximum, exponential sum, weighted accumulator, and normalized output. A new maximum rescales previous totals. Equal scores with values `[10,20,30]` produce 20, also when processed in batches under the declared numerical contract. See [online softmax](../../../corpus/schedules/online-softmax-b8-smoke.json).

## atomic_rmw

Indivisibly read, modify, and write shared state, returning the old value. A counter starting at 5 hands out 5, then 6 for successive increments. The admitted form is INT32 add, relaxed order, device scope; it does not synchronize every unrelated access. See [atomic reservation](../../../corpus/schedules/atomic-reservation-b8-smoke.json).
