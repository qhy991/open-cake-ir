# Common operators through small numerical examples

[中文原文](../../wiki/operators.md) · [English home](../README.md)

An operator is a defined computation. It may use several IR primitives or several GPU kernels. These examples explain mathematics, not measured GPU results.

[Guide index](README.md) · [Primitives](primitives.md) · [Workloads](workloads.md)

## Matrix multiplication, bias, and grouped GEMM

A matrix is a table. Paired products of `[2,3]` and `[4,5]` sum to 23; bias 1 makes 24. Linear layers use this idea. Large tasks accumulate across K tiles. Ragged groups have different valid row counts, and padding must not count as data. Block-scaled low-precision inputs require the correct scale association.

See [GEMM+bias](../../../corpus/schedules/gemm-bias-b1-smoke.json), [ragged groups](../../../corpus/schedules/ragged-grouped-gemm-b1-smoke.json), and [block scales](../../../corpus/schedules/block-scaled-gemm-b1-smoke.json).

## Affine computation

Multiply each row by its scale and add its bias: `[2,3]*4+1=[9,13]`. NCHW plane affine uses a separate coefficient for every batch/channel pair. FMA also fixes one-rounding semantics. The [eight-output GPU example](../AFFINE_PARENT_B200_CANARY_20260906.md) shows inputs, answers, and the exact evidence boundary.

## RMSNorm

Square a row, average it, and scale the original values by the reciprocal square root, then multiply weights:

`y=x/sqrt(mean(x²)+epsilon)*weight`

For `[3,4]`, mean square is 12.5. Unit weights and negligible epsilon give about `[0.849,1.131]`. No mean subtraction occurs. See [RMSNorm](../../../corpus/schedules/rmsnorm-b8-smoke.json).

## LayerNorm

Subtract the mean, normalize by variance plus epsilon, then apply weight and bias:

`y=(x-mean(x))/sqrt(mean((x-mean(x))²)+epsilon)*weight+bias`

For `[1,3]`, mean 2 gives centered `[-1,1]`; unit weight and zero bias produce nearly that pair. See [LayerNorm](../../../corpus/schedules/layernorm-b8-smoke.json).

## Softmax

Turn scores into nonnegative proportions summing approximately to one. Equal scores `[0,0]` give `[0.5,0.5]`. Subtract the maximum for numerical range, exponentiate, sum, and divide. See [softmax](../../../corpus/schedules/softmax-b8-smoke.json) and the tiled online variant in [primitives](primitives.md#online_softmax).

## ReLU and SwiGLU

ReLU maps `[-2,3]` to `[0,3]`. SwiGLU combines a gate and values, commonly `sigmoid(x)*x*value`; x=0 produces zero. The example builds sigmoid using an explicit tanh contract and elementwise multiplication. See [ReLU](../../../corpus/schedules/relu-b8-smoke.json) and [SwiGLU](../../../corpus/schedules/swiglu-b8-smoke.json).

## RoPE

Rotate a pair with supplied sine and cosine:

`y1=x1*cos-x2*sin`, `y2=x1*sin+x2*cos`.

`(1,0)` with cos=0 and sin=1 becomes `(0,1)`. The [RoPE plan](../../../corpus/schedules/rope-b8-fused.json) consumes coefficients; it does not claim that calculating sine and cosine is supported by that plan.

## Flash-KMeans

Assign each point its closest centroid. Point 3 and centroids `[0,5]` have squared distances `[9,4]`, so the index is 1. In many dimensions, sum squared differences. A term shared by every centroid may be omitted when only the winner matters. Matrix multiplication supplies many scores and argmin chooses positions. See the [plan](../../../corpus/schedules/flash-kmeans-b32-smoke-v2.json) and [task contract](workloads.md#flash-kmeans).

## Indexing, padding, combination, and state

Gather `[10,20,30]` at `[2,0]` to get `[30,10]`. Mask invalid rows using the declared valid prefix. Weighted combine may compute `0.25*10+0.75*20=17.5`. A state update changes caller storage `[1,2]` by `[3,4]` into `[4,6]`. Scan keeps running sums; top-k keeps values and positions.

See [gather](../../../corpus/schedules/indexed-gather-b8-smoke.json), [padding](../../../corpus/schedules/ragged-zero-pad-b1-smoke.json), [combine](../../../corpus/schedules/kda-weighted-combine-b8-smoke.json), and [state](../../../corpus/schedules/state-store-b8-smoke.json). Concurrent writes still need a supported ownership proof.

## Attention

Query describes what is sought, key describes how information is matched, and value supplies the information. Scores become softmax weights and combine values. Three equal weights on `[10,20,30]` give 20.

DSA/sparse MLA decode consumes supplied sparse indices. QSA prefill selects blocks then performs causal attention, excluding future positions, starting after projection. Kimi-K3 KDA decode updates convolution and recurrent state before gated normalization. Kimi megaop adds surrounding projections and stops before cross-device AllReduce. These are different [contracts](workloads.md); their results cannot be substituted for each other.
