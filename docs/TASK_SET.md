# The twenty-four-task Lab set, and why these twenty-four

Every task here is one frozen Workload Contract: a definition, an explicit tensor ABI,
five required input distributions, an independent oracle, and a predeclared tolerance.
Each one is portable — the same contract freezes for an Apple Metal device or an NVIDIA
Triton device, and `src/open_cake_ir/tasks/devices.py` owns the whole difference.

## What the set is selected for

A task earns a place by asking a Schedule something the other twenty-three do not. It does
**not** earn one by being a different operator name: five GELU vectorization variants in
the AKA corpus are five *Schedules* of one Workload, and the Lab already searches over
vectorization, so admitting all five would grow the count without growing the question.

The selection axis is therefore **structure**, and there are five structures the current
IR can express on both routes. The families are named after them.

| Family | Structure | Count |
|---|---|---|
| `activation` | Element-local. Each output reads one input coordinate. No reduction at all. | 8 |
| `rowwise` | Reduce along the row, then an epilogue that **multiplies** by the result. | 5 |
| `reductions` | One program per *feature*, folding **down the rows**, rank-1 outputs. | 4 |
| `optimizers` | Element-local, but writing several buffers and carrying accumulator state. | 3 |
| `contraction` | **Contract an axis**, so one loaded operand feeds many outputs. | 4 |

There is a second axis, and it separates the last family from the other four: **whether a
Schedule has any arithmetic to trade for traffic.** Measured with the Compiler's own work
model at the shapes they freeze, the twenty tasks in the first four families span **0.20 to
1.50** counted FLOPs per compulsory byte. Every one of them reads its inputs, does a few
operations per element, and writes its outputs; no Schedule for any of them can be
arithmetic-bound, so none of them can pose a question about keeping an accumulator
resident, tiling a contraction, or trading recomputation against bandwidth.

The `contraction` family reaches **20 to 31** FLOPs per byte at its frozen shape — an order
of magnitude past the entire rest of the set. `test_contraction_tasks.py` measures both
sides rather than asserting them, so the separation cannot quietly rot.

Within a family, each task is kept only if it stresses a different primitive, a different
tensor rank, a different reduction operator, or a different numerical hazard.

## The twenty-four

### `activation` — element-local, no reduction (8)

| Task | AKA parent | Why it is here |
|---|---|---|
| `silu` | — | The exp+reciprocal sigmoid gate, the family's baseline. |
| `swiglu` | — | Two same-shape operands: a gate, not a pointwise map. |
| `gelu_tanh` | — | The **only** task naming an instruction contract, and the reason `metal.precise.tanh.f32` exists. |
| `gelu_tanh_backward` | `gelu_tanh_backward_f32_vec4_direct_v1` | Reuses the tangent for `sech²` instead of forming a `cosh`; its allowance is derived from that cancellation. |
| `softsign` | `softsign_contiguous_fp32_int32_block256_v1` | **No transcendental at all.** Absolute value as `relu(x) + relu(-x)`. |
| `selu` | `selu_contiguous_fp32_int32_block256_v1` | A piecewise function with **no conditional**: the positive arm cancels exactly. |
| `prelu` | `prelu_nchw_per_channel_contiguous_fp32_i32_block256_v1` | The family's only **per-feature** operand broadcast along the row. |
| `softplus_gradient` | `softplus_gradient_fp32_contiguous_i32_gridstride_v1` | Sterbenz-exact `1 - exp(-y)` preserves a near-one ULP that `|dy|` then scales; it owns a wider allowance for that reason alone. |

Deliberately **not** included: `sigmoid` (it is `silu`'s gate with the multiply removed),
`elu` (SELU at unit outer scale — the same composition, a different constant),
`sigmoid_backward` and `rsqrt_gradient` (a two-input elementwise shape three other tasks
already carry).

### `rowwise` — reduce along the row, multiply by the result (5)

The distinction from RMSNorm and LayerNorm forward, which already exist in this Lab: those
reduce a *square*, so every term shares a sign and the epilogue then **divides** by that
same sum, normalizing its error away. These reduce signed terms and **multiply**, so the
reduction's error is amplified. That is why every task here derives its allowance from the
frozen row width instead of naming a constant.

| Task | AKA parent | Why it is here |
|---|---|---|
| `softmax_backward` | `softmax_backward_contiguous_fp32_int32_v1` | One plain sum; the product is formed in the epilogue. |
| `rmsnorm_input_gradient` | `row_rmsnorm_dx_contiguous_f32_i32_b128_v1` | One sum over a **triple** product. |
| `layernorm_backward_input` | `layer_norm_backward_input_contiguous_fp32_i32_block256_v1` | Two means, and **three tensor ranks in one ABI** (`[R,C]`, `[R]`, `[C]`). |
| `absmax_rescale` | `grouped_absmax_e4m3_fp32_i32_contiguous_16x16_v1` | The **only maximum** in the set. A fold that selects rather than accumulates contributes no rounding, and its allowance is flat where every other one grows. |
| `cosine_similarity` | `rowwise_cosine_similarity_contiguous_fp32_i32_block256_v1` | Three reductions collapsed into **one value per row** — the only rank-1 output from a row program. |

### `reductions` — one program per feature, folding down the rows (4)

This inverts which extent is parallel and which is sequential. A Schedule that is good at
folding a row is not automatically good at folding a column, and no other family can ask
that. The allowance here grows with the **row** extent, because the output *is* the
reduction and nothing divides its error away.

| Task | AKA parent | Why it is here |
|---|---|---|
| `layernorm_gamma_beta_backward` | `layernorm_gamma_beta_backward_contiguous_fp32_i32_block256_v1` | Two column reductions, two rank-1 outputs. |
| `bias_gradient_reduction` | `matmul_backward_bias_accumulate_contiguous_fp32_int32_block256_v1` | Accumulates into a prior gradient rather than overwriting. |
| `per_channel_moments` | `nhwc_raw_moments_contiguous_f32_i32_block256_v1` | Two reductions of the same tile; the raw second moment, not a centered variance. |
| `channel_absmax_scale` | `grouped_absmax_e4m3_fp32_i32_contiguous_16x16_v1` | A **maximum down the column** — the same operator as `absmax_rescale` on the other axis, which is exactly the comparison worth having. |

### `optimizers` — element-local, multi-output, carried state (3)

Element-local like `activation`, but two properties exist nowhere else: several output
buffers per element, so the Schedule owns a real store schedule; and inputs declared
**non-negative** because a reciprocal square root reads them.

| Task | AKA parent | Why it is here |
|---|---|---|
| `momentum_sgd` | `momentum_sgd_update_contiguous_fp32_i32_block256_v1` | Two outputs, exact arithmetic, no amplification — the family's baseline. |
| `adamw` | `adamw_contiguous_fp32_int32_block256_v1` | Three outputs; the epsilon sits **inside** the square root, and the allowance is derived from the resulting reciprocal. |
| `adadelta` | `derived_adadelta_update_contiguous_f32_i32_b256_v1` | Two accumulators, and a square root the vocabulary lacks, taken as `z * rsqrt(z)` with the epsilon keeping `z` strictly positive. |

### `contraction` — arithmetic-bound, one operand feeding many outputs (4)

The FLOP count grows as the product of three extents while the traffic grows as their sum.
Each baseline deliberately keeps the whole contracted operand register-resident: that is
the plain reading of the definition, and it is *why* the Metal route refuses larger shapes,
since its 1024 live FP32 values per lane is exactly the budget a tiled contraction would
free. Finding that trade is the search this family exists to pose.

| Task | AKA parent | Why it is here |
|---|---|---|
| `gemm` | `dd_matmul_bias_f32_bt_oc_c_block16_v1` | The canonical contraction; the shape every other one is measured against. |
| `gemm_silu` | `dd_matmul_bias_f32_bt_oc_c_block16_v1` | The same contraction with a **transcendental epilogue reading the accumulator** before it leaves the kernel — a fusion question the plain GEMM cannot ask. |
| `pairwise_sqdist` | `condensed_pairwise_l2_f32_i32_block256_v1` | Contracts a **rounded difference**, not a product, so no fused multiply-accumulate instruction can serve it. Highest intensity in the set. |
| `attention_decode` | `single_decode_attention_nhd_fp32_i32_hd128_v1` | **Two chained contractions with a data-dependent softmax between them** — four reductions in one Schedule, and the only task whose second contraction depends on the first's values. |

## What is deliberately excluded, and why

These were candidates and were dropped for a stated reason, not by oversight.

- **`prelu_input_gradient`** needs a true `indicator(x > 0)`. Unlike PReLU forward, the
  gradient has no value that supplies its own switch, so it needs a compare/select the IR
  does not have. It is the largest gap cluster in the AKA review (245 records).
- **Anything with a `log`** (softplus forward, cross-entropy, log-softmax) — 43 records.
- **Scan, sort, top-k, atomics, gather-by-index, dtype casts** — none is an admitted
  operation kind on these Targets.
- **`batch_norm_statistics`** as recorded uses Welford with a conditional on `M2 == 0`.
- The **FP8 cast** that consumes `channel_absmax_scale`'s output: no backend here can name
  the dtype. The scale is computed; the cast is stated as outside the contract.

## Provenance and what does not transfer

Twenty-one of the twenty-four carry an `aka_qualified_parent_lineage` provenance entry
naming the exact parent they were migrated from, and a test resolves each one against the
dataset rather than trusting the string. That entry records the **semantic** origin only. Each
Workload is its own frozen instance: shape, input distributions and tolerance are declared
here, and no recorded B200 correctness or performance result transfers to it on any device.

## Portability

All twenty-four lower on all four backends (`metal-m1-pro`, `metal-m2`, `triton-b200`,
`triton-b300`) — 96 combinations. Only three things differ between an Apple and an NVIDIA
freeze of the same task:

1. the `target=` / `backend=` the Schedule declares;
2. the `tanh` instruction contract, which `gelu_tanh` and `gelu_tanh_backward` name and
   which neither Target admits in the other's spelling;
3. the row width — Triton's `tl.arange` requires a power-of-two span, so a width the route
   cannot tile is refused at task creation rather than frozen into a Workload that has no
   Schedule.

`coalesced=False` and the row-per-program map are admitted by both, so the operation body
itself is byte-identical across routes.
