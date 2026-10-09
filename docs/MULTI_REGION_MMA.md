# Multi-region Triton MMA

The bounded multi-region route accepts sequential sibling loops or an existing
outer/inner pair. An MMA operand is a resident rank-two register value with one
producer. Its supported provenance is a global load followed by zero or more typed
casts/transposes, or a computed value defined outside every active MMA loop followed
by those same representation changes. A completed earlier sibling contraction or
reduction can supply the invariant value; an ordinary per-iteration value cannot.

`Schedule.staged_operand_provenance` resolves unique producer chains and typed axis
changes. Both contraction-axis analysis and Triton admission consume that query.
The backend checks availability and completion at each chain edge. Common typing,
SSA, read-before-write, loop escape, role, access-mask and input-effect checks still
own program legality. Arithmetic on a varying in-loop load is not supported by this
slice: it has no proven contraction-axis mapping. Computed values inside an active
ancestor loop are also outside the bounded invariant domain.

The retained attention shape has two sibling regions. QK walks the contraction K
axis, so emission initializes scores before that loop and adds every K tile.
Softmax computes weights after QK completes. AV walks output columns: its invariant
weights and loaded/transposed V tile produce a fresh chunk each iteration, which is
stored with that output coordinate. No AV accumulator is initialized or reused
across output columns. The existing emitter dispatch preserves authored casts,
transpose, masks, and the exact instruction precision; it does not move, duplicate,
or recompute invariant arithmetic.

CPU tests execute emitted source against independent unchunked attention math with
non-square row, contraction and output tails, and a nested cast/transpose GEMM with
inner K carry and per-row reset. They observe each output written once and unchanged
inputs. Negative cases cover malformed/ambiguous alias chains, varying arithmetic,
unfinalized values, output-axis escape, read-before-write, input writes, ancestor
carry and missing output ownership. The CPU interpreter is a control-flow/memory
semantic fixture, not FP32/FP16/BF16 hardware-rounding validation.

This resolves a lowering capability gap, not a measured speedup. Exact-target device
qualification retains the original workload inputs, oracle, tolerance and timing
boundary. In particular, an emitted `tl.trans` and the dot's orientation conversion
do not establish two physical layout exchanges or any LDS reduction.
