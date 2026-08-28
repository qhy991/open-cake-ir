# ADR 0037: A prefix scan is not a fold with a flag

Status: accepted for `open-cake-ir-sm100a-v29`, 2026-08-27.

## Outcome and non-goals

`SCAN` is one operation kind carrying an operator and a direction. It keeps the axis it
walks. It is not a mode of `REDUCE`, it is not composable from the elementwise
vocabulary, and it lands with the one operator a backend has a body for.

This does not add a scan across tile-loop iterations, an exclusive-prefix mode, a `max`
scan, or any claim that KDA's gate stage is now expressible. It adds the prefix and the
prefix only.

## Why it is a separate kind

A fold consumes an axis and a scan keeps it, so their results have different rank. Every
rule the IR states about a reduction is false of a scan: the collapsed-shape invariant,
the loop-carried accumulator, the identity a backend materialises before the loop. A
`reduce` with an `inclusive` flag would be one kind whose written shape depends on a
boolean, which is the shape of declaration this IR exists to refuse.

Nor is it a composition. The admitted elementwise operators — `square`, `rsqrt`, `exp`,
`tanh`, `add`, `sub`, `mul`, `div` — are each pointwise, and no arrangement of pointwise
operations produces a running prefix. A `TileLoop` orders operations over tiles but has no
general carried-value relation; the accumulator semantics derived for reductions and MMA
do not turn pointwise operations into a prefix recurrence.

## What the vocabulary admits

`ScanOp` has one member, `sum`, rather than reusing `ReduceOp` and refusing `max` at
emission. A word the Schedule may say and the Compiler cannot answer is the gap
`tools/ir_vocabulary.py` exists to expose; adding one deliberately would be the defect
that tool reports.

`ScanDirection` has both members because both reach the backend: `tl.cumsum(..., reverse=)`
is the same call, and the reverse prefix is what a backward pass needs. Deriving the
direction from context would leave the Schedule silent about a fact that changes every
output.

There is no `inclusive` parameter. An exclusive prefix is this operation followed by an
elementwise `sub` against the same input — a composition the vocabulary already covers,
and a second spelling of one is what this ADR declines to add.

## Placement

`SCAN` is in the Triton backend's outside-loop emitters and deliberately not the
inside-loop ones. A prefix scan inside a tile loop would need every earlier tile's
running total carried in; emitting a per-tile scan that silently restarts would be the
wrong answer rather than a refused one, so the position is refused.

## Acceptance evidence

`corpus/schedules/chunk-cumsum-b8-smoke.json` lowers to
`tl.cumsum(gate_tile.to(tl.float32), axis=0, reverse=False)` and, on a brokered B200,
compiled, launched once and matched the independent oracle on all 65,536 outputs at a
maximum deviation of `0.0` under the unchanged `1e-5` tolerance
(`open-cake-ir-sm100a-v29-draft`, schedule `34b747aa`, source `4f2760c7`).

Its drift sibling declares the result with the scanned axis dropped — the shape a fold
would produce — and is refused before lowering with `SCAN_SHAPE_MISMATCH`. That case is
the one this primitive most needs, because writing a scan as a reduction is the mistake
the two kinds' similarity invites, and nothing else in the gate would catch it.

## What this does not reach

The target was KDA's gate stage, `fla/ops/kda/gate.py::kda_gate_chunk_cumsum_vector_kernel`.
That kernel is

```python
b_A = tl.load(A_log + i_h)
b_gate = lower_bound * tl.sigmoid(exp(b_A) * b_s)    # the branch KDA forward measures
b_o = tl.cumsum(b_gate, axis=0)                      # chunk local
```

The scan is now the third of three parts. The gate arithmetic needs a per-head runtime
scalar operand, which this vocabulary still lacks — and note that the measured branch
uses `sigmoid`, which `swiglu-b8-smoke` already spells as `0.5 * (tanh(0.5x) + 1)`, so it
needs no `log` and no `softplus`. Those belong to the unbounded branch, which the KDA
forward benchmark does not take. Until the scalar operand lands, this Schedule takes the
gate values as input and proves the prefix; calling it the gate stage would be false.
