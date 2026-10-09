# Grouped attention backward starter

Tracking issue: [#408](https://github.com/qhy991/open-cake-ir/issues/408).
Draft PR: [#409](https://github.com/qhy991/open-cake-ir/pull/409).

`gqa_backward.py` constructs a complete pure CAKE Program for original C550-Bench
`L1/001_attention_softmax_dropout_value_matmul_backward`. It requires the
implemented BOOL storage interface. Runtime, native-build and device
qualification remain separately owned.

## Original semantics and ABI

The inputs retain their original order and storage:

| Tensor | Type and shape |
|---|---|
| `grad_attn_output` | BF16 `[B,Q,80,128]` |
| `attn_weights` | BF16 `[B,80,Q,K]` |
| `attn_weights_dropped` | BF16 `[B,80,Q,K]` |
| `value_states` | BF16 `[B,8,K,128]` |
| `dropout_mask` | BOOL `[B,80,Q,K]` |

Outputs are BF16 `grad_attn_scores[B,80,Q,K]` and
`grad_value_states[B,8,K,128]`, in that order. Query head `h` reads KV head
`h // 10`; each KV gradient aggregates its ten consecutive query heads.

The original custom factory returns Python float `attention_dropout=0.1`.
`source_for_workload` requires its exact `fixed_scalar_inputs` binding. The
factory and final original-ABI wrapper must verify the scalar. The candidate
uses the frozen constant and receives no extra scalar tensor.

Both matmul gradients, dropout division, softmax row reduction and group partials
use FP32. BF16 casts occur only at the two final outputs. The value gradient uses
the supplied dropped weights. Stored BF16 probabilities are not renormalized;
their rounded row sum need not be exactly one. The mask applies before softmax
backward, so a dropped position can still have a nonzero score gradient through
the softmax row term.

## Three-stage Program

1. Compute FP32 softmax row sums, streaming keys in 32-element tiles. Its only
   intermediate is `row_sum[B,80,Q]`.
2. Recompute the score dot products, apply mask and dropout scaling, subtract the
   stored row term and write score gradients directly.
3. For each KV/key coordinate, compute each query-head gradient with a separate
   query loop, add the ten FP32 results and write the value gradient. No expanded
   value tensor, per-head gradient tensor, or full score-sized scratch is stored.

The source uses existing loads, casts, arithmetic, reductions, loops and Program
composition. It is a conservative SIMT baseline. Its reduction order can differ
from a vendor FP32 matmul; original device tolerances decide acceptance. There
is no native accuracy or speed claim from source lowering.

## Resource estimate

For the tightest original case `[B,Q,K]=[32,691,773]`, row scratch is
**6.748 MiB per argument set**. The current two-arm, sixteen-fresh-set lifetime
therefore adds about **121.465 MiB** of declared scratch to the public tensor
lower bound.

For two copies of this starter, the model is `2I + 18O + 18S`, where I, O and S
are input, output and scratch bytes per argument set. Its largest value is
**60.487 GiB**, leaving **3.513 GiB** before accounting for context, compiler
private memory, allocator state and other costs. This is not measured free
memory or a device admission. A real memory gate remains necessary.

## Interfaces and evidence

- `source_for(B,Q,K, attention_dropout=0.1)` returns static CAKE Python source.
- `program_for(B,Q,K)` returns the complete Program.
- `source_for_workload(workload, case_id)` checks the exact seven-tensor ABI and
  custom-factory scalar binding.

At fixed software commit `92f74ab3`, all sixteen original shapes construct and
all 48 stages pass assessment and Triton source lowering. Four CPU contracts
pass without skips. Actual emitted arithmetic is compared with independent
small FP32 contractions, including BF16 probability sums, unchanged inputs,
masked score coupling, wrong dropout scaling and incorrect group mappings.
The smaller numerical controls use the same generator with fewer heads and
dimensions; full original geometry is covered separately by software lowering.

Generated sources, original workload identities and private reference material
remain outside Git. Native compilation, original 16-workload device correctness,
memory admission and qualified Program measurement are still required before
starting optimization with this baseline. No Compiler change or optimization
pass is promoted from this source-construction task.
