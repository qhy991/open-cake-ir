# Complete Decoder backward starter plan

Related issue: #427. This task starts at reviewed experimental source `192a28b0`.
It preserves the original 33 tensor inputs, fixed epsilon and ten gradient outputs.
Raw reference code, input-view observations and generated workloads remain external.

## Progress

- [x] Trace the original high-level semantics and Workload/Program boundaries.
- [x] Check the stage and rounding design, including two accumulation alternatives.
- [ ] Implement and verify small numerical and boundary controls.
- [ ] Assess and lower all 16 original workloads; report declared memory.
- [ ] Reconcile the design with implementation and obtain independent review.

## Caller and ownership

The public interface will be `decoder_backward.source_for_workload(workload,
case_id='primary')`. It returns a complete CAKE Program source or refuses an
original ABI, scalar or view mismatch. A separate `source_for` accepts explicit
dimensions for small CPU controls; those controls do not replace original cases.

The existing Bench binder owns the original oracle, tolerances and physical input
views. Its `zero_copy_dense_axis_permutation_before_native_submission` contract
permutes five observed inputs by `[0,2,1,3]`. Their candidate shapes are physically
`[B,S,H,D]`, while the original reference sees `[B,H,S,D]`. The starter reads that
storage directly and does not copy or change the original inputs. All other inputs
keep their original identity views. The starter must check the complete view map.

The Compiler owns typed Programs, Schedules and lowering; this task changes none
of those owners. Public inputs unused by the original backward expression still
remain in the ABI. Their declarations can bind to the first stage without adding
fake arithmetic. Workload validation checks that all inputs remain unchanged.

## Stage and rounding plan

Candidate matrix products use FP32 tile accumulators and preserve each original
BF16 result boundary. This does not assert the target library's internal reduction
order. Fusing a stage is allowed only if its original casts remain explicit.

| Stage | Result and required rounding |
|---|---|
| SwiGLU backward | Down-projection gradient rounds BF16 before use. Gate gradient rounds BF16. The up gradient rounds after multiplying by gate and again after multiplying by the BF16 SiLU derivative. Sigmoid and derivative arithmetic are FP32. |
| Down weight gradient | Reduce across the full batch and sequence; one final BF16 output. |
| FFN input gradient | Two input-gradient products each round BF16, then their sum rounds BF16. |
| Gate and up weight gradients | Two complete batch/sequence contractions; BF16 outputs. |
| Post-attention normalization weight gradient | FP32 product with the supplied FP32 normalized states, reduced across batch and sequence; FP32 output. |
| Post-attention normalization input gradient | Use original residual, variance, weight and epsilon in FP32. Round normalization result BF16 before adding the residual gradient; round that sum BF16. |
| Attention output gradient and output weight gradient | Two matrix products with their original BF16 result boundaries. |
| Attention weight gradient | Matmul result rounds BF16 before the softmax backward expression. |
| Softmax backward | Weight/gradient products, row sum and scaling are FP32; only the final logits gradient rounds BF16. Preserve the supplied attention weights; do not introduce another mask. |
| Rotated query gradient | Full key contraction, BF16 output. |
| Grouped key and value gradients | Each repeated-head matmul rounds BF16 before the four-head group reduction. Accumulate the reduction in FP32 and round once to BF16. |
| Inverse RoPE for query and key | Follow the reference's exact sign and half-rotation expression. Round each BF16 product before their BF16 sum. |
| QKV input gradient | Three matrix products each round BF16; add Q and K then round, add V then round. |
| Q, K and V weight gradients | Three complete batch/sequence contractions with BF16 outputs. |
| Input normalization weight gradient | FP32 product and full batch/sequence reduction; FP32 output. |
| Input normalization input gradient | FP32 derivative, BF16 normalization result, then BF16 residual addition producing `grad_input`. |

This gives 22 ordered stages when the paired FFN and triple QKV input-gradient
paths preserve their rounding inside single kernels. Intermediate head gradients
use concrete flattened `[B,S,H*D]` storage where useful. Original inputs retain the
bound physical ABI.

## Accumulation alternatives

Two designs avoid a `[B,M,N]` temporary for each weight gradient:

1. A sequence-tile outer loop and scalar batch inner loop carry one MMA accumulator.
   This is compact, but its legality and actual accumulation must be proved with a
   two-batch CPU counterexample before use.
2. Unroll the fixed batch count at authoring time. Each batch has its own sequence
   contraction into an FP32 tile, then FP32 tiles are added before the sole BF16
   output cast. This needs more source but keeps global scratch independent of batch.

The nested candidate is refused because the batch loop has no contraction-axis
carry, and the result cannot escape both loops. The fixed-batch candidate has a
proven sequence contraction for every batch. Its remaining refusal comes from the
backend's load-provenance check counting address-index reads as data operands.
Issue #435 owns that separate shared repair. The candidate selects fixed-batch
FP32 tile accumulation, with execution proof pending that reviewed successor.

Per-batch BF16 partial gradients are rejected because they add a rounding boundary
absent from the original expression. Large per-batch matrix scratch is rejected
because the measurement cohort retains many output/scratch sets. No failure here
authorizes a Compiler change inside the starter task.

## Design reconciliation

An independent source reader checked the entire original high-level backward
expression, including unused caches, four-head grouping, supplied normalization
values and each BF16 seam. The initial conservative alternative kept more than
thirty separate stages. The selected 22-stage composition keeps those casts inside
the fused FFN and QKV paths, reducing retained global intermediates.

The attention-output gradient uses explicit `[B,S,H,D]` internal storage so its
contraction axis stays a direct loop index. Score and logits gradients use
`[B*H,S,S]` storage so all output axes have direct program ownership. These are
candidate intermediates; the original five input permutations remain zero-copy.

## Verification and remaining risks

Controls must distinguish both batches in every weight gradient, test GQA grouping
and inverse-RoPE signs, and fail if any required BF16 boundary is removed. A small
complete chain must check all ten outputs, with FP32 normalization-weight outputs.
The full original workload set must retain every tensor, shape, epsilon and view.
Declared memory includes public outputs and all Program intermediates; the current
paired lower bound is `2I + 18O + 18S` when both arms use this starter, before runtime,
oracle and spill allocation. The original three-hour task budget remains unchanged.

Transpose-plus-MMA native execution and the full original device comparator are
separate gates. No GPU/provider work or performance result is part of this software
unit. Independent source review of the final implementation remains required.

## Current software and memory scope

At `d65f9de0`, seven arithmetic/ABI contracts pass. At the same fixed source, all
16 original metadata records and the retained input-view observations produce
complete Programs. B=1 cases lower 19 of 22 stages; larger batches lower 12 of 22.
All remaining refusals are the indexed-MMA provenance restriction in #435. This
is not complete software or device acceptance.

The largest per-case declared allocations are 2,887,864,320 input bytes,
655,400,960 output bytes and 1,321,396,224 scratch bytes. The maximum paired tensor
lower bound is 41,052,659,712 bytes, about 38.23 GiB. No case exceeds a 64 GiB
device from this bound alone. This is not memory admission: oracle temporaries,
runtime allocations and private spill remain outside the estimate.
