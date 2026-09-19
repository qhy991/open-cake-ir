# ADR 0074: FlashInfer starters compose exact spans and preserve FP16 GEMM

Status: accepted implementation scope; device qualification pending.

## Context

The owner requested all 26 FlashInfer pack tasks and B300 testing. Nine RMSNorm
captures were registered, but three were refused because a whole row was not a
power-of-two Triton span. Eight FP16 GEMMs lacked a Workload ABI path even though
Compiler casts, loads, stores and reductions already admitted FP16 operands.

## Decision

A normalization starter partitions the original row into disjoint static slices,
reduces all slice sums into one whole-row mean, then writes a tiled second pass.
One store operation owns the output. Nothing pads input shapes, changes epsilon,
relaxes the backend span check or permits multiple writers.

GEMM retains A[M,K] and B[N,K], FP16 inputs/output and FP32 intermediates. One CTA
computes one output using existing reductions, with exact K slices where needed.
This is a correctness starter, not a tensor-core performance claim. N and K remain
upstream constants; M is one declared upstream batch. Inputs are bounded to avoid
FP16 output overflow; this scoped input distribution is stated in the contract.
The Workload ABI and task byte accounting admit FP16 already supported by common
CUDA Evaluation. Numerical rounding is IEEE binary16, not BF16 substitution.

A complete pack inventory reports unimplemented tasks explicitly; an inventory
entry is not an admitted Workload. GQA, MLA and FP8 MoE require further integration.

## Gates and scope

No IR, Target, instruction contract or lowering changes: existing static slices,
reductions and casts retain their typing, analyses and emitted form (P1–P8).
Corpus expectations are unchanged. Tests cover exact shapes, transposed GEMM
storage, FP16 rounding, all-element comparison and B300 source generation.
Device compilation, correctness, CUPTI, profiler and framework evaluation remain
separate required gates. Reports record missing connectivity rather than GPU success.
