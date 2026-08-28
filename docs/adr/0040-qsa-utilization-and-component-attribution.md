# ADR 0040: QSA utilization is a whole-Program roofline claim

## Status

Accepted for the user-authorized continuing artifact-optimization loop. It changes no
Workload semantics, Candidate acceptance threshold, Compiler Revision, scientific Study,
checkpoint claim, or serving claim.

## Irreducible goal

Drive the fixed `target_t32768` QSA Program toward at least 70% MFU or BWU on one B200,
using Cake IR for authored candidates and on-device evidence for every decision. The
endpoint is the complete five-node Program from post-projection inputs to BF16 output,
not a selected kernel counter or the hidden direct-CUDA baseline.

## Fixed contract

- **Workload:** `contracts/workloads/qsa-prefill-t32768-v1.json`, case
  `target_t32768`, batch one, task-declared geometry, seeded inputs.
- **Program:** pool -> layernorm -> score/top-k -> expand -> selected causal GQA.
- **Correctness:** the existing external FP32 oracle and elementwise tolerance gate;
  intermediate diagnostics never replace final output correctness.
- **Primary timing:** the existing balanced candidate/baseline CUPTI cohorts, cold L2,
  no CUDA Graph, no profiler, with the task-owned 5% noise/materiality boundary.
- **Hardware:** one broker-owned NVIDIA B200, `sm_100a`, exclusive for timing/profile.
- **Baseline:** last valid tile256 Candidate; a structurally new Candidate must improve
  the complete Program beyond the unchanged materiality boundary to replace it.
- **Non-goals:** checkpoint reproduction, Qwen's reported 10.2x, serving TTFT/tok/s,
  multi-shape dispatch, or a general B200 utilization claim.

## Peak and utilization authorities

A target peak is measured, never copied from a product specification:

- memory bandwidth is the existing far-larger-than-L2 FP32 triad cohort;
- BF16 contraction peak is a vendor-library BF16 matmul;
- IEEE FP32 contraction peak is a vendor-library FP32 matmul with TF32 disabled;
- every cohort is exclusive, retains raw samples, and must satisfy the existing 5% CV
  gate before its median becomes a rate.

For Schedule node `i`, declared work supplies FLOPs, compulsory bytes, their error
directions, and the contraction contract. A dependency-ordered CUDA-event trace supplies
diagnostic node time without synchronizing between nodes. It runs only after the primary
timing cohorts and is never a frontier or promotion metric.

The complete sequential Program arithmetic floor is:

`sum_i(FLOPs_i / matching_peak_i)`

over nodes with a measured matching instruction rate. Program MFU is that floor divided
by the primary no-profiler Candidate time. It retains the work model's lower-bound/exact
status. Program BWU is `Program_bytes / (bandwidth_peak * Program_time)` only when the
byte count is exact; otherwise it remains an upper bound and cannot by itself prove the
70% endpoint. A later measured-traffic assay may supply a stronger BWU authority.

The goal is achieved only when either a sound Program MFU lower bound or an exact/measured
Program BWU is at least 0.70, final correctness passes, primary timing is stable, and a
fresh confirmation reproduces the result. NCU throughput percentages alone never satisfy
the goal.

## Component attribution

The evaluator may expose one optional diagnostic mode that records, for both Cake and
the fixed direct-CUDA reference:

- dependency-ordered per-kernel CUDA-event samples;
- whole-Program event samples from the same trace;
- medians, CVs, and each node's fraction of summed node medians;
- exact kernel ids and launch order.

It must not add device synchronizations between Program nodes, mutate the primary CUPTI
samples, or silently enable itself for ordinary search runs. Raw component samples are an
artifact; the run result carries only the bounded summary.

## Iteration and stop rules

1. Measure peak and component shares before selecting the next mechanism.
2. Prefer removing work, intermediate bytes, or launch/materialization boundaries over
   instruction tuning.
3. Use existing IR composition first. Add an IR primitive only when the required state,
   ownership, or operation cannot be expressed without an opaque QSA operation.
4. Change one live mechanism at a time. Three repetitions of the same compile/resource
   rejection close that branch unless a new mechanism changes the expected resource.
5. A correct improvement below 5% is retained as null evidence and does not replace the
   frontier. A profiler-only improvement does not promote.
6. Every terminal candidate, including invalid and null outcomes, remains append-only.

## Evidence boundary

SubCUDA cases are mechanism evidence, not performance transfer: R37 motivates eliminating
request-local attention launches while preserving one state owner; R2-R4 prove that local
attention speedups can fail the real semantic gate; Day3 proves that component fractions
and Amdahl arithmetic choose experiments but do not establish wall-time gains.
