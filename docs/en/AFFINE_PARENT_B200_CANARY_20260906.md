# Affine computation: one complete B200 correctness check

[中文原文](../AFFINE_PARENT_B200_CANARY_20260906.md) · [English home](README.md)

This fixed sample passed: eight outputs match bitwise, sixteen input values remain unchanged, and both output guards and the return interface pass. It is a new Compiler-v43 GPU correctness observation, with no timing or memory sanitizer.

## The computation

Two batches each have two channels and two spatial values. Each batch/channel pair has its own scale and bias:

`Y[n,c,s]=fma(X[n,c,s],scale[n,c],bias[n,c])`.

FMA rounds once. Using only the channel to select coefficients would get four answers wrong, which is why this sample is useful.

| Batch | Channel | Position | x | Scale | Bias | CPU/GPU answer |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | -3.65625 | -0.75 | -0.28125 | 2.4609375 |
| 0 | 0 | 1 | -2.5 | -0.75 | -0.28125 | 1.59375 |
| 0 | 1 | 0 | -1.34375 | 0.0625 | 0.25 | 0.166015625 |
| 0 | 1 | 1 | -0.1875 | 0.0625 | 0.25 | 0.23828125 |
| 1 | 0 | 0 | 0.96875 | 0.875 | -0.125 | 0.72265625 |
| 1 | 0 | 1 | 2.125 | 0.875 | -0.125 | 1.734375 |
| 1 | 1 | 0 | 3.28125 | -0.25 | 0.40625 | -0.4140625 |
| 1 | 1 | 1 | -3.59375 | -0.25 | 0.40625 | 1.3046875 |

## What was checked

Inputs follow the original parent formula at N=C=S=2. The selected Schedule comes unchanged from the committed twelve-parent re-audit and was lowered by released v43, without editing GPU source. The output starts as NaN and has guards on both ends. An independent triple-loop exact-fraction oracle checks every value, input preservation, guards, and interface. Collected bit patterns were recomputed locally with zero output mismatches and zero input changes.

These particular fraction results are exactly representable in FP32; this is not a universal floating-point FMA oracle. Two guards are not memcheck/racecheck. Local recomputation checks values and arrays; pointer/interface facts remain runtime judge observations.

The original AKA parent is l000075 from commit 387aa7faf521a0b72c994ff15a7638cd7e6a8583, row batch-sensitive-n2-c2-s2. Compiler/Schedule source is 26fbf8f3f2bc691707f82d0e79e7327386861feb; the task source is ed35501a982a74d45dad26f6ec1851083cdb2073. GPU Infra 0.17.0 used B200 GPU 0, shared broker job gpuq-9fd4ca406482. The unique run ended completed/valid, frontier_eligible=false because there was no timing stage. No retry or reroute occurred; the job released and shared services remained running.

## Retained evidence and scope

See [complete arrays](../data/affine-parent-v43-b200-canary-20260906/complete-output.json), [node result](../data/affine-parent-v43-b200-canary-20260906/run-result.json), [stage receipt](../data/affine-parent-v43-b200-canary-20260906/stage-receipt.json), [independent recomputation](../data/affine-parent-v43-b200-canary-20260906/verification.json), and [task notes](../../examples/gpu/affine_parent_canary/README.md). These are portable copies, not new owners of node lifecycle. Full route, snapshots, and logs remain at the external location in the Chinese source.

One fixed sample of one v43-lowered candidate passed on B200. It does not qualify all twelve parents, other shapes, dynamic host ABI, stream/state-return protocols, performance, or serving. The old v41 re-audit still says GPU not run; this is a separate observation.
