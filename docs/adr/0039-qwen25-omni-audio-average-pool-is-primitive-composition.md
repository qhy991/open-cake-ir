# ADR 0039: Qwen2.5-Omni audio average pool is primitive composition

Status: proposed for Compiler v29 review.

## Context

The Qwen2.5-Omni audio encoder in ApxInf commit
`32ba4d47b961b495dd878ccb99dfe6bd21d7970f` ends with a BF16
`avg_pool1d(kernel=2, stride=2)` over `[frames, 1280]`.  Its CUDA implementation loads
two BF16 frames, converts both values to FP32, accumulates in tap order, divides by two,
and converts the result once to BF16.

The current Schedule language already expresses that program as a global BF16 load of a
two-frame tile, an FP32 `reduce(sum)` over the frame axis, an FP32 divide by `2.0`, and
a BF16 store.  The Apple GPU family 9 Target admits every one of those operation kinds.
The gap is therefore in the finite Metal Adapter, not in the IR vocabulary or Target.

## Decision

Admit a third finite Metal primitive graph for `avg_pool1d(kernel=2, stride=2)`.  The
graph is recognized from Buffer types and shape relations, Operations, dependencies,
parameters, ProgramMap, and AccessMaps; neither the entry point nor a workload name
selects it.  It accepts positive feature widths and an input frame extent exactly twice
the output frame extent.  Consequently this first slice covers only even input-frame
extents.  It does not silently drop or pad an odd final frame.

One 32-wide SIMDgroup owns one output frame.  Its lanes cover features in steps of 32,
load the two BF16 taps, add them in FP32, divide by the exact binary32 value `2.0`, and
round once when storing BF16.  The lowering exposes two global buffers, zero threadgroup
memory, one threadgroup per output frame, and the existing safe Metal floating-point
flags.

The successor Corpus carries a positive graph and a scalar-drift falsifier.  The local
Apple probe compiles, links, and dispatches the generated MSL against an independent
bit-level BF16 oracle with both halfway parities, cancellation, and negative inputs.

## P1-P8 check

- **P1 Ergonomic / P3 Canonical:** no new syntax or operator name is added; the Schedule
  uses the existing load, reduction, elementwise, and store spelling.
- **P2 Performance-transparent / P5 Analysis-friendly:** the two-frame tile, one
  SIMDgroup role, FP32 intermediates, ProgramMap, and launch geometry remain explicit.
- **P4 Statically type-checked / P7 Analysis-consistent:** existing verifier rules check
  the BF16-to-FP32 reduction, scalar arithmetic, store conversion, and access shapes; no
  analysis model changes.
- **P6 Test-gated:** positive and falsifying Corpus cases, Adapter contracts, source
  compilation, and an Apple9 bit-exact dispatch probe gate the change.
- **P8 Hardware-grounded:** the emitted MSL and Apple9 probe witness the declared
  32-wide SIMDgroup execution and conversion behavior.

## Non-goals and consequences

This does not add general pooling syntax, arbitrary kernels or strides, padded pooling,
odd-frame tail handling, `im2col1d`, a Metal Evaluation driver, performance Calibration,
or scientific Evidence.  Because the upstream audio convolution can produce an odd frame
extent, this decision is a constrained operator subdomain rather than a claim that every
ApxInf audio input is replaceable.  The probe remains engineering verification.  Formal
Compiler release still requires the full Corpus Gate and an approval written outside the
implementation automation.
