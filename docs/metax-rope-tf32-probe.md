# Original RoPE angle TF32 probe

Related issue: #440. Related math work: #401.

The question is whether the existing explicit `triton.dot.fp32_tf32` route matches
the original benchmark's angle matmul under its retained BLAS HIGH context.
This does not repeat or replace the historical failed RNE10 hypothesis. The
contract names FP32 operands, FP32 accumulation and Triton's `tf32` option; it
does not promise a particular ten-bit rounding rule.

## Bounded comparison

Use the original B1/S2048, B16/S256 and B2/S131 workloads, resolved to their UUIDs
from the fixed benchmark metadata. The original factory provides INT64 positions
and its already-scaled FP32 frequencies. Keep the environment override, HIGH
precision and `allow_tf32` enabled exactly as observed; refuse a changed context
instead of resetting it.

The candidate uses zero-copy singleton views `[B,S,1]` and `[64,1]`. Masked loads
place the original value only in K lane zero of a physical K32 tile. All other
lanes are zero. Positions convert to FP32 before the dot. Frequencies remain the
left operand and positions the right operand, matching the original matmul order.
Transpose its result to `[B,S,64]`. The otherwise identical IEEE route is a control.
Neither route recomputes frequencies or uses an FP16 approximation.

Build uses the existing isolated compiler and sealed stage builder. Before
allocation, evaluation must validate the fixed source, original case, instruction,
emitted source, artifact and launch binding. Device comparison uses the existing
physical/runtime/PCI lease and native loader. Retain raw input and angle bits,
the original BLAS result, both route comparisons, input checks and module cleanup.
No timer, provider, search or full RoPE acceptance is involved.

Bit equality is an observation of this question, not a new tolerance for the
original benchmark. IEEE disagreement can be an expected control outcome.
Numerical observations from both predeclared routes remain visible. An execution,
identity, input-effect or cleanup failure stops further device work.

## Source and device boundaries

Production Target admission stays closed throughout the probe. Test-local Target
construction permits CPU assessment and emission only. Native compilation and a
successful device comparison are separate evidence, and a reviewed successor is
required for any future production admission.

P1–P8 scope: reuse existing Schedule, AccessMap, cast, singleton-view and MMA forms;
keep K padding and precision explicit; leave typing, effects and analyses with
their current owners. This tool adds no IR, layout algebra or hardware fact.
CPU contracts must include wrong-source, swapped-precision, workload, artifact
and launch counterexamples before independent review and any device execution.

This task inherits the experimental source at `3cc84f3b`. Its draft PR targets
`metax`, but it does not authorize merging unreviewed inherited work into a
maintained branch or `main`.
