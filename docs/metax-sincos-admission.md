# C550 sin/cos library admission

Tracking issue: #454. Related evidence: #401 and #445. This task changes only the
exact `xcore1002` declaration for `maca.sin.f32` and `maca.cos.f32`, plus its
verification and evidence records. TF32 admission is unchanged.

## Evidence supporting the declaration

The independent component probe at public source `379f8372` executed both named
library calls on MetaX C550 using the accepted MACA 3.5.3 / Triton 3.1.0 stack.
Job `maca-facf96c7db49` completed 20 native calls and 20,480 output checks with
zero mismatches, unchanged input bits and closed modules. The five input patterns
were each checked with two output poisons for each operation; these counts do not
represent independent random samples. The maximum observed finite absolute error
was `7.796446133134793e-08` against Python math on the decoded FP32 input.

The retained contract samples `[-65536,65536]`. Finite outputs must satisfy
`abs(error) <= 2e-6 + 2e-6*abs(reference)`. Exact zero inputs require sine to
preserve the sign bit and cosine to return exactly positive one. NaN and either
infinity require a NaN result, without a payload guarantee. Small nonzero values
use the finite tolerance, which does not prove subnormal preservation.

The bounded RNE10 RoPE successor at public source `da9390fc` then passed the
original Bench `ababa4c0`, all 16 Workloads and all ten rounds each, under job
`maca-c742da8d2a88`. All 160 native calls retained the input bits and closed their
modules. The original comparator reported zero absolute and relative error;
this is not a new bitwise claim. Its original HIGH numerical policy and input
factory stayed unchanged. The earlier IEEE-angle and TF32 failures remain
failed records. The bounded frequency conversion is a task-specific expression,
not a general TF32 implementation.

The component and full-task builds used the same accepted canonical host record,
runtime stack and relevant sin/cos emission. Their fixed private source bindings,
sealed artifacts, original comparison records and cleanup receipts remain in the
private qualification archive. Public source and tool reviews are in PR #430
and PR #447. No host capture or byte-identity inventory was regenerated for this
admission review.

## What the declaration means

The contracts name MACA FP32 software-library functions through the MCFATBIN
Triton route. They do not promise a native trigonometric ISA instruction,
correct rounding for every FP32 value, a universal ULP/error bound, NaN payloads,
subnormal preservation, speed, or automatic qualification of another SDK.
Task acceptance still requires its external device oracle. The static checks do
not prove that arbitrary runtime values lie inside the sampled interval.

## P1–P8 and verification

P1/P3: existing syntax and one registered contract per operation remain.
P2/P8: the explicit library choice and bounded device evidence remain visible.
P4/P5/P7: existing FP32 typing, effects, data analyses and source admission remain;
there is no new operation, layout model or analysis rule.
P6: real-Target source controls, closed-Target counterexamples, declared-contract
snapshots and the existing Corpus Gate run at a fixed successor commit.

The Corpus has no C550 sin/cos case. Its unreached-contract report must include
the two new declarations; no case is added to improve a count. No Corpus
expectation is refreshed. Software gate results and independent review will be
recorded after execution. This task runs no GPU or provider work and does not
merge the inherited experiment stack into a maintained branch.
