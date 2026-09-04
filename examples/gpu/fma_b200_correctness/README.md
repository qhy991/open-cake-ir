# Released v41 FMA B200 correctness

One immutable task evaluates the two unedited Schedules from released Compiler v41:
ordinary FP32 FMA and nested FMA with a separately rounded product. Each uses fixed
shape [8,128], three disjoint read-only FP32 inputs, and a supplied FP32 output whose
pointer and shape must be preserved. This is not a complete AKA parent or a timing task.

Six fixed input families cover cancellation, signed zero, subnormals, overflow,
Inf/quiet-NaN/signaling-NaN combinations, and seeded arbitrary binary32 bits.
The oracle implements infinite-precision integer product/add followed by binary32
round-to-nearest-even. The nested oracle performs an independently rounded multiply,
the inner FMA, then the outer FMA in exactly that order. Non-NaN outputs match bitwise;
NaN outputs must be quiet but their sign/payload is unspecified. Inputs remain bitwise
unchanged. Output sentinels always differ from the expected result class.

The evaluator is independent of generated kernel math and is frozen in the same
immutable candidate snapshot, following the existing GPU Infra cooperative-agent
trust model. It is not a hidden, adversarial evaluation boundary. CPU tests cross-check
the integer oracle against system fmaf and separately rounded multiplication, and
negative controls reject unfused, FTZ and incorrect NaN/zero outcomes.

`prepare.py` loads and verifies the released v41 lock, lowers the two canonical
Schedules, and creates a new package. Preserve the supplied virtualenv entry path.
The task runs correctness with broker shared capacity, then memcheck and racecheck
with exclusive capacity. Only the broker chooses a GPU; no retry or reroute is implicit.
The judge requires B200/sm100 and explicit clean sanitizer summaries. Infrastructure
or missing-output failure remains unknown; a numerical or sanitizer failure is invalid.

Expected evidence: 12 workloads per stage, 36 complete-output JSON artifacts and
36,864 output elements across the three stages. Bits, not JSON floating-point NaNs,
are retained. `verify_outputs.py` recomputes every collected output and all input
immutability checks without rerunning the GPU. It requires the complete three-stage
artifact multiset and does not trust recorded mismatch counts.

The node-owned terminal result and sanitizer logs remain separate evidence from this
recomputation. A valid run supports only these fixed-shape kernels/input families and
the stated ABI on the selected B200 target. No performance, framework, training,
arbitrary-shape or full-parent qualification follows.
