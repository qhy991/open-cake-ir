# M1 Pro successor design

The exact target `apple_gpu_family7` admits `Apple M1 Pro`, independently of
`apple_gpu_family8` / `Apple M2`. The target, parser, lowering preflight, host
projection and native runtime must agree; no automatic target downgrade exists.
Apple7 supports 1024 threads and 32 KiB threadgroup memory, per the
[Apple feature tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf).
Pipeline limits and SIMD width are still checked on the observed device.

P1/P3: reuse canonical tensor authoring and existing primitives without a second
layout language. P2/P8: preserve 32-lane striped ownership, explicit launch metadata,
safe math and inspectable source. P4/P5/P7: extend exact-target parser/refusals together;
Apple7 has no CUDA capability, occupancy facts or calibrated cost model. P6: preserve
all existing Corpus expectations, run the complete Gate and focused CPU/host tests,
and obtain independent ADR 0052 review before release or GPU evaluation.

Develop only in the isolated successor worktree; the original released checkout
and pinned Git retain the existing Compiler and Executor source bytes. Compiler
and affected Executor successor identities are assigned by their cycle commands.

Bounded local known-kernel optimization compares four distinct RMSNorm DAGs at
(128, 1024): canonical, weight first, prescaled square, and scale weights first.
The last changes the multiplication dependency after the inverse RMS calculation;
it is a performance hypothesis, not an assumed gain. Each run retains one randomized
search and two independent confirmation rounds, A/A control and separate profiling.
Run three complete repetitions with unchanged inputs, tolerances and stopping rules.
No cost ranking is available for Apple7; record this abstention before GPU time.
Raw receipts and the derived experiment report remain outside every checkout.
Future changes to lane mapping, vector access, shared reductions or dtype support
need their own legality/analysis update and successor review, justified by results.
