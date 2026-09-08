# M1 Pro successor design

The [Metal guide](metal.md) owns exact-target selection, supported operations, runtime
requirements and measurement semantics. This proposal extends that path to Apple M1 Pro.
The [Apple7 target definition](../compiler/targets/apple_gpu_family7.json) declares
1024 threads and 32 KiB threadgroup memory; the
current lowering uses one 32-lane threadgroup and no threadgroup memory. These limits
are not an occupancy or timing calibration. Pipeline limits and SIMD width remain
runtime checks on the observed device.

P1/P3: reuse canonical tensor authoring and existing primitives. P2/P8: preserve 32-lane
striped ownership, explicit launch metadata and safe math. P4/P5/P7: extend exact-target
parser/refusals and analysis abstentions together. P6: preserve Corpus expectations,
run the full Gate and focused portable tests, then obtain independent
[ADR 0052 review](adr/0052-independent-agent-release-review.md) before release or GPU evaluation.
Develop in an isolated successor worktree; pinned Git retains released source bytes,
and cycle commands assign Compiler and affected Executor successor identities.

Task experiments use the [TaskLab launcher](metal.md), with one fixed-shape Workload,
all required input distributions, explicit model/harness/effort and the common Ralph
stopping policy. No task owns a second optimization loop. Results must justify further
changes to lane mapping, vector access, shared reductions or dtype support, together
with their legality/analysis updates and successor review.
