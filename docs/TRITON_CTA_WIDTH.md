# Explicit Triton CTA width specialization

`Compiler.specialize_triton_warps(schedule, num_warps=..., schedule_id=...,
entry_point=...)` constructs one complete candidate. The caller chooses the width;
neither assessment nor lowering calls this pass automatically. This is the number
of warps per CTA, not producer/consumer warp specialization.

The two extraction cases are B300 RMSNorm-input-gradient and SwiGLU. Their retained
best candidates use 16 warps, with distinct math and memory workloads. This is a
reason to expose a reusable guarded transformation, not a reason to default all
kernels to 16. A fresh common evaluation determines the useful width per task.

## Contract

- Current evidence covers exact `sm_100a`/`sm_103a` Triton routes. Other targets are
  valid inputs but ineligible for this pass; no target is substituted.
- Input must already be lowering-eligible, have one zero-based role, and declare
  no per-role register split, residency, loops, persistent grid, explicit storage
  allocation or synchronization. Only global/register buffers and ordinary loads,
  pure elementwise arithmetic/casts, CTA reductions and stores are admitted.
- The caller requests a positive power-of-two count within the bound Target's
  `maximum_warps_per_cta`, checked before constructing the new role list. Triton
  preflight also owns this compile-option constraint for non-pass callers.
- Only Schedule id, entry point and role warp list change. The operation graph,
  access maps, global ABI and Workload binding are preserved. The full result is
  reassessed against the same Compiler and target. Original input is preserved.
- Existing width returns `unchanged`. Invalid and ineligible requests return a
  reason and no candidate, never a partly modified Schedule.

The emitted kernel AST is unchanged except its entry name, but the SDK is allowed
to select another reduction tree/layout for a different launch width. Bitwise
equivalence is not promised. Five-case external-oracle validation, fresh paired
timing and NCU attribution must qualify the actual binary under the original
numeric and measurement contracts. Register/shared-memory allocation, occupancy,
latency, numerical counterexamples and slow/null candidates are retained.

## Use and ownership

`tools/apply_warp_specialization.py --schedule input.py --num-warps 16
--schedule-id candidate-width16 --entry-point candidate_width16 --output /external/new`
materializes the complete Schedule and emitted source on a released Compiler.
Lab consumes the Schedule through its existing environment/build/evaluator.
The Compiler has no knowledge of which task, case or measurement selected it.

The paired engineering replay fixes the old baseline binary and evaluates the
current Compiler's width-1 floor separately from width-8/16 candidates. This
distinguishes compiler-floor movement from caller-selected width headroom. It is
not a randomized provider study or evidence of broad shape/architecture transfer.
F-2026-09-15-005 owns the extraction evidence and verification disposition.
