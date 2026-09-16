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

- Static admission covers exact `sm_100a`/`sm_103a` Triton routes. GPU performance
  evidence currently covers only FP32 128x1024 on `sm_103a`. Other targets are
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
[F-2026-09-15-005](../findings/2026-09-15-005-explicit-triton-cta-width.json)
owns the extraction evidence, both GPU rounds and verification disposition.

The reusable consumer is `tools/evaluate_triton_widths.py`. For example:

```sh
python tools/evaluate_triton_widths.py --source /absolute/frozen-checkout \
  --predecessor /external/retained-matrix --output /external/new-evaluation \
  --task softmax_backward --task cosine_similarity
```

The predecessor supplies each task's Workload, starter, evaluation protocol and
sealed `baseline-preflight/<task>/baseline/candidate.json`. Task names must be unique
identifiers. With no `--task`, the extraction pair is used. All requested tasks build
widths 1/8/16 and validate ABI pairs before any GPU work. The CPU parent must have
no GPU lease or selected device; the existing broker command allocates each GPU job.
The chosen frozen checkout must have released Compiler and matching Executor sources.
Each task then performs three searches, a fresh confirmation of the selected stable
correct candidate and primary-case NCU. The output is create-only. A missing/failed
receipt stops the sequence and retains the preceding evidence.

## Tick-tock readout

The [project cadence](../AGENTS.md#tick-tock-between-campaigns-and-revisions-outer-loop-cadence)
and [experience promotion rules](../AGENTS.md#kernelcompiler-co-evolution-experience-promotion)
remain the policy owners. This pass is one concrete application of them:

| Observation | Promoted owner | Retained limitation |
| --- | --- | --- |
| Compiler-generated FMA/infinity code failed common source admission | [Compiler source-admission repair](../findings/2026-09-13-002-triton-generated-fma-infinity-admission.json) | A successful compile is not GPU correctness or speedup. |
| Six declared warps reached a Triton SDK assertion | Triton preflight `TRITON_NUM_WARPS_UNSUPPORTED` | The constraint belongs to this backend, not a vendor-neutral Role parser. |
| Width changes recurred in RMSNorm gradient and SwiGLU | Explicit `specialize_triton_warps` candidate construction | No implicit width selection or bitwise reduction equivalence. |
| The same pass produced confirmed candidates for softmax backward and cosine similarity | Additional task evidence in the existing Finding | Same FP32 128x1024 B300 scope; no cross-shape or architecture conclusion. |
| Missing provider terminal or writes to a different directory ended a Run | Lab/provider protocol diagnosis | A local valid receipt does not retroactively qualify a faulted Run. |

The measured selections differ: RMSNorm gradient and cosine similarity selected
8 warps; SwiGLU and softmax backward selected 16. Width-1 comparisons were null or
below materiality. These observations keep the parameter choice in Lab rather than
turning this pass into a global default. The original Finding owns the actual
paired values and paths; summaries must use each confirmation's own baseline.

This tick-tock integration combines the unchanged, approved Compiler v85 source
with the newer runtime. [Current release status](../reports/current/STATUS.md) owns the
active version pointers. The historical GPU evidence remains bound to its recorded
Executor v126; integrating a newer Executor does not rebind or upgrade that evidence.
Next experiments can vary a shape or dtype under a new frozen workload boundary,
keeping the selected mechanism and fixed baseline explicit before measuring again.
