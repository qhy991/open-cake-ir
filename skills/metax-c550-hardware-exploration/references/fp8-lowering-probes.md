# Fixed FP8 GEMM lowering probes

Read this when changing K selection, residency or unrolling in the fixed
`maca.simt.fp8e4m3_compensated_fp32` route. The [Workload](../../../contracts/workloads/metax-fp8-e4m3-gemm-fp32-xcore1002-m64-n64-k64-v2.json)
owns the result and all 37 required input cases. These compiler observations
are not new instruction contracts.

## C550-2 Triton 3.6 observations, 2026-09-27

The probes start from resident compensated source generated at `b0709b38`.
Each changes one mechanism and retains FP32 Neumaier updates, output precision
and the two-row launch. The installed compiler was Triton API `3.6.0` in the
C550-2 MACA PyTorch container; these results do not qualify the older 3.1 route.

| Candidate | First observed result | Next decision |
| --- | --- | --- |
| Dynamic `tl.gather` of decoded resident tiles | TTIR/TTGIR contain `tt.gather`; MLIR-to-LLIR refuses generated `maca.shfl.sync` as an unknown custom op. Unchanged source compiles in the same environment. | This spelling cannot replace the admitted selection on this runtime. Another compiler requires a new probe; this is not a hardware impossibility. |
| FP8 loads inside each K iteration | Compiles to `mcfatbin`, with two loop-body loads and no `tt.dot` in TTIR/TTGIR. | Screen device correctness next. Extra reads and changed residency require measurement before replacement. |
| K64 `tl.static_range` on resident source | Compiles, removes the IR loop, retains repeated selection/reduction operations, and expands the native artifact from 46,428 to 356,828 bytes. | Selection elimination is not observed at TTIR/TTGIR. Unrolling and binary size alone establish no speedup. |

None of these probes executed a device kernel or measured latency. All remain
outside Compiler promotion. Inspect source, failed stage and control before
using an observation for another candidate.

## Evidence locators

These are external experiment locators, not files shipped in the skill:

- `Agent4Kernel/open-cake-ir-evidence/metax-fp8-gather-hypothesis-20260927/result.json`
  and `c5502-triton36/`: failure log, TTIR/TTGIR/MLIR and unchanged control.
- `Agent4Kernel/open-cake-ir-evidence/metax-fp8-streaming-hypothesis-20260927/result.json`
  and `c5502-triton36/`: streaming source and offline artifacts.
- `open-cake-ir-evidence/metax-fp8-unroll-20260927/result.json`
  and `c5502-triton36/`: full-unroll artifacts and comparison.

Resolve these against the actual experiment collection. If unavailable, this
page points to a prior observation, not a current verification. Do not replace
old inputs or runtime and call the old result replayed.

## Formal baseline entry

Use the task-owned `metax_fp8_gemm` starter, which projects the admitted example
and binds Workload metadata. The bare compiler example was refused at Schedule
Workload binding. The corrected `--baseline-only` path at `3d222f89` sealed a
C550-2 baseline without provider or GPU calls; all 37 cases preserve the primary
launch ABI. The host successor and correction integrated in PR #250 at `5368f77c`.

Formal candidate collections must be outside every source checkout, including
ancestor checkouts. The canonical bundle loader refused the initial collection
under the parent `Agent4Kernel` Git repository. Its external successor passed
the bundle loader and paired ABI check; that admission establishes no device
correctness or performance. The retained result is
`open-cake-ir-evidence/metax-fp8-formal-baseline-20260927/collection-result.json`.

Admit the exact Executor host and sealed artifact, prepare complete Workload
inputs/reference, then acquire the shared MACA broker. Promote only after
all-case device correctness and the declared timing and confirmation gates.

## RHS orientation correction

The v1 Workload defined `A @ B` while the resident instruction and its original frozen oracle compute `A @ B.T`. Square ABI checks missed this semantic mismatch. At `fe47e4cc`, the original frozen identity case has zero word differences against `A @ B.T` and 4,006 against `A @ B`; the new v1 identity input has 3,998 tolerance failures against the resident result. The pending sealed v1 request must not qualify this resident route. Its first broker attempt executed zero kernels. Keep the v1 contract and all historical observations intact. Use the explicit NT v2 successor, reseal the baseline under that Workload, and separately adapt streaming RHS addresses before its device check. The observation is retained at `open-cake-ir-evidence/metax-fp8-rhs-orientation-20260927/result.json`.

## Native launch ABI and repaired all-case replay

A captured Triton 3.6 native ELF can declare two zero scratch pointers beyond the
public TTGIR tensor signature. Derive and verify the launch count from the actual
native metadata; the retained 3.1 ELF has no such additional slots. Do not use
TTGIR alone or a universal vendor/version default. Nonzero or unmodeled scratch
requirements remain refused at compilation. The loader must reject a sealed
count mismatch before runtime calls.

The exact NT v2 sealed baseline at `875724d4` completed broker job
`maca-1a6e9f748221`: 37 cases and 151552 elements, zero mismatches/maximum absolute
error, unchanged inputs, 37 native module loads, 37 kernel calls and zero
compilation/fallback/timing calls. This closes
[F-2026-09-27-001](../../../findings/2026-09-27-001-metax-native-scratch-abi.json).
The collected worker artifacts are at
`open-cake-ir-evidence/metax-native-abi-request-875724d4/device-run/`; the case and
counter audit is `device-verification.json` in its parent. This is registered
Workload native correctness, not a GPU bitwise audit, timing or full-backend claim.

## Generated streaming route

The existing compensated contract now has a narrow explicit K1 loop form using
A[2,1]/B[64,1] and result[2,64] for the same fixed NT64 domain. Use the task-owned
`streaming_source(workload)` projection; keep the resident starter as the fixed
baseline. Loop entry owns both FP32 total and correction, and only the final
live-out store consumes their corrected result. Other consumers, K2, eight groups
or full unroll remain refused by the streaming guard. No new DType or hidden
memory access is introduced.

Read the current [platform result](../../../docs/metax-c550.md) and
[F-2026-09-27-002](../../../findings/2026-09-27-002-metax-fp8-streaming-lowering.json)
for generated-source qualification at `c3be4379`: all 37 cases pass; quality-passed
search and fresh confirmation repeat 26.880/155.904 us (5.80x on primary), with null
A/A and a correct separate instrumented profile. The authored-source 6.84x result
is not this generated-source score. Broader shapes, cache/load policies, consumers
and control options require their own evidence; do not widen the guard from this
one domain or infer a native FP8 matrix instruction.


## Generated-source representation screens (2026-09-27)

For the fixed NT v2 Workload at producer `c3be4379`, three native-source screens
used the generated streaming artifact as baseline. Each passed all 37 cases
(maximum absolute error 0, inputs unchanged), then replayed 20 raw MCPTI cohorts
(250 samples per arm). All timing quality gates passed; candidate and baseline
medians were 26.880 us in every screen.

| Isolated source change | TTGIR comparison / transpose counts | Correctness job | Paired job | Pair outcomes candidate / baseline / tied |
| --- | --- | --- | --- | --- |
| Omit K masks, retain row/store masks | 1 / 1 | `maca-6ba023cc1ba5` | `maca-81477c35c119` | 0 / 3 / 7 |
| Orient RHS load as a row, remove explicit transpose | 2 / 0 | `maca-ecc8db411a95` | `maca-0f95fa090fd4` | 1 / 0 / 9 |
| Combine both changes | 1 / 0 | `maca-79413adc4282` | `maca-3d3f0fe2dad9` | 1 / 0 / 9 |

**No promotion.** Fewer comparison or transpose operations did not yield a
material timing benefit in these screens. Keep the generated lowering unchanged;
these counts do not explain the authored/generated latency gap. The joint screen
also does not establish causal interaction. No fresh confirmation or instrumented
profile was run for these non-survivors. Allocation was `local_serialized`;
external GPU activity was not excluded.

Retained evidence is outside the source checkout at
`/Users/haiyan-infiniai/open-cake-ir-evidence/`, under
`metax-streaming-{kmask,rhsrow,combined}-paired-20260927/verification.json` and
its adjacent `device-run/timing-samples.json`. The K-mask seal report's original
baseline locator was stale; its appended `seal-interpretation.json` records the
actual generated-stream baseline used by pair validation. Do not reinterpret the
original report as a resident-baseline measurement.

When investigating the remaining gap, isolate another source difference and
replay the sealed device result; static IR simplification alone is insufficient
reason to add a MetaX lowering rule.
