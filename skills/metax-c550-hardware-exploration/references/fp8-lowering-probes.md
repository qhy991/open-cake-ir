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
