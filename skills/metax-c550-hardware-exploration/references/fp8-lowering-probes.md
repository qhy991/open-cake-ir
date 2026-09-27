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


## Scalar K and explicit FP16 dot precision screens

At `c3be4379`, replacing only the generated K singleton vector with a scalar
preserved all source masks, the RHS transpose and compensation. After removing
debug locations, TTGIR differed only in constant declaration order. Canonical
native-image extraction found the complete 13,872-byte ELF identical to the
qualified generated-stream baseline, including code and metadata. No distinct
native instruction mechanism remained to time. The unsubmitted resource waiter
was cancelled, the paired request was never launched, and the disposition is
**No promotion**. This exact comparison selects the next action; do not repeat it
as a routine fingerprint check. Evidence:
`/Users/haiyan-infiniai/open-cake-ir-evidence/metax-streaming-scalark-sealed-20260927/native-image-identity.json`
and `metax-streaming-scalark-request-20260927/cancellation.json`.

A separate native-source candidate explicitly decoded FP8 inputs to FP16 and
performed one FP16 `tl.dot` with FP32 accumulation (M16/N64, grid 4). All 254 finite
E4M3FN encodings roundtrip exactly through IEEE FP16 on CPU, and their 64,516
product pairs roundtrip exactly through IEEE FP32. These value facts do not
establish device accumulated-result accuracy. Its TTGIR contains MACA MMA and a
FP16 dot; it is not a native FP8 dot.

C550-2 job `maca-723eda450b48` executed all 37 NT v2 cases with unchanged inputs,
but six full-finite cases failed: suffixes `03`, `11`, `12`, `13`, `14`, `15`.
Eight outputs were outside tolerance; maximum absolute error was 0.0625.
No timing was run. **No promotion**: exact input representation does not make the
changed matrix accumulation numerically eligible. The raw receipt does not
identify the internal arithmetic stage responsible for the error. Evidence:
`/Users/haiyan-infiniai/open-cake-ir-evidence/metax-fp8-via-fp16dot-request-20260927/verification.json`,
its `device-preparation/correctness-output.json`, and the adjacent sealed
collection's `precision-preconditions.json`.


The numerical successor partitions each operand into four magnitude intervals
`[0, .125)`, `[.125, 2)`, `[2, 32)`, `[32, 512)`, computes 16 FP16 dot partials,
and combines them with FP32 Neumaier updates. CPU enumeration bounds each K64
partial at 921,600 integer quanta (20 bits); this supports representability,
not an unmeasured hardware arithmetic guarantee.

At the same producer, native-source job `maca-b5d1e0cba9fa` passed all 37 cases
with unchanged inputs and zero tolerance failures; maximum absolute error
0.015625 means this is not bitwise agreement. Paired search `maca-afdaa03d03e0`
and fresh confirmation `maca-83e79c2c85a7` passed quality and 10/10 pair wins:
10.240/26.624 us (2.60x) and 10.240/26.880 us (2.625x), respectively, against the
sealed generated-stream baseline. The generated-stream A/A control
`maca-7468491f8004` passed quality and was close-null at 26.880/26.624 us
(0.9905x, 0/8 wins and 2 ties). All 60 raw cohorts were replayed.

Separate profile `maca-3dcad48f50d5` has correct instrumented outputs and reports
140 registers/thread, 4,096 dynamic shared bytes, zero static shared and
function-local bytes. Achieved occupancy, bandwidth and instruction counters
were not collected. Allocation is `local_serialized`; external activity is not
excluded. Evidence:
`/Users/haiyan-infiniai/open-cake-ir-evidence/metax-fp8-bucket16dot-confirm-20260927/verification.json`.

This qualifies the **native-source candidate** in the fixed finite NT64 domain.
A composition of existing Cake cast/compare/select/FP16-MMA/arithmetic primitives
is already structurally accepted and eligible for lowering. Requalify that
Compiler-generated artifact before publishing its score or accepting a Lab
recipe. Do not introduce a new instruction or dtype solely to hide an expressible
composition, infer native FP8 dot, or substitute the native-source score for the
Cake-generated route.


The public `bucketed_source(workload)` projection now uses the canonical example
[`xcore1002_fp8_bucketed.py`](../../../examples/python/xcore1002_fp8_bucketed.py).
Producer `0183264c` passes all-case device correctness, paired search, fresh
confirmation, A/A and a separately validated profile. Read the
[platform result](../../../docs/metax-c550.md#有限-fp8-分桶-fp16-dot-组合)
for its generated-source score and evidence boundary. The native/generated gap
remains observed, not causally attributed; preserve both artifacts when choosing
the next source-level hypothesis. Promotion belongs to the task recipe because
existing Cake primitives already express the mechanism.


## Bucketed recipe endpoint and lifetime screens

Two generated-source screens at producer `0183264c` retain the frozen NT v2
Workload, M16/N64/grid4, 16 FP16 dots and the same FP32 Neumaier arithmetic.
Both pass all 37 cases with unchanged inputs, zero tolerance failures and maximum
absolute error 0.015625. The baseline is the qualified generated bucketed recipe,
not the earlier streaming or resident kernel.

| Source change | Correctness / paired jobs | Candidate / baseline median us | Pair wins candidate / baseline / ties | Launch-reported registers/thread |
| --- | --- | --- | --- | --- |
| Remove bucket0 `abs >= 0` and bucket3 `abs < 512` predicates | `maca-18786af04b14` / `maca-bedf30acd5b9` | 11.008 / 11.520 | 10 / 0 / 0 | 180 versus baseline 182 |
| Interleave each dot partial with its existing Neumaier merge; retain all original predicates | `maca-c1635e3e81f5` / `maca-7ee0c5aa6e84` | 11.520 / 11.520 | 0 / 2 / 8 | 186 versus baseline 182 |

Both paired quality gates pass, with 250 samples per arm and 20 raw cohorts
independently replayed per screen. The endpoint screen is a directional 1.0465x
improvement, below the fixed 1.05 materiality threshold: **No performance
promotion**. Do not relax that threshold after observing the score. CPU exhaustive
selection comparison covers 254 finite encodings and 1,016 bucket outputs,
including zero signs. TTGIR loses four float comparisons and four integer
multiplications; native `.text` shrinks from 19,320 to 18,296 bytes. These static
changes do not establish a qualified performance rule.

The interleaving screen preserves the same 271 named operations, parameters,
operands, buffers and dot order in another valid topological order. Native `.text`
grows to 19,416 bytes and launch-reported registers increase. **No promotion**:
shorter source-level partial lifetimes do not predict the realized allocation or
justify a backend scheduling rule from this probe. No fresh confirmation or
separate instrumented profile was run for either non-survivor. Launch resource
queries are distinct from an instrumented attribution profile. Allocation is
`local_serialized`, with external GPU activity not excluded.

Evidence under `/Users/haiyan-infiniai/open-cake-ir-evidence/`:
`metax-fp8-bucket-{endmask,interleaved}-paired-20260927/verification.json`,
adjacent `device-run/timing-samples.json`, and the corresponding sealed roots'
finite-domain / operation-graph equivalence and compiled-code comparisons.
