# Metal measurement successor: reduce separation between matched samples

Status: explicit v3 implemented at `a375e295`; targeted Python checks passed.
All eight native CPU snapshot cases passed in a separately user-approved
Xcode16/SDK15 environment with a private temporary module cache. Full CI at
`ea80a53f` passed. A separately compiled observer and external host proposal passed
preparation/admission; source integration and device qualification remain pending. Compiler and frozen results are unchanged.

## Evidence and ownership

The fixed `28e36de4` public Evaluation compared the retained R128 C1024 RMSNorm
archive with a Cake-authored redundant-row-reduction control. All outputs passed;
all ten pairs favored the original archive. The redundant arm passed all ten
relative-IQR gates, while the original arm failed three (pairs 0, 1 and 6).
The controlling result remains `measurement_quality_failed`; reverse-role and
independent A/A trials were not executed. Evidence is in
`~/.local/share/open-cake-ir/planning/metal-xcode16-20261006/g3-artifact-direction-controls-28e36de4/`.

The prior four C/W/W/C A/A trials passed with both 3 and 30 warmups. Their
duration-versus-preceding-gap correlations range from 0.008 to 0.086, and long
tails remain. They do not justify an arbitrary warmup or host pacing change.

The current plan executes 28 command buffers for one arm, then 28 for the other.
The redundant arm's median command interval is about 59.7 ms, versus 0.274 ms for
the original arm. Thus one arm's cohort occupies roughly seconds while the other
occupies milliseconds. Load-dependent state changes or unrelated activity may
interact with this separation; neither cause is established without attribution.

This is an Evaluation protocol hypothesis, not an IR or verifier defect.

## Proposed diagnostic treatment

Keep the same two sealed artifacts, exact host/toolchain, input cases, external
oracle, tolerances, 64 dispatches per command, 3 warmups per arm per pair, 25 timed
samples per arm per pair, ten AB/BA pairs, and the existing IQR and direction gates.
Interleave corresponding warmup samples, then corresponding timed samples in
the pair's declared arm order. This necessarily also moves snapshot persistence
from each arm's cohort end to the pair end. The treatment therefore bundles sample
order and persistence boundary; it cannot isolate an order-only causal effect.
Retain all raw timestamps and separate correctness snapshots.

This needs an explicit successor protocol rather than silently changing v2.
The old protocol remains replayable. Do not use fewer samples, an easier IQR gate,
extra discarded warmups, a runtime-selected batch size or a retry-until-stable loop.

## Implementation and preflight boundaries

- Add an explicit measurement-order declaration owned by the successor protocol;
  central plan generation and receipt validation must agree on exact global order.
- Existing Swift `snapshotFlushPlan` requires contiguous arm/cohort snapshots.
  An interleaved pair must instead retain a bounded pair-sized snapshot window,
  validated before dispatch. Do not flush between every short command, which would
  introduce disk I/O into the intended comparison. Preserve the existing 64 MiB
  bound or refuse before device work; this R128 C1024 pair requires 58,949,632 bytes
  for 56 launches. Do not silently raise the bound for other workloads.
- Retain the CPU-prepare/native-broker/CPU-compare phase split and fresh outputs.
- Test exact warmup/timed ordering, retained sample reconstruction, dropped,
  duplicated, misordered or wrongly paired observations, count accounting,
  pre-dispatch snapshot bounds and preserved old-protocol replay. Use synthetic
  timestamp inputs to verify classification parity when only order changes.
- Commit and pass applicable software gates before building a new observer.
  Bind its new helper explicitly in a successor host environment; do not edit or
  re-label the currently admitted helper or existing evidence.

## Bounded device validation after software acceptance

Predeclare a matched blocked/interleaved diagnostic using both exact artifacts,
including both role orientations and identical-artifact null controls. Both
protocols must use the same successor observer and captured host. Retain every
outcome and stop on environment/protocol/correctness failures. Measurement failures
remain outcomes, not permission to add runs. Fix the count and order before launch.
Only a reviewed, independently confirmed protocol may be used by later Ralph runs;
do not pool timings across protocols or report the deliberately inefficient control
as an optimization speedup.

No promotion is also a valid result. If interleaving fails, acquire better device
state/process attribution before introducing another scheduling mechanism.

## Implementation evidence (2026-10-06)

`fixed_baseline_paired_metal_v3` explicitly selects interleaved physical command
order; v1/v2 and task defaults keep their prior semantics. Execution and receipt
replay share `metal_cohort_calls`. v3 requires pair-sized snapshot admission
before the native lease, and the observer must report that it honored the pair
window. Attribution remains separate. No Compiler or author configuration changed.

At fixed checkout `a375e295`, the targeted Python suite passed **99 tests and
189 subtests**; eight native Swift tests were excluded from this Python run.
Counterexamples cover relabelled blocked receipts, missing/duplicate/misordered
samples, wrong pairing, pair snapshot overflow, and an observer ignoring the new
request. This is software verification, not measurement qualification.

The separate Xcode16/SDK15 CPU-only harness attempt stopped before compilation:
the sandbox refused the compiler's default ModuleCache output. No behavior case
ran and no GPU work occurred. Exact command and failure are retained in
`/private/tmp/cake-metal-interleaved-swift-a375e295/contract.json` and `build.json`.
The failed attempt is not normalized or reported as a pass. A separately approved
successor environment is required before continuing that native CPU acceptance.

The user then explicitly approved a successor CPU environment with a private
temporary module cache. At the same fixed source and Xcode16/SDK15, all eight
snapshot behavior cases passed, including interleaved pair boundaries, overflow,
owned copies, partial failure flushing and overwrite refusal. Evidence is in
`/private/tmp/cake-metal-interleaved-swift-a375e295-private-cache/`; the original
environment failure remains separate. This CPU-only build defines
`SNAPSHOT_TESTS`, excludes Metal device execution, and does not qualify timing.

## Proposed fixed diagnostic sequence

After the software and successor-host gates pass, execute exactly the following
six evaluations, each using the existing 10-pair policy. This is a bounded
engineering diagnostic, not a protocol superiority study or a Ralph campaign.

| Order | Protocol | Candidate | Baseline | Expected valid classification |
| --- | --- | --- | --- | --- |
| 1 | v2 blocked | redundant RMSNorm | original RMSNorm | second_arm_faster |
| 2 | v3 interleaved | redundant RMSNorm | original RMSNorm | second_arm_faster |
| 3 | v3 interleaved | original RMSNorm | redundant RMSNorm | first_arm_faster |
| 4 | v2 blocked | original RMSNorm | redundant RMSNorm | first_arm_faster |
| 5 | v2 blocked | original RMSNorm | original RMSNorm | close_null |
| 6 | v3 interleaved | original RMSNorm | original RMSNorm | close_null |

Maximum: 3,480 command buffers, 215,160 dispatches, 3,000 timed samples, no provider
calls, 300 seconds per evaluation. Stop on environment, protocol, correctness or
unexpected valid direction failures. An ordinary IQR failure is retained as an
outcome and does not add, retry or reorder evaluations. Do not pool protocols or
discard cohorts. A/A also requires the original [0.95, 1.05] ratio interval.

Even if all v3 controls pass, independent confirmation is still required before
promotion. If v3 fails, do not introduce another scheduling mechanism without
better attribution. This small, ordered diagnostic cannot establish causality,
exclude external GPU clients or establish generalization beyond R128 C1024.

## Full software gate and native preparation

CI `37408412819` completed on its actual merge checkout `ea80a53f`: Python
3.10/3.11/3.12 each passed 2,854 tests with 36 skips (3.12 also reports one warning).
This includes `metal@f619ff4c` research-document updates; they change no Compiler or
Executor source. The fixed checkout is
`/private/tmp/cake-metal-interleaved-ci-20261006` and the retained log is
`/private/tmp/metal-interleaved-ci-37408412819.log`.

The observer compiled successfully from that source using Xcode16/SDK15, the
explicit macOS15 target and an independent temporary module cache. Build records
and executable are in
`~/.local/share/open-cake-ir/planning/metal-xcode16-20261006/interleaved-observer-ea80a53f/`.
No old helper bytes were replaced.

Automatic approval rejected capture with `--replace` into the task's existing
host path because of the frozen-binding risk; that capture never ran. The safer
capture wrote a new external host proposal instead:
`/private/tmp/metal-interleaved-host-proposal-ea80a53f/runtime/hosts/apple_gpu_family9.json`.
Capture and admission passed with zero dispatches. Only the observer executable
record differs from the prior host document. Device, OS, SDK, Swift, Python and
archive helper records are unchanged. Inspection evidence lives in
`metal-xcode16-20261006/interleaved-host-ea80a53f/`; helper processes ended and no
broker lock holder remained. Integrating this proposal into a new source commit
awaits explicit user authorization; no measurement trial has run with it.
