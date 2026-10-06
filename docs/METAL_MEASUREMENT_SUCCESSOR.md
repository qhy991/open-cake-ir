# Metal measurement successor: reduce separation between matched samples

Status: proposal only; no protocol, Compiler or frozen result has changed.

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
Change only launch order within a pair: interleave corresponding warmup samples,
then interleave corresponding timed samples in the pair's declared arm order.
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
including both role orientations and identical-artifact null controls. Retain every
outcome and stop on environment/protocol/correctness failures. Measurement failures
remain outcomes, not permission to add runs. Fix the count and order before launch.
Only a reviewed, independently confirmed protocol may be used by later Ralph runs;
do not pool timings across protocols or report the deliberately inefficient control
as an optimization speedup.

No promotion is also a valid result. If interleaving fails, acquire better device
state/process attribution before introducing another scheduling mechanism.
