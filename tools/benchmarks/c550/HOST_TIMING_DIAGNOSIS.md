# Host timing diagnosis

Related issue: #456. This helper is not imported by the production evaluator.
It creates no Run, device lease, baseline, timing verdict or retry.

Use `instrument_host_phases(loaded, torch, timeline)` in a dedicated process
around one existing `MacaEventBenchmark` cohort. The caller must admit the exact
source, artifact, target and original correctness contract before execution,
and hold the existing physical device lock. `loaded` is the open native loader,
not the outer Torch wrapper. Retain `timeline.rows` even when capture fails.

The helper records monotonic wall and caller-thread CPU timestamps around event
submission, tensor argument preparation, native submission and end-event wait.
It delegates every operation and restores the original objects on exit.
It refuses overlapping instrumentation and more than 256 records. Run no other
Torch work concurrently in that process.

Instrumentation changes host overhead. These timestamps and instrumented device
events cannot replace production latency or a failed A/A. Thread CPU time omits
runtime helper threads; wall minus thread CPU is not proof of scheduler delay.
The two clocks bracket slightly different intervals and are not GPU timestamps.

Before a device diagnostic, fix its hypothesis, sample schedule and stop rules.
Retain all samples and stop on a correctness/runtime fault. If the suspected long
sample does not recur, report the cause unresolved. Do not repeat until a desired
result appears. Do not change mean-of-10, drop an observation, or switch timers
to make the historical same-artifact comparison pass.

CPU contracts exercise the real timer and native loader with device API doubles.
They check dispatch count, validation order, rejection, exception identity and
restoration. A subsequent bounded device diagnosis is recorded below. It does not qualify the performance route.
No promotion is proposed by this change.

## Fixed single-arm entry

`tools/diagnose_c550_event_interval.py` binds all sixteen same-source RMS
baselines, then selects original case index 1 before device admission. The block
is fixed: one preflight, eleven warmups, five instrumented samples, one postflight.
It uses the existing physical lock, original input factory/comparator and tensor
lifecycle. Every cohort output is checked. It never computes a paired verdict.
An existing output path refuses another invocation. Keep all terminal results;
creating a different path does not authorize a repeat.

Hypothesis: host argument preparation, event submission or runtime submission
may explain the earlier long interval. This block observes these phases with
wall and caller-thread CPU clocks. It neither reproduces an A/A nor determines
that the historical spike had the same cause. If no long sample recurs, the
historical cause stays unresolved. There is no automatic follow-up block.

Before execution, use a clean successor source, same-source baselines, accepted
CPU contracts and host/isolation gates. Record exact physical/runtime/PCI
binding and the argv outside source. The output explicitly states that it is
instrumented and not performance-qualified. The one device block below used this entry; do not repeat it as an A/A replacement.

## Retained device diagnosis

A private successor of public bb959965 passed 77 Linux CPU contracts, the unchanged
204-case Corpus Gate, host/isolation checks and sixteen same-source RMS native
bindings. One original case-index-1 diagnostic then completed eighteen native
calls with preflight, all cohort outputs and postflight accepted by the original
comparator. The module closed, workers exited and physical locks were released.

The five instrumented device events were approximately 32.000, 34.816, 32.256,
1030.144 and 85.760 microseconds. For the 1030.144-microsecond observation, the
wrapped native submission call occupied 996.209 microseconds of host wall time
and 994.819 microseconds of caller-thread CPU time. Argument preparation took
17.107 microseconds. This localizes the dominant host delay in that observation
to the native submission boundary, with caller CPU time nearly equal to wall
time. It does not identify the internal runtime operation, establish a runtime
bug, or prove the cause of the earlier uninstrumented 109.568-microsecond sample.

The fifth sample had 42.160 microseconds of argument preparation plus a
30.173-microsecond uninstrumented host gap before preparation. Thus preparation
alone does not explain every long interval. Host timestamps and device events
are different clock domains; their durations are observations, not an additive
decomposition of device execution. Instrumentation itself adds overhead.

No timer, threshold or baseline was promoted. The frozen A/A remains refused.
The evidence motivates investigating submission cost separately and reviewing a
fully queued device interval, including complete Program semantics, as a distinct
protocol. Neither is a demonstrated repair. Raw traces and original inputs remain
in private evidence; no new provider or optimization Run was started.
