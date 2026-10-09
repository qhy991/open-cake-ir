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
restoration. No physical GPU qualification has been performed for this helper.
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
instrumented and not performance-qualified. This entry has not yet run on a GPU.
