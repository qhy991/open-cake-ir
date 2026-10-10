# Default-stream gate component

Related issue: #458. The experimental successor at `3518b4d6` connects this component
to the existing paired worker under a distinct protocol. It starts no Run and owns
no device lease. Independent device and per-task qualification remain outstanding.

The caller supplies the admitted MACA function table, its original single-fill
reset callback and its complete launch callback. The component primes one event
pair, resets once before eleven ordinary warmups, then resets once per each of
five gated observations. Begin/full callback/end remain on stream 0. A separate
nonblocking stream holds a ready event until end submission succeeds.

The complete launch callback must retain its existing tensor/Program owner and
checks. `completed_calls` counts successful complete callbacks, not stage calls.
A partial callback is a failure; the owner must retain its actual stage count.
No C++ function here reads Program internals, selects kernels or removes checks.

Callback return codes retain their failure phase. Gate timeout invalidates the
observation. Cleanup releases every gate and drains both affected streams before
destroying events. If a drain fails, it retains native gate memory and runtime
resources until process exit. The caller must stop and preserve that failure;
it cannot reuse the component or report samples from the failed capture.

The CPU tests compile this exact C++ source and substitute runtime callbacks.
They cover full callback order, stream-zero submissions, partial callback and end
submission failures, cleanup failure, and bounded timeout. Their sentinel event
values are not physical timings. Broader task domains and device qualification of the paired protocol remain future
gates. Program mutation controls are covered below.

## Component tensor adapter

`open_cake_ir.evaluation.metax_queued_events.prepare_helper()` compiles host code before allocation.
`CompleteLaunchCapture` then binds an exact admitted C550 runtime. It delegates
the full call to the existing loader, checks actual stage deltas and catches
callback exceptions before crossing the C ABI. It accesses no private Program
children or prepared-storage tables. The component capture remains separate from the paired adapter described below.

Real Program CPU fixtures cover current/bound stream mismatch, storage rebound
after preparation, omitted stages, partial stage failure, duplicate argument sets,
physical-target versus codegen-family mismatch, and unsafe resource retention.
Native status zero alone does not imply capture completion or usable samples.

If `CaptureFailure.unsafe_to_release` is true, the adapter retains the tensor,
module, callback and helper owners and refuses further captures. A dedicated
component worker must persist the failure and terminate without running an
ordinary finally-unload path. Raw invalid event slots may be nonfinite; the
diagnostic writer must retain them with explicit nonfinite encoding. The adapter
is not a drop-in replacement for the production worker's teardown policy.

Independent Bench task and production-protocol qualification remain outstanding.
The adapter alone introduces no performance qualification.

## Dedicated component entry

`tools/qualify_c550_queued_program.py` accepts a same-source build of the retained
four-stage attention boundary fixture. It prepares the original five CPU cases
before allocation and checks all five on the device before capture. The fixed
block contains one normal cohort and one cohort with 5 ms host sleeps before
its five timed full launches, then original postflight and separate profiling.
Every cohort output is checked. There is no A/A or optimization verdict.

The delayed device intervals must all be shorter than the smallest observed
injected sleep; otherwise the control is inconclusive and stops. This is a large
causal delay control, not an estimate of kernel speed or a clock-domain sum.
The block is not repeated to obtain a preferred result.

A dedicated capture helper retains Program prepared sets on undrained failure.
The ordinary cohort helper is not used because it releases those sets on every
exception. The fatal entry persists its observation and uses process exit 74
without Python stack unwinding. If persistence fails, it still must not enter an
unsafe cleanup path. CPU subprocess controls cover both cases.

The bounded device result is recorded below. Host, source, build and exact
physical lock qualification preceded that block. Preserve the historical
A/A refusal, single-arm diagnostic and author qualification unchanged.

## Bounded component device evidence

A private successor of public 87b67327 passed 88 Linux contracts, the unchanged
204-case Corpus Gate, host/isolation checks, and the same-source four-stage native
build. The dedicated component block then completed once on C550.

- All five original attention input cases passed preflight; each ran four stages.
- Both capture cohorts checked all sixteen fresh-output Program calls. Each
  cohort had eleven warmups and five samples, with 64 observed stage calls.
- The injected sleeps were 5.0549–5.0585 ms. The corresponding device-event
  observations were 24.832–25.600 us, satisfying the declared delay-exclusion
  control. The normal cohort had 25.856–29.440 us observations. All values are
  retained; no sample was removed and the block was not repeated.
- Original postflight passed. Separate ordinary-launch profiling observed the
  four stages in order with zero non-target dispatches. Profiling was not active
  during event capture. Its span includes ordinary host submission gaps and is
  not a performance comparison with the queued interval.
- Total observed work was 160 native stage calls across this small component
  block. This is not the original Bench's sixteen-by-ten correctness test.
  Capture modules closed, the worker exited and physical locks were released.

This supports the queued-interval mechanism on that fixture and exact runtime.
It does not qualify a Bench task, explain the historical uninstrumented spike,
or demonstrate an operator/Compiler speedup. That component version had no paired
registration. The retained run records `No promotion`.

A production successor must review measurement semantics and safe worker teardown
as one change. The ordinary common capture/evaluation finally blocks cannot
release Program intermediates or modules after an undrained failure. After that
review and software acceptance, freeze a distinct protocol and establish fresh
per-task baselines, original correctness, common qualification and attribution.
The historical A/A refusal and previous diagnostics keep their original results.

## Experimental paired successor

`3518b4d6` moves the sole native source and capture adapter into the evaluation
package, then connects `MacaQueuedEventBenchmark` to the existing worker. The new
policy is `fixed_baseline_paired_maca_queued_event_v1`; legacy event records keep
their original kinds and semantics. This branch does not start a Bench search.

The existing AB/BA worker performs five samples per cohort and two cohorts per
arm. The reported mean therefore uses ten samples per arm, with eleven warmups
per cohort. An initial CPU-only implementation incorrectly requested ten per
cohort; its failed native-contract checks are retained. The successor restores
five plus five. No device sample was taken under the incorrect count.

The interval is the default-stream device event interval after the complete
launch has been queued. Reset remains one FP32 fill of four times the declared
L2 before warmups and before each sample. The helper compiles before allocation;
compilation inside a device lease is refused. Receipts bind exact target, stage
counts, samples, reset, stream and successful drain. No old sample is relabeled.

An unsafe drain or an existing terminal callback error retains the tensor,
module, callback and helper owners and propagates `UndrainedDeviceWork` to the
shared worker. Diagnostic formatting cannot replace the terminal signal.
Ordinary drained failures keep the normal release path.

CPU coverage includes the real C++ helper with runtime doubles and a complete
paired-worker, receipt and broker-validation replay. It verifies ten samples per
arm and all 136 stage calls for the synthetic two-stage fixture; those event
values are not hardware observations. Three independent code reviews found no
remaining concrete defect in this scope. At fixed `3518b4d6`, 100 Linux contracts passed without skips in a no-network,
no-device CPU environment. The wheel built successfully and contains the exact
runtime C++ helper. The local macOS packaging attempt lacked `bdist_wheel` and
remains a failed environment check; it was not repaired and rerun. The unchanged
115 Corpus source snapshots also passed. Independent device qualification remains
outstanding; CPU acceptance cannot open the Bench readiness gates.

This experimental ancestry still contains the older Bench Compiler. Integrating
the newer maintained MetaX capabilities and resolving the documented Finding ID
collisions are separate work before a new Compiler evaluation. Neither the old
three-task Bench correctness nor the closed component result automatically
qualifies the new source. Fresh qualification must preserve the original oracle,
separate profiling, and the ten-sample contract without redrawing closed A/A.
