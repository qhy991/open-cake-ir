# Default-stream gate component

Related issue: #458. This native component has no production adapter or admission.
It starts no Run and owns no device lease. Production qualification remains closed.

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
values are not physical timings. Broader task domains and a separately frozen production protocol remain future
gates. Program mutation controls are covered below.

## Component tensor adapter

`default_stream_adapter.prepare_helper()` compiles host code before allocation.
`CompleteLaunchCapture` then binds an exact admitted C550 runtime. It delegates
the full call to the existing loader, checks actual stage deltas and catches
callback exceptions before crossing the C ABI. It accesses no private Program
children or prepared-storage tables. It is not registered in the evaluator.

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
or demonstrate an operator/Compiler speedup. Production registration remains
closed. The retained run records `No promotion`.

A production successor must review measurement semantics and safe worker teardown
as one change. The ordinary common capture/evaluation finally blocks cannot
release Program intermediates or modules after an undrained failure. After that
review and software acceptance, freeze a distinct protocol and establish fresh
per-task baselines, original correctness, common qualification and attribution.
The historical A/A refusal and previous diagnostics keep their original results.
