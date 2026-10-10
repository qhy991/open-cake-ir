# Default-stream gate component

Related issue: #458. This native component has no production adapter or admission.
It starts no Run, owns no device lease, and has no device qualification.

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
values are not physical timings. Exact-target device semantics, a device entry and a separately frozen protocol
remain future gates. Program mutation controls are covered below.

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

Exact-target device validation remains outstanding. No production protocol or
performance qualification is introduced by the adapter.

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

This entry has no device result yet. Host, source, build and exact physical lock
qualification must precede its one bounded device block. Preserve the historical
A/A refusal, single-arm diagnostic and author qualification unchanged.
