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
