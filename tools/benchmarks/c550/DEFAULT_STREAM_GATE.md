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
values are not physical timings. Exact-target device semantics, Program mutation
controls, a tensor adapter, and a separately frozen protocol remain future gates.
