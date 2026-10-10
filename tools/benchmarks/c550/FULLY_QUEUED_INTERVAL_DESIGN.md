# Proposed fully queued C550 interval

Related issue: #456. Design review only. No new timer is implemented or admitted.
The original A/A and the one instrumented diagnostic remain sealed.

## What the evidence selects

The bounded diagnostic observed a roughly 1 ms native submission call with
almost the same caller-thread CPU time. Moving tensor checks outside the
interval alone cannot remove that submission cost. A C++ submission loop also
calls the same runtime, so fewer Python calls do not prove a stable device span.

| Option | What changes | Remaining uncertainty |
| --- | --- | --- |
| Prepack arguments | Moves Python ABI work before events | The observed runtime submission delay remains inside the event span. |
| Native submission loop | Removes Python between submissions | Runtime submission can still delay device work; its existing reset also differs. |
| Existing gated native timer | Releases work after start/kernel/end are queued | It owns a nonblocking stream, uses a different reset, and admits one dispatch. |
| Gate around the existing full launch | Queues start/full launch/end before release | Proposed. Default-stream ordering and failure cleanup need exact-target qualification. |

The last option is the smallest candidate to investigate. It preserves the
existing loader and all its tensor checks. It is not yet a demonstrated repair.

## Keep the Program owner

`LoadedProgram` binds a stream at construction. It retains arguments and views,
checks storage again before the first stage, launches stages in declared order,
and owns module teardown. `tasks.evaluate._admit_program_assay` currently refuses
the native/gated single-dispatch policies for either Program participant.
`metax_native_events.capture` also requires a single `LoadedMetaxCandidate`.
These are deliberate boundaries. Do not remove either refusal to enable a flag.

Avoid a new packed Program representation or direct reads of `_children` and
`_prepared`. A proposed gate can wrap the existing `loaded.launch(arguments)`
call on its existing stream. All stage validation and launch counting then stay
with their current owners, including checks that reject tensor storage changed
after preparation. The gate must admit a complete Program, not time only its
first or last stage.

## Proposed per-sample sequence

Use the current stream-0 launch contract and original single-fill cache reset.
A separate nonblocking gate stream holds a ready event behind a bounded host
callback. The callback must call no device API.

1. Allocate and retain fresh outputs/intermediates before timing. Check the full
   argument graph through its existing owner.
2. Submit the original cache reset on stream 0.
3. Queue the bounded host gate and ready event on the gate stream; make stream 0
   wait for that ready event.
4. Queue the begin event, call the existing complete launch, and queue the end
   event on stream 0. Keep all modules, argument objects and pointer slots alive.
5. Release the host gate only after successful end-event submission. Wait for
   completion, read the device interval, then validate every output and input.

This gate is a scheduling mechanism, not a source of correct results. The
original oracle and comparator still decide correctness. A long submission can
make the gate time out; timeout must be a visible failure, never a timing sample.
Default-stream semantics can interact with other streams. They must be checked
on this runtime; existing owned-stream evidence cannot establish them.

If launch or end-event submission fails after some stages, release the gate and
synchronize affected streams before destroying events or dropping tensors.
Retain successful stage counts and the failing phase. Do not report a partial
Program as completed, execute a fallback, or continue with the next sample.

## Ownership and evidence gates

- Evaluation owns the optional gate/timer adapter and native gate lifecycle.
  The ordinary loader still owns artifact authority, tensor legality and stage
  execution. No IR, Compiler rewrite, search loop or Run owner is added.
- A distinct timing contract must name the fully queued interval. Keep the
  original measurement kind unchanged. Do not reuse the existing gated record's
  `owned_nonblocking` stream label for stream 0.
- Both comparison arms must use the same new protocol, original oracle and
  reset, ten samples and arithmetic mean. Every sample remains retained. The
  old refusal cannot become a pass after this change.
- CPU controls must reject reordered/omitted stages, changed storage and wrong
  streams. Native helper controls must prove release after end submission and
  complete cleanup after launch/end failures and gate timeout.
- Device component qualification must check stream ordering, exact stage and
  sample counts, fresh outputs, the original comparator, resource lifetime and
  cleanup. Add deliberate bounded submission delay only as a predeclared
  diagnostic control; it cannot produce a production performance number.
- Only after those gates pass may a separately frozen protocol and new baseline
  qualification be proposed. No repeated A/A selection or new optimization Run
  is authorized by this document.

## Current decision

No promotion. The design has no device evidence. The standalone diagnostic
localizes one observation to native submission; it does not identify the runtime
internals or prove why the earlier uninstrumented sample was long. Continue
independent CPU/native Bench preparation while this route remains unqualified.
