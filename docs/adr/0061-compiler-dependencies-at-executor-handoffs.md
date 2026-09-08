# ADR 0061: Compiler dependencies at Executor handoffs

Status: proposed successor Executor boundary.

## Decision

The Compiler owns its target parser, target documents and transitive source closure.
A new Executor does not duplicate those files in its own source list or pin the mutable
current Compiler lock path as another independent authority. CampaignLock already records
the exact Compiler revision ID, relative manifest path and canonical identity; runtime
handoffs copy that reference and independently admit the released Compiler through its
existing revision loader.

The concrete broker requires the Campaign reference. Its worker request carries it, and
`tasks.evaluate` verifies it before interpreting artifacts, Workload or target. Successful
broker observations retain the complete `evaluator_request` bytes; process-failure artifacts
also retain that request. Common Evidence retention requires the request, and semantic
replay compares its Compiler reference with CampaignLock and rederives the existing broker
argument identity. A request with a self-consistent identity but another Compiler is refused.
Replay independently admits the Campaign's Compiler dependency before interpreting events.

The Metal builder likewise requires an explicit admitted reference and obtains its typed
Target from the Compiler revision, eliminating an unbound target-file load. QSA's launcher
copies its preflight Compiler reference into the existing judge command. Every QSA stage
verifies that command binding; its profiler child receives and verifies the same reference
before GPU admission. Direct-CUDA stages therefore cannot bypass the dependency merely
because they do not compile Cake source.

The standalone target-peak observation tool remains a calibration instrument with its
explicit target input and calibration/release acceptance obligations. This decision does
not turn invocation of that instrument into a qualified Campaign or measured target claim.

## What stays in the Executor

Imported package initialization and CLI code remain part of the runtime closure. Paired
CuTe/Triton authoring documentation also remains bound because it is visible treatment
material. Removing either merely because it is not a timer would leave the executable or
authoring environment under-specified. Shared serialization code is an Executor dependency.

## CPU semantic tests and release verification

CPU Lab tests use an explicitly injected synthetic Executor identity and host document.
They neither admit a real host nor run a release cycle, rehash the current source closure,
or restamp changed repository authorities. The same fixture is injected explicitly in
fresh-process CLI semantic tests. Real Executor descriptor/source verification stays in
the dedicated release/audit contract tests and release commands. Compiler source admission
continues to use its existing owner, with targeted dependency substitution tests.

This separates semantic regressions from a release-boundary mismatch. It does not claim
that unreviewed source is a released Executor or authorize provider/GPU experiments.

## Revision and replay boundary

These changes require an independently reviewed Executor successor. Released descriptors,
locks, historical broker requests, failed Runs and reports remain unchanged. The active
request contract requires the Compiler reference and retained request; it does not contain
a fallback parser for requests emitted by older Executors. Those historical records use
their pinned implementation. No Compiler primitive, target value, source closure or
Compiler release is changed by this decision.

## Validation

Tests cover missing/moving/stale references, a substituted target document rejected by the
Compiler owner, worker refusal before Workload interpretation, exact sealed-bundle request
transport, QSA command and profiler-child admission, and independent replay rejection even
when a foreign request has a consistent argument identity. CPU Lab lifecycle, paired
Evaluation, runtime configuration and task-launch tests retain their existing behavior.
No provider invocation, GPU dispatch or host qualification is used to validate this slice.
