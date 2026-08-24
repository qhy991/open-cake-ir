# ADR 0017: Retain the exact provider reference bundle

Status: accepted, 2026-08-24.

## Outcome and non-goals

Executor v25 retains the complete rendered reference bundle embedded in every provider
Turn as a `provider_reference_bundle` Evidence Object. The existing provider workspace
and reference-directory custody checks still prevent mutation; retention adds the bytes
needed for an auditor to inspect what the author actually received.

This does not add an agent state machine, copy the complete dynamic prompt, infer that a
bundle is clean merely because it has a digest, or reinterpret earlier Evidence. Content
classification remains the job of the frozen Study authorities, the implementation-free
starter gate and human review. Retention proves which classified bytes crossed the
authoring boundary.

## Authorities and dataflow

Workload, Study, Compiler and Authoring Environment contracts remain the authorities for
the bundle's facts. `CodexRunProvider` already owns the one materialized projection used
to render a Turn, so it returns those exact UTF-8 bytes with `ProviderTurn`. Lab writes
them to the existing no-follow content-addressed Evidence store and references the object
from `provider_turn_completed`. The CAS digest is an integrity projection, not another
source of reference policy.

Semantic replay for v25 and later requires exactly one such object per completed provider
Turn, verifies its object digest and UTF-8 bytes, and rejects additional unrecognized
objects. CAS deduplication means repeated identical bundles need no second physical copy.

## Failure and compatibility

If a v25 provider completes a Turn without the bundle, Lab records the consumed provider
tokens and seals a `harness_fault` with a missing endpoint before accepting any Candidate.
Empty or non-UTF-8 observations follow the same failure boundary. Missing information is
therefore not reported as a passed contamination audit.

Executor v24 and earlier Runs did not retain this object. Their frozen event sequences
remain replayable under that historical boundary; no object or conclusion is backfilled.

## Acceptance evidence

A concrete Codex provider test must show the same rendered bytes on initial and resumed
Turns. A v25 matched execution must archive and replay the exact CAS object in every Run.
A sibling provider that omits it must produce replayable `harness_fault` terminals, zero
system qualification and one missing endpoint per prescheduled Run.
