# ADR 0012: Profile each correctness-qualified search survivor

Status: accepted, 2026-08-24.

## Outcome and non-goals

A Study that requests the current attribution assay retains benchmark and profiler
evidence for every search survivor whose common Evaluation launch passes correctness.
Profiling still supplies diagnosis only: it does not rank candidates, contribute timing,
decide confirmation or change the Estimand. Incorrect candidates are not profiled because
an attribution receipt is required to describe a correct launch.

This change adds no agent role, runtime mode, event kind, artifact role or Evidence schema.
It composes the existing `search` and `attribution` Evaluation purposes.

## Canonical contract

The Study's sole attribution authority is `evaluation_protocol.attribution_evaluation`.
The current canonical operation is
`correctness_then_profile_each_search_survivor`. The earlier
`correctness_then_profile` spelling remains bounded read compatibility for frozen
Campaigns, where it meant profiling only the selected confirmatory-qualified candidate.

For the current operation, one Turn follows this dataflow:

```text
filter -> search(candidate i) -> if correct: attribution(candidate i)
       -> select by qualified search timing -> confirm selected candidate
       -> next-Turn feedback uses the selected candidate's retained profile
```

The search receipt owns whether its launch was correct. The attribution receipt owns raw
NCU bytes and their checked projection. The confirmatory receipt remains the only owner of
the Run endpoint. No later receipt backfills an earlier one.

## Failure and evidence boundary

Once declared, a missing, malformed or incorrect attribution attempt is a protocol fault;
absence is not reported as profiler coverage. Semantic replay requires exactly one
attribution receipt for every correctness-passing search receipt and none for the other
search candidates. It also permits the legacy selected-only relation only for the legacy
operation spelling.

Acceptance requires candidate-set contract evidence with at least two correct searched
survivors, including a non-selected survivor, plus a negative replay in which one required
profile is absent. GPU qualification remains a successor Study concern; historical raw
evidence is immutable.
