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
profile is absent. GPU qualification had to use a successor Study rather than rewriting
historical raw evidence.

The successor Study `open-cake-ir-candidate-set-system-v3` (canonical SHA
`180c0d4787c4f2361e5fe7bd342fbb7e07def6ca0f0e5a7551993d8fe4364b7e`) closes that
bounded GPU concern under Executor v18. Its external Campaign authority SHA is
`56bbcfe06caa4be7db31689d658ed59af3e2fe0ae2b4a88dc682ef9c8b373d8e`, and its Evidence
root is `/home/qinhaiyan/open-cake-ir-evidence/campaigns/candidate-set-campaign-live-v3`.
Each arm authored three launchable Candidates and searched two correct survivors. Both
the selected and non-selected survivor received attribution, so each Run retains two
search, two attribution and one confirmatory Receipt. All ten Receipts observe one target
kernel call and zero fallback calls; all four attribution Receipts carry eleven raw-checked
NCU metrics and no timing. Fresh-process audit reconstructs archive integrity, exact
profile coverage, semantic replay and adherence, and sets only
`system_qualification_passed=true`. The terminal seals are
`38a362b97a03747bd217bb070aa4a34e9f39de73f16012056ddbe52552dcd081` and
`84afb88f11f42907bb6c9a66e1ff52c4471f7450954fc08200b0b33d1edaa030`.
Estimand, estimate and uncertainty remain null; this is not a scientific Campaign or an
arm comparison.
