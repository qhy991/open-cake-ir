# ADR 0044: External evidence freezes gfx1151 Executor v1

Status: accepted after detecting an invalid same-id release attempt.

## Context

The gfx1151 Executor cycle discovers digest-bound references in repository-owned frozen
contracts, evidence, Campaign Locks and non-derived inventory. Runtime descriptors do
not witness themselves, and an external evidence root must first be registered in the
repository. Three retained AITER RMSNorm attempts lived outside the checkout and bound
three historical byte incarnations of `open-cake-ir-gfx1151-v1`, but no registration
made those bindings visible to the cycle. The working descriptor was consequently also
reissued under v1.

This was not safe merely because the planner allowed it. An Executor ID named by any
retained attempt is immutable: a failed attempt still owns its environment authority,
and the successful correctness attempt owns the exact Executor bytes it admitted.

## Decision

1. `inventory/AMD_GFX1151_EXECUTOR_V1_BINDINGS_20260826.json` registers all three
   external attempt bindings and the last repository working descriptor without
   pretending their different bytes were one canonical artifact.
2. The registration is a digest-bound witness. The release planner must report v1 as
   witnessed, refuse to reclaim its path and derive `open-cake-ir-gfx1151-v2`.
3. The 47-source v1 release attempt at `cb351264` is invalid because it reused an
   evidence-bound identity. It is retained only as an incident record and is not
   published to the canonical GitHub branch.
4. The 47-source closure is released once as v2 from a clean successor commit. Before
   any later Executor-pinned source changes, v2 receives its own repository-owned
   digest registration so that a future cycle necessarily derives v3.

## Boundary

The registration does not copy raw external evidence into Git, reinterpret a blocked
attempt as a success, or select one v1 incarnation as the bytes of every historical
attempt. Each evidence manifest remains authoritative for its own observation. The
registration exists only to preserve identity history and close the release-cycle gap.

Executor succession does not approve Compiler v29, freeze the RMSNorm Search Contract,
authorize GPU timing or establish an operator-performance result.
