# Acceptance gates

[中文阅读](zh-CN/ACCEPTANCE_GATES.md) · [Bilingual catalog](README.md)

Each gate names the uncertainty it settles and the action that changes if it fails. The
headings describe capabilities directly; internal migration sequence codes are not domain
language.

This document defines stable gates. It does not report which historical run passed them.
Current release authorities are generated in
[`../reports/current/STATUS.md`](../reports/current/STATUS.md); experimental conclusions
remain in their bound Study Reports and Evidence.

## Legacy recovery is durable

**Settles:** whether migration preserves history without copying legacy implementation
topology into active source.

Require a clean, remotely reachable final legacy revision, a complete source/evidence
manifest, an independently restorable history bundle, explicit secret exclusions, and
sufficient storage for rollback. Failure blocks legacy cleanup and any claim that active
source has a recoverable predecessor.

## Product and domain ownership is coherent

**Settles:** whether `open-cake-ir` has one product core, one-way dependencies, and one
owner for every fact.

Require:

- the independently usable Open Cake Compiler and dependent Research Lab;
- separate Compiler, Lab, Evaluation, and Evidence responsibilities;
- separate Workload and Study Contracts;
- complete Authoring Environment as the matched treatment;
- Estimand declared before execution and absent from non-scientific Claim Scopes;
- independent kernel and Compiler evolution loops;
- `matched_search` and the evidence-justified `portfolio` Study variants;
- artifact optimization as a non-scientific Claim Scope, not another execution mode.

Failure returns to Architecture and an ADR before runtime work proceeds.

## The standalone Compiler path works

**Settles:** whether the repository contains an independent compiler rather than an
experiment wrapper.

Through the public Compiler interface, one accepted Schedule must produce deterministic
Assessment and Lowering, while a rejected sibling produces a stable localized Finding.
Exact Target mismatch and missing analysis/calibration coverage must be explicit. Importing
the Compiler must not load provider, GPU, Lab, Evaluation, or Evidence dependencies.

Failure changes the owning Compiler implementation, not a campaign runner.

## A Compiler successor passes the full Corpus release boundary

**Settles:** whether a proposed semantic change preserves or deliberately updates every
reviewed Schedule expectation.

Require accepted and rejected cases, deterministic observations, exact Target definitions,
the complete source closure, a full Corpus Gate diff, and approval written outside the
release automation. A missing, malformed, or stale approval blocks release. Expectations
must never be regenerated merely to make the proposal pass.

Failure returns to Schedule, Target, verifier, analysis, or lowering design; the proposed
revision is not released.

## Common Evaluation preserves the assay boundary

**Settles:** whether both Authoring Environments reach the same correctness and measurement
path without arm-specific semantics leaking past artifact construction.

Require:

- sealed LaunchableCandidates from both Authoring Environments;
- authored, lowered, expanded source, PTX, CUBIN, and SASS as distinct artifact roles;
- correctness before timing or profiler collection;
- explicit route, launch, and fallback counts;
- separate search, confirmatory, and profiler Evaluation Receipts;
- raw profiler output whose summary can be recomputed;
- Measurement Quality distinct from Candidate correctness.

Failure blocks the affected Evaluation path and any dependent Lab claim.

## Evidence is independently replayable

**Settles:** whether every observation and terminal outcome can be reconstructed without
provider, GPU, network, or checkout mutation.

Require create-only no-follow writes, append-only typed events, one Terminal Archive shape
for every outcome, secret exclusion, deterministic fresh-process replay, and regeneration
of Run Audits after deleting derived reports. Replay rejects unknown semantic events and
derives selection and diagnoses from retained authorities.

Archive Integrity and Filesystem Custody remain orthogonal. An intact checkout or an
unanchored external archive may be inspected, but writer admission and claim-bearing
projections fail closed. Prospective custody requires the external writer-origin and
publication witnesses in [ADR 0058](adr/0058-external-writer-custody-anchors.md); open and
audit never backfill them.

## The zero-GPU Lab control plane implements the frozen Study

**Settles:** whether Study execution is preregistered rather than reconstructed after
observing outcomes.

Contract tests through `preflight -> execute -> audit` must prove:

- CampaignLock resolves exact Workload, Study, Authoring Environment, provider, Compiler,
  Executor, toolchain, and custody inputs;
- allocation and Run labels are fixed before outcomes;
- initial and resumed Turns preserve the frozen environment;
- reasoning effort and reference visibility are explicit treatment factors;
- clean-start references carry interfaces and contracts but no target implementation;
- every outcome, including faults and candidate rejection, yields a Terminal Archive;
- Integrity, Filesystem Custody, Protocol Adherence, Endpoint observation, and analysis
  inclusion remain separate;
- adhered candidate failure is observed while external faults remain missing data;
- checkpoints cannot be backfilled by a later Turn;
- scientific reports estimate only the preregistered Estimand;
- system qualification and artifact optimization emit no treatment estimate;
- no source-string assertion substitutes for invoking a public interface.
- successor Agent workspaces begin with exactly immutable `TASK.md` and `AGENTS.md`, add
  only `candidate-set.json`, and retain the complete two-file bundle per iteration;
- the external Ralph Controller—not the Agent—enforces token, wall-time,
  active-authoring, Turn, search, confirmatory and attribution limits;
- a tampered task file, StateCard or replayed task bundle fails semantic replay;
- frozen Prompt-based Studies remain replayable but are not the successor interface.

Failure fixes the owning Module. It never creates a copied versioned runner.

## Closed-provider qualification preserves the authoring contract

**Settles:** whether the real provider CLI honors a frozen closed Authoring Environment and
resume protocol.

Use a no-GPU qualification with an empty workspace, an initial submission, a resumed
update, and exact checks for usage, sandbox, working directory, thread continuity,
submission envelope, feature policy, and reference visibility. Both Cake IR and direct
CUDA/PTX authoring surfaces must qualify under the same provider treatment.

Provider or version drift requires a successor Authoring Environment and a new
qualification. A passed provider transport qualification does not authorize GPU work or a
scientific Campaign.

## Provider-default optimization preserves single-writer custody

**Settles:** whether engineering-only artifact optimization can expose provider-default
features without creating a second Candidate authority.

Require the exact provider feature policy and event vocabulary, typed auxiliary activity,
one primary submission writer, a final no-follow candidate-set envelope, and the external
Lab as the sole evaluator. Qualification may observe provider features; it does not
authorize external mutation, direct GPU measurement, or scientific use.

A bounded artifact-optimization Study may then verify that both Runs receive measured
feedback and promote at most one confirmatory-qualified Candidate each. All comparative
and scientific fields remain structurally absent.

## Bounded end-to-end system qualification composes the canonical path

**Settles:** whether Compiler, Lab, Evaluation, and Evidence compose through one real GPU
path.

Under a `system_qualification_only` Claim Scope, require one bounded Run per Authoring
Environment to pass clean-checkout preflight, execution, terminal sealing, semantic replay,
and independent offline audit. Each Run must retain at least one correctness-qualified
Evaluation Receipt from an actual target launch; a build-only result is insufficient.

The report may state that the system path qualified. It must keep Estimand, estimate, and
uncertainty absent, forbid pooling and comparative statistics, and avoid presenting
observed latencies as an arm comparison.

## Source-authority cutover is recoverable

**Settles:** whether `open-cake-ir` can become the sole runnable source owner without
losing history or rollback.

Require a remotely reachable source revision, released Compiler and Executor anchors,
zero dual-writes, complete legacy-manifest coverage, generated-current-view parity,
storage admission, a pinned rollback checkout, and explicit repository-owner approval.

The legacy tree remains read-only recovery evidence, not a second active implementation.

## Scientific completion is claim-specific

**Settles:** whether a completed Campaign supports the exact Estimand declared before it
ran.

Require all prescheduled Run Audits, preregistered inclusion and missingness handling,
complete Endpoint observations, estimate availability, uncertainty, contamination audit,
and a scope-limited Claim View. An adhered Run with no qualified Candidate is a negative
outcome; an external fault is missing data.

Fixed-shape success cannot satisfy portfolio, model-forward, or serving acceptance.
Unavailable evidence produces an unavailable estimate rather than replacement Runs,
imputation, or a broader narrative claim.
