# Acceptance gates

Each gate names the uncertainty it settles and the action that changes if it fails. Passing a narrow gate never
supports a broader scientific claim.

## Current gate state

| Gate | State | Evidence boundary |
| --- | --- | --- |
| G0 | passed | final HEAD/tree, 146-record manifest and verified complete-history bundle |
| G1 | passed | ADR 0001 plus the real second-use Portfolio extension in ADR 0002 |
| G2 | passed | deterministic public assess/lower, localized negative, exact Target and dependency isolation |
| G3 | passed | v10 17-case Corpus Gate, released content-bound Revision lock, reviewed approval and 33-source closure; the scientific-v3 Triton warp-specialized argmin failure is retained as a lowering-ineligible regression |
| G4 | passed for migrated evidence and the bounded attribution slice | common Flash-KMeans/TinyGEMM receipts, r37/r39 parity, full artifact roles, raw r45 replay, and raw-checked no-timing NCU attribution including all correct search survivors in the v18 qualification |
| G5 | passed for Evidence v2 | no-follow CAS, tamper/secret/concurrency tests, fresh-process replay, closed successor matched-event semantics and matched/Portfolio semantic replay pass |
| G6 | passed | 338 tests + 223 subtests cover candidate-set transport/selection/replay, unknown/start/diagnosis event tampering, paired implementation-free reference replacement, exact per-Turn reference-bundle retention and missing-bundle failure, exact reasoning-effort qualification/freezing, all-or-nothing calibrated ordering, every correctness-qualified search survivor's attribution and missing-profile rejection, two-part Estimand availability/missingness, tool-rich scratch lifecycles, artifact-only feedback-budget freezing, immutable CLI reports, Claim-Scope provider routing, external Campaign custody, Lab authority, GPU tutorial custody, preregistered ranking-calibration replay and all three matched Claim Scopes |
| G7 | passed | r4 proves singleton continuity; Codex 0.144.4 closed and tool-rich successors prove the two-arm candidate-set add/update lifecycle |
| G7F | passed | tool-rich v1 injects no feature disables and replays shell activity; its candidate-set successor preserves the single-writer envelope boundary, and artifact v4 replays one measured feedback transition per arm without claiming auxiliary-agent use |
| G8 | passed | r6 passes the singleton pilot; candidate-set v2 covers selected-only attribution, and v3 adds two adhered v18 Runs with ten replayed Receipts and complete correct-search-survivor profiles |
| G9 | passed | private `qhy991/open-cake-ir` main is the source authority; legacy is read-only with pinned rollback |
| G10 | historical projection only | r42 estimand unavailable and r45 bounded timing instability are preserved, not rerun |

## G0 — Legacy source set is durable

**Settles:** whether migration can preserve history without copying live implementation topology.

Require one clean, remotely reachable post-archive revision; repository/tree IDs; source/evidence manifests with
verified digests; explicit secret exclusions; and sufficient disk headroom. Failure blocks implementation and any
legacy cleanup.

## G1 — Product and domain design is coherent

**Settles:** whether `open-cake-ir` has the right product core, dependency direction and canonical owners.

Require acceptance of:

- compiler-first product plus dependent Research Lab;
- Compiler, Lab, Evaluation and Evidence Modules;
- separate Workload and Study Contracts;
- Authoring Environment as treatment;
- Estimand declared before execution and estimate derived afterward for a scientific Study; structurally absent for
  system qualification;
- independent inner kernel loop and governed outer compiler loop;
- `matched_search` plus the evidence-justified closed `portfolio` successor, with no additional runtime mode.
- artifact optimization as a non-scientific Claim Scope on `matched_search`, not another mode.

Any new ambiguity returns to top-level design and blocks further runtime change.

## G2 — Compiler tracer bullet passes

**Settles:** whether the new repository contains an independent compiler rather than an experiment wrapper.

Through the public Compiler Interface, one retained Schedule must produce deterministic assessment and lowering;
one rejected sibling must produce a stable localized Finding. Tests must also prove exact Target mismatch is
rejected, missing analysis/calibration coverage is explicit, and no provider/GPU/Lab/Evidence dependency is loaded.
Failure changes Compiler implementation, not a runner.

## G3 — Compiler Corpus Gate passes

**Settles:** whether the Schedule vocabulary and Compiler Revision generalize beyond one fixture.

Require retained accepted/rejected cases from r16, r25 and r31; deterministic canonical Schedule digests;
expected Findings and lowerings; exact Target definitions; and a content-addressed Compiler Revision. A full corpus
diff must be human-reviewed. Failure returns to Schedule/Target/Compiler design; the revision is not released.

## G4 — Common Evaluation parity passes

**Settles:** whether consolidation preserves scientific assay behavior without leaking arm semantics.

Require:

- sealed launchable artifacts from both Authoring Environments cross the same Evaluation Interface;
- exact r37 correctness dispositions and r39 raw timing summaries replay;
- authored, lowered, compiler-expanded source, PTX, CUBIN and SASS remain distinct artifact kinds;
- correctness always precedes timing and fallback/route counts are explicit;
- one Candidate can receive separate search and confirmatory Evaluation Receipts;
- a Study may add one post-confirmation attribution Receipt whose observed launch passes
  the same external oracle, has no timing, retains the exact profiler output and rejects a
  summary that cannot be recomputed from it;
- measurement quality failure is distinct from candidate disposition.

Failure blocks Lab implementation and deletion of legacy evaluators.

## G5 — Evidence integrity passes

**Settles:** whether every observation and terminal outcome is independently replayable.

Require create-only no-follow writes, CAS rehash, append-only event order, one Terminal Archive schema for success
and failure, deterministic fresh-process replay, secret exclusion, and regeneration of Run Audits after deleting
all reports. A current matched Study must name one closed event vocabulary; replay rejects unknown kinds and derives
selection and diagnoses from retained authorities rather than trusting their projections. Re-auditing the same
Campaign must produce the same canonical Run Audits. Failure blocks live Lab work.

## G6 — Zero-GPU Lab contract tests pass

**Settles:** whether the control plane implements the preregistered study rather than reconstructing it afterward.

Tests through `preflight -> execute -> audit` must prove:

- Campaign Lock references exact Workload/Study/Authoring Environment/provider/compiler/toolchain digests;
- allocation order and globally unique labels are fixed before outcomes;
- initial and resumed turns preserve cwd, sandbox, scaffold, provider and environment;
- provider reasoning effort is explicit, identical across arms and covered by the exact qualification digest;
- clean-start reference replacement is paired across arms, with an incomplete operation-free Schedule starter and
  an empty-body CUDA ABI starter so implementation contamination makes the gate fail;
- v25 retains the exact rendered reference bundle as an Evidence object for every completed Turn; absence records a
  harness fault and missing endpoint, while earlier Executors retain their historical replay boundary;
- success, candidate rejection, protocol fault and contamination all yield Terminal Archives;
- Archive Integrity, Protocol Adherence, Endpoint Observation and Analysis Inclusion remain orthogonal;
- compile/correctness/no-qualified outcomes are observed rather than complete-case deleted;
- provider/harness/custody/broker faults follow preregistered missingness and replacement rules;
- adhered candidate failure contributes an observed qualification endpoint, while an external fault or absent Run
  archive is missing data and cannot enter its denominator;
- Checkpoints distinguish `unreached`, `reached_no_qualified_candidate` and `reached_with_best`;
- successor matched Runs reject unknown events, changed Run starts and diagnoses not derivable from retained facts;
- threshold-crossing Turns cannot backfill earlier Checkpoints;
- a scientific Study Report estimates only the preregistered Estimand and reports availability/uncertainty, while a
  system-qualification report keeps those fields null;
- artifact optimization promotes only a per-Run confirmatory-qualified Candidate and never emits treatment
  statistics or scientific inclusion;
- no source-string assertion substitutes for invoking a public Interface.

Failure fixes the owning Module; it never creates a versioned successor runner.

## G7 — Live two-turn provider qualification passes

**Settles:** whether the real provider CLI honors the frozen Authoring Environment and resume contract.

Use one empty workspace, two Turns, one add then one update, no GPU and no scientific claim. Audit raw lifecycle,
usage, sandbox, cwd, thread continuity, tool surface and reference visibility. Provider/version drift requires a new
Authoring Environment revision and qualification, not a retry under the same Run.

The first two remote attempts remain intact negative archives: r1 observed transient invalidated authentication and
r2 exposed an invalid response schema. r3 passed after the schema repair; r4 is authoritative because it additionally
binds the closed Apps/MCP/browser/shell/subagent feature denylist. See
`inventory/G7_PROVIDER_AUTH_OBSERVATION_20260822.json`.

The closed and tool-rich Codex 0.144.4 successors also pass the same two-Turn, one-file add/update lifecycle for both
Open Cake and direct CUDA candidate-set envelopes. Their qualification records are
`contracts/providers/codex-cli-0.144.4-candidate-set-live-v1.json` and
`contracts/providers/codex-cli-0.144.4-candidate-set-tool-rich-v1.json`. This qualifies provider transport only;
GPU Evaluation remains owned by G8.

## G7F — Live tool-rich provider qualification passes

**Settles:** whether the final engineering optimization loop can expose provider-default features without breaking
Candidate custody or provider-event replay.

The pinned Codex 0.144.3 successor injects no `--disable` flags, binds output schema v2 and the
`tool_rich_candidate_v1` event contract, and observes a no-side-effect shell command on both initial and resumed
Turns. Raw activity and the terminal archive replay with integrity. This attests only observed capabilities under the
effective account/admin policy; it does not authorize external mutation, direct GPU measurement or scientific use.
See `inventory/FULL_FEATURE_PROVIDER_QUALIFICATION_20260823.json`.

The Codex 0.144.4 candidate-set successor additionally freezes the KDA-inspired ownership boundary: auxiliary agents
may investigate read-only, the primary provider thread is the sole envelope writer, and the external Lab alone may
evaluate Candidates. Qualification attests this transport boundary, not the quality of auxiliary-agent reasoning.

Three bounded artifact-optimization Campaigns exercised that boundary beyond the transport probe. v1 promoted one
Open Cake artifact and exposed v14's rejection of Direct CUDA scratch-file lifecycles. v2 promoted one Direct CUDA
artifact and exposed v15's remaining rejection when Open Cake added and updated its fixed envelope in the same Turn.
Executor v16 follows the already-declared authority boundary directly: every tool-rich file-change lifecycle is strict
typed auxiliary activity, while the final no-follow envelope alone submits Candidates. Both earlier faults remain
immutable. Their independent v3 successor has two adhered Runs, each with three authored Candidates, two searches,
one confirmatory promotion and one attribution Receipt. Fresh audit reports integrity, semantic replay,
`artifact_optimization_complete=true` and zero missing Runs under authority `b3d8002b…`; all scientific fields remain
null, so this qualifies the engineering loop without estimating a treatment effect.

## G8 — Bounded end-to-end pilot passes

**Settles:** whether Compiler, Lab, Evaluation and Evidence compose through the canonical path.

One small Run per Authoring Environment must pass clean-checkout preflight, execute, terminal seal and independent
offline audit under the `system_qualification_only` Claim Scope. Each Run must contain at least one Evaluation
Receipt accepted by semantic replay; a build-only rejection does not qualify the end-to-end path. The report may
set `system_qualification_passed`, but `estimand`, `estimate` and `uncertainty` remain null and
`estimand_available` remains false regardless of observed kernel outcomes. Comparative statistics and pooling are
forbidden. This is system qualification only and cannot estimate the paper treatment effect.

Remote r6 passes this gate. Both Runs are adhered, independently replay to search and confirmatory receipts, and
retain actual GPUQ job identities. Their provider-token totals were below the sole 80k checkpoint, so Run endpoints
remain `missing`; that is intentionally irrelevant to system qualification and cannot create a scientific estimate.
See `inventory/G8_SYSTEM_QUALIFICATION_20260822.json`.

Candidate-set v2 passes the same gate through Executor v14 and the frozen Study
`contracts/studies/matched-search-candidate-set-system-v2.json`. Each arm submits three non-deduplicated launchable
Candidates, searches two, and evaluates its selected Candidate once confirmatorily and once for attribution. The
eight receipts all pass the frozen tie-aware oracle with one target launch and zero fallback. Independent audit of
`/home/qinhaiyan/open-cake-ir-evidence/campaigns/candidate-set-campaign-live-v2` reconstructs integrity, semantic
replay, adherence and `system_qualification_passed=true` under Campaign authority
`88605aa1b1557b58aec9dacb4e043290d9a03205b0847d507fa72fc68fad633b`. Open Cake ends below the 80k checkpoint and
direct CUDA crosses it on the non-backfillable Turn, so their endpoint observations differ; both remain excluded
from scientific analysis and all estimand fields remain null.

Candidate-set v3 passes the current Executor-v18 protocol through frozen Study
`contracts/studies/matched-search-candidate-set-system-v3.json` (canonical SHA
`180c0d4787c4f2361e5fe7bd342fbb7e07def6ca0f0e5a7551993d8fe4364b7e`). Under Campaign
authority `56bbcfe06caa4be7db31689d658ed59af3e2fe0ae2b4a88dc682ef9c8b373d8e`, each arm
authors three launchable Candidates, searches two correct survivors, profiles both and
freshly confirms the selected survivor. The ten Receipts—two search, two attribution and
one confirmatory per arm—each observe one target launch and zero fallback; the four
attribution Receipts retain eleven checked NCU metrics and structurally no timing. Fresh
audit of `/home/qinhaiyan/open-cake-ir-evidence/campaigns/candidate-set-campaign-live-v3`
passes integrity, exact semantic replay and adherence with
`system_qualification_passed=true`. Open Cake remains below the turn-discrete checkpoint
and direct CUDA crosses it before confirmation can backfill the endpoint; scientific
inclusion remains forbidden and every estimand field remains null.

## G9 — Cutover is safe

**Settles:** whether `open-cake-ir` can become the sole runnable owner.

Require remote revision and Compiler Revision anchors, zero dual-writes, legacy-manifest coverage, generated-status
parity, disk-capacity gate, rollback to a pinned legacy checkout and explicit user approval. Only then may a full
matched-search Study Contract be frozen.

The private GitHub repository `qhy991/open-cake-ir` now anchors the independent `main` revision. The legacy tree is
retained only for historical replay and rollback; it is not a second writer. Capacity, cache exclusions, Evidence
custody, status parity and the pinned rollback procedure remain part of every later cutover audit.

## G10 — Scientific completion is claim-specific

**Settles:** whether a completed Campaign supports its declared Estimand.

Require every prescheduled Run Audit, preregistered inclusion/missingness handling, both parts of the Run endpoint,
estimate and uncertainty, contamination audit, explicit unavailable classifications and a scope-limited Claim View.
Portfolio and Serving require later independent gates; fixed-shape success cannot satisfy them.

Scientific v3 is the first completed local application of this gate. Fresh audit passes archive integrity and exact
semantic replay, but one provider fault and one harness fault leave two Runs missing. Direct CUDA qualifies in 2/3
Runs, Open Cake in 0/3, and the preregistered Estimand remains unavailable; the gate therefore reports no estimate or
uncertainty. Executor v21 and Compiler v10 repair the observed mechanisms only for successor Studies. Executor v22
also replaces the internally contradictory all-Runs-qualified availability rule for future Studies: a candidate
failure remains an observed qualification outcome, the complete two-part Estimand needs zero missing Runs and at
least one conditional latency per arm, and the estimate includes its declared qualification-rate difference. The
legacy v3 plan, Runs and report remain unchanged. Executor v23 preserves that Estimand and closes the semantic event
vocabulary only for successor matched Studies; it does not manufacture additional historical endpoint evidence.
Executor v24 makes reasoning effort a Study-owned, live-qualified factor so a future successor can use the paper's
`xhigh` without treating the distinct frozen `max` level as an alias.
Executor v25 retains the exact reference bytes supplied to each provider Turn. This makes future post-run reference
audit possible; it does not backfill the input bytes absent from earlier Campaigns or certify their contents by hash.
