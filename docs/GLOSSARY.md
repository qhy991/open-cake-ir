# Canonical glossary and claim language

This file owns the repository's human-readable terminology. Context documents explain
relationships; Contracts and released descriptors own exact machine semantics. A term
may be referenced anywhere, but its definition must not be independently restated.

Capitalization is meaningful when naming a domain concept. Lowercase prose is ordinary
English unless it links back to one of these concepts.

## Project names

### CAKE

The system described by the cited paper. The public paper does not make its unpublished
implementation part of this repository. Avoid using **CAKE** for a local result.

### open-cake-ir

This repository and independent reconstruction project. Use the lowercase, hyphenated
spelling for the repository or package.

### Cake IR

The hardware-explicit authoring representation used by an Open Cake Authoring
Environment. A Cake IR result requires a Schedule handled by a released Open Cake
Compiler; direct Triton source is not a Cake IR result.

### Open Cake Compiler

The independently usable product core in this repository. It assesses and lowers typed
Schedules without depending on the Research Lab, a provider, a Workload, GPU execution,
or Evidence storage.

## Compiler terms

### Schedule

Owner: Compiler Revision.

A typed, hardware-explicit declaration of roles, resources, operations, dependencies,
accesses, and pipeline decisions. Avoid **config**, **program** when only one Schedule is
meant, and **IR candidate**.

### Target

Owner: Target definition inside a Compiler Revision.

An exact architecture contract containing supported instructions, resources, and
synchronization. It is not a physical GPU allocation, a GPU marketing name, or a backend
flag.

### Finding

Owner: Compiler rule evaluated in an Assessment.

A localized statement that a Schedule satisfies, violates, or lacks modeled coverage for
a compiler rule. Avoid **error string** and **diagnostic blob**.

### Assessment

Owner: Compiler Revision.

The immutable Findings, modeled analysis, and lowering eligibility for one Schedule and
Target. It is not Workload correctness, GPU validation, or Lab preflight.

### Lowering

Owner: Compiler Revision.

The deterministic derivation of inspectable target source and source mapping from an
eligible Assessment. Lowering is not CUDA/Triton toolchain compilation, CUBIN loading, or
GPU execution.

### Compiler Revision

Owner: released compiler lock.

A content-bound combination of Schedule semantics, Target definitions, verifier,
analysis, lowering, Corpus expectations, and released calibration coverage. Avoid
**current compiler** and **Git HEAD** as identities.

### Corpus

Owner: Corpus manifest referenced by a Compiler Revision.

A versioned set of accepted and rejected Schedules with expected compiler observations.
The Corpus is not a collection of examples and a generic unit-test pass is not a Corpus
Gate.

### Corpus Gate

Owner: gate report plus external release approval.

The full-corpus review boundary for a proposed Compiler Revision. It proves agreement
with reviewed expectations; it does not prove GPU correctness or performance.

### Calibration

Owner: a target- and revision-bound calibration contract and its accepted measurements.

Evidence authorizing a bounded pre-GPU estimate or ordering decision. Calibration never
transfers implicitly between Targets, shapes, or Compiler Revisions.

## Workload and Research Lab terms

### Workload Contract

Owner: the content-bound Workload Contract.

The operator or Program semantics, input domain, materialization, oracle, tolerances,
correctness cases, and output contract. Avoid **benchmark config** and **task prompt**.

### Study Contract

Owner: the content-bound Study Contract.

The experimental design: Authoring Environments, allocation, budget, endpoints,
data-use rules, and, when scientific, a preregistered Estimand. It is not one execution
or a mutable campaign configuration.

### Claim Scope

Owner: Study Contract.

The closed data-use policy permitting exactly one of system qualification, artifact
optimization, or scientific analysis for a Campaign. A README label cannot widen it.

### Provider Feature Policy

Owner: Authoring Environment revision.

The exact provider feature exposure and provider-event interpretation. It describes what
the author can use; it does not authorize external side effects or determine Claim Scope.

### Authoring Environment

Owner: Study Contract references to frozen environment definitions.

The complete assigned environment through which an agent authors, receives diagnostics,
and submits sealed Candidates. The matched treatment is the complete Authoring
Environment, not syntax alone.

### Campaign

Owner: CampaignLock.

One execution instance of a Study Contract under resolved software, machine admission,
and Evidence custody. A Campaign is not a Study definition or an individual Run.

### CampaignLock

Owner: Lab preflight output.

The exact, immutable execution authorization resolving one Study Contract to released
Compiler, Executor, provider, toolchain, machine, and custody inputs. Use the spelling
`CampaignLock` in both code-oriented and prose references; do not alternate with
**Campaign Lock** as a second term.

### TaskPackage

Owner: deterministic Lab renderer under one CampaignLock.

The complete Agent-facing pair `TASK.md` and `AGENTS.md` for one Run. `TASK.md` projects
the frozen task authority; `AGENTS.md` projects stable behavior and tool rules. Neither
file owns mutable results, budgets consumed, frontier state, or Evaluation disposition.

### Ralph Controller

Owner: Lab Run control.

The external same-thread loop controller that admits an iteration, supplies a derived
StateCard, accounts for time/token/Turn/Evaluation budgets, and records a terminal reason.
It does not author Candidates or decide correctness and performance.

### StateCard

Owner: Ralph Controller projection from retained Run state.

The per-iteration JSON view of consumed and remaining budgets, prior bounded feedback,
and terminal status. It is evidence input to the next provider Turn, not task policy and
not a model-written summary.

### Run

Owner: Study Contract and Campaign execution.

One independently assigned replicate and the experimental unit of a comparative Study.
It is not a provider session, Turn, or broker job.

### Turn

Owner: Run protocol.

One provider interaction within a Run and its Candidate submission opportunity. Avoid
**round** and **checkpoint**.

### Candidate

Owner: Evidence Objects and Event Ledger after submission.

An immutable authored artifact identified by content and lineage. A mutable workspace
file is not yet a Candidate.

### Artifact Promotion

Owner: non-scientific artifact-optimization Study analysis.

Per-Run selection of one confirmatory-qualified Candidate by the declared latency and
tie-break rule. It is not an arm comparison, treatment effect, or production deployment.

### Endpoint

Owner: Study Contract.

A preregistered Run-level outcome consumed by Study analysis. A best log line or an
unqualified checkpoint median is not an Endpoint.

### Estimand

Owner: scientific Study Contract.

The target quantity defined by experimental unit, population, endpoint, contrast, and
intercurrent-event strategy. It is not an estimate or an observed result.

### Checkpoint

Owner: Run protocol.

A longitudinal view at a declared budget: unreached, reached without a qualified
Candidate, or reached with a qualified Candidate. It does not project future work.

### KernelSeed

Owner: Lab.

A fixed-cell Candidate accepted by its confirmatory endpoint protocol and eligible for a
separate portfolio Study. It is not a production kernel or generalization proof.

## Evaluation terms

### LaunchableCandidate

Owner: Evaluation boundary.

A Target, entry point, launch manifest, and complete role-labelled artifact byte set
ready for common evaluation. An arm output filename alone is insufficient.

### Evaluation Protocol

Owner: Study Contract reference to Workload-owned assays.

The frozen case, purpose, correctness-before-timing order, and measurement boundary. It
does not define Study analysis.

### Logical Evaluation Attempt

Owner: Evaluation.

One immutable Candidate evaluation retaining every broker job and the bounded zero-work
resubmission rule. It is not a retry mode or replacement Candidate.

### Evaluation Receipt

Owner: Evaluation.

The append-only correctness output, launch receipt, samples, and derived disposition for
one Candidate and evaluation purpose. It is not Candidate state, a score, or a Study
result.

### Portfolio Artifact

Owner: Evaluation.

The sole immutable mapping from frozen semantic keys to sealed specialist Candidates,
including fail-closed dispatch policy. It is not a mutable shape registry or serving
library.

### Measurement Quality

Owner: Evaluation.

Stability derived from retained raw cohorts for one measurement boundary, independent
of correctness. Timing instability does not make a Candidate incorrect.

## Evidence and report terms

### Evidence Object

Owner: Evidence store.

An immutable byte sequence identified by content. A path or output filename is only a
reference to it.

### Event Ledger

Owner: Evidence store.

An append-only ordered record of Evidence Object references and observed transitions. It
is not a mutable state file or an untyped log directory.

### Terminal Archive

Owner: Evidence store.

The complete replayable evidence for a Run regardless of success, failure, or protocol
deviation. There is no separate success and failure archive type.

### Integrity

Owner: Run Audit.

Whether archived bytes, references, and event ordering are complete and untampered.
Integrity is not scientific validity, correctness, or filesystem custody.

### Filesystem Custody

Owner: Run Audit under the live custody policy.

Whether audited paths have the owner and mode properties required by the single-writer
store. Present mode bits cannot prove continuous historical custody.

### Protocol Adherence

Owner: Run Audit.

Whether observed execution followed the frozen Study Contract and Campaign admission.
It is independent of archive Integrity and Candidate quality.

### Run Audit

Owner: Evidence replay.

A pure reconstruction of Integrity, Filesystem Custody, Protocol Adherence, and Endpoint
observations for one Run. It is not a Study result.

### Study Report

Owner: the Study Contract's Analysis Plan applied to accepted Run Audits.

A preregistered analysis containing inclusion, estimate, uncertainty, and availability.
It may estimate only an already declared Estimand.

### Claim View

Owner: derived projection from accepted Study Reports.

A deletable, reproducible view of supported claims and explicit unknowns. It never owns a
claim and must not be maintained independently in the README.

## Evidence strength and scope

These labels form boundaries, not a single automatic promotion ladder. A stronger label
requires its own named evidence.

| Label | What it establishes | What it does not establish |
| --- | --- | --- |
| `proposed` | A reviewed change may be implemented and gated | Released Compiler behavior or GPU evidence |
| `released` | The exact authority passed its release process | Workload correctness or performance |
| `structurally_expressible` | The vocabulary can state the mechanism | Backend lowering, correctness, or full-program coverage |
| `lowerable` | Assessment permits deterministic target-source emission or checked asset selection | Toolchain compilation or GPU execution |
| `compiled` | The frozen toolchain produced the target artifact | Correctness or speed |
| `correctness_qualified` | The named external oracle and cases accepted the artifact | Stable timing, profiler diagnosis, or broader shapes |
| `timed` | Valid samples exist for the named boundary | Attribution or end-to-end serving improvement |
| `profiled` | Correctness-qualified profiler evidence exists | Profiler-free speed or causal attribution by itself |
| `operator` | Evidence covers the named operator or fixed Program boundary | Model forward or serving behavior |
| `model_forward` | Evidence covers the declared model-forward path | Serving throughput, latency, or availability |
| `serving_e2e` | Evidence covers the named serving endpoint and complete request/token trajectory | Other deployments or workloads |
| `not_run` | The named evaluation was deliberately not executed | Failure |
| `unknown` | Available evidence cannot decide the claim | Negative outcome |
| `invalid` | The evidence or protocol cannot support the intended interpretation | A valid negative Candidate result |

Absolute latency always names its boundary: component, kernel, operator, complete Program,
model forward, or serving endpoint. Speedup without the matched absolute values and
boundary is incomplete.
