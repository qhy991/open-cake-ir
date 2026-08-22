# Top-level design

Status: accepted product shape; Compiler and two Study variants are implemented. Closed G7, tool-rich G7F and GPU
G8 qualifications pass; the private GitHub main revision completes sole-owner cutover.

## 1. Irreducible goal

Build an independently usable Cake-like compiler whose typed schedules, target contracts, localized findings and
lowering are inspectable and corpus-gated; then use a dependent Research Lab to test the complete authoring
environment through preregistered, replayable studies.

The system must answer two different questions without conflating them:

1. **Compiler question:** can a hardware schedule be expressed, assessed and lowered correctly under an exact
   Compiler Revision and Target?
2. **Study question:** under a fixed Workload Contract and matched design, what Run-level outcomes arise from
   assignment to an Open Cake versus direct CUDA Authoring Environment?

## 2. Non-goals

- Claiming the unpublished CAKE compiler or its reported 80M-token result has been reproduced.
- Designing the compiler around one Flash-KMeans campaign runner.
- Mutating a Compiler Revision during a Campaign.
- Treating syntax alone as the experimental treatment.
- Using one generic workflow DAG for fixed-shape search, portfolio construction and serving.
- Preserving `rXX`/`vN` implementation history as active source.
- Inferring missing endpoints, future checkpoints or plateau from incomplete Runs.

## 3. System map

```text
Standalone compiler path
------------------------
Schedule + Target + CompilerRevision
        -> Compiler.assess
        -> Assessment(findings, coverage, lowering eligibility)
        -> Compiler.lower
        -> Lowering(target source, source map, toolchain requirements)

Research path
-------------
WorkloadContract + StudyContract
        -> Lab.preflight
        -> CampaignLock(resolved immutable closure)
        -> variant-owned assigned Runs
             -> Provider
             -> AuthoringEnvironment Adapter
                  |-- OpenCakeEnvironment -> Compiler -> toolchain
                  `-- DirectCudaEnvironment -> CUDA toolchain
             -> sealed Candidate / optional LaunchableCandidate
             -> common Evaluation
             -> immutable Evidence
             -> bounded same-Run feedback or terminal
        -> Lab.audit
             -> RunAudit(integrity, adherence, endpoint observations)
             -> StudyReport(inclusion, estimate, uncertainty, availability)
             -> derived ClaimView
```

```text
Portfolio path
--------------
frozen KernelSeed + Workload case roles + CompilerRevision
        -> exact-shape Schedules -> distinct Lowerings / LaunchableCandidates
        -> PortfolioArtifact (sole semantic-key mapping)
        -> direct / dispatcher / postflight Evaluation
        -> correctness + per-boundary MeasurementQuality + bounded ClaimView
```

The dependency direction is strict:

```text
Lab -> Compiler
Lab -> Evaluation -> Evidence
Compiler -X-> Lab / Evaluation / Evidence / Provider / Workload
```

## 4. Four deep Modules

### 4.1 Compiler Module

The Compiler is the product core. Its Interface is independent of experiments:

```text
assess(Schedule, Target, CompilerRevision) -> Assessment
lower(eligible Assessment) -> Lowering
```

The Implementation hides Schedule parsing and typing, exact Target admission, verifier rules, analysis coverage,
localized Findings, cost-model coverage and deterministic lowering. `lower` never executes a GPU workload and
never records campaign state. Unsupported capability or absent calibration is explicit, not silently inherited.

A Compiler Revision owns the canonical Schedule semantics, Target definitions, verifier, analysis, lowering and
calibration references. A proposed revision becomes usable only after the full Corpus Gate and human merge.

### 4.2 Lab Module

The Lab owns study design resolution and Campaign control:

```text
preflight(WorkloadContract, StudyContract) -> CampaignLock
execute(CampaignLock) -> CampaignRef
audit(CampaignRef) -> StudyReport
```

Its Implementation hides allocation, Run/Turn lifecycle, provider invocation and resume equivalence, feedback
projection, budget accounting, checkpoint state, stopping, missingness and Study Analysis. Every side effect
rechecks dynamic provider, checkout and machine admission; preflight is not a lease on future external state.

The Lab has two real Authoring Environment Adapters:

- `OpenCakeEnvironment`: provider/scaffold/tool surface plus one frozen Compiler Revision and structured findings.
- `DirectCudaEnvironment`: the matched provider/scaffold with direct CUDA/PTX source and its toolchain feedback.

This Seam represents the complete assigned treatment bundle, not a source-file extension.

Provider variation is a second real Seam because Codex and DeepSeek already exist. Provider revisions are contract
inputs and are never pooled implicitly. GPUQ remains an internal dependency until a second execution Adapter is
actually retained.

### 4.3 Evaluation Module

Common Evaluation begins only after an Authoring Environment has produced an immutable LaunchableCandidate:

```text
evaluate(LaunchableCandidate, WorkloadContract, EvaluationProtocol, purpose)
    -> EvaluationReceipt
```

It owns input materialization, the external oracle, correctness, exact launch accounting, measurement quality and
timing. It does not own Cake/CUDA build semantics, treatment assignment, candidate lineage, Study Analysis or
evidence storage. Flash-KMeans and TinyGEMM2 justify explicit Workload Adapters; use a closed mapping, not dynamic
plugin discovery.

A Candidate may have multiple append-only Evaluation Receipts, uniquely identified by candidate digest,
Evaluation Protocol digest, environment admission and purpose. Search measurements guide the inner loop;
confirmatory measurements determine the endpoint. This prevents adaptive best-of-noise selection from becoming
the final claim silently.

Measurement quality is separate from candidate disposition. A noisy or unstable timing observation does not mean
the candidate itself is semantically unstable.

### 4.4 Evidence Module

The Evidence Module is the only durable observation writer. Its Interface stores immutable bytes, appends ordered
events and replays a Campaign without provider, GPU, network or checkout writes. The Implementation hides CAS
layout, no-follow writes, manifests, ledger chaining, terminal sealing and rehash.

Every Run produces one Terminal Archive, whether successful, externally failed or protocol-invalid. There is no
separate success archive and failure archive. Evidence never defines an Estimand; it supplies Run Audits to the
Lab's preregistered Study Analysis.

## 5. Canonical owners

| Fact | Canonical owner | What is only a reference or projection |
| --- | --- | --- |
| Schedule vocabulary and semantics | Compiler Revision | examples, docs, generated source |
| Exact target capabilities | Target inside Compiler Revision | GPU name, backend flag |
| Findings and analysis coverage | Compiler Revision + Assessment | stderr, prompt feedback |
| Lowering lineage | Lowering record | generated filename |
| Compiler release eligibility | Corpus manifest + Corpus Gate report + human merge | test count, Git message |
| Operator semantics, input domain and oracle | Workload Contract | task prompt, benchmark script |
| Unit, treatment, endpoints and Estimand | Study Contract | Campaign name, report narrative |
| Provider/scaffold/compiler/toolchain treatment | Authoring Environment revision referenced by Study Contract | command line |
| One execution authorization | Campaign Lock | mutable environment variables |
| Candidate bytes and lineage | Evidence Objects + Event Ledger | workspace copy, registry cache |
| Evaluation observation | Evaluation Receipt | feedback summary, median only |
| Archive Integrity and Protocol Adherence | Run Audit | process exit code alone |
| Estimate, uncertainty and availability | Study Report under the Study Contract's Analysis Plan | README status |
| Current supported claims | derived Claim View | Roadmap, changelog prose |
| Secret bytes | external host custody | key ID and non-secret custody receipt |

`CampaignLock` does not become a competing semantic owner. It resolves the exact digests and environmental
admission needed to execute one Study Contract. A Study Contract may authorize multiple Campaigns; pooling is
forbidden unless its Analysis Plan states how and why.

## 6. Contract model

### Workload Contract

Owns operator semantics, declared shape/input domain, data types, deterministic input plan, oracle, tolerances,
correctness cases and launch-level output contract. It is reusable across studies and does not contain provider,
budget or causal claims.

### Study Contract

The first implemented closed variant is `matched_search`. It references a Workload Contract and always owns:

- experimental unit: independent Run;
- assigned Authoring Environment revisions;
- budget grid, Run Protocol and Evaluation Protocol;
- protocol-deviation, missingness, replacement, quorum and inclusion rules;
- a content-bound Claim Scope and its Analysis Plan.

Claim Scope then closes the data policy. A scientific `matched_search` Study additionally owns its target population,
primary and secondary Run-level endpoints, balanced/randomized allocation, Estimand, contrast, intercurrent-event
strategy and any prespecified pooling rule. A `system_qualification_only` Study instead has exactly one Run per
Authoring Environment, declares no Estimand, forbids pooling and comparative statistics, and asks only whether every
prescheduled Run is intact, adhered, semantically replayable and contains an Evaluation Receipt.

An `artifact_optimization_only` Study also reuses `matched_search`, with one multi-Turn Run per Authoring
Environment. Its Authoring Environment injects no Codex feature disables and admits tool-rich provider events, but
its Analysis Plan can only promote one confirmatory-qualified Candidate per Run. It has no Estimand, arm comparison,
pooling or scientific inclusion. Provider feature exposure is independent of Claim Scope: it changes authoring
capability, never the authority of common Evaluation or Evidence.

For the initial scientific matched-search reconstruction, the primary endpoint is explicitly two-part at budget `B`:

1. whether a confirmatorily qualified candidate exists by `B`;
2. best confirmed speedup by `B`, conditional on qualification.

The scientific Study Report presents both rather than deleting Runs without a qualified candidate or assigning an
arbitrary performance value. It requires at least three Runs per arm; three Runs per arm support only a descriptive
contrast for the pinned setup, not a broad population claim. These endpoint and replicate rules do not apply to the
non-scientific G8 system qualification.

`tool_surface` in each arm names the Candidate submission Interface (`submit_schedule` or `submit_cuda`), not the
provider's auxiliary Apps/MCP/shell/browser/subagent catalog. The content-bound Provider Feature Policy owns that
catalog and event contract.

The second implemented closed variant is `portfolio`, justified by the real r43-r45 use case. It owns one
pre-held-out KernelSeed, one anchor plus two held-out Workload cases, exact-shape/no-retune specialization, a
manifest-owned dispatcher map, direct/dispatcher/postflight assays, fixed 5×25 kernel and dispatcher cohorts, CV
quality, an unsupported-key probe and a no-resampling descriptive Analysis Plan. It is data on the same Lab path,
not a new Module or execution mode, and it cannot repair a missing `matched_search` cell.

`Run Protocol`, `Evaluation Protocol` and `Analysis Plan` remain named sections of Study Contract until a real
second reuse case proves they need independent owners.

## 7. Run and checkpoint semantics

Run is the independent experimental unit. Turn, Candidate, Evaluation Receipt and Checkpoint are repeated
observations within a Run and never increase replicate count.

Each Checkpoint has exactly one of three states:

- `unreached`: the Run did not attain the budget; endpoint data are missing at that checkpoint;
- `reached_no_qualified_candidate`: the budget was observed and failure to qualify is a treatment outcome;
- `reached_with_best(candidate_digest)`: the budget was observed and the best eligible candidate is named.

A provider Turn completing beyond a checkpoint cannot backfill that smaller checkpoint. Plateau is an offline
diagnostic only until the paper's prespecified criterion becomes public.

Four orthogonal audit facts prevent status explosion:

| Fact | Examples |
| --- | --- |
| Archive Integrity | complete, missing object, digest mismatch, broken event order |
| Protocol Adherence | adhered, provider fault, harness fault, custody violation, contamination |
| Endpoint Observation | qualified, no qualified candidate, missing, measurement quality failure |
| Analysis Inclusion | included, excluded by preregistered rule, estimand unavailable |

Compile rejection, correctness rejection and reaching a budget without a qualified candidate are observed
treatment outcomes. Provider, broker, harness or custody faults are protocol deviations/missingness. Structural
audit success never implies inclusion in the Estimand.

## 8. Two evolution loops

### Inner loop: kernel evolution

```text
fixed CompilerRevision + fixed Study/Workload Contracts
  -> author Candidate
  -> assess/lower/build
  -> common Evaluation
  -> immutable evidence
  -> bounded same-Run feedback
  -> next Turn or terminal
```

### Outer loop: compiler evolution

```text
corpus or runtime evidence
  -> compiler change proposal
  -> coordinated IR / Target / verifier / analysis / lowering / calibration change
  -> full Corpus Gate
  -> human merge
  -> new CompilerRevision
  -> future Campaign only
```

The outer loop is a governed development process, not a runtime mode. An active Campaign can report a capability
gap but cannot install a new primitive or reinterpret earlier Candidates.

Clean-start provenance covers Compiler Revision, Corpus, scaffold, memory, diagnostics and the frozen reference bundle
embedded in every Turn prompt—not
only the agent workspace. A compiler evolved from the same task history is an explicit treatment prior. Such a
study may estimate package effectiveness but cannot claim task-naive discovery without a stronger firewall.

## 9. Claim-stage handoffs

Fixed-shape `matched_search` and its real successor handoff into `portfolio` are implemented.

```text
FixedShapeStudy -> sealed KernelSeed
KernelSeed + frozen case roles -> PortfolioStudy -> sealed PortfolioArtifact
PortfolioArtifact -> future ServingStudy -> system-level evidence
```

Portfolio is now the second closed Study Contract variant, still not a Module or runtime mode. Its current contract
is deliberately limited to the three actual r45 shapes; arbitrary populations, weighting and dynamic fallback remain
unsupported. Serving is still future work and requires a real framework, workload traces, concurrency policy and
no-profiler end-to-end endpoints.

## 10. Source shape

```text
src/open_cake_ir/
  compiler/                 # Schedule, Target, Finding, assessment, lowering, revision
  lab/                      # Study resolution, Campaign, Run/Turn, providers, arm environments
  evaluation/               # common oracle, correctness, timing, workload adapters
  evidence/                 # CAS, ledger, terminal archive, replay
corpus/                     # accepted/rejected Schedule observations and gate manifests
contracts/
  workloads/                # reusable Workload Contracts
  studies/                  # preregistered Study Contracts
  kernel-seeds/             # frozen pre-held-out kernel choices
  providers/                # provider capability receipts
tests/contracts/            # tests cross public Interfaces
migration/legacy_manifest.jsonl
```

Generated Campaigns live outside source and contain a resolved `campaign.lock.json`, event ledger, content-addressed
objects and deletable reports.

The public command families mirror the two products:

```text
open-cake-ir compiler assess|lower
open-cake-ir lab preflight|execute|audit
```

## 11. Concepts deliberately removed

- A campaign-centric definition of the entire product.
- A generic `RepresentationAdapter` hiding Compiler semantics.
- One monolithic Experiment Contract owning workload, study and execution facts.
- Candidate limited to one evaluation.
- Audit inventing an Estimand after execution.
- Success-only and failure-only archive paths.
- Combined states such as `compile_rejected_no_retry`.
- Per-round Policy, runner, verifier and auditor Modules.
- Runtime compiler self-modification.
- Generic DAG, dynamic plugin registry, speculative multi-backend/serving abstractions.
- r41/r42 and r43/r44/r45 successor wrappers as active code; stable semantics are folded into one current path.

By the deletion test, these concepts add caller knowledge without protecting a canonical fact. Historical behavior
remains recoverable from pinned Git and immutable evidence rather than surviving as active architecture.
