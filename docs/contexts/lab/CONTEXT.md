# Research Lab context

The Research Lab context owns study design and execution while treating compiler and workload revisions as frozen inputs.

## Language

**Workload Contract**:
An immutable definition of operator semantics, input domain, oracle, tolerances and correctness cases.
_Avoid_: Benchmark config, task prompt

**Study Contract**:
A frozen execution authority containing Authoring Environments, allocation, budget and data-use rules. A scientific
Study declares an Estimand; a system-qualification Study explicitly declares none.
_Avoid_: Experiment policy, campaign config

**Claim Scope**:
The content-bound data-use policy that permits system qualification, artifact optimization or scientific Study
Analysis, never more than one for the same Campaign.
_Avoid_: Report label, README disclaimer

**Provider Feature Policy**:
The Authoring Environment rule for feature overrides and provider-event interpretation. The closed research policy
injects the exact historical denylist; provider-default optimization injects no disables and uses the tool-rich event
contract. It describes exposure, not authority for external side effects.
_Avoid_: Claim Scope, tool availability claim

**Authoring Environment**:
The complete assigned environment through which an agent authors, receives diagnostics and produces sealed candidates.
_Avoid_: Representation, arm code path

**Campaign**:
One execution instance of a Study Contract under a resolved machine and software admission.
_Avoid_: Study, experiment round

**Run**:
One independently assigned replicate and the experimental unit of a comparative study.
_Avoid_: Session, GPU job

**Turn**:
One provider interaction within a Run and its resulting candidate submission opportunity.
_Avoid_: Round, checkpoint

**Candidate**:
An immutable authored artifact identified by content and lineage.
_Avoid_: Workspace file, current candidate

**Artifact Promotion**:
The per-Run selection of one confirmatory-qualified Candidate by declared latency rank and tie-break. It never forms
an arm comparison and does not itself create a KernelSeed.
_Avoid_: Estimand, winner, treatment effect

**Evaluation Receipt**:
An append-only observation of one Candidate under a named evaluation purpose, protocol and environment admission.
_Avoid_: Candidate state, score

**Endpoint**:
A preregistered Run-level outcome used by a Study Analysis.
_Avoid_: Best log line, checkpoint median

**Estimand**:
The target quantity defined by unit, population, endpoint, contrast and intercurrent-event strategy.
_Avoid_: Estimate, result

**Checkpoint**:
A longitudinal Run view at a budget that is unreached, reached without a qualified candidate, or reached with a best candidate.
_Avoid_: Replicate, future projection

**Kernel Seed**:
A fixed-cell candidate accepted by the confirmatory endpoint protocol but not yet validated as a library family.
_Avoid_: Production kernel, generalization proof

**Portfolio Artifact**:
The sole immutable exact-key mapping from frozen Workload cases to sealed specialist candidates plus fail-closed
dispatcher policy.
_Avoid_: Shape registry, serving library

**Measurement Quality**:
Per-boundary stability derived from retained raw cohorts, independent of candidate correctness.
_Avoid_: Candidate status, performance claim

## Relationships

- A **Study Contract** references one **Workload Contract** and assigns every **Run** to one **Authoring Environment**.
- A **Campaign** executes a Study Contract; multiple Campaigns are not pooled unless its analysis rule allows it.
- A **Run** contains ordered **Turns** and may produce multiple immutable **Candidates**.
- A **Candidate** may have multiple **Evaluation Receipts**, including search and confirmatory purposes.
- A scientific **Study Contract** declares an **Estimand** before execution; a later analysis only estimates it.
- A `system_qualification_only` **Claim Scope** has exactly one Run per Authoring Environment, forbids comparative
  statistics and pooling, and keeps Estimand, estimate and uncertainty structurally absent.
- An `artifact_optimization_only` **Claim Scope** has one multi-Turn Run per Authoring Environment and may perform
  only per-Run **Artifact Promotion**; all scientific inclusion remains false.
- A **Provider Feature Policy** belongs to the Authoring Environment, while Claim Scope owns only data use.
- A **Kernel Seed** may be consumed by the closed `portfolio` Study variant but does not imply arbitrary-shape
  generalization.

## Example dialogue

> **Researcher:** “This Run reached 100k tokens but produced no correct candidate. Is its endpoint missing?”
>
> **Statistician:** “No. That is an observed treatment outcome; only an external protocol fault creates missingness.”

## Flagged ambiguities

- “representation effect” is resolved to **Authoring Environment effect** unless a separate mechanism ablation exists.
- “best candidate” is not an Endpoint until a confirmatory Evaluation Receipt applies the preregistered rule.
- A system-qualification Study Report may qualify the execution path but can never expose a scientific estimate.
