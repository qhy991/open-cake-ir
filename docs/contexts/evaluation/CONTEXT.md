# Evaluation context

[中文阅读](../../zh-CN/contexts/evaluation/CONTEXT.md) · [Bilingual catalog](../../README.md)

Evaluation owns the common assay after an Authoring Environment seals a Candidate.
Canonical definitions are in the [`Glossary`](../../GLOSSARY.md#evaluation-terms).

## Owned terms

- [Launch Plan](../../GLOSSARY.md#launch-plan)

- [`LaunchableCandidate`](../../GLOSSARY.md#launchablecandidate)
- [`Evaluation Protocol`](../../GLOSSARY.md#evaluation-protocol)
- [`Logical Evaluation Attempt`](../../GLOSSARY.md#logical-evaluation-attempt)
- [`Evaluation Receipt`](../../GLOSSARY.md#evaluation-receipt)
- [`Portfolio Artifact`](../../GLOSSARY.md#portfolio-artifact)
- [`Measurement Quality`](../../GLOSSARY.md#measurement-quality)

## Responsibilities

- Require every Authoring Environment to cross the same LaunchableCandidate boundary.
- Materialize Workload cases and apply the external oracle and tolerances.
- Enforce correctness before timing or profiler collection.
- Keep search, confirmatory, and profiler purposes as separate Evaluation Receipts.
- Retain raw timing cohorts, launch/fallback counts, and profiler output needed for replay.
- Reject unsupported portfolio keys before launch.

Evaluation does not define the Estimand, Claim Scope, Study inclusion, or Claim View. It
records what happened at the declared assay boundary.

## Relationships

- The Workload Contract supplies semantics, case materialization, oracle, and tolerances.
- RunSpecification supplies exact execution admission; CampaignLock adapts legacy inputs.
- Evidence stores the Candidate and Receipt bytes after Evaluation observes them.
- Study analysis consumes Run Audits, not an evaluator's headline number.

## Boundary examples

Timing CV failure is Measurement Quality evidence, not Candidate incorrectness. A
correctness-qualified profiler launch supplies diagnosis but no profiler-free latency
sample. “Source” must name its role—authored source, lowered source, expanded source, PTX,
CUBIN, or SASS—rather than collapse distinct artifacts.

Operator materializers, oracles and exact semantic validators live in task modules. This context supplies their shared contracts and measurement mechanisms; see [task ownership](../../en/TASKS.md).

Compiler Program owns typed composition and bindings; every stage uses the same clean
Compiler commit. `evaluation/program.py` executes sealed Program bundles;
`evaluation/launch_plan.py` binds a LoweredProgram for existing direct task consumers.
Both consume Compiler-owned structure. Platform adapters supply exact devices, storage,
views and streams. Whole-program measurement includes every ordered kernel, with no
host-side task mathematics. Identity-bound single stages retain their existing kernel route.
