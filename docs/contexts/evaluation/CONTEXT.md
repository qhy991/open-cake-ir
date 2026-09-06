# Evaluation context

[中文阅读](../../zh-CN/contexts/evaluation/CONTEXT.md) · [Bilingual catalog](../../README.md)

Evaluation owns the common assay after an Authoring Environment seals a Candidate.
Canonical definitions are in the [`Glossary`](../../GLOSSARY.md#evaluation-terms).

## Owned terms

- [`LaunchableCandidate`](../../GLOSSARY.md#launchablecandidate)
- [`Evaluation Protocol`](../../GLOSSARY.md#evaluation-protocol)
- [`Logical Evaluation Attempt`](../../GLOSSARY.md#logical-evaluation-attempt)
- [`Evaluation Receipt`](../../GLOSSARY.md#evaluation-receipt)
- [`Portfolio Artifact`](../../GLOSSARY.md#portfolio-artifact)
- [`Measurement Quality`](../../GLOSSARY.md#measurement-quality)

## Responsibilities

- Require both Authoring Environments to cross the same LaunchableCandidate boundary.
- Materialize Workload cases and apply the external oracle and tolerances.
- Enforce correctness before timing or profiler collection.
- Keep search, confirmatory, and profiler purposes as separate Evaluation Receipts.
- Retain raw timing cohorts, launch/fallback counts, and profiler output needed for replay.
- Reject unsupported portfolio keys before launch.

Evaluation does not define the Estimand, Claim Scope, Study inclusion, or Claim View. It
records what happened at the declared assay boundary.

## Relationships

- The Workload Contract supplies semantics, case materialization, oracle, and tolerances.
- The CampaignLock supplies exact execution admission.
- Evidence stores the Candidate and Receipt bytes after Evaluation observes them.
- Study analysis consumes Run Audits, not an evaluator's headline number.

## Boundary examples

Timing CV failure is Measurement Quality evidence, not Candidate incorrectness. A
correctness-qualified profiler launch supplies diagnosis but no profiler-free latency
sample. “Source” must name its role—authored source, lowered source, expanded source, PTX,
CUBIN, or SASS—rather than collapse distinct artifacts.
