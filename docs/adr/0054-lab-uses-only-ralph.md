# ADR 0054: Lab uses only Ralph

Status: accepted by the repository owner, 2026-09-07; source implementation for an Executor successor.

## Decision

The owner explicitly retired the legacy prompt interface and its compatibility obligation.
This supersedes ADR 0048's compatibility section. All current matched-search Studies use
schema-v2 and `task_agents_ralph_v1`, including the Cake/native Triton comparison.

TASK.md and AGENTS.md are derived from the CampaignLock. A same-thread provider updates
candidate-set.json from each retained StateCard. The external Ralph controller owns budgets
and termination; Compiler filtering, external correctness, timing, profiling, promotion,
and Evidence custody keep their existing owners. Portfolio is a validation artifact path,
not an alternative agent authoring loop.

The prompt directory, qualification templates, legacy renderer/reference-directory path,
legacy Study/Lock admission and duplicate template entrypoints are removed. Provider
qualification always uses Ralph, with either closed-research or artifact-optimization
feature policy. Frozen pre-Ralph Study files are retired from the current checkout.

## History and release boundary

Historical releases, approvals, Executor identities and Evidence are not rewritten. Replay
of retired Studies requires their historical Git checkout; the current Lab rejects them.
The parent of this change contains the last complete pre-removal contract/prompt set.
This source change requires an Executor successor before live composition. CPU fixtures
are not host admission, provider qualification or GPU evidence. Compiler source is unchanged.

## Acceptance

Reject pre-Ralph Studies and Locks and prompt-bearing Ralph arms. Exercise first/resumed
turns, both authoring treatments, candidate-set custody, task-file tampering, budgets,
qualification receipts, filtering, confirmation and semantic replay through the Ralph path.
