# ADR 0047: Documentation separates stable rules, history, and current views

## Status

Accepted by the repository owner, 2026-08-30. This decision changes documentation
ownership only. It changes no Compiler Revision, Executor Revision, Workload Contract,
Study Contract, Evidence object, evaluator, or experimental claim.

## Problem

The repository currently asks several Markdown files to be stable explanations and live
status reports at the same time. In particular, the README, acceptance-gate document,
paper contract, coverage surveys, and `inventory/CURRENT_STATE.md` repeat release ids,
historical experiments, and current claims. Those copies drift whenever a release or
experiment advances.

Internal migration labels such as `G7`, `G7F`, and `G8` also escaped into user-facing
prose. They identify positions in an old migration checklist, not domain concepts. A
reader should see the capability that was qualified, not decode an internal sequence
number.

## Decision

Documentation has five lifecycle classes. Each fact has one class and one authority.

| Class | Purpose | Mutation rule | Authority |
| --- | --- | --- | --- |
| Stable documentation | Architecture, vocabulary, invariants, claim language, and operating rules | Edit only when the supported contract changes | The owning Context or accepted ADR |
| Decision history | Why a durable choice was made | Append a successor ADR; do not make old evidence current | ADRs |
| Executable authority | Semantics and exact released execution closure | Create a successor through the owning release path | Contracts, release locks, Targets, Corpus, and descriptors |
| Historical snapshot | What was observed or believed at a named date and revision | Freeze; corrections are new errata or successor reports | The named snapshot and retained Evidence |
| Current view | Human-readable projection of current released authorities and accepted reports | Delete and regenerate; never authorize execution or a claim | Canonical locks, indexes, Run Audits, and accepted Study Reports |

`docs/GLOSSARY.md` is the sole prose definition of canonical terms. Context documents
state ownership and relationships and link to those definitions rather than restating
them. Code/schema spellings remain authoritative at machine boundaries.

`reports/current/` contains generated current views. The README links to those views and
to their machine-readable inputs; it does not copy a release history or experimental
ledger. Until the repository has one accepted Study Report index, the generated status
view reports release authorities only and deliberately emits no repository-wide Claim
View.

Stable and operational documents use descriptive capability names such as
“closed-provider qualification”, “provider-default optimization qualification”, and
“bounded end-to-end system qualification”. Legacy gate codes may remain only where they
are part of an immutable artifact id, filename, or explicitly historical quotation; the
surrounding prose must explain the capability.

## New experiment rule

A new Candidate, Evaluation Receipt, Campaign, Run Audit, or Study Report does not modify
stable documentation. It appends Evidence and, if accepted, may change a regenerated
current Claim View. Stable documentation changes only when repeated evidence changes a
term, owner, invariant, supported interface, or claim boundary.

A correction never rewrites retained experimental bytes. It creates an erratum or
successor report and lets the current projection select the supported interpretation.

## Migration boundary

Historical paths are not moved merely to make the tree look cleaner. Existing snapshots
receive explicit lifecycle notices and remain available at their referenced paths. New
material follows this decision; physical convergence happens only at an unpinned
successor boundary.

## Acceptance evidence

- The README contains no hand-maintained release history or opaque migration-gate codes.
- Generated current status matches `compiler/revision.lock.json` and
  `inventory/EXECUTOR_REVISIONS.json`.
- Context documents do not carry parallel glossary definitions.
- Stable gate headings use capability names rather than sequence codes.
- A documentation contract test detects regeneration drift and reintroduction of opaque
  gate codes into stable entry documents.
