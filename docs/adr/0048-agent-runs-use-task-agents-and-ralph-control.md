# ADR 0048: Agent Runs use TASK.md, AGENTS.md, and external Ralph control

## Status

Accepted by the repository owner, 2026-08-30. Implemented by the matched-search schema-v2
successor and Executor `open-cake-ir-b200-v42`. This changes the Agent-facing Lab path; it
does not change Compiler semantics, Workload oracles, Evaluation, or historical Evidence.

## Problem

The original matched path embedded Workload, policy, reference material, output protocol,
and dynamic feedback in one arm-specific Prompt template. Every new Workload or authoring
policy risked another copied template, while the rendered Prompt duplicated facts already
owned by Workload, Study, Compiler, and Authoring Environment authorities.

KDA demonstrates a smaller Agent interface: one complete task document and one stable
agent-rules document, with an external judge retaining correctness, performance, budgets,
and history. The useful property is the ownership boundary, not the fiction that two
Markdown files replace the trusted evaluator.

## Decision

A new matched-search Study may declare `task_agents_ralph_v1`:

- `TASK.md` is rendered from the resolved Workload, Study, Target, scaffold, and arm
  references. It owns no mutable result.
- `AGENTS.md` states stable workspace, tool, evidence, and single-writer rules. It owns no
  task semantics or current frontier.
- The workspace initially contains exactly those two read-only files. The Agent may add or
  update only `candidate-set.json`.
- The provider invocation contains only a minimal instruction to read both files and an
  evidence-derived StateCard. It does not repeat the task authority.
- The same provider thread continues until an external Ralph controller terminates it.
- The exact TASK, AGENTS, and StateCard bytes supplied for every iteration are retained as
  the provider reference-bundle Evidence object.

The Ralph budget has independent bounds for provider tokens, wall time, active authoring
time, Turns, search Evaluations, confirmatory Evaluations, and attribution Evaluations. A
Turn starts only when the remaining Evaluation budget can admit its declared worst-case
assays. Run-level retries and replacement Runs remain forbidden; the existing one-time
same-authority zero-work GPU-admission recovery remains inside one Logical Evaluation
Attempt.

The trusted Compiler, Authoring Environment builders, broker Evaluator, Evaluation
Receipts, Event Ledger, Terminal Archive, Run Audit, and Study Report remain outside Agent
control. Markdown never becomes the correctness, performance, execution, or claim
authority.

## Compatibility

Frozen schema-v1 Studies retain their exact Prompt templates and replay rules. New Ralph
Studies do not contain a `prompt_template` field. The old path is a historical ingest and
replay edge, not a second recommended authoring interface.

## Failure semantics

- Changing TASK.md or AGENTS.md is contamination and terminates the Run.
- Missing or malformed StateCard input is a provider protocol fault.
- Token, time, Turn, or Evaluation exhaustion is a normal Ralph terminal reason, not a
  Candidate failure.
- Provider, harness, broker, or custody faults remain missing data under the Study's
  preregistered analysis.
- A system-qualification Ralph Run still produces no arm comparison or Estimand.

## Acceptance evidence

- Schema-v2 Preflight resolves the two-file interface and budget vector into CampaignLock.
- Each arm receives a distinct TASK.md and AGENTS.md with no mounted reference directory.
- Initial and resumed Turns preserve both files, one candidate-set lifecycle, and one
  provider thread.
- Token, wall-time, active-authoring, Turn, and Evaluation limits have deterministic tests.
- Two-Turn execution retains per-iteration task bundles and passes semantic replay.
- Tampered task files or replayed task-package bytes fail closed.
- Frozen schema-v1 Study and Evidence tests continue to pass.
