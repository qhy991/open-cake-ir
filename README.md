# open-cake-ir

`open-cake-ir` is a compiler-first, independent reconstruction of ideas described in
[CAKE: Compiler-Agent Co-Design for Frontier Kernel Evolution](https://arxiv.org/abs/2608.12629v1).
It does not contain or claim to reproduce the unpublished CAKE implementation.

In plain language, an author writes a typed, hardware-explicit Schedule. The Open Cake
Compiler checks it and deterministically lowers eligible operations to inspectable target
source. The dependent Research Lab can then evaluate sealed Candidates under frozen
Workload, Study, measurement, and Evidence rules.

第一次接触本项目，可以从
[`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) 开始；系统全貌见
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)，正式术语见
[`docs/GLOSSARY.md`](docs/GLOSSARY.md)。

## Product boundary

The repository has one product core and one dependent research application, supported by
two shared Modules:

1. **Open Cake Compiler** — Schedule semantics, exact Targets, localized Findings,
   modeled analysis, deterministic Lowering, Compiler Revisions, and the Corpus Gate.
2. **Research Lab** — frozen Study execution over complete Authoring Environments.
- **Evaluation** — common correctness, timing, and profiler assays after a Candidate is
  sealed.
- **Evidence** — immutable objects, append-only events, replay, Run Audits, and derived
  reports.

New matched-search Runs expose only a generated `TASK.md` and `AGENTS.md` to each Agent
workspace. An external Ralph Controller continues the same thread under independent
time, token, Turn, and Evaluation budgets; the trusted Evaluator and Evidence store remain
outside Agent control.

The dependency direction is permanent:

```text
Research Lab -> Open Cake Compiler
Research Lab -> Evaluation -> Evidence
Open Cake Compiler -X-> Lab / Provider / Workload / Evidence
```

## Current released authorities

The human-readable release view is generated at
[`reports/current/STATUS.md`](reports/current/STATUS.md). Its canonical inputs are:

- [`compiler/revision.lock.json`](compiler/revision.lock.json) for the released Compiler;
- [`inventory/EXECUTOR_REVISIONS.json`](inventory/EXECUTOR_REVISIONS.json) for the current
  Executor descriptor.

The README deliberately does not copy revision history, migration milestones, or
experimental results. Operator, Program, model, and serving claims remain bound to their
accepted Study Reports and Evidence until a repository-wide Claim View can be derived.

Regenerate the release view with:

```bash
.venv/bin/python tools/render_current_status.py --write
.venv/bin/python tools/render_current_status.py --check
```

## Evidence language

The project keeps these boundaries explicit:

- **proposed** is not released;
- **lowerable** is not compiled or GPU-correct;
- **correctness-qualified** is not timed or profiled;
- an operator or complete-Program result is not a model-forward or serving result;
- `unknown`, `not_run`, `invalid`, and an observed negative outcome are different states.

The complete definitions and permitted claim scopes are in
[`docs/GLOSSARY.md`](docs/GLOSSARY.md#evidence-strength-and-scope).

## Documentation map

| Question | Canonical document |
| --- | --- |
| What does the system do and where are its boundaries? | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| What does a term mean? | [`docs/GLOSSARY.md`](docs/GLOSSARY.md) |
| Where should a new document or experiment result go? | [`CONTEXT-MAP.md`](CONTEXT-MAP.md) |
| How do I write a complete Schedule? | [`compiler/AUTHORING_CONTRACT.md`](compiler/AUTHORING_CONTRACT.md) |
| How do I run or audit a Study? | [`docs/RUNBOOK.md`](docs/RUNBOOK.md) |
| What does the paper establish, report, or leave unknown? | [`docs/PAPER_CONTRACT.md`](docs/PAPER_CONTRACT.md) |
| Why was a durable design choice made? | [`docs/adr/`](docs/adr/) |
| What is released now? | [`reports/current/STATUS.md`](reports/current/STATUS.md) |

## Document and experiment lifecycle

A new experiment appends Candidate, Evaluation, Evidence, Run Audit, and Study Report
artifacts. It does not rewrite architecture, terminology, or an old snapshot. An accepted
result may change a regenerated Claim View. A stable document changes only when evidence
changes a supported concept, owner, invariant, interface, or claim boundary.

Historical paths remain available for replay. Current views are disposable projections;
they never select an Executor, authorize a Campaign, or turn a narrow result into a broader
claim. [ADR 0047](docs/adr/0047-documentation-separates-stable-history-and-current-views.md)
records this ownership rule.
