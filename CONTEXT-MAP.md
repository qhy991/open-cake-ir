# open-cake-ir documentation and context map

[中文阅读](docs/zh-CN/CONTEXT-MAP.md) · [Bilingual catalog](docs/README.md)

This is the navigation authority for repository documentation. Canonical term definitions
live in [`docs/GLOSSARY.md`](docs/GLOSSARY.md); this file says which document owns each kind
of information and where new material belongs.

## Documentation lifecycle

| Class | Contains | Update rule | Location |
| --- | --- | --- | --- |
| Stable documentation | Architecture, vocabulary, invariants, claim language, and operating rules | Change only with the supported contract | `docs/ARCHITECTURE.md`, `docs/GLOSSARY.md`, Context docs, Runbook |
| Decision history | Rationale and accepted trade-offs | Append or supersede with a new ADR | `docs/adr/` |
| Executable authority | Exact semantics, release closure, and execution policy | Create a successor through the owning gate | `compiler/`, `contracts/`, `runtime/`, released indexes |
| Historical snapshot | A dated survey, audit, observation, or migration state | Freeze; correct with an erratum or successor | dated `docs/`, `inventory/`, and retained Evidence |
| Current view | Human-readable current authorities and supported claims | Regenerate from canonical inputs | `reports/current/` |

The filename `inventory/CURRENT_STATE.md` predates this classification. It is a frozen
2026-08-25 migration snapshot, not a current view.

## Product contexts

`open-cake-ir` exposes two user-facing capabilities—the standalone Compiler and the
dependent Research Lab—implemented by four contexts. Evaluation and Evidence support the
Lab; they are not additional products.

| Context | Question | Context document |
| --- | --- | --- |
| Compiler | Is this hardware schedule valid, and what target source does it produce? | [`docs/contexts/compiler/CONTEXT.md`](docs/contexts/compiler/CONTEXT.md) |
| Research Lab | How does an agent improve Candidates under a frozen Study? | [`docs/contexts/lab/CONTEXT.md`](docs/contexts/lab/CONTEXT.md) |
| Evaluation | Is one sealed Candidate correct, and was its measurement valid? | [`docs/contexts/evaluation/CONTEXT.md`](docs/contexts/evaluation/CONTEXT.md) |
| Evidence | Can observations and terminal decisions be replayed independently? | [`docs/contexts/evidence/CONTEXT.md`](docs/contexts/evidence/CONTEXT.md) |

The Workload Contract is an external input owned by its content-bound definition. The Lab
does not redefine operator semantics or the oracle.

## Dependency direction

- **Research Lab -> Compiler**: a Cake IR Authoring Environment references one immutable
  Compiler Revision.
- **Research Lab -> Evaluation -> Evidence**: Lab controls execution, Evaluation applies
  common assays, and Evidence stores observations.
- **Evidence -> Lab Analysis**: pure Run Audits feed the preregistered Study analysis.
- **Compiler -> none**: the Compiler is independent of providers, Campaigns, Workloads,
  Evaluation, and Evidence.

Runtime evidence may motivate a Compiler proposal. Only a full Corpus Gate, external
release approval, and a released successor can change Compiler semantics.

## Where new information goes

| New information | Destination | Stable docs changed? |
| --- | --- | --- |
| Candidate or evaluation result | Campaign Evidence and Evaluation Receipt | No |
| Completed or failed Run | Terminal Archive and Run Audit | No |
| Accepted Study conclusion | Study Report; then regenerated Claim View | No |
| New Workload semantics or oracle | Successor Workload Contract | Only if the general boundary changes |
| New experimental design | Successor Study Contract or template | Only if the general boundary changes |
| Repeated compiler failure requiring a new rule or primitive | ADR, Compiler successor, Corpus Gate | Yes, when the decision is accepted |
| Correction to retained history | Erratum or successor report | Never rewrite the original Evidence |
| New paper version | New source revision and explicit comparison | Never overwrite the prior paper contract |

## Read order

### New user

1. [中文 Wiki](docs/wiki/README.md)
2. [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md)
3. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
4. [`docs/GLOSSARY.md`](docs/GLOSSARY.md) and [`compiler/AUTHORING_CONTRACT.md`](compiler/AUTHORING_CONTRACT.md)

### Researcher or experiment operator

1. [`docs/PAPER_CONTRACT.md`](docs/PAPER_CONTRACT.md)
2. [`docs/contexts/lab/CONTEXT.md`](docs/contexts/lab/CONTEXT.md)
3. [`docs/RUNBOOK.md`](docs/RUNBOOK.md)
4. The exact Workload Contract, Study Contract, and CampaignLock for the intended run

### Maintainer or auditor

1. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
2. [`docs/GLOSSARY.md`](docs/GLOSSARY.md)
3. [`docs/ACCEPTANCE_GATES.md`](docs/ACCEPTANCE_GATES.md)
4. [`reports/current/STATUS.md`](reports/current/STATUS.md)
5. [`docs/adr/`](docs/adr/) and the exact retained Evidence being audited
