# open-cake-ir context map

`open-cake-ir` contains four contexts with strict dependency direction.

## Contexts

- [Compiler](docs/contexts/compiler/CONTEXT.md) — defines and lowers typed hardware-explicit schedules.
- [Research Lab](docs/contexts/lab/CONTEXT.md) — specifies and executes studies over frozen authoring environments.
- [Evaluation](docs/contexts/evaluation/CONTEXT.md) — applies Workload-owned correctness and measurement assays to sealed candidates.
- [Evidence](docs/contexts/evidence/CONTEXT.md) — preserves observations and derives auditable views.

## Relationships

- **Research Lab -> Compiler**: a Cake Authoring Environment references one immutable Compiler Revision.
- **Research Lab -> Evaluation -> Evidence**: Lab controls execution; common assays emit raw receipts; Evidence stores them.
- **Evidence outputs -> Research Lab Analysis**: pure Run Audits feed the Study Analysis declared before execution.
- **Compiler -> none**: the Compiler is independent of providers, campaigns, workloads and evidence storage.
- Runtime evidence may motivate a compiler change proposal, but only an external Corpus Gate and human merge can
  create the next Compiler Revision.
