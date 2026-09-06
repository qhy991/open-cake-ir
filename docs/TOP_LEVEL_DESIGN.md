# Top-level design

Status: superseded as a current documentation owner by
[ADR 0047](adr/0047-documentation-separates-stable-history-and-current-views.md).

This path is retained because older links and Git history refer to it. It no longer
duplicates architecture, terminology, release status, migration progress, or experimental
claims.

Use the current canonical owners instead:

| Question | Owner |
| --- | --- |
| Product boundary, Modules, dataflow, and evolution loops | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Canonical terms and claim language | [`GLOSSARY.md`](GLOSSARY.md) |
| Context responsibilities and relationships | [`contexts/`](contexts/) |
| Current released Compiler and Executor | [`../reports/current/STATUS.md`](../reports/current/STATUS.md) |
| Exact Workload and Study semantics | Content-bound files under [`../contracts/`](../contracts/) |
| Durable design rationale | [`adr/`](adr/) |
| Historical migration state | Dated snapshots under [`../inventory/`](../inventory/) and Git history |

The original design document established the compiler-first product boundary, one-way Lab
dependency, canonical owners, separate kernel/compiler evolution loops, and the
`matched_search`/`portfolio` distinction. Those accepted decisions are retained in
[ADR 0001](adr/0001-compiler-first-with-dependent-lab.md),
[ADR 0002](adr/0002-portfolio-as-second-study-variant.md), and the current Architecture.

No new result should be appended here. A new experiment produces Evidence and a Study
Report; a new system decision produces a successor ADR.
