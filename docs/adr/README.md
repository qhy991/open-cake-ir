# Architecture decision records

ADRs record durable decisions, not current release status or experiment progress. A new
decision appends a new ADR; a successor names what it supersedes. Historical experiment
identifiers may appear when they are evidence for the decision, but user-facing summaries
must describe the capability in words.

The records currently fall into five themes:

- 0001–0006: product boundary, Study variants, repository lifecycle, and candidate sets;
- 0007–0019: Candidate identity, Evidence, ranking, and matched-study semantics;
- 0020–0029: KDA-derived expressibility and Compiler/Workload ownership;
- 0030–0037: release approval, custody, memory safety, and reusable IR primitives;
- 0038–0046: QSA Program expression, resource analysis, and lowering proposals;
- 0047: documentation ownership and temporal layers;
- 0048: two-file Agent tasks and externally controlled Ralph Runs.

Statuses have these meanings:

- **proposed**: implementation and review may proceed; no release or GPU result follows;
- **accepted**: the repository owner accepted the decision;
- **superseded**: a later ADR owns the current decision;
- **rejected**: the proposal is retained as decision history but must not be implemented.

See [ADR 0047](0047-documentation-separates-stable-history-and-current-views.md) for why
stable documentation, historical snapshots, and generated current views are separate.
