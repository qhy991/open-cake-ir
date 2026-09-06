# Workload Contract ownership

Every file in this directory is an immutable, content-bound Workload Contract. No README
entry makes one globally current: a Study Contract selects the exact Workload authority it
uses, and historical Studies continue to reference their frozen version.

Successor Workload Contracts may repair or extend semantics, materialization, oracle,
tolerance, or cases. They do not rewrite their predecessors. A new Study should choose the
latest applicable reviewed successor explicitly rather than infer it from a filename.

Canonical terminology is in
[`../../docs/GLOSSARY.md`](../../docs/GLOSSARY.md#workload-contract).
