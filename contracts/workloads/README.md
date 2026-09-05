# Workload Contract ownership

Every file in this directory is an immutable, content-bound Workload Contract. No README
entry makes one globally current: a Study Contract selects the exact Workload authority it
uses, and historical Studies continue to reference their frozen version.

Successor Workload Contracts may repair or extend semantics, materialization, oracle,
tolerance, or cases. They do not rewrite their predecessors. A new Study should choose the
latest applicable reviewed successor explicitly rather than infer it from a filename.

Canonical terminology is in
[`../../docs/GLOSSARY.md`](../../docs/GLOSSARY.md#workload-contract).

The Kimi-K3 B200 rank-local megaop has two separately frozen protocols:
[`kimi-k3-kda-decode-megaop-b200-v1.json`](kimi-k3-kda-decode-megaop-b200-v1.json)
retains the original ABA measurement, and
[`kimi-k3-kda-decode-megaop-b200-v2.json`](kimi-k3-kda-decode-megaop-b200-v2.json)
uses paired measurement and a median confidence interval. Both require exactly 18 graph
rows, 12 rank-local heads, head dimension 128, hidden size 7168, and 49 state slots,
with distinct 18-active-row and 17-active-row cases. These are Workload definitions;
registration alone does not provide a Lab evaluator or establish GPU correctness.
