# Workload Contract ownership

The B300 successors are [`rmsnorm-fp32-v2.json`](rmsnorm-fp32-v2.json),
[`gemm-bias-bf16-fp32-v2.json`](gemm-bias-bf16-fp32-v2.json) and
[`indexed-gather-bf16-v2.json`](indexed-gather-bf16-v2.json). They preserve the
v1 mathematics, ABI, 15 cases and tolerances while explicitly selecting
`sm_103a`. See the [B300 guide](../../docs/en/B300.md) for Python starting points
and the separate device/experiment qualification boundaries.

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

The standalone tile contracts
[`rmsnorm-fp32-v1.json`](rmsnorm-fp32-v1.json),
[`gemm-bias-bf16-fp32-v1.json`](gemm-bias-bf16-fp32-v1.json) and
[`indexed-gather-bf16-v1.json`](indexed-gather-bf16-v1.json) give the existing Corpus
examples explicit mathematical, ABI, input-domain and correctness authority. Their
standard-library CPU oracles and source-only Compiler baseline preparation are described
in [TILE_WORKLOADS](../../docs/en/TILE_WORKLOADS.md). These are independent standalone
definitions; GPU equivalence, timing and framework acceptance remain R2 pending.
