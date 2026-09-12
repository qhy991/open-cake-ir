# Workloads: complete problem definitions

[中文原文](../../wiki/workloads.md) · [Guide index](README.md) · [Operator explanations](operators.md)

A Workload fixes inputs, required outputs, correctness, and measurement. A shared operator name does not make different shapes or precision the same problem. This list is a contract catalog, not a claim that every task has a Lab evaluator, GPU qualification, or best performance. The Study selects the version.

## Standalone tile Workloads: normalization, GEMM and indexed reads

These contracts turn existing Corpus examples into standalone operators with explicit input domains, tensor ABI, mathematical references and all-element correctness rules.

| Contract | Computation and output |
| --- | --- |
| [RMSNorm FP32 v1](../../../contracts/workloads/rmsnorm-fp32-v1.json) | Normalize the last dimension and scale by gamma, with FP32 inputs and output. |
| [GEMM+bias BF16/FP32 v1](../../../contracts/workloads/gemm-bias-bf16-fp32-v1.json) | Compute `A @ B.T + bias` from BF16 A and B, with FP32 bias and output. |
| [Indexed gather BF16 v1](../../../contracts/workloads/indexed-gather-bf16-v1.json) | Select BF16 rows with paired expert/row IDs; either invalid ID produces positive zero for the whole row, and negative IDs never wrap. |
| [RMSNorm FP32 v2](../../../contracts/workloads/rmsnorm-fp32-v2.json) | Same mathematics and six cases, explicitly targeting B300 `sm_103a`. |
| [GEMM+bias BF16/FP32 v2](../../../contracts/workloads/gemm-bias-bf16-fp32-v2.json) | Same mathematics and five cases, explicitly targeting B300 `sm_103a`. |
| [Indexed gather BF16 v2](../../../contracts/workloads/indexed-gather-bf16-v2.json) | Same indexing semantics and four cases, explicitly targeting B300 `sm_103a`. |

See the [B300 guide](../B300.md) for Python starting points and separate experiment qualification.

The [tile Workload guide](../TILE_WORKLOADS.md) explains shapes, tolerances, the shared ABI and independent CPU oracle. [Baseline preparation](../../../examples/paired_triton/README.md) generates the paired native Triton source from the same IR as a declared common optimization starting point. The current delivery covers contracts, CPU references and source preparation; GPU compilation, correctness, timing, profiling and target-framework acceptance remain R2 pending.

## Flash-KMeans

BF16 points and centroids produce the nearest-centroid index under the FP32 distance/accumulation contract. The independent reference and tie-aware rule own acceptance; do not impose another tie policy afterward.

Contracts: [v1](../../../contracts/workloads/flash-kmeans-assign.json), [v2](../../../contracts/workloads/flash-kmeans-assign-v2.json). Code: [evaluation](../../../src/open_cake_ir/tasks/flash_kmeans/workload.py), [teaching tool](../../../examples/gpu/flash_kmeans_quickstart.py), [Compiler example](../../../corpus/schedules/flash-kmeans-b32-smoke-v2.json). Old Studies keep their bound versions.

## TinyGEMM2

A fixed small-batch linear layer computes input times transposed weight plus bias, then BF16 rounding. An independent FP32 reference and the declared output rules judge it. [v1](../../../contracts/workloads/tinygemm2-stage4.json) and [v2](../../../contracts/workloads/tinygemm2-stage4-v2.json) differ: v2 fixes materialized input, weight, bias, and reference bytes; a seed alone need not reproduce them.

The retained [Schedule](../../../corpus/schedules/tinygemm2-stage4-split-k.json) records the historical `checked_cuda_asset` route. The current Compiler has retired that dedicated route and checks its explicit structural refusal. Original files and observations remain intact. Replay old fixed-source results at their pinned Git revision; the refusal case does not establish current TinyGEMM2 generation support.

## DSA: sparse MLA one-step decode

Query, compressed KV cache, positional components, and supplied sparse indices produce masked scoring, softmax, and weighted values. Empty valid selections produce zero under the [v1 contract](../../../contracts/workloads/dsa-attention-sparse-mla-decode-v1.json). This fixed single-GPU DeepSeek-V3.2 task includes captured/generated rows, not full serving. Use the contract's executable scaling constant; it records a discrepancy with source prose. [WorkloadContract](../../../src/open_cake_ir/evaluation/workload.py) owns validation.

## QSA: long-sequence selection and attention

Post-projection q, k, v, index_q, and index_k feed causal complete-block pooling, LayerNorm, index-head scoring, block selection, token expansion, and selected causal attention. The [prefill contract](../../../contracts/workloads/qsa-prefill-t32768-v1.json) excludes earlier projections, RoPE, cache updates, and serving. Multiple stages require a complete Program; timing only top-k or attention is insufficient. The [evaluator](../../../src/open_cake_ir/tasks/qsa/evaluate.py) preserves that boundary. Declared smaller checks and target geometry do not establish a complete checkpoint configuration.

## Kimi-K3 KDA: stateful one-step core

Current-step inputs, gate parameters, convolution/recurrent state, and cache positions feed convolution update, recurrence, and sigmoid-gated RMSNorm. Selected state changes; unselected slots and padded rows remain protected. The [fused-decode contract](../../../contracts/workloads/kimi-k3-kda-fused-decode-v1.json) covers local head/activity cases and state pools across two calls. Prefill/chunk/model results cannot substitute.

## Kimi-K3 megaop: surrounding projections

Layer inputs, weights, and state feed qkvg/gate projections, the stateful core, then the local output projection. This is one B200 rank-local module, ending before BF16 TP AllReduce. [v1](../../../contracts/workloads/kimi-k3-kda-decode-megaop-b200-v1.json) uses a baseline-bracketed protocol; [v2](../../../contracts/workloads/kimi-k3-kda-decode-megaop-b200-v2.json) uses paired timing and confidence intervals. Rules cannot be interchanged after results.

Contract loading and boundary validation do not supply a complete megaop Lab evaluator or GPU results. Exact shapes, oracle, output storage, and state requirements remain in the contracts.

## Standalone AKA operator contracts

These migrated contracts fix the `sm_100a` input domain, tensor ABI and independent CPU
oracle. Historical AKA provenance does not qualify the current source on a GPU or establish performance.

| Contract | Computation and outputs |
| --- | --- |
| [GEMM NT+bias FP32](../../../contracts/workloads/aka-gemm-nt-bias-fp32-triton-b200-v1.json) | Weights use `[N,K]` storage; compute `input @ weight_nt.T + bias`. |
| [Histogram FP32](../../../contracts/workloads/aka-histogram-fp32-triton-b200-v1.json) | Count values in `[-4,4]`, ignore out-of-range values, and place the upper endpoint in the last bin. |
| [Max-pool1d NWC FP32](../../../contracts/workloads/aka-max-pool1d-nwc-fp32-triton-b200-v1.json) | Take window maxima along the sequence axis; padding is excluded rather than compared as zero. |
| [Momentum SGD FP32](../../../contracts/workloads/aka-momentum-sgd-fp32-triton-b200-v1.json) | Produce new parameters and momentum, with an explicit Nesterov branch and unchanged inputs. |
| [Residual LayerNorm FP32](../../../contracts/workloads/aka-residual-layernorm-fp32-triton-b200-v1.json) | Add the residual, normalize, and return residual, normalized values, mean and reciprocal standard deviation. |
| [Row gather FP32](../../../contracts/workloads/aka-row-gather-fp32-triton-b200-v1.json) | Copy FP32 rows using the declared indices into fresh nonaliasing output. |

## DeepSeek-V4-Pro routing slices

- [CSA indexer top-k](../../../contracts/workloads/deepseek-v4-csa-indexer-topk-fp32-triton-b200-v1.json) selects 1024 descending positions from 2048 unique scores. It excludes full CSA attention and short-context variable top-k.
- [MoE gate](../../../contracts/workloads/deepseek-v4-moe-gate-fp32-triton-b200-v1.json) computes `sqrt(softplus(logits))`, selects six experts using selection-biased scores, and normalizes the original scores with scale 2.5.

Both are standalone `sm_100a` routing contracts that exclude ties. They do not cover expert execution, shared experts, communication or complete serving, and do not imply current GPU qualification.

## Why some examples are not listed as Workloads

Softmax and RoPE also appear as [Corpus cases](../../../corpus/manifest.json) or independent measurements. [State update](../../../examples/gpu/state_store_b200_correctness/README.md) and [FMA](../../../examples/gpu/fma_b200_correctness/README.md) have fixed GPU tasks. A JSON file or README does not automatically make them a complete Workload callable by arbitrary Studies; task semantics, evaluator, and evidence each need delivery.
