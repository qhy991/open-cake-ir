# Open Cake English guide

[中文原文](../../wiki/README.md) · [English home](../README.md)

Start here to understand how an idea becomes a checked and measured GPU program. No compiler background is required.

## First reading route

1. [What the system does](../ARCHITECTURE.md).
2. [Try it without a GPU](../GETTING_STARTED.md).
3. [Read a Schedule](schedule.md).
4. [Understand common operators](operators.md).
5. [Read results and troubleshoot](results.md).

| Question | Page |
| --- | --- |
| What do load, mma, and top_k mean? | [IR primitives](primitives.md) |
| How can I author a Schedule in Python? | [Python frontend](../PYTHON_FRONTEND.md) |
| Which complete task definitions exist? | [Workloads](workloads.md) |
| How do I prepare the shared ABI and matched baselines for three standalone operators? | [Tile Workloads](../TILE_WORKLOADS.md) |
| How are two nested static TileLoops scoped? | [Loop scopes](../TRITON_LOOP_SCOPES.md) · [GEMM plan with tails](../../../corpus/schedules/gemm-bias-two-deep-tail-b1-smoke.json) |
| How do IR and native Triton share a baseline, build and Evaluation? | [Paired Triton](../PAIRED_TRITON.md) |
| How does AI work under a budget? | [Experiments](experiments.md) |
| How do I prepare real GPU work? | [Runbook](../../RUNBOOK.md) |
| How can an old task be replayed? | [Replay](replay.md) |
| Why does load shape matter? | [Access-domain repair](../ACCESS_DOMAIN_REPAIR_20260906.md) |
| What does a complete small GPU check look like? | [Eight-output affine example](../AFFINE_PARENT_B200_CANARY_20260906.md) |
| What is released? | [Generated status](../../../reports/current/STATUS.md) |
| Where are formal definitions? | [Glossary](../GLOSSARY.md) |
| Who owns each part? | [Context map](../../../CONTEXT-MAP.md) |
| Why was a design chosen? | [Decision directory](../adr/README.md) |
| How are documents maintained? | [Maintenance](maintaining.md) |

Use the [complete bilingual catalog](../../README.md) for all guides, decisions, surveys, and historical reports. This in-repository guide is checked with source; there is no separately maintained GitHub Wiki. Small numerical examples explain mathematics, while actual contracts and retained evidence define acceptance.
