# Open Cake documentation

[中文](../zh-CN/README.md) · [Complete bilingual catalog](../README.md) · [Repository home](../../README.md)

**Write an explicit GPU computation plan, check it, generate source, and validate the result.** open-cake-ir independently reconstructs parts of the public CAKE research ideas; it is not the paper's unpublished implementation.

A Workload defines the problem, a Schedule describes workers and data, the Compiler checks and translates it, Evaluation checks outputs and measurements, and Evidence preserves observations. A Study fixes a research question and fair comparison before execution.

## Start here

1. [System overview](ARCHITECTURE.md).
2. [Operators with small numbers](wiki/operators.md).
3. [First use without a GPU](GETTING_STARTED.md).
4. [Read a Schedule](wiki/schedule.md).
5. [Read results](wiki/results.md).
6. [Understand experiments](wiki/experiments.md).

The [English guide](wiki/README.md) also links every learning page. Use the [Glossary](GLOSSARY.md) as a lookup.

## Continue by purpose

| Purpose | English page |
| --- | --- |
| Authoring Schedules in Python | [Python frontend](PYTHON_FRONTEND.md) |
| Running FP32 elementwise and row reductions on Apple M2 | [Metal guide](../metal.md) |
| IR objects, implementation ownership and extension points | [IR guide](IR_GUIDE.md) |
| Initialization, accumulation and stores in two nested TileLoops | [Triton loop scopes](TRITON_LOOP_SCOPES.md) |
| Matched IR/native Triton optimization from one baseline | [Paired Triton](PAIRED_TRITON.md) |
| Operating a frozen Study | [Runbook](../RUNBOOK.md) |
| Understanding acceptance evidence | [Acceptance gates](../ACCEPTANCE_GATES.md) |
| Comparing paper claims and local evidence | [Paper contract](../PAPER_CONTRACT.md) |
| Module ownership | [Context map](../../CONTEXT-MAP.md) |
| Design rationale | [All decision records](adr/README.md) |
| Task definitions | [Workloads](wiki/workloads.md) |
| Shared ABI, CPU references and matched baselines for three standalone operators | [Tile Workloads](TILE_WORKLOADS.md) |
| Historical reconstruction | [Replay](wiki/replay.md) |
| The 677-row review and 56 qualified fixed instances | [AKA review](AKA_QUALIFIED_IR_REVIEW_AND_LAB_PLAN_20260903.md) |
| A complete eight-output GPU example | [Affine check](AFFINE_PARENT_B200_CANARY_20260906.md) |
| Current released authorities | [Generated status](../../reports/current/STATUS.md) |
| Surveys, history, and every language pair | [Complete catalog](../README.md) |

Metal is a correctness-first prototype: `serial_program_tile` uses one active thread per threadgroup to process the complete tile serially. It connects the Compiler to a direct runtime adapter; Metal Lab/Campaign integration, agent search, and performance optimization are not provided.

Original documents retain their paths. Chinese reading companions for English originals live under zh-CN; English counterparts to Chinese originals live here. Companions simplify explanations and link full historical tables rather than creating a second authority. Dates, scope, failures, and unverified outcomes retain their original meaning. Source generation, compilation, correctness, and performance remain distinct.
