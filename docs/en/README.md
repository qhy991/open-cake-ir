# Open Cake Technical Report: English Reading Guide

For native CUDA/PTX schedules and their exact compile boundary, see the [native backend guide](../NATIVE_CUDA.md).

[中文](../zh-CN/README.md) · [Complete bilingual catalog](../catalog.md) · [Repository home](../../README.md)

See the [B300 guide](B300.md) for the three Python operator starting points.

**Write an explicit GPU computation plan, check it, generate source, and validate the result.** open-cake-ir independently reconstructs parts of the public CAKE research ideas; it is not the paper's unpublished implementation.

A Workload defines the problem, a Schedule describes workers and data, the Compiler checks and translates it, Evaluation checks outputs and measurements, and Evidence preserves observations. A Study fixes a research question and fair comparison before execution.

## Quick lookup

| Purpose | Direct links |
|---|---|
| Inspect results and external-reference gaps | [Hardware results](../RESULTS.md) · [FlashInfer experimental status and English summary](../results/nvidia/FLASHINFER_STATUS.md) |
| Choose a platform or experiment workflow | [Platform and method catalog](../catalog.md) · [Lab workflow](wiki/experiments.md) |
| Read or cite the technical report | [Report chapters and citation](../README.md) · [Complete catalog](../catalog.md) |
| Locate implementation owners | [Source navigation](../../README.md#源码与文档结构) · [Development branches](../DEVELOPMENT_BRANCHES.md) |

<details>
<summary>Historical checkpoints — 2026-09-16</summary>

These dated notes retain their original scope. Current status and results are linked above.



- **2026-09-16 — the Hygon DCU (gfx938) is a third target that runs.** A Triton-lowered kernel
  compiles to an HSACO inside the DTK container, loads through `evaluation/hip_driver.py`,
  launches on a BW1101 and passes the external CPU oracle: `output_mismatches: 0`,
  `max_abs_error: 4.77e-07`. The route sits beside NVIDIA and Apple rather than stepping down
  from either — Hygon is its own vendor (`Vendor.HYGON`), sharing the
  `amdgcn-amd-amdhsa--` code object with AMD and nothing else.
  **It reports correctness, not latency**: gfx938 has no named timing source yet, so its Study
  states a measurement-coverage limitation instead of a paired assay and the receipt carries
  `timing: null`. A timing source is minted by measuring, not by adding a row — see the
  [DCU design record](../dcu-gfx938-design.md).

- **2026-09-16 — source identity is the git commit ([ADR 0065](../adr/0065-source-identity-is-the-commit.md)).**
  A Compiler is `open-cake-ir@<commit>` and an Executor is `<target>@<commit>`, so one shared
  source change no longer retires every other host's Executor.

</details>

## Start here

1. [System overview](ARCHITECTURE.md).
2. [Operators with small numbers](wiki/operators.md).
3. [First use without a GPU](GETTING_STARTED.md).
4. [Read a Schedule](wiki/schedule.md).
5. [Read results](wiki/results.md).
6. [Understand experiments](wiki/experiments.md).

The [English guide](wiki/README.md) also links every learning page. Use the [Glossary](GLOSSARY.md) as a lookup.

The [FlashInfer-Bench experimental status](../results/nvidia/FLASHINFER_STATUS.md) includes all 26 tasks, fixed-shape external comparisons, original-starter boundaries, independent NCU evidence and an English summary.

## Continue by purpose

| Purpose | English page |
| --- | --- |
| Authoring Schedules in Python | [Python frontend](PYTHON_FRONTEND.md) |
| Optimizing normalization tasks through TaskLab on Apple M1 Pro or M2 | [Metal guide](../metal.md) |
| IR objects, implementation ownership and extension points | [IR guide](IR_GUIDE.md) |
| Initialization, accumulation and stores in two nested TileLoops | [Triton loop scopes](TRITON_LOOP_SCOPES.md) |
| Matched IR/native Triton optimization from one baseline | [Paired Triton](PAIRED_TRITON.md) |
| Matched IR/native CuTeDSL GEMM on B300 | [Paired CuTeDSL](PAIRED_CUTE.md) |
| Operating a frozen Study | [Runbook](../RUNBOOK.md) |
| Reusing each task's best verified artifact in the next optimization | [Task incumbents](TASK_INCUMBENTS.md) |
| Reading evidence as a checklist and testing whether that feedback helps the Agent | [Rubric feedback](RUBRIC_FEEDBACK.md) |
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
| Surveys, history, and every language pair | [Complete catalog](../catalog.md) |

Metal uses `simd_program_tile` with 32 lanes per threadgroup and runs normalization tasks through the existing TaskLab/Ralph evaluation path. Host construction, warmed host calls and GPU command-buffer intervals are recorded separately; reported gains require the recorded noise controls to pass.

Original documents retain their paths. Chinese reading companions for English originals live under zh-CN; English counterparts to Chinese originals live here. Companions simplify explanations and link full historical tables rather than creating a second authority. Dates, scope, failures, and unverified outcomes retain their original meaning. Source generation, compilation, correctness, and performance remain distinct.

## Research design

[Executable optimization knowledge transfer](OPTIMIZATION_TRANSFER.md) describes the proposed
cross-hardware mechanism and its explanation-versus-rewrite ablation.

## Citation

These documents are the English reading edition of Haiyan Qin's open-cake-ir technical report.
The [report entry point](../README.md) defines its scope and provides the recommended citation,
BibTeX, and instructions for citing a fixed commit. Machine-readable metadata is maintained in
[CITATION.cff](../../CITATION.cff). Language companions may summarize the original chapters;
they do not define separate publications or independent experimental evidence.

For experimental results, retain the experiment's own source commit, target, workload, and
timing boundary in addition to the report chapter you cite.
