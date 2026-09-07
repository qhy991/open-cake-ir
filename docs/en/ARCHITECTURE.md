# The system: from an idea to a verifiable GPU program

[中文原文](../ARCHITECTURE.md) · [English home](README.md)

open-cake-ir makes computation choices explicit, locates errors, and preserves evidence for improvements. This page explains stable responsibilities. Read the [generated status](../../reports/current/STATUS.md) for released identities.

## Why write a Schedule?

Even summing each row of a table involves choices: which threads handle a row, how large each tile is, whether data can be reused, and whether writers collide. A Schedule describes those choices between the mathematical task and final machine code. The Compiler checks it and generates source; external checks and GPU measurements decide correctness and speed.

| Part | Responsibility | Main flow |
| --- | --- | --- |
| Compiler | Check and translate a complete hardware plan | Schedule → Assessment → Lowering |
| Research Lab | Organize AI authoring under a frozen design and budget | Workload + Study → Campaign |
| Evaluation | Check answers, measure, and collect diagnostics | Sealed candidate → receipts |
| Evidence | Preserve observations and rebuild conclusions | Objects → audit → report |

The Compiler works independently. Lab uses it, Evaluation, and Evidence; the Compiler never imports experimental or provider logic.

## Three different documents

For row normalization, Workload fixes inputs, mathematics, reference, and tolerance. Schedule assigns threads, storage, and operations. Study fixes the compared authoring environments, repetitions, budgets, and analysis. Changing a plan need not change the mathematical problem. Changing the scoring rule after seeing results destroys comparability.

## The compiler path

Format/type checks precede dependency, address, resource, and hardware checks. Assessment separates structural acceptance from backend eligibility and contains localized Findings. Each Finding retains its contract category, severity, and separate acceptance/lowering dispositions. `Assessment.findings` retains the blocking/report observations checked by the existing Corpus Gate; `Assessment.guidance` carries nonblocking hints. CLI, Lab, and profile reports expose both without treating hints as acceptance evidence or GPU measurements. Eligible plans generate source through `triton` or `cutlass_cute_dsl`, or select the admitted `checked_cuda_asset`.

The checked-asset route currently names `cake_tinygemm2_stage4_split_k`; it is a fixed linear-layer source, not arbitrary CUDA generation. `generated` reports this distinction. Analysis covers declared rules only: backend registers or implicit shared memory require compiled or device evidence.

## The agent and evidence loops

Lab prepares TASK.md for the problem and AGENTS.md for tool rules. An external Ralph controller supplies evidence-derived state and enforces budgets. AI submits candidates; the evaluator independently checks them. Earlier immutable candidates survive later edits.

`matched_search` handles fixed-task search. A separate `portfolio` Study combines qualified specialists through the same Lab path. Artifact-only optimization is a claim scope, not a third runtime. Serving needs later integration and evaluation.

Correctness, measurement stability, and application benefit are different facts. Faster operator code does not by itself make a model or service faster. Compiler changes happen between frozen Campaigns and update types, verification, analysis, and lowering together, followed by the full Corpus and independent review. Executor fixes a different closure: Lab, evaluation, evidence tools, and environment. Read the [Glossary](GLOSSARY.md) and [maintenance guide](wiki/maintaining.md) for exact ownership.
