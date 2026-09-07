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

Format/type checks precede dependency, address, resource, and hardware checks. Assessment separates structural acceptance from backend eligibility and contains localized Findings. Each Finding retains its contract category, severity, and separate acceptance/lowering dispositions. `Assessment.findings` retains the blocking/report observations checked by the existing Corpus Gate; `Assessment.guidance` carries nonblocking hints. CLI, Lab, and profile reports expose both without treating hints as acceptance evidence or GPU measurements. Eligible plans generate source through `triton`, `cutlass_cute_dsl`, or `metal`.

The dedicated `checked_cuda_asset` route is retired. Its original TinyGEMM2 Schedules remain explicit structure-refusal cases; replay old fixed-source observations at their pinned Git revision. Current `Lowering.generated` is true, while historical values retain their meaning. Analysis covers declared rules only: backend registers or implicit shared memory require compiled or device evidence.

### Internal ownership and adding a backend

`core.py` connects the public interface; `revision.py` admits a Revision and `corpus.py` compares each observed case with its expected result. Diagnostic types belong to `diagnostics.py`. The four common rule classes belong to `verifier/`; each backend owns its representation and control refusals without turning an expressible Schedule into an IR rejection.

[BACKENDS](../../src/open_cake_ir/compiler/backends/__init__.py) is the single static backend inventory. A backend implements `requirements`, `preflight`, and `emit`; Triton owns `pointer_type(DType)` for its compile signature. To add a backend, define its target, supported inputs and refusal conditions, implement that protocol, and register it once. Test actual supported and refused combinations. The CLI vocabulary view reads this same inventory. Full Corpus and independent review precede a successor release; registration alone establishes no device support.

[performance](../../src/open_cake_ir/compiler/performance/__init__.py) owns work, residency, profiling, compiled resources, empirical cost, ranking and utilization. Same-input intermediate results are derived once and passed explicitly without a global cache. New `tools/profile_lowered_kernel.py` reports use schema 2: `predicted.registers_per_thread_lower_bound` and `verdict.register_floor_sound` are removed; the measured physical-register occupancy-limit label is `registers`. Values, units and evidence domains remain unchanged, and historical schema 1 reports are not rewritten.

## The agent and evidence loops

Lab prepares TASK.md for the problem and AGENTS.md for tool rules. An external Ralph controller supplies evidence-derived state and enforces budgets. AI submits candidates; the evaluator independently checks them. Earlier immutable candidates survive later edits.

`matched_search` handles fixed-task search. A separate `portfolio` Study combines qualified specialists through the same Lab path. Artifact-only optimization is a claim scope, not a third runtime. Serving needs later integration and evaluation.

Correctness, measurement stability, and application benefit are different facts. Faster operator code does not by itself make a model or service faster. Compiler changes happen between frozen Campaigns and update types, verification, analysis, and lowering together, followed by the full Corpus and independent review. Executor fixes a different closure: Lab, evaluation, evidence tools, and environment. Read the [Glossary](GLOSSARY.md) and [maintenance guide](wiki/maintaining.md) for exact ownership.

Concrete implementations live under `src/open_cake_ir/tasks/`. Tasks supply contract validation, oracles and preparation; the common Lab and Evaluation never import concrete tasks. See [task ownership](TASKS.md).
