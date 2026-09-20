# The system: from an idea to a verifiable GPU program

[中文原文](../ARCHITECTURE.md) · [English home](README.md)

open-cake-ir makes computation choices explicit, locates errors, and preserves evidence for improvements. This page explains stable responsibilities. Read the [generated status](../../reports/current/STATUS.md) for released identities.

## Why write a Schedule?

Even summing each row of a table involves choices: which threads handle a row, how large each tile is, whether data can be reused, and whether writers collide. A Schedule describes those choices between the mathematical task and final machine code. The Compiler checks it and generates source; external checks and GPU measurements decide correctness and speed.

| Part | Responsibility | Main flow |
| --- | --- | --- |
| Compiler | Check complete programs and leaf schedules, apply explicit rewrites, generate source | Program / Schedule → assessment and lowering |
| Research Lab | Execute frozen permissions and budgets; allocate and analyze research conditions | Workload + RunSpecification → Run |
| Evaluation | Check answers, measure, and collect diagnostics | Sealed candidate → receipts |
| Evidence | Preserve observations and rebuild conclusions | Objects → audit → report |

The Compiler works independently. Lab uses it, Evaluation, and Evidence; the Compiler never imports experimental or provider logic.

## Separate task, implementation, execution and research ownership

For row normalization, Workload fixes mathematics, reference and tolerance. Schedule assigns threads, storage and operations; Program owns complete stage order, public tensors and bindings. RunSpecification freezes one optimization's versions, authoring environment, material and transform permissions, budgets and evaluation. Engineering Runs need no Study. Study preassigns the same Runs to conditions and owns their statistical analysis. Changing an implementation need not change the problem; changing scoring rules after results destroys comparability.

## The compiler path

Format/type checks precede dependency, address, resource, and hardware checks. Assessment separates structural acceptance from backend eligibility and contains localized Findings. Each Finding retains its contract category, severity, and separate acceptance/lowering dispositions. `Assessment.findings` retains the blocking/report observations checked by the existing Corpus Gate; `Assessment.guidance` carries nonblocking hints. CLI, Lab, and profile reports expose both without treating hints as acceptance evidence or GPU measurements. Eligible plans generate source through `triton`, `cutlass_cute_dsl`, `native_cuda`, or `metal`.

The dedicated `checked_cuda_asset` route is retired. Its original TinyGEMM2 Schedules remain explicit structure-refusal cases; replay old fixed-source observations at their pinned Git revision. Current `Lowering.generated` is true, while historical values retain their meaning. Analysis covers declared rules only: backend registers or implicit shared memory require compiled or device evidence.

### Internal ownership and adding a backend

`core.py` connects the public interface and composes lowering; `ir/program.py` owns typed composition and dataflow, `program.py` owns LoweredProgram and its code binding, and `program_passes.py` owns complete-program rewrites. `revision.py` admits a Revision and `corpus.py` compares observed cases with expectations. Diagnostic types belong to `diagnostics.py`. The four common rule classes belong to `verifier/`; each backend owns its representation and control refusals without turning an expressible Schedule into an IR rejection.

[BACKENDS](../../src/open_cake_ir/compiler/backends/__init__.py) is the single static backend inventory. A backend implements `requirements`, `preflight`, and `emit`; Triton owns `pointer_type(DType)` for its compile signature. To add a backend, define its target, supported inputs and refusal conditions, implement that protocol, and register it once. Test actual supported and refused combinations. The CLI vocabulary view reads this same inventory. Full Corpus gates a successor commit; integration and independent review follow the [branch workflow](../DEVELOPMENT_BRANCHES.md). Registration alone establishes no device support.

[performance](../../src/open_cake_ir/compiler/performance/__init__.py) owns work, residency, profiling, compiled resources, empirical cost, ranking and utilization. Same-input intermediate results are derived once and passed explicitly without a global cache. New `tools/profile_lowered_kernel.py` reports use schema 2: `predicted.registers_per_thread_lower_bound` and `verdict.register_floor_sound` are removed; the measured physical-register occupancy-limit label is `registers`. Values, units and evidence domains remain unchanged, and historical schema 1 reports are not rewritten.

## The agent and evidence loops

Lab freezes task material, rendered as TASK.md and AGENTS.md for CLI authors. A confined message author receives only permitted material and its own Run history. An external Ralph controller supplies evidence-derived state and enforces budgets. AI submits candidates or explicit transformation requests; the evaluator independently checks the resulting candidates. Earlier immutable candidates survive later edits.

Engineering optimization directly prepares a Run. A `matched_search` Study preassigns Runs; the legacy CampaignLock is an input adapter to the same search, budget, confirmation and audit engine. The former `portfolio` Study is retired under ADR 0071; historical replay uses its original commit. Serving needs later integration and evaluation.

Correctness, measurement stability, and application benefit are different facts. Faster operator code does not by itself make a model or service faster. Compiler changes happen between frozen Campaigns and update types, verification, analysis, and lowering together, followed by the full Corpus and the integration review specified by the [branch workflow](../DEVELOPMENT_BRANCHES.md). Executor fixes a different closure: Lab, evaluation, evidence tools, and environment. Read the [Glossary](GLOSSARY.md) and [maintenance guide](wiki/maintaining.md) for exact ownership.

Concrete implementations live under `src/open_cake_ir/tasks/`. Tasks supply contract validation, oracles and preparation; the common Lab and Evaluation never import concrete tasks. See [task ownership](TASKS.md).

## Executable optimization knowledge across hardware

The framework proposes turning agent-discovered fusion, tiling and memory-hierarchy
mechanisms into explicit guarded rewrites. Destination backends supply hardware-specific
implementations; Lab retunes parameters and verifies benefit. Extra mechanism material and
callable rewrites are separate experimental factors while base capabilities and validation
remain fixed. Complete Programs, independent Runs, controlled material/pass access and E/P
allocation have software implementations; cross-hardware effects remain unverified.
See the [mechanism and ablation design](OPTIMIZATION_TRANSFER.md).

## Lab lifecycle ownership

`lab/core.py` connects the public API and retains its six existing dependencies: project
root, clock, workload loader, schedule preparation, authoring validation and manifest
parser. Phase functions receive what they use directly, without a second context object.

See the [Lab implementation map](../../src/open_cake_ir/lab/README.md) for module ownership.

Execution records observations; replay independently checks their raw support. Shared
pure calculations do not replace either trust boundary. A first-provider fault returns
before task-package or empirical-model resolution. Authority checks precede Evidence
creation, and reporting invokes replay callbacks only when needed. Concrete tasks and
task wiring remains in `TaskLab`; public imports remain `open_cake_ir.lab`.
