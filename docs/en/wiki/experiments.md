# How an experiment runs

[中文原文](../../wiki/experiments.md) · [English home](../README.md)

Freeze the problem and judging rules first, let AI revise candidates, then retain independently replayable results. Exact commands live in the [Runbook](../../RUNBOOK.md).

[Guide index](README.md) · [Results](results.md)

## Choose an entry by purpose

Workload owns mathematics and the oracle; Compiler owns complete Programs, leaf Schedules
and explicit rewrites; Run freezes execution permissions and budgets; Study preassigns
Runs and analyzes their outcomes. Evaluation observes correctness and measurement, and
Evidence retains the original records. Engineering and research share one Run engine;
CampaignLock is a legacy input adapter.

There are three execution entries: fixed-candidate Evaluation for qualification and
diagnosis, an independent Run for bounded agent search and fresh confirmation, and Study
for preassigned repeated comparisons. The following are purposes, not new runtime modes:

| Purpose | Inputs | Question answered |
| --- | --- | --- |
| Optimize an existing implementation | Workload, starter/incumbent, exact target, baseline and budget | Can the agent produce a confirmed improvement on this task and target? |
| Reproduce an external implementation in Cake | Original semantics/oracle, declared reference and a separately bound external baseline | Can Cake recover its important structure, and what explains the remaining gap? |
| Generate from the specification | Specification, oracle, hardware/API contract and restricted reference access | What can the agent discover without a complete target implementation? |
| Transfer engineering knowledge | Destination task plus authorized mechanism material, evidence and transform grants | Does the mechanism apply and help after destination-specific adaptation? |
| Controlled comparison or ablation | Frozen assignments, splits, model, budgets, repetitions and analysis | Which treatment explains the observed effect? |
| Capability assessment and diagnosis | Fixed Program, reference mechanism or component probes | Is the gap in expression, lowering, toolchain, correctness or measurement? |

Ordinary task batches use independent Runs. External reproductions first check capability;
transfer discovery and destination adaptation precede frozen held-out studies. A Compiler
gap becomes a Finding and a separately verified successor commit between Runs, never a
silent change to a running experiment's compiler, oracle or measurement policy.

Reference permissions are orthogonal: clean_start excludes complete target implementations,
known_kernel_reproduction permits declared references, and direct_low_level governs native
authoring. The ordinary launcher supplies a starter and defaults to known_kernel_reproduction;
it is not clean-start evidence. Material and tool isolation must be enforced, not merely requested.
`--reference-access clean_start` delivers an empty Schedule derived from the Workload ABI,
mathematics and API documentation. The evaluator privately retains the complete starter
for fixed-baseline validation. Live author processes also need file access isolation
from reference implementations, prior sessions and other tasks.

E/P Studies independently control extra mechanism material and explicit transform calls.
All four cells share base IR/backend capabilities and validation. P0 may manually construct
the same optimization; P1's API itself contains knowledge. The
[ablation protocol](../../OPTIMIZATION_TRANSFER_ABLATION.md) owns exact statistical rules.

## Implementation and acceptance scope

Existing single-kernel routes remain target-specific. Complete-Program correctness uses
CUBIN, HSACO and MCFATBIN loaders; ordinary Run measurement still requires an implemented
whole-program adapter. Standalone MACA Program profiling is attribution, not a latency
sample or permission to reuse a single-dispatch timer. Platform records and original
evidence own remaining Metal composition, measurement and transfer-effect limitations.

Use launch_task.py for one task, launch_task_matrix.py for a batch, kernel_experiment.py
for explicit multi-node organization, lab run preflight/execute/audit for frozen Runs,
and transfer_study.py for E/P Studies. rewrite_collection.py assess is component-only;
qualify_tensor_program.py build/evaluate/profile handles admitted native Program assays.
Keep outputs outside source. A smoke or system-qualification report separates entrypoint
completion, correctness, measurement validity, baseline improvement and final confirmation;
it establishes no scientific transfer effect by itself.

## Two improvement loops

Candidate evolution changes tiling, roles, or composition for the same problem under one frozen Compiler. Compiler evolution fixes an expressibility or checking gap, updating types, rules, analysis, and lowering before a successor release. Changing the question, compiler, and timing together loses attribution.

A search fixes task and budget, obtains candidates, checks and compiles, compares complete required outputs, measures under the frozen protocol, preserves diagnostics, and either continues within budget or audits a terminal Run. Invalid candidates never proceed to timing. Cost ranking requires an explicitly bound empirical model with matching context; otherwise author order is retained. Estimates cannot replace measurement.

AI reads TASK.md and AGENTS.md and submits candidates. External Evaluation judges; Ralph tracks time, tokens, and attempt limits. Budget exhaustion can be a normal end.

Ordinary launches default to turn, time, compilation and evaluation limits. Provider tokens
remain recorded but do not stop or disqualify a Run unless `--token-budget` is supplied.
The frozen representation is `budget.limit: null` and an empty `checkpoints` list.
Already frozen Runs and their historical outcomes retain their original policy.

## Before running

| Check | Authority |
| --- | --- |
| Inputs, mathematics, outputs, tolerance, cases | Workload Contract |
| One optimization's environment, permissions, budget and stopping | RunSpecification |
| Research allocation, repetitions and analysis | StudyPlan when conducting a study |
| Compiler and full Corpus | Clean commit, `compiler/revision.json`, and Corpus Gate |
| Lab, evaluator, and host closure | Source commit and exact-target host capture |
| Actual AI binary and frozen capabilities | Provider qualification |
| Exact versions for this execution | Frozen Run inputs; legacy CampaignLock adapts into the same engine |
| GPU admission and new outputs | Controlled runtime configuration and external Evidence root |

Read [current release status](../../../reports/current/STATUS.md). The two no-GPU checks retained from the Chinese guide are:

```bash
.venv/bin/python tools/render_current_status.py --check
PYTHONPATH=src .venv/bin/python -m open_cake_ir.cli compiler check-corpus \
  --revision compiler/revision.json
```

Neither allocates a GPU or calls AI. A real run still needs the actual host, toolchain, driver, and controlled allocation. Old absolute paths do not prove current availability. Infrastructure templates contain simulated provider/toolchain facts and cannot be relabeled as a live environment.

## Fair timing and retained evidence

Use the same task, hardware, and declared timing boundary; pass correctness before samples. Baseline/candidate order can matter, so follow the contract's paired or interleaved protocol and retain raw cohorts. Profiler adds overhead and is diagnostic, not ordinary latency.

Retain immutable candidates, actual arguments, findings, build products, complete-output checks, raw timing, and stop reasons outside source. Keep failure, invalidity, not-run, and external faults distinct. Scientific summaries follow the preregistered Study, not the single best row found afterward.
