# How an experiment runs

[中文原文](../../wiki/experiments.md) · [English home](../README.md)

Freeze the problem and judging rules first, let AI revise candidates, then retain independently replayable results. Exact commands live in the [Runbook](../../RUNBOOK.md).

[Guide index](README.md) · [Results](results.md)

## Two improvement loops

Candidate evolution changes tiling, roles, or composition for the same problem under one frozen Compiler. Compiler evolution fixes an expressibility or checking gap, updating types, rules, analysis, and lowering before a successor release. Changing the question, compiler, and timing together loses attribution.

A search fixes task and budget, obtains candidates, checks and compiles, compares complete required outputs, measures under the frozen protocol, preserves diagnostics, and either continues within budget or audits a terminal Run. Invalid candidates never proceed to timing. Cost ranking needs matching released calibration; it cannot replace measurement.

AI reads TASK.md and AGENTS.md and submits candidates. External Evaluation judges; Ralph tracks time, tokens, and attempt limits. Budget exhaustion can be a normal end.

## Before running

| Check | Authority |
| --- | --- |
| Inputs, mathematics, outputs, tolerance, cases | Workload Contract |
| Environments, repetitions, budget, stopping | Study Contract |
| Compiler and full Corpus | Released lock and Gate |
| Lab, evaluator, and host closure | Executor descriptor |
| Actual AI binary and frozen capabilities | Provider qualification |
| Exact versions for this execution | CampaignLock produced by preflight |
| GPU admission and new outputs | Controlled runtime configuration and external Evidence root |

Read [current release status](../../../reports/current/STATUS.md). The two no-GPU checks retained from the Chinese guide are:

```bash
.venv/bin/python tools/render_current_status.py --check
.venv/bin/open-cake-ir compiler check-corpus --format text \
  --revision compiler/revision.lock.json
```

Neither allocates a GPU or calls AI. A real run still needs the actual host, toolchain, driver, and controlled allocation. Old absolute paths do not prove current availability. Infrastructure templates contain simulated provider/toolchain facts and cannot be relabeled as a live environment.

## Fair timing and retained evidence

Use the same task, hardware, and declared timing boundary; pass correctness before samples. Baseline/candidate order can matter, so follow the contract's paired or interleaved protocol and retain raw cohorts. Profiler adds overhead and is diagnostic, not ordinary latency.

Retain immutable candidates, actual arguments, findings, build products, complete-output checks, raw timing, and stop reasons outside source. Keep failure, invalidity, not-run, and external faults distinct. Scientific summaries follow the preregistered Study, not the single best row found afterward.
