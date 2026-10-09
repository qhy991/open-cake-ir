---
name: evolve
description: >-
  Coordinate bounded Cake kernel and compiler engineering rounds in open-cake-ir:
  review development tasks, diagnose maintenance opportunities, validate a successor,
  freeze an independent hardware Bench comparison, and select the next development
  round. Use for the 53-task development to Cake iteration to fixed-version Bench loop
  or read-only review of its evidence. Does not qualify hardware or turn engineering
  results into a formal version Study.
---

# Evolve

Complete the requested engineering round through the existing experiment owners. A
round can end with a useful kernel, a Compiler change, a material change, or No promotion.
No Compiler defect or speedup is required for the round to be valid.

Read the repository [Bench protocol](../../docs/BENCHMARK_PROTOCOL.md) and its
[branch workflow](../../docs/DEVELOPMENT_BRANCHES.md). The
[evolve design](../../docs/evolve-design.md) describes a larger proposed automated
protocol. Use only the implemented commands below; `advance`, native
`RevisionComparison`, and automatic recovery are not available.

## Recover the current round

Resolve the repository from this skill's location, including through a symlink.
Read the user's current goal, existing plan, allocation, compiler pin, launch owner,
terminal records, and original reference contracts. Preserve authorization already
given for this scope. A request to review or complete this skill alone does not start
another paid experiment.

For the external BW1100 two-host development allocation, use
[the evidence reader](../../tools/evolve.py) on each assigned host or on an explicit
local snapshot. Join observations against the original allocation, including missing
tasks. The reader does not connect to hosts itself. Exact commands and the supported
format are in [the evidence reference](references/evidence.md).

For native sealed Runs, use `tools/summarize_diagnoses.py --compiler-gaps` in their
compatible source checkout and inspect the retained candidate/evaluation/profiler
records. Semantic replay belongs to the producing Executor. Do not convert external
`ENDPOINT.json` files into native EvidenceStore authorities.

Infer progress from the owners' records. A process exit, a coordinator's `completed`
label, or an author's claim is insufficient by itself. Preserve terminal failures;
unknown attempts block automatic relaunch. A missing snapshot is unknown, not zero
cost or permission to reset a Run. Attach to the existing coordinator when it is live.

## Review development evidence

Review successful candidates, rejected candidates, unchanged starters, flat results,
and failed hypotheses. For each selected lead, retain the original candidate, emitted
source, exact workload/cases, assessment, evaluation, profiler, and negative examples.
State the smallest hypothesis and the observation that could disprove it.

Assign the maintenance owner before proposing a change:

| Evidence | Next action |
| --- | --- |
| An existing expression solves the problem | Keep the candidate; improve author guidance if reusable. |
| A repeated complete rewrite preserves semantics under known preconditions | Propose an explicit Compiler pass with counterexamples. |
| Legal IR cannot lower through the selected backend | Investigate backend lowering. |
| Required execution semantics cannot be expressed | Establish the missing commitment before changing IR, types and analyses. |
| A specific illegal behavior passes, or a legal behavior fails a rule | Fix the owning verifier rule with the minimal reproducer. |
| Generated source fails to compile for an unknown reason | Triage emission, toolchain and environment; do not assert an IR or verifier defect. |
| Measured ordering contradicts cost estimates beyond noise | Investigate the target's empirical cost model. |
| Oracle, precision, input ownership or timing differs | Fix the Workload/Evaluation boundary before claiming performance. |
| One kernel is faster but no reusable mechanism is established | Keep it as an incumbent or record No promotion. |

The 53-task starter is a development reference. Its diagnostic callable samples are
not community-Bench speedups. A low register count is evidence of resource use, not
proof of occupancy or end-to-end benefit. Counted errors are leads, not established
Compiler limitations. Historical routing remains historical; do not relabel old Runs.

## Make and validate the maintenance change

Use a tracking Issue and an independent task worktree under the branch workflow.
Never edit the Compiler, oracle, budget or measurement contract inside a frozen search.
Keep each maintenance change attributable to one supported hypothesis.

For a Compiler change, read P1–P8 in `AGENTS.md`, preserve the reproducer and negative
cases, run the affected contracts and full Corpus Gate at a clean commit, and obtain
independent review. Record source checks, device qualification and performance as
separate evidence. Merge or publish through the owning branch; do not substitute a
moving `main` for the version that actually produced a result.

Record the disposition in the owning Finding or investigation result. If nothing is
promoted, explain why and retain the useful candidate and failed attempts. A material
change is a material treatment; it is not a Compiler improvement. No new source or
material treatment means there is no reason to manufacture a version comparison.

## Freeze and run the independent Bench

Follow [the Bench reference](references/bench.md) before any comparison author starts.
Freeze the concrete Compiler conditions, Bench commit, exact task list, references,
author controls, complete allocation, time budget including confirmation, and stopping
rule. Use `tools/evolve.py freeze-bench` for the supported rolling fresh-search cohort.
This writes an immutable intent document; it does not prepare Runs or authorize launch.

Use existing platform launchers and device admission. Before launching, reconcile each
prepared Run's actual binding with its frozen allocation and controls. If the launcher
cannot represent them, fix and validate that launcher in a successor first. Do not
construct unsupported `StudyPlan` variants or pass this intent file as a native Run.
Read the installed `gpu-infra` lease lifecycle before device preparation or execution.
Reuse an existing qualified route; missing qualification is separate prerequisite work.

Use fresh author sessions and condition-specific directories. Keep the model, scaffold,
allowed material, reference access and budget equal. An agent improvement claim needs
fresh search; replaying `bench-best` measures the exported artifacts. Final heldout
research additionally needs qualified isolation and a preregistered Study; repeated
engineering Bench use is rolling validation.

The existing launch owner controls attempts, queues, timeouts and release. Inspect its
intake/deadline/terminal records before resuming. Never delete those records, reset the
time budget or replace an uncertain attempt. Two machines may execute disjoint frozen
allocations under their own qualified routes; the skill does not create another queue.

## Close the round and continue

Keep every assigned task and failure in the report. Report coverage/correctness first,
then qualified performance against the fixed reference, uncertainty and cost. Record
both Compiler and Bench commits. Retained starters, unsupported cases and missing
measurements remain visible. Do not report an aggregate only over the winners.

Read the raw Bench result and record a maintenance disposition in its existing owner:

| Result | Next-round decision |
| --- | --- |
| Correctness and applicable performance checks support adoption | Select the reviewed successor and name the accepted changes. |
| Correctness or coverage improves but performance is flat or worse | Make an explicit engineering tradeoff; preserve regressions and avoid a speedup claim. |
| Performance regresses without a compensating accepted benefit | Keep the current adopted version; retain the successor and negative evidence. |
| Noise, missing outcomes or protocol differences prevent a decision | Leave the decision unresolved; do not add favorable retries to the completed cohort. |
| No promotion | Keep the current version and state what evidence or new task would justify another round. |

Before starting the next 53-task round, name its new allocation, selected immutable
Compiler, original task contracts, model/scaffold, authorized budget and stop boundary.
State which development incumbents and experience are inherited, and validate them
against the selected version. Independent Bench reference artifacts stay fixed. A new
Bench commit, task population or reference contract starts a new comparison segment.
Continue automatically only within the user's already authorized round/resource scope.
Never infer authorization for unlimited rounds from a request to continue once.

Deliver the source/PR, evidence locators, selected next version, remaining uncertainty,
and next executable action. If only the skill was requested, finish its implementation
and validation without silently starting the next experiment.
