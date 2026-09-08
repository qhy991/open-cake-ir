# Organizing existing evidence into rubric feedback

[中文](../zh-CN/RUBRIC_FEEDBACK.md) · [English index](README.md) · [Acceptance gates](../ACCEPTANCE_GATES.md)

**The rubric is an evidence checklist for the Agent. It organizes feedback already visible to that arm, separating recorded results, missing evidence and suggested next checks.** It has no weighted score and makes no LLM judge call.

For example, a candidate refused by a backend support check needs the specific refusal examined. No GPU answer has been obtained, so this cannot be described as a GPU correctness failure. A candidate with stable exploratory timing still needs common Evaluation confirmation before it supports a confirmed latency claim.

## How it fits the existing path

Lab derives `rubric` from the StateCard's `previous_feedback` and adds it to the existing provider reference bundle. The Agent request uses that bundle and Evidence retains the same content. Raw feedback remains present. The checklist only organizes the current feedback; it reads no extra files, other-arm records or restricted reference implementations.

The implementation entry point is `rubrics.derive_rubric` in [Lab](../../src/open_cake_ir/lab/); the [task package](../../src/open_cake_ir/lab/task_package.py) owns delivery and retention. Equal input produces equal output. It uses the existing `matched_search` and Ralph task path, with no new configuration switch, runtime mode or provider call.

The bundle's `rubric` has `kind: evidence_rubric_v1` and `schema_version: 1`. Its `source` is `/state_card/previous_feedback`; each entry in `criteria` contains `criterion`, `state`, `evidence` and `guidance`. The `evidence` entries are JSON pointers into that bundle's raw feedback, so readers can check the source. `guidance` supplies fixed reading or verification advice. It is neither an executed change nor a promise that a proposed change will work.

| Criterion | What to read | What it does not establish |
| --- | --- | --- |
| `admission` | Whether the record is initial, refused or an existing observation | Admission implies a correct result |
| `correctness` | Whether feedback reports qualification or a correctness rejection | Static success proves GPU correctness or framework E2E acceptance |
| `measurement` | Whether measurement occurred and reported timing is stable | Stable means faster, or exploratory timing is confirmed |
| `confirmation` | Whether a confirmation result is reported | An unconfirmed low latency can authorize promotion |
| `profile` | Whether current feedback reports profiling material | An artifact's presence proves a bottleneck or mechanism |
| `findings` | Existing diagnostics, their locations and whether they block | Every diagnostic blocks, or no unreported problem exists |

These are separate observations, with no overall pass or total score. Missing observations, unrecognized evidence and malformed feedback retain their corresponding absent, unknown or malformed states. They do not become success. An initial request without a candidate result does not invent a failure.

## Correctness and performance retain their owners

The Workload Contract still owns semantics, inputs and oracle. The Verifier only blocks within its modeled domain; its success does not cover resources it did not analyze. Common Evaluation still owns external-reference correctness, paired timing and fresh confirmation. Candidate selection, promotion, budgets, stopping and diagnosis routing keep their existing authorities. The rubric does not participate in those decisions. Budget information stays in the existing StateCard, without a new efficiency score.

In particular, distinguish these cases:

- **Backend coverage gaps.** Unsupported register or residency controls describe a limit of the current representation or lowering. They are not GPU correctness failures and do not authorize an architecture downgrade.
- **Profiling material exists.** A profile record only reports the availability of material. A bottleneck claim needs the actual metrics and their measurement domain; a causal explanation also needs a controlled comparison.
- **Known evidence versus a proposed change.** “Timing is unstable” is an observation in the record. “Check measurement conditions before comparing” is advice. Until acted upon, that advice provides no new measurement or causal evidence.

This change belongs to Lab feedback presentation. It does not alter the Compiler Revision, IR vocabulary, Verifier rules or frozen historical Evidence. A Campaign continues to use its fixed Compiler. A missing primitive, if later demonstrated, needs the separate Compiler evolution, Corpus Gate and review process. Changes to replayable Lab runtime source must also follow the existing Executor successor lifecycle; old release identities are not rewritten.

## A future test of whether the feedback helps

**The following is a future experiment design, not a result established by this implementation.** The question is whether organizing the same evidence helps the Agent reach a correct, confirmed candidate sooner and repeat fewer failures.

Within one fixed Authoring Environment, compare two feedback treatments: deliver the old feedback as before; deliver the same old feedback plus its derived rubric. Both receive the same raw evidence and obey the same reference-access rules. Keep feedback presentation separate from a Cake-versus-native-code treatment.

Prespecify the task, single target shape and input distributions, oracle and tolerances, initial candidate, exact Compiler Revision, backend, toolchain, GPU, Agent model and reasoning effort, scaffold, candidate selection, timing boundaries, budgets and stopping rules in the Study. Arrange matched independent repetitions and run order. Additional feedback tokens count toward the same budget, without compensating calls or evaluations.

The two feedback implementations can bind the Executor Revisions before and after the rubric change. Check that feedback presentation is the only experimental difference, with other execution and evaluation rules matched and each revision's qualification requirements satisfied. This does not promise a new runtime toggle.

| Question | Prespecified observation |
| --- | --- |
| How soon is the first correct candidate reached? | Turns, recorded elapsed time, provider tokens and evaluation count to the first oracle-correct result; retain failures or missing outcomes when it is not reached |
| Are the same failures repeated less often? | Repeated failures grouped by a prespecified refusal reason and diagnostic location; record environment faults separately from incorrect candidates |
| Is the final result faster? | Common fresh-confirmation latency at the same budget; report qualification rate separately from conditional confirmed latency, without substituting the exploratory minimum |
| What does the benefit cost? | Provider tokens, total elapsed time, evaluation count and measurement cost; unavailable time or cost remains unknown |

Report the distribution across matched repetitions, failures, missing outcomes and stopping reasons. Do not select only successful Runs or add ad hoc reruns for environment faults. Explaining a speed change's mechanism additionally needs matched profiling and controlled-change evidence. The Study owns treatment and analysis; the CampaignLock owns execution bindings. This page is not a new contract or an experiment launcher.

Source code and tests for this change can at most establish that the projection respects evidence boundaries, the Agent request and retained bundle contain the same feedback, and existing decision rules still apply. Live provider qualification, GPU results, Agent benefit, IR superiority and target-framework E2E acceptance each need their own validation. See the [paired CuTeDSL path](PAIRED_CUTE.md) and the [decision to establish feedback before adding orchestration](../adr/0010-feedback-before-agent-orchestration.md).
