# ADR 0010: Establish feedback before adding agent orchestration

Status: accepted, 2026-08-24.

## Context

Tool-rich artifact Campaign v3 exposed the provider's full catalog but emitted no auxiliary-agent lifecycle. More
importantly, each first Turn crossed the 150k provider-token boundary, so neither Run received Evaluation feedback.
Adding a manager or requiring worker fan-out would therefore add another invocation, state owner and token-accounting
path without governing an observed next action.

KDA-style orchestration contributes a smaller invariant that is useful now: one worker owns candidate writes, a
read-only decision role may only steer later work, and an external judge alone owns correctness and performance.
The current Lab already has the first and third authorities: the resumed provider thread is the sole Candidate writer,
and Evaluation Receipts are the sole promotion authority.

## Decision

First demonstrate one bounded feedback transition on the existing path. A live artifact-only successor may replace
its inherited token grid with one explicit terminal provider-token checkpoint and a hard maximum-Turn count. Both
values must be supplied together, and this release operation is unavailable to system-qualification or scientific
Studies. The first such successor uses two Turns; Turn 2 receives only the Lab-owned build, measurement and attribution
feedback already declared by the Study.

Do not add a manager, persona state, new runtime mode or mandatory in-Turn subagents. Auxiliary agents remain permitted
read-only research activity, but their presence is not a result and is not inferred from feature exposure. The primary
provider thread remains the only Candidate-envelope writer.

A later manager is justified only after retained multi-Turn Evidence shows a repeated steering decision that cannot be
expressed by the existing feedback document. If introduced, it must be a fresh read-only invocation, choose from a
closed semantic vocabulary, provide no candidate implementation, use no GPU, and have its tokens and output sealed.

## Consequences

- The next experiment can answer whether measured feedback changes a candidate without adding another authority.
- Maximum Turns, rather than a hoped-for token estimate, is the hard provider-call bound.
- The terminal token checkpoint remains observational and turn-discrete; overshoot does not backfill it.
- Artifact promotion remains non-scientific and per-Run. The new horizon cannot support an arm comparison or paper
  claim.

## Validation

Artifact Campaign v4 binds Study SHA `3e698c91c152bfabe03ee7c4b9abe545e8b391cbc68de959d9881931de43b124`
and Campaign Lock SHA `45c2d30e39e14e7f0afd732f5efe0494e12a07d408826eb50a68d22f94ce4611`.
Both Runs adhere and execute exactly two same-thread Turns. Open Cake changes its second candidate set in response to
retained timing/NCU feedback
and improves its own confirmation from 1.851297 ms to 1.417239 ms. Direct CUDA also changes its second set but regresses
from 4.156109 ms to 8.814379 ms, so the per-Run promotion correctly retains Turn 1. Fresh audit reports archive
integrity and semantic replay true, zero missing Runs, and no Estimand. No Turn emits an auxiliary-agent lifecycle;
the Campaign validates feedback and role separation, not multi-agent effectiveness.

## KDA-internal reference audit, 2026-08-25

The current local KDA-internal experiment supplies a concrete future design without
changing this decision. Its persona flow gives a fresh, read-only manager only a compact
progress snapshot; the manager selects exactly one of `neutral`, `explore`, `exploit` or
`recover`, gives no technical solution, uses no tools or GPU, and falls back to neutral
when its structured output is invalid. A separate worker owns edits, while the
out-of-container judge remains authoritative. The manager decision and rendered persona
are archived per round.

Those mechanics satisfy the constraints above better than an unconstrained supervisor or
several writable workers would. They are a reference, not an authority imported into this
Lab: the inspected KDA-internal working tree is itself under active local modification,
and its continuous kernel-optimization rounds have a different state machine from a
frozen Open Cake Study. Adding the flow here would still require a Study-owned invocation
budget, sealed manager output/token use and retained evidence that the existing Lab
feedback leaves a recurring steering decision unresolved. None is currently present, so
this audit adds no manager field, mode, adapter or provider call.
