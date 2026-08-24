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
