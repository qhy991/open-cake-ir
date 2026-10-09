# Python candidate bundle optimization with retained research notes

Write one `candidate-set.py`. Import the Cake frontend once. Put complete
`@cake.schedule(...)` functions in the order that the Lab should consider them.
Use static `cake.program(...)` and `cake.stage(...)` for a permitted multi-stage
candidate. Use static `cake.transform(...)` only when this Run grants that call.
Keep the terminal response and the file format required by TASK.md.

## Before each proposal

Read TASK.md and the current StateCard. Read `optimization_history`, including
unselected candidates, measurement failures and omitted-entry counts. Read
`author_parents` before a transform call. Use only this Run's permitted materials.
If native context compaction occurred, recover these facts from the StateCard.
A memory of an earlier response is not a measurement record.

## Research notes in the permitted source

Use brief comments or a function docstring. State a public hypothesis and a
checkable result; do not write a private reasoning transcript. Do not create a
separate journal file, change JSON submission fields, or edit TASK.md/AGENTS.md.
The Lab retains each raw source file, including comments and docstrings.

Keep a short cumulative note at the top of the file, with at most six entries:

- State the mechanism tried, its candidate/Turn locator and the observed result.
- Distinguish rejected, incorrect, unmeasured, unstable, qualified search and
  independently confirmed outcomes. An unstable timing is not a slow kernel.
- State which hypothesis is still open and the next probe that can distinguish it.
- Preserve a useful negative result. Repeat that configuration only to test a
  stated change in conditions or a different hypothesis.

For each candidate or transform, add a short note with these fields:

- `Hypothesis`: the structural change and the expected resource or execution effect.
- `Based on`: the relevant candidate, Turn, Finding or measurement in this Run.
- `Evidence status`: observed fact, supported inference, or untested hypothesis.
- `Next check`: what correctness, timing or profiler result would support or reject it.

When the current interface blocks a concrete hypothesis, add:

- `Constraint`: the exact operation, access, dtype, loop or emitted-source region.
- `Attempt`: the smallest legal expression tried and the retained diagnostic.
- `Proposed owner`: author candidate, Lab recipe, Compiler pass, lowering,
  verifier, IR, or Evaluation. Mark uncertain ownership explicitly.
- `Compiler suggestion`: the needed capability or rewrite, its preconditions and
  a counterexample. Record `No promotion` when the evidence supports no change.

## Boundaries

Preserve the frozen Workload, public tensor ABI, exact Target and lowering route.
Keep all performance-relevant decisions visible. Submit structurally distinct
candidates; changes to names, comments or formatting are not new mechanisms.
Do not alter the frozen Compiler or evaluator during this Run. A native escape or
Compiler change requires a separate admitted Run or successor commit.

The external oracle, qualified paired timing and independent profiler own their
results. The Lab owns allocation, budgets, nomination and confirmation. Notes are
author observations and proposals; they cannot create a Finding, hardware fact,
performance result or permission to use withheld experience.
