# Metal evolution: Cake exploration and evidence notes

These instructions supplement the preceding complete Metal Python bundle scaffold.
TASK.md remains authoritative for the Workload, target, budget, reference access,
tool permissions and candidate output. This section grants no extra access.

Before proposing candidates, read the visible candidate feedback and Run-local
`optimization_history`, including omissions. Identify one falsifiable mechanism:
what changes in dataflow, load reuse, reduction structure, live intermediates or
work partition, and which observation would support or refute the change.

Inspect the supplied Cake API and diagnostics relevant to that mechanism. Try legal
canonical combinations of existing primitives through the permitted candidate
submission or granted transformations. The starter is an example, not the limit of
Cake's expressibility. Keep three questions separate: can Cake express the change,
does the lowering realize it, and does the device benefit? Renaming or changing
only the execution-group count is not a structural alternative.

When candidate-bound `generated_source` is actually delivered, inspect its exact
candidate, source Turn, stage, target and lowering route. Use the original line
numbers and CAKE_OP markers where present to compare one expected mechanism with
the emitted implementation. Metal's generated MSL is source code, not PTX, AIR or
Apple GPU machine code. Source alone does not establish physical register usage,
spills, occupancy or a performance gain. Missing, omitted or unavailable code and
profiler information remain unverified; do not infer their contents or invoke
extra tools to obtain them.

Keep a brief public hypothesis note at the start of each Schedule function body.
Name the Cake primitive or operation region, the expected lowering change, the
prior visible candidate/Turn evidence and the next check. Update the prior outcome
as supported, refuted or unknown. Do not provide private internal reasoning.
For a `cake.transform(...)` proposal, put the note beside that declaration and name
its proposal ordinal, parent and transformation. These notes remain in the original
author file; transform parameters and generated Program IR gain no new fields.

Carry forward only a short set of useful observations from this Run in the allowed
source file. A repeated failure is not automatically a Compiler bug. Distinguish
candidate errors, suspected verifier/lowering gaps, measurement limitations and
environment faults; cite actual diagnostic codes or visible receipts. Suggest the
responsible owner and a concrete follow-up when evidence warrants it.

Notes and source inspection do not select the winner, qualify a result or authorize
a Compiler change. Preserve negative and inconclusive outcomes. The Lab's fresh
confirmation and retained evidence remain authoritative. Write only the permitted
candidate file; do not create a memory database, alter TASK/AGENTS or change the
oracle, stopping conditions, reference policy or Compiler.
