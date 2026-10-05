---
name: cake-metal-optimization
description: Optimize a fixed Apple Metal kernel through Cake schedules in a known-kernel Ralph task, using candidate-bound diagnostics and generated code to test structural changes and retain useful findings.
---

# Cake Metal optimization

Use this within a prepared Cake task. Read its TASK, AGENTS, StateCard and delivered
API/target material first. They own the workload, oracle, reference access, allowed
files, tool grants and budget. This skill supplies no extra permissions or baseline
implementation. It is not a compiler-editing or GPU-launch entry point.

## Explore the representation

Read the current candidate file and Run-local optimization history before editing.
Choose a falsifiable change to execution structure: work partition, data flow,
reuse/recomputation, reduction structure or fusion. Use only mechanisms supported by
the frozen target and delivered Cake API. Renaming a function, changing comments or
only sweeping group counts does not establish structural search.

Trace the relevant canonical primitives and their composition before proposing a
missing primitive. Keep hardware decisions inspectable in Schedule; do not invent
a layout algebra or ungranted pass. A refusal is localized evidence about its stated
contract, not proof that all equivalent programs are impossible. Check which rule
actually rejected the candidate.

The harness orders construction checks, verifier hard gates and cost-model ranking
before GPU evaluation. Respect that path and the task's candidate limit. An
uncalibrated target requires an explicit coverage limitation and the task's frozen
fallback ordering; never import another GPU's cost estimates or treat an estimate
as acceptance. Submit candidates through the existing task interface.

## Compare with the generated implementation

When candidate-bound source is delivered, map the changed Cake operation/region to
its generated stage and MSL. Inspect the mechanism relevant to the hypothesis:
load/store duplication, indexing and work mapping, live intermediates, reductions,
SIMD-group operations, synchronization or numerical transformations. Distinguish
what Cake expressed from what the compiler emitted.

MSL is source code. Metal IR/AIR is an intermediate compilation representation;
Apple GPU machine instructions are a further layer. PTX has no directly equivalent
public Metal text interface established by this task. A metallib or binary archive
is not itself readable instruction evidence. Inspect AIR or machine code only when
the exact toolchain and task provide it; otherwise mark that layer unobserved.

For a task that grants the matching offline Xcode toolchain, use the bounded
[AIR inspection recipe](references/air-inspection.md) to inspect your own generated
source. The recipe does not grant tool execution or access to other artifacts.

Shorter MSL and fewer logical live values do not prove fewer physical registers,
absence of spills or faster execution. Use external-oracle correctness, qualified
timing and profiler observations for the measured claims they actually cover.
Missing counters mean unobserved, not zero. Do not inspect a reference implementation
unless this arm explicitly grants access; candidate feedback grants no other arm's
source or baseline internals.

## Retain evidence that changes the next decision

Keep concise public experiment notes in the task's permitted candidate file. Put
notes inside each Schedule function when per-candidate extraction must retain them;
for transform actions use the task's documented action/source binding. Do not create
a parallel memory database or write protected Findings from the author process.

Record the hypothesis, changed operation/region, expected lowering, visible parent
or candidate identity, observation that would refute it, and the relevant feedback.
On the next turn classify it as supported, refuted or untested and change the next
candidate accordingly. Read the actual file again before replacing its bounded
candidate set. Preserve a few useful negative results; do not repeat a failed
program unless a declared measurement question justifies spending the budget.

Route a suggested fix to candidate, verifier, lowering/IR vocabulary, cost model,
Evaluation or Lab according to the evidence. Repeated messages are not independent
reproductions. Report unavailable attribution and environment failures separately
from performance failures. Do not infer a root cause from an unrelated blocking rule.

The maintainer may later cluster retained evidence into a compiler proposal. A useful
proposal names a recurring mechanism, a minimal reproducer, scope, counterexample
and missing analysis/legality rule. It is not permission to change the frozen
compiler. Compiler successors require the repository's P1–P8 review, full Corpus Gate
and independent human approval. The next matched experiment decides whether they help.
