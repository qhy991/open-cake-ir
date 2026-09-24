# Python artifact optimization — scaffold v3

Use the supplied Workload Contract and Python starter to propose structurally distinct
Schedules for the exact target, backend and shape. Preserve the tensor ABI, arithmetic,
rounding, output ownership and lowering route. Submit each Python source as one
`python_source` member of the candidate-set envelope named by the StateCard.

The Lab already holds the frozen Workload. Do not copy or invent a
`workload_contract_sha256` in `@cake.schedule(...)`: after checking the target,
lowering route and public tensor ABI, the Lab binds the correct content identity.
An explicit conflicting identity is rejected. The Compiler can still inspect and lower
an unbound Schedule independently of the Lab.

Use the exact Target's instruction contracts and execution-group width; do not infer
support from a different device or backend. Backend diagnostics name the limits this
Compiler revision actually enforces. Do not replace a refused operation with an
unverified approximation or change the Workload's dtype or tolerance.

Generate a faithful complete candidate and a structurally distinct alternative when
expressible. A different name, formatting change, or execution-group count alone is
not a structural alternative. Local analyses and cost estimates filter candidates;
only the common Evaluation's oracle, paired timing and available profiler evidence
support correctness and performance conclusions.

The Lab owns compilation, GPU allocation, evaluation and stopping. Do not invoke
those tools, a GPU, the network or another task. Change only the named candidate
envelope; keep TASK.md and AGENTS.md unchanged. Do not submit serialized Schedule
JSON or a low-level baseline.

This v3 scaffold is a separate authoring treatment from v2. Keep its Runs separate;
do not pool or relabel prior evidence. One confirmed candidate establishes no
cross-shape or serving result.
