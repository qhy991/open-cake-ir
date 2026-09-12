# ADR 0064: Task incumbents are an append-only Lab projection

Status: accepted; requires successor Executors before live use.

## Decision

Artifact-optimization Campaigns may advance one incumbent for an exact comparison cell.
The cell key is Workload id and digest, case, exact target, lowering backend and complete
Evaluation Protocol digest. Compiler Revision is deliberately not part of the key: a
sealed incumbent artifact can remain the black-box opponent across Compiler ticks, while
the Campaign Lock still records the Compiler that produced each new candidate.

The promotion registry is an external custody-bearing EvidenceStore. Each promotion is
one sealed Run containing the complete launchable artifact objects, its confirmatory
receipt, its predecessor and a reference to the original Campaign evidence. The current
incumbent is derived by replaying the contiguous predecessor chain. There is no mutable
`current.json`, task score table or second performance ledger. CAS copies make the
incumbent self-contained; the original Campaign remains the observation authority.

A promotion requires selected-Run semantic replay, Archive Integrity, Filesystem Custody,
protocol adherence, external-oracle correctness, one kernel call, no fallback, stable
confirmatory timing and the Study-declared material win over the Campaign's fixed
baseline. After genesis, that fixed baseline must be the current incumbent exactly.
Close-null, slower, search-only, unstable, faulted and wrong-baseline results cannot
advance the chain. Concurrent writers serialize the current-read plus append; an
incomplete promotion remains a refusal, not a record repaired in place.

`launch_task.py --incumbent-registry ROOT` resolves the exact cell after constructing the
Workload and Study. If an incumbent exists, it materializes the registry CAS objects into
the new external workspace and binds that bundle through the existing `fixed_baseline`
Campaign Lock leaf. If no incumbent exists for that exact key, the launcher records the
explicit fallback and prepares the task's starter reference. The matrix driver passes one
registry root; each task resolves its own key. `baseline-selection.json` is a derived
workspace view, while the Campaign Lock's candidate identity is the execution authority.

## Reference baseline and experimental meaning

The task starter remains the stable reference baseline. Rolling incumbents are for
`artifact_optimization_only`; scientific matched studies continue to bind their
preregistered fixed reference. Periodic reference-baseline campaigns measure Compiler
floor movement separately from candidate-minus-incumbent headroom. Replacing every
scientific baseline with the latest winner would destroy longitudinal comparability.

An incumbent is a black-box baseline, not automatically visible source. Clean-start and
direct low-level authoring arms keep their declared reference restrictions. A known-kernel
reproduction Study may separately expose source when its treatment explicitly allows it.
Promotion does not create a Compiler pass, a portfolio, framework acceptance or a serving
artifact.

## Verification

Contract tests cover genesis, a second chained promotion, exact-baseline enforcement,
materiality refusal, copied-registry custody refusal, artifact materialization, launcher
freezing and matrix delegation. Live verification requires promoting one existing
custody-bearing M4 Campaign, launching the same exact cell with the registry and observing
that the new Campaign Lock binds the promoted candidate as its fixed baseline.
