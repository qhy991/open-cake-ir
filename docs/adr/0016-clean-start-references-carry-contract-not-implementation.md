# ADR 0016: Clean-start references carry contract, not implementation

Status: accepted, 2026-08-24.

## Outcome and non-goals

A matched clean-start reference fixture gives each arm the authoring contract needed
to begin, but no task implementation. The Open Cake starter exposes the external
buffers and output only; it deliberately has no program mapping, roles, scratch
allocations, pipelines, barriers or operations. The direct CUDA starter exposes the
canonical launch manifest and kernel signature, with a placeholder launch and an empty
body. Replacing references is one paired Study-successor operation.

This does not rewrite a frozen task-informed Study, add a paper/runtime mode, claim a
paper result, or reproduce the paper's unpublished isolation apparatus. The new fixture
still uses the local 150k budget and distinct `max` reasoning level; it settles only the
reference-access slice of a future clean-start protocol.

## Authorities and dataflow

The Workload Contract owns shapes, semantics and the oracle. The CUDA launch contract
owns the external argument ABI. The Compiler owns what constitutes a complete valid
Schedule. Each starter is only a projection of those authorities, while the frozen Study
owns the exact path and digest embedded in every Turn.

The Open Cake starter is intentionally rejected by `Schedule.from_dict` until an author
chooses exactly one program mapping and supplies a complete implementation. The direct
CUDA placeholder manifest is parseable so the host ABI is unambiguous, but its empty
kernel cannot pass external correctness. The Lab remains the external authoritative
Judge; an isolated provider workspace may return only its sealed candidate-set envelope.
This applies the useful KDA-internal ownership boundary without importing its manager or
agent state machine.

## Failure, compatibility and rollback

Existing full-skeleton Studies retain their bytes and task-informed meaning. A successor
that replaces only one arm's starter fails before output creation. Any implementation in
the Open Cake structural fields or CUDA function body fails the zero-GPU contamination
gate. The starter bytes are Study references rather than Executor source, so this change
does not manufacture an Executor revision.

Rollback means choosing a different paired reference bundle in a later successor; it
never means editing a frozen Study or historical Evidence.

## Acceptance evidence

The canonical successor tool must compute both reference digests, preflight the resulting
Study and refuse one-sided replacement. Contract tests must prove the Open Cake document
contains only the declared authoring interface and is not yet a valid Schedule, and prove
the direct CUDA manifest is canonical while its function body is empty.
