# Evaluation context

Evaluation owns the common assay after an Authoring Environment seals a LaunchableCandidate.

## Language

**LaunchableCandidate**:
A target, entry point, launch manifest and complete role-labelled artifact byte set ready for common evaluation.
_Avoid_: Arm output, source filename

**Evaluation Protocol**:
The frozen case, purpose, correctness-before-timing order and measurement boundary.
_Avoid_: Runner settings, Study analysis

**Evaluation Receipt**:
Raw correctness output, launch receipt, samples and derived disposition for one candidate/purpose.
_Avoid_: Candidate score, endpoint

**Logical Evaluation Attempt**:
One immutable candidate evaluation retaining every broker job, with at most one exact zero-work admission
resubmission.
_Avoid_: Retry mode, replacement candidate

**Portfolio Artifact**:
The sole exact semantic-key-to-candidate manifest used by dispatch and replay.
_Avoid_: Shape table, registry

**Measurement Quality**:
Stability derived from retained cohorts for one measurement boundary, independent of correctness.
_Avoid_: Candidate failure, performance claim

## Relationships

- Both Authoring Environments cross the same LaunchableCandidate boundary.
- Workload Contract supplies case materialization, oracle and tolerances.
- Correctness precedes timing; search and confirmatory receipts remain distinct.
- Portfolio dispatch derives keys from Workload cases and rejects unsupported keys before launch.
- Evidence stores raw receipt/artifact bytes; Evaluation never defines an Estimand or Claim View.

## Flagged ambiguities

- “source” is split into authored/lowered input, compiler-expanded source, IR, PTX, CUBIN and SASS roles.
- “unstable candidate” is invalid language when only timing CV failed; correctness and Measurement Quality are
  orthogonal.
