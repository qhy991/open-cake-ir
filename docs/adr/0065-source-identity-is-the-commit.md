# ADR 0065: Source identity is the commit, and a host capture is its own document

Status: accepted, 2026-09-16. Its "History stays where it is" clause is superseded by
[ADR 0067](0067-retired-release-outputs-live-on-the-history-branch.md). Supersedes the per-file closure and per-release approval
mechanisms of [ADR 0030](0030-compiler-release-approval-is-external.md),
[ADR 0049](0049-released-executor-descriptors-reserve-their-identities.md),
[ADR 0050](0050-released-compiler-locks-reserve-their-identities.md) and
[ADR 0052](0052-independent-agent-release-review.md). The invariants those records
protect are kept; only the mechanism changes.

## Decision

- **Code identity is the clean git commit of the checkout.**
  `open_cake_ir.source_identity.checkout_commit` returns HEAD only when the tree carries
  no modified tracked file and no untracked file; ignored files do not count. One check
  replaces every per-file digest list.
- **`compiler/revision.json` (schema 2) is the whole Compiler manifest.** It names the
  Corpus manifest and the calibration coverage. Every document under `compiler/targets/`
  is a declared Target, so adding a target is adding a document and nothing else.
- **A host is captured per exact target as `runtime/hosts/<target>.json`, and committed.**
  The Executor identity is `<target>@<commit>`, and its canonical digest covers the commit
  and that capture. A host capture changes when the host changes, never because shared
  source changed.
- **Releases are not minted per change.** The Corpus Gate runs in CI on the commit, and a
  merge to main carries one review. The author does not approve their own merge.
- **History stays where it is.** Released locks under `compiler/releases/`, descriptors
  under `runtime/executors/` and the frozen Executor inventory remain, and replay at their
  own commits with the tools of those commits.

## Why the mechanism changed

Measured on this repository before the change:

- Every released Executor pinned the same shared sources per file, so
  `refresh_inventory` retired every other target whose source map no longer matched. On
  `main`, one of five declared targets had a current Executor; on the AMD branch, three of
  seven, and the Apple M1 Pro pointer was lost for a day without anyone acting on it.
- 84 Compiler release directories and 129 Executor descriptors accumulated in 24 days. One
  bring-up day produced two Compiler releases and eleven Executor descriptors, and nine of
  its nineteen commits were release bookkeeping.
- The independent approval had become a signature: the two most recent approvals record
  that they rest on the author session's own reported Gate and change summary.

None of that is what the invariants were for. A Campaign still has to run on immutable,
identifiable code against a verified host; a git commit states that in one fact.

## Failure semantics

- A checkout with changes has no identity. The Compiler still assesses and lowers, and
  reports `open-cake-ir@uncommitted`; preflight, execution, the broker worker and replay
  refuse with the paths that differ.
- A target with no committed host capture is reported as having none. No other host is
  substituted, and no target inherits another's capture.
- A Campaign Lock keeps its reference shape. A reference now pins the commit through the
  Compiler and Executor identities, so replaying elsewhere or later refuses unless the
  checkout is clean at that commit.

## What this does not change

Exact target match with no step-down; an undeclared fact reported rather than substituted;
a report stating the domain it examined; on-device measurement as ground truth; Corpus
expectations never regenerated to make a Gate pass; Evidence append-only; campaigns
running only on frozen code.

## Validation

`tests/contracts/test_source_identity.py` covers the identity rules directly. The contract
suite covers the boundaries that consume them, and CI runs the Corpus Gate against
`compiler/revision.json` on every commit.
