# FMA v41 release review — 2026-09-04

## Current decision

Independent technical decision: **recommend-static-release**.
The four reviewed FMA cases are now admitted to the canonical Corpus manifest, and the
complete Gate matches **72/72 cases with 91 bound sources**. There are no identified
static release-blocking code defects.

Formal release has **not** been performed. The repository prohibits this automation
from writing `compiler/release-approval.json`; the user was asked whether their request
to publish grants a one-time exception after independent review. Until that explicit
answer or an externally written approval arrives, the old approval and v40 lock remain
unchanged. This document is technical review evidence, not a substitute approval file.

Exact approval boundary:

- Compiler proposal: `open-cake-ir-sm100a-v41-draft`.
- Gate: `compiler/corpus-gate-report.json`.
- Canonical Gate SHA256:
  `e0b15f2c63157cd3b4f470807208e3cc91a341432cc8b8a9838ec70218ad63fb`.
- Reviewed implementation: `c3304a59439704c3656c96c0e3d3e4a5a86dada8`,
  relative to `ff8903d9c7b085ed9439cd1a34ea00e69e8bfd41`.
- Subsequent Compiler-bound change: only canonical manifest admission of the exact
  four entries from [the reviewed proposal](AKA_FMA_CORPUS_PROPOSAL_20260904.json).

The digest above is recorded once for the new external approval boundary, not as a
claim of semantic or GPU correctness.

## Independent review

The delegated read-only reviewer `fma_release_review` completed two bounded reviews.
This was an actual returned advisory, unlike the earlier empty completion from the
separate `Review published AKA IR results` task.

First decision: `approve-for-corpus-admission`.

- Independently reran 15/15 CPU FMA contract tests on the exact remote source snapshot.
  The cache-writing offline compilation test was excluded from this read-only run and
  is not counted as independently repeated here.
- Constructed 89 public `Compiler.assess` probes, including 67 parser-admitted inputs,
  with zero exceptions and zero unexpected acceptance or rejection. Coverage included
  all operand/result dtype, space, rank and shape positions; missing/extra/unknown
  edges; instruction/operation mismatch; malformed/missing contracts; unsupported
  backend/target; and legal repeated/permuted operands.
- Checked seven nested dependency/lifetime variants. Read-before-write, self-read/write,
  missing producers and cycles were refused; removing redundant dependency annotations
  while retaining valid declared order and read edges was correctly admitted.
- Verified all four proposed cases' acceptance, lowering eligibility and finding codes.
- Did not modify source, approval, lock, Git refs or remote persistent state.

Final decision: `recommend-static-release`.

- Rebuilt the final Gate using the existing release tool's
  `--prepare-gate --verify` path: exit 0, no output-file write.
- Verified the first 68 manifest entries and all non-case manifest fields are unchanged
  from the reviewed implementation.
- Verified the four appended entries exactly equal the reviewed proposal.
- Verified source set, proposal, original released lock and original approval did not
  change during admission; no implementation source changed.
- `git diff --check` passed; the reviewer observed only the intended manifest/Gate
  changes and did not write them.

The primary task separately exercised 36 boundary inputs, including legal operand
reuse and order, with no unexpected outcome. These are supplementary checks, not
added to the independent review's counts because their coverage overlaps.

## Corpus adoption

The existing manifest is the sole Corpus authority. This change appends two positive
cases (ordinary and nested/rounded-producer FMA) and two negative cases (missing operand
and wrong instruction meaning). It does not regenerate any old expectation.
The earlier four-entry proposal remains a historical review input.

The normal release cycle produced the 72-case Gate and exited 3 because the old
approval does not bind it. That stop is intentional. The previous
[implementation report](AKA_FMA_IMPLEMENTATION_20260904.md) retains the earlier
68-case checkpoint and remote 208-test/real-compile evidence; this review supersedes
only its pending-Corpus-admission status.

## Scope and next action

This recommendation concerns static IR semantics, validation, analysis and lowering.
It is not FMA GPU numerical qualification, performance evidence, complete AKA-parent
expressibility or end-to-end framework acceptance. Cancellation, signed zero,
subnormal, Inf/NaN, nested FMA and rounded-producer numerical outputs still require
independent on-device checks under a frozen Workload contract.

Once an explicit permitted writer records approval for the exact Gate above, rerun the
existing release cycle and verify the released lock before publishing the released
Compiler. Do not change long-term AGENTS rules or invent a reviewer identity. No main
merge or GPU run has been performed by this release-preparation change.
