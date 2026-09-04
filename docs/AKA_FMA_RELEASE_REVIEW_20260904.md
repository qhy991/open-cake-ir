# FMA v41 release review — 2026-09-04

## Current decision

Independent technical decision: **recommend-static-release**.
The four reviewed FMA cases are now admitted to the canonical Corpus manifest, and the
complete Gate matches **72/72 cases with 91 bound sources**. There are no identified
static release-blocking code defects.

Formal static Compiler release is now **completed**: `open-cake-ir-sm100a-v41`.
The user explicitly granted the requested one-time permission to record the reviewed
Gate approval and publish v41. The approval was recorded separately from the unchanged
release cycle, which then exited 0 and produced a verified released lock. Long-term
AGENTS rules were not changed; the v40 archive remains intact. This document is review
evidence, not a substitute for `compiler/release-approval.json`.

Exact approval boundary:

- Compiler proposal: `open-cake-ir-sm100a-v41-draft`.
- Gate: `compiler/corpus-gate-report.json`.
- Canonical Gate SHA256:
  `372f3c182a06374cbb743eb9c5388fc0103b36a6faa323c2d291fef74628f56c`.
- Reviewed implementation: `c3304a59439704c3656c96c0e3d3e4a5a86dada8`,
  relative to `ff8903d9c7b085ed9439cd1a34ea00e69e8bfd41`.
- Subsequent Compiler-bound change: only canonical manifest admission of the exact
  four entries from [the reviewed proposal](AKA_FMA_CORPUS_PROPOSAL_20260904.json).

The digest above is recorded once for the new external approval boundary, not as a
claim of semantic or GPU correctness.

Digest-record correction: the first review mislabeled the serialized file's SHA256
(starting `e0b15f2c`) as the canonical Gate digest. The file includes a trailing newline;
the release builder's canonical JSON does not. The first authorized release attempt
therefore correctly exited 3 without replacing the v40 lock. The independent reviewer
rechecked `build_gate_report(...).canonical_sha256`, confirmed that the Gate file,
source, Target, manifest and proposal were unchanged from `1f2152f`, and reaffirmed
`recommend-static-release` for the same 72-case Gate. Only the digest record was
corrected; no semantic evidence or expected result was changed to make release pass.

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

## Post-release verification

Released commit `2ceb5b7b2ad4928b552e8941ce5d30d7d050c7dd` was transferred into a
new remote worktree, preserving both the pre-release and previous test snapshots.
As `qhy-sol` on `verda-b200x4`, the complete relevant suite passed **252/252 tests,
with no exclusions**. This includes the 16 FMA tests, the previous 197 shared tests,
and 39 Compiler/release contract tests. The five observation tests that required a
valid released identity now pass against v41 itself, rather than only the v40 baseline.

The existing Python 3.12.3 / Torch 2.12.1+cu130 / Triton 3.7.1 environment was reused,
with `CUDA_VISIBLE_DEVICES=""`. Both real GPU-free compilation cases passed again:
ordinary FMA emitted one `fma.rn.f32` and a 9,560-byte cubin; the nested case emitted
two FMA instructions, retained the separate multiplication, and produced a 9,824-byte
cubin. No GPU was allocated or executed.

Remote evidence root:
`/home/qhy-sol/aka-fma-v41-release-20260904-ba6kOC/`.
The tested worktree is `released-v41/`, the durable test log is
`v41-release-contracts.log`, and compiler outputs are in `released-triton-cache/`.
The original release tool's full `--verify` passed on this remote snapshot; its
worktree remained clean.

The independent reviewer subsequently returned **verified-release** for the same
commit: `build_release(...).verify(root)` returned true, released `Compiler.load`
and the 72-case Gate passed, and the lock correctly bound the approved canonical Gate,
approval and all 91 sources. Source, manifest, Target, proposal, Gate, v40 archive and
long-term AGENTS rules were unchanged by the release. That advisory is independent
of the primary task's remote test counts.

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

The existing cycle consumed the explicitly authorized approval for the canonical Gate
above. Local and remote release verification are complete, with 72/72 cases and 91
bound sources. No main merge or GPU run was performed. The remaining numerical-GPU
checks are a separate frozen-Workload task; log and cosine remain approved designs,
not features included in v41.
