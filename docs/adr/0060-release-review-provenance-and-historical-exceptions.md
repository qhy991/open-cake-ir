# ADR 0060: Release review provenance and historical exceptions

Status: proposed, 2026-09-09. The historical clarification below records existing
artifacts; the prospective validation change awaits independent review.
Related: [ADR 0030](0030-compiler-release-approval-is-external.md),
[ADR 0052](0052-independent-agent-release-review.md), and
[issue #73](https://github.com/qhy991/open-cake-ir/issues/73).

## Historical clarification, 2026-09-09

The approval records for Compiler v41, v42 and v43 record task-specific owner
delegation. They do not establish the standing independent approval procedure:

- [v41 approval](../../compiler/releases/v41/release-approval.json) names the
  repository owner, with Codex recording the authorization. Its basis and
  [review account](../AKA_FMA_RELEASE_REVIEW_20260904.md) identify the delegated
  `fma_release_review` agent as the actual technical reviewer. That account records
  a distinct read-only review, but does not record its effective model or reviewer
  session id. The approval is therefore not evidence of a separately performed
  human review or an allowed-model review under ADR 0052.
- [v42 approval](../../compiler/releases/v42/release-approval.json) expressly
  records owner-delegated Codex review without claiming a separate human or
  independent agent review.
- [v43 approval](../../compiler/releases/v43/release-approval.json) makes the same
  limitation explicit. The later B200 affine-parent canary's use of v43 does not
  supply the missing release-review independence.

These are historical task-specific exceptions, not standing authority for an
author to approve a later release. Owner authorization and independent technical
review are separate facts. This clarification does not rewrite their approvals,
locks, Gate reports or experiment evidence. It also does not infer that subsequent
independently reviewed releases lack independence merely because they share source
ancestry, or that a recorded device observation did not occur.

## Prospective decision

A future agent approval must locate its retained external review record. The
record identifies the actual author and reviewer sessions, effective model, exact
Gate and review decision. The approval basis references that record and explains
the examined evidence and limitations. Missing, unavailable, mismatched or stale
records refuse new release construction while preserving the prior authorities.
The reviewer creates the record and approval after examining the exact change;
neither is authored by the implementation or release cycle.

The validator checks consistency of retained declarations. It does not authenticate
a model provider or prove session identity from JSON. Launcher/session metadata
remains the source for the independent reviewer's identity check under ADR 0052.
Human approval remains supported without inventing an agent session.

The same reviewer session may review a later Gate after examining its changes and
creating a separate, exact review record. ADR 0052 requires separation from the
author; it does not prohibit a reviewer from performing successive reviews.
Reusing an old decision or record for a different Gate is refused. A global ban on
reviewer-session reuse would be a different policy, not a repair of that rule.

Historical approvals remain available with their original pinned implementation.
Their absence of a new provenance field cannot be repaired by backfilling a record
that purports to describe an unobserved historical session.

## Acceptance

Exercise the public release builder with missing, unreadable and stale records;
mismatched reviewer/model/author/Gate/decision; self-review; and an exact independent
approval. Exercise a second review in the same reviewer session using a new exact
record. Failed construction must preserve the old lock and approval. The first
release adopting this rule must itself provide the independently written record
and pass the full Corpus Gate.
