# ADR 0007: candidate evidence has one identity, and every frozen reference is a witness

## Outcome and non-goals

A Turn may submit several candidates, but every candidate must remain attributable from
provider output through filtering, search, selection, confirmation, feedback and offline
replay. A released Compiler Revision may have only one byte identity once any frozen
artifact refers to it.

This repair does not change Schedule semantics, introduce a ranking term, rewrite an old
observation, or claim new GPU evidence. Historical single-candidate Campaigns remain
read-only replay inputs through one bounded legacy adapter.

## Canonical identities and owners

- A provider Turn owns one ordered candidate list. Its candidate identity is
  `(turn, candidate_sha256)`; uniquely indexed `candidate_submission_0000`,
  `candidate_submission_0001`, ... object roles preserve that order without violating
  Evidence's one-role-per-event invariant.
- An Evaluation attempt is identified by
  `(turn, purpose, candidate_sha256)`. Neither `turn` nor `(turn, purpose)` is unique after
  candidate sets exist.
- `candidate_set_filtered` owns the complete pre-GPU order for the Turn.
- `candidate_selected` owns the one diagnostic subject and whether a qualified search
  result exists. The selected qualified candidate is derived from retained search receipts:
  qualification first, then lowest latency, then filter order.
- The selected candidate's `EnvironmentResult` owns the findings returned with its
  measurement. Findings from another candidate may not be substituted.
- The Compiler witness set is the union of every declared frozen-artifact source. Release
  policy consumes that one set; individual lifecycle scripts do not invent narrower
  definitions of history.

## State and failure semantics

For each Turn the Lab seals all submissions, records the full filter order, evaluates up to
the declared number of structurally distinct launchable candidates, and records a
selection. Only a qualified search receipt may win confirmation. If none qualifies, the
first searched candidate is retained only as the diagnostic subject; the Turn has no
qualified result and contributes no checkpoint latency.

Replay rebuilds the provider candidate list, launchables and receipts by the canonical
keys, recomputes selection, and checks the projected checkpoints. Missing coverage,
duplicate identities, an unqualified winner, a selection inconsistent with retained
receipts, or a filtered candidate not submitted by the provider fails replay.

The existing `open-cake-ir-sm100a-v4` collision is historical damage, not a migration that
can be repaired in place. Old observations stay byte-for-byte unchanged. The incident
record names every conflicting digest and its recoverability. The unrelated frozen v6
observation already consumed that otherwise unissued name, so the first unambiguous
successor is `open-cake-ir-sm100a-v7`.

This ADR supersedes ADR 0005 only on its promised directory move for the next Compiler and
Executor descriptors. Current consumers already bind `compiler/revision.lock.json` and
`runtime/executors/` as their authorities; moving them during this identity repair would
create a second writable authority or require a broader migration. The lifecycle invariant
is enforced now—released bytes and frozen Study Contracts are not rewritten—while a future
path migration requires its own complete successor design.

## Acceptance evidence

The smallest complete slice is a two-candidate fixture Campaign whose statically first
candidate is not the qualified measured winner. `execute` followed by `audit` must replay,
the next Turn must receive the winner's findings, and tampering with the selected candidate
must make semantic replay fail. Separate tests cover an unstable faster receipt, duplicate
program spellings that do not consume search budget, observation-to-Revision resolution,
and legacy single-candidate replay.
