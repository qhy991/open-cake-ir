# ADR 0031: Archive integrity is not filesystem custody

Status: accepted in the current unwitnessed Executor v27 working release, 2026-08-25.

## Outcome and non-goals

A read-only Evidence audit reports two orthogonal facts:

- `archive_integrity` says the authority, event chain, terminal seal and referenced
  object bytes form one valid sealed Run at the time they were read;
- `filesystem_custody_verified` says the audited paths also carry the owner and mode
  properties required by the live single-writer store.

An intact Git checkout with permissive modes must therefore remain replayable as
`archive_integrity=true, filesystem_custody_verified=false`. It cannot support promotion,
system qualification or a scientific estimate until custody or a separate external
archive anchor is established.

This does not introduce an archive format, signature service, mode manifest, migration
adapter or second auditor. It does not reinterpret frozen Evidence, and present mode bits
do not prove continuous historical custody.

## Authorities and dataflow

`EvidenceStore.audit_run` is the sole owner of both per-Run facts. It uses the existing
no-follow, descriptor-relative reader for content verification while separately observing
the modes and owner of the exact directories and files it reads. `Lab.audit` aggregates
the Run facts once into `archive_integrity_passed` and
`filesystem_custody_verified`; consumers do not recompute either.

```text
Evidence bytes --read-only, no-follow--> archive_integrity
same audited paths --------------------> filesystem_custody_verified
both + semantic replay + protocol ----> claim/estimate eligibility
```

`EvidenceStore.writer` retains the strict live-custody admission rule. Read-only audit may
observe weak modes; no write path may use them. The clone-time normalization helper is
removed because changing modes immediately before audit cannot prove historical custody
and is no longer required to inspect intact bytes.

## Failure and compatibility semantics

- Missing, malformed, linked, reordered or digest-mismatched content makes
  `archive_integrity=false` as before.
- Correct bytes under weak ownership or modes keep archive integrity and semantic replay
  observable but set filesystem custody false; claim-bearing projections remain false or
  unavailable.
- A live store created by the current writer reports both facts true.
- Frozen serialized qualification fields retain their historical spelling. The active
  `RunAudit.integrity` spelling is removed rather than kept as a second authority;
  consumers move atomically to `archive_integrity`.

## Acceptance evidence

One copied, byte-identical historical Campaign is made writable as a clone-time fixture.
Fresh Lab audit must still pass archive integrity and semantic replay, explicitly fail
filesystem custody and refuse system qualification. Opening that root as a writer must
fail. A content mutation must still fail archive integrity, while an unmodified live root
must report both facts true.
