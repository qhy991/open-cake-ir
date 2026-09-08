# ADR 0058: External writer-origin custody anchors

Status: proposed successor to ADR 0031's implementation of Filesystem Custody.

## Decision

Archive Integrity remains a byte and event-chain replay result. Filesystem Custody also
requires a prospective, external writer-origin witness. Current owner and mode bits alone
cannot distinguish a native writer archive from an ordinary Git checkout or copied tree.
Existing archives without these witnesses remain readable and have custody false under
this successor. Opening, auditing, restoring permissions, or copying old records into a
new root never creates or repairs an origin witness.

`EvidenceStore.create` admits only a new canonical path outside every ancestor Git
checkout. It creates an original private store marker and a create-only origin record in
an external owner-private registry. `OPEN_CAKE_CUSTODY_DIRECTORY` is caller configuration;
its default is `~/.local/share/open-cake-ir/custody-anchors`. The archive cannot select the
registry. The registry must be canonical, outside Git, separate from the archive, and
owner-private. Symlinks, shared modes and malformed records fail closed.

The origin binds the canonical root and original marker to device, inode, UID and ctime.
Normal writer activity changes children, leaving these original identities intact. Each
`start_run` separately registers its authority genesis, Run directory identity and original
private writer lock. Under that existing lock, event publication is followed by a
create-only external count/head witness using the existing event-chain identifier. Final
publication also witnesses the existing terminal seal and terminal file identity. Audit
compares every registered frontier with the replayed chain. There is no additional digest
inventory or duplicate payload store.

A fresh root therefore cannot grant custody to a copied old Run. Grafting history into an
already registered same-ID Run also fails: directory identity alone is not sufficient;
the writer must have witnessed its exact ordered frontier and terminal publication.
Reusing immutable CAS object bytes does not by itself adopt a Run history.

## Durability and failure

A failed event publication before persistence leaves the prior frontier usable. An event
persisted without its external witness cannot be adopted by a subsequent writer or audit.
A matching, witnessed terminal event whose terminal file was never published may complete
native pending-seal recovery. Once the seal witness exists, deleting or replacing the
terminal file is corruption, not recovery. If terminal publication succeeds but its external
seal witness fails, the archive may replay intact while custody is false; it is not
restamped. Partial or lost registry records fail closed without deleting retained bytes.

Only create/start/append/seal write witnesses. Read-only audit reevaluates live identities
and never caches a prior true result as current authority. Claim consumers continue to
require both Archive Integrity and Filesystem Custody for promotion, system qualification
and scientific estimates.

## Trust domain and historical interpretation

This boundary assumes the local OS, the writer UID and the caller-selected private registry
are trusted, and all native publications use the writer protocol. It detects ordinary
copies, checkouts, moves, substitutions and incomplete publication. It is not a proof of
continuous historical permissions or ACLs; it does not resist a malicious same-UID/root
actor rewriting both archive and registry, or coordinated filesystem rollback. Neither
mode bits nor filesystem identities justify that stronger claim.

Historical released implementations and their observations remain in pinned Git. Prior
mode-only `custody=true` reports are not rewritten or requalified. In particular, the
retained v28 RoPE report and pre-anchor v70 RMSNorm report keep their original policy and
observations; their unanchored archives cannot support new claim-bearing projections under
this successor. This is a prospective boundary correction, not evidence of fabricated
artifacts. A dated external erratum records the affected interpretation.

## Validation

Contract tests exercise native private creation/reopen, concurrent appends, Git-like modes,
copies/moves, restored marker modes, missing/tampered/symlinked anchors, fresh-root and
same-ID history grafts, publication failure boundaries and authorized pending-seal recovery.
The existing promotion projection refuses otherwise intact unanchored archives. Historical
in-checkout replay is checked read-only: Integrity true and Custody false. No mode repair,
automatic migration, provider call or GPU experiment is part of this change.
