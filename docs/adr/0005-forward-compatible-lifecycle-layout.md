# ADR 0005: Adopt lifecycle-first repository governance without moving history

Status: accepted, 2026-08-23; descriptor-placement commitment superseded by ADR 0007.

## Context

The Domain Contexts and their dependency direction are stable, but the repository still mixes active
Implementation, immutable definitions, released identities, historical Evidence and derived views. Historical paths
cannot be cosmetically reorganized: Compiler Revisions, Executor Revisions, Study Contracts, CampaignLocks and
Evidence retain exact path and digest references.

The most urgent failure is executable rather than visual. Generated Campaigns are documented as external to the
checkout, yet the Lab previously required only a create-only Evidence path and allowed that path inside the source
tree.

## Decision

Historical files remain byte-for-byte and path-for-path frozen. New repository material will converge by lifecycle:

```text
src/open_cake_ir/          active Implementation
definitions/<context>/     immutable declarative inputs owned by one Domain Context
releases/compiler/         released Compiler Revision descriptors
releases/executors/        released Executor Revision descriptors
evidence/                  immutable historical observations and archives
reports/current/           deletable projections generated from canonical inputs
```

The exact owner of a definition must be settled in the Context language before its first successor is placed under
`definitions/`; directory movement may not decide ownership implicitly. Released Revision descriptors belong under
`releases/`, not `definitions/`.

Every new CampaignLock and Campaign Evidence root is admitted through one Lab custody rule and must be create-only
outside the project checkout. CLI rejection occurs before reading execution inputs; the Lab repeats the same rule so
SDK callers cannot bypass it. Historical in-checkout CampaignLocks and Evidence remain available to read-only audit.

Executor v6 is the final transitional descriptor under `runtime/executors/` because its B200 observation already
binds that path. The next Executor Revision begins under `releases/executors/`. Existing expanded Executor archives
remain unchanged. A future archive may use an opaque source archive plus manifest to remove search pollution; a
cross-Revision CAS is not introduced until real storage pressure justifies its garbage-collection and liveness rules.

`inventory/` remains a legacy mixed projection until execution no longer reads it. New `reports/current/` output may
be introduced only after runtime selection binds an exact released authority elsewhere. A generated Catalog is for
discovery and cannot select an Executor, authorize a Campaign or redefine Study execution state.

## Consequences

- The checkout stops accumulating new Campaign runtime state while historical replay remains intact.
- Directory convergence happens at successor boundaries, without compatibility copies or rewritten hashes.
- Lifecycle and canonical owner are both visible; neither is inferred from a filename version.
- Executor v7 and the next Compiler Revision must use the new release layout or explicitly supersede this ADR.
- Catalog and Claim View work remains a later projection change, not part of Campaign custody enforcement.
