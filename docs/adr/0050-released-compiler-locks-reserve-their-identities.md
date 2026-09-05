# ADR 0050: a released Compiler lock reserves its identity

Status: accepted, 2026-09-06. This extends the release identity rule to Compiler
locks; it does not remove the explicit approval input or change IR semantics.

At main `3de2ff2`, Compiler v41 was released and used by the retained FMA task.
The canonical local witness scan found no v41 reference and stopped at v40. The
old cycle would therefore re-release changed sources as v41 without archiving its
prior lock. Local witness absence cannot prove that an externally used release is
free to overwrite. Adding another directory pattern cannot establish that absence.

The released lock now reserves its identity. Changed bound sources archive that
release and prepare a successor above both the current ordinal and known historical
references. Preparation retains the old lock and approval until the new Gate is
approved and verified; repeating preparation retains the same draft identity.
Repeating the cycle for an already matching release is a no-op. No release archive
is reclaimed. Historical references still reserve otherwise unresolved old ids.

Historical GPU task generators run from their complete Git source revisions. A
current main checkout is not made permanently compatible with every old Compiler
lock, and a documentation correction does not relabel old GPU evidence. The v41
task replay examples explain the separate source and output locations.

Regression coverage exercises a released lock with no local consumer witness,
repeated unapproved preparation, an approved successor, and unchanged-release
verification. It requires the previous four release authority files to survive.
