# Findings

Append-only records between campaign evidence and Compiler/Executor changes. One file
per finding, named `YYYY-MM-DD-NNN-slug.json`. The evidence ledger already records what
happened; a finding is the curated "so what" that a Revision decision can cite.

## Lifecycle

`decision` moves `proposed -> accepted | rejected | deferred` through the normal review
path. A finding closes only when `implemented_in` names the Revision that changed it and
`verified_by` names the verification that observed the change:

- **bug / protocol ticks** verify by *replay*: the cited evidence (workspace, captured
  streams, retained candidate sources) is re-processed under the successor Revision.
  Deterministic, no provider tokens.
- **capability / behavior findings** verify by *campaign*: new runs on the successor
  Revision, compared against frozen per-task baseline bundles so baseline movement and
  candidate headroom are reported separately.

## Record contract

- Cite evidence by workspace path, run id and event sequence number, or by object path
  under the campaign's evidence root. The ledger's own hash chain settles byte identity;
  findings do not copy digests (see AGENTS.md, "SHA checks are exceptional").
- Name `executor_revision`, `compiler_revision_id`, `backend` and `target` on every
  finding - cross-backend and cross-version comparison happens at this layer, never
  inside one campaign.
- `kind` is one of `bug`, `capacity`, `behavior`, `protocol`.
- Workspaces live outside the checkout and are never reset or deleted to make a finding
  convenient; a superseding run gets a suffixed workspace and both stay citable.
- The `NNN` in an id is allocated in submission order per date; before minting a finding,
  pull and take the next free number (the 2026-09-10-005 collision between two parallel
  sessions was caught at push and renumbered).

## Index

- F-2026-09-10-001 — multi-SIMD share DCE (bug, silu + swiglu recurrence; closed in Compiler v73)
- F-2026-09-10-002 — observer cohort snapshot bound (capacity, adamw + momentum_sgd corroboration; launcher check in Executor v90, observer charging still open)
- F-2026-09-10-003 — claude is_error zero-tolerance (protocol, softmax + swiglu; closed in Executor v90)
- F-2026-09-10-004 — baseline near vocabulary ceiling (behavior, 10 tasks)
- F-2026-09-10-005 — GEMM SiLU oracle overflow (bug, CPU oracle replay; fixed in Executor v88)
- F-2026-09-10-006 — explicit private epilogue fusion (capacity, implemented in Compiler v72; campaign verification pending)
- F-2026-09-10-007 — relative-path Write outside envelope (protocol, momentum_sgd; closed in Executor v90)
- F-2026-09-10-008 — CLI auto-compact outside event contract (protocol, momentum_sgd; closed in Executor v90)
- F-2026-09-10-009 — Claude quota-warning contract drift (protocol, closed in Executor v89)
- F-2026-09-10-012 — checkout-local Executor ordinals silently mint duplicate ids (bug, observed; plus the single current pointer)
- F-2026-09-10-011 — saved forward outputs declared with impossible signs (bug, 3 tasks; found by the first B300 sweep, closed in Executor v91)
- F-2026-09-10-010 — contraction regime shows material headroom over three campaigns; instability traced away from dispatch count to un-excluded external GPU activity (behavior, gemm)
- F-2026-09-10-013 — auto-compact window clamped to the CLI's assumed model context for unrecognized models; the bare status compaction notices reached the contract unnamed (protocol, gemm + pairwise_sqdist; closed in Executor v99)
- F-2026-09-10-014 — launcher's elementwise shape defaults overrode the contraction contract's extents, so every contraction task's baseline was refused on Metal (bug, 4 tasks; closed in Executor v99)
