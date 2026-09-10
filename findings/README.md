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

## Index

- F-2026-09-10-001 — multi-SIMD share DCE (bug, silu)
- F-2026-09-10-002 — observer cohort snapshot bound (capacity, adamw)
- F-2026-09-10-003 — claude is_error zero-tolerance (protocol, softmax)
- F-2026-09-10-004 — baseline near vocabulary ceiling (behavior, 5 tasks)

- F-2026-09-10-005 — GEMM SiLU oracle overflow (bug, CPU oracle replay; fixed in Executor v88)

- F-2026-09-10-006 — explicit private epilogue fusion (capacity, implemented in Compiler v72; campaign verification pending)
