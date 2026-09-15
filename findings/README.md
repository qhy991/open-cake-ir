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

- F-2026-09-15-004 — vendor, code-object family and lowering route are three axes, and shared code keeps collapsing them into hand-enumerated target ids; eight repairs in one bring-up are the same defect (capacity, proposed; declaring the code object is the next step warp_size and vendor began)
- F-2026-09-15-002 — Hygon DCU gfx938 is measurable AMD hardware, retiring F-2026-09-13-006's "no AMD device measurement is available"; its steps 2 and 3 and the first AMD Target land together (capacity, accepted; Gate 150/150, device-checked, ADR 0052 review and paired Executor successor pending)
- F-2026-09-15-003 — the AMDGCN Evaluation half exists in parts (HIP host admission, exact-HIP runtime custody, rocprofv3 trace projection) but no gfx target has an executable role, no timing path is named for a trace, and schema v2 identity pins gfx1151 while the Compiler target is gfx938 (capacity, accepted; the executable role, the AMD-parameterized Executor identity, the single host `kind` field and the named paired measurement source are implemented, an AMD evaluation driver and any timing calibration are not; two earlier observations in this record were read from a stale ref and are corrected in it)
- F-2026-09-14-001 — the Python FMA frontend defaults every target to `ptx.fma.rn.f32`, causing four fresh and four prior Metal candidates to be correctly refused as foreign instructions (capacity, Compiler frontend/Target/Metal contract proposal)
- F-2026-09-14-002 — M4 smoke timing certifies only 6/18 searches and 4/6 confirmations, with an AdamW direction reversal across stages (capacity, target-specific Lab measurement pilot; no withheld speedup claims)
- F-2026-09-14-003 — the one-turn M4 matrix measures first-shot candidate yield but cannot test whether localized diagnostics repair the next candidate (behavior, successor multi-turn Study proposal; no Compiler relaxation)
- F-2026-09-14-004 — a passing Corpus Gate named no Target, and apple_gpu_family7 and apple_gpu_family9 carry zero cases (protocol, Gate-report proposal)

- F-2026-09-13-002 — Compiler-generated exact FP32 FMA and infinity identities fail common Triton admission (bug, 10 B300 tasks; closed in Compiler v80 by 367-candidate replay and 16 isolated SDK builds)
- F-2026-09-13-003 — 3M-token Kimi sweep recurs at actual context compaction and missing/wrong-turn structured terminals (protocol, 12 B300 tasks; v4 compaction and exact terminal request implemented in Executor v114; live qualification pending)

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
- F-2026-09-10-017 — native/CuTe admission omitted explicit ownership, cache, scalar, state, target and direct-entry contracts (bug; closed in the retained Compiler v73 source replay, while successor integration verification remains pending)
- F-2026-09-10-011 — saved forward outputs declared with impossible signs (bug, 3 tasks; found by the first B300 sweep, closed in Executor v91)
- F-2026-09-10-010 — contraction regime shows material headroom over three campaigns; instability traced away from dispatch count to un-excluded external GPU activity (behavior, gemm)
- F-2026-09-10-013 — auto-compact window clamped to the CLI's assumed model context for unrecognized models; the bare status compaction notices reached the contract unnamed (protocol, gemm + pairwise_sqdist; closed in Executor v99)
- F-2026-09-10-014 — launcher's elementwise shape defaults overrode the contraction contract's extents, so every contraction task's baseline was refused on Metal (bug, 4 tasks; closed in Executor v99)
- F-2026-09-10-015 — Apple M4 has no exact family9 Compiler target or Executor admission path (capacity, closed in Compiler v74 and verified by an M4 campaign under Executor v100)
- F-2026-09-10-016 — Claude same-model internal activity can make aggregate modelUsage exceed the main-turn counters (protocol, closed in Executor v100 and verified by live kimi-k3 qualification)
- F-2026-09-10-018 — native event contract is pinned to one CLI build (protocol, proposed; the shared cause behind F-2026-09-10-003, F-008, F-009 and the kimi-k3 bring-up)
- F-2026-09-11-001 — Claude gateway tool ids may be reused only after the prior invocation completes (protocol, closed in Executor v101 and verified by a fresh layernorm campaign)
- F-2026-09-11-002 — CLI tool-progress heartbeat outside the native contract shadowed the named compaction refusal (protocol, gemm v99 retry; closed in Executor v103)
- F-2026-09-11-003 — a schema-rejected StructuredOutput call can recover through a later exact success (protocol, closed in Executor v102)
- F-2026-09-11-004 — output-column specialization cuts Metal contraction private state and confirms 23.6x on M1 Pro GEMM-SiLU (behavior; implemented in Compiler v79; M1 Pro successor campaigns on Executor v115 report flat compiler floor and a quality-passed 14.19x in-vocabulary K-split competitor, with the column mechanism directionally confirmed at 33.9x/36.2x search stage; M4 leg pending SDK repair)
- F-2026-09-11-005 — restricted NVIDIA counters require a bounded privileged NCU launch and caller-owned evidence (protocol, E107 attribution failure; closed in Executor v108 by two retained-candidate attribution replays)
- F-2026-09-11-006 — GELU Compiler output loses its libdevice dependency at common build admission (bug, closed in Compiler v76 by real baseline builds and both five-case B300 GPU replays)
- F-2026-09-11-007 — prepare every matrix baseline before provider spending and preregister sustained-search budgets and timing parameters (protocol, closed in Executor v109 by public baseline preparation and six fixed-shape controls; old timing failures unchanged)
- F-2026-09-12-001 — exact-R2 row-reduction scalarization confirms 1.107x on M4 channel absmax and a second stable 1.138x bias-gradient result under v78/v113 (behavior, promoted exact-task incumbent; Compiler-pass proposal still awaits M1 Pro, representative-shape and counterexample evidence)
- F-2026-09-12-002 — native heartbeats establish liveness but not semantic progress or enforcement of an in-flight provider budget (capacity, reconfirmed by one timeout and four 520k-1.49M-token calls under the v78/v113 matrix; provider-boundary proposal)
- F-2026-09-12-003 — per-Run promotion now feeds an append-only exact-task incumbent chain for the next artifact-optimization baseline (capacity, closed in family7 v109/family9 v110)
- F-2026-09-13-001 — M4 selected SDK 27 required Swift 6.4 while the installed CLT provided Swift 6.3.3 (capacity, closed by CLT 27, family9 Executor v113 and 29/29 baseline preflight)
- F-2026-09-13-005 — MLX's in-process JIT runs the emitted Metal route bit-identically on family7 across 95 differential cases, but sets one of three compile options and checks no pipeline commitment (capacity, second correctness host; closed by a 95/95 successor differential on Compiler v81; not a lowering route)
- F-2026-09-13-006 — naming AMD touches five shared sites, all pinned (three by the Compiler, two by the current family7 Executor); step 1 closed in Compiler v81 by declaring warp_size in every existing Target, while vendor identity, warpgroup width, neutrality fixtures and AMD device support remain future steps
- F-2026-09-13-004 — a per-role register budget dereferences the undeclared warpgroup width and crashes hardware conformance on every non-CUDA Target (bug, closed in Compiler v81 by public replay and the 145/145 Corpus Gate)
- F-2026-09-14-005 — subscription-authenticated Claude sessions report a surpassedThreshold fraction on the quota warning event, stranding provider qualification (protocol, closed in Executor v116 by replay qualification passing the quota guard into a live campaign)
- F-2026-09-15-001 — provider faults killed by the subscription quota wall recorded only a generic exit-code failure while the explaining rate-limit notice stayed in the retained stdout (protocol, 19 overnight instances; closed in Executor v118 by replaying all nineteen fault streams to their terminal notices)
