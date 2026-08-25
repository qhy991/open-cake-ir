# Audit findings register, 2026-08-25

This is the current-branch disposition of the external audit originally performed at
`f503de4` and rechecked at `e351b67`. The original prose lived in a divergent old
worktree; every conclusion below was reproduced or falsified again on the current
`codegen-prototype` line before changing its disposition.

## Disposition summary

| # | Finding | Current disposition |
| --- | --- | --- |
| 1 | `EpilogueFormula` was parsed but ignored | fixed in Compiler v23 |
| 2 | tutorial linked an obsolete, unparsable Schedule | fixed at the documentation authority |
| 3 | committed Evidence can fail custody checks after clone | operational repair added; archive design open |
| 4 | historical ranking checker duplicates current order | refuted; historical/current split is intentional |
| 5 | residency provenance was operator-supplied | fixed; host and time derive from the run |
| 6 | Compiler release writes its own approval | open governance defect |
| 7 | revision identities cause Study successor churn | open ownership defect |
| 8 | lowering profile is both routing and workload constraint | open architecture defect |
| 9 | emitter preconditions were late failures | fixed in Compiler v23 |

## Re-verification and action

### 1 and 9: backend semantics now fail before lowering

The defect remained present: changing the full CuTe-DSL assignment epilogue from
`centroid_sq_minus_two_dot` to `bias_add_bf16_round` left the Schedule accepted and
lowering-eligible, while source still contained centroid arithmetic. Backend preflight
is now the single owner of emitter-only requirements. Assessment projects those
requirements as lowering-blocking Findings, and direct emitter use consumes the same
preflight. The formula drift is a reviewed negative Corpus case rather than only a unit
test. ADR 0027 records the boundary.

TinyGEMM2 already expressed and checked the four-part CTA sum. Its epilogue formula was
only indirectly protected by a whole-Schedule digest; profile conformance now explicitly
requires `bias_add_bf16_round`. The kernel remains a closed source asset and truthfully
reports `generated=false`: schedule-level reduction semantics are solved, full code
generation is not.

### 2: fix the live tutorial, retain frozen history

`examples/gpu/flash-kmeans-b32-smoke.json` still carries the removed whole-operator MMA
formula and does not parse. It is retained because frozen Executor closures refer to it.
The actual quickstart already defaults to the valid `-v2` successor; the documentation
link now points to that same authority, and a contract test parses the linked example.
Runtime example JSON files are not Schedules, so “every JSON under examples parses as a
Schedule” would be an invalid gate.

### 3: custody and archive integrity are different claims

`EvidenceStore` correctly requires non-writable directories for a live Run. Git cannot
preserve those mode bits, so the same check can reject committed archives after a clone.
`tools/normalize_evidence_custody.py` makes the operational precondition checkable and
repairs it only under `--apply`; it does not silently manufacture a pass.

The design remains open. A future Executor successor should expose archive replay as
“hash-chain verified, filesystem custody not verified” rather than either claiming live
custody or refusing intact committed evidence. That status must be observable before
the operational tool can be retired.

### 4 and 5: one refuted, one fixed

The old v6 ranking checker intentionally replays the historical total order; current
ranking uses the newer preorder implementation. Editing the old checker would corrupt
negative evidence. Residency instruments now obtain UTC time and hostname from the
machine. Frozen calibration-v6 tooling is unchanged.

### 6: approval is still not an independent gate

`release_compiler_cycle.sh` still materializes `release-approval.json` after computing
the gate it approves. The Corpus expectations can fail, but the approval is a record of
the same actor's decision, not independent review. Until a distinct reviewer produces
that artifact, documentation and paper claims must not call it independent evidence.

### 7 and 8: the main remaining paper-level architecture debt

Study contracts inline changing Compiler/Executor identities, so routine revisions mint
near-identical successors. Separately, `metadata.profile` currently owns backend route,
entry ABI and bespoke workload conformance. This makes the profile table grow with the
corpus even though the Schedule body is compositional.

The minimal next redesign is to separate three existing facts, not add a framework:

1. Workload Contract owns external tensor and oracle semantics.
2. Schedule plus Target owns operation and hardware legality.
3. A small lowering route owns backend and entry ABI only.

Unknown workload names should not make an otherwise complete Schedule unlowerable. This
migration must replace the current writable authority in one successor; parallel profile
and route spellings would make the ownership problem worse.
