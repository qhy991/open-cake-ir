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
| 6 | Compiler release writes its own approval | successor mechanism fixed; current v24 approval is not independent |
| 7 | revision identities cause Study successor churn | fixed by stable templates and exact CampaignLocks |
| 8 | lowering profile is both routing and workload constraint | fixed in Compiler v24 |
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
only indirectly protected by a whole-Schedule digest; asset preflight now explicitly
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

The design remains open. A later archive-format migration should expose replay as
“hash-chain verified, filesystem custody not verified” rather than either claiming live
custody or refusing intact committed evidence. That status must be observable before
the operational tool can be retired.

### 4 and 5: one refuted, one fixed

The old v6 ranking checker intentionally replays the historical total order; current
ranking uses the newer preorder implementation. Editing the old checker would corrupt
negative evidence. Residency instruments now obtain UTC time and hostname from the
machine. Frozen calibration-v6 tooling is unchanged.

### 6: successor preparation no longer approves itself

`release_compiler_cycle.sh` now prepares the full failing-capable Corpus Gate but never
creates or changes `release-approval.json`. A missing, malformed or stale approval stops
the cycle with the prior lock and approval bytes intact. After a reviewer outside that
automation writes an approval bound to the exact Gate digest, rerunning the same command
consumes it through the existing release validator. An executable contract proves both
the refusal and success transitions.

The current v24 approval predates this repair and records the releasing repository
owner, so it is not retroactively independent evidence. The successor mechanism is
fixed; the evidence limitation closes only when a distinct reviewer actually approves a
future release. ADR 0030 owns this boundary without adding signatures, accounts or a
second release implementation.

### 7: stable design templates, exact execution locks

The defect was reproduced: five current zero-GPU Study fixtures differed from their
predecessors only because Compiler or Executor identity advanced. ADR 0028 now separates
stable experimental design from execution authority. A `template` Study has one legal
revision spelling, `{"binding":"current_release"}`. `Lab.preflight` resolves both
authorities exactly once; the resulting CampaignLock retains the Study digest plus exact
Compiler and Executor id, path, and digest. A `frozen` Study still requires exact
references and cannot follow current state. The two freeze tools replace both bindings
atomically before emitting a frozen successor.

The five current fixtures are now templates. Historical v37 files were not edited, and an
Executor release no longer requires five Study copies. This does not pretend that an old
unexecuted frozen Study verifies against a newer source tree: only terminal evidence earns
a copied Executor source archive.

### 8: route and Workload now have separate owners

Compiler v24 removes `metadata.profile` without a compatibility alias. Schedule syntax
has one typed `lowering = {backend, entry_point}` record. Its three backend values name
materialization mechanisms, not operators; the two generated-backend rows are independent
of the thirteen corpus program slices. Global Buffers derive the executable signature, so
the unused and already-wrong ABI labels are gone.

The Lab remains the Workload consumer: it checks the opaque Workload digest and exact
external tensor contract for the admitted case. The Compiler checks Schedule/Target
legality and backend support. As failure-capable evidence, a GEMM bias-extent variant now
lowers without a named-Workload exception, while the Lab would reject a tensor mismatch
inside a frozen Flash-KMeans Run.

TinyGEMM2 is the bounded exception, not a counterexample. Its route names one checked
source asset; the four-part CTA sum, BF16-round epilogue and whole-Schedule semantic pin
block only lowering, not IR acceptance. The reduction drift Corpus case is therefore
`accepted=true, lowering_eligible=false`. ADR 0029 records the ownership boundary.

Frozen pre-v24 observations and ranking calibrations keep their old Schedule/source
bytes. They are not re-labelled as v24 evidence: successor B200 observations and a
successor ranking calibration instrument are currently missing. The v24 Corpus Gate is
32/32 and the zero-GPU contract suite passes 387 tests plus 322 parameterized subtests;
those gates validate the ownership migration but do not substitute for the missing
on-device successors.
