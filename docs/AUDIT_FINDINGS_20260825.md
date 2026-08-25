# Audit findings register, 2026-08-25

An external read-only audit of this repository at `f503de4`, re-verified at `e351b67`.
Each finding records what was observed, what would falsify it, and its disposition.

Re-verification changed two of them. Finding 4 was refuted outright and is retained here
as refuted rather than deleted, because a register that keeps only its hits cannot be
audited. Finding 2 was materially narrower than first reported: the corpus half of it was
wrong and is corrected in place. Both corrections are stated where the finding is, not
folded away.

Findings are numbered as in the original report. This document is prose only; it pins no
source and belongs to no frozen closure.

## Disposition summary

| # | Finding | Disposition |
| --- | --- | --- |
| 1 | `EpilogueFormula` is declared but never read | open, needs Compiler successor |
| 2 | `examples/gpu/flash-kmeans-b32-smoke.json` does not parse | open, needs Executor successor |
| 3 | Committed Evidence fails its own auditor on permissions | **operationally fixed**, design open |
| 4 | `check_ranking_calibration.py` duplicates the ranking order | **refuted** |
| 5 | Provenance fields that cannot fail (`host`, `observed_at`) | **fixed here** |
| 6 | The Compiler release writes its own approval | open, design decision |
| 7 | Executor identity is inlined into every Study contract | open, design decision |
| 8 | The lowering profile table is the load-bearing constraint | open, design decision |
| 9 | Emitter preconditions have no Verifier finding | open, needs Compiler successor |

## 1. `EpilogueFormula` is parsed and then ignored

`src/open_cake_ir/compiler/ir.py` admits `EpilogueFormula` into
`EpilogueParameters.formula`. That parse is the field's only reader. Neither emitter nor
the Verifier consults it: `emit_cutedsl.py` hardcodes the centroid arithmetic in
`_emit_epilogue`, and `emit_triton.py` has no epilogue body at all.

Reproduced at `e351b67`: flipping `parameters.formula` from `centroid_sq_minus_two_dot`
to `bias_add_bf16_round` in `corpus/schedules/flash-kmeans-assignment-full.json` leaves
`accepted=True`, `lowering_eligible=True` and the finding list `['RESIDENCY_BOUND']`
unchanged, and the emitted CuTe-DSL source is byte-identical apart from the
`schedule_sha256` comment, centroid arithmetic included. A Schedule that declares a
biased BF16-rounded epilogue lowers to `norm[col] - 2.0*acc`.

This is a declaration the Compiler accepts and does not honour, which is worse than a
declaration it rejects. `bias_add_bf16_round` reaches lowering only through the TinyGEMM
asset template, so that member has never driven generated code.

Falsified by: a reader of `.formula` that changes emitted source, or a Verifier finding
that refuses a formula the selected backend does not implement.

Not fixed here because `ir.py` and `emit_cutedsl.py` are pinned by
`compiler/source_set.json`. Editing either requires a Compiler Revision successor, and
one was being minted concurrently when this audit ran. See "Why three findings are not
fixed here".

## 2. The beginner tutorial's example Schedule does not parse

`docs/GETTING_STARTED.md` points a first-time reader at
`examples/gpu/flash-kmeans-b32-smoke.json`. Loading it raises

    ScheduleParseError: schedule.operations[2].parameters unknown fields: ['formula']

It carries the removed `MmaFormula` field. The same is true of
`corpus/schedules/flash-kmeans-b32-smoke.json` and its `-shape-drift` sibling, but those
two are **not** a defect: `corpus/manifest.json` is what the Corpus is, both are absent
from it, and `tests/contracts/test_ir.py` derives its case list from the manifest for
exactly this reason. They are retained history and several frozen Study contracts,
release source sets and sealed Campaign authorities reference them. They must not be
deleted.

The `examples/gpu` copy is different: it is the artifact a new user is told to read, and
it is broken.

Falsified by: `Schedule.load` succeeding on every file under `examples/`.

Not fixed here because `examples/gpu/flash-kmeans-b32-smoke.json` is pinned by every
Executor Revision from v10 to v26. Editing it requires an Executor successor.

A durable fix should also add a contract test asserting that every Schedule under
`examples/` parses, so the tutorial cannot silently rot again.

## 3. Committed Evidence fails its own auditor

`src/open_cake_ir/evidence/store.py` refuses any evidence directory whose mode has
`0o022` set. Git records the executable bit and nothing else, so a checkout materialises
the committed archives under the caller's umask and the store then refuses to read its
own evidence.

This is stronger than an audit-gate gap. **The test suite does not pass on a fresh
clone.** Measured on a clean clone of `e351b67`, with no working-tree changes:

    2 failed, 339 passed, 226 subtests passed

    tests/contracts/test_compiler.py::…::test_released_compiler_source_closure_replays_from_evidence_v2
    tests/contracts/test_lab.py::…::test_historical_g8_r6_remains_a_non_scientific_replayable_qualification
    ValueError: evidence directory 'objects' is group/other writable

Control: both tests were re-run with every working-tree change stashed, and both still
failed, so the cause is the checkout and not any edit. Applying `0750` to directories and
`0440` to files under `evidence/` makes both pass, and `git status` does not observe the
change. That isolates the mechanism exactly.

Two consequences follow. The re-audit and independent-offline-audit gates in
`docs/ACCEPTANCE_GATES.md` cannot be executed against any evidence in this repository,
and the committed archives are in fact group-writable by every member of the owning
group.

Operationally fixed here by `tools/normalize_evidence_custody.py`, which reports
non-conforming paths and exits nonzero, and only repairs them under `--apply` -- a tool
that repaired what it reports would report a state it had just produced.
`docs/RUNBOOK.md` records the step. The tool is new, so it enters no frozen closure.

The design question stays open. The `0o022` refusal is right for a live Run, where
filesystem custody is what it asserts, and wrong for a committed archive, where git is
the integrity authority. The honest repair is to split them: a live Run verifies the hash
chain *and* custody; an archived Run verifies the hash chain and **records that custody
was not verified**, so absence of coverage cannot read as a passed check. That edit lands
in `store.py`, which the Executor Revision pins, so it needs a successor.

Falsified by: a fresh clone on which the two named tests pass without `--apply`.

## 4. Refuted: the ranking checker does not duplicate the shipped order

The original report held that `tools/check_ranking_calibration.py` hand-reimplements the
ranking key instead of importing `open_cake_ir.compiler.ranking.Cost.order`, and that its
third tie-break on `schedule_id` contradicts the shipped preorder, which deliberately
excludes it.

The observation is accurate and the conclusion is wrong. ADR 0018 records that the
shipped order *became* a preorder with abstention after calibration v6 was measured, and
states plainly that "the two frozen v6 repetitions remain valid negative evidence for the
old total-order decision" and that "calibration v6, its two raw records and its failed
decision remain unchanged and replay through their original source closure".

`check_ranking_calibration.py` is the frozen replay instrument for that historical
decision. Its key is the v6-era total order on purpose. The current instrument,
`tools/calibrate_gemm_ranking_interleaved.py`, imports `Cost` and `rank_for_cut`, honours
the abstention, reports the decisive/abstained split and refuses a domain with no
decisive subset. The historical/current split is already correct.

Changing the v6 checker would have broken the replay of a retained negative result. It
was not changed.

One residual and much smaller point: ADR 0011 describes the v6 checker as evaluating
"under the shipped ordering". That was true when written and became misleading when ADR
0018 changed the shipped ordering. ADRs are dated decision records, so the wording is
left as written; this entry is the cross-reference.

## 5. Fixed: provenance fields that could not fail

Two residency instruments wrote a hardcoded host literal and took their observation
timestamp from the command line:

- `tools/profile_lowered_kernel.py`
- `tools/observe_lowered_kernel.py`

Every record they produced therefore named the same host regardless of where it ran, and
carried whatever timestamp the operator typed. One committed residency record carries an
`observed_at` six hours later than the commit that introduced it, and all three are round
hours. A provenance field that the operator supplies is a field that reports a match it
just manufactured.

Both now derive `host` from `socket.gethostname()` and `observed_at` from the clock, and
the `--observed-at` argument is gone rather than left as a second authority.
`docs/RUNBOOK.md` and `docs/GETTING_STARTED.md` are updated to match.

`tools/calibrate_ranking_at_scale.py` has the same `observed_at` shape and was
deliberately **not** changed: it is pinned by the frozen calibration v6 contract and
`check_ranking_calibration.py` verifies its digest, so editing it would break the replay
of a retained negative result. Its host field was already derived correctly. Repairing
its timestamp requires a successor calibration, not an edit. That tool was edited during
this audit and restored once the pin was found; the digest was re-checked afterwards.

Falsified by: a record whose `host` differs from the machine that produced it.

## 6. The Compiler release writes its own approval

`tools/release_compiler_cycle.sh` regenerates the Corpus Gate report, then writes
`compiler/release-approval.json` itself with `reviewer` set to a string literal, then
feeds that file to the release as the approval. `src/open_cake_ir/compiler/release.py`
compares the persistent gate report against a gate report it recomputes from the same
inputs, and compares the approval's gate digest against that same recomputed value. Both
comparisons are between a deterministic function and its own output on unchanged inputs.
The `--verify` pass reads back what the previous loop iteration wrote.

Three of the four checks in the Compiler release cannot fail. The only falsifiable
assertion is that the matched case count equals the case count, and
`tools/refresh_corpus_expectations.py` adopts observed output as the new expectation, in
the same commit as the compiler change, in every commit that has moved it.

The Executor freeze is weaker still: `tools/release_executor.py` consults no prior
expectation at all, and its one guard, a refusal to reuse a released identity, is
disarmed by `tools/release_executor_cycle.sh`, which unlinks the descriptors above the
retained history before invoking it. `inventory/COMPILER_REVISION_IDENTITY_INCIDENT_20260824.json`
records what this already caused: seven distinct Compiler bodies sharing one identity.

This is a design decision, not a patch. Either the approval becomes an artifact a second
party produces, or it is removed from `docs/ACCEPTANCE_GATES.md` and stops being
described as a gate. A ceremony that cannot fail is worse than no ceremony, because it
manufactures the appearance of review.

## 7. Executor identity is inlined into every Study contract

Each frozen Study embeds the Executor identity and a canonical digest. With four Study
families and an Executor that revises many times a day, every Executor bump mints four
new near-identical contracts whose diff against their predecessor is three fields. The
great majority of the resulting contracts are referenced by nothing, and only a small
minority were ever executed.

One fact, many owners. Replacing the inline identity with a reference to the current
Executor descriptor, or generating Study successors at run time instead of committing
them, removes most of the churn without weakening any freeze guarantee.

## 8. The lowering profile table is the load-bearing constraint

The Schedule body is a real language. A composed operation that appears nowhere in the
corpus emits correctly, and retiling emits correctly; `emit_triton` is a genuine
structural traversal.

But `metadata.profile` is matched against a closed table, and a structurally valid
Schedule carrying an unknown profile name is refused outright by
`LOWERING_PROFILE_UNSUPPORTED`. Each admitted operator contributes a table row and a
bespoke conformance function, several of which assert literal constants or a literal
operation-id string. The corpus is growing faster than the language.

This sits in tension with ADR 0020, which correctly refuses to turn the external KDA
history into profiles and reports honest zero coverage against it. The repository applies
a strict expressibility standard outward and grows a lookup table inward. Both cannot
hold. Either conformance is derived from the Workload Contract and the profile table
degrades to backend routing, or the boundary is stated plainly: this is a compiler for a
named set of operators, and corpus size stops being evidence of reach.

## 9. Emitter preconditions have no Verifier finding

`compiler/AUTHORING_CONTRACT.md` promises that a blocking Finding is the reason an
Assessment is not lowering-eligible and that no such candidate reaches the toolchain.
Several emitter preconditions are not expressed as Findings: a single role and at most
one tile loop for the Triton backend, a single pipeline and exactly one MMA and one
epilogue for the CuTe backend.

A Schedule violating them is assessed `accepted` and `lowering_eligible`, and then fails
inside `lower()` with a `CompilerError` string. That is exactly the class of late failure
the Verifier exists to prevent, and it is a contract violation rather than a rough edge.

Promoting each precondition to a Finding is also the smallest concrete step toward
resolving 8, because it moves refusal from a profile-name lookup to a property of the
Schedule.

## Why three findings are not fixed here

Findings 1, 2 and 3 each touch a file inside a frozen closure:

- `src/open_cake_ir/compiler/ir.py` and `emit_cutedsl.py` are pinned by
  `compiler/source_set.json`;
- `src/open_cake_ir/evidence/store.py` and `examples/gpu/flash-kmeans-b32-smoke.json` are
  pinned by the Executor Revision.

Editing any of them requires minting a Compiler or Executor Revision successor. A
Compiler release cycle was in flight elsewhere while this audit ran. Minting a second
successor concurrently is precisely the mechanism that produced the identity incident
cited in 6, so these repairs are recorded rather than raced. They should land as one
reviewed successor cycle once no other cycle is in flight.
