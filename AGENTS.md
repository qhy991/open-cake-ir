# open-cake-ir local constraints

## Cake IR design principles (arXiv:2608.12629v1, Appendix B.1)

The paper states eight. They bind IR changes here; the global doctrine covers the rest.

- **P1 Ergonomic** — keep the editing model familiar to NumPy/PyTorch users, and avoid
  unnecessary destination-passing or grid bookkeeping.
- **P2 Performance-transparent** — keep performance-relevant hardware decisions visible
  and lowering behaviour inspectable.
- **P3 Canonical** — prefer one canonical form for each operation over equivalent
  alternative spellings.
- **P4 Statically type-checked** — use typing rules to constrain lowering and reject
  ill-typed programs during construction, not at emission.
- **P5 Analysis-friendly** — expose the information the supported static analyses need.
- **P6 Test-gated** — evaluate IR changes against the kernel-matrix tests.
- **P7 Analysis-consistent** — accompany changes to the IR data model with the
  corresponding analysis updates, in the same change.
- **P8 Hardware-grounded** — document the intended hardware behaviour of each operation.

## What the paper says not to do

- Layout is deliberately **not** a first-class abstraction. Do not introduce a layout
  algebra for an agent to manipulate. A Schedule records concrete storage and access
  commitments -- an SMEM view offset, an operand byte offset, a TMEM column range -- and
  the compiler verifies they stay mutually consistent. New primitives extend verification
  coverage without an agent learning a second language.
- Static analysis is a pre-compile gate **only within its modeled domain**. It does not
  prove global GPU correctness or capture all microarchitectural behaviour, and both
  false positives and false negatives occur. On-device measurement is the ground truth;
  a cost estimate never replaces it.
- Feedback to an author is localized correctness and performance diagnostics. A pass/fail
  bit, or one latency number, is the failure mode this harness exists to avoid.
- Timing-model coverage is evidence-gated per target. A target without its own calibration
  reports a coverage limitation; it never inherits another target's estimates.
- The paper uses human judgement to gate compiler evolution. This project's owner now
  permits an independent agent reviewer under ADR 0052; an automatic estimate still
  does not authorize a Revision.

## What the harness must do (S3)

- A blocking check names the affected program region and the class of contract violated.
  The four classes are schedule semantics, hardware conformance, data consistency and
  program safety; analysis may block only within them.
- Correctness is decided against an external reference across different shapes and input
  distributions. Final acceptance requires end-to-end evaluation in the target framework.
- The cost model ranks and filters candidates *before* they reach GPU time. It never
  decides acceptance.
- The compiler requires an exact target match. It reports missing device or toolchain
  support; it never steps a schedule down to another architecture.

## Learned here, not from the paper

Both of these were found by probing this repository and cost a wrong answer or a wrong
record before they were understood. They are corollaries of rules above, written as
things to do because the rules above did not stop either one.

- **A report states the domain it examined.** Omitting a resource the Schedule declares
  nothing for reads as "does not constrain" and means "was not looked at", and the reader
  will assume the generous one. Measured: `gemm-bias-b1-smoke` declares no shared memory,
  Triton allocates it for the `tl.dot` operands, and it bounds residency exactly as
  tightly as the registers the analysis does model.
- **A block from an unrelated rule is on loan.** When a probe shows a hazard is refused,
  ask which rule refused it. Twice here the answer was a rule about something else --
  tensor-memory ownership by a synchronisation rule, a register-held contraction
  accumulator by a lifetime rule -- and either can be scoped later by someone who checked
  every consequence they knew of. Close it properly or record that the block is borrowed.

## SHA checks are exceptional

- Do not compute, re-check, enumerate or report SHA-256 digests as a routine review,
  test or status ritual. A byte-identity relation already verified remains settled until
  one of its inputs or its owning boundary changes.
- A digest check is allowed only when exact byte identity is the live uncertainty at a
  frozen-release, external-handoff or explicitly pinned-evidence boundary, and a mismatch
  would change the next action. State that boundary instead of treating the digest as
  evidence of semantic correctness.
- Check each required identity relation once at that boundary, preferably through its
  existing release or audit command. Do not manually re-hash every referenced file, repeat
  the same check in later reviews, or mirror a digest catalogue in semantic contract tests.
- Reviews and status reports name artifact ids, paths and commits by default. Include a
  digest only for a newly established boundary, an actual mismatch or an explicit user
  request.
- Do not add new digest fields, projections, inventories or hash-only tests unless they
  replace materially more expensive work and their result selects a different action.

## Compiler evolution (S3, outer loop)

- A proposal is checked against P1-P8 before it is implemented. A new primitive must be
  performance-transparent and verification-friendly.
- A primitive and its analyses evolve together. Syntax without effects and legality rules
  makes the IR less analyzable, which is a reason to refuse it, not to defer them.
- Changes are test-gated across the kernel corpus and require independent review under
  ADR 0052 before release or merge.
- Recurring failures are what become new verifier rules, IR primitives, cost-model
  calibrations and reusable tactics. A one-off failure is not evidence for a rule.

## Tick-tock between campaigns and Revisions (outer loop cadence)

- Campaigns run only on frozen Revisions; that is the tock. A compiler or executor change
  is a tick that mints successor Revisions through the existing release cycles and never
  edits a pinned source in place.
- The unit of curation between the two is a Finding under `findings/`, append-only like
  evidence and following that directory's record contract. A finding cites workspaces,
  runs and event sequences by path and index -- the ledger's own hash chain settles byte
  identity, so findings do not copy digests.
- A finding closes only when `implemented_in` names the Revision that changed it and
  `verified_by` names its verification. Bug-fix and protocol ticks verify by replaying
  the cited failing evidence; capability ticks verify by new campaigns on the successor.
- Cadence is findings-triggered, not calendar-driven: sweep tasks until the set is covered
  or blocking findings accumulate, then tick, verify, and resume the sweep.
- The light closure convention: a tick's commit names the finding ids it addresses, and
  the finding's `implemented_in`/`verified_by` backfill lands in that same change.
- Cross-version comparison anchors on a fixed baseline bundle per task, so a tick reports
  two facts separately: baseline movement (compiler floor) and candidate-minus-baseline
  (provider headroom). Findings name the executor Revision and target; cross-backend
  comparison lives at the findings layer, never inside one campaign.

## The agent loop (S4)

Four stages, in order. A campaign that collapses them is not running this loop.

1. Generate **structurally distinct** candidates -- not variations of one shape.
2. Filter before GPU time: IR construction checks, then verifier hard gates, then
   cost-model ranking.
3. Evaluate survivors against the external oracle, with benchmarking **and profiler**
   evidence.
4. Route each diagnosis to whichever of candidate, verifier, cost model or IR vocabulary
   it belongs to.

The Workload Contract is the stable authority throughout, fixing shapes, oracle,
tolerances, hardware and permitted references. Retained results are what make decisions
auditable and recurring findings reusable.

## Reference access, per arm (S5)

Each arm's authoring environment declares what its author may see. Getting this wrong
invalidates the comparison, not just the run.

- **Clean start / frontier synthesis** -- may inspect the mathematical specification,
  evaluation contract, correctness oracle and high-level code. May **not** inspect a
  low-level target implementation (CUDA, PTX, SASS, or equivalent generated source). An
  external implementation may be run through the harness as a black-box baseline; its
  internals stay unavailable.
- **Known-kernel reproduction** -- may inspect the reference.
- **Direct CUDA/PTX** -- may write low-level code, may not inspect an existing target
  implementation.

## Measurement and replication (S5)

- On-GPU correctness checks and CUPTI timing on B200, with L2 flushed before every timed
  sample. Every reported candidate is compiled, correctness-checked and benchmarked at
  the listed shape.
- A replicated clean start fixes the agent and scaffold, model and reasoning effort, task
  statement, oracle, benchmark harness and the single target shape. Report median
  [min, max] across the matched runs, and retain the stopping and timing accounting.
- Model and scaffold are held fixed so a difference is attributable to the environment
  rather than to model capability.

## Portfolio generalization (S6)

- Generalization begins only after strong per-shape seeds exist. Scoring the inner loop
  on broad coverage weakens the signal it exists to produce.
- An incorrect or slow seed returns to the inner loop. It is never hidden behind a
  dispatcher predicate.
- Portfolio validation covers representative and held-out inputs, boundary and tail
  cases, overlapping or missing guards, and the fallback path.

- The Compiler is the product core. It must not import Lab, provider, workload, campaign, evidence-store or claim code.
- The Lab may depend on a frozen Compiler Revision; a campaign may never mutate that revision.
- Workload Contract owns operator semantics and oracle. Study Contract owns treatment, estimand and analysis.
- Study templates are stable and execution binding lives in the CampaignLock. Do not mint a frozen Study successor per
  Compiler or Executor Revision; re-stamping identity into every contract gives one fact many owners.
- `matched_search` and the evidence-justified exact-shape `portfolio` variant share one Lab path. Serving remains a
  future artifact handoff, not a runtime mode.
- `artifact_optimization_only` is a non-scientific Claim Scope on `matched_search`, not a mode. It may expose
  provider-default features, but Candidate promotion still requires common confirmatory Evaluation and never forms
  an arm comparison.
- KernelSeed and Workload-case specialization are owned by Lab; Compiler accepts complete Schedules and must remain
  unaware of held-out roles or Study policy.
- A Schedule declares its own `lowering` route — backend and entry point. There is no profile table to add a row to,
  and no per-operator conformance function to write. Refusal is a property of the Schedule: an unsupported dtype or
  operation body is a backend capability Finding before lowering, not a name lookup. A new operator that composes
  existing primitives therefore needs a Workload Contract, not a Compiler change; if it needs a Compiler change, say
  which primitive is missing rather than widening a route.
- Compiler changes require a full Corpus Gate and independent approval before producing a new Compiler Revision.
  The author and release cycle may prepare a release but may not write `compiler/release-approval.json`.
  A human reviewer or a distinct agent session may write it after reviewing the exact change and Gate. An agent
  reviewer must use a model in `ALLOWED_REVIEW_MODELS` in `src/open_cake_ir/compiler/release.py`; record the actual
  model and distinct author/reviewer session ids, with no silent model substitution or author self-approval.
  Missing, malformed, stale, disallowed-model or same-session approval exits 3 and preserves the prior lock and
  approval bytes. See ADR 0052, which supersedes ADR 0030's human-only interpretation.
- A file can be frozen by being named with a digest somewhere else, not only by living under `evidence/`. Before
  editing anything under `tools/`, `src/`, `examples/` or `corpus/`, check whether `compiler/source_set.json`, a
  `runtime/executors/*.json` closure or a `contracts/calibrations/*.json` plan pins its bytes. Editing a pinned file
  breaks the replay of whatever that digest supports; the answer is a successor, not an edit. If you already edited
  one, restore it and re-verify the digest before doing anything else.
- A Revision id is derived by its cycle script, never chosen by hand. Released Compiler locks and Executor
  descriptors reserve their identities even without local consumers; never delete or reuse one based on an
  absence of local witnesses. Compiler preparation reuses its pending draft until release, and an unchanged
  verified release is a no-op. See [ADR 0049](docs/adr/0049-released-executor-descriptors-reserve-their-identities.md)
  and [ADR 0050](docs/adr/0050-released-compiler-locks-reserve-their-identities.md).
- Never regenerate Corpus Gate expectations to make the gate pass — that reports a match it just manufactured. Adopt
  new expectations as a separate, reviewed act (`tools/refresh_corpus_expectations.py --write`). State the reason to
  the reviewer who writes the approval; do not write that basis yourself.
- More generally: a repair run just before the check it satisfies manufactures the state it then reports. If a check
  fails because of the environment, say so and stop — do not normalize the environment and rerun.
- Cake versus CUDA is an Authoring Environment assignment, not a syntax-only representation switch.
- Common Evaluation begins only after an arm produces a sealed launchable artifact.
- Every candidate, evaluation and terminal outcome is append-only; reports and status docs are derived views.
- An Evidence audit returns two orthogonal facts. `archive_integrity` can be true while `filesystem_custody_verified`
  is false: a git checkout replays correctly but carries permissive modes, because git records the executable bit and
  nothing else. Such an archive supports no promotion, system qualification or estimate. Never restore modes so a
  custody check passes — present mode bits cannot prove continuous historical custody (ADR 0031). A contract test that
  asserts a claim-bearing property therefore needs a custody environment; report that as an environment precondition
  rather than manufacturing one.
- Do not copy legacy `rXX`, `vN`, failure or archive runners. Historical implementation lives in pinned Git.
- No formal provider or GPU experiment is authorized before all applicable acceptance gates pass.
- Generated runs and secret bytes stay outside source. Cleanup of legacy data requires separate user authorization.
- Every new Campaign Lock, Evidence root and report stays outside the checkout; historical in-checkout Campaigns are
  read-only replay inputs. New lifecycle layout follows ADR 0005 only at successor Revision boundaries.
