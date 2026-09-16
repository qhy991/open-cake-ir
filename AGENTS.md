# open-cake-ir local constraints

## Targets are peers, and the shared layers are vendor-neutral

Three architecture families are in scope as peers: NVIDIA across its architectures, Apple
GPUs across theirs, and AMD. Where this section conflicts with another rule in this file,
this section wins. It adds no portability -- the exact target match, a refusal that never
steps a Schedule down, and the absence of a layout algebra all stand exactly as written.
It is about the layers every target shares.

- **A vendor arrives as documents plus one typed registration.** A new target is a Target
  document under `compiler/targets/`. A new vendor is that, plus a backend module owning
  its own `preflight`, its own calibration and its own tests, plus one `LoweringBackend`
  member with its `BACKENDS` row -- held equal by
  `tests/contracts/test_backend_boundaries.py:78`. The declaration mechanisms already work:
  `instruction_contracts` and `synchronization_contracts` are open per-target string sets,
  and `memory_spaces` and `operation_kinds` are per-target capability sets refused by name.
  Anything beyond those sites that a new vendor forces you to edit in shared code is a
  defect in that code. Naming AMD once today means editing `compiler/target.py:22`,
  `evaluation/artifacts.py:4`, `lab/executor.py:183`,
  `verifier/hardware_conformance.py:199` and `verifier/program_safety.py:19` -- every one
  of them shared code, so what should be a data addition still arrives as a code change.
- **A hardware fact is declared by the Target that owns it, once.** The tell is a number in
  shared Python that no Target document can state. `Target.warp_size` is `return 32`, nine
  shared computations multiply by it, and the evidence for 32 exists only as citation prose
  inside the Target documents -- value and evidence already have two owners.
  `backends/metal.py:62` keeps a second copy and `backends/native_cuda.py:556` a third as a
  literal. A backend may keep its own ISA constant; it may not be the neutral owner of one.
  A duplicate is a defect while it still agrees, because it is found when it stops: the
  exported `launch_cubin_once` transcribes `BINARY_VERSION != 100` and a B200
  shared-memory literal (`evaluation/cuda_driver.py:479`, `:488`) of the two checks the
  same file already does per target at `:234` and `:241`, and the permissive direction is
  the dangerous one -- an `sm_100a` cubin passes `100 != 100` under an `sm_103a` manifest.
- **Vendor identity is declared; no vendor lives in the `else`.** `warps_per_warpgroup` is
  `4 if self.compute_capability is not None else None`, and the same inference recurs at
  `target.py:340`, `performance/profile.py:405`, `backends/triton.py:213`,
  `backends/cutedsl.py:157`, `evaluation/artifacts.py:14`, `evaluation/paired.py:58` and
  `lab/executor.py:183`. Measured: a well-formed `gfx942` document is refused with
  `target.compute_capability is required for CUDA targets`; give the same document a
  fabricated capability pair and it parses, then silently carries `warp_size` 32 and a
  four-warp warpgroup rule against a 64-lane wavefront. Every rule keyed on vendor reads a
  positively declared field. That includes the incumbent: a host, arm or assay declaring no
  kind is an explicitly named pre-`kind` CUDA form, never the fall-through.
- **An undeclared fact is reported, never substituted and never dereferenced.** Reproduced
  through the public boundary: `corpus/schedules/metal-rmsnorm-primary.json`
  (`apple_gpu_family8`) with `registers_per_thread: 32` on its one role raises
  `TypeError: unsupported operand type(s) for %: 'int' and 'NoneType'` at
  `verifier/hardware_conformance.py:689`. The refusal that owns that input,
  `METAL_REGISTER_CAP_UNSUPPORTED`, never runs, because verification precedes preflight.
  The correct shape is sixteen lines above the crash: `TARGET_REGISTER_CAP_UNMODELED`
  reports "this limit was not checked". A rule that raises or silently skips outside its
  evidence is not a gate, and a fail-open default -- `_CONTRACT_DTYPES.get(...)` at
  `hardware_conformance.py:346`, whose own comment says "Adding a contract adds a row
  here" -- is worse than a refusal.
- **A shared rule quotes the Target's declared sets; it never restates them, and never
  reads how a mnemonic is spelled.** `hardware_conformance.py:199` intersects
  `target.synchronization_contracts` with the literal `{"mbarrier", "barrier.sync"}`, which
  is not closed even for NVIDIA because `sm_100a.json` declares `triton_program_order`, and
  `program_safety.py:19` keeps the same literal. `PLACED_CONTRACT_PREFIXES`
  (`ir/operations.py:79`) puts three NVIDIA mnemonic prefixes in the vendor-neutral IR, and
  the authoring schema turns them into a refusal of any other vendor's atom placement.
  Decide membership in what the Target declares, and quote that set back. Instruction
  *typing* stays with the instruction (ADR 0022) -- declaring it per Target would duplicate
  every contract that `sm_100a` and `sm_103a` both admit.
- **Construction admits structure; an ISA range is a hardware-conformance Finding.**
  `Schedule.from_dict` resolves no Target, so a range that holds only because one ISA
  encodes it that way cannot live there. `Role.from_dict` nonetheless enforces the
  `setmaxnreg` immediate (`ir/resources.py:83`, quoting the instruction in its own comment)
  and the schema republishes it to every author of every target, while the two siblings in
  that same file do it correctly: unbounded at parse, gated with the Target in scope.
- **A refusal names the class it owns, and no vendor the caller did not name.** Refusing
  `gfx942` is required; refusing it in CUDA's words is not. `executable_role`
  (`evaluation/artifacts.py:14`) is the whole Evaluation layer's target-to-executable map
  and falls through a hardcoded Apple id set into `cuda_architecture`, so `gfx942`,
  `apple_gpu_family10` and `sm_120a` all raise `unsupported exact CUDA target`. This is the
  borrowed-block rule below, still open.
- **A gate report names the targets it examined.** `check_corpus` closes each case to
  `{case_id, schedule, expected}`, and the word "target" does not occur in
  `compiler/corpus.py`. Measured: of five declared targets, `apple_gpu_family7` and
  `apple_gpu_family9` have zero cases and the full Gate passes. Report absence as missing;
  do not mint cases to satisfy a count. No test asserts a vendor-neutrality invariant of
  shared code today, and `tests/contracts/test_compiler_revision_architecture.py:119` pins
  the exact five-target set, so a sixth target fails a test rather than being checked by
  one. A synthetic third-vendor Target fixture is the hook this section needs; keep the pin.
- **One vendor word is not the neutral word.** `Role.warps` is the only accepted spelling
  for a role's execution slots, and `core.py:275` reports `total_warps` for every target.
  Name a shared concept for the concept; vendor words belong in per-target diagnostics and
  in emitted source, as `METAL_ROLE_UNSUPPORTED` (`backends/metal.py:140`) already does --
  it reads the IR's `warps` and reports "consecutive SIMD groups". This one is deferred,
  not optional: `semantic_sha256` covers the whole Schedule document, so a rename voids every
  calibration binding and lands as its own successor-Revision act, separate from and after
  the width fix above.

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
  commitments -- on sm_100a an SMEM view offset, an operand byte offset, a TMEM column
  range; on another target its own storage in its own words -- and
  the compiler verifies they stay mutually consistent. New primitives extend verification
  coverage without an agent learning a second language.
- Static analysis is a pre-compile gate **only within its modeled domain**. It does not
  prove global GPU correctness or capture all microarchitectural behaviour, and both
  false positives and false negatives occur. On-device measurement is the ground truth;
  a cost estimate never replaces it.
- Feedback to an author is localized correctness and performance diagnostics. A pass/fail
  bit, or one latency number, is the failure mode this harness exists to avoid.
- Timing-model coverage and hardware facts are both evidence-gated per target. A target
  without its own calibration reports a coverage limitation, and a fact its Target document
  does not declare is reported unmodeled; it never inherits another target's estimates, and
  never another target's constants.
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
- Changes are test-gated across the kernel corpus and reviewed once at the merge to
  `main`, by a session other than the author's (ADR 0065).
- Recurring failures are what become new verifier rules, IR primitives, cost-model
  calibrations, explicit transformation passes and reusable tactics. A one-off failure
  is not evidence for a rule.

## Kernel–Compiler co-evolution (experience promotion)

- At the end of a bounded kernel optimization or recurring-failure investigation,
  inspect the retained evidence for a reusable mechanism and record the disposition in
  the existing result. `No promotion` is valid. Routine edits and status checks do not
  trigger this process; neither a new pass nor a new Finding is required for every run.
- For a promising transformation, retain one mechanism's before/after programs,
  applicability and anti-conditions, correctness obligations, observed effects and
  negative/null results. Reference the original evidence; do not create a second
  experience ledger or maintain separate facts in Case, recipe and pass descriptions.
- Route the lesson to its owner: deterministic Schedule rewrites to Compiler passes;
  measured choices of when/with which parameters to apply them to Lab recipes or
  selection; missing legality checks to the verifier; missing expressibility to IR or
  lowering; estimation errors to target-specific calibration. An unexplained one-off
  result remains a case.
- Promote a pass when a real second use or a stable structural invariant justifies its
  scope. Make matching and semantic/target preconditions explicit; return a complete
  candidate or a reason for not applying it, preserving the original. Keep performance
  choices explicit and let Lab select them. A pass does not imply a speedup or authorize
  broader reference access.
- Exercise positive cases, valid-but-ineligible inputs and numerical/side-effect
  counterexamples. Confirm that the intended guard refuses each counterexample, not an
  unrelated rule. Test uses outside the extraction cases before claiming transfer;
  CPU/source checks do not establish GPU correctness or performance.
- Use the Finding and Revision cadence below for promotion, review and release. Judge
  the mechanism by verified results on subsequent kernels under matched budgets,
  attempts avoided and negative transfer—not by the number of passes accumulated.

## Tick-tock between campaigns and Revisions (outer loop cadence)

- Campaigns run only on frozen code; that is the tock. A compiler or executor change is a
  tick: it lands as commits on a branch and reaches campaigns when it merges, so the commit
  a Campaign pinned never changes under it (ADR 0065).
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
  complete implementation for the target in any low-level or compiled form -- CUDA, PTX,
  SASS or a cubin, MSL or a `metal_binary_archive`, and their equivalents. Access is a
  semantic artifact role, not a file extension (ADR 0062): a restriction that a different
  extension satisfies is not a restriction. An
  external implementation may be run through the harness as a black-box baseline; its
  internals stay unavailable.
- **Known-kernel reproduction** -- may inspect the reference.
- **Direct low-level** -- may write the target's own low-level language, may not inspect an
  existing target implementation. `lab/reference_access.py:15` already names the three
  categories neutrally; the vendor leak is `environment_kind` at `:51`, which refuses an
  unenumerated arm for an inherited implementation it never had. An arm kind that is not
  enumerated is refused for the reason that applies to it.

## Measurement and replication (S5)

- On-device correctness checks and target-native timing. A target declares its timer, what
  the measured interval includes, and the device state reset before every timed sample; on
  B200 and B300 that is CUPTI timing with L2 flushed before every timed sample. A target
  that cannot state all three reports a measurement-coverage limitation instead of a
  latency, and inherits no other target's timing semantics. Every reported candidate is
  compiled, correctness-checked and benchmarked at the listed shape.
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
- Compiler changes require a full Corpus Gate and one review at the merge to `main`. CI runs the Gate on the
  commit; the reviewer is the owner or an agent session other than the author's, and the merge commit names the
  reviewer. No release document is minted per change (ADR 0065).
- Source identity is the clean commit of the checkout, so editing a tracked file is ordinary work: commit it.
  What a historical record pins stays pinned at its own commit. Released locks under `compiler/releases/`,
  descriptors under `runtime/executors/`, frozen `contracts/calibrations/*.json` plans and retained Evidence replay
  with the tools of the commit that produced them, never against today's tree (ADR 0065).
- **Run the suite in its own worktree, at a commit.** `git worktree add /tmp/<name> <commit>`, then run there.
  Source identity is the clean commit, so a tracked file changing under a running suite takes its identity away
  mid-run and every test that needs a Compiler or an Executor fails from that moment. Measured: a suite in the
  shared checkout was 40 minutes in when another session committed `compiler/targets/gfx1151.json`, and several
  hundred tests turned red at once -- reading exactly like broken code, not like a checkout someone touched. This
  repository is worked on by several sessions at a time, so the shared checkout is the one place a long run cannot
  survive. A worktree costs a `git worktree add` and removes the whole failure mode; it also lets an agent keep
  editing while the run finishes. Report which commit the run was at.

- A host is captured once per exact target as `runtime/hosts/<target>.json` and committed. A target without one is
  reported as having none, and no other host is substituted for it. Recapture only when the host itself changes,
  with `tools/capture_executor_host.py --target <target>`.
- A Compiler identity is `open-cake-ir@<commit>` and an Executor identity is `<target>@<commit>`; neither is
  chosen by hand, and a checkout carrying changes has neither. Historical released identities under
  `compiler/releases/` and `runtime/executors/` stay reserved and are never deleted or reused; see
  [ADR 0049](docs/adr/0049-released-executor-descriptors-reserve-their-identities.md) and
  [ADR 0050](docs/adr/0050-released-compiler-locks-reserve-their-identities.md) for what they meant.
- Never regenerate Corpus Gate expectations to make the gate pass — that reports a match it just manufactured. Adopt
  new expectations as a separate, reviewed act (`tools/refresh_corpus_expectations.py --write`). State the reason to
  the reviewer who writes the approval; do not write that basis yourself.
- More generally: a repair run just before the check it satisfies manufactures the state it then reports. If a check
  fails because of the environment, say so and stop — do not normalize the environment and rerun.
- Cake versus a target's own low-level language is an Authoring Environment assignment, not a syntax-only
  representation switch. CUDA on B200/B300 is today's instance of that second arm; the arm is the
  environment, never the language.
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
