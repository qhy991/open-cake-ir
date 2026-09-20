# open-cake-ir local constraints

Paths below are relative to `src/open_cake_ir/` unless they start with `tests/`, `tools/`,
`compiler/targets/`, `corpus/` or `docs/`.

## Branch and worktree routing

Follow [Development branches](docs/DEVELOPMENT_BRANCHES.md), the canonical owner of the
maintained branches, task naming, shared-code routing, integration and synchronization.
New work uses `task/<platform-or-core>-<subject>` without a `codex/` prefix and its own
worktree. Inspect existing worktrees before editing; preserve other tasks' local branches
and active checkouts. Merged task branches are removed from GitHub; archive tags retain historical
work outside the maintained branches. A platform branch is a maintenance boundary,
not a fork of the shared Compiler.

## GPU resource discipline

Before preparing, launching or releasing GPU Evaluation work, read and apply the
installed `gpu-infra` skill's
[GPU lease lifecycle](https://github.com/qhy991/gpu-infra/blob/main/skills/gpu-infra/SKILL.md#gpu-lease-lifecycle).
That skill owns allocation and release procedure; keep its rules in that one place.
For this repository, a phase-boundary change is an Executor/source change: implement
and validate it in a successor commit before a new Campaign, preserving the Workload,
oracle and measurement contract. Existing frozen Campaigns retain their original code.

## Targets are peers, and the shared layers are vendor-neutral

Five vendors are declared as peers -- NVIDIA, Apple, AMD, Hygon and MetaX, the closed `Vendor`
set in `compiler/target.py` -- across eight declared targets. Where this section
conflicts with another rule in this file, this section wins. It adds no portability: the
exact target match, a refusal that never steps a Schedule down, and the absence of a layout
algebra all stand. Each invariant names the code that holds it and the test that pins it,
by symbol rather than by line so the citation survives an edit; a measured defect belongs
in `findings/`, not here.

- **A same-vendor target is a document.** `tests/contracts/test_target_documents.py::
  SameVendorTargetIsADocumentTest` loads a synthetic `sm_120a` and a synthetic
  `apple_gpu_family10` beside the declared documents, lowers one through Triton and
  one through Metal with no shared-code edit, routes the Triton emission offline, and
  checks that `sm_120a` refuses what it does not declare. A new vendor is that plus its
  own platform package and a `Vendor` member.
- **A hardware fact is declared by the Target that owns it, once.** `Target.from_dict`
  reads `warp_size` from the document and refuses an undeclared width; the slot budget
  `ResourceLimits.maximum_warps_per_cta` is derived from it, never declared beside it;
  `maximum_tensor_memory_bytes` is declared exactly by documents that declare the
  `tensor` space and is `None`, not zero, elsewhere. Pinned by
  `test_declared_warp_size.py::DeclaredWarpSize` and
  `test_target_documents.py::DeclaredDocumentsTest`.
- **Vendor identity is declared; no vendor lives in the `else`.** `Vendor` and
  `CodeObject` are closed enums (D2). Field admission in the parser keys on the code
  object, not the vendor: `compute_capability` and `warps_per_warpgroup` exist exactly on
  `cubin` documents. MACA's `triton_arch` is an explicit compatibility API value, distinct
  from its physical target and native codegen family (`test_metax_platform.py`).
  `test_vendor_neutrality.py::NoVendorInTheElseTest` fails on a shared
  rule that reads the presence of a CUDA field as identity, and the rest of that file
  holds the shared layers against a synthetic third-vendor fixture bound by no Revision.
- **An undeclared fact is reported, never substituted and never dereferenced.** A limit
  the Target does not state is a REPORT-level "not checked": `TARGET_REGISTER_CAP_UNMODELED`
  and `TARGET_WARPGROUP_WIDTH_UNMODELED` in `compiler/verifier/hardware_conformance.py`;
  Corpus case `gfx938-register-budget-unenforceable` pins the first. A fail-open default
  is worse than a refusal.
- **An instruction contract has one owner.** `compiler/ir/instruction_contracts.py` is
  the registry of every name a Target may declare (kind, operand and accumulator dtypes,
  placement, realized barrier mechanism); `revision._load_target` refuses a document that
  names a contract no record owns; the verifier reads the record and refuses an unowned
  name with `INSTRUCTION_CONTRACT_UNKNOWN`; a backend keeps only its emission spelling.
  Pinned by `test_instruction_contracts.py::AdmittedContractsHaveTheirAnalyses` and
  `test_barrier_mechanism_ownership.py::BarrierMechanismOwnership` (neither verifier file
  keeps its own copy of the barrier pair). Typing stays with the instruction (ADR 0022).
- **Construction admits structure; an ISA range is a hardware-conformance Finding.**
  `Schedule.from_dict` resolves no Target, so `Role.from_dict` parses a register budget
  unbounded and the backend that encodes the instruction owns its range;
  `test_register_split_ownership.py` pins it.
- **Route facts ride the emission, and the offline jail opens no document (D3).**
  `backends.triton.target_route_facts(target)` writes `code_object`, `triton_arch` and
  `warp_size` into the toolchain requirements, plus the declared native `codegen_arch`
  where it differs from the API architecture; `toolchain.triton_route(requirements)`
  derives everything else from the code object and refuses a missing key. No backend or
  toolchain module keeps a table of target ids, a compute-capability pair, or a lane
  width the document already declares. `test_vendor_neutrality.py::
  OfflineRouteMatchesEveryDeclaredDocumentTest` routes every declared document by its
  own facts.
- **A backend declares the code objects it emits, and the Compiler refuses the mismatch
  by name.** Every backend module carries `CODE_OBJECTS`; `Compiler.assess` emits
  `BACKEND_TARGET_UNSUPPORTED` when the Schedule's backend does not emit the Target's
  object, and skips that backend's preflight. The two remaining per-target gates are
  declared evidence sets, not capability tables: `backends/cutedsl.py::_TMEM_ROUTE_EVIDENCE`
  and `backends/cutedsl_register.py::REGISTER_ROUTE_EVIDENCE` (ADR 0070, a
  qualification act), `compiler/passes.py::_WARP_SPECIALIZATION_EVIDENCE` (bounded
  NVIDIA Triton evidence). Widening any of them is a reviewed act.
- **A refusal names the class it owns, and no vendor the caller did not name.**
  `evaluation/artifacts.py::executable_role` reads the declared `code_object` instead of
  partitioning target ids by hand; `test_declared_code_object.py`,
  `test_execution_platform.py::EveryDeclaredObjectIsARow` and
  `test_vendor_neutrality.py::RefusalOwnershipTest` hold the wording.
- **The target match is exact; nothing steps down.** `declared_target` refuses an id no
  document declares; the CUDA launch checks the binary version against the manifest's
  own declared compute capability (`evaluation/cuda_driver.py`); an id no document
  declares has no route (`test_vendor_neutrality.py::
  test_an_unlisted_target_is_refused_not_stepped_down`).
- **A backend is an emission mechanism, not a vendor.** `LoweringBackend` has four
  members, and the Triton route serves sm_100a, sm_103a, gfx938 and gfx1151 from one of
  them, and the MACA route serves xcore1002 through the same emitter. A new vendor
  reaching an existing emitter adds a Target document and, where
  needed, that backend's own preflight, calibration and tests -- no `LoweringBackend`
  member. `test_backend_boundaries.py` holds the registry equal to the enum and every
  backend's `CODE_OBJECTS` declared.
- **A Target addition is a reviewed decision, not a data drop (D2).** The declared set is
  pinned in `test_compiler_revision_architecture.py` and the admitted-contract snapshot
  in `test_instruction_contracts.py`; a new Target edits both, on purpose; the contract
  and capability sets a document declares are otherwise open and refused by name.
- **A gate report names the targets it examined.** `CorpusGateReport.unexamined_targets`
  reports absence beside coverage; `test_corpus_target_coverage.py` holds that absence is
  reported and never blocks the Gate (D10). Do not mint cases to satisfy a count:
  `apple_gpu_family7` and `apple_gpu_family9` are reported unexamined today.
- **Shared roles use execution groups.** Schedule v2 declares `Role.execution_groups`,
  and assessments report `total_execution_groups`. The Target owns each group's lane
  width. Native ISA and toolchain names remain in backend diagnostics and emitted source.
  Version 1 is replayed at its pinned commit; it is not silently translated at admission.


The five measured defects formerly tracked here are closed in F-2026-09-18-006,
which retains their original evidence and the software verification scope.

## Cake IR design principles (arXiv:2608.12629v1, Appendix B.1)

The paper states eight. They bind IR changes here.

- **P1 Ergonomic** — an editing model familiar to NumPy/PyTorch users; no unnecessary
  destination-passing or grid bookkeeping.
- **P2 Performance-transparent** — performance-relevant hardware decisions stay visible
  and lowering behaviour inspectable.
- **P3 Canonical** — one canonical form per operation.
- **P4 Statically type-checked** — typing rules reject ill-typed programs at construction,
  not at emission.
- **P5 Analysis-friendly** — expose what the supported static analyses need.
- **P6 Test-gated** — IR changes are evaluated against the kernel-matrix tests.
- **P7 Analysis-consistent** — a data-model change carries its analysis updates in the
  same change.
- **P8 Hardware-grounded** — each operation documents its intended hardware behaviour.

## What the paper says not to do

- Layout is deliberately **not** a first-class abstraction. A Schedule records concrete
  storage and access commitments -- on sm_100a an SMEM view offset, an operand byte offset,
  a TMEM column range; on another target its own storage in its own words -- and the
  compiler verifies they stay mutually consistent. Do not introduce a layout algebra.
- Static analysis is a pre-compile gate **only within its modeled domain**; false positives
  and false negatives both occur. On-device measurement is the ground truth; a cost
  estimate never replaces it.
- Feedback to an author is localized correctness and performance diagnostics. A pass/fail
  bit, or one latency number, is the failure mode this harness exists to avoid.
- Timing-model coverage and hardware facts are evidence-gated per target. A target without
  its own calibration reports a coverage limitation; a fact its document does not declare
  is reported unmodeled; nothing inherits another target's estimates or constants.
- Compiler evolution is gated by judgement, human or agent; an automatic estimate never
  authorizes a change.

## What the harness must do (S3)

- A blocking check names the affected program region and the class of contract violated:
  schedule semantics, hardware conformance, data consistency or program safety. Analysis
  may block only within them.
- Correctness is decided against an external reference across shapes and input
  distributions; final acceptance requires end-to-end evaluation in the target framework.
- The cost model ranks and filters candidates *before* GPU time. It never decides acceptance.
- The compiler requires an exact target match and reports missing device or toolchain
  support; it never steps a schedule down to another architecture.

## Learned here, not from the paper

- **A report states the domain it examined.** Omitting a resource the Schedule declares
  nothing for reads as "does not constrain" and means "was not looked at". Measured:
  `gemm-bias-b1-smoke` declares no shared memory, Triton allocates it for the `tl.dot`
  operands, and it bounds residency as tightly as the registers the analysis does model.
- **A block from an unrelated rule is on loan.** When a probe shows a hazard is refused,
  ask which rule refused it. Twice here it was a rule about something else -- tensor-memory
  ownership by a synchronisation rule, a register-held accumulator by a lifetime rule.
  Close it properly or record that the block is borrowed.

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

- A proposal is checked against P1-P8 before it is implemented. A primitive and its
  analyses evolve together: syntax without effects and legality rules is a reason to
  refuse it, not to defer them.
- Changes are test-gated across the kernel corpus. Follow the current
  [branch workflow](docs/DEVELOPMENT_BRANCHES.md) for task integration and independent
  review of Compiler changes before `main`; that owner-selected workflow replaces the
  earlier D9 direct-commit default. CI runs the Corpus Gate on applicable pushes and PRs;
  no per-change release document or machine-checked reviewer field is introduced.
- Recurring failures become verifier rules, IR primitives, transformation passes and
  reusable tactics. A one-off failure is not evidence for a rule.
- After a bounded optimization or recurring-failure investigation, record the promotion
  disposition in the existing result; `No promotion` is valid. Route the lesson to its
  owner: rewrites to Compiler passes, when-to-apply choices to Lab recipes, missing
  legality to the verifier, missing expressibility to IR or lowering. A promoted pass
  states its preconditions, returns a complete candidate or a reason, and is exercised on
  counterexamples that the intended guard -- not an unrelated rule -- refuses.

- In fusion, tiling and memory-hierarchy investigations, distinguish reusable rewrites
  from target-specific mappings and record the promotion disposition under the existing
  workflow; the [transfer design](docs/OPTIMIZATION_TRANSFER.md) owns the method. In controlled
  studies, the Study's material and pass-access treatment takes precedence: do not inject
  withheld experience or transformations into an ablation arm. This is a research priority,
  not evidence that transfer works or a requirement to mint a pass.

## Tick-tock between campaigns and Revisions

- Campaigns run only on frozen code; that is the tock. A compiler or executor change is a
  tick: it lands as a commit, so the commit a Campaign pinned never changes under it
  (ADR 0065).
- The unit of curation between the two is a Finding under `findings/`, append-only and
  following that directory's record contract; it cites workspaces, runs and event
  sequences by path and index and copies no digests.
- A finding closes only when `implemented_in` names the commit that changed it and
  `verified_by` its verification: bug and protocol ticks by replaying the cited failing
  evidence, capability ticks by new campaigns on the successor. A tick's commit names the
  finding ids it addresses and carries their backfill.
- Cadence is findings-triggered, not calendar-driven. Cross-version comparison anchors on
  a fixed baseline bundle per task and reports compiler floor and provider headroom
  separately; cross-backend comparison lives at the findings layer, never inside one campaign.

## The agent loop (S4)

Four stages, in order; a campaign that collapses them is not running this loop.
1. Generate **structurally distinct** candidates. 2. Filter before GPU time: IR
construction checks, verifier hard gates, cost-model ranking. 3. Evaluate survivors against
the external oracle with benchmarking **and profiler** evidence. 4. Route each diagnosis to
candidate, verifier, cost model or IR vocabulary. The Workload Contract is the stable
authority throughout.

## Reference access, per arm (S5)

Each arm's authoring environment declares what its author may see
(`lab/reference_access.py:15` names the three categories); getting this wrong invalidates
the comparison, not just the run.

- **Clean start / frontier synthesis** -- may inspect the specification, evaluation
  contract, oracle and high-level code; may **not** inspect a complete implementation for
  the target in any low-level or compiled form (CUDA, PTX, SASS, a cubin, MSL, a
  `metal_binary_archive`, and their equivalents). Access is a semantic artifact role, not a
  file extension (ADR 0062). An external implementation may run as a black-box baseline.
- **Known-kernel reproduction** -- may inspect the reference.
- **Direct low-level** -- may write the target's own low-level language, may not inspect an
  existing implementation. Cake versus that language is an Authoring Environment
  assignment, never a syntax switch; CUDA on B200/B300 is today's instance.

## Measurement, replication and portfolios (S5, S6)

- A target declares its timer, what the measured interval includes, and the device-state
  reset before every timed sample (on B200 and B300: CUPTI, L2 flushed). A target that
  cannot state all three reports a measurement-coverage limitation instead of a latency
  and inherits no other target's timing semantics.
- A replicated clean start fixes agent, scaffold, model, reasoning effort, task statement,
  oracle, harness and the single target shape; report median [min, max] and retain the
  stopping and timing accounting.
- Generalization begins only after strong per-shape seeds exist. An incorrect or slow seed
  returns to the inner loop, never hidden behind a dispatcher predicate. Portfolio
  validation covers held-out inputs, boundary and tail cases, overlapping or missing
  guards, and the fallback path.

## Boundaries, identity and evidence

- The Compiler is the product core and imports no Lab, provider, workload, campaign,
  evidence-store or claim code (`tests/contracts/test_compiler.py:101`); the common layers
  import no task implementation (`tests/contracts/test_task_boundaries.py:13`, ADR 0055).
- Workload Contract owns operator semantics and oracle; Study owns treatment, estimand and
  analysis; RunSpecification owns one execution's inputs, permissions and budgets. Lab owns
  KernelSeed and Workload-case specialization. The Compiler owns complete Programs, their
  leaf Schedules and deterministic rewrites, and stays unaware of Study policy.
- Study templates are stable and execution binding lives in RunSpecification. CampaignLock
  remains an input adapter to the same Run engine, not a second execution owner. Do not mint
  a frozen Study successor per Compiler or Executor change; the 36
  `flash-kmeans-r45-portfolio-reconstruction` successors that predate this rule live on
  the `history` tag together with the retired template and lifecycle.
- `matched_search` is the sole live Study kind; the Portfolio Study lifecycle is retired;
  `artifact_optimization_only` is a Claim Scope on it, not a mode, and promotion under it
  still requires common confirmatory Evaluation and forms no arm comparison.
  Independent engineering Runs need no Study; assigned Runs retain their Study's policy.
- A Schedule declares its own `lowering` route. Refusal is a property of the Schedule: an
  unsupported dtype or operation body is a backend capability Finding before lowering, not
  a name lookup. An operator that composes existing primitives needs a Workload Contract,
  not a Compiler change.
- Source identity is the clean commit of the checkout (`source_identity.py:39`;
  `tests/contracts/test_source_identity.py:45`). A Compiler identity is
  `open-cake-ir@<commit>`, an Executor identity `<target>@<commit>`; neither is chosen by
  hand, and a checkout carrying changes has neither. Editing a tracked file is ordinary
  work: commit it.
- **Run the suite in its own worktree, at a commit.** `git worktree add /tmp/<name> <commit>`, then run there.
  Source identity is the clean commit, so a tracked file changing under a running suite takes its identity away
  mid-run and every test that needs a Compiler or an Executor fails from that moment. Measured: a suite in the
  shared checkout was 40 minutes in when another session committed `compiler/targets/gfx1151.json`, and several
  hundred tests turned red at once -- reading exactly like broken code, not like a checkout someone touched. This
  repository is worked on by several sessions at a time, so the shared checkout is the one place a long run cannot
  survive. A worktree costs a `git worktree add` and removes the whole failure mode; it also lets an agent keep
  editing while the run finishes. Report which commit the run was at.
- A host is captured once per exact target as `runtime/hosts/<target>.json` and committed
  (`tools/capture_executor_host.py --target <target>`). A target without one is reported
  as having none (`lab/executor.py:300`); no other host is substituted.
- Never regenerate Corpus Gate expectations to make the gate pass -- that reports a match
  it just manufactured. Adopt new expectations as a separate act
  (`tools/refresh_corpus_expectations.py --write`) whose commit states the reason. More
  generally, a repair run just before the check it satisfies manufactures the state it
  reports: if a check fails because of the environment, say so and stop.
- Common Evaluation begins only after an arm produces a sealed launchable artifact. Every
  candidate, evaluation and terminal outcome is append-only; reports are derived views.
- `archive_integrity` and `filesystem_custody_verified` are orthogonal: a git checkout
  replays correctly and carries permissive modes, and such an archive supports no
  promotion, qualification or estimate. Never restore modes so a custody check passes
  (ADR 0031); report a missing custody environment as a precondition.
- No formal provider or GPU experiment runs before all applicable acceptance gates pass.
  Generated runs and secret bytes stay outside source; new Run specifications, Study plans,
  Campaign Locks, Evidence roots and reports stay outside the checkout; historical in-checkout Campaigns are read-only
  replay inputs. Legacy cleanup needs separate user authorization. Do not copy legacy
  `rXX`, `vN`, failure or archive runners.

## History

Retired release outputs -- `compiler/releases/`, `runtime/executors/`,
`evidence/executors/`, `evidence/calibration/`, `contracts/calibrations/`, the retired
runtime locks, the Study successors above and the retired AKA tools and tests -- live byte-identical
at their original paths under the `history` tag (`git show history:<path>`), as the ADR
that reverses ADR 0065's "History stays where it is" records. `docs/history/identities.json`
maps every retired Compiler Revision and Executor id to its producing commit and history
path, so findings and retained Evidence that cite them stay resolvable. Those identities
stay reserved (ADR 0049, ADR 0050) and nothing on `main` reads them: a historical record
replays at its own commit with the tools of that commit, never against today's tree.
