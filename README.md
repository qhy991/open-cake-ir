# open-cake-ir

`open-cake-ir` is a compiler-first, independent reconstruction of the ideas described in
[CAKE: Compiler-Agent Co-Design for Frontier Kernel Evolution](https://arxiv.org/abs/2608.12629v1).
It does not contain or claim to reproduce the unpublished CAKE implementation.

**想先看系统全貌？** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) 用六张图说明 Context 依赖方向、
Compiler 管线、Schedule 的硬件承诺、一次 Study Run 的状态机、Compiler Revision 的发布生命周期，
以及本仓库与论文架构的逐项差距。

**第一次接触编译器或 GPU kernel？请从
[`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) 开始。** 它用一个已经在 B200 上验证的真实例子解释
Schedule、Assessment、Lowering、CUBIN 和正确性检查，并提供可直接复制的命令。

In plain language: you describe *how* a GPU should divide and order work in a typed JSON Schedule; the Compiler
checks that plan and lowers it to inspectable target source. The optional Research Lab then lets agents improve
Candidates under frozen evaluation and evidence rules.

## Product boundary

The repository has one product core and one dependent research application, supported by two shared Modules:

1. **Open Cake Compiler** — typed hardware-explicit schedules, exact target definitions, localized findings,
   analysis, deterministic lowering, compiler revisions and corpus gates. It is usable without an agent campaign.
2. **Research Lab** — freezes an Open Cake Compiler revision and evaluates complete authoring environments against
   common external correctness and performance protocols.
- **Evaluation and Evidence** — shared Modules that apply Workload-owned correctness/measurement rules, retain typed
  receipts, store immutable observations, and replay Run integrity.

The dependency direction is permanent:

```text
Research Lab -> Open Cake Compiler
Research Lab -> Evaluation -> Evidence
Open Cake Compiler -X-> Lab / Provider / Workload / Evidence
```

## Status

- Compiler Revision `open-cake-ir-sm100a-v14` is content-bound to an exact `sm_100a` Target and a 21-case,
  37-source Corpus Gate. Its release lock binds the persistent Gate Report and approval. Historical v4
  observations are unchanged; an incident record accounts for their conflicting descriptor digests and the
  unresolvable v6 name, while v7 and v8 are archived by exact bytes. v9 closes the verifier/emitter gap that let an
  emitted MMA omit its instruction contract while still being marked lowering-eligible. v10 turns the pinned Triton
  backend's unsupported warp-specialized value-and-index argmin into structured lowering feedback. v11 separates the
  performance-semantic cost preorder from deterministic `schedule_id` serialization, so a tied cut is an abstention
  rather than an invented performance choice. v12 makes the existing lowering boundary public: `generated=true`
  means a backend emitted operation bodies from the Schedule, while TinyGEMM2's closed asset reports `false` instead
  of being indistinguishable from generation. v13 repairs the agent authoring schema against every Corpus Schedule
  and adds one KDA-derived `tanh` primitive with a standalone SwiGLU positive/shape-drift pair. v14 adds one
  deterministic indexed `top_k` primitive after KDA v1 exposed the paper-claimed vocabulary gap; its positive and
  shape-drift pair is gated, and a brokered B200 observation matches 64 ordered indices exactly under forced ties
  and negative infinities. The reviewed KDA baseline still has 0/57 complete-version coverage: grouping/masking,
  quantization-scale relations, grouped/ragged work, scatter and the program DAG remain absent. These slices prove
  primitive expressibility, not MoE or performance reproduction. Public ranking obeys its calibration
  coverage; current coverage remains empty rather than falling back to an uncalibrated order. A
  preregistered two-repeat B200 successor tested the Lab's actual three-to-two pruning decision across all 2,300
  eligible GEMM triplets and failed the fixed 5% boundary at 35.95% and 8.16%. A drift-controlled successor then
  classified every triplet as decisive or tied: 1,450 decisive cuts and 850 abstentions per repeat. One repeat still
  failed the unchanged boundary at 5.73% regret (the other reached 2.79%), so no coverage was promoted.
- The Lab implements two closed Study variants on the same control plane: `matched_search` and `portfolio`. The former
  owns provider Turns, resume, token checkpoints, candidate sealing and search/confirmatory Evaluation; the latter
  reconstructs the final three-shape specialist/dispatcher boundary without a new runtime mode. Within
  `matched_search`, `artifact_optimization_only` restores provider-default features while forbidding scientific
  comparison and promoting Candidates only through common confirmatory Evaluation. The Lab's ordered candidate-set
  envelope is implemented and exercised for both arms. Authoring and GPU-search budgets are independent, and
  the tool-rich prompts give auxiliary agents read-only investigations while the primary thread remains the sole
  submission writer; the sealed envelope is the handoff and the external Lab remains the sole Judge. Both live
  two-arm provider policies are qualified, and ADR 0009 is accepted after a bounded
  non-scientific B200 Campaign produced three launchable Candidates per arm, searched two per arm and replayed all
  eight selected-only search/confirmatory/attribution receipts. ADR 0012's Executor-v18 successor replays ten
  Receipts and profiles both correct search survivors in each arm, including the non-selected one. These are system
  qualifications, not arm comparisons.
- Evidence v2 uses a no-follow CAS, create-only event files, authority genesis, one terminal schema and read-only replay.
- Current Executor Revision `open-cake-ir-b200-v26` binds the 34-source runtime closure, including deterministic
  external Campaign custody, the exact Nsight Compute executable and replay-checked attribution profiles, plus the
  exact B200 host packages; v1–v5 are archived and v6–v25 are superseded descriptors. v19 canonically treats one
  observed same-path delete/add replacement as a resumed candidate update. v20 applies
  a cost order only when every launchable member is scored and emits a cost-model misranking diagnosis only when
  that order was actually applied; otherwise the whole set keeps provider order without inventing a ranking. v21
  also treats a resumed bare `add` label as the Lab-authoritative update after the pre-Turn existence check. v22
  gives successor scientific Studies a coherent two-part Estimand: adhered candidate failure is an observed
  qualification outcome, external faults remain missing, and the estimate includes the declared qualification-rate
  contrast. v23 gives every successor matched Study one closed `matched_run_v1` event vocabulary: replay validates
  the Run boundaries and payload/object coverage, and derives search selection and routed diagnoses from retained
  filter rows and Evaluation Receipts instead of trusting them as parallel truths. v24 removes the Lab's implicit
  `max` reasoning choice: qualification and live freezing must bind one explicit effort, identical across arms, and
  changing it without a matching qualification fails preflight. v25 retains the exact rendered reference bundle
  supplied on every provider Turn as a replay-checked Evidence object; absence becomes a harness fault and missing
  endpoint rather than an unverifiable clean-start pass. v26 projects the Compiler-owned lowering-generation fact
  through the public CLI without changing execution semantics. The current matched Study fixtures
  retain a correctness-qualified, no-timing profile for every searched
  survivor; only the
  selected survivor's projection becomes feedback. A frozen live successor now passes that exact protocol on B200
  after re-freezing its broker command.
  The current local suite passes 348 tests, including its parameterized contract cases. The earlier remote qualification passed its frozen
  contract suite, host admission and compile-only Triton check without launching a kernel.
- The beginner GPU quickstart passes on an exclusive B200: one candidate kernel launch from the loaded CUBIN,
  16,384 correct assignments, zero fallback calls, synchronized module unload, no performance measurement and no
  scientific claim.
- Historical Executor v10 digest `23a2c79f…` passes the separate correctness-qualified B200 NCU assay for that frozen
  Candidate: one target launch, zero fallback and zero timing, with all 11 profiler metrics replayed from retained
  raw CSV. This validates that historical mechanism; frozen v19 all-survivor coverage is independently qualified by
  the bounded successor above. Neither observation measures profiler speed or supports a scientific claim.
- The pinned Codex 0.144.3 tool-rich qualification passes with no injected feature disables and observes shell
  activity on both initial and resumed singleton Turns. The Lab derives the required live qualification from Claim Scope, so
  artifact-only composition accepts this tool-rich receipt while scientific/system scopes retain the closed receipt.
  Both closed and tool-rich Codex 0.144.4 candidate-set successors are separately live-qualified for Open Cake and
  direct CUDA. A second closed qualification now binds the paper-reported `xhigh` reasoning value to the same pinned
  executable and three-member envelope: both arms passed add/resume, usage and reference-visibility checks under
  external sealed Evidence. This proves provider transport at `xhigh`; no scientific Study has yet frozen the 80M
  endpoint, and no paper result is implied. Account/admin policy determines the effective catalog, while external
  mutation and direct GPU measurement remain unauthorized.
- The legacy source authority is the clean final Stage 6 revision `2fa79092...`, tree `b02d730b...`. It was initially
  observed 68 commits ahead; origin now carries the final revision and a verified complete-history bundle provides
  an independent recovery path under `migration/bundles/`.
- Historical r41/r42/r45 outcomes remain bounded negative/inconclusive evidence. G7 r4, the non-scientific G8 r6 and
  candidate-set system qualifications v2/v3 pass on the remote B200 host. The six-Run scientific matched-search v3
  Campaign also completes with archive integrity and semantic replay: direct CUDA qualifies in 2/3 Runs at 9.880473
  and 10.084415 ms, while Open Cake qualifies in 0/3 at the turn-discrete 150k checkpoint. One provider fault and one
  harness fault leave the preregistered Estimand unavailable, so no causal estimate is reported. Executor v20/v21 and
  Compiler v10 are create-only successors for the three runtime/compiler defects exposed by that immutable Evidence;
  Executor v22 corrects only the Analysis Plan of future Studies; Executor v23 closes semantic events only for new
  matched Studies; Executor v24 makes reasoning effort an explicit qualified treatment factor. This local
  150k, task-informed package study also freezes the provider's distinct `max` reasoning level and complete
  implementation skeletons. The later `xhigh` capability qualification does not rewrite that frozen treatment or
  supply an 80M clean-start Study. Serving and paper reproduction remain unsupported.
- `matched-search-clean-start-reference-v29.json` is a zero-GPU successor fixture that replaces both arm references
  together: Open Cake receives only its authoring interface and direct CUDA receives a canonical ABI with an empty
  kernel body. Its contamination gate validates reference access only; its inherited 150k/`max` treatment still
  cannot support a paper comparison.
- Four bounded tool-rich artifact-optimization Campaigns are retained. v1 promoted one Open Cake artifact and
  exposed v14's scratch-file event classification defect; v2 promoted one Direct CUDA artifact and exposed v15's
  remaining assumption when Open Cake added and updated the fixed envelope in one Turn. v16 uses the smaller authority
  model already declared by ADR 0004: every tool-rich file-change event is typed auxiliary activity, while only the
  final no-follow envelope is the submission authority. The independent v3 successor passes end to end: both Runs
  adhere and fresh audit replays all evidence, but its 150k boundary prevents a resumed Turn. v4 adds only an explicit
  8M terminal checkpoint and two-Turn hard bound. Both Runs then receive measured feedback: Open Cake promotes its
  Turn-2 artifact at 1.417239 ms after 1.851297 ms in Turn 1, while direct CUDA retains its 4.156109 ms Turn-1 artifact
  after Turn 2 reaches 8.814379 ms. All four v4 provider Turns emit command/file activity but no auxiliary-agent
  lifecycle, so v4 validates feedback and single-writer/judge separation—not multi-agent use. It remains
  non-scientific artifact optimization; no cross-arm estimate or paper claim is available.

## Read order by role

### New user

1. [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) — real B200 teaching smoke and basic vocabulary.
2. [`docs/contexts/compiler/CONTEXT.md`](docs/contexts/compiler/CONTEXT.md) — formal Compiler terms after the tutorial.
3. [`compiler/AUTHORING_CONTRACT.md`](compiler/AUTHORING_CONTRACT.md) — rules for writing complete Schedules.

### Researcher or experiment operator

1. [`docs/PAPER_CONTRACT.md`](docs/PAPER_CONTRACT.md) — paper facts, public artifacts and unknowns.
2. [`docs/TOP_LEVEL_DESIGN.md`](docs/TOP_LEVEL_DESIGN.md) — Modules, owners and two evolution loops.
3. [`docs/RUNBOOK.md`](docs/RUNBOOK.md) — provider, GPUQ, artifact optimization and Portfolio operations.

### Maintainer or auditor

1. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — the six diagrams, including the gap against the paper.
2. [`CONTEXT-MAP.md`](CONTEXT-MAP.md) — canonical domain language.
3. [`docs/ACCEPTANCE_GATES.md`](docs/ACCEPTANCE_GATES.md) — verified gates and claim boundaries.
4. [`docs/MIGRATION_PLAN.md`](docs/MIGRATION_PLAN.md) and
   [`migration/CAPABILITY_MATRIX.md`](migration/CAPABILITY_MATRIX.md) — migration provenance.
5. [`docs/adr/`](docs/adr/) — durable architecture decisions, including Rust and tool-rich optimization.

## Migration rule

Migrate canonical facts, stable capabilities and immutable evidence—not the historical `rXX`/`vN` implementation
topology. Active code has one current path; the verified legacy bundle exists only for historical recovery. A live
provider/GPU system qualification has passed. The private GitHub origin is the sole source revision authority;
legacy `cake-repro` remains read-only historical evidence under the G9 rollback contract.
