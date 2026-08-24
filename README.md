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

- Compiler Revision `open-cake-ir-sm100a-v8` is content-bound to an exact `sm_100a` Target and a 16-case,
  32-source Corpus Gate. Its release lock binds the persistent Gate Report and approval. Historical v4
  observations are unchanged; an incident record accounts for their conflicting descriptor digests and the
  unresolvable v6 name, while v7 is archived by exact bytes. Public ranking now obeys the Revision's
  calibration coverage; current coverage is empty rather than falling back to an uncalibrated order.
- The Lab implements two closed Study variants on the same control plane: `matched_search` and `portfolio`. The former
  owns provider Turns, resume, token checkpoints, candidate sealing and search/confirmatory Evaluation; the latter
  reconstructs the final three-shape specialist/dispatcher boundary without a new runtime mode. Within
  `matched_search`, `artifact_optimization_only` restores provider-default features while forbidding scientific
  comparison and promoting Candidates only through common confirmatory Evaluation. The Lab's ordered candidate-set
  envelope is implemented and exercised for both arms. Authoring and GPU-search budgets are independent, and
  the tool-rich prompts give auxiliary agents read-only investigations while the primary thread remains the sole
  submission writer. Both live two-arm provider policies are qualified, and ADR 0009 is accepted after a bounded
  non-scientific B200 Campaign produced three launchable Candidates per arm, searched two per arm and replayed all
  eight search/confirmatory/attribution receipts. This is system qualification, not an arm comparison.
- Evidence v2 uses a no-follow CAS, create-only event files, authority genesis, one terminal schema and read-only replay.
- Current Executor Revision `open-cake-ir-b200-v14` binds the 34-source runtime closure, including deterministic
  external Campaign custody, the exact Nsight Compute executable and replay-checked attribution profiles, plus the
  exact B200 host packages; v1–v5 are archived and v6–v13 are superseded descriptors. Current matched Study fixtures
  opt into a correctness-qualified, no-timing profiler assay; a live Study must re-freeze its exact broker command.
  The current local suite passes 319 tests plus 223 subtests. The earlier remote qualification passed its frozen
  contract suite, host admission and compile-only Triton check without launching a kernel.
- The beginner GPU quickstart passes on an exclusive B200: one candidate kernel launch from the loaded CUBIN,
  16,384 correct assignments, zero fallback calls, synchronized module unload, no performance measurement and no
  scientific claim.
- Historical Executor v10 digest `23a2c79f…` passes the separate correctness-qualified B200 NCU assay for that frozen
  Candidate: one target launch, zero fallback and zero timing, with all 11 profiler metrics replayed from retained
  raw CSV. This validates the attribution mechanism under v10, not current v14, kernel speed or a scientific Campaign.
- The pinned Codex 0.144.3 tool-rich qualification passes with no injected feature disables and observes shell
  activity on both initial and resumed singleton Turns. The Lab derives the required live qualification from Claim Scope, so
  artifact-only composition accepts this tool-rich receipt while scientific/system scopes retain the closed receipt.
  Both closed and tool-rich Codex 0.144.4 candidate-set successors are separately live-qualified for Open Cake and
  direct CUDA. Account/admin policy determines the effective catalog, and external mutation or direct GPU
  measurement remains unauthorized.
- The legacy source authority is the clean final Stage 6 revision `2fa79092...`, tree `b02d730b...`. It was initially
  observed 68 commits ahead; origin now carries the final revision and a verified complete-history bundle provides
  an independent recovery path under `migration/bundles/`.
- Historical r41/r42/r45 outcomes remain bounded negative/inconclusive evidence. G7 r4, the non-scientific G8 r6 and
  the candidate-set v2 system qualification pass on the remote B200 host. The latter has two adhered Runs and eight
  replayed Evaluation Receipts, but produces no treatment comparison: estimand, estimate and uncertainty remain
  null. No scientific Campaign, serving integration or paper-result claim has been run here.

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
