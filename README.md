# open-cake-ir

`open-cake-ir` is a compiler-first, independent reconstruction of the ideas described in
[CAKE: Compiler-Agent Co-Design for Frontier Kernel Evolution](https://arxiv.org/abs/2608.12629v1).
It does not contain or claim to reproduce the unpublished CAKE implementation.

## Product boundary

The repository has one product core and one dependent research application:

1. **Open Cake Compiler** — typed hardware-explicit schedules, exact target definitions, localized findings,
   analysis, deterministic lowering, compiler revisions and corpus gates. It is usable without an agent campaign.
2. **Research Lab** — freezes an Open Cake Compiler revision and evaluates complete authoring environments against
   common external correctness and performance protocols.
3. **Evaluation and Evidence** — share Workload-owned assays across treatments, retain typed receipts, store immutable
   observations, and replay Run integrity independently of the execution process.

The dependency direction is permanent:

```text
Research Lab -> Open Cake Compiler
Research Lab -> Evaluation -> Evidence
Open Cake Compiler -X-> Lab / Provider / Workload / Evidence
```

## Status

- Compiler Revision `open-cake-ir-sm100a-v3` is content-bound to an exact `sm_100a` Target and a six-case r16/r25/r31
  Corpus Gate. Its release lock binds the persistent Gate Report and approval; its complete source archive is
  externally anchored by the Terminal Archive seal.
- The Lab implements two closed Study variants on the same control plane: `matched_search` and `portfolio`. The former
  owns provider Turns, resume, token checkpoints, candidate sealing and search/confirmatory Evaluation; the latter
  reconstructs the final three-shape specialist/dispatcher boundary without a new runtime mode.
- Evidence v2 uses a no-follow CAS, create-only event files, authority genesis, one terminal schema and read-only replay.
- Current Executor Revision `open-cake-ir-b200-v2` binds the 30-source runtime closure and exact B200 host packages.
  The G8 r6 path remains bound to its complete archived v1 source closure. Local and
  remote contract suites pass 103 tests plus 13 subtests; remote host admission and compile-only Triton qualification
  pass without launching a kernel.
- The legacy source authority is the clean final Stage 6 revision `2fa79092...`, tree `b02d730b...`. It was initially
  observed 68 commits ahead; origin now carries the final revision and a verified complete-history bundle provides
  an independent recovery path under `migration/bundles/`.
- Historical r41/r42/r45 outcomes remain bounded negative/inconclusive evidence. G7 r4 and the non-scientific G8 r6
  system qualification pass on the remote B200 host. G8 produces no treatment comparison: estimand, estimate and
  uncertainty remain null. No scientific Campaign, serving integration or paper-result claim has been run here.

## Read order

1. [`CONTEXT-MAP.md`](CONTEXT-MAP.md) — compiler, lab and evidence domain language.
2. [`docs/PAPER_CONTRACT.md`](docs/PAPER_CONTRACT.md) — paper facts, public artifacts and unknowns.
3. [`docs/TOP_LEVEL_DESIGN.md`](docs/TOP_LEVEL_DESIGN.md) — system structure, owners and two feedback loops.
4. [`docs/MIGRATION_PLAN.md`](docs/MIGRATION_PLAN.md) — compiler-first one-way migration.
5. [`docs/ACCEPTANCE_GATES.md`](docs/ACCEPTANCE_GATES.md) — gates before code, pilot and cutover.
6. [`docs/adr/0001-compiler-first-with-dependent-lab.md`](docs/adr/0001-compiler-first-with-dependent-lab.md) —
   accepted product boundary.
7. [`migration/CAPABILITY_MATRIX.md`](migration/CAPABILITY_MATRIX.md) — final r16-r45 feature/evidence-to-owner map.
8. [`docs/RUNBOOK.md`](docs/RUNBOOK.md) — the canonical matched and Portfolio execution paths.
9. [`docs/adr/0003-rust-shadow-engine-after-v3.md`](docs/adr/0003-rust-shadow-engine-after-v3.md) — proposed
   language evolution after the Python v3 contract is frozen.

## Migration rule

Migrate canonical facts, stable capabilities and immutable evidence—not the historical `rXX`/`vN` implementation
topology. Active code has one current path; the verified legacy bundle exists only for historical recovery. A live
provider/GPU system qualification has passed; sole-owner cutover still requires the explicit G9 approval and Git
anchor in `docs/ACCEPTANCE_GATES.md`.
