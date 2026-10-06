# Shared research agenda and hardware experiments

[中文（canonical policy）](../RESEARCH_AGENDA.md) · [Repository](../../README.md) · [Technical report](../README.md)

**open-cake-ir studies how optimization discoveries become reusable, conditional transformations that reduce search and adaptation cost on subsequent tasks and hardware.** The Compiler remains the product core: it owns Program, Schedule, legality analyses, deterministic rewrites and lowering. Lab owns authors, materials, budgets and research assignments; independent Evaluation and Evidence own device observations and their interpretation.

The Chinese agenda is the canonical policy; this page is its English companion. The [roadmap](../ROADMAP.md) orders the work, [transfer design](OPTIMIZATION_TRANSFER.md) explains mechanisms, and the [E/P protocol](../OPTIMIZATION_TRANSFER_ABLATION.md) owns the detailed treatment and statistical rules. Platform results keep their own source, Workload and measurement scope.

## Research question

Under matched task semantics, reference access and declared budgets, which decisions benefit from free agent exploration, which should become conditional executable transformations, and does this division reduce later search while retaining competitive implementations?

Three hypotheses guide the work:

- **Representation and tools:** compare Cake with native DSL authoring using equivalent semantic tools, shared information and common evaluation. Report confirmed performance, cost, coverage and the gap to strong implementations.
- **Source experience and executable reuse:** distinguish generic knowledge from source-derived conditions; use E/P treatments to separate additional materials and transformation access. Report negative transfer and discovery/adaptation cost.
- **System evolution:** compare old/new analyses on a fixed candidate collection, then run fresh searches on successor code. Separate coverage, modeled-domain errors and downstream search effects.

These are hypotheses. Platform bring-up, a local speedup and a working pass are scoped evidence, not substitutes for controlled comparisons.

## Reusable knowledge and guarantees

A mechanism retains before/after programs, intent, semantic preconditions, target realization conditions, tunable target decisions, source evidence and counterexamples. Existing Findings, Compiler passes, Lab materials and result records remain their owners.

Semantic legality, target realizability and expected benefit are distinct. Static checks block only within their modeled domains; unsupported, unmodeled, unqualified and demonstrably invalid cases have different meanings. Cost estimates rank candidates; target measurements decide performance. Two individually valid programs are not thereby equivalent. State whether a guarantee comes from trusted construction, a proof in a specified model or finite device tests.

Prioritize tiling, reduction/data ownership, storage reuse and fusion with a qualified complete assay. Simple width changes are useful controls. Coordinated edits test whether executable transformations reduce implementation errors. Keep concrete storage/access commitments; do not introduce a layout algebra.

## Distinct comparisons

| Comparison | Interpretation |
|---|---|
| Cake vs native DSL plus equivalent tools on the same downstream stack | Representation/diagnostic package effect; if both use Cake internals, identify an author-interface comparison |
| Native DSL vs a qualified low-level route | Lowering and instruction-selection freedom; freeze common launch/ABI/constants constraints and separately report full capabilities |
| Non-LLM vs agent selection over the same action set | Selection-policy effect; unrestricted code generation is a separate system comparison |
| E/P within a shared Compiler and target | Additional materials and tool access; P's minimal interface already conveys knowledge |

Qualify authoring, compilation, loading and measurement before adding a native route. `native_cuda` is a Schedule emission backend, not proof that arbitrary PTX author submissions are supported. NVIDIA results do not qualify other vendors' routes. Match reference inputs: a complete optimized source is different from a mathematical specification. Shared-capability experiments and full-capability comparisons answer different questions. Nonsignificance alone does not establish equivalence.

## Native discovery and promotion

IR is the default development interface. A concrete expressibility or lowering hypothesis can motivate a separate, qualified native engineering Run. Retain the Workload, exact target, oracle and measurement requirements; declare inherited material, reference access and budget.

Promote choices to Lab recipes, reusable rewrites to Compiler passes, realization gaps to lowering, missing modeled legality to the Verifier, and genuine expressibility gaps to primitives together with types, effects and analyses. Review and verification precede a successor Run. No promotion is a valid disposition.

Do not switch routes or inject new knowledge within frozen Runs. A new Run does not erase prior reference access. Native results and re-expressed results require separate acceptance. An opaque low-level node does not inherit IR guarantees. Record manual discovery, implementation and review separately from autonomous methods, including their costs.

## Hardware branches share the question

Every platform can supply or receive experience. Choose reliably evaluable tasks; do not require every branch to duplicate the whole suite.

| Branch | Priority question |
|---|---|
| `nvidia` | Structural discoveries, strong native controls, explicit storage/synchronization and verification coverage |
| `metax` | Joint tiling under compiled-resource and driver constraints; reliable timing and localized refusals |
| `dcu` | Target retuning, numerical behavior and register/scratch observations; source-specific information and negative transfer |
| `amd` | Exact-target execution width, resource and toolchain differences; independently qualified target mappings |
| `metal` | Threadgroup ownership, storage and representation/measurement coverage |
| `main` | Shared semantics, methods and evidence integration; platform qualification stays with its records |

A new experiment states its hypothesis, mechanism/task/target, controls, reference permissions, budget, measurement contract and stopping/confirmation rule in the existing plan or Run/Study inputs. This introduces no new Run schema, Study kind or execution engine.

**New optimization Runs default to a three-hour total budget, including final confirmation.** Explicitly bind `wall_time_seconds=10800` and declare the confirmation reserve and compile/evaluation limits; do not rely on a launcher's historical defaults. Qualification probes have their own bounds. Budget changes must be declared consistently before a study; frozen Runs keep their original rules. Report wall time, tokens, CPU preparation, compilation, queueing and device use separately.

Separate source discovery, target adaptation and frozen target tests. Tasks used to select mechanisms or parameters are development evidence. Unseen shapes, new operator families and new targets are distinct generalization settings. Repeated Runs and timing samples do not create more independent tasks. Block treatments by device and time, and preserve assigned failures and missing endpoints.

## Related work and contribution boundary

The [canonical agenda](../RESEARCH_AGENDA.md#与相关路线的关系) maps CAKE/TIRx, Exo/MLIR Transform, AutoTVM/MetaSchedule, AKG, KernelBlaster/AccelOpt and AI lowering to the questions above. Reusing these foundations is explicit. A collection of backends, agents and memories is not itself a novelty claim. Each contribution needs a concrete mechanism, controlled evidence, maintenance costs and a stated domain.
