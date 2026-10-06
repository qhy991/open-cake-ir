# Shared research agenda and hardware experiments

[中文（canonical policy）](../RESEARCH_AGENDA.md) · [Repository](../../README.md) · [Technical report](../README.md)

**open-cake-ir studies how optimization discoveries become reusable, conditional transformations that reduce search and adaptation cost on subsequent tasks and hardware.** The Compiler remains the product core: it owns Program, Schedule, legality analyses, deterministic rewrites and lowering. Lab owns authors, materials, budgets and research assignments; independent Evaluation and Evidence own device observations and their interpretation.

The Chinese agenda is the canonical policy; this page is its English companion. The [roadmap](../ROADMAP.md) orders the work, [transfer design](OPTIMIZATION_TRANSFER.md) explains mechanisms, and the [E/P protocol](../OPTIMIZATION_TRANSFER_ABLATION.md) owns the detailed treatment and statistical rules. Platform results keep their own source, Workload and measurement scope.

## Research question

Under matched task semantics, reference access and declared budgets, which decisions benefit from free agent exploration, which should become conditional executable transformations, and does this division reduce later search while retaining competitive implementations?

Four hypotheses guide the work:

- **H1 Representation and diagnostics:** compare Cake with native DSL authoring under matched information, semantic tools and evaluation. Cost savings must retain quality and coverage; persistent native advantages narrow the useful domain of structured authoring.
- **H2 Source experience and executable reuse:** compare generic and source-derived materials, then separate materials from execution access through E/P. Target-answer leakage invalidates the comparison.
- **H3 Route selection (proposed):** compare a development-frozen policy with the best fixed route selected on development tasks and equal allocation, under one total budget. Post-hoc per-task selection is only a descriptive upper bound.
- **H4 System evolution:** replay fixed candidates on old/new versions, then run fresh searches on held-out tasks. Repairing only the original failing cases does not establish downstream search gains.

These are hypotheses. Nonsignificance alone establishes neither equivalence nor falsification. Begin with bounded H1/H2 pilots; use their signal and qualified capabilities to scope H3/H4.

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

## Route selection across frozen Runs (H3, proposed)

Lab may study allocation across separately frozen Runs. Each Run retains its author environment, source and route. Freeze the policy, triggers and aggregate budget on development tasks; the default multi-route optimization task still receives three hours in total. Child Runs share this allowance, including handoff, failed compilation and confirmation costs. Retain inherited reference access. This proposal adds no automatic fallback, new Study kind or execution engine.

Workload Contract owns semantics and the oracle; Study owns treatments and estimands; RunSpecification owns execution inputs, permissions and budgets; Target/Executor own target and execution facts. A conceptual task tuple does not replace these authorities. Qualify each native author route on its exact target.

## Quality endpoints and lifetime cost

The owner-approved [ten-sample mean design](../MEAN10_TIMING.md) specifies ten timed samples per implementation, arithmetic-mean selection and diagnostic-only dispersion for successor development Runs. Raw capture validity remains mandatory. Activation requires successor qualification; frozen results retain their original policy.

Freeze the starting baseline separately from a strong reference. Baseline speedup measures improvement; a predeclared quality endpoint such as `L_confirm ≤ (1 + δ) × L_ref`, after common acceptance gates, measures competitiveness. Declare the reference, δ and primary endpoint before execution. A suggested 5% tolerance is not a universal rule. Existing E/P endpoints retain their original definition.

First-hit costs require scheduled independent confirmation checkpoints; preserve right-censoring for unmet endpoints. Timing samples estimate measurement noise, repeated Runs estimate author variation, and cross-task inference clusters by task. Small pilots report per-task results; choose formal scale from independent pilot variability and the effect of interest.

Report both marginal reuse value and cumulative value after discovery, extraction, implementation, target adaptation and maintenance. Keep human time, GPU time, tokens and money separate unless conversion assumptions are stated. Include low-headroom, unsupported and negative-transfer cases alongside promising structural tasks. Distinguish semantic violations, unmodeled behavior, expressibility/lowering/toolchain gaps, numerical failures, invalid timing and valid no-gain results. Refusal audits use qualified, bounded diagnostic environments; stronger optional proof coverage is reported separately from common acceptance.

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
