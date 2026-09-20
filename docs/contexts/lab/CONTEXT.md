# Research Lab context

[中文阅读](../../zh-CN/contexts/lab/CONTEXT.md) · [Bilingual catalog](../../README.md)

The Research Lab owns independent Run execution and Study allocation/analysis while treating Workload, Compiler, and
Executor revisions as frozen inputs. Canonical definitions are in the
[`Glossary`](../../GLOSSARY.md#workload-and-research-lab-terms).

## Owned terms

- [`Study Contract`](../../GLOSSARY.md#study-contract)
- [`StudyPlan`](../../GLOSSARY.md#studyplan) and [`RunSpecification`](../../GLOSSARY.md#runspecification)
- [`Claim Scope`](../../GLOSSARY.md#claim-scope)
- [`Provider Feature Policy`](../../GLOSSARY.md#provider-feature-policy)
- [`Authoring Environment`](../../GLOSSARY.md#authoring-environment)
- [`Campaign`](../../GLOSSARY.md#campaign) and
  [`CampaignLock`](../../GLOSSARY.md#campaignlock)
- [`TaskPackage`](../../GLOSSARY.md#taskpackage),
  [`Ralph Controller`](../../GLOSSARY.md#ralph-controller), and
  [`StateCard`](../../GLOSSARY.md#statecard)
- [`Run`](../../GLOSSARY.md#run), [`Turn`](../../GLOSSARY.md#turn),
  [`Checkpoint`](../../GLOSSARY.md#checkpoint), and [`Endpoint`](../../GLOSSARY.md#endpoint)
- [`Candidate`](../../GLOSSARY.md#candidate)
- [`Artifact Promotion`](../../GLOSSARY.md#artifact-promotion)
- [`Estimand`](../../GLOSSARY.md#estimand)
- [`KernelSeed`](../../GLOSSARY.md#kernelseed)

The Lab references but does not redefine the
[`Workload Contract`](../../GLOSSARY.md#workload-contract),
[`Evaluation Receipt`](../../GLOSSARY.md#evaluation-receipt),
[`Portfolio Artifact`](../../GLOSSARY.md#portfolio-artifact), or
[`Measurement Quality`](../../GLOSSARY.md#measurement-quality).

## Responsibilities

- Freeze execution inputs, permissions and budgets in RunSpecification; adapt legacy
  Study/CampaignLock inputs into that same execution path.
- Render one immutable TASK.md/AGENTS.md TaskPackage per successor Run and let the
  external Ralph Controller supply only evidence-derived iteration state.
- Assign every Run to one complete Authoring Environment.
- Keep one primary author as the sole Candidate submission writer.
- Bound Turns, authoring budget, GPU-search budget, checkpoints, and stopping rules.
- Send sealed Candidates through the common Evaluation boundary.
- Apply only the Analysis Plan and Claim Scope frozen before execution.

The Lab does not redefine Workload semantics, decide Candidate correctness, write an
Evaluator's Receipt, or turn system qualification into a scientific comparison.

## Relationships

- An engineering Run needs no Study. StudyPlan preassigns conditions and repetitions to
  the same Run engine, with one Authoring Environment per Run and frozen analysis.
- The legacy Campaign executes one Study Contract; pooling remains subject to its
  Analysis Plan. The adapter retains that policy without owning another execution loop.
- A Run contains ordered Turns and may submit multiple immutable Candidates.
- One Candidate may receive distinct search, confirmatory, and profiler Evaluation
  Receipts.
- A scientific Study declares an Estimand before execution. A system-qualification or
  artifact-optimization Claim Scope structurally has no treatment estimate.
- KernelSeed and standalone portfolio artifact utilities remain available; there is no
  Portfolio Study lifecycle. A seed does not establish arbitrary-shape or serving generalization.

## Boundary examples

A Run that adheres to the protocol but finds no qualified Candidate is an observed
negative outcome. A provider, broker, custody, or harness fault is missing data. Neither
may be silently converted into the other.

Artifact Promotion chooses one confirmatory-qualified Candidate inside one Run. It is not
an arm winner, a causal effect, or a production deployment decision.

Concrete task code and the TaskLab composition root live outside the common engine; see [task ownership](../../en/TASKS.md).

## Implementation owners

`lab/core.py` is the public facade. It owns only the existing project root, clock,
workload loader, schedule preparation, authoring validation and manifest parser.
Phase functions receive those dependencies explicitly; there is no second context object.

Run parsing occurs once in preflight. Task-specific admission checks the frozen inputs
before runtime side effects; legacy Study policy is checked at its input boundary.
See the [implementation map](../../../src/open_cake_ir/lab/README.md)
for module ownership and the independent replay package. Historical campaigns replay at their
pinned source commit; a refactor never rewrites their records.

## Native pairing implementation

Each NativeBackend row binds a NativeAdapter: its factories, source admission, launch
block and baseline projection. Pairing delegates through that interface without selecting
a default implementation. Dependencies are resolved at call time. See
[ADR 0072](../../adr/0072-backend-owned-input-and-native-adapters.md).
