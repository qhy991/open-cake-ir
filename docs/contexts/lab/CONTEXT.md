# Research Lab context

The Research Lab owns Study design and execution while treating Workload, Compiler, and
Executor revisions as frozen inputs. Canonical definitions are in the
[`Glossary`](../../GLOSSARY.md#workload-and-research-lab-terms).

## Owned terms

- [`Study Contract`](../../GLOSSARY.md#study-contract)
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

- Resolve a Study template and released inputs exactly once into a CampaignLock.
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

- A Study Contract references one Workload Contract and assigns every Run to one Authoring
  Environment.
- A Campaign executes one Study Contract; pooling across Campaigns is forbidden unless the
  Study's Analysis Plan declares it.
- A Run contains ordered Turns and may submit multiple immutable Candidates.
- One Candidate may receive distinct search, confirmatory, and profiler Evaluation
  Receipts.
- A scientific Study declares an Estimand before execution. A system-qualification or
  artifact-optimization Claim Scope structurally has no treatment estimate.
- A KernelSeed may enter a separate portfolio Study; it does not imply arbitrary-shape or
  serving generalization.

## Boundary examples

A Run that adheres to the protocol but finds no qualified Candidate is an observed
negative outcome. A provider, broker, custody, or harness fault is missing data. Neither
may be silently converted into the other.

Artifact Promotion chooses one confirmatory-qualified Candidate inside one Run. It is not
an arm winner, a causal effect, or a production deployment decision.
