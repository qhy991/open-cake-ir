# Glossary: code names in plain English

[中文定义原文](../GLOSSARY.md) · [English home](README.md)

This is the English translation of the canonical Chinese glossary. The source page owns prose definitions; machine spellings remain owned by code and contracts. Use it as a lookup, not a list to memorize.

## Basic reading vocabulary

These explanations do not promise that every use of a concept is supported. Exact constraints belong to the corresponding contract.

| Term | Plain meaning | Example or distinction |
| --- | --- | --- |
| GPU / CPU | GPU handles many similar tasks in parallel; CPU handles general computation and control | CPU prepares input; GPU processes many rows |
| Operator / kernel | An operator is a mathematical task; a kernel is a GPU program executed by a launch | One attention operator may launch several kernels |
| Tensor / shape | An array organized by dimensions, and the size of each dimension | [2,3] means two rows of three values |
| Dtype / FP32 / BF16 | Number storage types, using four and two bytes respectively with different precision | Fewer bits can lose precision |
| Buffer | Named Schedule data with shape, type, and storage placement | x is input; x_tile is a working tile |
| Tile | A small piece processed together | 128 values in chunks of 32 form four tiles |
| Thread / warp / CTA | A thread is a worker; an NVIDIA warp has 32 threads; a CTA is a cooperating thread block | A CTA can contain several warps |
| Global / register / shared / tensor memory | Distinct GPU storage spaces with different capacity and cooperation rules | Renaming a space does not make a transfer legal |
| ABI | The caller/callee agreement on argument order, types, and results | Passing a value differs from passing its address |
| Runtime scalar / constant | A value supplied at invocation versus one fixed in the authored plan | A varying learning rate cannot silently become 0.1 |
| Broadcast / splat | Reuse aligned smaller-shape values, or explicitly replicate one value | Turning 2 into [2,2,2] needs defined semantics |
| Mask / predicate | Which positions participate, and a condition deciding true or false | A partial final tile must not read past input |
| State / in-place update | Caller-owned memory modified for later use | Update a counter rather than return a substitute |
| Rounding / FMA | Map an exact value to representable precision; FMA rounds a product plus addend once | Separate multiply/add can round twice |
| NaN / Inf / subnormal | Special floating-point categories: not-a-number, infinity, very small values | Correctness contracts specify their behavior |
| Oracle / tolerance | Independent expected answer and allowed deviation | Do not widen a threshold after observing a failure |
| Static / dynamic | Reasoning from a plan versus observing execution | Source eligibility differs from GPU correctness |
| Benchmark / profiler | Measure duration versus inspect where work/resources go | Profiled duration is not ordinary latency |
| Upper/lower bound / estimate | At most, at least, or a value without a guaranteed direction | At least ten is not exactly ten; unknown is not zero |
| Median / CV | Middle sorted sample, and variability relative to the mean | Check stability rather than choose a best sample |
| Freeze / successor | Keep one execution's content fixed; create a new version | New results do not overwrite old evidence |

## Project names

### CAKE

The system and research described by the paper. This project uses the public description and does not present itself as the unpublished original source.

### open-cake-ir

This repository: an independent Compiler implementation and a research system that depends on it.

### Cake IR

The language used here to describe GPU execution plans. Exact fields and semantics belong to the code, schema, Authoring Contract, and released revision.

### Open Cake Compiler

An independently usable compiler that assesses a Schedule, reports analysis, and generates eligible target source. It does not call AI, allocate GPUs, or declare experimental wins.

## Compiler terms

### Schedule

A complete execution plan specifying Buffers, worker roles, dependencies, loops, coordinates, storage, and lowering. Compiler typing and semantics own its meaning; it does not replace Workload mathematics.

### Target

An exact hardware target and declared capabilities bound by a Compiler Revision. It is not an arbitrary available GPU or permission to fall back to another architecture.

### Finding

A localized diagnostic with a code, position, explanation, and blocking effect, owned by its checking rule. It may be informational rather than an error.

### Assessment

The result of checking one Schedule under an exact target and revision. Structural acceptance and backend eligibility are separate, and neither is GPU correctness or Lab admission.

### Lowering

Deterministically generated target source and mapping information, or selection of an admitted fixed source asset. Owned by the Compiler Revision; it has not yet been toolchain-compiled or executed.

### Compiler Revision

A frozen compiler binding semantics, targets, checking, analysis, generation, Corpus expectations, and calibration coverage. Its release lock owns identity; a moving Git branch does not.

### Executor Revision

A released descriptor binding Lab, evaluation, evidence tooling, and Python/profiling environment. Source and actual host are checked separately. It differs from Compiler identity and does not imply every Workload has a live evaluator.

### Corpus

The compiler positive/negative case set with reviewed expected behavior, owned by the release-bound manifest. Loose examples or unit tests do not replace it.

### Corpus Gate

The complete comparison of Corpus observations with reviewed expectations, owned by its report and release approval. It does not establish GPU correctness or speed.

### Calibration

Measurement-based support for an estimate in a specified domain. Bound calibration contracts and accepted records own it; it does not transfer automatically across target, shape, or revision.

## Workload and Research Lab terms

### Workload Contract

The problem definition: mathematics, input domain and generation, external reference, tolerances, test rows, and outputs. It owns these facts rather than merely listing benchmark arguments.

### Study Contract

The experimental design: authoring environments, allocation, budget, references, and analysis. It is not mutable configuration for one execution.

### Claim Scope

The kind of conclusion and use allowed by the Study, distinguishing system qualification, artifact optimization, and scientific analysis. README prose cannot widen it.

### Provider Feature Policy

The frozen Authoring Environment policy for available AI features and event interpretation. It neither determines a conclusion nor authorizes external actions.

### Authoring Environment

The complete tools, references, feedback, and submission interface supplied to an author. Its frozen Study reference owns it; a comparison cannot be named only by source syntax.

### Campaign

One actual execution of a Study, with versions, host requirements, and evidence location fixed by CampaignLock. It is distinct from the design and from an individual Run.

### CampaignLock

The preflight-produced exact execution binding for Workload, Compiler, Executor, provider, toolchain, host, and custody. Later main changes do not update it.

### TaskPackage

The TASK.md and AGENTS.md bundle rendered deterministically by Lab from CampaignLock. It states the problem and rules, not mutable scores, consumed budget, or a current best candidate.

### Ralph Controller

The external Lab iteration controller that decides whether another Turn may start, supplies StateCard, accounts for budgets, and records stopping. It neither authors candidates nor judges correctness.

### StateCard

A next-Turn view derived from retained Run facts: spent and remaining budget and known feedback. It is not an agent-editable policy.

### Run

One independently allocated repetition and the comparison unit in a Study. It may contain several Turns; it is not a terminal process or broker job.

### Turn

One provider interaction and candidate submission opportunity in a Run. The Run protocol owns it; round/checkpoint are not alternative names for the same thing.

### Candidate

A submitted, content-fixed artifact with identity and provenance, owned by Evidence objects and events. Editable workspace files are not yet sealed candidates.

### Artifact Promotion

Selection inside an artifact-only Study using confirmatory evaluation and fixed rules. It is not an arm comparison or production deployment.

### Endpoint

A preregistered Run-level outcome, such as whether a qualified candidate exists at the terminal budget. It is not an attractive log line chosen afterward.

### Estimand

The exact quantity a scientific Study intends to estimate, defined before execution with population, contrast, and exceptional-case handling.

### Checkpoint

An observation at a specified budget: not reached, reached without a qualified candidate, or reached with one. The Run protocol owns it; later results do not predict or fill earlier checkpoints.

### KernelSeed

A candidate qualified under a frozen narrow confirmatory scope that may enter a later portfolio Study. Owned by Lab; it is not automatically a general production operator.

## Evaluation terms

### LaunchableCandidate

A sealed candidate with exact target, entry, launch description, and complete role-tagged artifacts, admitted to common Evaluation. A filename alone is insufficient.

### Evaluation Protocol

Workload-owned assays referenced by the Study, including rows, purposes, and correctness before timing. It does not define scientific analysis.

### Logical Evaluation Attempt

One logical evaluation of a fixed candidate, retaining broker jobs and bounded zero-work admission recovery. It cannot change the candidate or authorize arbitrary retries.

### Evaluation Receipt

An append-only record of launches, correctness, samples, and disposition for one candidate assay. It is neither the Study conclusion nor an author-reported score.

### Portfolio Artifact

The fixed semantic-key map to sealed specialists and the refusal rule for unsupported inputs, owned by Evaluation. It is not a mutable runtime shape registry.

### Measurement Quality

Evaluation evidence about sample stability at a named boundary. Unstable timing and an incorrect answer are different facts.

## Evidence and report terms

### Evidence Object

Immutable stored bytes identified by content. A path locates an object but does not establish content identity.

### Event Ledger

Ordered, typed, append-only references to evidence and observed transitions, owned by Evidence. It is not an overwritten state file or arbitrary log directory.

### Terminal Archive

The complete replay materials for a terminal Run, including successes, failures, and deviations. Evidence uses one shape rather than a special success archive.

### Integrity

The Run Audit finding that archive bytes, references, and event order are intact. This does not prove adherence, correctness, or trusted historical permissions.

### Filesystem Custody

The Run Audit check of path ownership, modes, and single-writer requirements. Tightening permissions today cannot prove continuous historical custody.

### Protocol Adherence

Whether actual execution followed the frozen Study and prerequisites, reconstructed from events. It is separate from intact files and candidate quality.

### Run Audit

Read-only reconstruction of integrity, custody, protocol adherence, and endpoint observations from retained Evidence. It precedes the Study result.

### Study Report

Application of the frozen analysis plan to qualified Run Audits, reporting inclusion, results, uncertainty, and missingness for the preregistered question.

### Claim View

A disposable view generated from accepted Study Reports. It does not own conclusions, and README must not maintain an independent copy.

## Evidence strength and scope

These scopes are not an automatic ladder. Each stronger statement needs its own evidence.

| Label | Supported statement | Does not automatically support |
| --- | --- | --- |
| proposed | A bounded reviewed proposal may enter implementation/checking | Release or GPU acceptance |
| released | A fixed revision passed its release process | Workload correctness or speed |
| structurally_expressible | Vocabulary can describe the mechanism | Backend, GPU, or whole-program coverage |
| lowerable | Source generation or fixed-asset selection is eligible | Toolchain compilation or launch |
| compiled | A fixed toolchain produced a target artifact | Correctness or speed |
| correctness_qualified | Bound input/reference checks passed | Stable timing or other shapes |
| timed | Valid samples exist for the stated boundary | Bottleneck explanation or serving gains |
| profiled | A correct candidate has corresponding diagnostic evidence | Profiler-free latency or causality |
| operator | A named operator or fixed Program is covered | Model or serving requests |
| model_forward | The declared model-forward path is covered | Serving throughput, latency, availability |
| serving_e2e | Complete requests and token trajectories are covered | Other deployments or workloads |
| not_run | An assay was deliberately not executed | A failure |
| unknown | Evidence is insufficient | An established negative result |
| invalid | Evidence/protocol cannot support the intended interpretation | That every negative result is invalid |

State timing boundaries and both baseline/candidate absolute times with speedups. See [reading results](wiki/results.md).
