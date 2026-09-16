# ADR 0065: SoL-ExecBench task import, namespace and gate translation

Status: proposed. The first task (`fib_rmsnorm_h4096`) is implemented against this
decision; the remaining tasks are not imported by it.

## Context

`flashinfer-bench-tasks` packages 120 single-operator optimization tasks for
`NVIDIA B300 SXM6 AC` (sm_103). They arrive from two different upstream authorities:
26 tasks whose definitions come from FlashInfer-Bench (`fi_api:` tags,
`upstream/definitions/`), and 94 tasks from `nvidia/SOL-ExecBench` subset L1
(`docs/L1-CATALOG.json`). The pack's own baselines come from a third place again
(SoL-Contest-InfiniAI v3 for the 26, `qhy991/SOL-Baseline` for 16 of the L1 tasks).

Their target already exists here: `sm_103a` names `NVIDIA B300 SXM6 AC` among its
devices, and `triton-b300` routes to it with `timing_source` `cupti`. What does not
already exist is an agreement about what an imported task *is*, because the upstream
scoring, tolerance and shape protocol each conflict with a rule this project already
holds.

AKA v3 is the precedent for the import shape itself: external records as semantic
provenance only, with a new fixed-shape ABI, a new independent CPU oracle, and no
imported candidate source or timing.

## Decision

### Namespace

One root namespace `solx`, two sub-namespaces, one per upstream authority.

| | FlashInfer-Bench definitions | SOL-ExecBench L1 |
| --- | --- | --- |
| contract | `contracts/workloads/solx-fib-<slug>-<dt>-<backend>-r<R>-v<N>.json` | `solx-l1-…` |
| operator | `solx_fib_<slug>_<dt>` | `solx_l1_<slug>_<dt>` |
| module | `src/open_cake_ir/tasks/solx_fib/` | `src/open_cake_ir/tasks/solx_l1/` |
| `--task` | `fib_<slug>` | `l1_<slug>` |

The slug drops the pack's ordinal (`001_`, `L1_011_`): all 120 slugs are distinct
without it, and the ordinal is an index into a pack rather than a semantic fact. It is
retained in provenance as the join key back to the upstream task directory.

Two modules rather than one, because an Executor closure pins individual files. Adding
an L1 task must not dirty the closure that the FlashInfer-Bench tasks were released
under. The prefix also means the operator table says where each task came from without
anyone opening a file.

### The upstream score is descriptive, not the estimand

The pack scores `geomean(speedup vs PyTorch eager)`. Its own README records that task
`012`'s 3971x measures how slow that reference is: the same kernel is 8.62x against
FlashInfer. A geomean over a deliberately slow reference is exactly the single number
this harness exists to avoid reporting.

The upstream reference belongs in a fixed-baseline bundle as an external black-box
baseline. The primary performance score stays `task_efficiency_v1`. An upstream-shaped
geomean may appear in a report as a descriptive comparison view, labelled as such.

### `matched_ratio` is not inherited; the replacement is stricter, and says so

The pack passes a workload at `required_matched_ratio: 0.99` with `atol = rtol = 1e-2`,
which admits one percent wrong elements. This project compares every element of every
required case. An imported contract therefore predeclares its own elementwise `atol`
and `rtol`, and its `tolerance_rationale` states plainly that the upstream gate is a
different gate, so a PASS on one side is never read as a PASS on the other.

### One shape per contract; the rest of the axis is recorded, not claimed

An upstream task carries 7 to 48 workloads over continuous axes (57 of the 120 declare
two variable axes, 24 declare three, 9 declare four). A Workload Contract here binds
one shape with its required input cases. The remaining upstream extents are recorded in
provenance as a fact about the upstream and excluded in `semantics.exclusions`; they are
a portfolio question a later Study may take up, and listing them in a contract that
never examines them would report a domain it did not look at.

The seed extent is chosen for what it exercises, not for what it flatters. For
`fib_rmsnorm_h4096` it is `batch_size = 170`, the extent at which the upstream baseline
was weakest.

### `int64` and `bool` are narrowed explicitly or not imported

16 of the 120 tasks declare `int64` inputs (mostly `position_ids`) and 6 declare `bool`
masks. `DType` names neither. An importing contract may declare `int32` instead when it
can state a provable bound for that task's values, and must record the narrowing in
`semantics.exclusions`. This is an ABI change, not an equivalence. A task that cannot
state such a bound is not imported; widening the IR vocabulary is a separate proposal
under P1-P8 with its own Compiler Revision.

### The upstream baselines are declared restricted artifacts

`tasks/*/baseline/`, `dcu_sol/*` and `upstream/blob/` hold complete low-level target
implementations and materialized inputs. Each imported contract names them in
`provenance` with `kind: restricted_artifact`, so a clean-start or direct-low-level arm's
refusal is a property of the contract rather than a convention. Per ADR 0062 this is a
semantic role, not a file extension.

An importer reads `definition.json`, `workload.jsonl` and `TASK.json`'s axes and
constants. It does not read the baseline directories, and it does not copy the pack's
`known_issues` or `optimization_hints`: those are another author's search results, not
this task's semantics.

## Consequences

`src/open_cake_ir/tasks/**` is not in `compiler/source_set.json`, so importing a task is
not a Compiler Revision and needs no Corpus Gate approval. It is inside the Executor
closure, so each batch of tasks mints a successor Executor Revision and never edits a
released one.

A task's availability is decided by asking the Target, not by naming devices:
`admit_dtype` and `admit_operations` refuse a backend whose route cannot name `bf16`
or whose Target declares no `cast`. Measured at this ADR's implementation, the BF16
RMSNorm import is admitted on `triton-b200` and `triton-b300`, refused on the three
Metal backends for `bf16`, and refused on `triton-dcu` because `gfx938` declares only
`load`, `elementwise`, `reduce` and `store`. That last refusal is a real coverage
limit of the DCU line for every BF16 task in this pack, and is reported as such rather
than worked around.

Two upstream tasks in the pack, `rmsnorm_h1536` and `rmsnorm_h7168`, carry hidden sizes
that are not powers of two. `admit_width` refuses them on every Triton route. They are
not importable as written; that is a Finding about the width rule, not a reason to relax
it here.

Registration is not a launchable task: `deepseek_v4_*` is registered without being in
`--task`, and `dsa/` has a contract with no oracle. An imported task counts as landed
only when its contract, oracle, starter and `--task` entry all exist, as this one's do.
