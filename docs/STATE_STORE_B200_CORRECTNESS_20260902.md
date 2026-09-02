# Released state-store B200 correctness attempt, 2026-09-02

## Result

Independent target correctness was **not obtained**. The single authorized Kernel
Infra run reached broker GPU custody, but the broker process could not read the
node-owned candidate judge (`0600 qinhaiyan:gpuq-users`). It failed before importing
the generated Triton module or invoking the wrapper. The lifecycle is `infra_error`,
validity is `unknown`, and frontier eligibility is false. No retry, reroute, benchmark,
profile, performance measurement, approval, release, tag, PR, or main merge occurred.

## Fixed authorities and boundary

- Source start: local `main`, fetched `origin/main`, and the isolated worktree all began
  at `7fdac036626a4eaf0a0e047dca1565c0a88f19aa` with a clean tree. Work continued only
  on `codex/state-store-b200-correctness-20260902`.
- Released Compiler: `compiler/revision.lock.json`, revision
  `open-cake-ir-sm100a-v40`; public `compiler check-corpus` passed 68/68 cases. The
  released Gate is `b57fdea8f50375bd7757b5c6774c5d2a1bae19196c7c8fbb3ee56627bca4a58f`.
- Positive Schedule: `corpus/schedules/state-store-b8-smoke.json`. The candidate Triton
  source was emitted unchanged through released `Compiler.assess_file` and
  `Compiler.lower`; it compiled as Python, passed the state-pointer-to-`tl.store`, no
  replacement `[8,128]` allocation, and empty-tuple wrapper static checks.
- Negative Schedules were rejected before GPU: owner drift by
  `STATE_STORE_PROGRAM_OWNER` at `access_maps[2].indices[0]`; axis drift by
  `STATE_STORE_PROGRAM_AXIS_COVERAGE` at `program_map.axes[1]`.
- AKA v5 qualified parent: `contiguous_apply2_add_fp32_u32_block256_v1`; source record
  `data_movement_and_layout__memory_addressing__analysis__l000001_b200_v1__directderived_sol_ultra_v2`;
  original parent `categories/data_movement_and_layout/memory_addressing/analysis.jsonl:1`;
  portable bundle
  `/Users/haiyan-infiniai/AKA-worktrees/direct-derived-parent-completion/datasets/curated/cuda_kernel_parent_completions_v5/bundles/data_movement_and_layout__memory_addressing__analysis__l000001_b200_v1`;
  historical locator
  `b200-local/directderived-contiguous-fp32-add-u32-v1-452f56c1e353`.
- Parent authority boundary: source-complete direct-derived contract with no upstream
  repository/revision authority. This attempt concerns only fixed shape `[8,128]`
  (1,024 elements), not arbitrary `n` or complete-parent coverage.

## Kernel Infra identities and custody

- Task: `open-cake-state-store-b8x128-b200-correctness-v1`, identity
  `83371caed32a9bdc5a24f79aea81be6756794a3c2e60df43f8a78bf5ec406491`.
- Candidate identity: `0c816ba6c7ed0eefcb2f88e9fa1a074b61104e41f54761cdc95c9ccb0a477914`.
- Bundle: `def0707fd417480b7fa7a89c0a48ed93`.
- Route: `verda-b200x4-state-store`, fixed run
  `open-cake-state-store-b8x128-b200-correctness-v1-a191081fd0d3`.
- Broker job: `gpuq-bcfdf702d9ef`, shared mode, one GPU, physical GPU 0. The live route
  observation bound Kernel Infra `0.17.0`, daemon instance
  `warm-sun-begins-fin-03-pid1599063-start4660606`, broker `0.5.3` instance
  `warm-sun-begins-fin-03-pid2434-start4200`, B200/CUDA/sm100 capabilities, and more
  than 20 GiB free disk.
- Node-owned stage receipt recorded broker acceptance, GPU 0 assignment, exit 1, and
  missing judge result. Stderr localized the first failure to `PermissionError` while
  reading `candidate/evaluate.py`; therefore this is infrastructure `unknown`, not GPU
  numerical `invalid`.

## Complete-output workloads

| Workload | Required checks | Observed result |
| --- | --- | --- |
| `zeros-b8x128` | 1,024-element bitwise state result, update unchanged, empty tuple, stable state pointer | Not executed; `unknown` |
| `signed-integers-b8x128` | Same, against independent CPU float32 | Not executed; `unknown` |
| `alternating-binary-fractions-b8x128` | Same, exact binary-fraction domain | Not executed; `unknown` |
| `seeded-scaled-integers-b8x128` | Same, deterministic seed `20260902` | Not executed; `unknown` |

No `state_before`, `update_before`, expected, or actual complete-output artifact was
created because the judge did not start. Consequently there is no mismatch count or
max-error observation for any workload, and no independent target correctness claim.

## Preserved evidence and cleanup

The create-only task root is
`/Users/haiyan-infiniai/.codex/runs/open-cake-state-store-b200-correctness-20260902`.
It retains the route receipt, terminal status/wait observations, a 72 KiB fetched
node-run mirror, and create-only `collection-01`. Raw logs and mirrors remain outside
the repository. Final `node-status` showed no active runs, services, ready deployments,
broker running jobs, or queued jobs; GPU 0/1/2 were idle and the pre-existing unrelated
GPU 3 foreign allocation remained untouched. No shared daemon or broker was stopped.

The checked-in generator, correctness-only task, candidate-owned judge, catalog, and
CPU/static contract tests remain reusable after the external Kernel Infra candidate-file
readability boundary is corrected by its owner. This attempt does not repair or
manufacture that permission state.
