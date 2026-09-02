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

## V2 permission-repair successor

The create-only successor task was
`open-cake-state-store-b8x128-b200-correctness-v2`. It preserved the same released
Compiler, generated candidate, four `[8,128]` workloads, complete-output oracle,
bitwise/ABI/in-place acceptance boundary, and correctness-only resource stage. It was
not a retry of the v1 identity. The fixed route was
`verda-b200x4-state-store/open-cake-state-store-b8x128-b200-correctness-v2-bd76e612edb0`;
the broker job was `gpuq-1795099eb4b5`, shared mode on physical GPU 1.

V2 also terminated `infra_error` with validity `unknown` and no frontier eligibility.
It passed the v1 failure point: the node-owned candidate files were `0640
qinhaiyan:gpuq-users`, the broker imported `evaluate.py` and the generated Triton
wrapper, and GPU execution reached the first case's post-compute artifact-write step.
The next infrastructure boundary failed: the correctness stage directory was `2750
qinhaiyan:gpuq-users`, so the broker identity could not create either
`zeros-b8x128.complete-output.json` or the stage `result.json`. The node-owned receipt
therefore records exit 1 and a missing judge result. No output arrays, mismatch counts,
or max-error values were durably produced.

| Workload | V2 observation | Validity |
| --- | --- | --- |
| `zeros-b8x128` | Wrapper execution reached artifact creation, but no complete output or metrics were retained | `unknown` |
| `signed-integers-b8x128` | Not reached | `unknown` |
| `alternating-binary-fractions-b8x128` | Not reached | `unknown` |
| `seeded-scaled-integers-b8x128` | Not reached | `unknown` |

The exact deployed GPU Infra source was commit
`6a1dff5b45d286838ab6846a2595409da307a878`, Kernel Infra `0.16.0`, daemon instance
`warm-sun-begins-fin-03-pid4082710-start13477426`, socket
`/tmp/kernelinfra-open-cake-state-store-b200-v2-6a1dff5.sock`, and state root
`/home/qinhaiyan/open-cake-state-store-b200-v2-6a1dff5/state`. The daemon ran with
umask `0027`, `KERNELINFRA_RUN_DIR_MODE=750`, and
`KERNELINFRA_RUN_FILE_MODE=640`; the state root was `2750
qinhaiyan:gpuq-users` and not world-readable. The task deployment occupied 1.9 MiB.

The first infrastructure successor `6a1dff5` corrected a separate defect in
`a0abba8`: an unset file-mode variable still preserved a fleet-delivered `0600`
candidate instead of applying the documented broker-safe default. The v2 run then
exposed that the earlier stage-directory change remained umask-sensitive. GPU Infra
commit `41341f8` makes each broker-owned stage output directory explicitly `0770` after
creation and adds a regression assertion. Both permission repairs passed the focused
candidate/store/runner tests and the full 75-test repository suite. `41341f8` was not
deployed and no further GPU submission was made.

The create-only external evidence root is
`/Users/haiyan-infiniai/.codex/runs/open-cake-state-store-b200-correctness-20260902-v2-6a1dff5`.
It retains the route, terminal status and wait observations, broker/node observations,
a 72 KiB mirror, and create-only `collection-01`. The task-owned broker job was
terminal and absent from the running queue before cleanup. The exact task daemon was
stopped gracefully and its socket and process disappeared; the shared broker, shared
daemon, foreign jobs, and foreign GPU allocations were not modified. The remote
task-owned state and local mirrors were retained.

Neither v1 nor v2 establishes `fixed_instance independent target correctness`.
Together they provide infrastructure diagnostics only. They do not support arbitrary
`n`, complete-parent coverage, source-independent semantics, performance, operator,
model, serving, or training claims.

## V3 stage-directory-fixed successor

The create-only v3 task was
`open-cake-state-store-b8x128-b200-correctness-v3`. It retained the exact released
Compiler v40 generated candidate and the same four correctness workloads and
acceptance boundary. It was neither a retry nor a reroute of v1 or v2. Its fixed route
was
`verda-b200x4-state-store/open-cake-state-store-b8x128-b200-correctness-v3-796ec1898a2d`;
the broker job was `gpuq-dc17a6353a4c`, shared mode on physical GPU 2.

V3 crossed the v2 stage-directory boundary: the node-owned stage directory was
`0770 qinhaiyan:gpuq-users`, the broker judge exited 0, and it created a stage
`result.json` plus one complete-output artifact for each of the four workloads. Those
five judge-owned files were created as `0600 gpuq:gpuq-users`. The daemon identity
therefore could not read `result.json` to validate and aggregate it, and the collector
could not read the complete-output artifacts. The node-owned run terminated
`infra_error`, validity `unknown`, with no frontier eligibility and no stage receipt.

| Workload | V3 observation | Durable correctness fields | Validity |
| --- | --- | --- | --- |
| `zeros-b8x128` | Complete-output file exists but is unreadable to daemon/collector | 1,024-element values, mismatch counts, max error, update immutability, state pointer and empty tuple are unavailable | `unknown` |
| `signed-integers-b8x128` | Complete-output file exists but is unreadable to daemon/collector | Same fields unavailable | `unknown` |
| `alternating-binary-fractions-b8x128` | Complete-output file exists but is unreadable to daemon/collector | Same fields unavailable | `unknown` |
| `seeded-scaled-integers-b8x128` | Complete-output file exists but is unreadable to daemon/collector | Same fields unavailable | `unknown` |

The exact deployment was GPU Infra commit
`41341f8a233f87148ceaab7f0599b346268622bf`, Kernel Infra `0.16.0`, daemon instance
`warm-sun-begins-fin-03-pid567579-start13537190`, socket
`/tmp/kernelinfra-open-cake-state-store-b200-v3.sock`, and state root
`/home/qinhaiyan/open-cake-state-store-b200-v3/state`. It used umask `0027`,
`KERNELINFRA_RUN_DIR_MODE=750`, and `KERNELINFRA_RUN_FILE_MODE=640`; the state root
was `2750 qinhaiyan:gpuq-users` and the deployment occupied 1.9 MiB.

The create-only external evidence root is
`/Users/haiyan-infiniai/.codex/runs/open-cake-state-store-b200-correctness-20260902-v3-41341f8`.
It retains the route, terminal status/wait, node/broker observations and create-only
`collection-01`. Direct fetch created no mirror, and collection recorded
`fetch_failed`, because export encountered the same unreadable complete-output file.
The task-owned broker job was observed completed with exit 0 and absent from the
running queue. The exact v3 daemon was stopped gracefully; its process and socket
disappeared. The shared daemon, shared broker and foreign jobs were not modified, and
the task-owned remote state was retained without chmod or deletion.

This failure exposed a third, separate cross-identity handoff defect. GPU Infra commit
`97a2bff` reuses the existing execution guard inside broker stages and, after the judge
exits, finalizes only that broker UID's task-owned stage directories and regular files
to the existing `0770` and `KERNELINFRA_RUN_FILE_MODE` boundaries. It adds no new
permission key, queue or allocator. Focused guard/runner tests passed 7/7, the full
repository passed 76/76, and an explicit `0027` plus `0750/0640` lifecycle check
observed the stage result as `0640`. The fix was not deployed and no further GPU
submission was made.

V3 therefore still does not establish `fixed_instance independent target
correctness`. V1, v2 and v3 remain three distinct infrastructure-unknown records and
support no arbitrary-`n`, complete-parent, source-independent semantics, performance,
coverage, operator, model, serving or training claim.
