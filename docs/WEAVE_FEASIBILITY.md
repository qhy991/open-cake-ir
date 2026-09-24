# Weave design versus current Cake IR

Evidence date: 2026-09-24. Compiler base: `3e5f9bff651e9dfc56731a6f7bce3ac6855873fc` (`origin/main` at branch creation). Paper: [Weave, arXiv:2609.21483v1](https://arxiv.org/html/2609.21483v1), submitted 2026-09-18. This is an offline expressibility and evaluation-boundary audit; it contains no new GPU measurement.

## Decision

Cake can express and lower the **local MoE arithmetic** in its current fixed-shape, single-device domain. It cannot currently express or evaluate Weave's defining **expert-parallel, cross-GPU persistent megakernel with per-layer/per-GPU dynamic SM partition and tile stealing**. A port of the existing Cake MoE task would not test that claim: it uses `T=1`, FP8, one B300 target and eight ordered kernels; Weave evaluates BF16, 2k–8k routed tokens, EP=4, four H100s and one persistent kernel per layer.

## What Weave actually does

After routing, each GPU knows local and incoming token-expert counts. The paper models computation as `(X_local + X_in) × 6HI` FLOPs, deduplicated dispatch as `X_in_uniq × 2H` bytes, and combine as `X_in × 2H` bytes (§3.1, equations 1–3). Calibrated bandwidth and TFLOPS curves by SM count, a chunk-efficiency correction and an overlap factor drive a runtime joint search over communication SM count `c` and chunk count `K` (§3.1–3.3). The launch has one CTA per H100 SM; each CTA chooses its communication or computation role using `blockIdx.x < c*`. Global atomic counters allocate tiles. Communication CTAs perform some GEMM work after dispatch, and every CTA participates in final combine (§4). This requires cross-rank token transfer, synchronization and ownership in addition to local GEMM.

The published evaluation is on 4×H100 SXM, EP=4, BF16, six models and three sequence lengths (§5.1–5.2). The reported 2.89× MoE-layer and 1.33× end-to-end geometric means in the abstract are aggregates across its stated baselines; the per-baseline geometric means vary (§5.2–5.3). The paper's end-to-end number combines SGLang attention time with measured MoE time (§5.3); it is a composed latency estimate, not evidence here of a complete Cake serving integration. These are paper results, not Cake measurements.

## Existing Cake coverage and gaps

| Weave requirement | Current evidence | Assessment |
| --- | --- | --- |
| Routing, local expert projections, activation, weighted combine | `src/open_cake_ir/tasks/solx_fib/moe.py::author_plan` constructs eight Cake stages; ADR 0075 records six B300 development correctness cases. `test_solx_fib_launch_plans.py` checks all eight stages lower. | Present for the fixed local task, with different precision and geometry. |
| Standalone MoE gate | `contracts/workloads/deepseek-v4-moe-gate-fp32-triton-b200-v1.json` and `contracts/workloads/README.md` cover score routing only. | Present as a separate workload, without expert compute or communication. |
| Local grouped/ragged GEMM primitives | `docs/adr/0025-ragged-grouped-gemm-is-primitive-composition.md` and `corpus/schedules/ragged-grouped-gemm-b1-smoke.json` cover a local BF16 contraction. | Useful building block; not Weave's distributed pipeline. |
| Exact H100 target | `compiler/targets/` declares B200 (`sm_100a`) and B300 (`sm_103a`), but no H100 (`sm_90` family) document. `Compiler.assess` returns `TARGET_UNSUPPORTED` for the probed `sm_90a`. | Blocking for paper-exact reproduction; Cake refuses architecture fallback by design. |
| Cross-GPU dispatch/combine | `compiler/ir/vocabulary.py::OperationKind` has no remote transfer or communication operation. `compiler/ir/program.py::Program` owns one target and static same-stream stages; `evaluation/cuda_driver.py::PersistentCudaCandidate.launch` admits one logical GPU. | Blocking for EP=4 execution and external oracle/measurement. |
| Runtime partition of CTAs into SM roles | `Role.execution_groups` divides execution groups **inside a CTA**, not CTAs across SMs. `ProgramMap.persistent` walks static work; the new explicit `cooperative` commitment now lowers to a qualified native B300 launch and passed scoped work-claim correctness. | Cooperative residency is available, but Cake still cannot represent `blockIdx.x < c*(routing)` with different CTA bodies. |
| Dynamic tile claiming, stealing and cross-CTA pipeline readiness | Cake's narrow INT32 `atomic_rmw` returns a unique old counter value that can address a work-item load. Triton and the new native CUDA PTX route both lower this single-role slice; ordinary and persistent native B300 forms passed scoped development correctness on B300-M4. Cake still has no modeled cross-CTA readiness signal/wait, CTA role transition or proof of ownership/liveness for the five-stage DAG. | Single-role dynamic work claiming is expressible; Weave's cross-role stealing remains blocked. |
| Online `c,K` model and calibration | Existing cost work ranks static candidates before evaluation; target documents carry hardware facts but no per-SM communication/compute throughput curves or chunk-efficiency calibration for EP. | New measurement/model contract required; paper's H100 curves cannot be inherited by B200/B300. |

## Reproducible offline probes

All probes were run in detached worktree `/tmp/cake-weave-validation-check-3e5f9bff` at commit `3e5f9bff` so the Compiler source identity remained stable. No GPU was allocated.

1. `PYTHONPATH=src python3 -m unittest tests.contracts.test_solx_fib_launch_plans.CompletePackPlanTests.test_all_26_tasks_have_one_owning_implementation tests.contracts.test_solx_fib_launch_plans.CompletePackPlanTests.test_attention_and_moe_stages_are_assessed_and_lowered_by_compiler tests.contracts.test_solx_fib_launch_plans.CompletePackPlanTests.test_moe_retains_complete_geometry_and_runtime_scalars` — 3 tests passed.
2. Constructing the frozen SoL MoE `WorkloadContract`, then calling `moe.launch_plan(workload)` and `Compiler.load(root).lower_program(plan)` produced eight `sm_103a` lowerings. The contract has six cases: `primary`, `zero_hidden`, `all_nonlocal`, `mixed_local`, `scaled`, `ties`.
3. Changing that plan's first Schedule target to `sm_90a` yielded `lowering_eligible=False` with `TARGET_UNSUPPORTED`. Changing an operation kind to `remote_load` raised `ScheduleParseError` because it is outside the admitted operation vocabulary. Changing one Program stage target from `sm_103a` to `sm_100a` raised `ValueError: program stage 'moe_group_scores' target differs`.

These probes establish only current admission/lowering boundaries. They do not establish that any particular proposed communication/scheduling extension is correct or fast.

## Focused spatial, temporal and steal probes

`tools/probe_weave_schedule.py` is a repeatable, CPU-only admission probe. It was run in a clean detached worktree at `e67fd64cba81d1ecf55f201e3d0c005095e17899` with `PYTHONPATH=src python3 tools/probe_weave_schedule.py`. The existing atomic-reservation contracts and persistent traversal test also ran there: 9 tests, 1 environment skip, 0 failures, using the repository's test virtual environment.

| Mechanism attempted | Observed admission / lowering | What it proves |
| --- | --- | --- |
| Static persistent spatial work mapping | Accepted; emitted `tl.range(program_id, TOTAL_TILES, NUM_CTAS)` with 592 CTAs walking 1,024 tiles. | Cake can keep CTAs resident and traverse a fixed work domain. This does not partition CTAs by runtime routing results. |
| Two declared roles on Triton | Refused with `TRITON_ROLE_COUNT`. | Existing `Role` partitions execution groups *within* a CTA; Triton only lowers one such role. There is no CTA-class assignment for `blockIdx.x < c*`. |
| Cross-role producer/consumer without handshake | Refused with `OP_CROSS_ROLE_RACE`. | A `depends_on` edge does not establish concurrent-worker visibility. |
| Cross-role producer/consumer with declared `barrier.sync` | Refused with `TRITON_BARRIER_UNSUPPORTED` and `TRITON_ROLE_COUNT`. | The verifier can name a handshake, but this backend cannot emit it. It also cannot realize the two roles. |
| Static chunk loop | Accepted. | Fixed loop/chunk structure is expressible. |
| Loop bound naming a runtime routing count | Refused with `LOOP_STOP_PROGRAM_UNKNOWN`. | Current `LoopStop` derives from a declared Program axis; this probe cannot bind a device-resident routing count to a runtime-chosen `K`. This single refusal is not a proof that every conceivable encoding of `K` is impossible. |
| Persistent atomic work claim, old value used to load and store a work item | Accepted; emitted `tl.atomic_add`, a runtime-indexed `tl.load`, and a persistent loop with 148 CTAs over 256 logical tiles. | The building block for dynamic tile assignment exists for one worker type. The probe is an INT32 work-list slice, not GEMM or a communication/computation pipeline. |
| Communication producer hands its claimed tile to computation role | Refused with `OP_CROSS_ROLE_RACE`. | Reusing the atomic claim does not provide Weave's safe role transition or chunk-ready handoff. |

The Cake `Program` executes separately lowered stages in order on one stream (`evaluation/program.py::_launch`), so reordering or statically chunking those stages cannot reproduce same-kernel communication/compute overlap. The probe generated source only; it did not build a GPU binary or validate output on a device. The user's four-GPU cap was respected: **zero GPUs allocated**, because the complete Weave schedule fails compiler admission before device evaluation. A future device experiment needs a complete admitted candidate, external oracle and the repository's acceptance gates and GPU lease procedure.

## Smallest useful successor validation

1. Define a new EP MoE Workload Contract with routing, placement, deduplication, dispatch/combine semantics, BF16 cases, cross-rank oracle and explicit device reset/timer/profiler policy. Keep the existing frozen SoL and V4 gate contracts intact.
2. Add a reviewed exact H100 Target, choosing the `sm_90` family codegen variant from hardware and implementation evidence, only if reproducing the paper's H100 protocol; otherwise state a B200/B300 adaptation with its own hardware measurements. A Target addition alone gives no EP capability.
3. Design the shared multi-device Program/effect and synchronization semantics first: remote buffer ownership, transfer completion, cross-CTA and cross-rank readiness, work-queue uniqueness and liveness. Check P1–P8 with counterexamples; do not hide communication or SM role changes behind an opaque MoE opcode.
4. Implement one complete, correct dispatch → local expert compute → combine slice with an external oracle, then an overlap slice, then runtime `(c,K)` and stealing. Each compiler primitive and its analysis must land together. Measure per-rank latency and profiler overlap; qualify cost curves on the exact target. Do not infer speedup from offline lowering or a cost estimate.

Shared IR, Program and verifier work belongs on `task/core-*` against `main`; NVIDIA target, backend and on-device validation belong on `task/nvidia-*` against `nvidia`, following `docs/DEVELOPMENT_BRANCHES.md`. Formal GPU work remains subject to the repository's GPU lease and acceptance gates. This audit's promotion disposition is **no Compiler promotion**: it identifies missing contracts and evidence, but has not validated a reusable new primitive.

## Native atomic successor contract

The native CUDA successor names `ptx.atom.relaxed.gpu.global.add.s32` in the shared
instruction registry. It is a second physical realization of the existing typed
`atomic_rmw(add, int32, relaxed, device)` effect, not a new atomic opcode or a
second spelling of MoE. P1/P3 retain the ordinary returned-old-value operation;
P2 makes the physical PTX route inspectable; P4/P5 reuse its existing state,
index, result and bounds checks; P6 requires accepted/refused kernel cases and
the Corpus Gate; P7 keeps typing and admission with the primitive; P8 binds the
emission to an exact Target declaration and PTX ISA form. NVIDIA Target admission
and native lowering belong in the platform task and are not implied by this
registry row alone. Relaxed atomic add guarantees a unique claimed index; it does not
publish a preceding payload or make a later worker's read ready. Weave's chunk
handoff will require a separate release/acquire and liveness design.
