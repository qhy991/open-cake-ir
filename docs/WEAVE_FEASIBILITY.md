# Weave design versus current Cake IR

This opening audit is fixed to its 2026-09-24 Compiler base. The successor
evidence and remaining Compiler boundary are recorded at the end of this page.

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
| Runtime partition of CTAs into SM roles | `Role.execution_groups` divides execution groups **inside a CTA**, not CTAs across SMs. `ProgramMap.persistent` statically strides CTAs over work in `backends/triton.py::_emit_persistent_header`. | Current persistence cannot represent `blockIdx.x < c*(routing)` with different CTA bodies. |
| Dynamic tile claiming, stealing and cross-CTA pipeline readiness | Cake's narrow INT32 `atomic_rmw` can return a unique old counter value, and that value can address a work-item load inside a persistent Triton kernel. Cake has no modeled cross-CTA readiness signal/wait, CTA role transition or proof of ownership/liveness for the five-stage DAG. Native CUDA preflight also refuses `program_map.persistent`. | Single-role dynamic work claiming is expressible; Weave's cross-role stealing remains blocked. |
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

The next native CUDA slice names `ptx.atom.relaxed.gpu.global.add.s32` in the shared
instruction registry. It is a second physical realization of the existing typed
`atomic_rmw(add, int32, relaxed, device)` effect, not a new atomic opcode or a
second spelling of MoE. P1/P3 retain the ordinary returned-old-value operation;
P2 makes the physical PTX route inspectable; P4/P5 reuse its existing state,
index, result and bounds checks; P6 requires accepted/refused kernel cases and
the Corpus Gate; P7 keeps typing and admission with the primitive; P8 binds the
emission to an exact Target declaration and PTX ISA form. NVIDIA Target admission
and native lowering belong in the platform task and are not implied by this
registry row. Relaxed atomic add guarantees a unique claimed index; it does not
publish a preceding payload or make a later worker's read ready. Weave's chunk
handoff will require a separate release/acquire and liveness design.

## 2026-09-25 successor evidence and next lowering boundary

The B300 task branch now emits a one-GPU cooperative worker Program from three
complete FP32 FMA leaf Schedules. It has typed CTA classes, work queues,
release/acquire tile handoffs, chunks and a steal window. B300-M4 verified
eight spatial/temporal/steal plans against an independent CPU oracle at
`cake-worker-b300-m4-dafb2a70/`. A subsequent two-GPU development run at
`cake-peer-payload-b300-m4-a81940b0/` bound one intermediate to peer storage:
all eight plans remained numerically exact and exercised system-scope flag
publication. A separate Cake-generated system-scope atomic work claim bound
peer-owned state and passed three cases at
`cake-system-peer-host-b300-m4-f0ae3b36/`. These evidence directories live
under `open-cake-ir-workspaces/evidence/weave-b300-m4-20260924/` or the
corresponding `20260925/` root. They do not make the worker Program an EP4
program: all of its CTAs execute on one GPU.

The distinct `weave-ep4-bf16-moe-b300-v1` Workload Contract now owns a
four-rank CPU oracle, expert placement, five routing distributions and
per-rank local/incoming/unique-token counts. A direct CUDA/PTX mailbox
reference, **outside Cake lowering**, passed balanced four-GPU BF16 output
correctness and a skewed plan with 14 actually stolen tiles at
`open-cake-ir-workspaces/evidence/weave-b300-m4-20260925/` (see
`experiments/weave/native_b300/EP4_MAILBOX.md`). Neither that small geometry
nor the Cake synthetic worker has measured MoE latency or serving benefit.

The current NVIDIA result projection does not establish that Triton has no
remaining MoE headroom: `docs/results/nvidia/records.json` row
`nvidia-result-042` (`Alpha-MoE`) has null baseline and candidate latency and
status `未实测`. The new BF16 EP4 contract likewise has no qualified Triton
candidate. A native CUDA/PTX performance decision needs a matched workload,
target, correctness gate and timing interval for both paths; the raw mailbox
correctness result and the FP32 worker probes cannot substitute for that
comparison.

The NVIDIA task branch at `49e1ba3a` now lowers one BF16 expert-projection
row dot from seven ordinary Cake operations (`load`, `cast`, `mul`, `reduce`,
`store`) into a one-warp CUDA kernel. The same mathematical Schedule is also
eligible for Triton with a changed lowering route, so it offers a controlled
leaf comparison once B300 execution is available. This native leaf has passed
offline Compiler and Corpus gates; it has not been nvcc-compiled or checked
against the B300 oracle, and it is not connected to the EP4 worker Program.
At NVIDIA task commit `a9151838`, the leaf additionally accepts a local
expert coordinate through an ordinary INT32 load and `scalar_buffer`
AccessMap. The native weight load masks an out-of-range coordinate to BF16
zero, matching the eligible Triton route. A five-case matched correctness
bundle includes both valid local experts and two invalid indices, but remains
unexecuted on B300-M4. This still lacks gated activation, down projection,
remote dispatch/combine and a four-rank Cake lowering.

`examples/programs/weave-local-expert-ffn-b300.json` now expresses the whole
**local expert calculation** as three complete Cake Schedules: BF16 up/gate
projection, FP32 gated activation over two nonoverlapping subranges, then
FP32-activation/BF16-weight down projection. The current Triton route lowers
all three ordered stages offline, with explicit singleton views between their
global tensors. This verifies composability of the existing IR and exact
local shapes; it does not fuse stages, execute on B300 or implement the
five-stage cross-GPU persistent worker.

The NVIDIA task branch at `e4cbd5b7` now lowers the same three-stage local
calculation through native CUDA: a selected-expert BF16 up/gate projection,
an explicit FP32 `up * gate/(1+exp(-gate))` activation, and a down projection
that keeps activation FP32 while reading BF16 weights. Native and Triton
Schedules for each stage retain the same operation graph and AccessMaps.
The native Program passes offline tests and the Corpus Gate, but has no nvcc
or B300 oracle result and still launches three kernels instead of one
distributed persistent kernel.

The origin's weighted combine is also expressible without a MoE opcode:
`examples/schedules/triton/weave-weighted-combine-t{7,8}-h16.json`
loads each token's two FP32 contributions and route weights, multiplies,
reduces the route axis and rounds once to BF16. Both Workload token extents
lower through Triton offline. The missing part is the ranked mailbox effect
that binds remote contributions to this complete Schedule and a four-GPU
oracle.
The NVIDIA task branch at `59bd7385` now emits the matched native CUDA
combine for both `T=7` and `T=8`, with one FP32 route reduction and one BF16
rounding per token. Its offline tests and Corpus Gate pass. The ranked
mailbox effect, nvcc/device correctness and performance comparison remain
unqualified.

An exact native-route admission audit of the existing T=1 FP8 SoL MoE plan
explains why it cannot simply become the EP4 kernel. `moe_gemm1` is blocked by
FP8 dtype admission, dynamic access indices, cast realization, arithmetic
and operation kinds; `moe_gemm2` has the same classes of gap. The routing
stages additionally need top-k, coordinate, compare and select lowerings.
These are multiple independently owned primitives and effects, not a missing
backend name. The BF16 EP4 development contract deliberately starts after
routing with expert IDs and weights as inputs; it does not pretend to cover
the FP8 task's routing and block-scale semantics.

The next Compiler tick must compose that visible expert math with ranked
mailbox effects, remote queue and payload ownership, system-scope
publication and liveness analysis. [ADR 0084](adr/0084-ep4-worker-needs-ranked-mailbox-effects.md)
states the four-rank admission and verification obligations. The CPU
`experiments/weave/dispatch_ledger.py` now separates deduplicated remote
payload slots from one-per-route compute tasks, including the skew and tail
capacity bounds. `experiments/weave/rank_plan.py` derives per-rank `c/K/steal`
domains and uneven tail completion counts: `T=7, K=2` requires 8 then 6
route contributions. The measured direct CUDA reference still sends one
payload per route and requires `T % K == 0`. A bounded CPU event model explores
dispatch, regular compute, local stealing and early combine over these plans;
it does not establish GPU memory order, occupancy or timing.

The NVIDIA task branch at `7fd4eaf5` adds an unmeasured `T=7` successor
while leaving the measured `T=8` source unchanged. Its host-compiled chunk
helper covers uneven partitions; it has no nvcc or B300 oracle result.
The measured `T=8` runs had the same `K` on every rank; source inspection found that
their return path indexed the origin counter with the compute rank's `K`.
An unmeasured successor at `fd222407` passes all source-rank chunk counts
and prepares `K=(2,3,7,1)` as a device counterexample. It has passed CPU
input/plan checks but not nvcc or GPU execution.
The NVIDIA task branch at `d772da55` also has an unmeasured direct CUDA
successor that separates remote payload slots from per-route compute tasks.
Its frozen `skew_to_rank0` and `tail_tokens` input/oracle bundles check the
expected 24/64 and 10/14 payload/task counts respectively; CPU preparation
and a host-only syntax check passed. Neither bundle has an nvcc or GPU result,
so no deduplication or performance claim is established for Cake.

The existing single-device launch paths refuse the EP4 Workload Contract.
No raw mailbox source is promoted as an opaque MoE instruction.
