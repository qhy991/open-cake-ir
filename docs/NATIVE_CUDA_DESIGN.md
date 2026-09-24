# Native CUDA lowering (development successor)

The native backend owns operation traversal, warp dispatch, shared storage, tensor-memory
allocation and release, TMA tensor maps, instruction descriptors and circular stage/phase
protocols. NVIDIA nvcc/ptxas owns compilation and physical register allocation. The target
admits B200 / sm_100a and B300 / sm_103a separately, using their existing Target
contracts and the documented common tcgen05 f16 instruction subset. Neither target
inherits timing calibration or device evidence from the other. Existing releases replay
from their pinned Git. Successor preparation preserves released locks and approvals;
the canonical release cycle derives the new identity after the source change.

## Minimal contract and P1–P8

P1/P3: preserve Schedule and its operation DAG. Add `native_cuda` and one missing transfer:
`load(movement="tmem", source_atom=...)`. Transfer is independent of a formula. Existing
load, MMA, elementwise, argmin and store compose both GEMM+bias and KMeans. No operator
name dispatch, layout language, framework compiler or kernel template is introduced.

P2/P5/P8: support row-major BF16/FP16 operands in explicitly swizzled shared tiles,
K-major tcgen05 f16-family atoms, FP32 accumulators, CTA-group one, and 128-row TMEM
readout. One epilogue thread owns one row; stores therefore explicitly use
`coalesced=false`. Source maps identify operations and resource declarations. Circular
TMA stages use ready and consumed barriers with per-slot parity; consumers release
storage only after asynchronous MMA completes. The compiler derives addresses, never
algorithmic precision or placement. Register-role redistribution, cluster operations
and unsupported range controls are refused. Reported physical
registers and device correctness remain unknown until target compilation/evaluation.

P4/P7: the new transfer is typed tensor FP32 to identically shaped register FP32, with
an explicit tcgen05 load atom. The verifier checks these effects; other backends refuse
it. Backend admission also checks role ownership, access coordinates, instruction
shapes, synchronization scope and supported resource refinements before emission.

P6: implementation precedes focused public-interface and adversarial tests. The full
Corpus Gate keeps prior expectations. Independent ADR0052 review binds the exact new
commit and Gate; the author never writes release approval.

Only explicitly represented loops and accesses are supported. Examples are known-kernel
reproduction/structural cases, not clean-start or performance evidence. Remote testing
requires the active delivery's separate authorization and exact target contract.
Exact-target compilation, external-oracle checks, matched timing with L2 flush and profiler
remain separate evidence gates. Framework acceptance is not inferred from this compiler change.

## Persistent native traversal successor for Weave investigation

The native backend now admits the existing `ProgramMap.persistent` form when the
Schedule declares `residency.ctas_per_multiprocessor` and the exact Target has an
observed multiprocessor count. It launches `min(logical tiles, SMs × requested CTAs
per SM)` CTAs, then decodes successive logical tiles in the declared traversal
order inside each CTA. The CTA's TMEM allocation remains live for its traversal;
each logical tile completes and invalidates its own pipeline/completion barriers
before the next iteration. The emitter places a CTA rendezvous at that boundary.
The host launch and toolchain metadata use the same derived grid.

P1/P3: reuse the canonical `ProgramMap`, `Residency`, operation DAG and existing
native CUDA route; no second grid size or kernel-template API is introduced.
P2: residency and traversal remain visible in the Schedule and generated CUDA.
P4: the common verifier requires residency and checks its resource bound; native
preflight requires exact Target occupancy and refuses a register cap it cannot
enforce. P5: work analysis continues to count logical tiles rather than resident
CTAs. P6: focused source and refusal tests plus the Corpus Gate cover the change.
P7: no IR data model changes; native preflight and emission acquire this
existing semantic form together. P8: each logical tile repeats the declared
TMA/MMA/barrier sequence on the same CTA. CTA-to-SM placement and actual
resident occupancy still require target observation, not an inference from the
grid size alone.

This successor establishes a native CUDA persistent tensor-core lowering
building block. It does **not** implement Weave's routing-dependent communication
SM count, inter-GPU transfer, cross-CTA readiness, or communication-worker GEMM
stealing. Those require separately admitted effects and backend mechanisms; a
PTX atomic or barrier spelling alone cannot supply their ownership/liveness proof.

`ProgramMap.cooperative=true` is a further explicit launch commitment (ADR
0082). On B300, native CUDA now queries `cudaDevAttrCooperativeLaunch` and the
compiled kernel's actual active blocks per SM, refuses a shortfall against the
declared CTA residency, and uses `cudaLaunchCooperativeKernel` with the same
derived grid and exact Target. Its metadata reports the commitment. Other
backends refuse it before emission. This promises an admitted cooperative
launch; it introduces no implicit cross-CTA wait or payload publication.

At `606a6f1e`, the B300-M4 CUDA 13.1 toolchain compiled both cooperative
native routes: the one-warp INT32 work claim used 20 registers with no spills,
and the tensor-core GEMM route used 80 registers and one barrier, also without
spills. Each ran under a separate one-GPU exclusive broker lease and was
verified against an external CPU oracle after release:

- `gpuq-c73b88f136fd`: work claim passed all three contention,
  nonzero-counter and invalid-route cases.
- `gpuq-4cebccd99fa8`: persistent BF16 GEMM+bias on A[32768,256] and
  B[256,256] matched 8,388,608 FP32 outputs with zero mismatches and zero
  maximum error under the predeclared `1e-4` gate. The 148 CTAs traversed
  1,024 logical output tiles.

Raw inputs, source, compile logs, broker receipts, device outputs and
post-release reports are retained at
`B300-M4:/home/qinhaiyan/cake-weave-b300-m4-606a6f1e-cooperative/`.
These scoped development checks do not test cross-CTA release/acquire, CTA
role switching, NVLink or Weave communication/computation overlap.

## Native one-warp FP32 FMA leaf

`native_cuda_pointwise.py` admits a complete rank-two row Schedule whose one
warp loads three FP32 operands, executes the Target-declared
`ptx.fma.rn.f32` instruction, and stores one FP32 result. The row/lane
AccessMaps, immutable inputs, fresh output, equal register widths, cache
choice, and no-resource execution are checked before emission. Its host ABI
uses the existing native route, including optional cooperative persistent
launch. `examples/schedules/native/fma-row-b512.json` is a B300 instance.
The generated instruction is explicit PTX rather than a C++ expression left
to nvcc's contraction decision. This is a reusable arithmetic leaf for a
future worker Program lowering; it does not yet fuse stages or implement the
cross-GPU MoE pipeline.

At clean source commit `2ccf404b`, 19 related contract tests and the 179-case
Corpus Gate passed. CUDA 13.1 nvcc/ptxas compiled the Cake-generated B300
cooperative kernel with 16 registers, no spills and no barriers. Broker job
`gpuq-9d4b3e68f245` used one exclusive B300-M4 GPU; after lease release, an
independent CPU `a*b+c` oracle matched all 8,192 outputs exactly. Inputs,
generated source, compile log, admission receipt and device result are retained
outside source at
`open-cake-ir-workspaces/evidence/weave-b300-m4-20260924/cake-native-fma-b300-m4-2ccf404b/`
and `B300-M4:/home/qinhaiyan/cake-native-fma-b300-m4-2ccf404b/`. This check
establishes leaf lowering correctness, not Program v2 worker execution or
performance.

## Cooperative worker Program lowering slice

The native backend now lowers one admitted Program v2 form into one CUDA
cooperative kernel: three complete FP32 FMA row Schedules over the same tile
domain, two device-scope release/acquire handoffs, two CTA classes, three
atomic-claim queues and one bounded steal window. Stage bodies are generated
from their Schedule operations and Program bindings; stage names are labels,
not MoE opcodes. The emitted host launch checks exact Target, cooperative
support and compiled residency, resets the derived internal state on the
launch stream, and returns a status slot for invalid device-resident controls.
The caller allocates the state described in toolchain requirements. Other
topologies and non-native leaf routes are refused.
The state includes a successful-steal count and a first-combine observation
(`compute_done + 1`, with zero reserved for "not observed") so the device
experiment can distinguish an exercised schedule from a merely valid output.
At fixed source commit `dafb2a70`, 7 focused Program/worker tests and the
179-case Corpus Gate passed. B300-M4's CUDA 13.1 nvcc/ptxas compiled the
generated cooperative worker kernel with 30 registers, no spills and no
barriers; the compile log is retained at
`B300-M4:/home/qinhaiyan/cake-worker-b300-m4-dafb2a70/compile.log`.
Broker job `gpuq-d78103dea1dc` ran eight predeclared plans on one exclusive
B300-M4 GPU. After the lease ended, the independent CPU oracle matched every
element of all three stage outputs in every plan; queue completion, readiness
flags, chunk counts and runtime status also passed. The broker no longer listed
this job or its allocation on the subsequent status check. The first broker
attempt, `gpuq-13fdadd3f9db`, exited before CUDA because the standalone test
command omitted its `run` argument; its separate receipt and log are retained.

| Plans | Runtime `(c, K, steal budget)` | Successful steals | Compute counter at first combine |
| --- | --- | ---: | ---: |
| Spatial split | `(12/36/120, 1, 0)` | `0/0/0` | `512/512/512` |
| Temporal chunks | `(120, 2/4/8, 0)` | `0/0/0` | `278/140/140` |
| Steal window | `(120, 2, 64/256)` | `64/233` | `260/471` |

Inputs, generated source, compile log, both broker receipts, device snapshots
and post-release report are retained at
`open-cake-ir-workspaces/evidence/weave-b300-m4-20260924/cake-worker-b300-m4-dafb2a70/`
and `B300-M4:/home/qinhaiyan/cake-worker-b300-m4-dafb2a70/`.
These counters establish schedule execution in this synthetic Program; they
are not a latency, SM-activity, NVLink or MoE measurement.

The B300 successor also accepts a Program v2 handoff whose intermediate
payload is explicitly stored on a peer GPU. Its ready flag uses
`st.release.sys.global.s32` / `ld.acquire.sys.global.s32`; a device-scope
handoff keeps its original `.gpu` instructions. The generated host `create`
requires local queue state, inspects the payload's actual peer owner, checks
the peer's exact B300 identity and directed P2P capabilities, then enables
peer access. The development Evaluation adapter admits only the named
system-scope intermediate on a peer device; every other tensor stays local.
CUDA 13.1 nvcc/ptxas compiled this bounded successor with 30 registers and
no spills. At fixed code commit `a81940b0`, 14 related tests and the 179-case
Corpus Gate passed. Broker job `gpuq-9712bbf86cf7` then used two exclusive
B300-M4 GPUs: GPU 0 ran every Cake worker CTA and GPU 1 owned `middle0`.
The standalone driver did not enable peer access. All eight predeclared
`c/K/steal` plans matched every element of all three stage outputs exactly
against the post-release CPU oracle; inputs were unchanged and the queues,
flags and status passed. The `K=2/4/8` plans observed 259/197/196 completed
compute tiles at the first combine, and budgets 64/256 yielded 64/164 actual
stolen tiles. Source, input/oracle, compile log, broker receipt, device state
and report are retained at
`open-cake-ir-workspaces/evidence/weave-b300-m4-20260925/cake-peer-payload-b300-m4-a81940b0/`
and `B300-M4:/home/qinhaiyan/cake-peer-payload-b300-m4-a81940b0/`.
The broker no longer listed this allocation after completion. Event counters
are scheduling observations, not latency or SM-overlap measurements.

The worker CTAs still execute on one GPU even when an intermediate allocation
belongs to a peer. The development launch adapter is not a sealed Evaluation
candidate. Full inter-GPU dispatch/combine, grouped expert GEMMs, per-layer
cost model and MoE Workload acceptance remain separate required work.

### Two-rank worker Program v3 successor

The bounded native successor maps Program v3's first CTA class to rank 0 and
regular compute to rank 1. Rank 0 owns dispatch, optional compute stealing
and combine; rank 1 owns regular compute. A single rank-0 state allocation
holds the work queues, per-tile flags and chunk counts. Both generated
cooperative kernels use system-scope PTX atomics and release/acquire on that
state; one intermediate resides on rank 1 and the next on rank 0. The host
ABI checks every declared tensor owner and each directed peer pair before
launch, resets state once on rank 0, checks compiled cooperative residency
on both GPUs, then launches both rank-specific worker loops. The three FP32
FMA leaf bodies still come from their complete Schedules. Other placements
and math remain refused. The single-device Evaluation adapter cannot launch
this form; a shared development adapter now allocates tensors by declared
rank, isolates rank-0 queue state, checks argument order and storage, then
requires one combined two-kernel launch and a synchronized status read.

This is a two-rank development mechanism, not an EP4 MoE implementation. It
still needs exact-source nvcc compilation and a two-GPU independent oracle
before any device claim. A sealed distributed Evaluation candidate and timing
protocol remain separate work even after the development adapter runs.

## Native PTX returned-old-value work claim prototype

A second, explicitly bounded path in the same `native_cuda` backend lowers a
one-warp INT32 Schedule with `load`, `atomic_rmw(add, relaxed, device)` and
`store`. Each lane owns one routed slot. The backend emits
`atom.relaxed.gpu.global.add.s32` only when the runtime index is in bounds,
uses the returned old value for an optional indirect work-table load, and
stores one lane-owned result. It reuses `ProgramMap.persistent` and the native
host launch; no MoE opcode, task lookup or raw source escape is admitted.
Both B200 and B300 Target documents explicitly name the PTX instruction
contract. A persistent launch additionally needs the exact Target's observed
occupancy. B300-M4 supplied all four fields through `cudaDeviceGetAttribute`
under broker job `gpuq-8bf1f1ff977e`; the retained observation is
`B300-M4:/home/qinhaiyan/cake-weave-b300-m4-occupancy-20260924/facts.json`.
A declared load cache policy is refused until the native
emitter realizes it rather than being silently dropped.

The PTX instruction provides an atomic old value under device-scope relaxed
ordering, as specified by the [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-atom).
It neither publishes a separate payload nor waits for another CTA's chunk.
The example `examples/schedules/native/atomic-work-claim.json` is a source
and admission probe. Profiler evidence and a cross-role steal remain separate
from this leaf. The P1–P8 check for
the shared atomic contract is in `docs/WEAVE_FEASIBILITY.md`; this backend
slice preserves that operation's existing typing and adds exact Target and
emission refusals.

Both ordinary-grid and persistent B300 forms compiled with CUDA 13.1
nvcc/ptxas and independently matched a permutation oracle across three route
distributions on B300-M4. Their source commits are `ebf0eaac` and `19bcc545`,
respectively. Inputs, source, broker receipt, device snapshots and post-release
reports are retained at
`B300-M4:/home/qinhaiyan/cake-weave-b300-m4-ebf0eaac-probe02/` and
`B300-M4:/home/qinhaiyan/cake-weave-b300-m4-19bcc545-persistent/`.
The broker allocated one shared GPU to each sequential correctness run and
the allocations were released before CPU verification. Both forms passed
all three cases: 2,048 same-expert routes, balanced routes with nonzero
initial counters, and skewed routes containing invalid expert ids. These
development checks establish no latency, profiler overlap, cross-CTA handoff,
or multi-GPU MoE correctness.

### B300 system-scope work claim successor

The shared IR now distinguishes `atomic_rmw(add, relaxed, system)` from device
scope. Triton refuses the stronger request rather than emitting `scope="gpu"`.
Native CUDA selects `atom.relaxed.sys.global.add.s32` only when the exact
Target declares that instruction; B300 admits it and B200 does not inherit
that admission. At clean source commit `de45cf2c`, 30 focused contracts and
the 179-case Corpus Gate passed. CUDA 13.1 ptxas compiled the generated B300
kernel with 20 registers and no spills. One exclusive broker GPU in job
`gpuq-9d4930d83925` passed the same three independent permutation-oracle
cases, and the job disappeared from subsequent broker status. Source, inputs,
compile log, receipt, device snapshots and post-release report are at
`open-cake-ir-workspaces/evidence/weave-b300-m4-20260925/cake-system-atom-b300-m4-de45cf2c/`
and `B300-M4:/home/qinhaiyan/cake-system-atom-b300-m4-de45cf2c/`.

That Cake Schedule used a local state pointer. System-scope instruction
correctness does not grant access to a peer-owned allocation. A distributed
launcher must check every exact device pair's peer access and native P2P
atomic support, bind remote storage and prove payload publication separately.
The direct four-GPU EP4 mailbox experiment exercises those mechanisms outside
Cake; the Compiler still cannot lower its remote address and flag effects.

A separate two-GPU development binding at the same fixed Compiler commit
`de45cf2c` passed GPU 1's `counts` allocation to Cake code running on GPU 0.
The runner checked the selected pair's peer access and native P2P atomic
attributes before enabling peer access. Broker job `gpuq-b126c81629bb` used
two exclusive B300-M4 GPUs and passed the all-one-expert, balanced-nonzero and
skewed-invalid route cases against the post-release permutation oracle.
Inputs, Cake source, compile log, peer observation, receipt and device report
are retained at
`open-cake-ir-workspaces/evidence/weave-b300-m4-20260925/cake-system-peer-atom-b300-m4-de45cf2c/`
and `B300-M4:/home/qinhaiyan/cake-system-peer-atom-b300-m4-de45cf2c/`.
The broker no longer listed the allocation after completion. This proves one
exact peer-state work-claim binding, not a Cake-owned pointer-placement
protocol, payload handoff, EP4 execution or latency.

At successor commit `f0ae3b36`, the generated native host `create` now
inspects each system-atomic state pointer with `cudaPointerGetAttributes`.
For a peer-owned device allocation it checks the owner's exact B300 identity,
directed peer access and native P2P atomics, then enables the peer mapping.
Device-scope atomic schedules emit none of these checks. The fixed successor
passed 20 related tests and the 179-case Corpus Gate; CUDA 13.1 compiled its
kernel without spills. In broker job `gpuq-7d83558601a0`, two exclusive GPUs
passed all three post-release permutation-oracle cases with GPU 1 owning
`counts` and GPU 0 executing the Cake kernel. The test driver did **not**
enable peer access, so the generated host ABI exercised that path. Receipt,
source, compile log and observed output are retained at
`open-cake-ir-workspaces/evidence/weave-b300-m4-20260925/cake-system-peer-host-b300-m4-f0ae3b36/`
and `B300-M4:/home/qinhaiyan/cake-system-peer-host-b300-m4-f0ae3b36/`.
The peer mapping is CUDA-context state and remains enabled until that context
ends; a sealed distributed Executor still needs explicit context ownership.
This qualification covers one remote atomic state pointer only. It does not
publish a payload, schedule remote expert work or time an MoE layer.

PTX encoding reference: NVIDIA PTX ISA, sections 9.7.17.4 (matrix descriptors),
9.7.17.8 (TMEM allocation), 9.7.17.9 (TMEM transfer), and 9.7.17.10 (tcgen05 MMA):
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html .

## Transport and finite support domain

`load(movement="global")` may explicitly stage a masked half/BF16 global tile into
shared memory. Threads apply the declared swizzle, fence generic stores into the async
proxy, synchronize the producer warp and arrive at the ready mbarrier. This handles
contiguous K65/K66 inputs without hidden padding or host copies. `movement="tma"`
requires aligned global strides; its rank-2 box extends with a unit batch coordinate
when the AccessMap names a scalar batch ProgramAxis. Mixed global/TMA producers use
one ready barrier. Every stage has one MMA issuing thread; multiple MMA nodes queue
from that thread before the consumed commit, so the commit covers every stage reader.

An MMA owns a distinct completion barrier. Readout waits completion, fences thread
synchronization into tcgen05, issues complete-warp TMEM loads and waits for their
completion. Readout shares the contraction loop's parent scope. Four readout warps
must start at a multiple of four; their role-local rows then match physical TMEM lanes.
All output-tile readouts finish before the next output-tile iteration, and allocation
owners deallocate only after all readers finish. Pipeline stages overlap copy and MMA;
CTA barriers occur at allocation and output-tile/contraction boundaries, not between
individual stage operations.

Native code currently supports M128 atoms, N in [8,256] at multiples of eight, K16
atoms and K16/32/64 staged tiles selected by full-row 32/64/128-byte swizzles. Register
arithmetic supports FP32 add/sub/mul/div/relu/square and explicit floating-point casts.
Streaming argmin preserves global indices, lowest-index ties and centroid-tail masks.
Resident argmin uses local tile positions and is admitted only when a complete,
zero-origin candidate domain fits in one tile. `Schedule.argmin_domain` follows the
actual access and producer graph for both native CUDA and Triton. A 64-column physical
tile with four declared candidates therefore excludes columns 4 through 63. Numeric
broadcasts with conflicting candidate bounds are refused, not used to truncate the
candidate domain. Unsupported origins and nonzero subranges receive local findings.
The shared query does not model a runtime-valid candidate prefix. It refuses that
domain before emission; global storage capacity cannot replace a runtime extent.
The global addressing domain is bounded to signed-32-bit element counts. Unsupported
placement, precision, resource refinements, range controls and role arrangements fail
locally; there is no scalar-MMA or target fallback.

Native examples are complete hardware Schedules, not backend-renamed Triton inputs.
The KMeans core explicitly takes `centroid_sq`, matching the existing portfolio runtime
preparation contract. The original Workload still owns tokens/centroids and strict ids.
Norm preparation and its costs belong in qualification accounting; core-launch timing
is not the end-to-end timing of the original two-input Workload.

## Exact compile and launch boundary

Use explicit `--gpu-architecture=compute_100a --gpu-code=sm_100a` (or the matching
103a pair), never `-arch` shorthand that also emits non-specific virtual code. This was
confirmed by an independent CPU nvcc diagnostic: the shorthand failed PTXAS admission;
the same source with the exact pair compiled. `--fmad=false` keeps decomposed arithmetic
from acquiring an undeclared contraction. `CUtensorMap` retains its native alignment.

The emitted C ABI separates create/launch/destroy. Create encodes descriptors, validates
the exact device, records buffers and configures shared memory. It does not copy/pad,
compute norms or run kernels. Launch checks the device binding and launches on the
supplied stream. The caller owns all buffer storage and synchronization. The handle must
outlive queued launches. The emitted metadata is a projection of Schedule/Target, not a
second workload registry. GPU correctness is pending until the separate qualification
worker evaluates the unchanged oracle and case requirements.

### Two-rank worker queue scope

The bounded rank-placed worker keeps its queue state on rank 0. Dispatch and
combine cursors, dispatch completion, the communication-worker barrier, the
steal permit counter and the stolen-tile counter have only rank-0 users. Their
returned-old-value increments use the Target's
`ptx.atom.relaxed.gpu.global.add.s32` contract. The compute cursor and
completion counter have writers on both ranks; their increments retain
`ptx.atom.relaxed.sys.global.add.s32`. Cross-rank ready flags retain
system-scope release/acquire; chunk completion uses a system-scope counter,
while the consumer also waits on each payload's ready flag. This scope
selection follows the Program's fixed rank and queue ownership. The peer pair is
checked by the generated host ABI. The scope change has no B300 timing claim
until the successor source is compiled and measured on device.

### BF16 row-dot expert projection leaf

`examples/schedules/native/bf16-row-dot-h16-i32.json` composes two BF16
global loads, two explicit FP32 casts, a lane-wise multiply, a CTA-scoped SUM
and one FP32 global store. The native SIMT path assigns one weight row to one
CTA and reduces 16 or 32 products with a fixed warp shuffle tree. Its
AccessMaps state the vector, weight-row and output coordinates; admission
rejects another reduction, dtype, mapping, cache hint or resource shape.
The participating lanes use an exact 16- or 32-bit shuffle mask and power-of-two
width, following the [CUDA warp synchronization constraints](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-sync-intrinsic-constraints).
This makes one expert projection expressible through Cake without an opaque
MoE operation. It does not implement expert selection, the gated activation,
the second projection, cross-GPU dispatch or a performance claim. B300 nvcc
and oracle validation remain required before using this leaf in an Evaluation.

## Completion phases and nested reduction scopes

A contraction loop carries its TMEM accumulator across K iterations. Its non-pipeline
completion barrier publishes that final value once in the contraction's parent scope:
initialize count one, issue all K iterations, commit once from the MMA issuing thread,
wait successfully on phase zero before TMEM readout, then invalidate after all readouts.
It never advances through unobserved intermediate phases. The stage ready/consumed
barriers continue their independent per-slot phase handshakes during the contraction;
every used consumed slot is explicitly drained before invalidation. TMA and MMA remain
concurrent roles. This follows [PTX primary-phase rules](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#primary-phase).

The immediate parent of a contraction owns its completion-barrier invalidation; a more
distant ancestor cannot invalidate it again. Likewise, carried argmin state initializes
on each dynamic entry to its own reduction loop, outside that loop but inside any
surrounding row/batch loops. State survives centroid tiles, never unrelated row tiles.
Unknown loop buffers or out-of-rank dimensions are common Schedule-semantics Findings.
Native shared staging outside a contraction pipeline and TMEM swizzles are explicit
backend refusals until their instruction and storage effects have an implementation.

The completion consumer contract is checked for every declared waiter: it must be a
TMEM LOAD of that MMA's result in the contraction parent scope. Other signal uses or
per-K consumers are refused, so final publication does not discard an observable
per-iteration notification. The original ready/free per-stage protocol is separate.

A single K tile is represented canonically by a contiguous root LOAD/MMA DAG and a
one-stage pipeline, with no authored TileLoop. The same emission path visits that
stage once and publishes once. More stages without a K loop are refused; declaring a
single-trip TileLoop remains a common canonical-form error. K2 and longer contractions
use their declared loop and stage count. The focused publication-scope checks cover
K1, K2, repeated ring wraps, multiple MMA results and nested row/centroid iterations.
