# B300-M4 CTA scheduling mechanism probe

`worker_pipeline.cu` is a direct CUDA/PTX reference for the control mechanisms
Cake does not yet express. It is a synthetic **single-GPU** pipeline: dispatch
copies one 16-element token tile, compute performs a small FP32 matrix-vector
tile, and combine copies the result. These operations deliberately stand in
for MoE communication and grouped GEMM; they do not implement either one.

One cooperative launch creates 148 CTAs, matching the observed SM count of the
exact `sm_103a` Target. Every CTA reads device-resident `(c, K, steal_limit)`.
The host admits the Target's two declared B300 device names; B300-M4 exposed
`NVIDIA B300 SXM6 AC` on a broker-assigned card, which a first prototype had
incorrectly refused before kernel launch.
`blockIdx.x < c` selects communication workers; other CTAs compute. The
runtime values change the worker split and the chunk size without recompiling
the kernel. The host refuses a device without cooperative-launch support or
enough occupancy for this grid. [CUDA's cooperative launch contract](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html#when-to-use-cudalaunchcooperativekernel)
is the liveness premise for the cross-CTA waits. Ordinary `ProgramMap.persistent`
currently does not make that promise.

The phase protocol is:

1. Communication CTAs claim dispatch tiles with device-scope relaxed PTX atomics.
   Each tile's payload is published by `st.release.gpu.global.u32` on its own
   ready flag after the whole warp finishes writing.
2. Compute CTAs claim tiles from a second counter and use
   `ld.acquire.gpu.global.u32` on the exact tile flag before reading it.
3. Once all dispatch tiles finish, communication CTAs claim compute tiles until
   the first chunk is complete, the explicit steal budget expires, or the
   compute queue empties. Every communication CTA then reaches one bounded
   phase barrier. Compute CTAs keep processing their claimed work.
4. All CTAs may claim combine tiles after that barrier. A combine waits until
   its entire chunk has completed compute, then acquires its own tile's ready
   flag before reading the result. The release/acquire pair, rather than a
   relaxed completion count, owns payload visibility.

This is a bounded kernel protocol, not an IR vocabulary decision. The test ABI
has 16 separately allocated buffers: input, weights, dispatched values,
computed values, output, two per-tile flags, per-chunk completion counts,
seven work/phase counters, four summary statistics, three per-tile execution
counts, a device-resident three-value plan, status and per-CTA role trace. The standalone runner
must reset flags, counters, execution counts and status before each launch.

The external oracle checks the matrix-vector output and exactly one dispatch,
compute and combine execution per tile. It checks `computed_by_compute +
stolen == tile_count`, the steal budget, the runtime CTA role split and that
each tested plan reaches completion. The first combine records the number of
completed compute tiles, exposing whether K let combine start before all
compute finished. A nonzero stolen count is needed to say the steal path was
exercised. Inputs, compile output, broker receipts and device snapshots stay
outside the source checkout. Correctness and performance are separate phases;
this reference makes no timing or NVLink claim.

Before Cake can admit the same structure, its shared IR and verifier need to
name CTA classes separately from warp `Role`, device-resident runtime split
inputs, bounded work queues, release/acquire payload ownership, cooperative
residency and phase liveness. The native backend then owns PTX emission; the
Workload Contract owns the external oracle and per-rank routing semantics.

## B300-M4 development observation

The exact source at `863eb3fd` compiled with CUDA 13.1 nvcc/ptxas for
`sm_103a` (48 registers, no spills). Under one exclusive broker GPU,
`gpuq-07e90df5e32c` completed three preregistered plans. The independent
CPU oracle ran after the GPU lease was released and found the matrix-vector
output correct, every dispatch/compute/combine tile executed exactly once,
the requested CTA split, and the declared steal budget held in each case.

| Plan | `c` | `K` | Stolen compute tiles | Compute tiles done at first combine |
| --- | ---: | ---: | ---: | ---: |
| No steal | 12 | 1 | 0 / 512 | 512 / 512 |
| Chunk pipeline | 36 | 4 | 0 / 512 | 506 / 512 |
| Steal heavy | 120 | 2 | 256 / 512 | 424 / 512 |

The K=4 and K=2 rows show a combine began before all compute tiles finished;
the K=2 row actually exercised compute stealing. These counters do not measure
simultaneous SM activity or latency. Raw inputs, compile log, broker receipt,
device snapshots and post-release report are retained at
`B300-M4:/home/qinhaiyan/cake-weave-worker-b300-m4-863eb3fd/` and mirrored
outside the checkout under `open-cake-ir-workspaces/evidence/weave-b300-m4-20260924/`.
An earlier run at `5cb9c344` was refused by the host's too-narrow B300 device
name check before kernel launch; its failure log and receipt are retained
separately. No GPU processes or lease from this task remained after the
successful run.

## Isolated spatial, temporal and steal sweep

At source commit `3e5715cf`, a second standalone test fixed 512 tiles, width
16 and 32 FP32 matrix-vector repetitions while changing one runtime plan
dimension at a time. Nine plans were prepared before GPU admission, then run
through one exclusive B300-M4 broker GPU in job `gpuq-314534c59efc`. The
independent CPU oracle ran after lease release. All nine plans passed output,
exactly-once execution, CTA split, queue completion and steal-budget checks.

| Controlled plans | `(c, K, steal budget)` | Stolen tiles | Compute tiles complete at first combine |
| --- | --- | ---: | ---: |
| Spatial split | `(12, 1, 0)` / `(36, 1, 0)` / `(120, 1, 0)` | `0` / `0` / `0` | `512` / `512` / `512` |
| Temporal chunks | `(36, 2, 0)` / `(36, 4, 0)` / `(36, 8, 0)` | `0` / `0` / `0` | `512` / `506` / `509` |
| Steal budget | `(120, 2, 0)` / `(120, 2, 64)` / `(120, 2, 256)` | `0` / `64` / `256` | `274` / `260` / `424` |

The spatial rows show that the device-resident cutoff changes CTA roles while
preserving correctness. The temporal rows show early combine for `K=4/8` in
this single run. The steal rows show exact budget use for two nonzero budgets.
The first-combine counter is an event-order observation, not a timing or SM
overlap measurement; it may vary with scheduling. The synthetic dispatch is an
HBM copy, so none of these rows verifies NVLink, grouped GEMM or end-to-end
MoE. The device snapshots, broker receipts, compile output and post-release
report are retained outside source at
`open-cake-ir-workspaces/evidence/weave-b300-m4-20260924/cake-weave-worker-sweep-b300-m4-3e5715cf/`
and at
`B300-M4:/home/qinhaiyan/cake-weave-worker-sweep-b300-m4-3e5715cf/`.
The first broker attempt exited before touching CUDA because the standalone
runner lacked its receipt path. The successful retry used a separate receipt;
both logs are retained. The successful job released its GPU allocation.

## Four-GPU peer-read prerequisite

A separate B300-M4 broker job `gpuq-2ffcda714d1a` requested exactly four
GPUs, the user's resource ceiling. An `sm_103a` CUDA kernel on each destination
read a peer-owned INT32 array from each of the other three GPUs through CUDA
unified virtual addressing. All 12 directed pairs reported peer access, and
all 12 outputs matched an independent CPU oracle after lease release. The
four broker-visible devices each reported B300 compute capability 10.3 and
148 SMs. Source inputs, compile output, device observations, broker receipt
and oracle report are retained outside source at
`open-cake-ir-workspaces/evidence/weave-b300-m4-20260924/cake-weave-peer-b300-m4-4gpu-3e5715cf/`
and at `B300-M4:/home/qinhaiyan/cake-weave-peer-b300-m4-4gpu-3e5715cf/`.
The broker job finished and released all four cards.

Peer readability establishes a transport prerequisite only. This test has no
concurrent producer/consumer handoff, NVLink bandwidth measurement, grouped
GEMM, expert routing or MoE output. The worker prototype above remains a
single-GPU computation with an HBM-copy stand-in for dispatch.

## Two-GPU system-scope handoff prerequisite

NVIDIA's [CUDA memory model](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html)
limits `.gpu` synchronization scope to one GPU; peer-GPU payload publication
requires `.sys` scope and, for a flag in GPU memory, native P2P atomic support
for the exact pair. B300-M4 broker job `gpuq-a9a69b66ff70` allocated two GPUs.
`cudaDeviceGetP2PAttribute` reported native peer atomic support in both
directions. A producer on one GPU wrote an INT32 payload and published a local
flag with PTX `st.release.sys.global.u32`; a concurrently launched peer kernel
waited with `ld.acquire.sys.global.u32` before reading that payload. Both
directions passed eight rounds each against the post-release CPU oracle, for
16/16 correct handoffs. The broker released both GPUs.

Inputs, PTX-bearing source, compile output, capability observations, device
outputs and broker receipt are retained outside source at
`open-cake-ir-workspaces/evidence/weave-b300-m4-20260924/cake-weave-sys-handoff-b300-m4-2gpu-3e5715cf/`
and at
`B300-M4:/home/qinhaiyan/cake-weave-sys-handoff-b300-m4-2gpu-3e5715cf/`.
This small protocol probe does not qualify a high-load all-to-all or a Cake
Program v2 lowering. The shared worker descriptor now preserves either
`device` or `system` scope; it still refuses execution until the verifier,
native emission and Evaluation state/reset owners are complete.

## Observed SM placement for the CTA split

The direct prototype now records PTX `%smid` together with each CTA's role in
`role_trace`. At source commit `58936b53`, B300-M4 exclusive job
`gpuq-2f1223552f27` reran the three original plans. The independent oracle
again passed all outputs and exact-once checks. Each of the three launches
observed **148 distinct SM IDs for 148 CTAs**; the communication CTA counts
`12/36/120` therefore corresponded to `12/36/120` distinct communication
SMs in these runs. The trace, compile output, broker receipt and post-release
report are at
`open-cake-ir-workspaces/evidence/weave-b300-m4-20260924/cake-weave-smid-b300-m4-58936b53/`
and `B300-M4:/home/qinhaiyan/cake-weave-smid-b300-m4-58936b53/`.
This is an observed placement of this one compiled kernel. Cooperative launch
guarantees enough residency for all participating CTAs, but the worker
lowering still needs to check its own occupancy and record its own SM
distribution before claiming an exact SM split.
