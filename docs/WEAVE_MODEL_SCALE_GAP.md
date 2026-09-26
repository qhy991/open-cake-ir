# Weave model-scale Cake gap on B300

This is an implementation audit, not a performance result or a new Workload
Contract. The separate open-baseline task owns its provisional Qwen3-30B
geometry and CPU oracle. The frozen `weave-ep4-bf16-moe-b300-v1` Cake
development Workload remains T7/T8, E8, top-2, H16, I32; its passing runs
do not imply model-scale throughput.

## Admitted domains and remaining joins

| Mechanism | Current Cake/B300 evidence | Model-scale gap |
| --- | --- | --- |
| Ranked payload/task/return effects | `RankedMailboxEffects` types rank owners, system release/acquire, capacities, `c/K/steal`; the B300 source passed five small cases and replayed three former liveness failures after the warp-uniform fix. A separate core task branch at `52e5c88f` now types logical tiles and their stage CTA work units. | The live NVIDIA emitter still claims one `(source,item,route)` task. Core schema 2 has no B300 ranked lowering or four-device result. |
| Inline expert math | `native_cuda_ep_math.py` verifies a three-stage Program and combine Schedule, then emits a one-warp SIMT BF16 row-dot / FP32 activation/down body. | Admission fixes H16, I32, two local experts and T7/T8. Increasing constants would leave one-token/route SIMT work and not implement grouped tensor-core GEMM. |
| Tensor-core local FFN | A complete native CUDA TMA/`tcgen05` GEMM Schedule exists. Separate no-bias model-width up/gate (`cc1715b5`) and down (`de0c4824`) tiles each passed two full one-GPU FP32 comparisons bitwise against an independent FP64 oracle. A model-width SwiGLU Schedule (`cb72263a`) explicitly casts FP32 to BF16; a three-stage Program (`98431873`) binds up/gate → SwiGLU → down without an implicit cast. CUDA 13.1 compiled all three Program stages on B300-M4 with 74/24/74 registers per thread and no spills. One brokered B300 GPU run passed two full stage-by-stage cases bitwise against an independent oracle. | The ordered Program is one fixed expert tile; the model-scale bridge below invokes it from host orchestration rather than a tile-keyed GPU worker. No distributed EP4 or performance result follows. |
| Dynamic expert-bin input | Cake now admits a complete Schedule using three metadata loads, returned-old `atomic_rmw`, BF16 row load and two reservation-owned indexed stores. The exact B300 native CUDA emitter (`66e97f3a`) compiled and passed a brokered one-GPU oracle on all 16,384 routes; host admission rejected duplicate local experts and unreset counts. The earlier standalone PTX probe (`a4bcc968`) remains separate evidence. | Its host domain check synchronizes and copies route metadata before launch, and each expert bin has a fixed 2,048-row capacity. There is no cross-device release/acquire publication, tile-ready queue or FFN invocation from these bins. |
| Routed arithmetic bridge | A one-GPU run (`975abc3a`) connected Cake bins to 194 padded 128-row Cake FFN tiles across all 128 experts and retained every FP32 route contribution. Post-lease CPU weighted combine matched the independent model-scale FP64 oracle at the predeclared tolerance: 0 / 4,194,304 failing elements. | Four source ranks were staged on one GPU; host code read counts, padded tails and selected weights, and CPU performed the final combine. It has no GPU tile-ready queue, distributed return, `c/K/steal` result or qualified latency. |
| Model top-8 combine | A complete Cake LOAD→MUL→SUM→CAST→STORE Schedule (`fd698ba9`) lowered to 256-thread native CUDA CTAs. A separate one-GPU run consumed the bridge's saved FP32 route contributions and passed the same independent FP64 oracle with 0 / 4,194,304 failures. | Combine ran in a separate GPU job on retained contributions, rather than as the return phase of a live four-GPU ranked layer. It establishes no dispatch/compute/return overlap or speedup. |

The tensor-tile source, exact synthetic Schedule, nonblocking
`RESIDENCY_BOUND` finding, nvcc command/log and cubin are retained in
`cake-weave-b300-tensor-tile-probe-1c59d3af/` under the 2026-09-25 Weave
evidence root. Its compilation establishes a real lowering reuse candidate;
it proves no expert arithmetic, queue protocol or performance.
The model-width up/gate and down Schedules, generated sources, dyadic
input/oracle cases, compiled products, single-GPU broker receipts and
bitwise reports are separately retained in
`cake-weave-model-upgate-tile-cc1715b5/` and
`cake-weave-model-down-tile-de0c4824/`. Their exact fixed shapes and input
domains do not qualify the full model-scale FFN. The ordered Program, three
generated sources, two full CPU input/oracle cases and CPU nvcc/PTXAS build
products and one-GPU stage-by-stage bitwise device report are in
`cake-weave-model-local-ffn-98431873/`. This establishes the fixed tile's
ordered arithmetic, not dynamic routing or an EP4 result.

## Provisional Qwen3-30B geometry pressure

The open-baseline task's independent contract currently names EP4, 2,048
total tokens (512/rank), E128, top-8, H2048 and I768 with BF16 weights.
Projecting **the current small-worker mailbox layout unchanged** onto those
extents gives 1,536 remote payload slots, 16,384 task slots and 4,096 return
slots per rank. Its largest fields would be 6,291,456 BF16 payload bytes,
33,554,432 FP32 contribution bytes and 2,097,152 BF16 output bytes. The
whole projected mailbox is 42,360,880 bytes (40.40 MiB) per rank, or
161.59 MiB across four ranks. One rank's 32 expert weight shards would be
201,326,592 bytes for up/gate and 100,663,296 for down. These are storage
arithmetic under the old structs, **not** an admitted source, allocation or
measured L2/throughput estimate. The model contract's seeded synthetic routes
also differ from the paper's ShareGPT routing.

The same saved synthetic route IDs expose a temporal packing constraint. With
128-row expert tiles, accumulating across all 128-token source waves requires
194 logical tiles and 8,448 padded rows; 70 full tiles become publishable only
in the final wave, and 124 partial tasks require terminal flush. Treating
each source rank's wave as an independent expert bin instead would create
2,048 logical tiles and 245,760 padded rows. A CPU-only thresholded policy that
flushes bins at 64 rows after each wave creates 256 logical tiles and publishes 70
in the second wave, with 16,384 padded rows. The route-keyed task manifests,
counterexamples and exact source are retained in
`cake-weave-model-tile-flush-22d3f77d/` under the same external evidence
root. This is a deterministic publication witness, not GPU execution or a
timing model. It motivates a tile queue that accumulates across temporal
chunks and permits an explicit partial-bin publication rule.
The B300 `cake-weave-model-expert-bin-pack-a4bcc968/` record separately
validates the actual BF16 row and return-key placement for all routes on one
GPU. Its relaxed GPU-scope atomic is only a row reservation; it cannot stand
in for the system release/acquire handoff needed by the ranked EP4 worker.
The successor `cake-weave-model-expert-bin-cake-66e97f3a/` retains the
Cake-generated source, Compiler findings, exact toolchain requirements,
nvcc/PTXAS products and all-route device oracle. Its Schedule expresses
dynamic indices and reservation ownership in existing IR primitives. The
pre-launch host domain check proves the fixed capacity for the synthetic
contract but must be replaced or accounted for before any performance claim.
The `cake-weave-routed-ffn-bridge-975abc3a/` record connects that Schedule
to the tensor-core FFN Program on one GPU. It proves routed arithmetic for
the baseline's fan-in synthetic case while exposing the remaining scheduling
boundary: Python host orchestration, not the ranked worker, owns tile
formation, padding, weight selection and completion.
The `cake-weave-model-combine-fd698ba9/` record separately validates GPU
top-8 combine on those retained contributions. Its 223 BF16 bit differences
from the CPU FP64 combine are within the unchanged external tolerance; the
claim is oracle correctness, not bitwise equality or a layer latency.
The separate core schema-2 analysis (`cake-ranked-stage-tasks-52e5c88f/`)
derives that each logical FFN tile expands into **24 up/gate, 128 activation
and 32 down CTA work units**. Under the 64-row temporal policy, the safe
per-rank queue upper bounds are 255 logical tiles and 46,920 stage work
units, with 765 stage-completion slots. The saved route plan observes 64
logical tiles and 11,776 stage work units per destination rank. These are
CPU capacity/plan facts, not an emitted live GPU queue.

## Required joint change

1. **Math and mapping.** Build complete Cake Schedules for an expert up/gate
   tile and down tile at the model's BF16/FP32 accumulation contract. Reuse
   the existing TMA/`tcgen05` native emitter where its concrete SMEM views,
   TMEM columns and barrier lifetimes apply. Admit dynamic local expert
   selection and token bins with explicit byte offsets/strides, without a
   first-class layout algebra or an opaque MoE opcode. Verify each tile
   against the independent oracle before embedding it in EP4.
2. **Task/effect ownership.** A logical expert tile groups token routes;
   one ready Program-stage CTA work unit is the compute claim unit. All CTA
   work units of a predecessor stage must finish before the next stage is
   published. Derive destination queue capacity, source return ownership and
   chunk-completion thresholds from that decomposition. A route cannot
   publish twice or disappear when tiles group tokens. The verifier must
   cover partial bins, empty experts, tail tokens, duplicate destination
   payloads, aliasing, release/acquire and warp-uniform loop exits.
3. **CTA resource transition.** Current ranked workers launch 32 threads per
   CTA. The admitted tensor tile uses six warp roles in a 192-thread CTA
   (four epilogue, one MMA, one copy), plus SMEM and TMEM resources. A
   communication CTA that later steals a stage work unit must own the
   maximum-stage resources and synchronization edges needed for that
   transition. The spatial `c` budget
   must be checked against the actual resident 192-thread cooperative grid,
   not inherited from the small SIMT worker's `SMS=148` grid assumption.
   On B300-M4, `cake-weave-model-ffn-residency-a71c9eca/` records CUDA's
   occupancy query for the exact up/gate and down stages at 192 threads and
   49,200 B dynamic SMEM: **one active CTA/SM** on 148 SMs, or 148 CTAs as
   each stage's queried cooperative grid bound. Communication and computation
   CTAs must share that total budget if the future combined worker retains
   the same limit; the ranked worker itself has not been compiled or queried.
   A successor one-GPU control probe (`cake-weave-model-tile-cooperative-
   3423429c/`) completed 64- and 148-CTA cooperative grids over four
   source-chunk waves with the tensor-core worker's 192 threads, 49,200 B
   dynamic SMEM and a real `tcgen05.alloc/dealloc` transition. It covered
   `c=1,74,147,148`; at `c=147` a 32-task steal budget claimed 32 dummy
   tiles, and at `c=148` a 64-task budget claimed all 64. The first source
   (`01643c32`) failed with CUDA 719 when a persistent CTA relinquished TMEM
   after every task; holding it until the CTA finished all waves resolved
   the observed failure. These are resource/control probes with no FFN math
   or peer mailbox, so they do not qualify the complete ranked worker.
   The next one-GPU probe (`cake-weave-model-activation-queue-f20e9d45/`)
   replaced dummy claims with the real Cake model-width SwiGLU row stage.
   Four cooperative cases each completed 8,192 activation CTA work units
   over four waves with bitwise BF16 oracle equality. At `c=147` and
   `c=148`, communication CTAs actually stole 4,096 and 8,192 stage units.
   Its input repeats one dense-dyadic tile and it still has no TMA/MMA stage,
   live expert-bin readiness or P2P handoff.
   The next one-GPU probe (`cake-weave-model-upgate-queue-afe75c51/`)
   executed the **real Cake up/gate TMA/MMA body** from host-encoded tensor
   maps stored in device global memory. Four cooperative cases each
   completed 1,536 N-subtile CTA work units (64 logical tiles × 24) over
   four waves with bitwise FP32 oracle equality. At `c=147` and `c=148`,
   communication CTAs stole 768 and 1,536 real up/gate subtasks. The
   persistent worker compiled with 79 registers/thread and no spills.
   Inputs still repeat one dense-dyadic tile; dynamic expert selection,
   activation/down completion and P2P return remain open.
   The corresponding one-GPU down probe (`cake-weave-model-down-queue-
   9c9dc158/`) completed 64 logical tiles × 32 N-subtile work units under
   the same four-wave controls, again with bitwise FP32 oracle equality.
   Its `c=147` and `c=148` cases actually stole 1,024 and 2,048 down tasks.
   A successor activation→down chain (`cake-weave-model-activation-down-
   chain-08b46b5c-r2/`) executes both real stages in **one cooperative
   launch**, with a grid barrier after each stage of each source wave and
   an async-proxy fence between activation's global BF16 writes and down's
   TMA reads. Each of four cases completed 8,192 activation and 2,048 down
   units, with both output tensors bitwise equal to the retained Cake FFN
   oracles. At `c=147`, communication CTAs stole 4,104 activation plus
   1,016 down units; at `c=148`, they stole all 10,240 units. The 192-thread
   worker compiled with 78 registers/thread and no spills, and CUDA queried
   one active CTA/SM on 148 SMs. The barrier is wave-wide, not a per-tile
   ready publication. The next one-GPU chain (`cake-weave-model-full-ffn-
   chain-3777c9a1/`) joins the Cake up/gate, activation and down bodies in
   **one cooperative launch**. Four cases each processed 1,536 / 8,192 /
   2,048 stage CTA units, with zero bit mismatches in all three saved stage
   tensors. At `c=147`, communication CTAs stole 803 / 4,069 / 1,016
   units respectively; at `c=148` they stole every unit. The worker compiled
   with 88 registers/thread and no spills and again queried one active
   CTA/SM. It uses successive wave-wide stage barriers, so per-tile
   readiness and overlap, dynamic expert choice and P2P traffic remain open.
4. **Complete host and measurement contract.** Preserve exact four-rank
   tensor placement, selected peer pairs, broker ownership, reset and
   all-rank statuses. Establish one qualified complete-layer four-device
   CUPTI/L2-reset interval before ranking `c/K/steal` or comparing against
   the pinned open baseline. The captured B300 helper currently returns
   `ENODEV`; the one-shot Nsight activity trace is development evidence only.

## Bounded implementation order

- Integrate the separate core schema-2 tile/stage effect after its review,
  then lower the evidenced host tile formation, padding, expert weight
  selection and **stage CTA completion counters** into a live B300 queue.
  Replace the validated wave-wide stage barriers with tile-local ready
  publication. Publish activation tasks only after all 24 up/gate subtasks for their
  logical tile complete, and down tasks only after all 128 activation
  subtasks complete; then publish the route-keyed return after all 32 down
  subtasks. Verify empty/tail bins and dynamic expert/weight descriptors
  before coupling the worker to the four-rank mailbox.
  Communication CTA steal must claim the same ready stage unit as computation
  CTAs; connect the admitted GPU combine to the live ranked return path.
  Exercise empty, highly skewed and tail experts, and replay small T7/T8
  counterexamples before a new Campaign. Move pre-launch domain checks off
  the critical path only with equally explicit admission/failure signals.
- Finally evaluate spatial `c`, temporal chunks and stealing under a common
  complete-layer timer and the upstream baseline's exact source/semantics.
  A cost estimate only filters candidates; on-device correctness and the
  qualified interval decide acceptance.

Promotion disposition: **no promotion** from the synthetic tensor-tile
compile. Existing small-worker source remains a correctness prototype until
the tile math, ranked effects and their analyses evolve together and pass the
new Workload, Corpus Gate and device evidence.
