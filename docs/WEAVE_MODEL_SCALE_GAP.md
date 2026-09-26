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
   A tile-ready successor (`cake-weave-model-tile-ready-capped-449c19b8/`)
   replaces those stage barriers with per-tile GPU-scope release/acquire
   completion: 24 up/gate subtasks publish 128 activation rows, which
   publish 32 down subtasks. Two separate one-GPU runs at the same source
   commit each completed all 11,776 units in all four `c`/steal cases,
   matched all three saved stage tensors bitwise and observed successor
   claims while other tiles' predecessor tasks remained unfinished.
   The `c=148` case stole every unit in both runs. An earlier fetch-add/
   subtract permit variant (`cake-weave-model-tile-ready-ffn-f1b6fdfe/`)
   passed three cases but timed out at `c=148`; replacing its reservation
   with bounded CAS made that exact case complete twice. This does not
   prove arbitrary interleavings safe, and neither version includes
   dynamic expert bins, P2P communication or qualified latency.
   The next backend tick (`native_cuda_tile_stage.py`, source `a8cd20f5`)
   removes the remaining hand-written FFN math from the worker template.
   It reuses native CUDA TMA/MMA lowering for up/gate and down and the model
   activation emitter for SwiGLU/BF16 cast, with operation IDs mapped back
   to the complete Cake Program. Six related tests and the 179-case Corpus
   Gate passed at that fixed commit. On B300-M4, one brokered GPU run of
   the generated source passed all four 11,776-unit cases bitwise, including
   full `c=148` stealing. The generated worker used 96 registers/thread,
   versus 84 in the hand-composed version, with the same one-CTA/SM
   shared-memory occupancy; no qualified timing compares them. This is
   offline stage composition, not yet schema-2 ranked-tile backend lowering.
   A successor store-lowering tick (`6c70bb37`) proves full M128×N64 output
   ownership from the exact Schedule axes and AccessMap, traps an invalid
   N-subtile, then omits the per-element store mask inside the worker.
   PTXAS register use fell from 96 to 88 per thread without spills; the
   same one-GPU four-case oracle, completion and steal checks passed.
   Occupancy remains one CTA/SM, and no latency or speedup claim follows.
   A further one-GPU step (`cake-weave-tile-ready-two-expert-9a0d6eeb/`)
   selects separate up/gate and down TMA weight descriptors by each tile's
   expert ID. Sixty-four tiles alternate two distinct input/weight sets with
   independent saved stage oracles. All four c/steal cases again completed
   11,776 work units with bitwise agreement at every stage; `c=148` stole
   every unit. IDs were prepacked before launch, so this validates selected
   weight mapping, not live expert-bin formation or four-rank routing.
   A separate four-GPU transport probe (`cake-weave-ep4-p2p-bin-a14d758a/`)
   routes the exact 16,384 synthetic model route rows into destination-owned
   expert bins using system-scope peer atomic row reservations and release
   flags. Its independent post-lease oracle verified every return key,
   destination expert and BF16 row bitwise: 4,039 / 4,196 / 4,016 / 4,133
   rows by owner, including 12,271 remote-owner routes. At a 64-row partial
   threshold, a clean temporary schema-2 integration check accepts the same
   route IDs as 64 logical tiles and 11,776 stage tasks per owner. These are
   still **separate** transport, CPU publication and FFN worker results;
   no GPU tile consumer has yet acquired the bin flags or returned a full
   EP4 layer contribution.
   A follow-on bridge (`cake-weave-ep4-worker-bridge-014d1f55/`) consumes
   those actual four-rank P2P bin rows. The host first checks every route key
   and BF16 row, then materializes the schema-2 threshold-64 plan into 64
   padded M128 tiles per owner. Four B300 workers use the Cake-generated
   three-stage FFN and 32 rank-local expert weight descriptors. Under
   `c=147`, each processed 11,776 stage CTA units and stole 5,888. All
   16,384 route contributions matched the earlier Cake bridge bitwise;
   CPU weighted combine had 0 / 4,194,304 elements outside the independent
   oracle's unchanged 0.01/0.01 tolerance (maximum absolute error 0.0078125).
   The peer bin and worker were separate jobs; tile formation, route-key
   gathering and final combine ran on the host. Live GPU tile publication,
   rank return, overlap and qualified latency remain unverified.
   A subsequent four-GPU return probe (`cake-weave-ep4-return-combine-
   5cc8b83a/`) writes those down outputs to origin-rank contribution slots
   by deterministic route key, publishes a system-scope release flag, and
   has each origin GPU acquire all 4,096 ready slots before its Cake-lowered
   T512/top-8 combine. All 16,384 returned FP32 contributions and the
   4,194,304 GPU BF16 output elements matched the prior Cake results
   bitwise; the independent FP64 CPU oracle again had zero elements beyond
   the fixed 0.01/0.01 tolerance. The bin, FFN and return/combine remain
   **separate jobs with host tile formation**. A single live ranked worker,
   full communication-compute overlap and qualified layer timing remain open.
   The next four-GPU transport successor (`cake-weave-ep4-gpu-tile-gather-
   8346d6e0/`) records route-to-bin locations during peer dispatch and
   publishes a per-route system release flag. Destination GPU gather CTAs
   acquire the flags and form all 64 M128 BF16 tiles per owner; every GPU
   tile tensor matched the prior checked host materialization bitwise,
   while the 16,384-route bin oracle still passed. Dispatch kernels were
   submitted before gather kernels without a host device sync between them.
   The tile-key plan remains CPU supplied, and the FFN/return kernels were
   not in this same allocation; overlap and latency were not measured.
   One-allocation development successor (`cake-weave-ep4-live-chain-
   46a4b844/`) now submits peer dispatch, GPU tile gather, the three-stage
   Cake worker, P2P return, GPU acquire and Cake T512 combine for all four
   B300 ranks before the first host device-wide sync. At `c=147`, each rank
   completed 64 logical tiles, 11,776 stage units and 5,888 real steals.
   All 16,384 FP32 returned route contributions and 4,194,304 final BF16
   outputs matched the prior Cake results bitwise; zero outputs exceeded the
   independent CPU oracle's 0.01/0.01 tolerance. The **CPU still supplies**
   the threshold-64 tile-key plan and stage-task waves. Per-rank streams
   order local phases, and no qualified profiler/timer establishes actual
   communication-compute overlap or a speedup. The runtime tile publisher
   and variable plan admission are the next Compiler/backend boundary.
   A separate Nsight CUDA-activity replay (`cake-weave-ep4-live-nsys-
   2b8f97e7/`) passed the same complete oracle and retained six kernel
   phases on each rank. In that single profiled run, cross-rank dispatch and
   gather activity intersected for four rank pairs, gather and FFN did not
   intersect, and FFN/return activity intersected for six pairs. The FFN
   kernel under `c=147`, budget 5,888 occupied about 218–220 ms of each
   rank's profiled timeline; earlier ranks subsequently waited for return
   flags from the last rank. Nsight profiling, one sample and no L2 reset
   make these diagnostic intervals unsuitable for a latency/speedup claim.
   A controlled spatial development sweep (`cake-weave-ep4-spatial-sweep-
   4900b162/`) then held the model, routes, tile plan, Cake kernels and
   source commit fixed across `(c,budget)=(1,0),(74,5888),(147,5888),
   (148,11776)`. All four passed the full oracle. Actual steals at `c=74`
   were 4,654–5,063, demonstrating that budget is a cap, not a target; an
   earlier host check that required equality was corrected in `4900b162`.
   A structural bound on non-steal work per ordinary compute CTA is about
   81, 80, 5,888 and 0 work units respectively. The four single-run Nsight
   FFN activity medians were 2.711, 2.862, 218.787 and 3.875 ms. A further
   `c=147` run changed only the budget to 11,776; its generated CUDA source
   was byte-equal to the half-budget case, actual steals rose to 11,643–
   11,662 and the single-run FFN activity median was 3.810 ms, again with a
   passing oracle. This diagnoses steal-cap tail work on the tested plan;
   it is **not** a qualified speedup or general plan ranking. Promotion
   disposition: no Compiler pass or automatic budget rule yet. The exact
   residual-work bound belongs in a future Lab pre-GPU filter once the
   complete-layer timer and broader routing confirm its decision value.
   A subsequent temporal-binding audit found the retained threshold-64 plan
   has rank-specific early tile counts: wave 1/2 = 17/15, 20/12, 14/18 and
   19/13. The earlier complete-chain runs used 17/15 for every rank. Their
   arithmetic and spatial steal observations remain valid because all tiles
   were prepared before the FFN launch, but they do not validate rank 1–3's
   declared publication waves. Source `cd2f5a11` derives stage-task counts
   from each rank's materialized plan; one four-GPU `c=74` replay passed
   those exact wave counts and the unchanged full oracle. Its plan and wave
   counts are still CPU supplied, so GPU runtime tile publication and a
   meaningful temporal-`K` sweep remain open.
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
  The one-GPU tile-ready prototype has validated the 24→128→32 predecessor
  chain on repeated fixed tiles; connect its completion to the real
  expert-bin publication and publish the route-keyed return after all 32 down
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
