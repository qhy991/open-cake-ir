# Weave model-scale Cake gap on B300

This is an implementation audit, not a performance result or a new Workload
Contract. The separate open-baseline task owns its provisional Qwen3-30B
geometry and CPU oracle. The frozen `weave-ep4-bf16-moe-b300-v1` Cake
development Workload remains T7/T8, E8, top-2, H16, I32; its passing runs
do not imply model-scale throughput.

## The two admitted domains today

| Mechanism | Current Cake/B300 evidence | Model-scale gap |
| --- | --- | --- |
| Ranked payload/task/return effects | `RankedMailboxEffects` types rank owners, system release/acquire, capacities, `c/K/steal`; the B300 source passed five small cases and replayed three former liveness failures after the warp-uniform fix. | Effect key is one `(source,item,route)` compute task; no tensor-core tile task, expert bin, tile completion or multi-warp role transition. |
| Inline expert math | `native_cuda_ep_math.py` verifies a three-stage Program and combine Schedule, then emits a one-warp SIMT BF16 row-dot / FP32 activation/down body. | Admission fixes H16, I32, two local experts and T7/T8. Increasing constants would leave one-token/route SIMT work and not implement grouped tensor-core GEMM. |
| Tensor-core local FFN | A complete native CUDA TMA/`tcgen05` GEMM Schedule exists. Separate no-bias model-width up/gate (`cc1715b5`) and down (`de0c4824`) tiles each passed two full one-GPU FP32 comparisons bitwise against an independent FP64 oracle. A model-width SwiGLU Schedule (`cb72263a`) explicitly casts FP32 to BF16; a three-stage Program (`98431873`) binds up/gate → SwiGLU → down without an implicit cast. CUDA 13.1 compiled all three Program stages on B300-M4 with 74/24/74 registers per thread and no spills. | The ordered Program is one fixed expert tile. Its full device oracle is pending. There is no routed token bin, rank placement, tile-keyed mailbox or transition from the current 32-thread ranked CTA to the 192-thread tensor-core worker. No EP4 or performance result follows. |

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
products are in `cake-weave-model-local-ffn-98431873/`; they do not yet
establish device correctness for the full Program.

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

## Required joint change

1. **Math and mapping.** Build complete Cake Schedules for an expert up/gate
   tile and down tile at the model's BF16/FP32 accumulation contract. Reuse
   the existing TMA/`tcgen05` native emitter where its concrete SMEM views,
   TMEM columns and barrier lifetimes apply. Admit dynamic local expert
   selection and token bins with explicit byte offsets/strides, without a
   first-class layout algebra or an opaque MoE opcode. Verify each tile
   against the independent oracle before embedding it in EP4.
2. **Task/effect ownership.** A tile rather than a token-route becomes the
   compute claim unit. Derive destination queue capacity, source return
   ownership and chunk-completion thresholds from that complete Program and
   its tile decomposition. A route cannot publish twice or disappear when
   tiles group tokens. The verifier must cover partial bins, empty experts,
   tail tokens, duplicate destination payloads, aliasing, release/acquire
   and warp-uniform loop exits.
3. **CTA resource transition.** Current ranked workers launch 32 threads per
   CTA. The admitted tensor tile uses six warp roles in a 192-thread CTA
   (four epilogue, one MMA, one copy), plus SMEM and TMEM resources. A
   communication CTA that later steals a tile must own all resources and
   synchronization edges needed for that transition. The spatial `c` budget
   must be checked against the actual resident 192-thread cooperative grid,
   not inherited from the small SIMT worker's `SMS=148` grid assumption.
4. **Complete host and measurement contract.** Preserve exact four-rank
   tensor placement, selected peer pairs, broker ownership, reset and
   all-rank statuses. Establish one qualified complete-layer four-device
   CUPTI/L2-reset interval before ranking `c/K/steal` or comparing against
   the pinned open baseline. The captured B300 helper currently returns
   `ENODEV`; the one-shot Nsight activity trace is development evidence only.

## Bounded implementation order

- Validate the ordered local expert FFN Program against its independent
  stage-by-stage oracle on B300, then add dynamic expert-owned token bins and
  test varied expert loads. Retain nvcc/PTXAS, oracle and profiler evidence.
  The current fixed expert tile and dyadic projection cases are correctness
  seeds, not a routed model-scale result.
- Then add tile-keyed ranked effects, capacity/liveness analyses and native
  emission in a successor Compiler commit; replay small T7/T8 counterexamples
  and the separate model-scale Workload before a new Campaign.
- Finally evaluate spatial `c`, temporal chunks and stealing under a common
  complete-layer timer and the upstream baseline's exact source/semantics.
  A cost estimate only filters candidates; on-device correctness and the
  qualified interval decide acceptance.

Promotion disposition: **no promotion** from the synthetic tensor-tile
compile. Existing small-worker source remains a correctness prototype until
the tile math, ranked effects and their analyses evolve together and pass the
new Workload, Corpus Gate and device evidence.
