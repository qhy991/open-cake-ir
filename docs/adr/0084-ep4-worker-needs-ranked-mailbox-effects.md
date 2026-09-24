# ADR 0084: EP4 workers need ranked mailbox effects

Status: proposed. This extends ADR 0083's worker-plan direction to the
four-rank MoE Workload; it does not admit a new Program schema or qualify a
distributed Cake executable. The current version-3 Program remains an exact
two-rank FP32 control slice.

## Evidence and the remaining gap

The frozen `weave-ep4-bf16-moe-b300-v1` Workload owns four-rank expert
placement, routed inputs, five routing distributions and an independent FP64
oracle. The direct CUDA/PTX development reference in
`experiments/weave/native_b300/ep4_mailbox.cu` passed balanced BF16 output
correctness on four B300-M4 GPUs and a skewed plan with 14 actual stolen
tiles. That source is outside Cake lowering, uses a small fixed geometry, has
no qualified latency, and does not implement dispatch deduplication or the
Workload's `T=7` tail case. It proves neither a Cake EP4 backend nor a
performance gain.
Separate unmeasured NVIDIA task successors at `fd222407` and `d772da55`
address origin-ranked `K` and then distinct remote payload/task queues in
direct CUDA. Prepared Workload oracle bundles and host checks do not
substitute for nvcc, four-GPU correctness or a Cake effect analysis.

Cake Program version 3 can name two ranks, one tensor owner per tensor, one
state owner and system-scope handoffs. Its two worker classes are assigned to
different ranks. EP4 needs **both** communication and computation CTA classes
on **each** of four ranks, a rank-local mailbox on each GPU, and dynamic
source/destination ranks selected by routing. Repeating the version-3 kernel
four times would leave remote ownership and the cross-rank queue protocol
outside the IR. The local expert math is available as three complete Cake
Schedules in `examples/programs/weave-local-expert-ffn-b300.json`; the NVIDIA
task branch at `e4cbd5b7` emits an ordered native CUDA counterpart. Neither
Program is a fused or distributed worker.
The NVIDIA task branch at `c822fb10` can emit inline device helpers from the
complete local expert and origin combine Schedules under an explicit
`ranked_mailbox` rewrite request. Those helpers provide mathematical source
maps but no reservation, peer ownership, readiness or launch protocol.

The shared `evaluation/ranked_launch.py` development adapter now checks one
rank-local input set and `c/K/steal` plan per rank, exact storage owners and
nonaliasing, the compiled mailbox size and its BF16 output view, and stable
execution contexts. Each invocation requires an Evaluation-owned reset and
one combined launch followed by synchronized statuses from every rank.
Platform callbacks still own CUDA, peer admission and the actual reset;
this adapter does not seal a distributed candidate or qualify measurement.

`evaluation/ranked_manifest.py` is a separate pre-seal boundary: it binds
one frozen distributed Workload case, the complete effect/local/combine math,
exact Compiler lowering, rank/expert-sharded tensor ABI and all four runtime
plans. It refuses wrong placement, source identity, grid or controls. The
existing single-device manifest and `LaunchableCandidate` remain closed to
this form until compilation and a distributed loader can seal and replay it.

## Proposed Program boundary

Keep one complete Program as the mathematical authority. Its rank-leading
tensor shapes and stage bindings must state which local shard a stage sees;
the binding is a concrete contiguous slice, not a layout algebra. Each rank
replicates the same two CTA classes, with its own device-resident `c_r`,
`K_r` and bounded steal allowance. The exact Target and each rank's compiled
residency determine whether one cooperative CTA per SM can launch. The
three-stage local expert Program supplies projection and activation math to
the compute phase; the Compiler must not replace it with an opaque MoE
instruction or paste the direct-reference body as unexamined source.

The worker execution descriptor needs these **effects**, each tied to named
buffers and operations:

1. **Remote payload reservation and route enqueue.** A source rank derives
   destination from each checked expert ID. All routes of one `(source rank,
   token)` to the same remote destination share **one** BF16 payload slot;
   each route still reserves its own compute-task slot with expert and route
   metadata referring to that payload. A local route reads the source's
   hidden tensor without a remote payload reservation. Both reservations
   return old indices checked against their distinct capacities. Payload
   bytes and task metadata each have a ready flag published with
   system-scope release after their writes.
2. **Inbound claim and local expert compute.** Regular compute CTAs and
   communication CTAs during the steal window claim from the *same*
   rank-local task head. Each claim owns one ready task, acquires its
   metadata and (for remote work) the referenced payload's ready flag,
   then runs the complete local expert math with that task's local expert
   coordinate. Two tasks may read one payload, but neither task can execute
   twice.
3. **Return contribution.** The compute owner writes its FP32 contribution to
   the source rank's `(token, route)` slot and publishes a system-scope ready
   flag. The origin's combine phase may read it only after acquire. A
   chunk-completion counter can signal progress, but a relaxed increment
   alone is not a payload publication.
4. **Bounded combine.** The source rank combines exactly its declared routes
   with their weights and rounds its output to BF16. `K_r` partitions its
   token range into complete chunks, including a tail; no chunk may omit or
   duplicate a token. Completion may overlap later compute, subject to the
   per-contribution ready flags. For `T=7, K_r=2`, the two completion
   thresholds are 8 and 6 contributions, not a reused uniform counter.

The first Compiler slice is `compiler/ir/ranked_mailbox.py`:
`RankedMailboxEffects` declares the three owners, reservation modes, keys,
system-scope ready handoffs, rank-local controls, shared task-queue steal
window and per-launch reset. Its analysis derives separate payload, task and
return capacities from a complete local Program and complete combine
Schedule. The ordinary Program launcher is unchanged; a ranked lowering
without a dedicated backend refuses explicitly. This typed effect object
does not yet prove remote memory visibility or emit a four-rank kernel.
The NVIDIA task at `5b0af054` is a development successor that emits a
four-rank source from this effect object and the complete Cake math. It
declares the additional B300 PTX system-atomic contracts, checks pointer
owners and input ID/weight bounds in its host ABI, and probes selected
directed P2P pairs at launch. Its source and runner remain unqualified until
nvcc, all required device cases and formal Evaluation pass.

The mailbox state and every tensor shard have one declared owning rank. A
remote pointer carries that owner through lowering and Evaluation. Admission
checks the exact directed peer pair for peer access and native P2P atomics
before enabling access; an observation on another pair is not transferable.
Rank-local cursors use GPU-scope atomics where no peer reads or writes them.
Remote reservations, completion and publication use the Target's declared
system-scope PTX forms. This ownership classification belongs to the effect
analysis, not an emitter's guess from pointer spelling.

## Static checks before native emission

- The Workload's `R`, `T`, top-k, expert count and weight geometry bind all
  shard extents. In the present EP4 Workload, one destination can receive
  at most `(R-1)*T` **remote payloads** and `R*T*top_k` **compute tasks**;
  the `T=8` skew case reaches 24 and 64 respectively, while `T=7` reaches
  21 and 56. These capacities are derived from Workload routing and must
  not be inferred from the direct reference's one-slot-per-route mailbox.
  `experiments/weave/dispatch_ledger.py` projects the two domains without
  claiming a Cake effect or GPU implementation.
- `experiments/weave/rank_plan.py` derives rank-local CTA populations and
  per-chunk completion thresholds from that ledger. It accepts uneven tail
  chunks and refuses zero-worker classes, invalid `K_r`, over-budget stealing
  and incomplete return domains. The measured direct CUDA source still
  requires `T % K_r == 0`. The NVIDIA task's separate `T=7` successor has
  host-tested chunk arithmetic but no nvcc or device oracle result; the CPU
  plan does not make either kernel tail-qualified. The measured runs shared
  one `K` across ranks and their return path used the compute rank's `K`
  to update an origin counter. A further unmeasured successor at `fd222407`
  passes the four origin chunk counts explicitly and prepares
  `K=(2,3,7,1)` as the device counterexample.
- `experiments/weave/ep4_event_model.py` explores bounded fair orders of
  dispatch, regular compute, local CTA stealing and combine. It checks unique
  route completion, publication-before-consumption in its logical event order,
  steal bounds and early chunk combine. It does not model GPU memory
  visibility, compiled occupancy, true CTA residency or elapsed time, so a
  green simulation cannot discharge the device liveness obligation below.
- Every payload, metadata slot, ready flag, contribution and public output
  has one writer and an owning rank. The reserved slot and returned source
  coordinate determine the only legal write address. Invalid expert IDs,
  negative reservations or overflow refuse before a dereference; no target
  fallback or silent clamp is permitted.
- The phase graph has a producer path for every wait. Each rank must reserve
  at least one communication and one computation CTA (`0 < c_r < N_r`), and
  each producer must be resident or able to progress while consumers wait.
  All source ranks' dispatch completion is required before an empty inbound
  queue can be declared terminal. A cooperative launch fact by itself does
  not prove this liveness condition.
- Release follows all payload-writing lanes' rendezvous; acquire dominates
  every payload read. Queue claims may be relaxed returned-old-value atomics
  because uniqueness and publication are distinct obligations. Cross-rank
  flags cannot be weakened to device scope.
- State reset, chosen timer and measured interval remain Evaluation-owned
  facts. A cost model may choose `c_r/K_r` candidates before GPU time; it
  cannot accept one. A Compiler change after a Campaign is a successor
  commit, not a mutation of the frozen Run.

Counterexamples must include an overfull skewed payload or task queue, a
duplicate remote payload, a missing route task, wrong origin rank, an
acquire omitted before the first payload or task-metadata
read, a relaxed publication flag, a zero-worker class, a future-tile wait
cycle, invalid `K_r` or tail partition, a missing peer pair and a compiled
occupancy shortfall. The owning rule must refuse each one; an unrelated
verifier block is not evidence for it.

## Verification and performance decision

Cake P1/P3 preserve the ordinary mathematical editing model and one
canonical mailbox effect document; capacities are derived rather than
declared twice. P2 exposes rank-local `c/K/steal`, queue owners and scope.
P4/P5 require construction-time key/owner/shape checks and capacity analysis,
followed by the separate liveness and peer-pair gates above. P6 gates the
new effect through the kernel corpus and adversarial examples. P7 places
syntax beside analysis in the first slice and requires a later backend tick
before executable admission. P8 keeps system-scope PTX and cooperative launch
grounded in the exact Target and actual selected peer pairs.

The executable successor must bind the typed effect to a complete ranked
Program, native lowering and counterexamples together (Cake P1–P8).
First run the full
Compiler Corpus Gate at a fixed commit. Then, under the B300-M4 broker and
at most four GPUs, require nvcc compilation, all five Workload cases against
the external oracle, input preservation, queue/counter evidence, and a skew
case with actual stealing. Tail, local-only and remote-only routing are not
optional. Seal a distributed candidate before formal Evaluation. CUPTI
intervals with the target's L2 reset and profiler evidence then measure
per-rank overlap and whole-MoE time.

The current NVIDIA results have no qualified Triton MoE latency, so Triton
headroom remains unknown. Compare native and Triton only under the same
Workload, shape, routing, oracle, reset and timer. Native PTX is promoted for
measured gains or a demonstrated Triton expressibility limit, with its exact
Target contracts and source retained; no offline estimate establishes that
choice.
