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

1. **Remote dispatch reservation.** A source rank derives destination from a
   checked expert ID, atomically reserves one slot in that destination's
   mailbox, writes source/token/route/expert metadata and BF16 payload, then
   publishes that slot's ready flag with system-scope release. Reservation
   returns the old slot index; a range check precedes every write.
2. **Inbound claim and local expert compute.** Regular compute CTAs and
   communication CTAs during the steal window claim from the *same*
   rank-local head. Each claim owns one ready slot, consumes payload only
   after system-scope acquire, and runs the complete local expert math with
   that slot's local expert coordinate. A claim cannot execute twice.
3. **Return contribution.** The compute owner writes its FP32 contribution to
   the source rank's `(token, route)` slot and publishes a system-scope ready
   flag. The origin's combine phase may read it only after acquire. A
   chunk-completion counter can signal progress, but a relaxed increment
   alone is not a payload publication.
4. **Bounded combine.** The source rank combines exactly its declared routes
   with their weights and rounds its output to BF16. `K_r` partitions its
   token range into complete chunks, including a tail; no chunk may omit or
   duplicate a token. Completion may overlap later compute, subject to the
   per-contribution ready flags.

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
  shard extents. The worst-case inbound capacity is derived from routed
  work across **all** sources; skew to one rank must not overflow its
  mailbox. Deduplication is a separate declared semantic relation, not a
  capacity optimization silently assumed from the direct reference.
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

Counterexamples must include an overfull skewed mailbox, duplicate route
reservation, wrong origin rank, an acquire omitted before the first payload
read, a relaxed publication flag, a zero-worker class, a future-tile wait
cycle, invalid `K_r` or tail partition, a missing peer pair and a compiled
occupancy shortfall. The owning rule must refuse each one; an unrelated
verifier block is not evidence for it.

## Verification and performance decision

The implementation tick must land Program syntax, effects, legality,
lowering and counterexamples together (Cake P1–P8). First run the full
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
