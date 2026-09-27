# B300 persistent event pipeline: bounded first step

Code commit `23167cbb` changes the exact native CUDA ranked-tile lowering from
one 96-CTA cooperative grid **per event** to one such grid **per rank and
layer**. The frozen four-rank BF16 Workload, Cake FFN bodies, source-event
ordering, `c/K/steal` controls, tile capacities, return/combine path and
pointer ABI v4 remain the same. The host resets all state before launching
one grid on each rank's stream. Each grid walks the `5*K` events, then releases
its CTA-owned tensor memory. The existing return and combine kernels follow
on the same rank-local stream.

## Current event protocol

For each event, the source rank waits for the preceding event's tile snapshot,
the first `c` CTAs dispatch into peer bins, and a system-scope release
publishes source completion. Each destination derives its expert snapshot,
assigns tiles and stage tasks, gathers tile rows, and publishes
`wave_consumed[event]`. Every consumer crosses the generic-to-async proxy
fence before Cake's TMA reads. Workers claim finite per-tile up/gate,
activation and down work; bounded stealing counts against the rank-local
budget. A grid barrier completes all FFN work before that rank enters the next
event. The old cross-stream planner dependency is absent: all producers and
consumers are CTAs of a resident cooperative grid on the same GPU.

The wait graph advances by event index. A source's prior-snapshot wait reads
event `e-1`; a destination's source-completion wait reads the same event's
source after its dispatch; the next event starts only after the current
grid-wide FFN completion. The host still checks exact B300 identity, peer
reachability and cooperative occupancy before launch and checks all ranks'
status afterward. A bounded hot8 K=2 device replay is described below;
broader correctness and arbitrary cross-rank liveness remain open.

## Reset ordering correction and bounded replay

The earlier per-event source `57bc00ec` timed out in broker job
`gpuq-18e11319d003` after a complete first hot8 K=2 launch: its next launch
never returned before the 12-minute run limit. The retained worker log has
the first launch's 40 rank/event phase records, but no accepted second output.
This does not by itself identify the stalled instruction.

The host reset boundary had a separate, concrete ordering defect:
`ranked_tile_reset` issued device `cudaMemset` operations through the default
stream, while every rank's worker stream was created with
`cudaStreamNonBlocking`. NVIDIA documents that [device `cudaMemset` may return
before it completes](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html)
and that [nonblocking streams do not synchronize with the legacy default
stream](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html).
A source rank could therefore publish into a peer Bin while that peer's reset
was still outstanding. Successor `3e1ecb94` enqueues each reset in its own
compute stream and synchronizes all four streams before launching any grid.
The fix changes neither the frozen Workload nor its oracle.

At `3e1ecb94`, 15 related contracts and the 179-case Corpus Gate pass; B300
NVCC reports 93 registers, 9,216 B static shared memory and zero spills for
the persistent worker. Isolated broker job `gpuq-8fea3de7b6ba` replayed the
same hot8 K=2, c=64, budget=46920 case with two warmups and three CUPTI
samples. All five complete-layer launches returned, the final 4,194,304
outputs had zero FP64-oracle failures and zero bit differences from the
retained v3 output, all 80 tile-event counts matched the CPU plan, and all
200 expected phase records were present. Every CUPTI sample had four reset
kernels, 16 layer kernels and 92 layer memsets. The report is retained at
`/home/qinhaiyan/cake-weave-peer-reset-3e1ecb94-hot8-k2/report.json`.
This bounded replay supports the reset correction; it does not prove the old
timeout's sole cause, arbitrary cross-rank liveness or closure of the open
c=95 route-location Finding.

## Required next transition

This first step is **sequential across events within each rank**. It removes
repeated cooperative launches and allows CTA-owned resources to persist, but
does not yet let communication for event `e+1` overlap Cake FFN work for event
`e` on the same rank. The full temporal design needs a bounded producer and
consumer state machine within this grid: communication CTAs may start the
next source event once the previous tile snapshot is gathered, while compute
CTAs finish already published stage tasks. Tile storage and task counters are
indexed by event; admission must show that every waited-on producer CTA is
resident and can make progress. The cross-CTA generic-store to async-TMA
handoff needs its own reviewed proxy-ordering rule and counterexamples before
removing the current grid barrier. No dependent planner kernel in another
stream may be required for progress.

Before promotion, compile the clean commit with B300 NVCC, replay the four
frozen routes at K=1/2/4 against the external FP64 oracle and CPU tile-event
planner, check c/steal controls including the open c=95 mismatch Finding,
retain profiler evidence, and qualify the target's four-device timing reset
and matched open baseline. This task branch is a development successor, not a
claim of Weave speedup or full temporal overlap.
