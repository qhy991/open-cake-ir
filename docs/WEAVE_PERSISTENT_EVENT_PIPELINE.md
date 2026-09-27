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

Job `gpuq-7ba6ab50e85e` then ran 12 separately sealed controls at
`3e1ecb94`: four frozen routes × K=4/2/1, uniformly c=64 and budget 46920.
The post-lease audit found zero FP64-oracle failures across 50,331,648 output
elements, exact CPU/GPU agreement for all 960 rank/event tile-count slots,
zero output bit differences from the retained v3 controls, and no steal-cap
or P2P payload discrepancy. The job completed and released its four-GPU
lease; its report is retained at
`/home/qinhaiyan/cake-weave-peer-reset-3e1ecb94-matrix12/report.json`.
This establishes the sequential persistent lifecycle for those twelve
frozen configurations. The c=95 Finding, boundary/tail guards outside those
cases, same-rank cross-event overlap and qualified timing remain open.

For a matched launch-structure comparison, the per-event path received the
same reset correction at `cbb68331`; its 15 related contracts, 179-case Corpus
Gate and exact B300 NVCC build passed. Broker job `gpuq-a336552fe1d2` then
ran old/new/new/old under one four-GPU lease on the mixed K=4, c=64,
budget=46920 case. Every arm passed the FP64 oracle, CPU tile plan, v3 bitwise
output and CUPTI reset/activity checks. The per-event source emitted 92 layer
kernels per sample; the persistent source emitted 16. Their four CUPTI
development medians were 9.97/10.09/10.06/9.82 ms, yielding old/new
aggregate medians of 9.89/10.08 ms. Fewer kernel launches alone showed no
gain under this assay; the result is not qualified latency. The retained
report is `/home/qinhaiyan/cake-weave-persistent-vs-event-reset-abba-r3/report.json`.
Promotion disposition: **no merge or Lab schedule rule** from the sequential
persistent scaffold. Same-rank cross-event communication/FFN overlap is the
next performance mechanism to implement and validate.

## One-event source lookahead experiment

Successor `1aff5d01` adds a bounded first overlap step. After the grid has
gathered and published event `e`'s tile snapshot, communication CTAs owned by
the next source rank may dispatch event `e+1` into peer bins while computation
CTAs work on event `e`'s immutable tile rows. Each source CTA waits for the
previous snapshot's peer flags before dispatch; all source writers issue a
system fence. The end-of-event grid barrier waits for both the prefetched
dispatch and current FFN before block 0 publishes the next source completion.
Thus the changed interval overlaps **source dispatch with FFN**, while tile
planning and row gather for the next event still wait. No dependent kernel in
another stream is needed for progress.

The lookahead path records, per rank and event, whether completed Cake stage
tasks increased while the source CTA performed the next dispatch. A positive
flag is an actual-work overlap observation; an empty flag is inconclusive
about latency. The experiment keeps the same source/plan seal, FP64 oracle,
CPU tile-event planner and four-GPU broker boundary. `1aff5d01` passes 15
related contracts, the 179-case Corpus Gate and real B300 NVCC (95 registers,
9,216 B static shared memory, zero spills). Isolated four-GPU broker job
`gpuq-3439e7bcfa6c` completed the first mixed K=4, c=64, budget=46920
lookahead case. Its post-lease audit found zero FP64-oracle failures over
4,194,304 output elements, exact agreement for all 80 rank/event tile-count
slots and zero bit differences from v3. All 80 rank/event overlap records were
retained. Rank 0 at event 14 recorded a positive flag: its completed Cake
stage-task count increased while its communication CTAs dispatched the next
source event. This is direct evidence of useful-work overlap in **one** frozen
complete-FFN case, not a general liveness result or qualified latency gain.
The output and overlap reports are retained at
`/home/qinhaiyan/cake-weave-lookahead-1aff5d01/` and mirrored locally.
Broker job `gpuq-34381e97168b` extended the same source to four frozen
routes × K=4/2/1. The post-lease audit found zero FP64-oracle failures over
50,331,648 output elements, exact CPU/GPU agreement for 960 tile-event
slots, and zero bit differences from the corresponding v3 outputs. All phase
and overlap records were retained. Completed Cake tasks advanced during
next-source dispatch in fanin K=4, fanin K=2 and mixed K=4; the other nine
controls had no positive overlap flag in this run. The report is retained at
`/home/qinhaiyan/cake-weave-lookahead-1aff5d01-matrix12/` and mirrored
locally. This establishes bounded useful-work overlap on three frozen
controls, not a general performance gain. Before promotion, check further
c/steal controls and compare against the equal-reset persistent control
under the Target's qualified timing contract.

Same-lease ABBA job `gpuq-465099593875` compared the equal-reset sequential
persistent source (`3e1ecb94`) with lookahead (`1aff5d01`) on the mixed K=4,
c=64, budget=46920 control. All four arms passed the FP64 oracle, CPU tile
plan, v3 bitwise comparison and CUPTI reset/activity checks; each sample had
16 layer kernels and 92 layer memsets. Old/new/new/old development medians
were 10.01/10.03/10.07/9.81 ms, giving aggregate old/new medians of
9.91/10.05 ms. The retained report is
`/home/qinhaiyan/cake-weave-lookahead-vs-persistent-abba/report.json`.
Useful-work overlap in the three matrix controls did **not** yield a latency
gain in this matched mixed-route assay. The lookahead still defers the next
source-completion publication until the current FFN finishes, so next-event
planning and gather do not overlap. Promotion disposition: **no merge or Lab
rule** from this bounded lookahead result.

## Required next transition

The original one-grid step was **sequential across events within each rank**.
The bounded lookahead successor overlaps next-event source dispatch with
current FFN, but waits before planning and gathering the next event. The full
temporal design still needs a bounded producer and consumer state machine
within this grid so communication CTAs can plan and publish future tile work
while compute CTAs finish earlier stages. Tile storage and task counters are
indexed by event; admission must show that every waited-on producer CTA is
resident and can make progress. The cross-CTA generic-store to async-TMA
handoff needs its own reviewed proxy-ordering rule and counterexamples before
removing the current grid barrier. No dependent planner kernel in another
stream may be required for progress.

A concrete successor can split the resident grid by CTA role. The first `c`
communication CTAs advance a producer loop across events: dispatch, wait for
the source publication, snapshot and sort experts, assign tiles and gather
rows, then release-publish each event's tile-ready flag. They need a bounded
barrier **among those `c` CTAs** at each producer phase; a whole-grid barrier
would again wait for the current FFN. The remaining `96-c` CTAs acquire
tile-ready events and claim complete Cake stage work units from the existing
per-tile queues. After all events are produced, communication CTAs can borrow
remaining work under the same steal budget. All CTAs join only at final
completion before TMEM release and return/combine. The producer for event
`e` may wait for the `e-1` snapshot or the same event's source; it must never
wait for an unscheduled CTA or for consumer completion at event `e`. This
wait graph, per-event capacity and cross-CTA proxy handoff need explicit
Compiler admission/analysis and B300 counterexamples before launch.

## Dual-role grid development result

Code commit `b360415f` implements this producer/consumer split in the B300
native CUDA lowering. Six per-event barriers involve only the `c`
communication CTAs; the other `96-c` CTAs acquire each published event and
execute the existing complete Cake stage-task queue. Communication CTAs join
that queue after producing all events and remain subject to the same steal
budget. The producer fences generic tile-row stores before release-publishing
tile readiness; consumers acquire readiness and issue an async-proxy fence
before Cake TMA reads. The final grid barrier waits for every CTA before TMEM
release. This is a development memory-ordering implementation, still requiring
counterexamples and broader device verification.

The clean commit passes 15 related contracts, the 179-case Corpus Gate and
B300 NVCC (84 registers, 9,216 B static shared memory, zero spills). Broker
job `gpuq-330dbd2e9627` completed the frozen mixed K=4, c=64,
budget=46920 case on four GPUs. Its post-lease audit found 0/4,194,304
FP64-oracle failures, exact agreement for all 80 rank/event tile counts,
zero bit differences from v3, matching P2P payload counts and actual stolen
work [5426, 7129, 7305, 7293] by rank. All 80 overlap records were present;
four recorded Cake stage-task completions during the **next event's dispatch,
planning and gather**: rank 0 at completed event 0, and ranks 1–3 at
completed event 14. Reports are retained at
`/home/qinhaiyan/cake-weave-dual-role-b360415f/` and mirrored locally.
This proves one complete-layer execution and useful-work overlap on this
frozen route. It does not prove all c/K domains, arbitrary liveness, the open
c=95 case, qualified latency, or a gain against the equal-reset sequential
control. Promotion disposition remains **no merge or Lab rule** pending those
gates.

Broker job `gpuq-6a9bd1a3b62a` extended `b360415f` to four frozen routes
× K=4/2/1 at c=64 and budget 46920. All 12 separately sealed controls
passed: 0/50,331,648 FP64-oracle failures, exact agreement for 960 CPU/GPU
tile-event count slots, zero output bit differences from v3, and no payload or
steal-cap discrepancies. Every control retained positive same-rank evidence
that Cake stage work advanced while communication CTAs produced the next
event. The report and overlap records are retained at
`/home/qinhaiyan/cake-weave-dual-role-b360415f-matrix12/` and mirrored
locally. These results establish the bounded pipeline across the frozen
route/K matrix on that code commit; they do not transfer automatically to a
successor memory-ordering revision or qualify a latency gain.

The first barrier revision counted communication CTA arrivals with a plain
atomic add. That atomicity alone did not state how each CTA's prior tile and
metadata writes reached the last arrival. Successor `c8550363` uses PTX
`atom.acq_rel.sys.global.add.s32` on each per-event, per-phase arrival, then a
system-scope release for the ready flag; waiters use a system-scope acquire.
The [PTX memory model](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
defines the acquire and release patterns used for this handoff. Its clean
commit passes 15 related contracts, the 179-case Corpus Gate and B300 NVCC
(84 registers, 9,216 B static shared memory, zero spills). The frozen
`b360415f` matrix remains a separate run. Broker job `gpuq-9972c6f1c4cf`
validated `c8550363` on the frozen mixed K=4, c=64, budget=46920 case:
0/4,194,304 FP64-oracle failures, 80 exact CPU/GPU event-count slots,
zero bit differences from v3, and four positive same-rank task-progress
overlap records. The report is retained at
`/home/qinhaiyan/cake-weave-dual-role-acqrel-c8550363/` and mirrored
locally. This one-case successor result does not yet inherit the earlier
four-route matrix's coverage or qualify latency. Broker job
`gpuq-b982dd9436ba` subsequently ran its own four-route × K=4/2/1 matrix.
All twelve separately sealed controls passed with 0/50,331,648 FP64-oracle
failures, exact agreement for 960 CPU/GPU tile-event slots, zero bit
differences from v3, and positive useful-work overlap records in every
control. Its report is retained at
`/home/qinhaiyan/cake-weave-dual-role-acqrel-c8550363-matrix12/` and
mirrored locally. The successor now has the same frozen route/K correctness
scope as `b360415f`; neither is a qualified timing result. Same-lease ABBA
job `gpuq-9d2f8521be29` compared `c8550363` with the equal-reset sequential
persistent source `3e1ecb94` on mixed K=4, c=64, budget=46920. All four
arms passed the oracle, tile plan and v3 bitwise checks. With L2 flushed
before each sample, the development CUPTI medians were 10.051 ms for the
sequential control and 10.074 ms for the producer/consumer grid, a 0.998x
old/new ratio. The retained report is
`/home/qinhaiyan/cake-weave-dual-role-acqrel-vs-persistent-abba/report.json`.
Observed work overlap alone did not improve this complete-layer assay.

Broker job `gpuq-e6b37e5feec8` also checked six sealed spatial/steal plans
on the frozen mixed K=4 route: c=1 with budgets 0/5888, c=74 with budgets
0/5888/46920, and c=95 with budget 46920. All six passed the FP64 oracle
over 25,165,824 output elements, matched 480 CPU tile-event count slots and
were bitwise equal to v3. Each had a positive next-event production versus
Cake-task progress flag. Zero budget produced zero stolen tasks; c=74,
budget=5888 reached the cap on every rank. The c=95 arm passed this one run,
but does not close F-2026-09-27-001's intermittent route-location mismatch.
Reports are retained at
`/home/qinhaiyan/cake-weave-dual-role-acqrel-c8550363-sweep6/` and mirrored
locally. This adds spatial and steal correctness coverage, not a timing rule.

Job `gpuq-734fbadaded6` profiled those six sealed controls with two warmups
and three L2-flushed CUPTI samples each. The post-lease audit checked the
FP64 oracle and v3 bitwise output again, plus four reset kernels, 16 layer
kernels, 92 layer memsets and complete phase records per sample. Mixed K=4
development medians were 132.41/131.65 ms for c=1 at budgets 0/5888;
24.55/14.48/10.42 ms for c=74 at budgets 0/5888/46920; and 10.72 ms for
c=95 at budget 46920. The c=74, budget=5888 arm exhausted its permit on
every rank. The same-lease c=64, budget=46920 ABBA above had a roughly
10.07 ms producer/consumer median. The report is retained at
`/home/qinhaiyan/cake-weave-dual-role-acqrel-c8550363-profile6/report.json`.
These are development observations on one route: they guide barrier and
partition investigation, but do not qualify a c selection or external
baseline speedup.

Successor `00a2fc14` scopes only the **producer CTA barrier** to the device:
its arrivals use `atom.acq_rel.gpu.global.add.s32`, with GPU-scope
release/acquire phase flags. Cross-GPU payload, bin, tile and source-completion
handoffs retain system scope. The PTX instructions are explicit B300 Target
contracts and lowering requirements. This clean commit passed 15 related
contract tests, seven instruction-contract tests and the 179-case Corpus
Gate. B300 NVCC compiled the main kernel at 84 registers and 9,216 B static
shared memory with zero spills. Its one-case broker job `gpuq-f9ec0886774d`
passed the mixed K=4, c=64, budget=46920 oracle over 4,194,304 elements,
matched all 80 CPU/GPU tile-event slots, was bitwise equal to v3, and recorded
positive same-rank next-event overlap on all four ranks. Evidence is retained
at `/home/qinhaiyan/cake-weave-device-barrier-00a2fc14/` and mirrored locally.

Two independent four-GPU same-lease ABBA jobs compared `c8550363` against
`00a2fc14` on that control. Job `gpuq-aa942d495aec` measured old/new
development CUPTI medians of 9.991/9.526 ms (old/new 1.049x); job
`gpuq-027d892feb0a` measured 10.131/9.079 ms (1.116x) on a different
physical GPU set. Each arm had two warmups, three L2-flushed samples, four
reset kernels, 16 layer kernels, 92 layer memsets, 400 phase records, zero
oracle failures and zero v3 bit mismatches. The two reports and raw CUPTI
records are retained at
`/home/qinhaiyan/cake-weave-device-vs-system-barrier-abba-00a2fc14/` and
`/home/qinhaiyan/cake-weave-device-vs-system-barrier-abba-00a2fc14-rep2/`,
and mirrored locally. The repeated direction supports investigating this
local barrier cost, but the 1.049x–1.116x range is a development observation:
clock calibration, common Candidate timing and a matched external baseline
remain unqualified.

Broker job `gpuq-15c78b154f1c` separately validated `00a2fc14` over four
frozen routes × K=4/2/1, all at c=64 and budget=46920. All 12 sealed
controls passed with 0/50,331,648 oracle failures, 960 exact CPU/GPU
tile-event slots, zero v3 output bit mismatches, positive useful-work
overlap in every control and nonzero stolen tasks in every control. The
report is retained at
`/home/qinhaiyan/cake-weave-device-barrier-00a2fc14-matrix12/` and
mirrored locally. This restores the bounded route/K correctness scope for
the device-scope successor; the c=95 intermittent Finding remains open. The
broker receipt retained the prior matrix label due to script templating;
the sealed source, build, launch manifests and leaf case records all name
`00a2fc14`, and the receipt's job id binds the device outputs.

Before promotion, check further c/steal controls including the open c=95
mismatch Finding, retain profiler evidence, and qualify the target's
four-device timing reset and matched open baseline. This task branch remains
a development successor, not a claim of Weave speedup or full temporal
overlap.
