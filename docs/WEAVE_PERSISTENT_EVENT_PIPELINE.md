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

Job `gpuq-0f37f64eb3a3` extended `00a2fc14` to six sealed mixed K=4
spatial/steal controls: c=1 at budgets 0/5888, c=74 at 0/5888/46920,
and c=95 at 46920. All six passed with 0/25,165,824 oracle failures,
480 exact CPU/GPU event-count slots, zero v3 bit mismatches and positive
next-event overlap. The two budget-zero controls stole zero tasks; c=74
with budget 5888 reached exactly 5888 stolen tasks on every rank. The
report is retained at
`/home/qinhaiyan/cake-weave-device-barrier-00a2fc14-sweep6/` and mirrored
locally. One passing c=95 trial does not close F-2026-09-27-001's
intermittent route-location mismatch.

The next kernel/lowering hypothesis at `55e1cf2a` fused the separate
producer handoffs after tile assignment and task-count expansion. CTA 0
retained an internal `__syncthreads()` before expanding tasks, while all
communication CTAs joined one device-scope barrier before gathering. This
reduced the producer's per-event barrier count from six to five without
changing the ranked effects or the Workload. Its fixed-commit checks passed
22 related contracts and the 179-case Corpus Gate; B300 NVCC used 84
registers, 9,216 B static shared memory and no spills. Job
`gpuq-53b1b04354f3` passed the frozen mixed K=4 case with 0/4,194,304
oracle failures, 80 exact event slots, v3 bitwise agreement and positive
overlap on all ranks. Job `gpuq-70c45c0647c8` passed the four-route ×
K=4/2/1 matrix with 0/50,331,648 oracle failures, 960 exact event slots,
v3 bitwise agreement and positive overlap in all 12 controls. Evidence is
retained under `/home/qinhaiyan/cake-weave-fused-handoff-55e1cf2a/` and
`/home/qinhaiyan/cake-weave-fused-handoff-55e1cf2a-matrix12/` and mirrored
locally.

Two same-lease ABBA jobs compared `00a2fc14` with `55e1cf2a` on mixed
K=4, c=64, budget=46920. `gpuq-d7c86efc2e11` measured old/new development
CUPTI medians of 9.395/8.880 ms (1.058x old/new); `gpuq-6bae3e44fdb4`
measured 9.181/9.337 ms (0.983x). Both passed the oracle, tile plan and
v3 bitwise checks. The tile-assignment and task-expansion phase intervals
fell from roughly 3.8k+5.2k to 2.0k+2.0k cycles per event, but that small
local saving did not produce repeatable complete-layer improvement. Raw
CUPTI and phase records are retained at
`/home/qinhaiyan/cake-weave-fused-vs-device-abba-55e1cf2a/` and
`/home/qinhaiyan/cake-weave-fused-vs-device-abba-55e1cf2a-rep2/` and
mirrored locally. Promotion disposition: **no promotion**; successor
`58aed769` reverted this fusion. The active code again has the `00a2fc14`
producer-barrier structure, while its source identity is the new commit.
This redirected work to the source-dispatch/completion path, whose
per-event wait was around 350k cycles in those traces.

Successor `db16d3af` vectorized each remote 4 KB BF16 payload copy with
`ld.global.v4.b32` and `st.global.v4.b32`. The Bin payload slots are
16-byte aligned; the initial version also required the caller's hidden
pointer to be aligned. The [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
requires natural 16-byte alignment for these accesses, and B300 `cuobjdump`
showed `LDG.E.128` and `STG.E.128` in the compiled main kernel. Its clean
commit passed 22 related contracts, the 179-case Corpus Gate and B300 NVCC
at 84 registers with no spills. Its mixed K=4 single-grid job passed the
oracle and v3 bitwise comparison; the 12-route/K matrix passed with
0/50,331,648 oracle failures, 960 exact event slots and 12 positive
overlap controls. On mixed K=4, same-lease `gpuq-7844211fd52b` measured
9.503/9.508 ms old/new: no complete-layer improvement despite reducing
the source-completion phase median from about 360k to 318k cycles. The
communication-heavier fanin K=4 route showed small same-direction ABBA
differences in two jobs: 8.513/8.409 ms (`gpuq-8cb31ac785e5`) and
8.462/8.366 ms (`gpuq-054f0086a448`). All compared arms passed the
external oracle, tile plan and v3 bitwise checks. The retained roots use
the `cake-weave-vector-payload-db16d3af` and
`cake-weave-vector-vs-device-*-db16d3af` names and are mirrored locally.

The initial alignment rule narrowed the pointer ABI without a matching
Workload constraint. Broker job `gpuq-cca52dbe0c94` confirmed it rejected
a hidden BF16 pointer offset by two bytes. Successor `a6fcc377` removed
that admission change: aligned input keeps the PTX vector path, while a
BF16-aligned subview uses the original scalar copy. Its 179-case Corpus
Gate passed; the 22-test local contract invocation had two environment
errors because macOS offered no usable temporary directory, so it is not
recorded as a passing suite. B300 NVCC again used 84 registers and had no
spills. Separate four-GPU jobs `gpuq-2fcf501c0f27` (aligned) and
`gpuq-59ad31561265` (hidden pointer offset by two bytes on every rank)
both passed 0/4,194,304 oracle failures, 80 exact event slots and v3
bitwise comparison, while retaining positive overlap.

Two same-lease fanin K=4 ABBA jobs measured the final `a6fcc377` against
`00a2fc14`: 8.524/8.367 ms (`gpuq-fae3d2be75f0`, old/new 1.019x) and
8.593/8.457 ms (`gpuq-f07edad98695`, 1.016x). These are repeated small
development differences on a remote-payload-heavy route, not a qualified
speedup. The final source's four-route × K=4/2/1 matrix
`gpuq-9797dbb39967` passed with 0/50,331,648 oracle failures, 960 exact
event slots, zero v3 bit mismatches and positive overlap in all 12
controls. Spatial/steal job `gpuq-8a3021dc7c33` passed six sealed mixed
K=4 controls with 0/25,165,824 oracle failures, 480 exact event slots,
zero v3 bit mismatches and six positive overlap flags; budget zero stole
zero tasks, and c=74 at budget 5888 hit the cap on every rank. Its c=95
pass does not close F-2026-09-27-001. These reports and raw ABBA records
are retained under the matching `/home/qinhaiyan/cake-weave-vector-fallback-a6fcc377*/`
roots and mirrored locally. Promotion disposition: keep as a B300 native
CUDA development candidate; no Lab rule or external baseline claim.

The next Compiler/lowering/kernel tick grouped four Cake activation rows in
one CTA. The B300 SwiGLU Schedule now declares a four-row ProgramMap tile,
four execution groups and `[4,768]` register values; the native CUDA
emitter maps one warp to each row. Ranked-tile analysis therefore derives
`24+32+32=88` tasks per tile instead of `24+128+32=184`, reducing the
255-tile task bound from 46,920 to 22,440. The Evaluation admission and
pointer ABI follow the new bound; the Workload and oracle are unchanged.
The fixed code commit `48b811b3` passed 33 related contracts and the
179-case Corpus Gate. B300 NVCC compiled the complete grid at 84 registers,
9,216 B static shared memory and no spills. The standalone activation
emitter also compiled at 25 registers and no spills. The original attempt
`gpuq-ba4dd7aa6379` was refused before device execution because its frozen
pointer adapter still expected 46,920 slots. Successor `c3034259` corrected
that adapter, and the fixed-commit test now calls its CPU-only contract
preflight. The failed job remains separate from every later result.

Job `gpuq-07acaac5a977` validated `48b811b3` on mixed K=4, c=64,
budget=22,440: 0/4,194,304 FP64-oracle failures, 80 exact event slots,
zero v3 bit differences and positive same-rank overlap. Two same-lease
old/new/new/old CUPTI comparisons used the **same 22,440 budget** for the
one-row control `a6fcc377` and the four-row successor. Job
`gpuq-32ba52458409` measured complete-layer development medians of
9.749/7.067 ms (old/new 1.379x); `gpuq-b442e54ddf05` measured
9.043/6.990 ms (1.294x) on another physical GPU set. All eight arms
passed the oracle, tile plan and v3 bitwise comparison. The main
`tile_schedule_probe` kernel fell from roughly 8.3–8.6 ms to 6.4–6.5 ms
in those paired records. The reports and raw CUPTI traces are retained at
`/home/qinhaiyan/cake-weave-activation4-vs-row1-abba-48b811b3/` and its
`-rep2` successor, and mirrored locally.

Job `gpuq-7a3df15db65e` checked the four frozen routes × K=4/2/1 at
c=64 and budget 22,440. All 12 sealed controls passed with
0/50,331,648 oracle failures, 960 exact CPU/GPU event-count slots, zero
v3 bit differences and positive useful-work overlap. Job
`gpuq-bf74c63e46ac` then checked six spatial/steal plans on mixed K=4:
c=1 at budgets 0/5888, c=74 at 0/5888/22,440 and c=95 at 22,440.
All six passed with 0/25,165,824 oracle failures, 480 exact event slots,
zero v3 bit differences and positive overlap. Both zero-budget controls
stole zero tasks. The reduced task population no longer exhausted the
5888 permit in the c=74 control; a separate sealed c=74, budget=2048 job
`gpuq-2ebc3ba3c014` passed 0/4,194,304 oracle failures and reached
**exactly 2048 stolen tasks on each rank**. One passing c=95 control
does not close F-2026-09-27-001. These reports are retained under the
matching `/home/qinhaiyan/cake-weave-activation4-48b811b3*/` roots and
mirrored locally. Promotion disposition: retain the four-row B300
Schedule/lowering as a development candidate; do not infer a generic
cross-target rule or an external-baseline speedup from this scope.

Successor `1c02cd74` uses all six worker warps for the Cake activation:
the Schedule declares a six-row ProgramMap tile, six execution groups and
`[6,768]` register values. The native CUDA lowering gives one row to each
warp and masks rows beyond 128 in the final work unit, which has only two
valid rows. Ranked-tile analysis derives `24+22+32=78` tasks per tile and
a 19,890-task bound. The fixed commit passed 33 related contracts and the
179-case Corpus Gate. B300 NVCC compiled standalone activation at 25
registers and the complete grid at 84 registers, 9,216 B static shared
memory, with no spills. Job `gpuq-28c73e1326aa` passed mixed K=4,
c=64, budget=19,890 with 0/4,194,304 oracle failures, 80 exact event
slots, zero v3 bit differences and positive overlap. This includes the
masked activation tail in a complete layer.

Two same-lease ABBA jobs compared the four-row `48b811b3` control with
the six-row successor at the **same 19,890 budget**. Job
`gpuq-7e5afc2adc8c` measured old/new development medians of
6.796/6.730 ms (1.010x); `gpuq-f257e40154a6` measured 7.000/6.696 ms
(1.045x) on a different GPU set. Every arm passed the external oracle,
tile plan and v3 bitwise comparison. The direction repeated, but the
incremental gain is small and variable; it does not replace the much
larger one-row to four-row finding. Reports and raw CUPTI records are
retained at `/home/qinhaiyan/cake-weave-activation6-vs-row4-abba-1c02cd74/`
and its `-rep2` successor, and mirrored locally.

Job `gpuq-539e3bed3d9a` checked four frozen routes × K=4/2/1 at c=64
and budget 19,890. All 12 controls passed with 0/50,331,648 oracle
failures, 960 exact CPU/GPU event-count slots, zero v3 bit mismatches and
positive overlap. Job `gpuq-abdfebc0ec7b` checked six mixed K=4
spatial/steal controls: c=1 at budgets 0/5888, c=74 at 0/5888/19,890
and c=95 at 19,890. All six passed with 0/25,165,824 oracle failures,
480 exact event slots, zero v3 bit mismatches and positive overlap; the
zero-budget controls stole zero tasks. A separate c=74, budget=1024 job
`gpuq-adc8823fb59f` passed 0/4,194,304 oracle failures and reached
**exactly 1024 stolen tasks on each rank**. One c=95 pass does not close
F-2026-09-27-001. Results are retained under the matching
`/home/qinhaiyan/cake-weave-activation6-1c02cd74*/` roots and mirrored
locally. Promotion disposition: keep this six-warp B300 Schedule and
lowering as a development candidate; no generic rule or Lab recipe yet.

The next bounded spatial diagnosis kept the same `1c02cd74` source and
mixed K=4 workload. CUPTI job `gpuq-4268c9028bc2` measured development
medians of 120.56/120.16 ms for c=1 at budgets 0/5888; 13.28/7.12/7.19
ms for c=74 at budgets 0/5888/19,890; and 7.84 ms for c=95 at budget
19,890. Every arm passed the external oracle, v3 bitwise output, and the
four-rank reset and activity audit. The c=95 samples did not reproduce
F-2026-09-27-001 and do not close it.

Two same-source, same-budget c scans reversed their order on separate
four-GPU leases. Job `gpuq-475b7c1687b2` profiled c=48/56/64/74 at
8.251/7.726/7.005/6.846 ms; job `gpuq-582b7efa2a21` ran c=74/64/56/48
and reported c=48/56/64/74 at 7.989/7.725/6.815/6.793 ms. All eight
controls passed the oracle, v3 bitwise comparison, L2 reset and CUPTI
activity coverage. Thus c=48 and c=56 were slower than c=64/74 in both
orders on this route. Direct same-lease ABBA `gpuq-5cb9c26c249d`
measured c=64/c=74 at 6.806/6.919 ms, a small difference opposite the
sequential profiles. The profile reports are retained under the matching
`/home/qinhaiyan/cake-weave-activation6-1c02cd74-c-profile4*/` roots;
the paired report is at
`/home/qinhaiyan/cake-weave-activation6-c64-vs-c74-abba-1c02cd74/`.
They are mirrored locally. Promotion disposition: **no c-selection rule**
from these small, order-sensitive differences; c=64 remains the fixed
development comparison control.

An external handoff audit now relates the model-scale Cake `fanin_v2` case
to the retained SGLang `v0.5.12.post1` + DeepEP `1.2.1` fallback. A CPU-only
full-array comparison found zero differences across all four ranks in the
BF16 hidden values, expert IDs, FP32 route weights, and every BF16 gate,
up and down weight after Cake's declared up/gate packing. The FP32 oracle
outputs are bitwise identical. The report and comparison script are retained
at `/home/qinhaiyan/cake-weave-vs-sglang-fanin-input-match-20260928/` and
mirrored locally. This establishes an exact input/oracle relation for this
synthetic case, not equivalence to the paper's SGLang version or its
ShareGPT evaluation.

A separate CPU-only output relation compared the six-warp Cake fanin K=4
device output with the warmed SGLang/DeepEP output. Both passed that same
FP64 oracle and tolerance, but **2,763,349/4,194,304 BF16 output bit
patterns differed** across the four ranks. The numerical implementations
are therefore not bitwise interchangeable. The retained
`output_relation.json` sits beside the input-match report.

Two separate four-GPU broker jobs tested whether the baseline could share
Cake's low-overhead CUPTI activity window. Jobs `gpuq-a6a4a9314469` and
`gpuq-822103215b27` each passed the original baseline FP64 oracle over
4,194,304 values, with no failures. Per-rank cupti-python activities,
including the successor's callback-before-enable and forced-flush change,
retained only four sample kernels each and omitted the FFN and combine
stages. Their coverage-failure reports and raw records remain at the
`/home/qinhaiyan/cake-weave-sglang-cupti-feasibility*-20260928/` roots.
Neither run supports a four-rank GPU latency.

A third independent job `gpuq-b1952cc5c401` enabled cupti-python before
the first baseline forward. Its FP64 oracle again passed with zero failures,
but its four rank files retained only initialization activities and **zero
sample kernels**. The original, forced-flush and early-enable overlays thus
all fail complete-layer coverage in this multi-process path. The third
coverage-failure record is retained at
`/home/qinhaiyan/cake-weave-sglang-cupti-before-warmup-20260928/` and
mirrored locally. No latency is selected from any of the three.

A third broker job, `gpuq-97abd4ebc7fc`, used one warmup, a synchronized
L2 clear per rank and a Torch profiler trace on the same baseline inputs. Its
post-release oracle passed with zero failures. All four traces retained 49
CUDA kernels each, including four DeepEP dispatch and four combine kernels;
the CPU trace recorded eight bmm and four SiLU operations per rank. The
trace timestamps fall inside their rank's wall-clock sample window after
accounting for profiler setup. The trial's developmental joint kernel span
was 12.764 ms, while each profiled host window was about 243 ms and spent
about 210 ms before its first kernel. That instrumented span is **not** a
Cake-vs-baseline latency ratio. The trace audit, failed first timestamp
audit, corrected containment audit, four trace files and oracle result are
retained at `/home/qinhaiyan/cake-weave-sglang-warm-trace-20260928/` and
mirrored locally. The baseline's declared `qualified_latency_ns` remains
null. A shared low-perturbation four-rank timer and target reset contract
are still required before an external speedup claim.

An out-of-process Nsight Systems successor tested a lower-perturbation
trace route on the same frozen baseline and input snapshot. Broker job
`gpuq-480e8fd1e65f` completed after one warmup and one synchronized L2
clear per rank; the post-release FP64 oracle passed with 0/4,194,304
failures. Its four-process report has one NVTX complete-layer window per
rank of 2.289–2.353 ms. Inside each window, the unified GPU timeline
contains 49 kernels and 12 GPU copy records: four DeepEP dispatches, eight
BF16 GEMM kernels, four SiLU kernels and four DeepEP combines. A retained
unsigned-char fill kernel precedes each NVTX window, and the CPU-only audit
reports a **2.149 ms developmental joint GPU activity span**. The source
overlay, Nsight report, SQLite export, oracle result and audit are retained
at `/home/qinhaiyan/cake-weave-sglang-nsys-warm-20260928/` and mirrored
locally. This single Nsight observation uses a different profiler from the
Cake CUPTI ABBA and is not a qualified Cake-to-baseline speedup. A common
timer, repeated matched samples and target clock/reset qualification
remain required.

The matching Cake fanin K=4/c=64/budget=19,890 Nsight trial uses the same
input and oracle snapshot as that baseline. Its first external root,
`/home/qinhaiyan/cake-weave-cake-nsys-fanin-20260928/`, ended at profiler
argument parsing in broker job `gpuq-10091f12229d`: the host's Nsight
Systems 2025.5 does not accept `--cuda-trace-scope=process-tree`. No device
sample or latency came from that job. Successor broker job
`gpuq-d5a5e7d5c69f` ran and released four GPUs normally. The post-release
audit passed the shared FP64 oracle with **0/4,194,304** output failures,
matched the v3 output bitwise, and matched all 80 CPU tile-event slots;
the four ranks stole 1,509/1,549/1,462/1,493 stage tasks.

The successor Nsight report has one complete-layer NVTX window of 7.847 ms
around the four-rank sample, agreeing with the 7.838 ms host window. Inside
it, each device has one worker, scatter, wait and combine kernel, ten GPU
copies and 23 memsets; an L2 clear kernel precedes the window on every
device. The development joint GPU activity span is **7.733 ms**; the worker
kernel alone takes 5.937–6.086 ms per rank. The baseline's single Nsight
span was 2.149 ms on the exact same inputs. The observed Cake span is about
3.60 times longer, but these are single instrumented observations with
different Nsight versions and separate leases, so this is a bottleneck lead,
not a qualified speedup or slowdown estimate. The report, SQLite export,
oracle and analysis are retained at
`/home/qinhaiyan/cake-weave-cake-nsys-fanin-v2-20260928/` and mirrored
locally. The worker is the first target for the next compiler/lowering/kernel
trial.

A CPU-only N=128 feasibility check first showed that `Compiler.assess`
admitted both modified up/gate and down Schedules without blocking findings;
standalone native CUDA lowering emitted grids `[1,12,1]` and `[1,16,1]` at
65,584 dynamic shared bytes each. Commit `0177ca57` added a distinct N128
Program and bounded worker lowering. Ranked-tile analysis derives
`12+22+16=50` tasks per logical tile and a 12,750-stage-task capacity,
versus N64's `24+22+32=78` and 19,890. NVCC compiled the full N128 library
with 145 registers/thread, zero spills and 9,216 static shared bytes; the
same-commit N64 control uses 84 registers/thread with zero spills. The N128
variant keeps the existing N64 Program and experiment artifacts intact.

The first four-GPU N128 job `gpuq-dd1cdbaba524` failed before model-kernel
execution because Evaluation admission still required 19,890 stage tasks.
The failed root is retained at
`/home/qinhaiyan/cake-weave-n128-nsys-fanin-0177ca57/`. Commit `2c638aad`
changed that admission to require the capacity from the checked ranked-tile
analysis. A CPU replay of the failed N128 Program, lowering and exact plan
then passed at `/home/qinhaiyan/cake-weave-n128-admission-replay-2c638aad/`;
25 related contracts and all 179 Corpus Gate cases passed at the fixed
commit. The workload, FP64 oracle, route snapshot and L2 reset did not change.

Successor job `gpuq-057623aa7116` completed on four B300 GPUs. Its
post-release audit found **0/4,194,304** FP64-oracle failures, zero v3 bit
differences and exact agreement on all 80 CPU tile-event slots; the ranks
stole 951/1,038/1,048/968 tasks. The complete Nsight window is 7.455 ms,
the host window 7.447 ms, and the single development joint GPU activity
span **7.335 ms**, with per-rank worker kernels of 5.672–5.810 ms. That is
shorter than the earlier N64 Nsight span of 7.733 ms, but separate leases
and one sample per arm cannot establish a speedup. The N128 root is
`/home/qinhaiyan/cake-weave-n128-nsys-fanin-2c638aad/` and mirrored locally.

Two same-commit, same-input, same-lease CUPTI comparisons then exercised
N64 and N128 at c=64 with their respective full analyzed steal capacities,
19,890 and 12,750. Every arm performed two warmups and three L2-flushed
samples on four GPUs. All 12 measured samples in each four-arm trial had complete
16-kernel and 92-memset layer coverage, plus four reset kernels. Every arm
passed the FP64 oracle, v3 bitwise check and CPU tile-event plan.
In `gpuq-4111913e8f16` (N64/N128/N128/N64), the first three arm medians
were 6.447/6.210/6.224 ms, but the final N64 arm took **27.082 ms**;
its worker kernels, not a trace gap, took roughly 18–26 ms across devices.
The retained `interpretation-v2.json` explicitly selects no performance
ratio from that trial; the automatic 2.696 ratio in its original report is
invalid as a performance estimate. The cause of this order-specific slow
arm remains unproven.

Independent reversed-order `gpuq-f1c87673109b` (N128/N64/N64/N128)
measured arm medians 6.184/6.373/6.315/5.997 ms. Its N64 and N128
two-arm medians are **6.344 and 6.091 ms**, a **1.042x** N64/N128
development ratio on this route. The first ABBA's stable opening pair and
the separate Nsight observations point in the same direction, but the
unexplained 27 ms arm, small absolute saving and missing target clock
qualification prevent a general performance claim. Both comparison roots
are retained under `/home/qinhaiyan/cake-weave-n64-vs-n128-*-2c638aad-fanin/`
and mirrored locally. Promotion disposition: retain N128 as a bounded
development candidate; do not select it by default or generalize beyond
fanin K=4/c=64 until the route/K matrix, timing stability and profiler
qualification are checked.

Before promotion, resolve the open c=95 mismatch Finding, qualify the
target's four-device timing reset, and run a matched open baseline under
the same measurement contract. This task branch remains a development
successor, not a qualified claim of Weave speedup or full temporal overlap.
