# B300 single-grid ranked-tile protocol candidate

Code commit `33f28b88` is a development successor to the resident CTA transport
proof. It puts source dispatch, expert-bin snapshot and tile construction,
row gather, and the three Cake FFN stage-task bodies in **one cooperative
grid per rank and source event**. The generated pointer ABI is version 4;
an older compiled library is refused by the loader. The exact domain is
four B300 ranks, model-width BF16 tensors, K=1/2/4 uniform across ranks,
1–95 communication CTAs in a 96-CTA grid, 255 logical tiles and at most
46,920 stage work units per rank.

## One event

1. If this rank is the event's source, block 0 waits until every destination
   has published the preceding event's tile snapshot. The first `c` CTAs
   then dispatch this source chunk into expert-owned P2P bins. All writer
   threads issue a system fence, the grid synchronizes, and source block 0
   release-publishes `source_wave_done` to every destination.
2. Destination block 0 acquire-waits for the source completion. After a grid
   barrier, the first 32 CTAs sort one local expert snapshot each. Block 0
   assigns tile keys, valid rows and stage-task counts, with a grid barrier
   between producers and consumers.
3. All 96 CTAs gather tile rows in a grid-stride loop. After system fences
   and a grid barrier, block 0 release-publishes `wave_consumed[event]`.
   The next source's wait uses that flag, so later source rows cannot enter
   this event's expert snapshot. Every thread then executes
   `fence.proxy.async.global` before Cake's TMA reads those tile rows:
   the gather's generic global stores and the TMA async-proxy reads now
   occur inside the same kernel, so the [PTX proxy-fence rule](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
   applies at this new boundary.
4. CTAs claim complete Cake stage work units from per-tile GPU-scope queues.
   CTAs in the `c` communication class need a bounded steal permit; the
   other `96-c` CTAs are ordinary computation workers. Predecessor-stage
   completion uses the existing GPU-scope acquire/release contract. A final
   grid barrier completes the event before that rank's next event kernel.

After all events, the existing return scatter, ready checks and Cake
weighted combine run on the same rank-local stream. This removes the old
cycle where a resident worker waited for planner kernels in another stream
on the **same GPU**. It does not by itself prove arbitrary cross-rank
interleaving liveness or that a malformed device state can always report a
failure before a broker timeout. The host still owns prelaunch route checks,
exact target and peer admission, cooperative occupancy admission, complete
state reset, all-rank status checks and isolated-process cleanup.

## Verification boundary

At `33f28b88`, 21 related contract tests and the 179-case Corpus Gate pass
in a detached fixed-commit worktree. The emitted source maps each of the 27
ranked effects and 25 Cake math operations once. The later clean code commit
`f7ae17fe` passes 28 related contracts and the same 179-case Corpus Gate.
Its exact source compiled with B300-M4 CUDA 13.1 NVCC outside a GPU lease at
`/home/qinhaiyan/cake-weave-single-grid-f7ae17fe/build-mixed-k4-c1-b5888/`:
the build produced a 502,584-byte ELF with pointer ABI version 4. PTXAS
reported 80 registers and 9,216 bytes of static shared memory for
`tile_schedule_probe`. A later isolated four-GPU development trial
`gpuq-da4acd47acce` used the sealed ABI-v4 library for the frozen mixed route
at K=4, c=1 and steal budget=5888. It completed and the post-lease audit
found zero FP64-oracle failures across 4,194,304 outputs, zero bit differences
from the earlier v3 K=4 output, and exact agreement for all 80 rank/event tile
counts. Owner tile totals were [80, 64, 64, 64]; actual stolen task counts
were [147, 116, 121, 123]. The report is retained at
`/home/qinhaiyan/cake-weave-single-grid-f7ae17fe-r2/report.json` and mirrored
under the local `cake-weave-single-grid-f7ae17fe-r2` evidence directory.
The subsequent broker job `gpuq-f10e54a7c490` ran all four frozen routes at
K=4/2/1 with individually sealed plans and at most four GPUs. Its post-lease
audit found zero FP64-oracle failures over 50,331,648 output elements, exact
agreement for all 960 rank/event tile counts, and zero bit differences from
the corresponding v3 outputs. Job `gpuq-6cc09f030482` then held the mixed
route at K=4 and ran five sealed spatial/steal controls: c=1 with budgets
0/5888, and c=74 with budgets 0/5888/46920. All five passed the same checks
over 20,971,520 output elements and 400 event counts. With c=74, budget 5888
was exhausted on every rank; budget 46920 permitted actual stolen counts
[10877, 8882, 8914, 8931]. The reports are retained under
`/home/qinhaiyan/cake-weave-single-grid-f7ae17fe-{matrix1,sweep1}/report.json`
and mirrored under the corresponding local evidence directories. These are
development correctness results; they do not establish a timing improvement.
The local integration now carries `RankedTileLaunchManifest` and a
byte-bearing `RankedTileCandidate` for pointer ABI v4: they refuse a wrong
Workload case, source, rank plan or library bytes before CUDA state creation.
Its create-only Lab builder writes the exact lowered source, invokes the
Target's NVCC route outside a GPU lease, retains failures and seals the
resulting ELF bytes in the same process. It persists one artifact-role
record for source, ELF and manifest; the next process checks these bytes
once before loading. That CPU build handoff has now run on B300-M4.
It is not yet the common sealed Evaluation Candidate or a qualified
device/timing result.

The source-declaration audit counted 57,496 B per CTA before compiler
padding. PTXAS instead reports 9,216 B static plus 49,200 B dynamic shared
memory, or 58,416 B per CTA. Four CTAs would require 233,664 B, which is
192 B above the B300 Target's declared 233,472 B per SM. The cooperative
grid has 96 CTAs across a declared 148 SMs, so it needs at least one resident
CTA per SM; the host's `cudaOccupancyMaxActiveBlocksPerMultiprocessor` check
remains authoritative before launch. The source-only four-CTA estimate must
not be used as an occupancy result.

## Development timing and open diagnostic

Job `gpuq-a2d23c16f7ac` sampled seven mixed-route controls under CUPTI with
two warmups and three L2-flushed samples each. The post-lease audit found the
expected four reset kernels, 92 layer kernels and 92 layer memsets per K=4
sample (52/92 at K=2, 32/92 at K=1), with zero oracle failures. Median
four-device GPU activity spans, including the layer's internal reset, were
102.49/99.89/97.67 ms for K=4/2/1 at c=1, budget 5888. At K=4, c=74 and
budgets 0/5888/46920, they were 45.83/16.84/11.13 ms. These control names
are not matched performance treatments against the earlier v3 path: c now
also partitions the real P2P source dispatch, while the older transport had
different parallelism.

Job `gpuq-637084680df8` explored c=8/16/32/48/64/74 with budget 46920;
their valid CUPTI medians were 21.39/14.48/12.92/11.74/10.69/10.99 ms.
The c=95 arm returned device status 1018 after two successful warmups, so the
seven-arm job is failed; its six completed arms passed the oracle and activity
checks. Status 18 is the gather check that a published route location belongs
to the tile's expert. The retained failure is at
`/home/qinhaiyan/cake-weave-single-grid-f7ae17fe-c-sweep/c95_b46920/launch_failure.json`.
Successor `c4c79545` records the key, event and slot on that error. Its exact
source passed 15 related contracts, the unchanged 179-case Corpus Gate and
real B300 NVCC. One isolated trial and a separate run with two warmups plus
20 CUPTI samples at c=95 passed the oracle and tile checks; the failure did
not recur, so its cause and frequency remain open. No control is promoted
from this sweep. The next investigation should localize the single-grid
source-dispatch, tile-plan and Cake FFN phase costs, and reproduce or close
status 18 before broadening the admitted domain. Comparison with the matched
open baseline waits for the common Candidate and four-device reset/clock
gates; these CUPTI spans are development evidence, not qualified speedups.

The successor `15fb89e7` bounds each expert's bitonic snapshot sort by the
next power of two of its populated row count. At the mixed K=4, c=64,
budget=46920 control, 15 related contracts and the 179-case Corpus Gate pass;
B300 NVCC reports the same 80 registers, 9,216 B static shared memory and
zero spills. The ELF shrank from 502,584 B to 383,800 B. One four-GPU trial
passed the FP64 oracle over 4,194,304 elements, matched all 80 CPU event
counts and was bitwise equal to the v3 output. Job `gpuq-643d4d4f94c9` then
ran all four frozen routes at K=4/2/1 with separately sealed c=64,
budget=46920 plans: 50,331,648 output elements had zero FP64-oracle failures,
all 960 rank/event counts matched the CPU plan, and all outputs were bitwise
equal to v3. The same-lease ABBA development
comparison `gpuq-c4fd0c759ea8` ran old/new/new/old on one broker allocation
with the same frozen mixed case, control and L2-flush protocol. Its four
per-arm CUPTI medians were 10.71/9.99/9.54/10.98 ms; the old/new aggregate
medians were 10.85/9.76 ms (1.11× old-over-new). Each arm passed its oracle,
tile-plan and activity checks. The report is retained at
`/home/qinhaiyan/cake-weave-sort-paired-abba-15fb89e7/report.json` and
mirrored locally. A second same-lease ABBA comparison
`gpuq-33897db5cffc` isolated `c4c79545` (the diagnostic predecessor) from
`15fb89e7`: old/new/new/old medians were 10.79/12.18/9.68/11.06 ms, and
old/new aggregate medians were both about 10.93 ms. One new arm had longer
late-event tails on one rank; its cause is unproven. This run-to-run variation
prevents attributing the first ABBA's 1.11× result to dynamic sorting alone.
The code-size and correctness changes are established; timing improvement
and an external-baseline gain remain unqualified. Promotion disposition:
**no merge or Lab rule** from these development timing comparisons.

Diagnostic successor `57bc00ec` records same-CTA PTX `clock64` differences at
eleven boundaries inside every event kernel. It passed 15 related contracts,
the 179-case Corpus Gate and B300 NVCC (80 registers, 9,216 B static shared,
zero spills). Four-GPU job `gpuq-446a8b285c35` passed the mixed K=4, c=64,
budget=46920 FP64 oracle over 4,194,304 elements, matched all 80 CPU tile
event counts and retained five launches × four ranks × 20 events = 400 phase
records. Across its three CUPTI samples, source-rank dispatch took a median
407,159 clock cycles per source event. Expert snapshot sorting had a median
12,592 cycles across rank/events. The heavy FFN work clustered at terminal
events 14 and 19: ranks 1–3 each published 32 tiles in both events and spent
about 4.1 million same-CTA cycles in each FFN interval; terminal events 4 and
9 published no tiles. The records and post-lease audit live at
`/home/qinhaiyan/cake-weave-phase-57bc00ec/`. These counters exclude final
TMEM release and are not cross-GPU or qualified wall-clock latency. They
motivate a controlled K comparison and investigation of late tile work rather
than another unmeasured change to the sorting network.
