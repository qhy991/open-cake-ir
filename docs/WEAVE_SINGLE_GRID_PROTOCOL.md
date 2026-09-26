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
ranked effects and 25 Cake math operations once. No B300-M4 NVCC/PTXAS
build or device run exists for this successor: direct and jump SSH both timed
out while preparing the earlier cross-stream candidate. The static checks
do not measure the new planner's register/static shared footprint, confirm
cooperative residency, or verify exact GPU tile events and FP64 outputs.

A source-declaration audit counts 8,192 B for the per-CTA expert sort,
84 B for source-dispatch scratch and 20 B for worker claims, in addition to
49,200 B of emitted dynamic shared memory: 57,496 B before compiler
padding. Four such CTAs would require 229,984 B, below the B300 Target's
declared 233,472 B per SM. This rules out an obvious shared-memory-only
four-CTA refusal; it does **not** establish actual occupancy. The compiled
register/static-shared report and `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
gate remain authoritative before the cooperative launch.

The first device validation should compile this clean commit outside a GPU
lease, then use the broker for at most four GPUs. Start with one mixed-route
K=4,c=1,budget=5888 case under an isolated process and bounded runtime;
after it passes, audit K=1/2/4 and the other frozen routes against the
external FP64 oracle and CPU event-tile planner. CUPTI intervals remain
development evidence until the four-device reset/clock and sealed Candidate
gates are qualified. Promotion disposition: **no merge or Lab rule** from
the static single-grid candidate.
