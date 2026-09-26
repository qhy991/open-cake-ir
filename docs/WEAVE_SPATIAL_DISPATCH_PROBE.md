# B300 spatial communication CTA investigation

The existing ranked-tile source launches route dispatch in a separate kernel.
Its `communication_ctas` argument classifies CTAs within a later worker grid
as borrow-eligible; it does not assign communication work to those CTAs or
reserve physical SMs. This is the remaining spatial-schedule gap.

The standalone successor at `2e12ce78` gives that control real work. One
96-CTA grid per rank assigns the first `c` CTAs to model-width BF16 P2P
expert-bin transport and the remaining `96-c` CTAs to an independently
checked BF16-to-FP32 tensor transform. It uses PTX system-scope atomics and
release stores for the route bins, plus `%globaltimer` and active CTA
counters for a within-device overlap observation. The source builds with
40 registers, no spills and one barrier. This prototype owns neither the
Cake FFN stage bodies nor the existing tile planner.

The exact frozen fan-in Workload input was replayed on B300-M4 under one
exclusive four-GPU broker job `gpuq-b5a0b4200178`. Separate c=1 and c=74
processes both passed a post-lease CPU oracle over all 16,384 route keys,
their BF16 bin rows and the independent FP32 transform. Each rank reported
the requested CTA counts (1/95 or 74/22) and an overlap flag set while
both CTA roles were active. The within-device group-window intersections
were about 37–39 microseconds at c=1 and 130–141 microseconds at c=74.
Raw binaries and rows remain at
`B300-M4:/home/qinhaiyan/cake-weave-spatial-dispatch-2e12ce78/`; the small
reports are mirrored under
`open-cake-ir-workspaces/evidence/weave-b300-m4-20260927/cake-weave-spatial-dispatch-2e12ce78/`.

The next backend step must let resident communication CTAs feed the exact
Cake FFN worker through source-completion and tile-ready handoffs, retaining
the four-rank output oracle and explicit failure progress. The prototype
alone establishes no model FFN overlap, complete-layer latency, or speedup.
Promotion disposition: no public Compiler lowering or automatic Lab rule
from this standalone proof.

## Cross-stream progress audit of the first FFN integration candidate

`4100a9d5` moves source dispatch into `tile_schedule_probe` and launches
that 96-CTA cooperative worker before separate tile-planner kernels. The
worker waits on `wave_consumed[event]` for the planner, while the planner
waits on `source_wave_done` from the worker. Host stream ordering and
system-scope publication make the *data* dependency explicit, but progress
also depends on the worker and planner kernels executing concurrently on
one GPU. The Target's 148 SMs and the runtime occupancy check admit the
96-CTA cooperative grid; they do not themselves guarantee that another
stream's dependent planner kernel runs while the grid is resident.

The CUDA [stream guidance](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-host-programming.html)
says different streams **may** execute concurrently under resource and
dependency conditions. The [programmatic dependent launch guidance](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)
explicitly warns that relying on opportunistic concurrent execution for
progress is unsafe. [Green contexts](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__GREEN__CONTEXTS.html)
can partition SMs, but NVIDIA likewise states that disjoint partitions
do not guarantee concurrent kernel execution or forward progress, so
they do not close this dependency. This is a forward-progress gap in
`4100a9d5`, not a
measured deadlock. The direct and jump SSH paths to B300-M4 timed out before
that candidate's successor source could be compiled; no FFN device result
exists for it.

The next implementation should put source dispatch, expert snapshot and
tile planning, row gather, and Cake stage-task execution into one bounded
cooperative grid per rank. A grid-wide barrier can separate each local
phase; system-scope release/acquire mailboxes still synchronize ranks.
Keeping the planner in the grid removes the cross-stream scheduling
dependency, while exact-target residency, cross-rank source order, finite
worker progress and the four-case output oracle still require verification.
Until then, keep this integration candidate on the task branch and make no
liveness or performance claim from it.
