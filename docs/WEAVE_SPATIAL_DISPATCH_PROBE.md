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
