# Tile-keyed ranked effects: structural admission

`RankedTileEffects` schema 2 is the shared Compiler contract for moving from
one token-route work claim to an expert-owned tile and its stage work units.
It composes a complete
local Program with the source-rank weighted-combine Schedule. It introduces
no opaque MoE operation and does not reinterpret the frozen ranked-mailbox
version-1 effects.

The payload is keyed by source item and destination rank. Each expert-bin row
retains its `(source_rank, item, route)` identity. A **logical tile** is keyed
by `(destination_rank, expert, tile_index)` and publishes a valid-row count.
Sources reserve rows in a destination-owned expert bin through a returned-old
**system-scope** atomic. A GPU-scope atomic cannot own the cross-rank row
index used by the evidenced P2P dispatcher; stage-task claims stay GPU-scope
because they are local to the destination rank.
Its claimable **stage task** is keyed by that tile plus the Program stage and
one flattened ProgramMap CTA coordinate. Every CTA of a predecessor stage
must complete before a successor stage becomes ready.
Return slots retain the original source/item/route key. Full tiles may publish
when filled; a partial tile may publish at a declared source-chunk wave
threshold; every remaining partial tile must publish after all dispatch.
Communication CTAs may steal the same stage task a computation CTA would
claim, with the largest local stage's execution-group and storage requirement.
One CTA has not been shown to execute the entire 128-row FFN tile.

The structural analysis derives, for R ranks, T items/rank, K distinct
experts/item, E experts, B rows/tile, L=E/R experts/rank and at most W source
waves:

- remote payload slots/rank: `(R-1)T`;
- packed route rows/rank: at most `RTK`;
- rows/expert: at most `RT`, **provided runtime admission checks distinct
  expert IDs per item**;
- return slots/source rank: `TK`;
- logical tile slots/destination rank: at most
  `min(RTK, ceil(RTK/B) + L*W - 1)` with wave-end partial publication, or
  `min(RTK, ceil(RTK/B) + L - 1)` when the threshold equals B and only terminal
  partial publication remains.
- stage task slots/destination rank: the logical tile bound multiplied by
  the sum of each local Program stage's ProgramMap CTA count;
- stage completion slots/destination rank: the logical tile bound multiplied
  by the number of ordered Program stages.

The task bound counts each full B-row publication and at most one additional
partial fragment per expert per allowed wave. Because a nonempty fragment
consumes a route row, the final `-1` is safe even when `RTK` is divisible by
B. The analysis derives each stage's finite CTA grid from its own ProgramMap,
then reports the largest execution-group count and shared/tensor allocation
among the sequential stages. A stage with a persistent, cooperative or
implicit grid is refused here; its work-unit decomposition is not proven.
The Target owns the lane width and any actual cooperative residency decision.

`RankedTileAnalysis.check_plan` replays one materialized task plan against
the declared route IDs. It checks every source/item/route exactly once,
expert placement, dense per-expert tile indices, valid-row extents, the
wave-end partial threshold, terminal flush and the static capacities. Its
publication markers are a serial witness to when rows are *eligible*;
they do not establish live CUDA memory ordering or concurrent progress.

Using the separately developed model-width Cake Program and source combine
Schedule with EP4, T512, K8, E128, B128, W4 and a 64-row wave threshold
produces 1,536 payload slots, 16,384 packed rows, 255 **logical tiles**,
46,920 **stage tasks**, 765 stage-completion slots and 4,096 return slots
per rank. The three model FFN stages require 24 up/gate, 128 activation and
32 down CTAs per logical tile. Steal requires at least six execution groups, 49,152
bytes of SMEM and 32,768 bytes of TMEM. These are **safe capacities and
resource requirements**, not allocations or achieved occupancy. At the
no-early-flush threshold B128 the logical-tile bound is 159 and the stage-task
bound is 29,256. The CPU route
packing study observed 64 logical tiles per destination rank under the 64-row
threshold, requiring 11,776 stage work units per destination rank.
The retained 16,384-route synthetic input and both CPU tile manifests
replayed through this checker: the 64-row policy has 64 tasks per owner
(128 early partial plus 128 terminal tasks total), while the no-early policy
has owner counts 47/53/42/52 (70 full plus 124 terminal tasks total).

The current core API deliberately refuses `lower_ranked_tiles`: no backend
yet proves device peer ownership, system release/acquire handoffs, exact
CTA residency, partial-bin flush completion, return-key uniqueness in a
live queue, or warp-uniform steal termination. A runtime route-domain check
and an exact Target qualification must precede such emission. Compiler
analysis alone does not certify deadlock freedom or performance.
