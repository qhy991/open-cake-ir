# Tile-keyed ranked effects: structural admission

`RankedTileEffects` is the shared Compiler contract for moving from one
token-route work claim to an expert-owned tile task. It composes a complete
local Program with the source-rank weighted-combine Schedule. It introduces
no opaque MoE operation and does not reinterpret the frozen ranked-mailbox
version-1 effects.

The payload is keyed by source item and destination rank. Each expert-bin row
retains its `(source_rank, item, route)` identity; a task is keyed by
`(destination_rank, expert, tile_index)` and publishes a valid-row count.
Return slots retain the original source/item/route key. Full tiles may publish
when filled; a partial tile may publish at a declared source-chunk wave
threshold; every remaining partial tile must publish after all dispatch.
Communication CTAs may steal only the same tile task a computation CTA would
claim, with the complete local Program's resource requirement.

The structural analysis derives, for R ranks, T items/rank, K distinct
experts/item, E experts, B rows/tile, L=E/R experts/rank and at most W source
waves:

- remote payload slots/rank: `(R-1)T`;
- packed route rows/rank: at most `RTK`;
- rows/expert: at most `RT`, **provided runtime admission checks distinct
  expert IDs per item**;
- return slots/source rank: `TK`;
- tile task slots/destination rank: at most
  `min(RTK, ceil(RTK/B) + L*W - 1)` with wave-end partial publication, or
  `min(RTK, ceil(RTK/B) + L - 1)` when the threshold equals B and only terminal
  partial publication remains.

The task bound counts each full B-row publication and at most one additional
partial fragment per expert per allowed wave. Because a nonempty fragment
consumes a route row, the final `-1` is safe even when `RTK` is divisible by
B. The analysis also reports the largest execution-group count and shared/
tensor allocation among the sequential local Program stages. The Target owns
the lane width and any actual cooperative residency decision.

Using the separately developed model-width Cake Program and source combine
Schedule with EP4, T512, K8, E128, B128, W4 and a 64-row wave threshold
produces 1,536 payload slots, 16,384 packed rows, 255 tile tasks and 4,096
return slots per rank. Steal requires at least six execution groups, 49,152
bytes of SMEM and 32,768 bytes of TMEM. These are **safe capacities and
resource requirements**, not allocations or achieved occupancy. At the
no-early-flush threshold B128 the tile-task bound is 159. The CPU route
packing study observed 64 tasks per destination rank under the 64-row
threshold, below the static bound.

The current core API deliberately refuses `lower_ranked_tiles`: no backend
yet proves device peer ownership, system release/acquire handoffs, exact
CTA residency, partial-bin flush completion, return-key uniqueness in a
live queue, or warp-uniform steal termination. A runtime route-domain check
and an exact Target qualification must precede such emission. Compiler
analysis alone does not certify deadlock freedom or performance.
