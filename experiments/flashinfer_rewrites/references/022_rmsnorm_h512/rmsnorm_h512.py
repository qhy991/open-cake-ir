import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Structurally novel design vs. the existing pool:
#
#   step_1_large.py  : grid=(B,)               , 1 row per program
#   step_2_small.py  : grid=(cdiv(B,RPB),)     , RPB rows per program
#
# Both reload the 1 KB weight tensor `O(num_programs)` times. For
# B=11949/14521 (the workloads that dominate the geomean), that is
# `O(B)` (step_1) or `O(B/RPB)` (step_2) reloads -- the L1 will hit, but the
# instruction stream still issues a load + cast + broadcast per program.
#
# This candidate is *persistent*: grid is bounded by SM count, each program
# loads `w` ONCE (kept in registers / L1 across the program's lifetime), then
# walks `tiles_per_program` row-tiles in a host-free loop. For B=14521 the
# weight is read ~num_programs ~= 296 times total instead of ~14521 times, and
# the launch-overhead floor is amortized across hundreds of tiles.
#
# Tiny-B falls back to a lean single-row kernel because there isn't enough
# work to fill a persistent grid, and the inner walker loop is pure overhead
# when each program owns one tile. This satisfies the skill-memory rule
# "cannot use one config for all M" via a B-bucketed dispatch.
# ---------------------------------------------------------------------------


# ---------- non-persistent path: tiny B (B <= NUM_SMS roughly) -------------

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
    ],
    key=["HIDDEN_SIZE"],
)
@triton.jit
def _rmsnorm_tiny_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_xb, stride_xh,
    stride_yb, stride_yh,
    EPS: tl.constexpr,
    INV_HIDDEN: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)

    x_ptrs = X_ptr + row * stride_xb + cols * stride_xh
    y_ptrs = Y_ptr + row * stride_yb + cols * stride_yh
    w_ptrs = W_ptr + cols

    x_bf = tl.load(x_ptrs, eviction_policy="evict_first")
    w_bf = tl.load(w_ptrs, eviction_policy="evict_last")

    x_f32 = x_bf.to(tl.float32)
    w_f32 = w_bf.to(tl.float32)

    sq = x_f32 * x_f32
    mean_sq = tl.sum(sq, axis=0) * INV_HIDDEN
    inv_rms = tl.rsqrt(mean_sq + EPS)

    y_f32 = x_f32 * inv_rms * w_f32
    tl.store(y_ptrs, y_f32.to(tl.bfloat16))


# ---------- persistent path: medium / large B ------------------------------
#
# grid = (NUM_PROGRAMS,). Each program:
#   1. loads w once,
#   2. iterates tile_id = pid, pid + grid, pid + 2*grid, ... while in-bounds,
#   3. each iteration processes ROWS_PER_BLOCK consecutive rows.
#
# `ROWS_PER_BLOCK` is autotuned. Autotune key includes a coarse B bucket so
# tiny vs. medium vs. large pick different (rows, warps, stages).

@triton.autotune(
    configs=[
        # ROWS_PER_BLOCK = 1
        triton.Config({"ROWS_PER_BLOCK": 1}, num_warps=1, num_stages=2),
        triton.Config({"ROWS_PER_BLOCK": 1}, num_warps=2, num_stages=2),
        triton.Config({"ROWS_PER_BLOCK": 1}, num_warps=4, num_stages=2),
        # ROWS_PER_BLOCK = 2
        triton.Config({"ROWS_PER_BLOCK": 2}, num_warps=2, num_stages=2),
        triton.Config({"ROWS_PER_BLOCK": 2}, num_warps=4, num_stages=2),
        triton.Config({"ROWS_PER_BLOCK": 2}, num_warps=4, num_stages=3),
        # ROWS_PER_BLOCK = 4
        triton.Config({"ROWS_PER_BLOCK": 4}, num_warps=2, num_stages=2),
        triton.Config({"ROWS_PER_BLOCK": 4}, num_warps=4, num_stages=2),
        triton.Config({"ROWS_PER_BLOCK": 4}, num_warps=4, num_stages=3),
        triton.Config({"ROWS_PER_BLOCK": 4}, num_warps=8, num_stages=2),
        # ROWS_PER_BLOCK = 8
        triton.Config({"ROWS_PER_BLOCK": 8}, num_warps=4, num_stages=2),
        triton.Config({"ROWS_PER_BLOCK": 8}, num_warps=4, num_stages=3),
        triton.Config({"ROWS_PER_BLOCK": 8}, num_warps=8, num_stages=2),
    ],
    key=["HIDDEN_SIZE", "B_BUCKET"],
)
@triton.jit
def _rmsnorm_persistent_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_xb, stride_xh,
    stride_yb, stride_yh,
    B,                              # runtime row count
    EPS: tl.constexpr,
    INV_HIDDEN: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    B_BUCKET: tl.constexpr,         # autotune key only
    ROWS_PER_BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0).to(tl.int64)
    grid = tl.num_programs(axis=0).to(tl.int64)

    # Tile ceiling = cdiv(B, ROWS_PER_BLOCK). Computed in-kernel from the
    # runtime row count `B` and the autotuned constexpr `ROWS_PER_BLOCK`
    # because Triton does not accept meta-lambdas for runtime/constexpr
    # kernel arguments (only for `grid` and autotune Config values). The
    # launcher's grid is bounded by this same expression, so the two stay
    # consistent at every ROWS_PER_BLOCK.
    NUM_TILES = (B + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK

    cols = tl.arange(0, BLOCK_SIZE)
    row_lane = tl.arange(0, ROWS_PER_BLOCK).to(tl.int64)
    w_ptrs = W_ptr + cols

    # --- weight loaded ONCE for this program's lifetime ----------------------
    # It will sit in L1 across every iteration of the walker loop; we still
    # explicitly hoist the upcast so the compiler doesn't redo it per tile.
    w_bf = tl.load(w_ptrs, eviction_policy="evict_last")
    w_f32 = w_bf.to(tl.float32)

    # --- host-free walker over tiles ----------------------------------------
    tile = pid
    while tile < NUM_TILES:
        row_base = tile * ROWS_PER_BLOCK
        row_offs = row_base + row_lane
        row_mask = row_offs < B  # only the very last tile can be partial

        x_ptrs = X_ptr + row_offs[:, None] * stride_xb + cols[None, :] * stride_xh
        y_ptrs = Y_ptr + row_offs[:, None] * stride_yb + cols[None, :] * stride_yh

        x_bf = tl.load(
            x_ptrs,
            mask=row_mask[:, None],
            other=0.0,
            eviction_policy="evict_first",
        )
        x_f32 = x_bf.to(tl.float32)

        sq = x_f32 * x_f32
        mean_sq = tl.sum(sq, axis=1, keep_dims=True) * INV_HIDDEN  # [R,1]
        inv_rms = tl.rsqrt(mean_sq + EPS)                          # [R,1]

        y_f32 = x_f32 * inv_rms * w_f32[None, :]
        tl.store(y_ptrs, y_f32.to(tl.bfloat16), mask=row_mask[:, None])

        tile += grid


# ---------- launcher --------------------------------------------------------

def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


# Cache device SM count once -- torch.cuda.get_device_properties is not
# free and we hit run() once per workload row.
_NUM_SMS_CACHE = {}


def _num_sms(device) -> int:
    key = device.index if device.index is not None else 0
    n = _NUM_SMS_CACHE.get(key)
    if n is None:
        n = torch.cuda.get_device_properties(device).multi_processor_count
        _NUM_SMS_CACHE[key] = n
    return n


def _b_bucket(b: int) -> int:
    if b <= 64:
        return 0
    if b <= 1024:
        return 1
    if b <= 16384:
        return 2
    return 3


@torch.no_grad()
def run(hidden_states: torch.Tensor, weight: torch.Tensor, output: torch.Tensor):
    """
    Destination-passing-style fused RMSNorm.

    Args:
        hidden_states: [B, H] bf16, H == 512.
        weight:        [H]    bf16.
        output:        [B, H] bf16, pre-allocated by the harness.
    """
    assert hidden_states.dim() == 2, "expected 2-D hidden_states"
    B, H = hidden_states.shape
    assert H == 512, f"this kernel specializes to hidden_size=512, got {H}"
    assert weight.shape == (H,), f"weight shape mismatch: {weight.shape} vs ({H},)"
    assert output.shape == hidden_states.shape, "output shape mismatch"
    assert hidden_states.is_cuda and weight.is_cuda and output.is_cuda

    if B == 0:
        return

    BLOCK_SIZE = _next_pow2(H)  # 512, statically equal to HIDDEN_SIZE

    # --- tiny-B path: not enough rows to benefit from a persistent walker ---
    # The walker's while-loop body and the trailing mask check are pure
    # overhead when every program owns exactly one tile. Use the lean
    # non-persistent kernel instead.
    num_sms = _num_sms(hidden_states.device)
    if B <= num_sms:
        _rmsnorm_tiny_kernel[(B,)](
            hidden_states, weight, output,
            hidden_states.stride(0), hidden_states.stride(1),
            output.stride(0), output.stride(1),
            EPS=1e-6,
            INV_HIDDEN=1.0 / H,
            HIDDEN_SIZE=H,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return

    # --- persistent path: medium / large B ----------------------------------
    B_BUCKET = _b_bucket(B)

    def grid(meta):
        rpb = meta["ROWS_PER_BLOCK"]
        num_tiles = (B + rpb - 1) // rpb
        # SM-bounded: never launch more programs than tiles, and never launch
        # more than ~2 waves of SMs (any further amortization is wasted).
        return (min(num_tiles, num_sms * 2),)

    # NUM_TILES is computed inside the kernel from B and the autotuned
    # ROWS_PER_BLOCK (see comment in the kernel body): Triton rejects
    # meta-lambdas for runtime/constexpr kernel arguments, so the tile
    # ceiling cannot be passed host-side when ROWS_PER_BLOCK is autotuned.
    _rmsnorm_persistent_kernel[grid](
        hidden_states,
        weight,
        output,
        hidden_states.stride(0), hidden_states.stride(1),
        output.stride(0), output.stride(1),
        B,
        EPS=1e-6,
        INV_HIDDEN=1.0 / H,
        HIDDEN_SIZE=H,
        BLOCK_SIZE=BLOCK_SIZE,
        B_BUCKET=B_BUCKET,
    )
