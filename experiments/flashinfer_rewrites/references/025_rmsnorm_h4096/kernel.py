import torch
import triton
import triton.language as tl

HIDDEN = 4096
EPS = 1e-5

# Persistent multi-row autotuned kernel for prefill (bs >= 512).
@triton.autotune(
    configs=[
        triton.Config({"ROWS_PER_TILE": 4}, num_warps=4, num_stages=3),
        triton.Config({"ROWS_PER_TILE": 8}, num_warps=4, num_stages=3),
        triton.Config({"ROWS_PER_TILE": 4}, num_warps=8, num_stages=3),
    ],
    key=[],
)
@triton.jit
def _rmsnorm_persistent_kernel(
    X, W, Y,
    n_rows,
    eps,
    HIDDEN_SIZE: tl.constexpr,
    ROWS_PER_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)

    cols = tl.arange(0, HIDDEN_SIZE)
    w = tl.load(W + cols).to(tl.float32)

    start = pid * ROWS_PER_TILE
    stride = grid_size * ROWS_PER_TILE
    for tile_start in tl.range(start, n_rows, stride):
        for tr in tl.static_range(0, ROWS_PER_TILE):
            row = tile_start + tr
            if row < n_rows:
                row_off = row * HIDDEN_SIZE
                x = tl.load(X + row_off + cols).to(tl.float32)
                ss = tl.sum(x * x, axis=0) / HIDDEN_SIZE
                inv_rms = tl.rsqrt(ss + eps)
                y = (x * inv_rms) * w
                tl.store(Y + row_off + cols, y.to(tl.bfloat16))


# One-program-per-row kernel — used for small AND mid batches (bs < 512).
# Round-1 evidence (ins-bn-1, ins-sc-1): the persistent grid `max(1, batch//4)`
# launched ≤42 programs at bs=170, leaving the B200's 592 warp slots idle.
# Per-row launches scale 1:1 with batch and saturate naturally up to ~592.
@triton.jit
def _rmsnorm_small_kernel(
    X, W, Y,
    eps,
    HIDDEN_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, HIDDEN_SIZE)
    row_off = row * HIDDEN_SIZE
    x = tl.load(X + row_off + cols).to(tl.float32)
    w = tl.load(W + cols).to(tl.float32)
    ss = tl.sum(x * x, axis=0) / HIDDEN_SIZE
    inv_rms = tl.rsqrt(ss + eps)
    y = (x * inv_rms) * w
    tl.store(Y + row_off + cols, y.to(tl.bfloat16))


_SM_COUNT = None


def _sm_count():
    global _SM_COUNT
    if _SM_COUNT is None:
        _SM_COUNT = torch.cuda.get_device_properties(0).multi_processor_count
    return _SM_COUNT


# Fork threshold raised from 32 → 512. At B200's 148 SMs / 592 warp slots,
# one-program-per-row covers bs up to ~592 without occupancy loss; round-1
# valley (bs 34..170) sat in the gap.
FORK_THRESHOLD = 512


@torch.no_grad()
def run(hidden_states, weight):
    batch_size, hidden_size = hidden_states.shape
    assert hidden_size == HIDDEN
    x = hidden_states.contiguous()
    w = weight.contiguous()
    y = torch.empty_like(x)

    if batch_size < FORK_THRESHOLD:
        _rmsnorm_small_kernel[(batch_size,)](
            x, w, y, EPS, HIDDEN_SIZE=HIDDEN,
        )
    else:
        sm = _sm_count()
        # Cap at SM*4 = 592 programs; autotune picks ROWS_PER_TILE.
        # Use ROWS_PER_TILE=4 as the divisor so the grid is sized for the
        # smallest tile in the autotune config space.
        grid = min(sm * 4, max(1, batch_size // 4))
        _rmsnorm_persistent_kernel[(grid,)](
            x, w, y,
            batch_size,
            EPS,
            HIDDEN_SIZE=HIDDEN,
        )
    return y


def kernel_function(hidden_states, weight):
    return run(hidden_states, weight)
