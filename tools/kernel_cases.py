"""How to build one call's inputs for each profile this repository emits.

Shared because two instruments need it and neither owns it: `observe_lowered_kernel.py`
compares the result against the oracle here, and `profile_lowered_kernel.py` throws the
result away and measures the launch. Writing the inputs twice would let one instrument
profile a kernel the other never checked.

Each case returns `(inputs, reference, distance)`. `distance` is None for a float-valued
kernel, where the question is how far off the answer is rather than whether the index it
chose was as good.
"""

from __future__ import annotations


def global_shapes(document: dict) -> dict[str, tuple[int, ...]]:
    """Each global buffer's declared shape, which is what a case builds tensors from."""

    return {
        buffer["name"]: tuple(buffer["shape"])
        for buffer in document["buffers"]
        if buffer["space"] == "global"
    }


def _flash_kmeans_case(shapes, torch):
    """Inputs and the float32 answer for the assignment kernel.

    The oracle is independent of the kernel: argmin of squared euclidean distance with
    the token norm elided, which is constant per row and cannot change the argmin.
    """

    tokens_n, dimension = shapes["tokens"]
    centroids_k, _ = shapes["centroids"]
    tokens = torch.randn(tokens_n, dimension, device="cuda", dtype=torch.bfloat16)
    centroids = torch.randn(centroids_k, dimension, device="cuda", dtype=torch.bfloat16)
    left, right = tokens.to(torch.float32), centroids.to(torch.float32)
    distance = (right * right).sum(dim=1)[None, :] - 2.0 * (left @ right.t())
    inputs = (
        tokens,
        centroids,
        (right * right).sum(dim=1).contiguous(),
        torch.empty(tokens_n, centroids_k, device="cuda", dtype=torch.float32),
        torch.full((tokens_n,), -1, device="cuda", dtype=torch.int32),
    )
    return inputs, torch.argmin(distance, dim=1).to(torch.int32), distance


def _softmax_case(shapes, torch):
    """Inputs and the float32 answer for the row softmax.

    torch.softmax is the oracle rather than a hand-written exp-and-divide, so the
    comparison is against an implementation that did not come from the same reasoning
    the Schedule did.
    """

    x = torch.randn(shapes["x"], device="cuda", dtype=torch.float32)
    reference = torch.softmax(x, dim=-1)
    return (x, torch.empty_like(x)), reference, None


def _rmsnorm_case(shapes, torch):
    """Inputs and the float32 answer for the row RMSNorm.

    The oracle is written from the definition rather than from the Schedule's own
    decomposition, so it does not inherit the Schedule's reasoning about epsilon
    placement -- which is the part a normalization gets wrong.
    """

    x = torch.randn(shapes["x"], device="cuda", dtype=torch.float32)
    gamma = torch.randn(shapes["gamma"], device="cuda", dtype=torch.float32)
    inverse = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    return (x, gamma, torch.empty_like(x)), x * inverse * gamma, None


CASES = {
    "rmsnorm_b8_smoke": _rmsnorm_case,
    "flash_kmeans_assignment_full": _flash_kmeans_case,
    "softmax_b8_smoke": _softmax_case,
}
