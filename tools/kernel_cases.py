"""One call's arguments for any Schedule, and the answer for each profile that has one.

Shared because two instruments need it and neither owns it: `observe_lowered_kernel.py`
compares the result against the oracle here, and `profile_lowered_kernel.py` throws the
result away and measures the launch. Writing the inputs twice would let one instrument
profile a kernel the other never checked.

The split is where the ownership is. **Shapes, dtypes and argument order belong to the
Schedule** -- the emitted host function already validates them against the same
declarations -- so `build_inputs` derives them rather than restating them per operator,
and a new operator adds nothing here for its tensors. **The answer belongs to the
operator**, and no amount of declaration produces it, so that is the one thing an oracle
is.

An oracle reads the inputs that were built rather than assuming a relation between them.
Flash-KMeans is called in production with a centroid-norm vector equal to the squared row
sums of its centroids, and the earlier version of this file constructed one -- which made
the check test the kernel on the single input family it was designed for. The kernel's
contract is arithmetic over whatever vector it is handed, and that is what is checked.
"""

from __future__ import annotations

_TORCH_DTYPE = {
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
    "int32": "int32",
}


def global_shapes(document: dict) -> dict[str, tuple[int, ...]]:
    """Each global buffer's declared shape."""

    return {
        buffer["name"]: tuple(buffer["shape"])
        for buffer in document["buffers"]
        if buffer["space"] == "global"
    }


def build_inputs(document: dict, torch) -> tuple:
    """One call's arguments, in the order the entry point takes them.

    The emitted host function takes the global buffers in declaration order, so that
    order is the argument order and nothing here needs to know it. An input buffer gets
    values; anything the kernel writes gets zeros, because its contents before the launch
    are not part of what is being checked.
    """

    arguments = []
    for buffer in document["buffers"]:
        if buffer["space"] != "global":
            continue
        dtype = getattr(torch, _TORCH_DTYPE[buffer["dtype"]])
        shape = tuple(buffer["shape"])
        if buffer["mode"] == "input" and dtype.is_floating_point:
            arguments.append(torch.randn(shape, dtype=dtype, device="cuda"))
        else:
            arguments.append(torch.zeros(shape, dtype=dtype, device="cuda"))
    return tuple(arguments)


def _flash_kmeans_oracle(inputs, torch):
    """Float32 argmin of the distance the kernel is asked to minimise.

    The token norm is elided because it is constant per row and cannot change an argmin.
    The centroid-norm vector is whatever was handed in, so this checks the arithmetic and
    not a relation the caller happened to satisfy.
    """

    tokens, centroids, centroid_sq, _, _ = inputs
    distance = centroid_sq[None, :] - 2.0 * (
        tokens.to(torch.float32) @ centroids.to(torch.float32).t()
    )
    return torch.argmin(distance, dim=1).to(torch.int32), distance


def _softmax_oracle(inputs, torch):
    """`torch.softmax`, so the comparison is against an implementation that did not come
    from the same reasoning the Schedule did."""

    return torch.softmax(inputs[0], dim=-1), None


def _rmsnorm_oracle(inputs, torch):
    """Written from the definition rather than from the Schedule's own decomposition, so
    it does not inherit the Schedule's reasoning about where epsilon goes."""

    x, gamma, _ = inputs
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * gamma, None


def _layernorm_oracle(inputs, torch):
    """`torch.nn.functional.layer_norm`, so the comparison is against an implementation
    that did not come from the Schedule's own decomposition of the variance."""

    x, gamma, beta, _ = inputs
    return (
        torch.nn.functional.layer_norm(x, (x.shape[-1],), gamma, beta, eps=1e-5),
        None,
    )


ORACLES = {
    "layernorm_b8_smoke": _layernorm_oracle,
    "flash_kmeans_assignment_full": _flash_kmeans_oracle,
    "softmax_b8_smoke": _softmax_oracle,
    "rmsnorm_b8_smoke": _rmsnorm_oracle,
}
