"""Current input construction layered over the frozen calibration helper.

The retained helper remains the authority for every dtype it knew.  This adapter owns
only FP8 E4M3 and related scale inputs, which were added after those bytes became part of
historical calibration evidence.
"""

from __future__ import annotations

from kernel_cases import _TORCH_DTYPE as _RETAINED_TORCH_DTYPE
from kernel_cases import build_inputs as _retained_build_inputs


def build_inputs(document: dict, torch) -> tuple:
    if not any(buffer["dtype"] == "fp8_e4m3" for buffer in document["buffers"]):
        return _retained_build_inputs(document, torch)

    arguments = []
    for buffer in document["buffers"]:
        if buffer["space"] != "global":
            continue
        dtype_name = (
            "float8_e4m3fn"
            if buffer["dtype"] == "fp8_e4m3"
            else _RETAINED_TORCH_DTYPE[buffer["dtype"]]
        )
        dtype = getattr(torch, dtype_name)
        shape = tuple(buffer["shape"])
        if buffer["mode"] != "input":
            value = torch.zeros(shape, dtype=dtype, device="cuda")
        elif buffer.get("scale_of") is not None:
            # Non-unit, positive scales make a dropped or mis-associated scale visible.
            value = 0.05 + 0.15 * torch.rand(shape, dtype=dtype, device="cuda")
        elif buffer["dtype"] == "fp8_e4m3":
            # PyTorch does not sample directly into every float8 dtype.
            value = torch.randn(shape, dtype=torch.float32, device="cuda").to(dtype)
        elif dtype.is_floating_point:
            value = torch.randn(shape, dtype=dtype, device="cuda")
        else:
            value = torch.zeros(shape, dtype=dtype, device="cuda")
        arguments.append(value)
    return tuple(arguments)
