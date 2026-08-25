"""Current input construction layered over the frozen calibration helper.

The retained helper remains the authority for every dtype it knew.  This adapter owns
only FP8 E4M3, related scale inputs, and runtime extent inputs added after those bytes
became part of historical calibration evidence.
"""

from __future__ import annotations

from math import prod

from kernel_cases import _TORCH_DTYPE as _RETAINED_TORCH_DTYPE
from kernel_cases import build_inputs as _retained_build_inputs


def build_inputs(document: dict, torch) -> tuple:
    entry_point = document.get("lowering", {}).get("entry_point")
    indexed_routes = {
        "cake_indexed_gather_b8_smoke",
        "cake_kda_weighted_combine_b8_smoke",
        "cake_atomic_reservation_b8_smoke",
    }
    extent_contracts = {
        buffer["valid_extent"]["buffer"]: buffer["shape"][
            buffer["valid_extent"]["dimension"]
        ]
        for buffer in document["buffers"]
        if buffer.get("valid_extent") is not None
    }
    if entry_point not in indexed_routes and not extent_contracts and not any(
        buffer["dtype"] == "fp8_e4m3" for buffer in document["buffers"]
    ):
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
        if entry_point in indexed_routes and buffer["name"] == "expert_ids":
            # Rotate valid expert ids per token and retain KDA's -1 no-route sentinel.
            # This makes a Cartesian-product lowering, a fixed coordinate and missing
            # lower-bound mask all disagree with the oracle.
            base = torch.tensor(
                [0, 1, 2, 3, 3, 2, 1, -1], dtype=dtype, device="cuda"
            )
            value = torch.stack(
                [torch.roll(base, shifts=token) for token in range(shape[0])]
            )
        elif entry_point in indexed_routes and buffer["name"] == "row_ids":
            slots = torch.arange(shape[1], dtype=dtype, device="cuda")
            value = torch.stack(
                [(slots + token) % 8 for token in range(shape[0])]
            )
        elif (
            entry_point == "cake_kda_weighted_combine_b8_smoke"
            and buffer["name"] == "expert_rows"
        ):
            # Small integers and power-of-two route weights make the FP32 sum and final
            # BF16 rounding exact. The observation then isolates indexing, masking,
            # weighting and reduction instead of accepting an accumulation-order delta.
            value = (
                torch.arange(prod(shape), device="cuda")
                .reshape(shape)
                .remainder(17)
                .sub(8)
                .to(dtype)
            )
        elif (
            entry_point == "cake_kda_weighted_combine_b8_smoke"
            and buffer["name"] == "route_weights"
        ):
            base = torch.tensor(
                [1.0, 0.5, -0.25, 2.0, -1.0, 0.125, 0.25, -0.5],
                dtype=dtype,
                device="cuda",
            )
            value = torch.stack(
                [torch.roll(base, shifts=token) for token in range(shape[0])]
            )
        elif (
            entry_point == "cake_atomic_reservation_b8_smoke"
            and buffer["name"] == "counts"
        ):
            # Non-zero caller state proves the atomic returns the preceding value rather
            # than an index synthesized from a zero-based lane or program coordinate.
            value = torch.tensor([3, 5, 7, 11], dtype=dtype, device="cuda")
        elif buffer["name"] in extent_contracts:
            capacity = extent_contracts[buffer["name"]]
            # Empty, partial, full and another partial group are all observable. The
            # relation verifier proves this input has four entries; the values derive
            # from its one owned fact, capacity, rather than from workload names.
            if shape != (4,) or capacity < 2:
                raise ValueError(
                    "no observation input is registered for this extent contract"
                )
            value = torch.tensor(
                [0, capacity // 2 - 1, capacity, capacity // 2 + 1],
                dtype=dtype,
                device="cuda",
            )
        elif buffer["mode"] != "input":
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
