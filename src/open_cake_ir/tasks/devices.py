"""Portable device registry: one backend names one Target, one route and its contracts.

A task's mathematics is device-independent, so a Workload should not be an Apple artifact
that a CUDA run has to re-derive. What actually differs between an Apple and an NVIDIA
run of the same Workload is small and enumerable, and this module is where it is written
down instead of being spread through each family's authoring code:

* the Compiler Target and the one device it admits;
* the lowering route the Schedule declares (`metal` or `triton`);
* the instruction contract for `tanh`, which each Target names in its own vocabulary --
  `metal.precise.tanh.f32` against Metal's named-precision function, `libdevice.tanh.f32`
  against the CUDA one. They are different functions and neither Target admits the other's
  spelling, which is the point of naming a contract at all;
* whether the route can tile a row whose width is not a power of two. The Metal emitter
  stripes any width over 32 lanes; Triton's `tl.arange` requires a positive power-of-two
  span, so a Workload frozen at an odd width has no Triton Schedule to compare against
  and this registry refuses to create one rather than emitting a Schedule that cannot
  lower.

`open_cake_ir.tasks.apple` stays the authority for the Apple entries; the assertion below
keeps this table from drifting away from it.
"""
from __future__ import annotations

from open_cake_ir.tasks.apple import BACKENDS as _APPLE

# The largest `tl.arange` span the Triton backend admits, from its own refusal text.
TRITON_MAXIMUM_TILE = 1 << 20

BACKENDS = {
    "metal-m1-pro": {"target": "apple_gpu_family7", "device_name": "Apple M1 Pro",
                     "provenance_token": "M1_Pro", "route": "metal",
                     "tanh_contract": "metal.precise.tanh.f32", "power_of_two_width": False},
    "metal-m2": {"target": "apple_gpu_family8", "device_name": "Apple M2",
                 "provenance_token": "M2", "route": "metal",
                 "tanh_contract": "metal.precise.tanh.f32", "power_of_two_width": False},
    "triton-b200": {"target": "sm_100a", "device_name": "NVIDIA B200",
                    "provenance_token": "B200", "route": "triton",
                    "tanh_contract": "libdevice.tanh.f32", "power_of_two_width": True},
    "triton-b300": {"target": "sm_103a", "device_name": "NVIDIA B300",
                    "provenance_token": "B300", "route": "triton",
                    "tanh_contract": "libdevice.tanh.f32", "power_of_two_width": True},
}

# One registry, not two: the Apple rows here must stay exactly what the Apple tasks use.
assert all(all(BACKENDS[name][field] == device[field] for field in device)
           for name, device in _APPLE.items()), "Apple backend registry drift"


def backend_for_target(target: object) -> str | None:
    """Name the single backend that admits one Compiler target, or None."""
    return next((name for name, device in BACKENDS.items() if device["target"] == target), None)


def device_name(target: object) -> str:
    """Return the exact device a task admits for one bound Target."""
    backend = backend_for_target(target)
    if backend is None:
        raise ValueError("Workload target has no admitted device")
    return BACKENDS[backend]["device_name"]


def admit_width(backend: str, columns: int) -> None:
    """Refuse a frozen row width the backend's route provably cannot tile."""
    device = BACKENDS[backend]
    if not device["power_of_two_width"]:
        return
    if columns > TRITON_MAXIMUM_TILE or columns & (columns - 1):
        raise ValueError(
            f"{backend} tiles a row with tl.arange, which requires a positive power-of-two "
            f"span no larger than {TRITON_MAXIMUM_TILE}; {columns} is not one")
