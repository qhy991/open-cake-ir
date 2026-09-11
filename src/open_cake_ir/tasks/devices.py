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

import math

from open_cake_ir.tasks.apple import BACKENDS as _APPLE

# The largest `tl.arange` span the Triton backend admits, from its own refusal text.
TRITON_MAXIMUM_TILE = 1 << 20
# `snapshotPayloadLimit` in evaluation/metal/observer.swift. The native observer charges a
# participant's whole tensor ABI for every launch it holds in one snapshot cohort, so the
# cohort's pending payload is that ABI times the cohort's route calls.
SNAPSHOT_PAYLOAD_LIMIT = 64 * 1024 * 1024

BACKENDS = {
    "metal-m1-pro": {"target": "apple_gpu_family7", "device_name": "Apple M1 Pro",
                     "provenance_token": "M1_Pro", "route": "metal",
                     "tanh_contract": "metal.precise.tanh.f32", "power_of_two_width": False},
    "metal-m2": {"target": "apple_gpu_family8", "device_name": "Apple M2",
                 "provenance_token": "M2", "route": "metal",
                 "tanh_contract": "metal.precise.tanh.f32", "power_of_two_width": False},
    "metal-m4": {"target": "apple_gpu_family9", "device_name": "Apple M4",
                 "provenance_token": "M4", "route": "metal",
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


def admit_cohort_payload(workload, case_id: str, route_calls_per_cohort: int) -> None:
    """Refuse a shape whose snapshot cohort cannot fit the observer's payload bound.

    The bound is enforced inside the pinned native observer, which refuses the cohort
    before a single dispatch. A task that exceeds it therefore fails deterministically at
    its first evaluation, after the campaign has already spent provider tokens authoring
    candidates for it -- which is what F-2026-09-10-002 recorded for every multi-buffer
    optimizer task at the launcher's default shape. Checking the same arithmetic here
    turns that into a refusal at launch, naming the shape that would fit.

    This does not change the bound. Charging a cohort for what correctness actually needs
    is the real fix and it lives in the observer, whose bytes are pinned by the Executor
    host environment; until that happens this keeps campaigns from paying to discover it.
    """
    if type(route_calls_per_cohort) is not int or route_calls_per_cohort <= 0:
        raise ValueError("cohort route-call count must be a positive integer")
    tensors = workload.tensor_abi(case_id)
    elements = sum(math.prod(argument.shape) for argument in tensors)
    per_launch = elements * 4
    pending = per_launch * route_calls_per_cohort
    if pending > SNAPSHOT_PAYLOAD_LIMIT:
        admissible = SNAPSHOT_PAYLOAD_LIMIT // (len(tensors) * 4 * route_calls_per_cohort)
        raise ValueError(
            f"snapshot cohort would hold {pending / 2**20:.1f} MiB against the observer's "
            f"{SNAPSHOT_PAYLOAD_LIMIT // 2**20} MiB bound: this Workload declares "
            f"{len(tensors)} FP32 tensors of {elements // len(tensors)} elements and the "
            f"protocol holds {route_calls_per_cohort} route calls per cohort. At this ABI "
            f"the largest admissible tensor is {admissible} elements")
