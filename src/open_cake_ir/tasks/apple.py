"""Apple backend registry shared by every task that targets a Metal device.

One backend names exactly one Compiler target and one admitted device; the provenance
token keeps every already frozen Workload document byte-identical.
"""
from __future__ import annotations

BACKENDS = {
    "metal-m1-pro": {"target": "apple_gpu_family7", "device_name": "Apple M1 Pro",
                     "provenance_token": "M1_Pro"},
    "metal-m2": {"target": "apple_gpu_family8", "device_name": "Apple M2",
                 "provenance_token": "M2"},
    "metal-m4": {"target": "apple_gpu_family9", "device_name": "Apple M4",
                 "provenance_token": "M4"},
}


def backend_for_target(target: object) -> str | None:
    """Name the single backend that admits one Compiler target, or None."""
    return next((name for name, device in BACKENDS.items() if device["target"] == target), None)


def device_name(target: object) -> str:
    """Return the exact device a task admits for one bound Metal target."""
    backend = backend_for_target(target)
    if backend is None:
        raise ValueError("Workload target has no admitted Apple device")
    return BACKENDS[backend]["device_name"]
