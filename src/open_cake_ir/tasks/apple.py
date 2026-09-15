"""Metal-only view of the one device registry in `open_cake_ir.tasks.devices`.

Apple-specific code asks here so that a Metal path cannot accidentally be handed a CUDA
or AMDGCN row. It holds no rows of its own: a second table is what let two task families
freeze at "Metal only" while five others admitted every backend, and a view cannot drift
from what it is a view of.
"""
from __future__ import annotations

from open_cake_ir.tasks.devices import BACKENDS as _ALL

BACKENDS = {name: device for name, device in _ALL.items() if device["route"] == "metal"}


def backend_for_target(target: object) -> str | None:
    """Name the single Metal backend that admits one Compiler target, or None."""
    return next((name for name, device in BACKENDS.items() if device["target"] == target), None)


def device_name(target: object) -> str:
    """Return the exact device a task admits for one bound Metal target."""
    backend = backend_for_target(target)
    if backend is None:
        raise ValueError("Workload target has no admitted Apple device")
    return BACKENDS[backend]["device_name"]
