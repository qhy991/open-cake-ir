"""Allocation facts from agent-gpu-broker, independent of the kernel API.

The broker is trusted under GPU Infra's cooperating-Unix-user boundary. It
overwrites these facts in the child environment. Device admission still checks
the actual target; an allocation is not a hardware or timing qualification.
"""
from __future__ import annotations

import os
import re
from collections.abc import Mapping

from open_cake_ir.compiler.target import Vendor, declared_target

_BACKENDS = {Vendor.NVIDIA: "nvidia", Vendor.APPLE: "metal",
             Vendor.AMD: "amd", Vendor.HYGON: "hygon"}


def backend_for_target(target: str) -> str:
    return _BACKENDS[declared_target(target).vendor]


def validate_allocation(value, *, target: str, job_id: str) -> None:
    backend = backend_for_target(target)
    if (not isinstance(value, Mapping)
            or set(value) != {"job_id", "backend", "mode", "device_ids", "occupancy_scope"}
            or value["job_id"] != job_id
            or re.fullmatch(r"gpuq-[0-9a-f]{12}", job_id) is None
            or job_id == "gpuq-000000000000"
            or value["backend"] != backend or value["mode"] != "exclusive"
            or not isinstance(value["device_ids"], list) or len(value["device_ids"]) != 1
            or type(value["device_ids"][0]) is not int or value["device_ids"][0] < 0
            or value["occupancy_scope"] != ("cooperative" if backend == "metal" else "system")):
        raise ValueError("GPU Infra allocation differs from the exact target")


def observe_allocation(target: str) -> dict:
    ids = os.environ.get("GPUQ_DEVICE_IDS", "")
    if re.fullmatch(r"[0-9]+", ids) is None:
        raise ValueError("GPU Infra requires exactly one assigned device")
    value = {"job_id": os.environ.get("GPUQ_JOB_ID", ""),
             "backend": os.environ.get("GPUQ_BACKEND"),
             "mode": os.environ.get("GPUQ_MODE"), "device_ids": [int(ids)],
             "occupancy_scope": os.environ.get("GPUQ_OCCUPANCY_SCOPE")}
    validate_allocation(value, target=target, job_id=value["job_id"])
    if value["backend"] in {"amd", "hygon"}:
        if (os.environ.get("HIP_VISIBLE_DEVICES") != ids
                or os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("ROCR_VISIBLE_DEVICES")):
            raise ValueError("HIP visibility differs from broker allocation")
    elif value["backend"] == "nvidia" and os.environ.get("CUDA_VISIBLE_DEVICES") != ids:
        raise ValueError("CUDA visibility differs from broker allocation")
    elif value["backend"] == "metal" and (ids != "0" or os.environ.get("CUDA_VISIBLE_DEVICES")
                                           or os.environ.get("HIP_VISIBLE_DEVICES")):
        raise ValueError("Metal visibility differs from broker allocation")
    return value
