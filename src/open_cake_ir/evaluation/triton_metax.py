"""Exact C550 admission under the existing local single-device broker."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os

from open_cake_ir.compiler.target import CodeObject, declared_target


@dataclass(frozen=True)
class MetaxDeviceAdmission:
    broker_job_id: str
    target: str
    device_arch: str
    device_name: str
    warp_size: int
    pci_bus_id: str
    gpu_uuid: str | None = None


def observe_local_metax(target_id: str) -> MetaxDeviceAdmission:
    from .local_broker import observe_local_job

    target = declared_target(target_id)
    if target.code_object is not CodeObject.MCFATBIN or os.environ.get("GPUQ_JOB_ID"):
        raise ValueError("MACA admission requires its own local-broker target")
    job_id = observe_local_job("maca")
    import torch
    from triton.runtime import driver

    if (not isinstance(getattr(torch.version, "maca", None), str)
            or not torch.version.maca or torch.cuda.device_count() != 1):
        raise ValueError("MACA admission requires a MACA PyTorch and one visible device")
    properties = torch.cuda.get_device_properties(0)
    actual = driver.active.get_current_target()
    if ((actual.backend, actual.arch, actual.warp_size) != ("maca", target.triton_arch, target.warp_size)
            or properties.name not in target.device_names or properties.warp_size != target.warp_size):
        raise ValueError("MACA framework target differs from the declared device")
    # The compatibility API reports (8,0); the native runtime identifies XCORE1002
    # with (10,2). These are two different API facts, never CUDA target admission.
    api = ctypes.CDLL("libmcruntime.so")
    query = api.mcDeviceGetAttribute
    query.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    query.restype = ctypes.c_int
    values = []
    # SDK mcDeviceAttributeComputeCapabilityMajor/Minor, observed from 3.5.3 headers.
    for attribute in (75, 76):
        value = ctypes.c_int()
        status = query(ctypes.byref(value), attribute, 0)
        if status or value.value < 0:
            raise ValueError(f"MACA native architecture query failed with status {status}")
        values.append(value.value)
    native_arch = f"xcore{values[0] * 100 + values[1]}"
    if native_arch != target.target_id:
        raise ValueError(f"MACA native target {native_arch!r} differs from {target.target_id!r}")
    pci = f"{properties.pci_domain_id:04x}:{properties.pci_bus_id:02x}:{properties.pci_device_id:02x}"
    return MetaxDeviceAdmission(job_id, target_id, native_arch, properties.name,
                                properties.warp_size, pci)
