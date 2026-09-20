"""Exact C550 admission under the existing local single-device broker."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Mapping

from open_cake_ir.compiler.target import CodeObject, declared_target


@dataclass(frozen=True)
class MetaxDeviceAdmission:
    broker_job_id: str
    target: str
    device_arch: str
    device_name: str
    warp_size: int
    pci_bus_id: str
    runtime_library: str
    gpu_uuid: str | None = None


def validate_maca_admission(value, *, target_id: str, job_id: str) -> None:
    """Read the native identity retained by a MACA launch, without inventing a UUID."""
    target = declared_target(target_id)
    if (target.code_object is not CodeObject.MCFATBIN or not isinstance(value, Mapping)
            or not isinstance(job_id, str) or re.fullmatch(r'maca-[0-9a-f]{12}', job_id) is None
            or job_id == 'maca-000000000000' or value.get('broker_job_id') != job_id
            or value.get('target') != target_id or value.get('device_arch') != target_id
            or value.get('device_name') not in target.device_names
            or value.get('warp_size') != target.warp_size or value.get('gpu_uuid') is not None
            or not isinstance(value.get('pci_bus_id'), str)
            or re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}', value['pci_bus_id']) is None
            or not isinstance(value.get('runtime_library'), str)
            or not Path(value['runtime_library']).is_absolute()):
        raise ValueError('MACA recorded device admission differs')


def observe_local_metax(target_id: str, *, runtime_library: str) -> MetaxDeviceAdmission:
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
    api = ctypes.CDLL(runtime_library)
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
                                properties.warp_size, pci, runtime_library)
