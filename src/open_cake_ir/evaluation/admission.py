"""Observed exact-device admission shared by matched and Portfolio workers."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .cuda_driver import CudaDeviceAdmission
from open_cake_ir.compiler.target import cuda_target


def observe_exclusive_b200() -> CudaDeviceAdmission:
    """Retain the historical B200-only public admission boundary."""
    return observe_exclusive_cuda("sm_100a")


def observe_exclusive_cuda(target_id: str) -> CudaDeviceAdmission:
    """Reject the r41 clean-card race before compile, module load or launch."""

    target = cuda_target(target_id)

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    job_id = os.environ.get("GPUQ_JOB_ID")
    if not visible or "," in visible or not job_id or not job_id.startswith("gpuq-"):
        raise ValueError("gpu_admission_differs")
    executable = Path("/usr/bin/nvidia-smi")
    if not executable.is_file():
        raise ValueError("gpu_admission_differs")
    completed = subprocess.run(
        [
            str(executable),
            "-i",
            visible,
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )
    active = [line for line in completed.stdout.decode(errors="replace").splitlines() if line.strip()]
    if completed.returncode != 0 or active:
        raise ValueError("gpu_admission_differs")
    torch = __import__("torch")
    if (
        torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(0) not in target.device_names
        or torch.cuda.get_device_capability(0) != target.compute_capability
    ):
        raise ValueError("gpu_admission_differs")
    properties = torch.cuda.get_device_properties(0)
    return CudaDeviceAdmission(
        torch.cuda.get_device_name(0),
        torch.cuda.get_device_capability(0),
        str(getattr(properties, "uuid", "")),
        job_id,
        "exclusive",
    )
