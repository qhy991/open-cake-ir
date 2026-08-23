"""Observed exclusive-B200 admission shared by matched and Portfolio workers."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .cuda_driver import CudaDeviceAdmission


def observe_exclusive_b200() -> CudaDeviceAdmission:
    """Reject the r41 clean-card race before compile, module load or launch."""

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
        or torch.cuda.get_device_name(0) != "NVIDIA B200"
        or torch.cuda.get_device_capability(0) != (10, 0)
    ):
        raise ValueError("gpu_admission_differs")
    properties = torch.cuda.get_device_properties(0)
    return CudaDeviceAdmission(
        "NVIDIA B200",
        (10, 0),
        str(getattr(properties, "uuid", "")),
        job_id,
        "exclusive",
    )
