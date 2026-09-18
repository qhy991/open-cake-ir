"""Observed exact-device admission shared by matched and Portfolio workers.

Two allocators can hand a CUDA device to a worker. The cluster allocator issues a `gpuq`
job and an exclusive lease, which is what the paired CUPTI assay's timing evidence
requires (ADR 0011, 0056). The local broker issues a `cuda` job for one device on one
machine, the way it already does for an Apple GPU and a DCU; that admission supports a
correctness check and never a paired timing receipt, and `paired.admit_device_identity`
refuses the latter by the job it names. The device observation is the same in both:
`nvidia-smi` sees no other compute process on the visible device and torch reports the
one device the Target declares.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .cuda_driver import CudaDeviceAdmission
from .platforms import platform_for
from open_cake_ir.compiler.target import CodeObject, Target, declared_target


def _cubin_target(target_id: str) -> Target:
    target = declared_target(target_id)
    # This observes a CUDA device through nvidia-smi and torch.cuda, which a Target
    # producing another object has nothing to say to.
    if target.code_object is not CodeObject.CUBIN:
        raise ValueError(
            f"CUDA device admission observes a cubin target; {target_id!r} declares "
            f"{target.code_object.value}"
        )
    return target


def _observe_cuda_device(target: Target, visible: str | None, job_id: str, mode: str) -> CudaDeviceAdmission:
    """One visible, otherwise idle device that is the one the Target declares."""
    if not visible or "," in visible:
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
        mode,
    )


def observe_exclusive_cuda(target_id: str) -> CudaDeviceAdmission:
    """Reject the r41 clean-card race before compile, module load or launch."""

    target = _cubin_target(target_id)
    prefix = platform_for(target).exclusive_job_prefix
    job_id = os.environ.get("GPUQ_JOB_ID")
    if not job_id or not job_id.startswith(f"{prefix}-"):
        raise ValueError("gpu_admission_differs")
    return _observe_cuda_device(target, os.environ.get("CUDA_VISIBLE_DEVICES"), job_id, "exclusive")


def observe_local_cuda(target_id: str) -> CudaDeviceAdmission:
    """Admit this process's local-broker job and the one visible CUDA device (D6).

    The job comes from the local broker under the cubin row's own local prefix, never
    from the cluster allocator; `CUDA_VISIBLE_DEVICES` must still name exactly one
    device, because the broker serializes a machine and does not choose a card.
    """

    target = _cubin_target(target_id)
    if os.environ.get("GPUQ_JOB_ID"):
        raise ValueError("gpu_admission_differs")
    from .local_broker import observe_local_job

    try:
        job_id = observe_local_job(platform_for(target).local_job_prefix)
    except (ValueError, OSError) as error:
        raise ValueError("gpu_admission_differs") from error
    return _observe_cuda_device(target, os.environ.get("CUDA_VISIBLE_DEVICES"), job_id,
                                "local_serialized")
