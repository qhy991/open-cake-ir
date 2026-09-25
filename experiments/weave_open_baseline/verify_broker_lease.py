"""Verify this running B300 broker lease against its saved admission receipt.

The B300-M4 broker v0.6 provides CUDA_VISIBLE_DEVICES to children but does not
inject GPUQ_JOB_ID/GPUQ_MODE. Its receipt plus live status own those facts.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


GPUQ = "/home/qinhaiyan/agent-gpu-broker/bin/gpuq"


def verify(receipt_path: Path) -> tuple[str, str]:
    receipt = json.loads(receipt_path.read_text())
    status = json.loads(subprocess.check_output([GPUQ, "status", "--json"], text=True))
    if status.get("probe_error") is not None:
        raise RuntimeError("Broker device observation is unavailable")
    if (receipt.get("schema") != "gpuq.admission-receipt.v1"
            or receipt.get("mode") != "exclusive"
            or receipt.get("gpu_count") != 4
            or status.get("instance_id") != receipt.get("broker_instance_id")):
        raise RuntimeError("Broker receipt identity or exclusive EP4 allocation differs")
    running = [job for job in status.get("running", [])
               if job.get("job_id") == receipt.get("job_id")]
    if len(running) != 1:
        raise RuntimeError("Receipt's broker job is not running")
    job = running[0]
    ids = receipt.get("gpu_ids")
    if (job.get("mode") != "exclusive" or job.get("gpu_count") != 4
            or job.get("gpu_ids") != ids
            or job.get("owner") != receipt.get("owner")
            or job.get("admission_receipt_sha256") != receipt.get("receipt_sha256")
            or not isinstance(ids, list) or len(ids) != 4
            or any(type(gpu) is not int or gpu < 0 for gpu in ids)
            or len(set(ids)) != 4):
        raise RuntimeError("Live broker allocation differs from its admission receipt")
    selected = ",".join(map(str, ids))
    if os.environ.get("CUDA_VISIBLE_DEVICES") != selected:
        raise RuntimeError("Broker device visibility differs from its admission receipt")
    return receipt["job_id"], selected


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_broker_lease.py <admission.json>")
    job_id, device_ids = verify(Path(sys.argv[1]))
    print(job_id, device_ids)
