"""Run a Metal helper only during an existing broker's process-owned lease.

Fresh admission happens in a short-lived child that execs the native helper.
The reusable Python author/build process never holds a new allocation. Existing
Evaluation allocations are borrowed and remain owned by their original worker.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from open_cake_ir.evaluation.local_broker import admit_local_job, observe_local_metal_job
from open_cake_ir.evaluation.source_bootstrap import module_command


def run_metal_process(executable: Path, request: Path, *, target: str,
                      timeout_seconds: int) -> subprocess.CompletedProcess:
    executable, request = Path(executable), Path(request)
    if (not executable.is_absolute() or not executable.is_file()
            or not request.is_absolute() or not request.is_file()):
        raise ValueError('Metal device process requires prepared absolute executable and request paths')
    command = [str(executable), str(request)]
    if os.environ.get('METAL_JOB_ID') and not os.environ.get('METAL_BROKER_LOCK_FD'):
        raise ValueError('Metal allocation identity lacks its live descriptor')
    if os.environ.get('GPUQ_JOB_ID') and os.environ.get('METAL_BROKER_LOCK_FD'):
        raise ValueError('Metal device process has conflicting allocation owners')
    if os.environ.get('GPUQ_JOB_ID'):
        # This admission's owner outlives the helper; do not acquire another lease.
        from open_cake_ir.evaluation.gpuq import observe_allocation
        observe_allocation(target)
        return subprocess.run(command, capture_output=True, timeout=timeout_seconds)
    if os.environ.get('METAL_BROKER_LOCK_FD'):
        observe_local_metal_job()
        return subprocess.run(command, capture_output=True, timeout=timeout_seconds,
                              pass_fds=(int(os.environ['METAL_BROKER_LOCK_FD']),))
    # The fixed Swift helpers do not fork. The child execs that helper, so
    # subprocess.run kills and reaps the device-owning PID on timeout. Keep the
    # caller's process group: its outer supervisor must also cancel this helper.
    return subprocess.run(module_command(sys.executable, __name__,
        '--executable', str(executable), '--request', str(request)),
        cwd=request.parent, capture_output=True, timeout=timeout_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    if (not args.executable.is_absolute() or not args.executable.is_file()
            or not os.access(args.executable, os.X_OK)
            or not args.request.is_absolute() or not args.request.is_file()):
        parser.error('prepared Metal helper executable and request are required')
    admit_local_job('metal')
    # Same PID, same descriptor, same process group. Exit, failure and supervisor
    # timeout all end device ownership before the parent parses the report.
    os.execv(str(args.executable), [str(args.executable), str(args.request)])
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
