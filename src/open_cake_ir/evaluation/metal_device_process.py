"""Run a Metal helper only during an existing broker's process-owned lease.

Fresh admission happens in a short-lived child that execs the native helper.
The reusable Python author/build process never holds a new allocation. Existing
Evaluation allocations are borrowed and remain owned by their original worker.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import os
from pathlib import Path
import subprocess
import sys

from open_cake_ir.evaluation.local_broker import admit_local_job, observe_local_metal_job
from open_cake_ir.evaluation.source_bootstrap import module_command


def _record_job(path: Path | None, job: str):
    if path is not None:
        with path.open('x') as output:
            json.dump({'job_id': job}, output)


def read_metal_job(path: Path) -> str:
    """Read the process admission retained before the trusted helper was exec'd."""
    if path.is_symlink() or not path.is_file():
        raise ValueError('Metal process allocation record is unavailable')
    record = json.loads(path.read_bytes())
    from .attempts import valid_job_mode
    if (not isinstance(record, dict) or set(record) != {'job_id'}
            or not isinstance(record['job_id'], str)
            or not (valid_job_mode(record['job_id'], 'exclusive')
                    or re.fullmatch(r'metal-[0-9a-f]{12}', record['job_id'])
                    and valid_job_mode(record['job_id'], 'local_serialized'))):
        raise ValueError('Metal process allocation record differs')
    return record['job_id']


def run_metal_process(executable: Path, request: Path, *, target: str,
                      timeout_seconds: int, allocation_output: Path | None = None,
                      queue_seconds: float = 0) -> subprocess.CompletedProcess:
    executable, request = Path(executable), Path(request)
    if (not executable.is_absolute() or not executable.is_file()
            or not request.is_absolute() or not request.is_file()):
        raise ValueError('Metal device process requires prepared absolute executable and request paths')
    if (not math.isfinite(queue_seconds) or queue_seconds < 0
            or allocation_output is not None and (
                not allocation_output.is_absolute() or allocation_output.exists()
                or allocation_output.is_symlink())):
        raise ValueError('Metal process queue or new allocation output differs')
    command = [str(executable), str(request)]
    if os.environ.get('METAL_JOB_ID') and not os.environ.get('METAL_BROKER_LOCK_FD'):
        raise ValueError('Metal allocation identity lacks its live descriptor')
    if os.environ.get('GPUQ_JOB_ID') and os.environ.get('METAL_BROKER_LOCK_FD'):
        raise ValueError('Metal device process has conflicting allocation owners')
    if os.environ.get('GPUQ_JOB_ID'):
        # This admission's owner outlives the helper; do not acquire another lease.
        from open_cake_ir.evaluation.gpuq import observe_allocation
        _record_job(allocation_output, observe_allocation(target)['job_id'])
        return subprocess.run(command, capture_output=True, timeout=timeout_seconds)
    if os.environ.get('METAL_BROKER_LOCK_FD'):
        _record_job(allocation_output, observe_local_metal_job())
        return subprocess.run(command, capture_output=True, timeout=timeout_seconds,
                              pass_fds=(int(os.environ['METAL_BROKER_LOCK_FD']),))
    # The fixed Swift helpers do not fork. The child execs that helper, so
    # subprocess.run kills and reaps the device-owning PID on timeout. Keep the
    # caller's process group: its outer supervisor must also cancel this helper.
    arguments = ['--executable', str(executable), '--request', str(request),
                 '--queue-seconds', str(queue_seconds)]
    if allocation_output is not None:
        arguments.extend(('--allocation-output', str(allocation_output)))
    return subprocess.run(module_command(sys.executable, __name__, *arguments),
        cwd=request.parent, capture_output=True, timeout=timeout_seconds + queue_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--allocation-output', type=Path)
    parser.add_argument('--queue-seconds', type=float, default=0)
    args = parser.parse_args()
    if (not math.isfinite(args.queue_seconds) or args.queue_seconds < 0
            or args.allocation_output is not None and not args.allocation_output.is_absolute()):
        parser.error('valid queue budget and absolute allocation output are required')
    if (not args.executable.is_absolute() or not args.executable.is_file()
            or not os.access(args.executable, os.X_OK)
            or not args.request.is_absolute() or not args.request.is_file()):
        parser.error('prepared Metal helper executable and request are required')
    job = admit_local_job('metal', queue_seconds=args.queue_seconds)
    _record_job(args.allocation_output, job)
    # Same PID, same descriptor, same process group. Exit, failure and supervisor
    # timeout all end device ownership before the parent parses the report.
    os.execv(str(args.executable), [str(args.executable), str(args.request)])
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
