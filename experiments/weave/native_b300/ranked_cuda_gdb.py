"""Bounded CUDA-GDB snapshot of an unchanged ranked development bundle.

Copy beside `adapter_run.py` and run only inside its four-GPU broker job.
CUDA-GDB starts the exact bundle runner, interrupts it after eight seconds,
records host and CUDA focus, then exits. The diagnostic deliberately stops
this one test process; its output is not an oracle or a timing sample.
"""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys


HERE = Path(__file__).resolve().parent
GDB = Path('/usr/local/cuda-13.1/bin/cuda-gdb')


def main():
    if not GDB.is_file() or not GDB.stat().st_mode & 0o111:
        raise RuntimeError('qualified CUDA-GDB executable is unavailable')
    log = HERE / 'cuda_gdb.log'
    if log.exists():
        raise RuntimeError('CUDA-GDB log must be create-only')
    argv = [str(GDB), '--batch', '-q',
            '-ex', 'set pagination off',
            '-ex', 'set confirm off',
            '-ex', 'run',
            '-ex', 'info cuda devices',
            '-ex', 'info cuda kernels',
            '-ex', 'info cuda warps',
            '-ex', 'bt 12',
            '--args', sys.executable, str(HERE / 'adapter_run.py'), 'run']
    with log.open('w') as output:
        process = subprocess.Popen(argv, cwd=HERE, stdout=output,
                                   stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            if process.poll() is None:
                os.kill(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=25)
            except subprocess.TimeoutExpired:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
    print(f'CUDA-GDB diagnostic exit={process.returncode}; retained {log}')
    return 0 if process.returncode == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
