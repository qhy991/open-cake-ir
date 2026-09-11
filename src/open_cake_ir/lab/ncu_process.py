"""NCU's restricted-counter privilege boundary; ordinary execution stays ordinary."""

from __future__ import annotations

import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from typing import Mapping, Sequence

from open_cake_ir.lab.process import (
    SupervisedProcessOutputLimit, SupervisedProcessTimeout, run_supervised,
)


def requires_sudo() -> bool:
    if sys.platform != "linux" or os.geteuid() == 0:
        return False
    try:
        parameters = Path("/proc/driver/nvidia/params").read_text()
    except FileNotFoundError:
        return False
    return re.search(r"^RmProfilingAdminOnly:\s*1\s*$", parameters, re.MULTILINE) is not None


def run_ncu(arguments: Sequence[str], *, cwd: Path, environment: Mapping[str, str],
            timeout_seconds: int = 900, maximum_output_bytes: int = 16 * 1024 * 1024):
    """Use sudo only for an observed restricted driver, with a root-owned deadline.

    The ordinary caller can signal its direct sudo relay, which forwards TERM to
    the privileged timeout process. Never use an ordinary killpg to clean up
    privileged descendants. The inner deadline also survives caller termination.
    """
    if not requires_sudo():
        return run_supervised(arguments, cwd=cwd, environment=environment,
                              timeout_seconds=timeout_seconds,
                              maximum_output_bytes=maximum_output_bytes)
    if timeout_seconds <= 0 or maximum_output_bytes <= 0 or not arguments:
        raise ValueError("NCU command or bounds differ")
    for name in ("CUDA_VISIBLE_DEVICES", "GPUQ_JOB_ID"):
        if not environment.get(name):
            raise ValueError(f"privileged NCU requires broker {name}")
    names = ("PATH", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "GPUQ_JOB_ID",
             "LD_LIBRARY_PATH", "TMPDIR", "CUDA_MODULE_LOADING")
    command = ["/usr/bin/sudo", "-n", "--", "/usr/bin/timeout", "--signal=TERM",
               "--kill-after=5s", f"{timeout_seconds}s", "/usr/bin/env",
               *(f"{name}={environment[name]}" for name in names if name in environment),
               *arguments]
    pending = None
    def interrupted(signum, frame):
        nonlocal pending
        if pending is None:
            pending = signum
    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    process = None
    try:
        for s in previous:
            signal.signal(s, interrupted)
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            process = subprocess.Popen(command, cwd=cwd, env=dict(environment),
                stdin=subprocess.DEVNULL, stdout=stdout_file, stderr=stderr_file,
                start_new_session=True)
            deadline = time.monotonic() + timeout_seconds + 15
            forwarded = False
            while process.poll() is None:
                if pending is not None and not forwarded:
                    process.send_signal(signal.SIGTERM)
                    forwarded = True
                    deadline = min(deadline, time.monotonic() + 15)
                if time.monotonic() >= deadline:
                    process.send_signal(signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired as error:
                        raise RuntimeError("privileged NCU cleanup unknown after relay deadline") from error
                    break
                time.sleep(0.05)
            process.wait()
            sizes = (stdout_file.tell(), stderr_file.tell())
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout = stdout_file.read(maximum_output_bytes)
            stderr = stderr_file.read(maximum_output_bytes)
            if pending is not None:
                raise SystemExit(128 + pending)
            if process.returncode in (124, 137):
                raise SupervisedProcessTimeout(stdout, stderr)
            if max(sizes) > maximum_output_bytes:
                raise SupervisedProcessOutputLimit(stdout, stderr)
            return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


def write_new(path: Path, payload: bytes, owner: tuple[int, int] | None = None) -> None:
    """Create profile artifacts as the admitted caller, without repairing custody."""
    uid, gid = os.geteuid(), os.getegid()
    change = owner is not None and owner != (uid, gid)
    if change and uid != 0:
        raise ValueError("profile output owner differs from ordinary caller")
    try:
        if change:
            os.setegid(owner[1])
            os.seteuid(owner[0])
        with path.open("xb") as stream:
            stream.write(payload)
    finally:
        if change:
            os.seteuid(uid)
            os.setegid(gid)
