"""Serialize local Metal jobs for one user, then exec the bound common worker.

This is process admission only. It does not evaluate a candidate, own a Ralph loop,
or promise exclusive physical GPU access against WindowServer or unrelated apps.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import uuid

from .attempts import valid_job_mode


def _lock_path() -> Path:
    return Path(tempfile.gettempdir()) / f"open-cake-ir-metal-{os.geteuid()}.lock"


def _acquire(path: Path) -> int:
    fd = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid() or metadata.st_nlink != 1:
            raise ValueError("local Metal broker lock ownership differs")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except BaseException:
        os.close(fd)
        raise


def observe_local_metal_job() -> str:
    job = os.environ.get("METAL_JOB_ID", "")
    raw_fd = os.environ.get("METAL_BROKER_LOCK_FD", "")
    if not valid_job_mode(job, "local_serialized") or not raw_fd.isdecimal() or int(raw_fd) < 3:
        raise ValueError("local Metal broker admission is missing")
    fd = int(raw_fd)
    metadata = os.fstat(fd)
    expected = _lock_path().stat(follow_symlinks=False)
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1 or (metadata.st_dev, metadata.st_ino) != (expected.st_dev, expected.st_ino)):
        raise ValueError("local Metal broker descriptor is not the shared user lock")
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return job


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-module", required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if re.fullmatch(r"open_cake_ir\.[a-zA-Z0-9_.]+", args.worker_module) is None:
        parser.error("worker must be a bound open_cake_ir module")
    if not args.request.is_absolute() or not args.output.is_absolute() or args.output.exists():
        parser.error("request/output must be absolute and output must be new")
    job = "metal-" + uuid.uuid4().hex[:12]
    print(f"[metal-run] accepted job {job}", file=sys.stderr, flush=True)
    try:
        fd = _acquire(_lock_path())
    except BlockingIOError:
        result = {"schema_version": 1, "job_id": job, "mode": "local_serialized", "admitted": False,
                  "error": "metal_broker_busy", "failure_class": "admission", "receipt": None,
                  "counters": {name: 0 for name in ("compiler_invocations", "module_loads", "preflight_calls", "kernel_calls", "timing_samples", "fallback_calls")}}
        with args.output.open("x") as stream:
            json.dump(result, stream)
        return 0
    # Exec preserves the supervisor-owned process group and this lock. A timeout
    # in the existing supervisor kills the worker and its native children together.
    os.set_inheritable(fd, True)
    environment = dict(os.environ, METAL_JOB_ID=job, METAL_BROKER_LOCK_FD=str(fd))
    os.execvpe(sys.executable, [sys.executable, "-m", args.worker_module,
        "--request", str(args.request), "--output", str(args.output)], environment)
    return 1  # exec never returns normally


if __name__ == "__main__":
    raise SystemExit(main())
