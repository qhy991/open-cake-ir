"""Admit local single-device jobs, then exec the bound common worker.

This is process admission only. It does not evaluate a candidate, own a Ralph loop,
or promise exclusive physical GPU access against WindowServer or unrelated apps.

The allocation, not the API: an Apple GPU, a Hygon DCU and a CUDA device outside the
cluster allocator are each one visible device on one machine, and all are reached this
way. Each kind keeps its own lock and its own job prefix, so a DCU job is not recorded
as a Metal one -- the same mislabelling as a DCU latency recorded as CUPTI, arriving
through the allocator instead of the timer. The kinds are the local prefixes the
execution platform rows declare; a CUDA job issued here is admitted for correctness and
never for the paired CUPTI assay, which requires the exclusive cluster lease.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import time
import uuid
from dataclasses import asdict

from .attempts import valid_job_mode
from .platforms import PLATFORMS


LOCAL_KINDS = tuple(
    row.local_job_prefix for row in PLATFORMS.values() if row.local_job_prefix is not None)


def _lock_path(kind: str = "metal") -> Path:
    if kind not in LOCAL_KINDS:
        raise ValueError(f"local broker kind {kind!r} is unsupported")
    return Path(tempfile.gettempdir()) / f"open-cake-ir-{kind}-{os.geteuid()}.lock"


def _device_lock_path(kind: str, device: int) -> Path:
    _selection_environment(kind, device)
    if type(device) is not int or device < 0:
        raise ValueError('device-scoped local admission requires an explicit physical device')
    base = _lock_path(kind)
    return base.with_name(f'{base.stem}-device-{device}{base.suffix}')


def _acquire(path: Path, *, shared: bool = False) -> int:
    fd = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid() or metadata.st_nlink != 1:
            raise ValueError("local broker lock ownership differs")
        fcntl.flock(fd, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        return fd
    except BaseException:
        os.close(fd)
        raise


def observe_local_job(kind: str = "metal") -> str:
    """Observe this process's own local-broker admission for one allocation kind."""
    job = os.environ.get("METAL_JOB_ID", "")
    raw_fd = os.environ.get("METAL_BROKER_LOCK_FD", "")
    if (not valid_job_mode(job, "local_serialized") or not job.startswith(f"{kind}-")
            or not raw_fd.isdecimal() or int(raw_fd) < 3):
        raise ValueError(f"local {kind} broker admission is missing")
    fd = int(raw_fd)
    metadata = os.fstat(fd)
    scope = os.environ.get('OPEN_CAKE_LOCAL_LOCK_SCOPE', 'user')
    selected = os.environ.get('OPEN_CAKE_LOCAL_DEVICE')
    if scope not in {'user', 'device'}:
        raise ValueError('local broker lock scope differs')
    if selected is not None:
        keys = _selection_environment(kind, int(selected)) if selected.isdecimal() else ()
        if not keys or any(os.environ.get(key) != selected for key in keys):
            raise ValueError('local broker device mapping differs from admission')
    if scope == 'device' and (selected is None or not selected.isdecimal()):
        raise ValueError('device-scoped local admission lacks a physical device')
    path = _device_lock_path(kind, int(selected)) if scope == 'device' else _lock_path(kind)
    expected = path.stat(follow_symlinks=False)
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1 or (metadata.st_dev, metadata.st_ino) != (expected.st_dev, expected.st_ino)):
        raise ValueError('local broker descriptor is not the admitted shared lock')
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    barrier = os.environ.get('OPEN_CAKE_LOCAL_LEGACY_FD')
    if scope == 'device':
        if barrier is None or not barrier.isdecimal() or int(barrier) < 3 or int(barrier) == fd:
            raise ValueError('device-scoped local admission lacks the legacy compatibility lock')
        observed = os.fstat(int(barrier))
        legacy = _lock_path(kind).stat(follow_symlinks=False)
        if (not stat.S_ISREG(observed.st_mode) or observed.st_uid != os.geteuid()
                or observed.st_nlink != 1 or (observed.st_dev, observed.st_ino) != (legacy.st_dev, legacy.st_ino)):
            raise ValueError('legacy compatibility lock differs')
        fcntl.flock(int(barrier), fcntl.LOCK_SH | fcntl.LOCK_NB)
    elif barrier is not None:
        raise ValueError('user-scoped local admission cannot carry a device compatibility lock')
    return job


def observe_local_metal_job() -> str:
    """The Metal spelling every retained Metal run replays through."""
    return observe_local_job("metal")


class LocalBrokerBusy(BlockingIOError):
    def __init__(self, kind: str, job_id: str):
        super().__init__(f"{kind}_broker_busy")
        self.job_id = job_id


_VISIBILITY_KEYS = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "MACA_VISIBLE_DEVICES")


def _selection_environment(kind: str, device: int | None) -> tuple[str, ...]:
    row = next((row for row in PLATFORMS.values() if row.local_job_prefix == kind), None)
    if row is None:
        raise ValueError(f"local broker kind {kind!r} is unsupported")
    if device is not None:
        if type(device) is not int or device < 0:
            raise ValueError("local physical device must be a nonnegative integer")
        if not row.local_visibility_environment:
            raise ValueError(f"local broker kind {kind!r} has no device-selection API")
    return row.local_visibility_environment


def admit_local_job(kind: str, *, device: int | None = None, queue_seconds: float = 0,
                    lock_scope: str = 'user') -> str:
    """Acquire this process's local job after its CPU preparation completes.

    The default user scope remains compatible with frozen workers. Device scope
    requires an explicit physical ordinal and shares the legacy user lock while
    holding its own device lock exclusively. Thus device-scoped jobs on different
    cards can overlap; legacy workers still exclude them without knowing a device.

    Ownership lasts until process exit, exactly as in the exec entry point. The
    descriptor deliberately remains open through failures and final reporting:
    a partially constructed device object cannot cause an early in-process unlock.
    Call only from a short-lived worker's process entry, never a reusable host.
    """
    selection = _selection_environment(kind, device)
    if (type(queue_seconds) not in (int, float) or not math.isfinite(queue_seconds)
            or queue_seconds < 0):
        raise ValueError("local queue seconds must be finite and nonnegative")
    if lock_scope not in {'user', 'device'}:
        raise ValueError('unknown local lock scope')
    path = _device_lock_path(kind, device) if lock_scope == 'device' else _lock_path(kind)
    if any(os.environ.get(key) for key in ('METAL_BROKER_LOCK_FD', 'GPUQ_JOB_ID',
                                         'OPEN_CAKE_LOCAL_LEGACY_FD', 'OPEN_CAKE_LOCAL_LOCK_SCOPE')):
        raise ValueError('a local job cannot nest an existing allocation')
    job = f"{kind}-" + uuid.uuid4().hex[:12]
    deadline = time.monotonic() + queue_seconds
    while True:
        barrier_fd = None
        try:
            # Old workers take an exclusive user lock and cannot declare their
            # device. New workers hold it shared, then lock their own device.
            # This permits new/new cross-device concurrency without bypassing
            # the old admission boundary. Waiting retains neither descriptor.
            if lock_scope == 'device':
                barrier_fd = _acquire(_lock_path(kind), shared=True)
            try:
                fd = _acquire(path)
            except BaseException:
                if barrier_fd is not None:
                    os.close(barrier_fd)
                raise
            break
        except BlockingIOError as error:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LocalBrokerBusy(kind, job) from error
            time.sleep(min(.1, remaining))
    try:
        if device is not None:
            for key in _VISIBILITY_KEYS:
                os.environ.pop(key, None)
            for key in selection:
                os.environ[key] = str(device)
            os.environ['OPEN_CAKE_LOCAL_DEVICE'] = str(device)
        os.set_inheritable(fd, True)
        if barrier_fd is not None:
            os.set_inheritable(barrier_fd, True)
            os.environ['OPEN_CAKE_LOCAL_LEGACY_FD'] = str(barrier_fd)
            os.environ['OPEN_CAKE_LOCAL_LOCK_SCOPE'] = 'device'
        os.environ.update(METAL_JOB_ID=job, METAL_BROKER_LOCK_FD=str(fd))
    except BaseException:
        os.close(fd)
        if barrier_fd is not None:
            os.close(barrier_fd)
        raise
    print(f"[{kind}-run] accepted job {job}", file=sys.stderr, flush=True)
    return job


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-module")
    parser.add_argument("--kind", choices=LOCAL_KINDS, default="metal",
                        help="which local device family this job serializes")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-device", type=int)
    parser.add_argument("--local-lock-scope", choices=("user", "device"), default="user")
    parser.add_argument("--local-queue-seconds", type=float, default=0)
    parser.add_argument("--probe-target", help="exact MACA device admission only; no kernel or timing")
    parser.add_argument("--runtime-library", type=Path)
    args = parser.parse_args(argv)
    if args.probe_target is not None:
        if (args.kind != 'maca' or args.request is not None or args.worker_module is not None
                or args.runtime_library is None or not args.runtime_library.is_absolute()
                or not args.output.is_absolute() or args.output.exists()):
            parser.error('MACA probe requires only its exact target, runtime library and new absolute output')
        from .triton_metax import observe_local_metax
        try:
            job = admit_local_job(args.kind, device=args.local_device, queue_seconds=args.local_queue_seconds, lock_scope=args.local_lock_scope)
            admission = observe_local_metax(args.probe_target, runtime_library=str(args.runtime_library))
            result = {'schema_version': 1, 'scope': 'local_device_admission_only',
                      'admitted': True, 'job_id': job, 'mode': 'local_serialized',
                      'physical_device': args.local_device, 'lock_scope': args.local_lock_scope, 'device_admission': asdict(admission),
                      'kernel_calls': 0, 'timing_samples': 0}
        except (ValueError, OSError, RuntimeError) as error:
            result = {'schema_version': 1, 'scope': 'local_device_admission_only',
                      'admitted': False, 'error': str(error), 'kernel_calls': 0, 'timing_samples': 0}
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2)
        return 0 if result['admitted'] else 1
    if args.worker_module is None or args.request is None or args.runtime_library is not None:
        parser.error('worker execution requires --worker-module and --request')
    if re.fullmatch(r"open_cake_ir\.[a-zA-Z0-9_.]+", args.worker_module) is None:
        parser.error("worker must be a bound open_cake_ir module")
    if not args.request.is_absolute() or not args.output.is_absolute() or args.output.exists():
        parser.error("request/output must be absolute and output must be new")
    try:
        admit_local_job(args.kind, device=args.local_device, queue_seconds=args.local_queue_seconds, lock_scope=args.local_lock_scope)
        # Exec preserves the supervisor-owned process group and lock. Its
        # timeout kills the worker and native children together as before.
        from .source_bootstrap import module_command
        os.execvpe(sys.executable, module_command(sys.executable, args.worker_module,
            "--request", str(args.request), "--output", str(args.output)), dict(os.environ))
    except LocalBrokerBusy as error:
        result = {"schema_version": 1, "job_id": error.job_id, "mode": "local_serialized", "admitted": False,
                  "error": f"{args.kind}_broker_busy", "failure_class": "admission", "receipt": None,
                  "counters": {name: 0 for name in ("compiler_invocations", "module_loads", "preflight_calls", "kernel_calls", "timing_samples", "fallback_calls")}}
        with args.output.open("x") as stream:
            json.dump(result, stream)
        return 0
    return 1  # exec never returns normally


if __name__ == "__main__":
    raise SystemExit(main())
