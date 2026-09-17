"""One process-group supervisor for provider, toolchain and broker commands."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

_SENSITIVE_ENVIRONMENT_NAME = re.compile(
    r"(?:^|_)(?:API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?|AUTH)(?:$|_)",
    re.IGNORECASE,
)


def sanitized_environment(removed: Sequence[str] = ()) -> dict[str, str]:
    """Project the host environment while excluding explicit and common secret names."""

    explicit = set(removed)
    return {
        name: value
        for name, value in os.environ.items()
        if name not in explicit and _SENSITIVE_ENVIRONMENT_NAME.search(name) is None
    }


class SupervisedProcessTimeout(TimeoutError):
    """A timed-out process group plus its retained streams."""

    def __init__(self, stdout: bytes, stderr: bytes) -> None:
        super().__init__("supervised process timed out")
        self.stdout = stdout
        self.stderr = stderr


class SupervisedProcessOutputLimit(RuntimeError):
    """A process exceeded the bounded retained stdout/stderr contract."""

    def __init__(self, stdout: bytes, stderr: bytes) -> None:
        super().__init__("supervised process output limit exceeded")
        self.stdout = stdout
        self.stderr = stderr


def run_supervised(
    arguments: Sequence[str],
    *,
    cwd: str | Path,
    timeout_seconds: int,
    environment: Mapping[str, str] | None = None,
    maximum_output_bytes: int = 64 * 1024 * 1024,
) -> subprocess.CompletedProcess[bytes]:
    """Run without shell; on timeout kill and reap the whole new process group."""

    if not arguments or timeout_seconds <= 0 or maximum_output_bytes <= 0:
        raise ValueError("supervised command or timeout differs")
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(
            list(arguments),
            cwd=cwd,
            env=dict(environment) if environment is not None else None,
            stdin=subprocess.DEVNULL,
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=True,
        )
        timed_out = False
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        stdout_size = stdout_file.tell()
        stderr_size = stderr_file.tell()
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read(maximum_output_bytes)
        stderr = stderr_file.read(maximum_output_bytes)
        if timed_out:
            raise SupervisedProcessTimeout(stdout, stderr)
        if stdout_size > maximum_output_bytes or stderr_size > maximum_output_bytes:
            raise SupervisedProcessOutputLimit(stdout, stderr)
        return subprocess.CompletedProcess(list(arguments), process.returncode, stdout, stderr)
