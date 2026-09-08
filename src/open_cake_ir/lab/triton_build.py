"""Supervised kernel-only Triton compilation with an explicit Linux filesystem jail.

The coordinator never imports candidate modules. No unsandboxed fallback exists.
The read-only mounts are runtime dependencies, not the author's workspace or HOME.
"""
from __future__ import annotations

from open_cake_ir.serialization import canonical_json_bytes

import base64
import json
from hashlib import sha256
import os
from pathlib import Path
import sys
import subprocess
import tempfile
from typing import Mapping

from open_cake_ir.compiler.toolchain import (
    TritonCompilation, compile_triton, validate_triton_kernel,
)
from .faults import CandidateCompileRejected, RunProtocolFault
from .process import run_supervised, SupervisedProcessTimeout, SupervisedProcessOutputLimit


class IsolatedTritonCompiler:
    """One explicitly pinned Python/Triton runtime under bubblewrap, CPU compilation only."""

    def __init__(self, *, python: str, bubblewrap: str, runtime_roots: list[str],
                 triton_version: str, timeout_seconds: int = 600):
        if sys.platform != "linux":
            raise ValueError("native Triton build requires Linux bubblewrap filesystem isolation")
        self.python = Path(os.path.abspath(python))
        self.bubblewrap = Path(bubblewrap).resolve(strict=True)
        # ELF interpreters and shared libraries name guest paths such as /lib64.
        # Resolving those aliases here would mount only /usr/lib64 in the jail.
        destinations = tuple(Path(os.path.abspath(p)) for p in runtime_roots)
        self._runtime_mounts = tuple((p.resolve(strict=True), p) for p in destinations)
        host_home = Path.home().resolve(strict=True)
        if (not self.python.is_file() or not self.bubblewrap.is_file() or not triton_version
            or not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool)
            or timeout_seconds <= 0 or not self.runtime_roots
            or any(not source.is_dir() or host_home.is_relative_to(source)
                   for source, _ in self._runtime_mounts)
            or not any(self.python.is_relative_to(p) for p in self.runtime_roots)):
            raise ValueError("isolated Triton runtime mount contract differs")
        self.triton_version = triton_version
        self.timeout_seconds = timeout_seconds

    @property
    def runtime_roots(self) -> tuple[Path, ...]:
        """Declared guest destinations; checked host sources are fixed separately."""
        return tuple(destination for _, destination in self._runtime_mounts)

    def check_executor(self, executor, *, author_workspace: str | Path) -> None:
        """Bind the isolated invocation to the already-admitted runtime owner."""
        host = executor.document["host_environment"]
        invocation = Path(host["python"]["invocation_path"])
        expected_python = Path(os.path.abspath(invocation))
        if self.python != expected_python or self.triton_version != host["packages"]["triton"]:
            raise ValueError("isolated Triton runtime differs from the frozen Executor")
        workspace = Path(author_workspace).resolve()
        checkout = Path(__file__).resolve().parents[3]
        if any(workspace.is_relative_to(source) or checkout.is_relative_to(source)
               for source, _ in self._runtime_mounts):
            raise ValueError("runtime mounts must not expose author workspace or Compiler checkout")

    @property
    def identity(self) -> dict[str, object]:
        # Exact runtime package/source identity is owned by the frozen Executor.
        return {"kind": "bubblewrap_triton_kernel_v1", "python": str(self.python),
                "bubblewrap": str(self.bubblewrap),
                "bubblewrap_sha256": sha256(self.bubblewrap.read_bytes()).hexdigest(),
                "runtime_roots": [{'source': str(source), 'destination': str(destination)}
                                  for source, destination in self._runtime_mounts],
                "triton_version": self.triton_version, "timeout_seconds": self.timeout_seconds}

    @property
    def canonical_sha256(self) -> str:
        return sha256(canonical_json_bytes(self.identity)).hexdigest()

    def compile(self, source: bytes, requirements: Mapping[str, object]) -> TritonCompilation:
        validate_triton_kernel(source, requirements)
        package_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="open-cake-isolated-triton-") as directory:
            root = Path(directory)
            (root / 'request.json').write_text(json.dumps({
                "source": source.decode(), "requirements": dict(requirements),
                "triton_version": self.triton_version,
            }))
            argv = [str(self.bubblewrap), '--die-with-parent', '--new-session',
                    '--unshare-all', '--clearenv', '--proc', '/proc', '--dev', '/dev',
                    '--tmpfs', '/tmp', '--dir', '/home', '--dir', '/home/build']
            for source_path, destination in self._runtime_mounts:
                argv += ['--ro-bind', str(source_path), str(destination)]
            argv += ['--ro-bind', str(package_root), '/compiler-src',
                     '--bind', str(root), '/build', '--chdir', '/build',
                     '--setenv', 'HOME', '/home/build', '--setenv', 'PATH', '/usr/bin:/bin',
                     '--setenv', 'PYTHONPATH', '/compiler-src',
                     '--setenv', 'PYTHONDONTWRITEBYTECODE', '1',
                     '--setenv', 'TRITON_CACHE_DIR', '/tmp/triton-cache',
                     str(self.python), '-s', '-m', 'open_cake_ir.lab.triton_build', '/build/request.json']
            try:
                result = run_supervised(argv, cwd=root, environment={},
                                        timeout_seconds=self.timeout_seconds)
            except (SupervisedProcessTimeout, SupervisedProcessOutputLimit) as error:
                raise RunProtocolFault('harness_fault', str(error), artifact_payloads={
                    'toolchain_stdout': error.stdout, 'toolchain_stderr': error.stderr,
                }) from error
            except OSError as error:
                diagnostic = f'{type(error).__name__}: {error}'
                raise RunProtocolFault('harness_fault', diagnostic, artifact_payloads={
                    'toolchain_stderr': diagnostic.encode(),
                }) from error
            if result.returncode:
                diagnostic = (result.stderr or result.stdout).decode(errors='replace')[-4096:]
                artifacts = {'toolchain_stdout': result.stdout, 'toolchain_stderr': result.stderr}
                if result.returncode != 2:
                    raise RunProtocolFault('harness_fault', 'isolated Triton build unavailable: ' + diagnostic,
                                           artifact_payloads=artifacts)
                raise CandidateCompileRejected(diagnostic, artifact_payloads=artifacts)
            record = json.loads((root / 'compilation.json').read_text())
            if record['compiler_version'] != self.triton_version:
                raise RunProtocolFault('harness_fault', 'isolated Triton runtime version differs')
            return TritonCompilation(source, record['target'], record['entry_point'],
                {k: base64.b64decode(v, validate=True) for k, v in record['artifacts'].items()},
                record['threads_per_cta'], record['dynamic_shared_bytes'], record['compiler_version'])


def _compile_failure(error: Exception) -> tuple[bool, str]:
    """Classify one failure and retain every linked exception's type and message.

    Infrastructure anywhere in the cause/context chain overrides a candidate-looking
    wrapper. Inspect both links, including suppressed display context, once per object.
    """
    external = (ImportError, OSError, subprocess.SubprocessError)
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    messages: list[str] = []
    infrastructure_failure = False
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        messages.append(f'{type(current).__name__}: {current}')
        infrastructure_failure |= isinstance(current, external)
        # Stack order keeps the explicit cause before the implicit context.
        pending.extend(link for link in (current.__context__, current.__cause__) if link is not None)
    diagnostic = '\n'.join(messages)
    if infrastructure_failure:
        return False, diagnostic
    if isinstance(error, (ValueError, SyntaxError)):
        return True, diagnostic
    # These optional runtime types are consulted only after a compilation error.
    # Import failure never selects a fallback compiler or changes the diagnostic.
    try:
        from triton.compiler.errors import CompilationError
        from triton.runtime.errors import OutOfResources
    except (ImportError, OSError):
        return False, diagnostic
    return isinstance(error, (CompilationError, OutOfResources)), diagnostic


def _worker(path: str) -> int:
    """Trusted supervisor entry; input is validated again before any candidate import."""
    import importlib.metadata
    request = json.loads(Path(path).read_text())
    if importlib.metadata.version('triton') != request['triton_version']:
        raise ValueError('pinned Triton version differs')
    source = request['source'].encode()
    validate_triton_kernel(source, request['requirements'])
    try:
        result = compile_triton(source, request['requirements'])
    except Exception as error:
        candidate_rejection, diagnostic = _compile_failure(error)
        print(diagnostic, file=sys.stderr)
        return 2 if candidate_rejection else 1
    Path('/build/compilation.json').write_text(json.dumps({
        'target': result.target, 'entry_point': result.entry_point,
        'artifacts': {k: base64.b64encode(v).decode() for k, v in result.artifacts.items()},
        'threads_per_cta': result.threads_per_cta, 'dynamic_shared_bytes': result.dynamic_shared_bytes,
        'compiler_version': result.compiler_version,
    }))
    return 0


if __name__ == '__main__':
    raise SystemExit(_worker(sys.argv[1]))
