"""Supervised kernel-only Triton compilation with an explicit Linux filesystem jail.

The coordinator never imports candidate modules. No unsandboxed fallback exists.
The read-only mounts are runtime dependencies, not the author's workspace or HOME.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import sys
import subprocess
from typing import Mapping

from open_cake_ir.compiler.toolchain import (
    TritonCompilation, compile_triton, validate_triton_kernel,
)
from .faults import RunProtocolFault
from .isolated_build import IsolatedCompiler
from .process import run_supervised


class IsolatedTritonCompiler(IsolatedCompiler):
    """One explicitly pinned Python/Triton runtime under bubblewrap, CPU compilation only."""

    label = "Triton"
    worker_module = "open_cake_ir.lab.triton_build"

    def __init__(self, *, python: str, bubblewrap: str, runtime_roots: list[str],
                 triton_version: str, timeout_seconds: int = 600,
                 build_environment: Mapping[str, str] | None = None):
        super().__init__(python=python, bubblewrap=bubblewrap, runtime_roots=runtime_roots,
                         timeout_seconds=timeout_seconds)
        if not triton_version:
            raise ValueError("isolated Triton runtime mount contract differs")
        # The jail runs --clearenv, so a fact that lives only in the invoking shell's
        # environment does not survive it. That is correct and costs a CUDA host nothing.
        # On the Hygon DTK host it costs two things, both measured: without
        # LD_LIBRARY_PATH `import triton` fails with "libgalaxyhip.so.5: cannot open
        # shared object file" while the file is mounted and present, and without
        # ROCM_PATH clang-18 reports "cannot find ROCm device library". The host declares
        # what it needs; nothing is inherited.
        self.build_environment = dict(build_environment or {})
        declared_paths = [part for value in self.build_environment.values()
                          for part in value.split(":") if part.startswith("/")]
        if any(not Path(part).is_dir() for part in declared_paths):
            raise ValueError("isolated Triton build environment names a missing directory")
        if any(not any(Path(part).is_relative_to(destination)
                       for destination in self.runtime_roots)
               for part in declared_paths):
            raise ValueError("isolated Triton build environment names a path outside every mount")
        self.triton_version = triton_version

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
                "bubblewrap_sha256": self._bubblewrap_sha256(),
                "runtime_roots": self._runtime_mount_identity(),
                "build_environment": dict(sorted(self.build_environment.items())),
                "triton_version": self.triton_version, "timeout_seconds": self.timeout_seconds}

    def _request(self, source: bytes, requirements: Mapping[str, object]) -> dict[str, object]:
        return {"source": source.decode(), "requirements": dict(requirements),
                "triton_version": self.triton_version}

    def _jail_environment(self, requirements: Mapping[str, object]) -> list[tuple[str, str]]:
        return [("TRITON_CACHE_DIR", "/tmp/triton-cache"), *sorted(self.build_environment.items())]

    def _supervise(self, argv, *, cwd, environment, timeout_seconds):
        return run_supervised(argv, cwd=cwd, environment=environment, timeout_seconds=timeout_seconds)

    def _receipt(self, root: Path, source: bytes, requirements: Mapping[str, object],
                 streams: dict[str, bytes]) -> TritonCompilation:
        record = json.loads((root / 'compilation.json').read_text())
        if record['compiler_version'] != self.triton_version:
            raise RunProtocolFault('harness_fault', 'isolated Triton runtime version differs')
        return TritonCompilation(source, record['target'], record['entry_point'],
            {k: base64.b64decode(v, validate=True) for k, v in record['artifacts'].items()},
            record['threads_per_cta'], record['dynamic_shared_bytes'], record['compiler_version'],
            record['code_object'])

    def compile(self, source: bytes, requirements: Mapping[str, object]) -> TritonCompilation:
        validate_triton_kernel(source, requirements)
        return self._compile_in_jail(source, requirements)


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
        'compiler_version': result.compiler_version, 'code_object': result.code_object,
    }))
    return 0


if __name__ == '__main__':
    raise SystemExit(_worker(sys.argv[1]))
