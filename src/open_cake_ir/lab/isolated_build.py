"""One bubblewrap jail for every kernel-only compile the Lab supervises.

The coordinator never imports candidate modules. No unsandboxed fallback exists. The
jail mounts the declared runtime roots read-only, the Compiler's own source tree at
`/compiler-src` (src only: no Target document and no checkout, D3), and one scratch
directory at `/build` carrying `request.json` in and `compilation.json` out; it runs
`--clearenv`, so a fact that lives only in the invoking shell does not survive it.

`triton_build.py` and `cute_build.py` each carried this scaffold in full. They differ
in what they pin (a Triton version and a declared build environment; a CuTe SDK pin
and a cuobjdump), in the variables their toolchain reads inside the jail, and in how
they read a receipt back -- so those are the hooks, and everything else is here once.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Mapping

from open_cake_ir.serialization import canonical_json_bytes

from .faults import CandidateCompileRejected, RunProtocolFault
from .process import SupervisedProcessOutputLimit, SupervisedProcessTimeout


class IsolatedCompiler:
    """A pinned interpreter under bubblewrap, CPU compilation only; subclasses pin a toolchain."""

    # Names the toolchain in refusals ("Triton", "CuTe").
    label: str = ""
    # The module bubblewrap runs with /build/request.json; its `_worker` writes the receipt.
    worker_module: str = ""

    def __init__(self, *, python: str, bubblewrap: str, runtime_roots: list[str],
                 timeout_seconds: int, executables: tuple[Path, ...] = ()) -> None:
        if sys.platform != "linux":
            raise ValueError(f"native {self.label} build requires Linux bubblewrap filesystem isolation")
        self.python = Path(os.path.abspath(python))
        self.bubblewrap = Path(bubblewrap).resolve(strict=True)
        # ELF interpreters and shared libraries name guest paths such as /lib64.
        # Resolving those aliases here would mount only /usr/lib64 in the jail.
        destinations = tuple(Path(os.path.abspath(path)) for path in runtime_roots)
        self._runtime_mounts = tuple((path.resolve(strict=True), path) for path in destinations)
        home = Path.home().resolve(strict=True)
        if (not self.python.is_file() or not self.bubblewrap.is_file()
            or any(not executable.is_file() for executable in executables)
            or type(timeout_seconds) is not int or timeout_seconds <= 0
            or not self._runtime_mounts
            or any(not source.is_dir() or home.is_relative_to(source)
                   or self._mount_refused(source, destination)
                   for source, destination in self._runtime_mounts)
            or any(not any(path.is_relative_to(root) for root in self.runtime_roots)
                   for path in (self.python, *executables))):
            raise ValueError(f"isolated {self.label} runtime mount contract differs")
        self.timeout_seconds = timeout_seconds

    def _mount_refused(self, source: Path, destination: Path) -> bool:
        """A subclass's own refusal of one runtime mount, beyond the shared contract."""
        return False

    @property
    def runtime_roots(self) -> tuple[Path, ...]:
        """Declared guest destinations; checked host sources are fixed separately."""
        return tuple(destination for _, destination in self._runtime_mounts)

    def _runtime_mount_identity(self) -> list[dict[str, str]]:
        return [{"source": str(source), "destination": str(destination)}
                for source, destination in self._runtime_mounts]

    def _bubblewrap_sha256(self) -> str:
        return sha256(self.bubblewrap.read_bytes()).hexdigest()

    @property
    def identity(self) -> dict[str, object]:
        raise NotImplementedError

    @property
    def canonical_sha256(self) -> str:
        return sha256(canonical_json_bytes(self.identity)).hexdigest()

    # -- hooks -------------------------------------------------------------------------

    def _request(self, source: bytes, requirements: Mapping[str, object]) -> dict[str, object]:
        """The request.json the worker validates again before any candidate import."""
        raise NotImplementedError

    def _jail_environment(self, requirements: Mapping[str, object]) -> list[tuple[str, str]]:
        """Variables the toolchain reads inside the jail, in the order they are set."""
        raise NotImplementedError

    def _supervise(self, argv: list[str], *, cwd: Path, environment: dict, timeout_seconds: int):
        """Run the jail; the subclass module owns the supervisor name its tests patch."""
        raise NotImplementedError

    def _streams(self, root: Path, result) -> dict[str, bytes]:
        """Retained toolchain streams; a subclass may add what its jail wrote."""
        return {"toolchain_stdout": result.stdout, "toolchain_stderr": result.stderr}

    def _receipt(self, root: Path, source: bytes, requirements: Mapping[str, object],
                 streams: dict[str, bytes]):
        """Read compilation.json back into the toolchain's typed compilation."""
        raise NotImplementedError

    # -- the scaffold ------------------------------------------------------------------

    def _argv(self, root: Path, requirements: Mapping[str, object]) -> list[str]:
        package_root = Path(__file__).resolve().parents[2]
        argv = [str(self.bubblewrap), "--die-with-parent", "--new-session", "--unshare-all",
                "--clearenv", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                "--dir", "/home", "--dir", "/home/build"]
        for source_path, destination in self._runtime_mounts:
            argv += ["--ro-bind", str(source_path), str(destination)]
        argv += ["--ro-bind", str(package_root), "/compiler-src",
                 "--bind", str(root), "/build", "--chdir", "/build",
                 "--setenv", "HOME", "/home/build", "--setenv", "PATH", "/usr/bin:/bin",
                 "--setenv", "PYTHONPATH", "/compiler-src",
                 "--setenv", "PYTHONDONTWRITEBYTECODE", "1"]
        for name, value in self._jail_environment(requirements):
            argv += ["--setenv", name, value]
        argv += [str(self.python), "-s", "-m", self.worker_module, "/build/request.json"]
        return argv

    def _compile_in_jail(self, source: bytes, requirements: Mapping[str, object]):
        with tempfile.TemporaryDirectory(prefix=f"open-cake-isolated-{self.label.lower()}-") as directory:
            root = Path(directory)
            (root / "request.json").write_text(json.dumps(self._request(source, requirements)))
            argv = self._argv(root, requirements)
            try:
                result = self._supervise(argv, cwd=root, environment={},
                                         timeout_seconds=self.timeout_seconds)
            except (SupervisedProcessTimeout, SupervisedProcessOutputLimit) as error:
                raise RunProtocolFault("harness_fault", str(error), artifact_payloads={
                    "toolchain_stdout": error.stdout, "toolchain_stderr": error.stderr,
                }) from error
            except OSError as error:
                diagnostic = f"{type(error).__name__}: {error}"
                raise RunProtocolFault("harness_fault", diagnostic, artifact_payloads={
                    "toolchain_stderr": diagnostic.encode(),
                }) from error
            streams = self._streams(root, result)
            if result.returncode:
                diagnostic = (result.stderr or result.stdout).decode(errors="replace")[-4096:]
                # Exit 2 is the worker's own classification of a candidate rejection;
                # anything else is the toolchain, not the candidate.
                if result.returncode != 2:
                    raise RunProtocolFault("harness_fault",
                                           f"isolated {self.label} build unavailable: " + diagnostic,
                                           artifact_payloads=streams)
                raise CandidateCompileRejected(diagnostic, artifact_payloads=streams)
            return self._receipt(root, source, requirements, streams)
