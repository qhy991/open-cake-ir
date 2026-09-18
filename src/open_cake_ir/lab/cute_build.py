"""One filesystem-isolated CuTe toolchain feeding the common CUBIN launch boundary."""
from __future__ import annotations

from open_cake_ir.serialization import canonical_json_bytes

import base64
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping

from open_cake_ir.compiler.cute_toolchain import (
    CuTeCompilation, compile_cute, validate_cute_kernel, validate_cute_compilation,
)
from .faults import RunProtocolFault
from .isolated_build import IsolatedCompiler
from .process import run_supervised


class IsolatedCuTeCompiler(IsolatedCompiler):
    """Pinned SDK 4.5.2 under bubblewrap; no coordinator candidate imports or fallback."""

    label = "CuTe"
    worker_module = "open_cake_ir.lab.cute_build"

    def __init__(self, *, python: str, bubblewrap: str, runtime_roots: list[str],
                 cuobjdump: str, cutlass_version: str, timeout_seconds: int = 600):
        self.cuobjdump = Path(os.path.abspath(cuobjdump))
        super().__init__(python=python, bubblewrap=bubblewrap, runtime_roots=runtime_roots,
                         timeout_seconds=timeout_seconds, executables=(self.cuobjdump,))
        if cutlass_version != "4.5.2":
            raise ValueError("isolated CuTe runtime mount contract differs")
        self.cutlass_version = cutlass_version

    def _mount_refused(self, source: Path, destination: Path) -> bool:
        # The CuTe jail refuses the checkout and /dev as runtime roots at construction;
        # the Triton jail checks the checkout against the Executor's workspace instead.
        checkout = Path(__file__).resolve().parents[3]
        return (checkout.is_relative_to(source)
                or source.is_relative_to('/dev') or destination.is_relative_to('/dev'))

    def check_executor(self, executor, *, author_workspace: str | Path) -> None:
        host = executor.document["host_environment"]
        if (self.python != Path(os.path.abspath(host["python"]["invocation_path"]))
            or any(host["packages"].get(name) != self.cutlass_version for name in (
                "nvidia-cutlass-dsl", "nvidia-cutlass-dsl-libs-base", "nvidia-cutlass-dsl-libs-cu13"))):
            raise ValueError("isolated CuTe runtime differs from the frozen Executor")
        workspace = Path(author_workspace).resolve()
        if any(workspace.is_relative_to(source) for source, _ in self._runtime_mounts):
            raise ValueError("runtime mounts must not expose the author workspace")

    @property
    def identity(self) -> dict[str, object]:
        return {"kind": "bubblewrap_cute_kernel_v1", "python": str(self.python),
                "bubblewrap": str(self.bubblewrap),
                "bubblewrap_sha256": self._bubblewrap_sha256(),
                "runtime_roots": self._runtime_mount_identity(),
                "cuobjdump": str(self.cuobjdump), "cutlass_version": self.cutlass_version,
                "timeout_seconds": self.timeout_seconds}

    def _request(self, source: bytes, requirements: Mapping[str, object]) -> dict[str, object]:
        return {"source": source.decode(), "requirements": dict(requirements),
                "cutlass_version": self.cutlass_version, "cuobjdump": str(self.cuobjdump)}

    def _jail_environment(self, requirements: Mapping[str, object]) -> list[tuple[str, str]]:
        return [("CUDA_VISIBLE_DEVICES", ""), ("CUTE_DSL_ARCH", requirements['target']),
                ("CUTE_DSL_DISABLE_FILE_CACHING", "1"), ("CUTE_DSL_CACHE_DIR", "/tmp/cute-cache")]

    def _supervise(self, argv, *, cwd, environment, timeout_seconds):
        return run_supervised(argv, cwd=cwd, environment=environment, timeout_seconds=timeout_seconds)

    def _streams(self, root: Path, result) -> dict[str, bytes]:
        streams = super()._streams(root, result)
        guard_path = root / "artifacts" / "cuda_call_guard.json"
        if guard_path.is_file():
            streams["toolchain_cuda_call_guard"] = guard_path.read_bytes()
        return streams

    def _receipt(self, root: Path, source: bytes, requirements: Mapping[str, object],
                 streams: dict[str, bytes]) -> CuTeCompilation:
        try:
            record = json.loads((root / "compilation.json").read_text())
            if (not isinstance(record, dict) or set(record) != {"target", "entry_point", "artifacts", "threads_per_cta",
                              "dynamic_shared_bytes", "compiler_version"}
                or not isinstance(record["artifacts"], dict)):
                raise ValueError("CuTe compilation receipt fields differ")
            artifacts = {key: base64.b64decode(value, validate=True)
                         for key, value in record["artifacts"].items()}
            compilation = CuTeCompilation(source, record["target"], record["entry_point"], artifacts,
                record["threads_per_cta"], record["dynamic_shared_bytes"], record["compiler_version"])
            validate_cute_compilation(compilation, source, requirements)
            return compilation
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise RunProtocolFault("harness_fault", "invalid CuTe compilation receipt: " + str(error),
                                   artifact_payloads=streams) from error

    def compile(self, source: bytes, requirements: Mapping[str, object]) -> CuTeCompilation:
        validate_cute_kernel(source, requirements)
        return self._compile_in_jail(source, requirements)


class CuTeToolchainBuilder:
    """Seal both authoring representations against one ordered Workload tensor ABI."""

    def __init__(self, *, workload, case_id: str, isolated_compiler=None):
        if workload is None or not case_id:
            raise ValueError("CuTe builder Workload and case must be bound together")
        self._workload, self._case_id, self._isolated = workload, case_id, isolated_compiler

    def build(self, request):
        from open_cake_ir.evaluation.core import LaunchableCandidate, TensorLaunchManifest
        requirements = request.toolchain_requirements
        validate_cute_kernel(request.source, requirements)
        abi = self._workload.tensor_abi(self._case_id)
        if (requirements["signature"] != [{"name": row.name, "dtype": row.dtype} for row in abi]
            or request.target != requirements["target"]
            or request.target != self._workload.document["semantics"].get("target")
            or request.entry_point != requirements["kernel_entry_point"]):
            raise ValueError("CuTe build requirements differ from the Workload tensor ABI")
        if self._isolated is None:
            raise RunProtocolFault("harness_fault", "CuTe requires filesystem-isolated compilation")
        compiled = self._isolated.compile(request.source, requirements)
        validate_cute_compilation(compiled, request.source, requirements)
        manifest = TensorLaunchManifest.for_workload(self._workload, self._case_id,
            target=compiled.target, kernel_name=compiled.entry_point, grid=requirements["grid"],
            block=requirements["block"], dynamic_shared_memory_bytes=0, hidden_null_pointer_parameters=0)
        manifest_bytes = canonical_json_bytes(manifest.as_dict())
        payloads = {request.source_role: request.source,
                    "compiler_expanded_source": compiled.artifacts["source"],
                    "ptx": compiled.artifacts["ptx"], "cubin": compiled.artifacts["cubin"],
                    "toolchain_resource_report": compiled.artifacts["toolchain_resource_report"],
                    "launch_manifest": manifest_bytes}
        return LaunchableCandidate(request.candidate_sha256, request.target, compiled.entry_point,
            {role: sha256(payload).hexdigest() for role, payload in payloads.items()},
            sha256(manifest_bytes).hexdigest(), payloads)


def _compile_failure(error: Exception) -> tuple[bool, str]:
    """A compiler diagnostic rejects a candidate; broken tools remain harness faults."""
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    messages = []
    infrastructure = False
    candidate = isinstance(error, (ValueError, SyntaxError))
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        messages.append(f"{type(current).__name__}: {current}")
        infrastructure |= isinstance(current, (ImportError, OSError, subprocess.SubprocessError))
        candidate |= (type(current).__module__.startswith("cutlass.") and type(current).__name__ in {
            "DSLUserCodeError", "DSLOperationBuildError", "DSLAstPreprocessorError", "MLIRError"})
        candidate |= (type(current).__module__, type(current).__name__) == (
            "cutlass.cute.nvgpu.common", "OpError")
        pending.extend(link for link in (current.__context__, current.__cause__, getattr(current, "cause", None))
                       if isinstance(link, BaseException))
    return candidate and not infrastructure, '\n'.join(messages)


def _worker(path: str) -> int:
    request = json.loads(Path(path).read_text())
    if any(importlib.metadata.version(name) != request["cutlass_version"] for name in (
        "nvidia-cutlass-dsl", "nvidia-cutlass-dsl-libs-base", "nvidia-cutlass-dsl-libs-cu13")):
        raise RuntimeError("pinned CuTe SDK package versions differ")
    source = request["source"].encode()
    validate_cute_kernel(source, request["requirements"])
    try:
        result = compile_cute(source, request["requirements"], output_directory=Path('/build/artifacts'),
                              cuobjdump=request["cuobjdump"])
    except Exception as error:
        rejected, message = _compile_failure(error)
        print(message, file=sys.stderr)
        return 2 if rejected else 1
    Path('/build/compilation.json').write_text(json.dumps({
        "target": result.target, "entry_point": result.entry_point,
        "artifacts": {key: base64.b64encode(value).decode() for key, value in result.artifacts.items()},
        "threads_per_cta": result.threads_per_cta, "dynamic_shared_bytes": result.dynamic_shared_bytes,
        "compiler_version": result.compiler_version}))
    return 0


if __name__ == '__main__':
    raise SystemExit(_worker(sys.argv[1]))
