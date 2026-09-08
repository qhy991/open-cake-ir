"""Concrete Metal binary-archive builder at the existing Lab BuildRequest seam.

The released OpenCakeEnvironment owns assessment/lowering authority. This builder
accepts only its bounded Metal lowering requirements, compiles a real archive, and
checks an archive-only reload in a fresh process before sealing LaunchableCandidate.
It performs no Evaluation, dispatch, optimization iteration, or qualification.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import tempfile

from open_cake_ir.compiler.target import Target
from open_cake_ir.evaluation.core import LaunchableCandidate
from open_cake_ir.evaluation.artifacts import METAL_TARGETS
from open_cake_ir.evaluation.metal_manifest import MetalTensorLaunchManifest, compile_options
from .environments import BuildRequest
from .faults import CandidateCompileRejected, RunProtocolFault

ROOT = Path(__file__).resolve().parents[3]
SWIFT_SOURCE = Path(__file__).with_name("metal") / "archive.swift"


def _json(document: object) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _external_root(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("Metal build output root must be absolute")
    resolved = path.resolve()
    if resolved == ROOT or ROOT in resolved.parents:
        raise ValueError("Metal build artifacts must stay outside the checkout")
    return resolved


def _validate_report(report: object, target: str, expected_names: tuple[str, ...], *, rebuilt: bool) -> dict:
    if not isinstance(report, dict) or report.get("status") != "completed":
        raise RunProtocolFault("harness_fault", "Metal archive helper did not complete")
    host, pipeline = report.get("host"), report.get("pipeline")
    if (set(report) != {"schema_version", "status", "dispatches", "host", "pipeline", "archive_miss_policy", "source_library_rebuilt", "stage"}
            or type(report.get("schema_version")) is not int or report["schema_version"] != 1
            or type(report.get("dispatches")) is not int or report["dispatches"] != 0
            or report.get("stage") != "strict_pipeline_load"
            or report.get("source_library_rebuilt") is not rebuilt
            or report.get("archive_miss_policy") != "failOnBinaryArchiveMiss"
            or not isinstance(host, dict) or set(host) != {"device_name", "device_registry_id", "operating_system", "target"}
            or host.get("target") != target or host.get("device_name") not in expected_names
            or any(not isinstance(value, str) or not value for value in host.values())
            or not isinstance(pipeline, dict)
            or set(pipeline) != {"thread_execution_width", "max_total_threads_per_threadgroup", "static_threadgroup_memory_bytes"}
            or type(pipeline.get("thread_execution_width")) is not int or pipeline["thread_execution_width"] != 32
            or type(pipeline.get("max_total_threads_per_threadgroup")) is not int
            or pipeline["max_total_threads_per_threadgroup"] < 32
            or type(pipeline.get("static_threadgroup_memory_bytes")) is not int or pipeline["static_threadgroup_memory_bytes"] != 0):
        raise RunProtocolFault("harness_fault", "Metal archive helper observations differ from the admitted pipeline")
    return report


@dataclass(frozen=True)
class MetalArchiveHost:
    """Actual Swift helper invocation; outputs are retained under an external root."""
    executable: Path
    toolchain: dict
    timeout_seconds: int = 120

    @classmethod
    def build(cls, output_root: Path, *, swiftc: str = "swiftc", timeout_seconds: int = 120):
        root = _external_root(output_root)
        root.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="metal-host-", dir=root))
        version = subprocess.run([swiftc, "--version"], capture_output=True, timeout=timeout_seconds)
        (directory / "swift-version.stdout").write_bytes(version.stdout)
        (directory / "swift-version.stderr").write_bytes(version.stderr)
        if version.returncode:
            raise RunProtocolFault("harness_fault", "Swift toolchain version observation failed")
        executable = directory / "metal-archive-host"
        command = [swiftc, "-O", str(SWIFT_SOURCE), "-o", str(executable)]
        (directory / "build-command.json").write_bytes(_json(command))
        completed = subprocess.run(command, capture_output=True, timeout=timeout_seconds)
        (directory / "build.stdout").write_bytes(completed.stdout)
        (directory / "build.stderr").write_bytes(completed.stderr)
        if completed.returncode or not executable.is_file():
            raise RunProtocolFault("harness_fault", f"Metal archive host build failed; see {directory}")
        return cls(executable, {"swift_version": version.stdout.decode().strip(),
                               "swift_build_command": command}, timeout_seconds)

    def invoke(self, request: dict, directory: Path) -> dict:
        directory.mkdir(parents=False, exist_ok=False)
        path = directory / "request.json"
        path.write_bytes(_json(request))
        completed = subprocess.run([str(self.executable), str(path)], capture_output=True, timeout=self.timeout_seconds)
        (directory / "stdout.json").write_bytes(completed.stdout)
        (directory / "stderr.log").write_bytes(completed.stderr)
        if completed.returncode:
            raise RunProtocolFault("harness_fault", f"Metal archive helper process failed; see {directory}")
        try:
            report = json.loads(completed.stdout)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise RunProtocolFault("harness_fault", "Metal archive helper returned malformed output") from error
        if not isinstance(report, dict):
            raise RunProtocolFault("harness_fault", "Metal archive helper returned non-object output")
        return report

    def reload(self, archive_path: Path, manifest: MetalTensorLaunchManifest, *,
               expected_device_names: tuple[str, ...], expected_host: dict, directory: Path) -> dict:
        """Compile-only archived-library reload with strict cache hit and no source."""
        report = self.invoke({"action": "reload", "target": manifest.target,
            "expected_device_names": list(expected_device_names), "entry_point": manifest.kernel_name,
            "archive_path": str(Path(archive_path).resolve()), "source_path": None,
            "expected_host": expected_host}, directory)
        if report.get("status") != "completed":
            raise RunProtocolFault("harness_fault", f"sealed Metal archive refused strict reload: {report.get('error')}")
        report = _validate_report(report, manifest.target, expected_device_names, rebuilt=False)
        if report["host"] != expected_host:
            raise RunProtocolFault("harness_fault", "Metal archive reload host differs")
        return report


class MetalToolchainBuilder:
    def __init__(self, *, workload, case_id: str, output_root: Path,
                 host: MetalArchiveHost | None = None, project_root: Path = ROOT):
        if workload.target not in METAL_TARGETS:
            raise ValueError("Metal builder requires a supported exact Metal target")
        self.workload, self.case_id = workload, case_id
        self.output_root = _external_root(output_root)
        self.host = host
        self.target = Target.load(Path(project_root) / "compiler" / "targets" / f"{workload.target}.json")
        self.workload.tensor_abi(case_id)

    def _manifest(self, request: BuildRequest) -> MetalTensorLaunchManifest:
        requirements = request.toolchain_requirements
        expected = {"source_language": "metal", "compiler": "MTLDevice.makeLibrary", "target": self.workload.target,
            "language_standard": "metal2.3", "fast_math_enabled": False, "threadgroup_memory_bytes": 0,
            "execution_model": "simd_program_tile", "active_threads_per_threadgroup": 32,
            "buffer_order": [arg.name for arg in self.workload.tensor_abi(self.case_id)],
            "threads_per_threadgroup": [32, 1, 1]}
        if (set(requirements) != set(expected) | {"threadgroups_per_grid"}
                or any(requirements.get(key) != value for key, value in expected.items())
                or type(requirements.get("fast_math_enabled")) is not bool
                or type(requirements.get("threadgroup_memory_bytes")) is not int
                or type(requirements.get("active_threads_per_threadgroup")) is not int
                or request.target != self.workload.target or request.source_role != "lowered_source"):
            raise ValueError("Metal builder requires the exact admitted Compiler lowering and Workload ABI")
        return MetalTensorLaunchManifest.for_workload(self.workload, self.case_id, target=request.target,
            kernel_name=request.entry_point, grid=requirements["threadgroups_per_grid"], block=requirements["threads_per_threadgroup"])

    def build(self, request: BuildRequest) -> LaunchableCandidate:
        manifest = self._manifest(request)
        self.output_root.mkdir(parents=True, exist_ok=True)
        if self.host is None:
            self.host = MetalArchiveHost.build(self.output_root)
        directory = Path(tempfile.mkdtemp(prefix="metal-build-", dir=self.output_root))
        source = directory / "lowered.metal"
        source.write_bytes(request.source)
        archive = directory / "pipeline.binary.metallib"
        report = self.host.invoke({"action": "build", "target": request.target,
            "expected_device_names": list(self.target.device_names), "entry_point": request.entry_point,
            "archive_path": str(archive), "source_path": str(source), "expected_host": None}, directory / "compile")
        if report.get("status") != "completed":
            if report.get("stage") in {"source_compile", "function_lookup", "archive_serialize", "strict_pipeline_load"}:
                raise CandidateCompileRejected(str(report.get("error")), artifact_payloads={"metal_build_report": _json(report)})
            raise RunProtocolFault("harness_fault", f"Metal build host refused: {report.get('error')}")
        _validate_report(report, request.target, self.target.device_names, rebuilt=True)
        if not archive.is_file() or not archive.stat().st_size:
            raise RunProtocolFault("harness_fault", "Metal builder produced no binary archive")
        replay = self.host.reload(archive, manifest, expected_device_names=self.target.device_names,
                                 expected_host=report["host"], directory=directory / "reload")
        build_report = {"schema_version": 1, "compile_options": compile_options(), "toolchain": self.host.toolchain,
                        "build": report, "archive_only_reload": replay}
        payloads = {"lowered_source": request.source, "metal_binary_archive": archive.read_bytes(),
                    "metal_build_report": _json(build_report), "launch_manifest": _json(manifest.as_dict())}
        # This is the existing common artifact seal boundary, not a separate digest inventory.
        return LaunchableCandidate(request.candidate_sha256, request.target, request.entry_point,
            {role: sha256(data).hexdigest() for role, data in payloads.items()}, manifest.canonical_sha256, payloads)
