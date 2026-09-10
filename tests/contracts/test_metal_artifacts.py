"""Metal build/ABI seam tests. Native coverage creates pipelines but never dispatches.

The small WorkloadContract fixture tests tensor ABI projection, not task admission or
Study/Executor qualification. Native artifacts are genuine compiled binary archives.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import platform
import shutil
import tempfile
import unittest
from unittest.mock import Mock

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.compiler.target import Target
from open_cake_ir.evaluation.artifacts import METAL_TARGETS, executable_role, required_build_roles
from open_cake_ir.evaluation.core import LaunchableCandidate
from open_cake_ir.evaluation.metal_manifest import MetalTensorLaunchManifest
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.lab.environments import BuildRequest
from open_cake_ir.lab.faults import RunProtocolFault
from open_cake_ir.lab.metal_build import MetalArchiveHost, MetalToolchainBuilder
from open_cake_ir.lab.metal_host import inspect_metal_host
from tools.metal import rmsnorm
from tests.contracts._executor_fixture import compiler_reference

ROOT = Path(__file__).resolve().parents[2]


def abi_fixture(target: str = "apple_gpu_family7"):
    return WorkloadContract({"workload_id": "artifact-abi-fixture", "operator": "test_geometry",
        "semantics": {"target": target, "candidate_abi": {"inputs": ["x", "weight"], "outputs": ["out"]}},
        "cases": [{"case_id": "odd", "shape": {"R": 3, "C": 7}, "seed": 7001, "mode": "uniform"}],
        "tensors": {name: {"shape": shape, "dtype": "fp32", "layout": "contiguous_row_major"}
                    for name, shape in (("x", ["R", "C"]), ("weight", ["C"]), ("out", ["R", "C"]))}})


def request_fixture(target: str = "apple_gpu_family7"):
    source = rmsnorm.source(3, 7, target=target)
    compiler = Compiler.load(ROOT, ROOT / "compiler/revision.lock.json")
    if compiler.state != "released":
        raise AssertionError("native test requires the existing released Compiler")
    assessment = compiler.assess(frontend.parse(source).document)
    lowering = compiler.lower(assessment)
    return BuildRequest(sha256(source.encode()).hexdigest(), lowering.source.encode(), "lowered_source",
                        lowering.source_sha256, lowering.target, lowering.route.entry_point,
                        lowering.toolchain_requirements)


def live_metal_target(executable: Path, directory: Path) -> str | None:
    """Name the exact target of the device actually present, or None.

    Each admitted target is offered to the helper, which accepts only its own exact
    device. This identifies the host; it never substitutes another Apple GPU.
    """
    for target in sorted(METAL_TARGETS):
        names = Target.load(ROOT / "compiler/targets" / f"{target}.json").device_names
        try:
            inspect_metal_host(executable, target=target, expected_device_names=list(names),
                               directory=directory / f"inspect-{target}")
        except (ValueError, RunProtocolFault):
            continue
        return target
    return None


class MetalArtifactContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="cake-metal-artifacts-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.workload = abi_fixture()
        self.request = request_fixture()

    def manifest(self):
        return MetalTensorLaunchManifest.for_workload(self.workload, "odd", target="apple_gpu_family7",
            kernel_name=self.request.entry_point, grid=[3, 1, 1], block=[32, 1, 1])

    def test_cuda_minimum_constructor_and_backend_build_products_are_distinct(self):
        payload = b"fixture CUDA executable, not executed"
        digest = sha256(payload).hexdigest()
        legacy = LaunchableCandidate(digest, "sm_100a", "kernel", {"cubin": digest}, digest, {"cubin": payload})
        self.assertEqual(set(legacy.artifact_roles), {"cubin"})
        self.assertEqual(executable_role("sm_103a"), "cubin")
        self.assertEqual(executable_role("apple_gpu_family7"), "metal_binary_archive")
        self.assertEqual(executable_role("apple_gpu_family9"), "metal_binary_archive")
        self.assertEqual(required_build_roles("metal"), {"metal_binary_archive", "metal_build_report", "launch_manifest"})
        self.assertNotIn("lowered_source", required_build_roles("metal"))
        self.assertEqual(required_build_roles("triton"), {"compiler_expanded_source", "ptx", "cubin", "launch_manifest"})
        with self.assertRaises(ValueError):
            executable_role("apple_gpu_family10")

    def test_wrong_backend_executable_roles_and_missing_executables_refuse(self):
        payload = b"synthetic role-contract bytes"
        digest = sha256(payload).hexdigest()
        for target, roles in (("apple_gpu_family7", {"cubin": digest}),
                              ("sm_100a", {"metal_binary_archive": digest}),
                              ("apple_gpu_family7", {"lowered_source": digest}),
                              ("apple_gpu_family7", {"metal_binary_archive": digest, "cubin": digest})):
            with self.subTest(target=target, roles=roles), self.assertRaises(ValueError):
                LaunchableCandidate(digest, target, "kernel", roles, digest)

    def test_manifest_roundtrip_and_exact_workload_projection(self):
        manifest = self.manifest()
        self.assertEqual(MetalTensorLaunchManifest.from_dict(manifest.as_dict()), manifest)
        manifest.check_workload(self.workload, "odd")
        self.assertEqual(manifest.block_threads, 32)
        self.assertEqual([row[0] for row in manifest.tensor_abi], ["x", "weight", "out"])
        wrong = manifest.as_dict()
        wrong["tensor_abi"][0]["shape"] = [3, 8]
        with self.assertRaisesRegex(ValueError, "selected Workload"):
            MetalTensorLaunchManifest.from_dict(wrong).check_workload(self.workload, "odd")

    def test_manifest_refuses_unadmitted_launch_math_target_and_abi(self):
        changes = ({"target": "sm_100a"}, {"target": []}, {"grid": [True, 1, 1]}, {"block": [64, 1, 1]},
                   {"threadgroup_memory_bytes": True}, {"execution_model": "serial_program_tile"},
                   {"active_threads_per_threadgroup": 1}, {"archive_miss_policy": "compile_on_miss"},
                   {"kernel_name": "bad-name"}, {"schema_version": True})
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                MetalTensorLaunchManifest.from_dict({**self.manifest().as_dict(), **change})
        for mutation in (lambda d: d["tensor_abi"].reverse(),
                         lambda d: d["tensor_abi"][0].update(dtype="bf16"),
                         lambda d: d["compile_options"].update(fast_math_enabled=0)):
            document = self.manifest().as_dict()
            mutation(document)
            with self.assertRaises(ValueError):
                MetalTensorLaunchManifest.from_dict(document)

    def test_builder_refuses_bad_requirements_before_host_or_filesystem_work(self):
        host = Mock()
        root = self.directory / "new-build-root"
        builder = MetalToolchainBuilder(compiler_reference=compiler_reference(ROOT), workload=self.workload, case_id="odd", output_root=root, host=host)
        changes = ({"compiler": "metal"}, {"buffer_order": ["weight", "x", "out"]},
                   {"threads_per_threadgroup": [1, 1, 1]}, {"fast_math_enabled": 0},
                   {"language_standard": "metal3.0"}, {"unknown_policy": True})
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                builder.build(replace(self.request, toolchain_requirements={**self.request.toolchain_requirements, **change}))
        with self.assertRaises(ValueError):
            builder.build(replace(self.request, source_role="authored_source"))
        host.invoke.assert_not_called()
        self.assertFalse(root.exists())
        with self.assertRaises(ValueError):
            MetalToolchainBuilder(compiler_reference=compiler_reference(ROOT), workload=self.workload, case_id="odd", output_root=ROOT / "runs")

    @unittest.skipUnless(platform.system() == "Darwin" and platform.machine() == "arm64" and shutil.which("swiftc"),
                         "native compile-only Metal archive test requires Apple Silicon and Swift")
    def test_real_archive_build_and_source_free_strict_reload(self):
        host = MetalArchiveHost.build(self.directory)
        # The Apple target is a property of this host, not of the fixture's default.
        target = live_metal_target(host.executable, self.directory)
        if target is None:
            self.skipTest("no admitted exact Apple Metal device is present")
        workload, request = abi_fixture(target), request_fixture(target)
        builder = MetalToolchainBuilder(compiler_reference=compiler_reference(ROOT), workload=workload, case_id="odd", output_root=self.directory, host=host)
        candidate = builder.build(request)
        self.assertEqual(set(candidate.artifact_roles), required_build_roles("metal") | {"lowered_source"})
        manifest = MetalTensorLaunchManifest.from_dict(json.loads(candidate.artifact_payloads["launch_manifest"]))
        report = json.loads(candidate.artifact_payloads["metal_build_report"])
        self.assertEqual(candidate.launch_spec_sha256, manifest.canonical_sha256)
        self.assertEqual(report["archive_only_reload"]["source_library_rebuilt"], False)
        self.assertEqual(report["build"]["dispatches"], 0)
        source_free = self.directory / "source-free"
        source_free.mkdir()
        archive = source_free / "pipeline.binary.metallib"
        archive.write_bytes(candidate.artifact_payloads["metal_binary_archive"])
        self.assertGreater(archive.stat().st_size, 0)
        replay = host.reload(archive, manifest, expected_device_names=builder.target.device_names,
                            expected_host=report["build"]["host"], directory=source_free / "replay")
        self.assertFalse(replay["source_library_rebuilt"])
        self.assertEqual(replay["dispatches"], 0)
        # Missing symbol and invalid compiled bytes must refuse without source fallback.
        with self.assertRaises(RunProtocolFault):
            host.reload(archive, replace(manifest, kernel_name="missing_function"),
                        expected_device_names=builder.target.device_names, expected_host=report["build"]["host"],
                        directory=source_free / "wrong-entry")
        corrupt = source_free / "corrupt.binary.metallib"
        corrupt.write_bytes(b"not a Metal binary archive")
        with self.assertRaises(RunProtocolFault):
            host.reload(corrupt, manifest, expected_device_names=builder.target.device_names,
                        expected_host=report["build"]["host"], directory=source_free / "corrupt")
        with self.assertRaises(RunProtocolFault):
            host.reload(archive, manifest, expected_device_names=builder.target.device_names,
                        expected_host={**report["build"]["host"], "operating_system": "different OS"},
                        directory=source_free / "wrong-host")


if __name__ == "__main__":
    unittest.main()
