"""Public no-GPU resource reporting, context custody, and portable replay contracts."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import CompiledResources, Compiler  # noqa: E402
from open_cake_ir.compiler.performance.compiled_resources import (  # noqa: E402
    load_compiled_resources,
)
from open_cake_ir.compiler.toolchain import compile_triton, _parse_cuobjdump_resources  # noqa: E402

RESOURCE_TEXT = """Resource usage:
 Common:
  GLOBAL:0
 Function unrelated:
  REG:24 STACK:8 SHARED:0 LOCAL:0 CONSTANT[0]:944
 Function _cake_gemm_bias_b1_smoke_kernel:
  REG:128 STACK:376 SHARED:1024 LOCAL:0 CONSTANT[0]:944 TEXTURE:0
"""


class CompiledResourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        cls.path = ROOT / "corpus/schedules/gemm-bias-b1-smoke.json"
        cls.assessment = cls.compiler.assess_file(cls.path)
        cls.lowering = cls.compiler.lower(cls.assessment)
        # Clearly synthetic bytes: these fixtures test custody and parsing. Actual
        # CUBIN resource observations are exercised by the external no-GPU canary.
        cls.binary = b"\x7fELF-resource-contract-fixture"
        cls.resources = CompiledResources(
            source_sha256=cls.lowering.source_sha256,
            cubin_sha256=sha256(cls.binary).hexdigest(),
            target="sm_100a", entry_point="_cake_gemm_bias_b1_smoke_kernel",
            threads_per_cta=128, registers_per_thread=128,
            static_shared_bytes=1024, dynamic_shared_bytes=32784,
            stack_bytes=376, local_bytes=0,
            compiler_version="fixture", inspector_version="fixture",
        )

    def test_inspection_selects_one_function_and_keeps_storage_units(self) -> None:
        self.assertEqual(_parse_cuobjdump_resources(RESOURCE_TEXT, self.resources.entry_point), {
            "registers_per_thread": 128, "stack_bytes": 376,
            "static_shared_bytes": 1024, "local_bytes": 0,
        })
        for report in (RESOURCE_TEXT.replace("REG:128", ""), RESOURCE_TEXT + RESOURCE_TEXT):
            with self.subTest(report=report):
                with self.assertRaises(ValueError):
                    _parse_cuobjdump_resources(report, self.resources.entry_point)
        with self.assertRaises(ValueError):
            _parse_cuobjdump_resources(RESOURCE_TEXT, "absent")

    def test_public_profile_tightens_the_resource_bound_without_predicting_stalls(self) -> None:
        before = self.compiler.profile(self.assessment).as_dict()
        after = self.compiler.profile(self.assessment, compiled_resources=self.resources).as_dict()
        self.assertGreater(before["residency"]["ctas_per_sm_upper_bound"], 4)
        self.assertEqual(after["residency"]["ctas_per_sm_upper_bound"], 4)
        self.assertEqual(after["residency"]["binding_resource"], "registers")
        metrics = {item["metric"]: item for item in after["ncu_metrics"]}
        self.assertEqual(metrics["launch__registers_per_thread"]["value"], 128)
        self.assertEqual(metrics["launch__registers_per_thread"]["estimate_kind"], "exact")
        self.assertEqual(metrics["launch__occupancy_limit_shared_mem"]["value"], 6)
        self.assertEqual(metrics["sm__warps_active.avg.pct_of_peak_sustained_elapsed"]["value"], 25.0)
        self.assertEqual(metrics["sm__throughput.avg.pct_of_peak_sustained_elapsed"]["estimate_kind"], "unknown")
        self.assertEqual(metrics["smsp__warp_issue_stalled_barrier_per_warp_active.pct"]["estimate_kind"], "uncalibrated_risk")
        self.assertEqual(after["compiled_resources"]["stack_bytes"], 376)
        self.assertNotIn("spill_bytes", after["compiled_resources"])
        self.assertEqual(self.compiler.rank([self.assessment])[0], ())

    def test_stale_source_target_entry_and_launch_observations_are_refused(self) -> None:
        for changes in (
            {"source_sha256": "0" * 64}, {"target": "sm_90a"},
            {"entry_point": "unrelated"}, {"threads_per_cta": 256},
        ):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, "different source, target, entry or launch"):
                    self.compiler.profile(self.assessment, compiled_resources=replace(self.resources, **changes))
        changed = json.loads(self.assessment.schedule_bytes)
        changed["residency"]["registers_per_thread"] = 64
        assessment = self.compiler.assess(changed)
        with self.assertRaises(ValueError):
            self.compiler.profile(assessment, compiled_resources=self.resources)
        forged = replace(self.assessment, schedule_bytes=json.dumps(changed).encode())
        with self.assertRaises(ValueError):
            self.compiler.profile(forged, compiled_resources=self.resources)

    def test_invalid_allocation_values_are_not_accepted_as_exact_measurements(self) -> None:
        for value in (True, -1, 1.5, None):
            with self.subTest(value=value):
                document = self.resources.as_dict()
                document["registers_per_thread"] = value
                with self.assertRaises(ValueError):
                    CompiledResources.from_dict(document)

    def _write_report(self, directory: Path) -> Path:
        artifacts = directory / "0000"
        artifacts.mkdir()
        (artifacts / "kernel.cubin").write_bytes(self.binary)
        (artifacts / "lowered.py").write_text(self.lowering.source)
        path = directory / "report.json"
        path.write_text(json.dumps({"schema_version": 1, "rows": [{
            "profile": self.compiler.profile(self.assessment, compiled_resources=self.resources).as_dict()
        }]}))
        return path

    def test_portable_report_reuses_the_same_allocation_without_gpu_imports(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = self._write_report(Path(folder))
            self.assertEqual(load_compiled_resources(path)[self.lowering.source_sha256], self.resources)
            guard = """import importlib.abc,runpy,sys
class NoGPU(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','triton','cuda','cupti'}:
   raise AssertionError('GPU dependency imported during portable report replay: '+fullname)
sys.meta_path.insert(0,NoGPU())
sys.argv=['tools/report_schedule_profile.py',*sys.argv[1:]]
runpy.run_path('tools/report_schedule_profile.py',run_name='__main__')
"""
            result = subprocess.run([
                sys.executable, "-B", "-c", guard, "--revision", "compiler/revision.json",
                "--compiled-report", str(path), "--json", str(self.path),
            ], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            profile = json.loads(result.stdout)["rows"][0]["profile"]
            self.assertEqual(profile["compiled_resources"], self.resources.as_dict())
            feedback = subprocess.run([
                sys.executable, "-B", "src/open_cake_ir/tasks/qsa/project_feedback.py", "compiler",
                "--revision", "compiler/revision.json", "--compiled-report", str(path), str(self.path),
            ], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(feedback.returncode, 0, feedback.stderr)
            self.assertEqual(json.loads(feedback.stdout)["static_profile"]["compiled_resources"], self.resources.as_dict())

    def test_portable_report_refuses_changed_binary_or_source(self) -> None:
        for name in ("kernel.cubin", "lowered.py"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                path = self._write_report(Path(folder))
                with (Path(folder) / "0000" / name).open("ab") as stream:
                    stream.write(b"changed")
                with self.assertRaisesRegex(ValueError, "differs from its observation"):
                    load_compiled_resources(path)

    def test_bulk_profile_keeps_an_uncovered_backend_explicitly_static(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            report = self._write_report(Path(folder))
            result = subprocess.run([
                sys.executable, "-B", "tools/report_schedule_profile.py", "--revision",
                "compiler/revision.json", "--compiled-report", str(report), "--json",
                "corpus/schedules/flash-kmeans-assignment-full.json",
            ], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        profile = json.loads(result.stdout)["rows"][0]["profile"]
        self.assertIsNone(profile["compiled_resources"])
        self.assertTrue(any("does not cover this lowering backend" in item for item in profile["abstentions"]))

    def test_portable_report_does_not_follow_artifact_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = self._write_report(root)
            original = root / "0000/kernel.cubin"
            moved = root / "fixture.cubin"
            original.rename(moved)
            original.symlink_to(moved)
            with self.assertRaisesRegex(ValueError, "escapes its output directory"):
                load_compiled_resources(path)

    def test_shared_compiler_uses_an_explicit_target_and_never_initializes_gpu_handles(self) -> None:
        class Compiled:
            metadata = SimpleNamespace(name="kernel", num_ctas=1, num_warps=4, shared=32,
                                       global_scratch_size=0, profile_scratch_size=0)
            asm = {"source": "expanded", "ttir": "ir", "ttgir": "gpu ir", "llir": "llvm",
                   "ptx": ".target sm_100a\n", "cubin": b"\x7fELFfixture"}

            def __getattr__(self, name):
                raise AssertionError("unexpected runtime access: " + name)

        modules = {name: ModuleType(name) for name in (
            "triton", "triton.backends", "triton.backends.compiler", "triton.compiler"
        )}
        seen = {}
        def fake_compile(source, *, target, options):
            seen.update(target=target, options=options)
            return Compiled()
        modules["triton.backends.compiler"].GPUTarget = lambda *args: args
        modules["triton.compiler"].ASTSource = lambda *args: args
        modules["triton.compiler"].compile = fake_compile
        requirements = {"compiler": "triton", "source_language": "python", "target": "sm_100a",
                        "kernel_entry_point": "kernel", "signature": {}, "compile_constants": {},
                        "compile_options": {"num_warps": 4}}
        with patch.dict(sys.modules, modules), patch("importlib.metadata.version", return_value="fixture"):
            compilation = compile_triton(b"def kernel(): pass\n", requirements)
        self.assertEqual(seen["target"], ("cuda", 100, 32))
        self.assertEqual(compilation.threads_per_cta, 128)
        self.assertEqual(compilation.dynamic_shared_bytes, 32)
        self.assertEqual(compilation.artifacts["cubin"], Compiled.asm["cubin"])
        with self.assertRaises(ValueError):
            compile_triton(b"source", {**requirements, "target": "sm_90a"})


if __name__ == "__main__":
    unittest.main()
