"""The bounded C550 route and its own refusals; no device evidence is manufactured."""

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.frontend import parse
from open_cake_ir.compiler.target import CodeObject, Target, TargetParseError, Vendor
from open_cake_ir.compiler.toolchain import triton_route
from open_cake_ir.evaluation.platforms import platform_for
from open_cake_ir.tasks.normalization.study import evaluation_policy
from open_cake_ir.tasks.workloads import create_task
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.lab.executor import triton_version

ROOT = Path(__file__).resolve().parents[2]


class MetaxPlatformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_c550_rmsnorm_reaches_maca_with_its_declared_family_and_lane_width(self):
        document, source = create_task("rmsnorm", backend="triton-metax", rows=8, columns=128)
        assessment = self.compiler.assess(parse(source).document)
        self.assertTrue(assessment.lowering_eligible, [f.code for f in assessment.findings])
        lowering = self.compiler.lower(assessment)
        facts = lowering.toolchain_requirements
        route = triton_route(facts)
        self.assertEqual((route.gpu_backend, route.architecture, route.warp_size), ("maca", 80, 64))
        self.assertEqual(facts["codegen_arch"], "xcore1000")
        self.assertEqual(document["semantics"]["target"], "xcore1002")
        self.assertEqual(route.binary_role, "mcfatbin")
        self.assertNotIn("ptx", route.artifact_roles)
        self.assertNotIn("llir", route.artifact_roles)

    def test_native_cuda_cannot_claim_the_maca_target(self):
        _, source = create_task("rmsnorm", backend="triton-metax", rows=8, columns=128)
        schedule = parse(source).document
        schedule["lowering"]["backend"] = "native_cuda"
        result = self.compiler.assess(schedule)
        self.assertFalse(result.lowering_eligible)
        self.assertIn("BACKEND_TARGET_UNSUPPORTED", [f.code for f in result.findings])

    def test_compatibility_architecture_is_not_a_cuda_hardware_capability(self):
        original = json.loads((ROOT / "compiler/targets/xcore1002.json").read_text())
        target = Target.from_dict(original)
        self.assertIs(target.vendor, Vendor.METAX)
        self.assertIs(target.code_object, CodeObject.MCFATBIN)
        self.assertIsNone(target.compute_capability)
        self.assertIsNone(target.warps_per_warpgroup)
        for value in (None, True, "80", 0):
            with self.subTest(value=value), self.assertRaises(TargetParseError):
                Target.from_dict({**original, "triton_arch": value})
        with self.assertRaisesRegex(TargetParseError, "no CUDA compute capability"):
            Target.from_dict({**original, "compute_capability": [8, 0]})

    def test_fp8_copy_preserves_its_storage_type_without_arithmetic(self):
        source = '''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="copy", target="xcore1002", backend="triton", entry_point="copy")
def candidate(lm, x: cake.Tensor((8, 128), "fp8_e4m3"), y: cake.Tensor((8, 128), "fp8_e4m3", mode="output")):
    compute = lm.role(execution_groups=[0])
    row = lm.program(x, axis=0, dimension=0, tile=1)
    with compute:
        value = lm.load(x[row, 0:128], id="load")
        lm.store(y[row, 0:128], value, id="store")
'''
        result = self.compiler.assess(parse(source).document)
        self.assertTrue(result.lowering_eligible, result.findings)
        lowering = self.compiler.lower(result)
        self.assertEqual(dict(lowering.toolchain_requirements["signature"]),
                         {"x": "*fp8e4nv", "y": "*fp8e4nv"})
        self.assertNotIn(".to(tl.", lowering.source)

    def test_explicit_numeric_conversions_keep_their_tensor_pointer_types(self):
        for source_dtype, destination, pointer in (
            ("fp16", "fp32", "*fp16"), ("bf16", "fp32", "*bf16"),
            ("fp32", "fp16", "*fp32"), ("fp32", "bf16", "*fp32"),
            ("int32", "fp32", "*i32"),
        ):
            source = f'''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="convert", target="xcore1002", backend="triton", entry_point="convert")
def candidate(lm, x: cake.Tensor((8, 128), "{source_dtype}"), y: cake.Tensor((8, 128), "{destination}", mode="output")):
    compute = lm.role(execution_groups=[0])
    row = lm.program(x, axis=0, dimension=0, tile=1)
    with compute:
        value = lm.load(x[row, :], id="load")
        converted = lm.cast(value, to="{destination}", id="cast")
        lm.store(y[row, :], converted, id="store")
'''
            with self.subTest(source=source_dtype, destination=destination):
                assessment = self.compiler.assess(parse(source).document)
                self.assertTrue(assessment.lowering_eligible, assessment.findings)
                lowering = self.compiler.lower(assessment)
                self.assertEqual(lowering.toolchain_requirements["signature"]["x"], pointer)
                self.assertIn(".to(tl.", lowering.source)
                # Admitting INT32 buffers does not admit an unspecified float-to-int cast.
                document = parse(source.replace(f'to="{destination}"', 'to="int32"')
                                 .replace(f'y: cake.Tensor((8, 128), "{destination}"',
                                          'y: cake.Tensor((8, 128), "int32"')).document
                if source_dtype != "int32":
                    refused = self.compiler.assess(document)
                    self.assertIn("CAST_DTYPE_UNSUPPORTED", [f.code for f in refused.findings])

    def test_existing_mixed_precision_and_integer_tasks_reach_the_platform(self):
        for task in ("add_rmsnorm_bf16", "aka_row_gather", "aka_momentum_sgd"):
            with self.subTest(task=task):
                _, source = create_task(task, backend="triton-metax", rows=8, columns=128)
                assessment = self.compiler.assess(parse(source).document)
                self.assertTrue(assessment.lowering_eligible, assessment.findings)
                self.assertEqual(self.compiler.lower(assessment).target, "xcore1002")

    def test_gelu_uses_the_declared_maca_math_contract_and_refuses_borrowed_names(self):
        for task in ("gelu_tanh", "gelu_tanh_backward"):
            _, source = create_task(task, backend="triton-metax", rows=8, columns=128)
            document = parse(source).document
            assessment = self.compiler.assess(document)
            self.assertTrue(assessment.lowering_eligible, assessment.findings)
            self.assertIn("libdevice.tanh(", self.compiler.lower(assessment).source)
            operation = next(op for op in document["operations"] if op["id"] == "tanh")
            self.assertEqual(operation["parameters"]["instruction"]["contract"], "maca.tanh.f32")
            for borrowed in ("libdevice.tanh.f32", "ocml.tanh.f32"):
                operation["parameters"]["instruction"]["contract"] = borrowed
                refused = self.compiler.assess(document)
                self.assertFalse(refused.lowering_eligible)
                self.assertIn("TARGET_INSTRUCTION_UNSUPPORTED", [f.code for f in refused.findings])

    def test_native_timer_policy_preserves_all_cases_and_owns_its_profile(self):
        document, _ = create_task("rmsnorm", backend="triton-metax", rows=8, columns=128)
        workload = WorkloadContract(document)
        policy = evaluation_policy(workload)
        self.assertEqual(tuple(policy["validation_case_ids"]), workload.case_ids)
        self.assertEqual(policy["search_evaluation"], "correctness_then_paired_mcpti_dispatch")
        self.assertEqual(policy['paired_timing']['route_calls_per_cohort'], 36)
        self.assertEqual(platform_for("xcore1002").measurement_source, 'mcpti_dispatch')
        self.assertEqual(platform_for("xcore1002").attribution, "inside_evaluate")

    def test_flagtree_distribution_version_is_not_used_as_triton_api_version(self):
        host = {"packages": {"flagtree": "0.5.1+metax3.1"}, "runtime": {"triton_version": "3.1.0"}}
        self.assertEqual(triton_version(host), "3.1.0")
        self.assertEqual(triton_version({"packages": {"triton": "3.7.1"}}), "3.7.1")

    def test_worker_refuses_host_drift_before_device_admission_or_evaluation(self):
        from open_cake_ir.tasks import evaluate as worker
        executor = SimpleNamespace(admit_host=Mock(side_effect=ValueError("captured host drift")))
        authority = SimpleNamespace(executor=executor, request={"purpose": "confirmatory"},
                                    candidate=SimpleNamespace(target="xcore1002"))
        with patch("open_cake_ir.evaluation.triton_metax.observe_local_metax") as observe, \
             patch.object(worker, "_evaluate_tile_candidate") as evaluate:
            with self.assertRaisesRegex(ValueError, "captured host drift"):
                worker._evaluate_metax_candidate(authority, {}, collect_timing=False)
            executor.admit_host.assert_called_once()
            observe.assert_not_called()
            evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
