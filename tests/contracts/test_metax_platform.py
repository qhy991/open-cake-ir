"""The bounded C550 route and its own refusals; no device evidence is manufactured."""

import json
from pathlib import Path
import unittest

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

    def test_non_fp32_copy_is_refused_by_the_maca_dtype_rule(self):
        source = '''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="copy", target="xcore1002", backend="triton", entry_point="copy")
def candidate(lm, x: cake.Tensor((8, 128), "bf16"), y: cake.Tensor((8, 128), "bf16", mode="output")):
    compute = lm.role(execution_groups=[0])
    row = lm.program(x, axis=0, dimension=0, tile=1)
    with compute:
        value = lm.load(x[row, 0:128], id="load")
        lm.store(y[row, 0:128], value, id="store")
'''
        result = self.compiler.assess(parse(source).document)
        self.assertFalse(result.lowering_eligible)
        self.assertIn("MACA_DTYPE_UNQUALIFIED", [f.code for f in result.findings])
        self.assertNotIn("TARGET_OPERATION_UNSUPPORTED", [f.code for f in result.findings])

    def test_missing_timer_is_a_coverage_limitation_and_preserves_all_cases(self):
        document, _ = create_task("rmsnorm", backend="triton-metax", rows=8, columns=128)
        workload = WorkloadContract(document)
        policy = evaluation_policy(workload)
        self.assertEqual(tuple(policy["validation_case_ids"]), workload.case_ids)
        self.assertEqual(policy["search_evaluation"], "correctness_only")
        self.assertNotIn("paired_timing", policy)
        self.assertIsNone(platform_for("xcore1002").measurement_source)
        self.assertEqual(platform_for("xcore1002").attribution, "unavailable")

    def test_flagtree_distribution_version_is_not_used_as_triton_api_version(self):
        host = {"packages": {"flagtree": "0.5.1+metax3.1"}, "runtime": {"triton_version": "3.1.0"}}
        self.assertEqual(triton_version(host), "3.1.0")
        self.assertEqual(triton_version({"packages": {"triton": "3.7.1"}}), "3.7.1")


if __name__ == "__main__":
    unittest.main()
