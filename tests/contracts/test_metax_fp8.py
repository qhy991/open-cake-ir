"""MACA storage and widening do not silently admit unqualified FP8 arithmetic."""
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.compiler.backends import triton
from open_cake_ir.compiler.backends.common import EmitError
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import declared_target

ROOT = Path(__file__).resolve().parents[2]


class MetaxFP8Admission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_storage_and_both_decoders_preserve_the_explicit_pointer_abi(self):
        for suffix, output_type in (("copy", "*fp8e4nv"),
                                    ("decode-fp16", "*fp16"), ("decode-fp32", "*fp32")):
            with self.subTest(route=suffix):
                result = self.compiler.assess_file(ROOT / f"corpus/schedules/xcore1002-fp8-{suffix}.py")
                self.assertTrue(result.lowering_eligible, result.findings)
                lowering = self.compiler.lower(result)
                self.assertEqual(dict(lowering.toolchain_requirements["signature"]),
                                 {"x": "*fp8e4nv", "out": output_type})
                self.assertEqual(lowering.toolchain_requirements["warp_size"], 64)
                self.assertNotIn("tl.dot(", lowering.source)

    def test_scalar_compiler_failure_is_refused_before_emission(self):
        path = ROOT / "corpus/schedules/xcore1002-fp8-scalar-cast-unsupported.py"
        result = self.compiler.assess_file(path)
        owned = [f for f in result.findings if f.code == "MACA_FP8_SCALAR_CAST_UNSUPPORTED"]
        self.assertEqual(len(owned), 1)
        self.assertEqual(owned[0].path, "operations[1]")
        self.assertFalse(result.lowering_eligible)
        with self.assertRaises(EmitError):
            triton.emit(Schedule.from_dict(frontend.read_schedule(path).document), declared_target("xcore1002"))

    def test_other_cast_directions_have_their_own_refusal(self):
        template = (ROOT / "corpus/schedules/xcore1002-fp8-decode-fp32.py").read_text()
        for source_dtype, destination in (("fp8_e4m3", "bf16"), ("fp32", "fp8_e4m3"),
                                           ("fp16", "fp8_e4m3"), ("bf16", "fp8_e4m3")):
            with self.subTest(source=source_dtype, destination=destination):
                source = template.replace('"fp8_e4m3"', '"SOURCE"').replace('"fp32"', f'"{destination}"')
                source = source.replace('"SOURCE"', f'"{source_dtype}"')
                document = frontend.parse(source).document
                result = self.compiler.assess(document)
                self.assertFalse(result.lowering_eligible)
                self.assertIn("MACA_FP8_CAST_UNQUALIFIED", [f.code for f in result.findings])
                self.assertNotIn("CAST_DTYPE_UNSUPPORTED", [f.code for f in result.findings])
                # This measured MACA limitation must not restrict another vendor.
                document["target"] = "sm_103a"
                findings = triton.preflight(Schedule.from_dict(document), declared_target("sm_103a"))
                self.assertFalse(any(f.code.startswith("MACA_") for f in findings))

    def test_fp8_matrix_does_not_inherit_storage_qualification(self):
        source = (ROOT / "corpus/schedules/xcore1002-dot-fp16-64.py").read_text()
        source = source.replace('"fp16"', '"fp8_e4m3"').replace("triton.dot.fp16_fp32", "triton.dot.fp8e4m3_fp32")
        document = frontend.parse(source).document
        result = self.compiler.assess(document)
        self.assertFalse(result.lowering_eligible)
        codes = [f.code for f in result.findings]
        self.assertIn("TARGET_INSTRUCTION_UNSUPPORTED", codes)
        self.assertNotIn("MACA_DTYPE_UNQUALIFIED", codes)
        # Compiler stops at the undeclared instruction before backend preflight.
        # Direct emission must independently refuse the unqualified FP8 operation.
        findings = triton.preflight(Schedule.from_dict(document), declared_target("xcore1002"))
        self.assertIn("MACA_FP8_OPERATION_UNQUALIFIED", [f.code for f in findings])
        with self.assertRaises(EmitError):
            triton.emit(Schedule.from_dict(document), declared_target("xcore1002"))
