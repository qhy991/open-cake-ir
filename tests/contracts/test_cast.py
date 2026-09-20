from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler, frontend  # noqa: E402


class CastPrimitiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_bf16_to_fp32_conversion_is_declared_and_lowered(self) -> None:
        assessment = self.compiler.assess_file(
            ROOT / "corpus/schedules/cast-b8-smoke.json"
        )

        self.assertTrue(assessment.accepted)
        self.assertTrue(assessment.lowering_eligible)
        source = self.compiler.lower(assessment).source
        self.assertIn("# CAKE_OP:cast_x", source)
        self.assertIn("y_tile = x_tile.to(tl.float32)", source)

    def integer_cast(self, source_dtype="int32", output_dtype="fp32"):
        source = f'''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="integer-cast", target="gfx1151", backend="triton", entry_point="run", grid=(1, 1, 1))
def candidate(lm, x: cake.Tensor((32,), "{source_dtype}"),
              y: cake.Tensor((32,), "{output_dtype}", mode="output")):
    compute = lm.role(execution_groups=[0])
    with compute:
        values = lm.load(x[:], id="load_x")
        converted = lm.cast(values, to="{output_dtype}", id="cast_x")
        lm.store(y[:], converted, coalesced=False, id="store_y")
'''
        return self.compiler.assess(frontend.parse(source).document)

    def test_signed_integer_to_float_is_an_explicit_directed_conversion(self):
        assessment = self.integer_cast()
        self.assertTrue(assessment.lowering_eligible, assessment.findings)
        source = self.compiler.lower(assessment).source
        self.assertIn("converted = values.to(tl.float32)", source)

    def test_reverse_and_other_integer_conversions_remain_refused_by_cast_typing(self):
        for source, output in (("fp32", "int32"), ("int32", "bf16"),
                               ("int32", "fp16"), ("int32", "int32")):
            with self.subTest(source=source, output=output):
                assessment = self.integer_cast(source, output)
                self.assertFalse(assessment.lowering_eligible)
                self.assertIn("CAST_DTYPE_UNSUPPORTED", [f.code for f in assessment.findings])

    def test_result_dtype_must_equal_the_declared_conversion(self) -> None:
        assessment = self.compiler.assess_file(
            ROOT / "corpus/schedules/cast-b8-smoke-result-drift.json"
        )

        self.assertFalse(assessment.accepted)
        self.assertFalse(assessment.lowering_eligible)
        self.assertIn(
            "CAST_RESULT_DTYPE",
            [finding.code for finding in assessment.findings],
        )


if __name__ == "__main__":
    unittest.main()
