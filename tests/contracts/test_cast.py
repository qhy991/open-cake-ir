from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler  # noqa: E402


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
