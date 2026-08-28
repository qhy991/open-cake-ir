from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler  # noqa: E402


class ReluPrimitiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_relu_is_one_canonical_unary_elementwise_lowering(self) -> None:
        assessment = self.compiler.assess_file(
            ROOT / "corpus/schedules/relu-b8-smoke.json"
        )

        self.assertTrue(assessment.accepted)
        self.assertTrue(assessment.lowering_eligible)
        source = self.compiler.lower(assessment).source
        self.assertIn("# CAKE_OP:relu", source)
        self.assertIn("y_tile = tl.maximum(x_tile, 0.0)", source)
        self.assertNotIn("qsa", source)

    def test_relu_refuses_a_result_dtype_drift_before_lowering(self) -> None:
        assessment = self.compiler.assess_file(
            ROOT / "corpus/schedules/relu-b8-smoke-dtype-drift.json"
        )

        self.assertFalse(assessment.accepted)
        self.assertFalse(assessment.lowering_eligible)
        self.assertEqual(
            [finding.code for finding in assessment.findings],
            ["ELEMENTWISE_RESULT_DTYPE", "RESIDENCY_BOUND"],
        )


if __name__ == "__main__":
    unittest.main()
