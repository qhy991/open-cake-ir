from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler  # noqa: E402


class IndexExpandPrimitiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_affine_block_expansion_is_explicit_and_flat(self) -> None:
        assessment = self.compiler.assess_file(
            ROOT / "corpus/schedules/index-expand-b8-smoke.json"
        )

        self.assertTrue(assessment.accepted)
        self.assertTrue(assessment.lowering_eligible)
        source = self.compiler.lower(assessment).source
        self.assertIn("# CAKE_OP:expand_tokens", source)
        self.assertIn("block_tile[:, None] * 4", source)
        self.assertIn("tl.reshape(expand_tokens_matrix, (32,))", source)

    def test_non_int32_source_fails_before_lowering(self) -> None:
        assessment = self.compiler.assess_file(
            ROOT / "corpus/schedules/index-expand-b8-smoke-dtype-drift.json"
        )

        self.assertFalse(assessment.accepted)
        self.assertFalse(assessment.lowering_eligible)
        self.assertIn(
            "INDEX_EXPAND_DTYPE",
            [finding.code for finding in assessment.findings],
        )


if __name__ == "__main__":
    unittest.main()
