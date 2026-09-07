"""Exact Target admission for the gfx1151 register-only reshape primitive."""

from __future__ import annotations

import unittest
from pathlib import Path

from open_cake_ir.compiler.ir import OperationKind
from open_cake_ir.compiler.target import Target


ROOT = Path(__file__).resolve().parents[2]


class Gfx1151ReshapeAdmissionTests(unittest.TestCase):
    def test_reshape_is_admitted_by_gfx1151_only(self) -> None:
        gfx1151 = Target.load(ROOT / "compiler/targets/gfx1151.json")
        sm100a = Target.load(ROOT / "compiler/targets/sm_100a.json")

        self.assertIn(OperationKind.RESHAPE, gfx1151.operation_kinds)
        self.assertNotIn(OperationKind.RESHAPE, sm100a.operation_kinds)


if __name__ == "__main__":
    unittest.main()
