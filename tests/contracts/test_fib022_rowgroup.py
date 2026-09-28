"""FIB 022 row-group mapping uses its own H512 ABI on exact NVIDIA Targets."""
from __future__ import annotations

import unittest
from pathlib import Path

from open_cake_ir.compiler import Compiler, Program
from open_cake_ir.compiler.frontend import parse
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.solx_fib.nvidia_rmsnorm_rowgroup import (
    qualified_rowgroup16_source, row_group_source,
)
from open_cake_ir.tasks.solx_fib.workload import SPECS, workload_document

ROOT = Path(__file__).resolve().parents[2]
TASK = "fib_rmsnorm_h512"


class Fib022RowGroupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_every_official_batch_lowers_at_its_exact_target(self) -> None:
        for backend, target in (("triton-b200", "sm_100a"),
                                ("triton-b300", "sm_103a")):
            for rows in SPECS[TASK]["batches"]:
                workload = WorkloadContract(workload_document(
                    TASK, rows=rows, columns=512, backend=backend))
                for tile in (4, 8, 16):
                    with self.subTest(target=target, rows=rows, tile=tile):
                        schedule = parse(row_group_source(
                            workload, rows_per_cta=tile)).document
                        self.assertEqual(schedule["program_map"]["axes"][0]["tile"], tile)
                        self.assertEqual(schedule["program_map"]["axes"][1]["tile"], 512)
                        reduction = next(op for op in schedule["operations"]
                                         if op["id"] == "sum_square")
                        self.assertEqual(reduction["parameters"]["axis"], 1)
                        assessment = self.compiler.assess(schedule)
                        self.assertFalse(
                            [f for f in assessment.findings if f.blocks_lowering],
                            assessment.findings)
                        lowered = self.compiler.lower_program(
                            Program.from_schedule(schedule))
                        lowered.validate_binding()
                        leaf = lowered.lowerings[0]
                        self.assertEqual(leaf.target, target)
                        self.assertEqual(leaf.toolchain_requirements["grid"],
                                         [(rows + tile - 1) // tile, 1, 1])
                        self.assertIn("mean_square = totals / 512.0", leaf.source)
                        self.assertIn("mask=", leaf.source)

    def test_021_measured_recipe_does_not_claim_022(self) -> None:
        workload = WorkloadContract(workload_document(
            TASK, rows=539, columns=512, backend="triton-b300"))
        with self.assertRaisesRegex(ValueError, "B300 task 021"):
            qualified_rowgroup16_source(workload)


if __name__ == "__main__":
    unittest.main()
