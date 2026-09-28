"""FIB 021 row-group Cake mapping and exact NVIDIA target boundaries."""
from __future__ import annotations

import unittest
from pathlib import Path

from open_cake_ir.compiler import Compiler, Program
from open_cake_ir.compiler.frontend import parse
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.solx_fib.nvidia_rmsnorm021 import (
    qualified_rowgroup16_source, row_group_source)
from open_cake_ir.tasks.solx_fib.workload import SPECS, workload_document

ROOT = Path(__file__).resolve().parents[2]
TASK = "fib_rmsnorm_h128"


class Fib021RowGroupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_all_official_batches_lower_each_bounded_row_group(self) -> None:
        for backend, target in (("triton-b200", "sm_100a"),
                                ("triton-b300", "sm_103a")):
            for rows in SPECS[TASK]["batches"]:
                workload = WorkloadContract(workload_document(
                    TASK, rows=rows, columns=128, backend=backend))
                for tile in (4, 8, 16):
                    with self.subTest(rows=rows, tile=tile, target=target):
                        schedule = parse(row_group_source(
                            workload, rows_per_cta=tile)).document
                        self.assertEqual(schedule["outputs"], ["out"])
                        self.assertEqual(schedule["program_map"]["axes"][0]["tile"], tile)
                        self.assertEqual(schedule["program_map"]["axes"][1]["tile"], 128)
                        self.assertEqual(
                            [op["id"] for op in schedule["operations"]
                             if op["kind"] == "load"], ["load_x", "load_weight"])
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
                        self.assertIn("axis=1", leaf.source)
                        self.assertIn("mask=", leaf.source)

    def test_other_target_task_and_row_tile_are_refused(self) -> None:
        workload = WorkloadContract(workload_document(
            TASK, rows=24, columns=128, backend="triton-b300"))
        with self.assertRaisesRegex(ValueError, "must be 4, 8, 16, 32 or 64"):
            row_group_source(workload, rows_per_cta=2)
        with self.assertRaisesRegex(ValueError, "bounded to four official large batches"):
            row_group_source(workload, rows_per_cta=32)
        other_target = WorkloadContract(workload_document(
            TASK, rows=24, columns=128, backend="triton-gfx1151"))
        with self.assertRaisesRegex(ValueError, "exact B200 or B300 task"):
            row_group_source(other_target)
        other = WorkloadContract(workload_document(
            "fib_rmsnorm_h512", rows=7, columns=512, backend="triton-b300"))
        with self.assertRaisesRegex(ValueError, "exact B200 or B300 task"):
            row_group_source(other)

    def test_large_row_groups_lower_at_exact_targets_only(self) -> None:
        for backend, target in (("triton-b200", "sm_100a"),
                                ("triton-b300", "sm_103a")):
            for rows in (49532, 65016, 396256, 520128):
                workload = WorkloadContract(workload_document(
                    TASK, rows=rows, columns=128, backend=backend))
                for tile in (32, 64):
                    with self.subTest(target=target, rows=rows, tile=tile):
                        source = row_group_source(workload, rows_per_cta=tile)
                        schedule = parse(source).document
                        self.assertEqual(schedule["program_map"]["axes"][0]["tile"], tile)
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

    def test_measured_t16_recipe_keeps_its_batch_boundary(self) -> None:
        for rows in (49532, 65016, 520128):
            with self.subTest(rows=rows):
                workload = WorkloadContract(workload_document(
                    TASK, rows=rows, columns=128, backend="triton-b300"))
                source = qualified_rowgroup16_source(workload)
                self.assertEqual(source, row_group_source(
                    workload, rows_per_cta=16))
                assessment = self.compiler.assess(parse(source).document)
                self.assertFalse(
                    [f for f in assessment.findings if f.blocks_lowering],
                    assessment.findings)
                self.assertIn("axis=1", self.compiler.lower(assessment).source)
        for rows in (24, 2528, 396256):
            with self.subTest(rows=rows):
                workload = WorkloadContract(workload_document(
                    TASK, rows=rows, columns=128, backend="triton-b300"))
                with self.assertRaisesRegex(ValueError,
                                            "qualified only at R49532, R65016 and R520128"):
                    qualified_rowgroup16_source(workload)
        b200 = WorkloadContract(workload_document(
            TASK, rows=49532, columns=128, backend="triton-b200"))
        with self.assertRaisesRegex(ValueError, "require B300"):
            qualified_rowgroup16_source(b200)


if __name__ == "__main__":
    unittest.main()
