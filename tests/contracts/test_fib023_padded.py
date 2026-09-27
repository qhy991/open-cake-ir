"""FIB 023's one-pass masked Cake mapping and its owned refusal boundary."""
from __future__ import annotations

import unittest
from pathlib import Path

from open_cake_ir.compiler import Compiler, Program
from open_cake_ir.compiler.frontend import parse
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.solx_fib.b300_rmsnorm023 import padded_one_pass_source
from open_cake_ir.tasks.solx_fib.workload import SPECS, workload_document

ROOT = Path(__file__).resolve().parents[2]
TASK = "fib_rmsnorm_h1536"


class Fib023PaddedMappingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_every_official_batch_lowers_one_masked_row_program(self) -> None:
        for rows in SPECS[TASK]["batches"]:
            with self.subTest(rows=rows):
                workload = WorkloadContract(workload_document(
                    TASK, rows=rows, columns=1536, backend="triton-b300"))
                schedule = parse(padded_one_pass_source(workload)).document
                self.assertEqual(schedule["outputs"], ["out"])
                self.assertEqual(schedule["program_map"]["axes"][1]["tile"], 2048)
                self.assertEqual(schedule["tile_loops"], [])
                self.assertEqual(
                    [op["id"] for op in schedule["operations"]
                     if op["kind"] == "load"], ["load_x", "load_weight"])
                assessment = self.compiler.assess(schedule)
                self.assertFalse(
                    [f for f in assessment.findings if f.blocks_lowering],
                    assessment.findings)
                lowered = self.compiler.lower_program(Program.from_schedule(schedule))
                lowered.validate_binding()
                self.assertEqual(lowered.lowerings[0].toolchain_requirements["grid"],
                                 [rows, 1, 1])
                source = lowered.lowerings[0].source
                self.assertIn("BLOCK_COLUMN=2048", source)
                self.assertIn("column_offsets < 1536", source)
                self.assertIn("square_sum / 1536.0", source)

    def test_unpadded_full_slice_is_refused_by_the_arange_rule(self) -> None:
        workload = WorkloadContract(workload_document(
            TASK, rows=539, columns=1536, backend="triton-b300"))
        source = padded_one_pass_source(workload)
        source = source.replace(
            "    column = lm.program(x, axis=1, dimension=1, tile=2048)\n", "")
        source = (source.replace("x[row, column]", "x[row, :]")
                        .replace("weight[column]", "weight[:]")
                        .replace("out[row, column]", "out[row, :]"))
        assessment = self.compiler.assess(parse(source).document)
        self.assertFalse(assessment.lowering_eligible)
        self.assertIn("TRITON_ARANGE_RANGE_UNSUPPORTED",
                      {finding.code for finding in assessment.findings})

    def test_other_target_and_operator_are_refused_by_the_task_recipe(self) -> None:
        b200 = WorkloadContract(workload_document(
            TASK, rows=7, columns=1536, backend="triton-b200"))
        with self.assertRaisesRegex(ValueError, "exact B300 task"):
            padded_one_pass_source(b200)
        other = WorkloadContract(workload_document(
            "fib_rmsnorm_h512", rows=7, columns=512, backend="triton-b300"))
        with self.assertRaisesRegex(ValueError, "exact B300 task"):
            padded_one_pass_source(other)


if __name__ == "__main__":
    unittest.main()
