"""KDA's within-chunk decay product uses the existing typed scan operation."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

import jsonschema

from open_cake_ir.compiler import Compiler, Schedule, Target
from open_cake_ir.compiler.ir import ScheduleParseError
from open_cake_ir.compiler.performance.work import work_bound
from open_cake_ir.compiler.schema import schedule_schema
from open_cake_ir.compiler.verifier import verify

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "corpus/schedules/chunk-cumsum-b8-smoke.json"


def product_document() -> dict:
    value = json.loads(SOURCE.read_text())
    value["schedule_id"] = "chunk-cumprod-b8-kda-witness"
    value["lowering"]["entry_point"] = "cake_chunk_cumprod_b8_kda_witness"
    scan = value["operations"][1]
    scan["id"] = "chunk_cumprod"
    scan["parameters"]["op"] = "mul"
    value["operations"][2]["depends_on"] = ["chunk_cumprod"]
    return value


class PrefixProductContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_forward_and_reverse_multiply_prefixes_lower_explicitly(self):
        value = product_document()
        jsonschema.Draft202012Validator(schedule_schema()).validate(value)
        for direction, reverse in (("forward", "False"), ("reverse", "True")):
            with self.subTest(direction=direction):
                value["operations"][1]["parameters"]["direction"] = direction
                assessment = self.compiler.assess(value)
                self.assertTrue(assessment.lowering_eligible,
                                [(f.code, f.path) for f in assessment.findings if f.blocks_lowering])
                source = self.compiler.lower(assessment).source
                self.assertIn(
                    f"tl.cumprod(gate_tile.to(tl.float32), axis=0, reverse={reverse})",
                    source,
                )
                bound = work_bound(Schedule.from_dict(value))
                self.assertFalse(bound.flops_exact)
                self.assertIn("chunk_cumprod", bound.uncounted_arithmetic)

    def test_type_shape_and_closed_operator_refusals_remain_localized(self):
        value = product_document()
        value["operations"][1]["parameters"]["op"] = "product"
        with self.assertRaisesRegex(ScheduleParseError, "schedule.operations\\[1\\].parameters.op"):
            Schedule.from_dict(value)
        target = Target.load(ROOT / "compiler/targets/sm_100a.json")
        value = product_document()
        next(b for b in value["buffers"] if b["name"] == "cumulative_tile")["dtype"] = "bf16"
        self.assertIn("SCAN_DTYPE_MISMATCH",
                      {finding.code for finding in verify(Schedule.from_dict(value), target)})
        value = product_document()
        next(b for b in value["buffers"] if b["name"] == "cumulative_tile")["shape"] = [32,128]
        self.assertIn("SCAN_SHAPE_MISMATCH",
                      {finding.code for finding in verify(Schedule.from_dict(value), target)})


if __name__ == "__main__":
    unittest.main()
