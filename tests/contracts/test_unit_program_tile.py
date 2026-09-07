"""A unit-width program tile is still a vector when AccessMap says program_tile."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler  # noqa: E402


SCHEDULE_PATH = ROOT / "corpus/schedules/rmsnorm-b8-smoke.json"


def _unit_row_schedule() -> dict[str, object]:
    document = json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))
    document["schedule_id"] = "rmsnorm-b8-unit-row-program-tile"
    document["program_map"]["axes"][0]["tile"] = 1
    matrix = {"x_tile", "sq", "normed", "y_tile"}
    vector = {"sumsq", "meansq", "shifted", "inv_rms"}
    for buffer in document["buffers"]:
        if buffer["name"] in matrix:
            buffer["shape"] = [1, 128]
        elif buffer["name"] in vector:
            buffer["shape"] = [1]
    return document


class UnitProgramTileTests(unittest.TestCase):
    def test_access_map_owns_vector_semantics_even_when_tile_is_one(self) -> None:
        compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        assessment = compiler.assess(_unit_row_schedule())

        self.assertTrue(assessment.accepted, assessment.findings)
        self.assertTrue(assessment.lowering_eligible, assessment.findings)
        self.assertNotIn(
            "ACCESS_PROGRAM_AXIS_UNTILED",
            {finding.code for finding in assessment.findings},
        )
        lowering = compiler.lower(assessment)
        self.assertEqual(lowering.toolchain_requirements["grid"], [512, 8, 1])
        self.assertEqual(
            lowering.toolchain_requirements["compile_constants"]["BLOCK_ROW_BLOCK"],
            1,
        )
        self.assertIn(
            "row_block_offsets = row_block * BLOCK_ROW_BLOCK + "
            "tl.arange(0, BLOCK_ROW_BLOCK)",
            lowering.source,
        )

    def test_unit_vector_keeps_shape_distinct_from_a_scalar_program_coordinate(self) -> None:
        compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        document = _unit_row_schedule()
        for access in document["access_maps"]:
            for index in access["indices"]:
                if index.get("source") == "program_tile" and index.get("name") == "row_block":
                    index["source"] = "program"
        assessment = compiler.assess(document)
        self.assertFalse(assessment.accepted)
        self.assertIn("LOAD_ACCESS_SHAPE_MISMATCH", {finding.code for finding in assessment.findings})


if __name__ == "__main__":
    unittest.main()
