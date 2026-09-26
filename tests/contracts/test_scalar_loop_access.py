"""A unit tile can address a chunk axis without adding a value axis."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Schedule
from open_cake_ir.compiler.ir.vocabulary import AccessIndexKind


ROOT = Path(__file__).resolve().parents[2]


def document() -> dict:
    value = json.loads((ROOT / "corpus/schedules/fma-b8-smoke.json").read_text())
    value["schedule_id"] = "fma-b8-scalar-chunk"
    for buffer in value["buffers"]:
        if buffer["space"] == "global":
            buffer["shape"] = [8, 2, 128]
    for access in value["access_maps"]:
        access["indices"] = [
            {"source": "program", "name": "batch"},
            {"source": "loop", "name": "chunk"},
            {"source": "dimension", "dimension": 2},
        ]
    value["tile_loops"] = [{
        "name": "chunks", "iterator": "chunk", "buffer": "a", "dimension": 1,
        "tile": 1, "body": [op["id"] for op in value["operations"]],
        "range_options": {
            "num_stages": 1, "loop_unroll_factor": 1, "flatten": False,
            "warp_specialize": False, "disallow_acc_multi_buffer": True,
            "disable_licm": False,
        },
    }]
    return value


class ScalarLoopAccess(unittest.TestCase):
    def test_unit_loop_is_scalar_and_lowered_as_chunk_coordinate(self):
        value = document()
        schedule = Schedule.from_dict(value)
        self.assertEqual(schedule.access_maps[0].indices[1].source, AccessIndexKind.LOOP)
        self.assertFalse(schedule.access_maps[0].indices[1].is_vector)
        compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        assessment = compiler.assess(value)
        self.assertEqual([f.code for f in assessment.findings if f.blocks_lowering], [])
        source = compiler.lower(assessment).source
        self.assertIn("chunk_offsets = chunk + tl.arange(0, BLOCK_CHUNKS)", source)
        self.assertIn("chunk * D_A_2", source)
        self.assertNotIn("chunk_offsets * D_A_2", source)

    def test_wide_loop_and_short_source_are_refused(self):
        compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        value = document()
        value["tile_loops"][0]["tile"] = 2
        self.assertIn("ACCESS_LOOP_SCALAR_TILE",
                      {f.code for f in compiler.assess(value).findings})
        value = document()
        next(b for b in value["buffers"] if b["name"] == "b")["shape"][1] = 1
        self.assertIn("ACCESS_LOOP_EXTENT_MISMATCH",
                      {f.code for f in compiler.assess(value).findings})


if __name__ == "__main__":
    unittest.main()
