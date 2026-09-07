"""Contract evidence for the KDA-derived FP8 block-scale vertical slice."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import jsonschema

from open_cake_ir.compiler.backends.triton import emit
from open_cake_ir.compiler.ir import Schedule, ScheduleParseError
from open_cake_ir.compiler.schema import schedule_schema
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.verifier import verify

ROOT = Path(__file__).resolve().parents[2]
POSITIVE = ROOT / "corpus" / "schedules" / "block-scaled-gemm-b1-smoke.json"
DRIFT = ROOT / "corpus" / "schedules" / "block-scaled-gemm-b1-smoke-scale-drift.json"
TARGET = Target.load(ROOT / "compiler" / "targets" / "sm_100a.json")


def _document(path: Path = POSITIVE) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _codes(document: dict) -> set[str]:
    return {finding.code for finding in verify(Schedule.from_dict(document), TARGET)}


class BlockScaleContractTest(unittest.TestCase):
    def test_schema_and_typed_ir_share_the_relation(self) -> None:
        document = _document()
        jsonschema.Draft202012Validator(schedule_schema()).validate(document)
        schedule = Schedule.from_dict(document)
        relation = schedule.buffer("sfa").scale_of
        self.assertEqual(relation.buffer, "a")
        self.assertEqual(relation.granularity, (1, 128))
        self.assertEqual(relation.axis_order, (1, 0))

    def test_axis_order_is_structurally_closed(self) -> None:
        document = _document()
        document["buffers"][2]["scale_of"]["axis_order"] = [1, 1]
        with self.assertRaisesRegex(ScheduleParseError, "repeats a data axis"):
            Schedule.from_dict(document)

    def test_derived_shape_can_fail(self) -> None:
        document = _document()
        document["buffers"][2]["shape"] = [3, 16]
        self.assertIn("SCALE_SHAPE_MISMATCH", _codes(document))

    def test_scale_dtype_can_fail(self) -> None:
        document = _document()
        document["buffers"][2]["dtype"] = "fp16"
        self.assertIn("SCALE_DTYPE", _codes(document))

    def test_load_propagation_can_fail(self) -> None:
        document = _document()
        document["buffers"][7]["shape"] = [4, 16]
        document["buffers"][7]["scale_of"]["granularity"] = [1, 64]
        self.assertIn("SCALE_RELATION_DRIFT", _codes(document))

    def test_mma_association_can_fail(self) -> None:
        self.assertIn("MMA_SCALE_ASSOCIATION", _codes(_document(DRIFT)))

    def test_backend_subset_can_fail_without_erasing_the_relation(self) -> None:
        document = _document()
        for index in (3, 8):
            document["buffers"][index]["shape"] = [2, 2]
            document["buffers"][index]["scale_of"]["granularity"] = [64, 128]
        self.assertIn("MMA_BLOCK_SCALE_UNLOWERABLE", _codes(document))

    def test_emission_applies_each_scale_to_its_partial_dot(self) -> None:
        source = emit(Schedule.load(POSITIVE), TARGET).source
        self.assertIn("a_tile_block_0, a_tile_block_1 = tl.split", source)
        self.assertIn("sfa_tile_block_0[:, None] * sfb_tile_block_0", source)
        self.assertIn("sfa_tile_block_1[:, None] * sfb_tile_block_1", source)
        self.assertNotIn("tl.dot_scaled", source)


if __name__ == "__main__":
    unittest.main()
