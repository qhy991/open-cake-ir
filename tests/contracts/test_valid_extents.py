"""Contract evidence for device-resident valid-prefix extents."""

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
POSITIVE = ROOT / "corpus" / "schedules" / "ragged-zero-pad-b1-smoke.json"
GROUPED = ROOT / "corpus" / "schedules" / "ragged-grouped-gemm-b1-smoke.json"
GROUP_DRIFT = (
    ROOT / "corpus" / "schedules" / "ragged-grouped-gemm-b1-smoke-group-drift.json"
)
TARGET = Target.load(ROOT / "compiler" / "targets" / "sm_100a.json")


def _document() -> dict:
    return json.loads(POSITIVE.read_text(encoding="utf-8"))


def _codes(document: dict) -> set[str]:
    return {finding.code for finding in verify(Schedule.from_dict(document), TARGET)}


class ValidExtentContractTest(unittest.TestCase):
    def test_schema_and_typed_ir_share_one_relation(self) -> None:
        document = _document()
        jsonschema.Draft202012Validator(schedule_schema()).validate(document)
        relation = Schedule.from_dict(document).buffer("ragged").valid_extent
        self.assertEqual(relation.dimension, 1)
        self.assertEqual(relation.buffer, "lengths")
        self.assertEqual(relation.indexed_by, (0,))

    def test_index_axes_are_structurally_unique(self) -> None:
        document = _document()
        document["buffers"][0]["valid_extent"]["indexed_by"] = [0, 0]
        with self.assertRaisesRegex(ScheduleParseError, "repeats a data axis"):
            Schedule.from_dict(document)

    def test_valid_dimension_can_fail(self) -> None:
        document = _document()
        document["buffers"][0]["valid_extent"]["dimension"] = 3
        self.assertIn("VALID_EXTENT_DIMENSION", _codes(document))

    def test_extent_dtype_can_fail(self) -> None:
        document = _document()
        document["buffers"][1]["dtype"] = "fp32"
        self.assertIn("VALID_EXTENT_DTYPE", _codes(document))

    def test_extent_storage_contract_can_fail(self) -> None:
        document = _document()
        document["buffers"][1]["mode"] = "output"
        self.assertIn("VALID_EXTENT_BUFFER_CONTRACT", _codes(document))

    def test_data_storage_contract_can_fail(self) -> None:
        document = _document()
        document["buffers"][0]["space"] = "register"
        document["buffers"][0]["mode"] = "scratch"
        self.assertIn("VALID_EXTENT_DATA_SPACE", _codes(document))

    def test_derived_extent_shape_can_fail(self) -> None:
        document = _document()
        document["buffers"][1]["shape"] = [5]
        self.assertIn("VALID_EXTENT_SHAPE_MISMATCH", _codes(document))

    def test_backend_subset_can_fail_without_erasing_the_relation(self) -> None:
        document = _document()
        document["buffers"][0]["valid_extent"]["indexed_by"] = [0, 2]
        document["buffers"][1]["shape"] = [4, 16]
        self.assertIn("VALID_EXTENT_ACCESS_UNLOWERABLE", _codes(document))

    def test_emission_derives_the_extent_load_and_row_predicate(self) -> None:
        source = emit(Schedule.load(POSITIVE), TARGET).source
        self.assertIn(
            "load_ragged_ragged_valid_extent = tl.load(lengths + group)", source
        )
        self.assertIn(
            "mask=ragged_d1_offsets[:, None] < load_ragged_ragged_valid_extent",
            source,
        )
        self.assertNotIn("valid_extent", source.split("# CAKE_OP:store_dense", 1)[1])

    def test_grouped_gemm_composes_without_new_vocabulary(self) -> None:
        schedule = Schedule.load(GROUPED)
        self.assertEqual(_codes(json.loads(GROUPED.read_text())), {
            "RESIDENCY_BOUND"
        })
        source = emit(schedule, TARGET).source
        self.assertIn("group = tl.program_id(0)", source)
        self.assertIn("load_a_a_valid_extent = tl.load(lengths + group)", source)
        self.assertIn("group * N_M_BLOCK * N_K_LOOP", source)
        self.assertIn("group * N_N_BLOCK * D_B_2", source)
        self.assertIn("group * D_C_1 * D_C_2", source)
        self.assertIn("acc = tl.zeros((16, 16), tl.float32)", source)
        self.assertIn("acc += tl.dot(a_tile, tl.trans(b_tile))", source)
        self.assertEqual(source.count("for k in tl.range("), 1)

    def test_scalar_program_extent_cannot_overrun_another_buffer(self) -> None:
        document = json.loads(GROUP_DRIFT.read_text(encoding="utf-8"))
        self.assertIn("ACCESS_PROGRAM_EXTENT_MISMATCH", _codes(document))


if __name__ == "__main__":
    unittest.main()
