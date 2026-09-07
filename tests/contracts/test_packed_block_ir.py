"""Contracts for the minimal GGML packed-block storage relation.

This slice owns record custody only.  It deliberately adds no decode, quantize, dot,
Target or lowering behavior, so admitting these documents cannot be mistaken for Q4
MMVQ execution support.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from open_cake_ir.compiler.ir import (
    PACKED_BLOCK_FORMATS,
    ByteOrder,
    DType,
    PackedBlockField,
    PackedBlockFormat,
    Schedule,
    ScheduleParseError,
)
from open_cake_ir.compiler import emit_cutedsl, emit_triton
from open_cake_ir.compiler.schema import schedule_schema
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.verifier import verify


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "corpus/schedules/swiglu-b8-smoke-gfx1151.json"
TARGET = Target.load(ROOT / "compiler/targets/gfx1151.json")


def _document() -> dict[str, object]:
    document = json.loads(BASE.read_text(encoding="utf-8"))
    document["buffers"].extend(
        [
            {
                "name": "weight_q4_raw",
                "space": "global",
                "dtype": "uint8",
                "shape": [2, 18],
                "mode": "input",
                "packed_block": {
                    "format": "ggml_q4_0_v1",
                    "record_axis": 1,
                },
            },
            {
                "name": "activation_q8_raw",
                "space": "global",
                "dtype": "uint8",
                "shape": [3, 36],
                "mode": "input",
                "packed_block": {
                    "format": "ggml_q8_1_v1",
                    "record_axis": 1,
                },
            },
        ]
    )
    return document


def _buffer(document: dict[str, object], name: str) -> dict[str, object]:
    return next(
        value
        for value in document["buffers"]
        if value["name"] == name
    )


def _packed_codes(document: dict[str, object]) -> set[str]:
    schedule = Schedule.from_dict(document)
    return {
        finding.code
        for finding in verify(schedule, TARGET)
        if finding.code.startswith("PACKED_BLOCK_")
    }


class PackedBlockRegistryTests(unittest.TestCase):
    def test_registry_owns_exact_q4_and_q8_record_abi(self) -> None:
        q4 = PACKED_BLOCK_FORMATS[PackedBlockFormat.GGML_Q4_0_V1]
        q8 = PACKED_BLOCK_FORMATS[PackedBlockFormat.GGML_Q8_1_V1]

        self.assertEqual(DType.UINT8.itemsize, 1)
        self.assertEqual(DType.INT8.itemsize, 1)
        self.assertEqual(
            (q4.logical_extent, q4.record_bytes, q4.record_alignment_bytes),
            (32, 18, 2),
        )
        self.assertIs(q4.byte_order, ByteOrder.LITTLE)
        self.assertEqual(
            q4.fields,
            (
                PackedBlockField("d", 0, DType.FP16, 1),
                PackedBlockField("qs", 2, DType.UINT8, 16),
            ),
        )
        self.assertEqual(
            q4.nibble_logical_order,
            tuple(value for byte in range(16) for value in (byte, byte + 16)),
        )
        self.assertEqual(
            (q8.logical_extent, q8.record_bytes, q8.record_alignment_bytes),
            (32, 36, 4),
        )
        self.assertIs(q8.byte_order, ByteOrder.LITTLE)
        self.assertEqual(
            q8.fields,
            (
                PackedBlockField("d", 0, DType.FP16, 1),
                PackedBlockField("s", 2, DType.FP16, 1),
                PackedBlockField("qs", 4, DType.INT8, 32),
            ),
        )
        self.assertEqual(q8.nibble_logical_order, ())


class PackedBlockParseAndSchemaTests(unittest.TestCase):
    def test_parser_and_generated_schema_admit_both_exact_relations(self) -> None:
        document = _document()

        schedule = Schedule.from_dict(document)
        packed_schema = schedule_schema()["properties"]["buffers"]["items"][
            "properties"
        ]["packed_block"]

        self.assertFalse(packed_schema["additionalProperties"])
        self.assertEqual(
            packed_schema["required"],
            ["format", "record_axis"],
        )
        self.assertEqual(
            packed_schema["properties"]["format"]["enum"],
            ["ggml_q4_0_v1", "ggml_q8_1_v1"],
        )
        relations = {
            buffer.name: buffer.packed_block for buffer in schedule.buffers
        }
        self.assertEqual(
            relations["weight_q4_raw"].format,
            PackedBlockFormat.GGML_Q4_0_V1,
        )
        self.assertEqual(relations["weight_q4_raw"].record_axis, 1)
        self.assertEqual(
            relations["activation_q8_raw"].format,
            PackedBlockFormat.GGML_Q8_1_V1,
        )
        self.assertEqual(_packed_codes(document), set())

    def test_relation_rejects_extra_fields_in_parser_and_schema(self) -> None:
        document = _document()
        _buffer(document, "weight_q4_raw")["packed_block"]["extra"] = True

        with self.assertRaisesRegex(ScheduleParseError, "unknown fields.*extra"):
            Schedule.from_dict(document)
        packed_schema = schedule_schema()["properties"]["buffers"]["items"][
            "properties"
        ]["packed_block"]
        self.assertFalse(packed_schema["additionalProperties"])
        self.assertNotIn("extra", packed_schema["properties"])


class PackedBlockVerifierTests(unittest.TestCase):
    def test_record_extent_drift_is_local_for_all_four_neighbours(self) -> None:
        cases = (
            ("weight_q4_raw", 17),
            ("weight_q4_raw", 19),
            ("activation_q8_raw", 35),
            ("activation_q8_raw", 37),
        )
        for name, extent in cases:
            with self.subTest(buffer=name, extent=extent):
                document = _document()
                _buffer(document, name)["shape"][1] = extent
                self.assertEqual(
                    _packed_codes(document),
                    {"PACKED_BLOCK_RECORD_EXTENT"},
                )

    def test_int8_cannot_pretend_to_be_the_raw_record_storage(self) -> None:
        document = _document()
        _buffer(document, "activation_q8_raw")["dtype"] = "int8"

        self.assertEqual(_packed_codes(document), {"PACKED_BLOCK_DTYPE"})

    def test_record_axis_must_resolve(self) -> None:
        document = _document()
        _buffer(document, "weight_q4_raw")["packed_block"]["record_axis"] = 2

        self.assertEqual(_packed_codes(document), {"PACKED_BLOCK_RECORD_AXIS"})

    def test_record_axis_must_be_the_contiguous_last_dimension(self) -> None:
        document = _document()
        packed = _buffer(document, "weight_q4_raw")
        packed["shape"] = [18, 2]
        packed["packed_block"]["record_axis"] = 0

        self.assertEqual(_packed_codes(document), {"PACKED_BLOCK_RECORD_AXIS"})

    def test_packed_storage_has_one_stage(self) -> None:
        document = _document()
        _buffer(document, "weight_q4_raw")["stages"] = 2

        self.assertEqual(_packed_codes(document), {"PACKED_BLOCK_STAGES"})

    def test_record_start_obeys_the_format_alignment(self) -> None:
        document = _document()
        _buffer(document, "weight_q4_raw")["byte_offset"] = 1

        self.assertEqual(_packed_codes(document), {"PACKED_BLOCK_ALIGNMENT"})

        q4_aligned = _document()
        _buffer(q4_aligned, "weight_q4_raw")["byte_offset"] = 2
        self.assertNotIn("PACKED_BLOCK_ALIGNMENT", _packed_codes(q4_aligned))

        q8_misaligned = _document()
        _buffer(q8_misaligned, "activation_q8_raw")["byte_offset"] = 2
        self.assertEqual(
            _packed_codes(q8_misaligned),
            {"PACKED_BLOCK_ALIGNMENT"},
        )

    def test_packed_storage_cannot_also_be_an_fp8_scale_relation(self) -> None:
        document = _document()
        packed = _buffer(document, "weight_q4_raw")
        packed["scale_of"] = {
            "buffer": "up",
            "granularity": [1, 1, 1],
            "axis_order": [0, 1, 2],
        }

        self.assertIn("PACKED_BLOCK_SCALE_CONFLICT", _packed_codes(document))


class RawByteBackendSurfaceTests(unittest.TestCase):
    def test_dtype_tables_name_raw_bytes_without_claiming_decode(self) -> None:
        self.assertEqual(emit_triton._TL_DTYPE[DType.UINT8], "tl.uint8")
        self.assertEqual(emit_triton._TL_DTYPE[DType.INT8], "tl.int8")
        self.assertEqual(emit_triton._TORCH_DTYPE[DType.UINT8], "torch.uint8")
        self.assertEqual(emit_triton._TORCH_DTYPE[DType.INT8], "torch.int8")
        self.assertEqual(emit_cutedsl._CUTLASS_DTYPE[DType.UINT8], "cutlass.Uint8")
        self.assertEqual(emit_cutedsl._CUTLASS_DTYPE[DType.INT8], "cutlass.Int8")
        self.assertEqual(emit_cutedsl._TORCH_DTYPE[DType.UINT8], "torch.uint8")
        self.assertEqual(emit_cutedsl._TORCH_DTYPE[DType.INT8], "torch.int8")
        self.assertTrue(
            {DType.UINT8, DType.INT8} <= emit_triton.SUPPORTED_DTYPES
        )
        self.assertTrue(
            {DType.UINT8, DType.INT8} <= emit_cutedsl.SUPPORTED_DTYPES
        )

    def test_triton_projects_raw_signature_while_cute_refuses_packed_semantics(self) -> None:
        document = _document()
        document["buffers"].append(
            {
                "name": "plain_i8",
                "space": "global",
                "dtype": "int8",
                "shape": [4],
                "mode": "input",
            }
        )
        schedule = Schedule.from_dict(document)

        triton_failures = emit_triton.preflight(schedule, TARGET)
        emission = emit_triton.emit(schedule, TARGET)
        cute_codes = {
            finding.code for finding in emit_cutedsl.preflight(schedule, TARGET)
        }

        self.assertEqual(triton_failures, ())
        self.assertEqual(
            emission.toolchain["signature"]["weight_q4_raw"], "*u8"
        )
        self.assertEqual(emission.toolchain["signature"]["plain_i8"], "*i8")
        self.assertIn("torch.uint8", emission.source)
        self.assertIn("torch.int8", emission.source)
        self.assertIn("CUTE_PACKED_BLOCK_UNSUPPORTED", cute_codes)


if __name__ == "__main__":
    unittest.main()
