"""Lowering contracts for identity-only packed and signed raw-byte copies.

These cases prove parser-to-Triton custody only. They deliberately contain no field
decode, quantization, integer dot, correction, MMVQ integration or performance claim.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from open_cake_ir.compiler import Compiler, emit_triton
from open_cake_ir.compiler.ir import PackedBlockFormat, Schedule
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.verifier import verify


ROOT = Path(__file__).resolve().parents[2]
TARGET = Target.load(ROOT / "compiler/targets/gfx1151.json")
SM100_TARGET = Target.load(ROOT / "compiler/targets/sm_100a.json")
REVISION = ROOT / "compiler/revision.json"
Q4 = ROOT / "corpus/schedules/packed-q4_0-record-copy-gfx1151.json"
Q8 = ROOT / "corpus/schedules/packed-q8_1-record-copy-gfx1151.json"
I8 = ROOT / "corpus/schedules/int8-byte-copy-gfx1151.json"
DIRECT_DIMENSION = (
    ROOT
    / "corpus/schedules/packed-q8_1-direct-dimension-gfx1151-unsupported.json"
)
LOOP_SCHEDULE = ROOT / "corpus/schedules/flash-kmeans-b32-smoke-v2.json"
RESIDENCY_REPORT = "RESIDENCY_TARGET_UNMODELED"
ARANGE_FINDING = "TRITON_ARANGE_RANGE_UNSUPPORTED"
NEW_CASE_IDS = (
    "packed-q4_0-record-copy-gfx1151-accepted",
    "packed-q8_1-record-copy-gfx1151-accepted",
    "int8-byte-copy-gfx1151-accepted",
    "packed-q8_1-direct-dimension-gfx1151-unsupported",
)


class PackedRawCopyLoweringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = Compiler.load(ROOT, REVISION)

    def test_parser_verifier_and_preflight_admit_the_three_identity_copies(self) -> None:
        q4 = Schedule.load(Q4)
        q8 = Schedule.load(Q8)
        i8 = Schedule.load(I8)

        self.assertEqual(
            q4.buffer("src_raw").packed_block.format,
            PackedBlockFormat.GGML_Q4_0_V1,
        )
        self.assertEqual(
            q8.buffer("src_raw").packed_block.format,
            PackedBlockFormat.GGML_Q8_1_V1,
        )
        self.assertIsNone(i8.buffer("src_raw").packed_block)
        for schedule in (q4, q8, i8):
            with self.subTest(schedule=schedule.schedule_id):
                self.assertEqual(
                    [finding.code for finding in verify(schedule, TARGET)],
                    [RESIDENCY_REPORT],
                )
                self.assertEqual(emit_triton.preflight(schedule, TARGET), ())

    def test_positive_lowerings_have_raw_signatures_power_of_two_tiles_and_masks(self) -> None:
        cases = (
            (Q4, "*u8", "torch.uint8", 18, 32),
            (Q8, "*u8", "torch.uint8", 36, 64),
            (I8, "*i8", "torch.int8", 32, 32),
        )
        forbidden = (
            "tl.dot(",
            "sudot4",
            "nibble",
            "decode",
            "quantize",
            "dequantize",
            "mmvq",
        )
        for path, pointer, torch_dtype, record_bytes, tile in cases:
            with self.subTest(schedule=path.name):
                assessment = self.compiler.assess_file(path)
                self.assertTrue(assessment.accepted, assessment.findings)
                self.assertTrue(assessment.lowering_eligible, assessment.findings)
                self.assertEqual(
                    [finding.code for finding in assessment.findings],
                    [RESIDENCY_REPORT],
                )

                lowering = self.compiler.lower(assessment)
                requirements = lowering.toolchain_requirements
                self.assertEqual(
                    requirements["signature"],
                    {"src_raw": pointer, "dst_raw": pointer},
                )
                self.assertEqual(requirements["grid"], [4, 1, 1])
                self.assertEqual(
                    requirements["compile_constants"]["N_BYTE"], record_bytes
                )
                self.assertEqual(
                    requirements["compile_constants"]["BLOCK_BYTE"], tile
                )
                self.assertIn(
                    "byte_offsets = byte * BLOCK_BYTE + "
                    "tl.arange(0, BLOCK_BYTE)",
                    lowering.source,
                )
                self.assertIn("mask=byte_offsets < N_BYTE,", lowering.source)
                self.assertIn(
                    "mask=byte_offsets < D_DST_RAW_1,", lowering.source
                )
                self.assertIn(torch_dtype, lowering.source)
                lowered_source = lowering.source.lower()
                for marker in forbidden:
                    self.assertNotIn(marker, lowered_source)

    def test_direct_record_dimension_is_valid_ir_but_unlowerable_triton(self) -> None:
        schedule = Schedule.load(DIRECT_DIMENSION)
        self.assertEqual(
            [finding.code for finding in verify(schedule, TARGET)],
            [RESIDENCY_REPORT],
        )
        failures = emit_triton.preflight(schedule, TARGET)
        self.assertEqual(
            [(failure.code, failure.path) for failure in failures],
            [
                (ARANGE_FINDING, "access_maps[0].indices[1]"),
                (ARANGE_FINDING, "access_maps[1].indices[1]"),
            ],
        )

        assessment = self.compiler.assess_file(DIRECT_DIMENSION)
        self.assertTrue(assessment.accepted)
        self.assertFalse(assessment.lowering_eligible)
        self.assertEqual(
            [finding.code for finding in assessment.findings],
            [RESIDENCY_REPORT, ARANGE_FINDING, ARANGE_FINDING],
        )

    def test_preflight_checks_only_real_program_tile_and_tile_loop_ranges(self) -> None:
        program_tile_document = json.loads(Q4.read_text(encoding="utf-8"))
        program_tile_document["program_map"]["axes"][1]["tile"] = 24
        program_tile = Schedule.from_dict(program_tile_document)
        self.assertEqual(
            [
                (failure.code, failure.path)
                for failure in emit_triton.preflight(program_tile, TARGET)
                if failure.code == ARANGE_FINDING
            ],
            [(ARANGE_FINDING, "program_map.axes[1].tile")],
        )

        loop_document = json.loads(LOOP_SCHEDULE.read_text(encoding="utf-8"))
        loop_document["tile_loops"][0]["tile"] = 48
        loop = Schedule.from_dict(loop_document)
        self.assertEqual(
            [
                (failure.code, failure.path)
                for failure in emit_triton.preflight(loop, SM100_TARGET)
                if failure.code == ARANGE_FINDING
            ],
            [(ARANGE_FINDING, "tile_loops[0].tile")],
        )

    def test_manifest_retains_the_four_reviewed_packed_storage_cases(self) -> None:
        manifest = json.loads((ROOT / "corpus/manifest.json").read_text())
        self.assertEqual(tuple(case["case_id"] for case in manifest["cases"][-6:-2]), NEW_CASE_IDS)



if __name__ == "__main__":
    unittest.main()
