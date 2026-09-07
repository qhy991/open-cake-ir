"""Focused contracts for the two-source-tile loop-carried top-k cadence."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from open_cake_ir.compiler import Compiler, EmitError
from open_cake_ir.compiler.backends import triton
from open_cake_ir.compiler.ir import Schedule, ScheduleParseError
from open_cake_ir.compiler.schema import schedule_schema
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.verifier import FindingCategory, verify

ROOT = Path(__file__).resolve().parents[2]
STREAMING = ROOT / "corpus/schedules/top-k-streaming-b8-smoke.json"
RESIDENT = ROOT / "corpus/schedules/top-k-b8-smoke.json"
INT32_STREAMING = ROOT / "corpus/schedules/top-k-int32-streaming-b8-drift.json"
TARGET = Target.load(ROOT / "compiler/targets/sm_100a.json")


def _document(path: Path = STREAMING) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _top_k(document: dict) -> dict:
    return next(operation for operation in document["operations"] if operation["kind"] == "top_k")


def _two_tile_document(path: Path = STREAMING) -> dict:
    document = _document(path)
    _top_k(document)["parameters"]["source_tiles_per_merge"] = 2
    return document


def _finding(schedule: Schedule, code: str):
    return next(item for item in verify(schedule, TARGET) if item.code == code)


class CanonicalSyntaxTest(unittest.TestCase):
    def test_omission_is_the_one_source_tile_cadence(self) -> None:
        operation = Schedule.load(STREAMING).operation("select_blocks")
        assert operation is not None

        self.assertEqual(operation.parameters.source_tiles_per_merge, 1)

    def test_explicit_two_is_typed_and_schema_admitted(self) -> None:
        document = _two_tile_document()
        schedule = Schedule.from_dict(document)
        operation = schedule.operation("select_blocks")
        assert operation is not None

        self.assertEqual(operation.parameters.source_tiles_per_merge, 2)
        self.assertEqual(
            list(Draft202012Validator(schedule_schema()).iter_errors(document)),
            [],
        )

    def test_explicit_one_another_factor_or_null_is_not_a_second_spelling(self) -> None:
        for value in (1, 3, None):
            with self.subTest(value=value):
                document = _document()
                _top_k(document)["parameters"]["source_tiles_per_merge"] = value

                with self.assertRaisesRegex(
                    ScheduleParseError,
                    r"operations\[1\]\.parameters\.source_tiles_per_merge",
                ):
                    Schedule.from_dict(document)

                errors = list(
                    Draft202012Validator(schedule_schema()).iter_errors(document)
                )
                self.assertTrue(
                    any(
                        tuple(error.absolute_path)[-1:] == ("source_tiles_per_merge",)
                        for error in errors
                    )
                )


class LegalityTest(unittest.TestCase):
    def test_two_tiles_requires_loop_carried_state(self) -> None:
        document = _two_tile_document(RESIDENT)
        schedule = Schedule.from_dict(document)

        finding = _finding(schedule, "TOP_K_MERGE_CADENCE_REQUIRES_ACROSS_LOOP")
        self.assertEqual(
            finding.path,
            "operations[1].parameters.source_tiles_per_merge",
        )
        self.assertIs(finding.category, FindingCategory.DATA_CONSISTENCY)

    def test_delayed_results_cannot_be_read_inside_the_same_loop(self) -> None:
        document = _two_tile_document()
        document["tile_loops"][0]["body"].append("store_indices")
        schedule = Schedule.from_dict(document)

        finding = _finding(schedule, "TOP_K_MERGE_CADENCE_OUTPUT_READ_IN_LOOP")
        self.assertEqual(finding.path, "operations[2].reads")
        self.assertIn("top_indices", finding.message)
        self.assertIs(finding.category, FindingCategory.DATA_CONSISTENCY)

    def test_current_triton_slice_refuses_control_flow_drift(self) -> None:
        cases = (
            ("loop_unroll_factor", 2),
            ("warp_specialize", True),
            ("flatten", True),
        )
        for field, value in cases:
            with self.subTest(field=field):
                document = _two_tile_document()
                document["tile_loops"][0]["range_options"][field] = value
                schedule = Schedule.from_dict(document)

                findings = [
                    item
                    for item in triton.preflight(schedule, TARGET)
                    if item.code == "TRITON_TOP_K_TWO_TILE_CONTROL_FLOW_UNSUPPORTED"
                ]
                self.assertEqual(len(findings), 1)
                self.assertFalse(findings[0].blocks_acceptance)
                self.assertTrue(findings[0].blocks_lowering)
                self.assertNotIn(findings[0].code, {item.code for item in verify(schedule, TARGET)})
                assessment = Compiler.load(ROOT, ROOT / "compiler/revision.json").assess(document)
                self.assertTrue(assessment.accepted)
                self.assertFalse(assessment.lowering_eligible)
                with self.assertRaisesRegex(EmitError, "two-source-tile top_k control flow"):
                    triton.emit(schedule, TARGET)
                self.assertEqual(
                    findings[0].path,
                    f"tile_loops[0].range_options.{field}",
                )
                self.assertIs(
                    findings[0].category,
                    FindingCategory.HARDWARE_CONFORMANCE,
                )

    def test_existing_int32_carried_state_exclusion_remains_the_owner(self) -> None:
        schedule = Schedule.from_dict(_two_tile_document(INT32_STREAMING))
        codes = {item.code for item in verify(schedule, TARGET)}

        self.assertIn("TOP_K_INT32_ACROSS_LOOP_UNLOWERABLE", codes)
        self.assertNotIn("TOP_K_MERGE_CADENCE_REQUIRES_ACROSS_LOOP", codes)

    def test_supported_fp32_cadence_has_no_new_blocking_finding(self) -> None:
        schedule = Schedule.from_dict(_two_tile_document())
        blocked = {
            item.code
            for item in verify(schedule, TARGET)
            if item.blocks_lowering
        }

        self.assertFalse(
            blocked
            & {
                "TOP_K_MERGE_CADENCE_REQUIRES_ACROSS_LOOP",
                "TOP_K_MERGE_CADENCE_OUTPUT_READ_IN_LOOP",
                "TRITON_TOP_K_TWO_TILE_CONTROL_FLOW_UNSUPPORTED",
            }
        )


if __name__ == "__main__":
    unittest.main()
