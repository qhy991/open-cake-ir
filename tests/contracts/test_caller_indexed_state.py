from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.emit_cutedsl import preflight as cute_preflight
from open_cake_ir.compiler.emit_triton import emit
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.profile_model import profile_envelope
from open_cake_ir.compiler.schema import schedule_schema
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.verifier import verify
from open_cake_ir.compiler.work import work_bound


ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "corpus/schedules/caller-indexed-state-b4-smoke.json"
SHIFT = ROOT / "corpus/schedules/disjoint-state-shift-b4-smoke.json"
SHIFT_OVERLAP = (
    ROOT / "corpus/schedules/disjoint-state-shift-b4-smoke-overlap-drift.json"
)
OUTER = ROOT / "corpus/schedules/outer-b8-smoke.json"
OUTER_DRIFT = ROOT / "corpus/schedules/outer-b8-smoke-shape-drift.json"
SCALAR = ROOT / "corpus/schedules/scalar-broadcast-b8-smoke.json"
TARGET = Target.load(ROOT / "compiler/targets/sm_100a.json")


def _document(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _blocking(document: dict) -> set[str]:
    return {
        finding.code
        for finding in verify(Schedule.from_dict(document), TARGET)
        if finding.blocks_lowering
    }


def _buffer(document: dict, name: str) -> dict:
    return next(item for item in document["buffers"] if item["name"] == name)


def _operation(document: dict, name: str) -> dict:
    return next(item for item in document["operations"] if item["id"] == name)


class CallerIndexedStateContractTest(unittest.TestCase):
    def test_positive_is_typed_verified_profiled_and_lowered(self) -> None:
        document = _document(STATE)
        Draft202012Validator(schedule_schema()).validate(document)
        schedule = Schedule.from_dict(document)
        self.assertEqual([], [f.code for f in verify(schedule, TARGET) if f.blocks_lowering])

        source = emit(schedule, TARGET).source
        compile(source, str(STATE), "exec")
        self.assertIn("state + slot * 12", source)
        self.assertIn("tl.where((slot >= 0), next_tile, 0.0)", source)
        self.assertIn("tuple(state.stride()) != (12, 4, 1)", source)
        self.assertNotIn("not state.is_contiguous()", source)

        bound = work_bound(schedule)
        self.assertIsNotNone(bound)
        self.assertIn("state", bound.partially_addressed)
        self.assertIn("output", bound.partially_addressed)
        profile = profile_envelope(schedule, TARGET, lowered_source=source)
        self.assertEqual(["state"], profile.lowering["strided_global_buffers"])
        scoreboard = next(
            metric
            for metric in profile.ncu_metrics
            if metric.metric
            == "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"
        )
        self.assertIn(
            "concrete-strided global buffer state", scoreboard.reasons
        )

    def test_stride_and_relation_failures_are_local(self) -> None:
        cases = []

        dense = _document(STATE)
        _buffer(dense, "state")["strides"] = [8, 4, 1]
        cases.append((dense, "BUFFER_STRIDES_NONCANONICAL"))

        overlapping = _document(STATE)
        _buffer(overlapping, "state")["strides"] = [0, 4, 1]
        cases.append((overlapping, "BUFFER_STRIDES_WRITABLE"))

        wrong_dtype = _document(STATE)
        _buffer(wrong_dtype, "cache_indices")["dtype"] = "fp32"
        cases.append((wrong_dtype, "UNIQUE_INDEX_BUFFER_CONTRACT"))

        bad_sentinel = _document(STATE)
        _buffer(bad_sentinel, "cache_indices")["unique_index"]["sentinel"] = 0
        cases.append((bad_sentinel, "UNIQUE_INDEX_SENTINEL_RANGE"))

        wrong_domain = _document(STATE)
        _buffer(wrong_domain, "cache_indices")["unique_index"]["dimension"] = 1
        cases.append((wrong_domain, "STATE_STORE_DOMAIN_MISMATCH"))

        no_provenance = _document(STATE)
        del _buffer(no_provenance, "cache_indices")["unique_index"]
        cases.append((no_provenance, "STORE_VALID_INDEX_PROVENANCE"))

        for document, expected in cases:
            with self.subTest(expected=expected):
                self.assertIn(expected, _blocking(document))

    def test_out_of_rank_unique_state_index_fails_closed(self) -> None:
        document = _document(STATE)
        operation = _operation(document, "store_state")
        access = next(
            item
            for item in document["access_maps"]
            if item["operation"] == operation["id"] and item["buffer"] == "state"
        )
        valid_if = operation["parameters"]["valid_if"]
        component = next(
            item
            for item in access["indices"]
            if item.get("source") == "buffer" and item.get("name") == valid_if
        )
        access["indices"].remove(component)
        access["indices"].insert(0, {"source": "dimension", "dimension": 0})
        access["indices"].append(component)

        assessment = Compiler.load(ROOT, ROOT / "compiler/revision.json").assess(
            document
        )
        self.assertIn(
            "ACCESS_RANK",
            {finding.code for finding in assessment.findings},
        )

    def test_inactive_policy_is_destination_specific(self) -> None:
        output_no_effect = _document(STATE)
        _operation(output_no_effect, "store_output")["parameters"][
            "inactive"
        ] = "no_effect"
        self.assertIn(
            "OUTPUT_STORE_INACTIVE_POLICY", _blocking(output_no_effect)
        )

        state_zero = _document(STATE)
        _operation(state_zero, "store_state")["parameters"][
            "inactive"
        ] = "write_zero"
        self.assertIn("STATE_STORE_INACTIVE_POLICY", _blocking(state_zero))

    def test_disjoint_state_writers_are_proved_not_asserted(self) -> None:
        positive = Schedule.load(SHIFT)
        self.assertEqual(
            [], [f.code for f in verify(positive, TARGET) if f.blocks_lowering]
        )
        source = emit(positive, TARGET).source
        compile(source, str(SHIFT), "exec")
        self.assertEqual(4, source.count("tl.store("))

        self.assertIn("BUFFER_MULTIPLE_WRITERS", _blocking(_document(SHIFT_OVERLAP)))

    def test_cute_route_refuses_the_successor_storage_relations(self) -> None:
        document = _document(STATE)
        document["lowering"] = {
            "backend": "cutlass_cute_dsl",
            "entry_point": "cake_caller_indexed_state_b4_cute",
        }
        findings = {
            finding.code
            for finding in cute_preflight(Schedule.from_dict(document), TARGET)
        }
        self.assertIn("CUTE_STATE_UNSUPPORTED", findings)
        self.assertIn("CUTE_BUFFER_STRIDES_UNSUPPORTED", findings)
        self.assertIn("CUTE_UNIQUE_INDEX_UNSUPPORTED", findings)


class OuterPrimitiveTest(unittest.TestCase):
    def test_outer_is_typed_counted_and_lowered(self) -> None:
        document = _document(OUTER)
        Draft202012Validator(schedule_schema()).validate(document)
        schedule = Schedule.from_dict(document)
        self.assertEqual([], [f.code for f in verify(schedule, TARGET) if f.blocks_lowering])
        source = emit(schedule, TARGET).source
        compile(source, str(OUTER), "exec")
        self.assertIn("left_tile[:, None].to(tl.float32)", source)
        self.assertIn("right_tile[None, :].to(tl.float32)", source)
        bound = work_bound(schedule)
        self.assertIsNotNone(bound)
        self.assertEqual(32, bound.flops)
        profile = profile_envelope(schedule, TARGET, lowered_source=source)
        self.assertEqual(1, profile.lowering["outer_operations"])

    def test_outer_shape_drift_is_rejected_before_lowering(self) -> None:
        self.assertEqual(
            {"OUTER_RESULT_SHAPE"}, _blocking(_document(OUTER_DRIFT))
        )

    def test_rank_zero_outer_input_fails_closed(self) -> None:
        document = _document(OUTER)
        operation = _operation(document, "form_product")
        _buffer(document, operation["reads"][0])["shape"] = []

        assessment = Compiler.load(ROOT, ROOT / "compiler/revision.json").assess(
            document
        )
        self.assertIn(
            "OUTER_INPUT_SHAPE",
            {finding.code for finding in assessment.findings},
        )


class RegisterScalarTest(unittest.TestCase):
    def test_runtime_scalar_broadcasts_without_an_array_axis(self) -> None:
        document = _document(SCALAR)
        Draft202012Validator(schedule_schema()).validate(document)
        schedule = Schedule.from_dict(document)
        self.assertEqual([], [f.code for f in verify(schedule, TARGET) if f.blocks_lowering])
        source = emit(schedule, TARGET).source
        compile(source, str(SCALAR), "exec")
        self.assertIn("scaled = value_tile * scale_value", source)

    def test_rank_zero_is_register_scratch_only(self) -> None:
        document = _document(SCALAR)
        scalar = _buffer(document, "scale_value")
        scalar["space"] = "global"
        scalar["mode"] = "input"
        with self.assertRaisesRegex(ValueError, "empty only for register scratch"):
            Schedule.from_dict(document)


if __name__ == "__main__":
    unittest.main()
