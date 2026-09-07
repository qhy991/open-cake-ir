"""Both authoring boundaries expose one canonical packed-Q8 primitive vocabulary."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import jsonschema

from open_cake_ir.compiler.ir import (
    CastParameters, DType, ElementwiseOp, ElementwiseParameters, OperationKind,
    OverflowPolicy, ReduceOp, ReduceParameters, ReductionAlgorithm,
    ReductionScope, ReshapeParameters, RoundingMode, Schedule, ScheduleParseError,
)
from open_cake_ir.compiler.schema import schedule_schema

ROOT = Path(__file__).resolve().parents[2]
POSITIVE = ROOT / "corpus/schedules/packed-q8_1-producer-gfx1151.json"
VALIDATOR = jsonschema.Draft202012Validator(schedule_schema())


def document():
    return json.loads(POSITIVE.read_text())


def operation(doc, name):
    return next(op for op in doc["operations"] if op["id"] == name)


class KernelAParserTests(unittest.TestCase):
    def test_existing_parameter_constructors_keep_their_defaults(self):
        reduction = ReduceParameters(ReduceOp.SUM, 0, ReductionScope.CTA)
        arithmetic = ElementwiseParameters(ElementwiseOp.MUL, None, None, None)
        cast = CastParameters(DType.FP16)
        self.assertIs(reduction.algorithm, ReductionAlgorithm.BACKEND)
        self.assertTrue(reduction.across_loop)
        self.assertIsNone(arithmetic.rounding)
        self.assertIsNone(cast.rounding)
        self.assertIsNone(cast.overflow)

    def test_all_ten_amd_inputs_agree_at_parser_and_schema_boundaries(self):
        paths = list((ROOT / "corpus/schedules").glob("*gfx1151*.json"))
        self.assertEqual(len(paths), 10)
        for path in paths:
            with self.subTest(schedule=path.name):
                value = json.loads(path.read_text())
                VALIDATOR.validate(value)
                Schedule.from_dict(value)

    def test_reshape_and_explicit_policies_are_typed(self):
        schedule = Schedule.from_dict(document())
        self.assertIsInstance(schedule.operation("reshape_blocks").parameters, ReshapeParameters)
        self.assertIs(schedule.operation("abs_activation").parameters.op, ElementwiseOp.ABS)
        self.assertIs(schedule.operation("make_d_fp32").parameters.op, ElementwiseOp.DIVIDE_NO_NAN)
        self.assertIs(schedule.operation("round_q").parameters.rounding, RoundingMode.NEAREST_AWAY_FROM_ZERO)
        cast = schedule.operation("cast_q_int8")
        self.assertIs(cast.kind, OperationKind.CAST)
        self.assertIs(cast.parameters.to, DType.INT8)
        self.assertIs(cast.parameters.rounding, RoundingMode.TOWARD_ZERO)
        self.assertIs(cast.parameters.overflow, OverflowPolicy.FORBID)
        self.assertIs(schedule.operation("reduce_sum").parameters.algorithm, ReductionAlgorithm.XOR_TREE_32)

    def test_invalid_parameters_are_refused_by_both_boundaries(self):
        invalid = [
            ("reshape_blocks", {"shape": [16, 32]}),
            ("round_q", {"op": "round"}),
            ("round_q", {"op": "round", "rounding": "nearest_even"}),
            ("round_q", {"op": "round", "rounding": "nearest_away_from_zero", "overflow": "ieee"}),
            ("round_q", {"op": "round", "rounding": "nearest_away_from_zero", "scalar": 0.0}),
            ("round_q", {"op": "round", "rounding": "nearest_away_from_zero", "broadcast_axis": 0}),
            ("abs_activation", {"op": "abs", "rounding": "nearest_away_from_zero"}),
            ("make_d_fp32", {"op": "divide_no_nan", "overflow": "ieee"}),
            ("cast_q_int8", {"to": "int8", "rounding": "toward_zero"}),
            ("cast_q_int8", {"to": "int8", "overflow": "forbid"}),
            ("cast_q_int8", {"to": "int8", "rounding": "unknown", "overflow": "forbid"}),
            ("cast_q_int8", {"to": "int8", "rounding": "toward_zero", "overflow": "unknown"}),
            ("cast_q_int8", {"to": "int8", "rounding": "toward_zero", "overflow": "forbid", "scalar": 0.0}),
            ("reduce_sum", {"op": "sum", "axis": 1, "scope": "cta", "algorithm": "warp_magic"}),
        ]
        for name, parameters in invalid:
            with self.subTest(operation=name, parameters=parameters):
                value = document()
                operation(value, name)["parameters"] = parameters
                with self.assertRaises(ScheduleParseError):
                    Schedule.from_dict(value)
                with self.assertRaises(jsonschema.ValidationError):
                    VALIDATOR.validate(value)

    def test_legacy_elementwise_cast_is_not_a_second_authoring_spelling(self):
        value = document()
        cast = operation(value, "cast_q_int8")
        cast["kind"] = "elementwise"
        cast["parameters"] = {"op": "cast", "rounding": "toward_zero", "overflow": "forbid"}
        with self.assertRaises(ScheduleParseError):
            Schedule.from_dict(value)
        with self.assertRaises(jsonschema.ValidationError):
            VALIDATOR.validate(value)


if __name__ == "__main__":
    unittest.main()
