"""Parser and generated-schema contracts for the Kernel A arithmetic vocabulary."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from open_cake_ir.compiler.ir import (
    ElementwiseOp,
    ElementwiseParameters,
    OperationKind,
    OverflowPolicy,
    ReduceOp,
    ReduceParameters,
    ReductionAlgorithm,
    ReductionScope,
    ReshapeParameters,
    RoundingMode,
    Schedule,
    ScheduleParseError,
)
from open_cake_ir.compiler.schema import schedule_schema


ROOT = Path(__file__).resolve().parents[2]
SWIGLU = ROOT / "corpus/schedules/swiglu-b8-smoke-gfx1151.json"
RMSNORM = ROOT / "corpus/schedules/llama-rmsnorm-mul-b8-gfx1151-r1-w8.json"


def _document(path: Path = SWIGLU) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _operation(document: dict[str, object], op_id: str) -> dict[str, object]:
    return next(
        operation
        for operation in document["operations"]
        if operation["id"] == op_id
    )


def _elementwise(parameters: dict[str, object]) -> ElementwiseParameters:
    document = _document()
    operation = _operation(document, "half_gate")
    operation["parameters"] = parameters
    return Schedule.from_dict(document).operation("half_gate").parameters


class KernelAParserTests(unittest.TestCase):
    def test_historical_parameter_constructors_keep_their_defaults(self) -> None:
        reduction = ReduceParameters(
            op=ReduceOp.SUM,
            axis=0,
            scope=ReductionScope.CTA,
        )
        elementwise = ElementwiseParameters(
            ElementwiseOp.MUL,
            None,
            None,
            None,
        )

        self.assertIs(reduction.algorithm, ReductionAlgorithm.BACKEND)
        self.assertIsNone(elementwise.rounding)
        self.assertIsNone(elementwise.overflow)

    def test_reshape_has_one_empty_parameter_spelling(self) -> None:
        document = _document()
        operation = _operation(document, "half_gate")
        operation["kind"] = "reshape"
        operation["parameters"] = {}

        parameters = Schedule.from_dict(document).operation("half_gate").parameters

        self.assertIsInstance(parameters, ReshapeParameters)
        extra = copy.deepcopy(document)
        _operation(extra, "half_gate")["parameters"] = {"shape": [16, 32]}
        with self.assertRaisesRegex(ScheduleParseError, "unknown fields.*shape"):
            Schedule.from_dict(extra)

    def test_abs_divide_no_nan_round_and_cast_parse_with_exact_arity_metadata(self) -> None:
        absolute = _elementwise({"op": "abs"})
        divide = _elementwise({"op": "divide_no_nan"})
        rounded = _elementwise(
            {"op": "round", "rounding": "nearest_away_from_zero"}
        )
        casted = _elementwise(
            {
                "op": "cast",
                "rounding": "toward_zero",
                "overflow": "forbid",
            }
        )

        self.assertEqual((absolute.op, absolute.arity_needed), (ElementwiseOp.ABS, 1))
        self.assertEqual(
            (divide.op, divide.arity_needed),
            (ElementwiseOp.DIVIDE_NO_NAN, 2),
        )
        self.assertEqual(rounded.op, ElementwiseOp.ROUND)
        self.assertIs(rounded.rounding, RoundingMode.NEAREST_AWAY_FROM_ZERO)
        self.assertIsNone(rounded.overflow)
        self.assertEqual(casted.op, ElementwiseOp.CAST)
        self.assertIs(casted.rounding, RoundingMode.TOWARD_ZERO)
        self.assertIs(casted.overflow, OverflowPolicy.FORBID)

    def test_round_requires_its_only_admitted_mode_and_forbids_overflow(self) -> None:
        invalid = (
            {"op": "round"},
            {"op": "round", "rounding": "nearest_even"},
            {"op": "round", "rounding": "toward_zero"},
            {
                "op": "round",
                "rounding": "nearest_away_from_zero",
                "overflow": "ieee",
            },
            {
                "op": "round",
                "rounding": "nearest_away_from_zero",
                "scalar": 0.0,
            },
            {
                "op": "round",
                "rounding": "nearest_away_from_zero",
                "broadcast_axis": 0,
            },
        )
        for parameters in invalid:
            with self.subTest(parameters=parameters), self.assertRaises(
                ScheduleParseError
            ):
                _elementwise(parameters)

    def test_cast_requires_explicit_valid_rounding_and_overflow(self) -> None:
        invalid = (
            {"op": "cast"},
            {"op": "cast", "rounding": "toward_zero"},
            {"op": "cast", "overflow": "forbid"},
            {"op": "cast", "rounding": "unknown", "overflow": "forbid"},
            {"op": "cast", "rounding": "toward_zero", "overflow": "unknown"},
            {
                "op": "cast",
                "rounding": "toward_zero",
                "overflow": "forbid",
                "scalar": 0.0,
            },
            {
                "op": "cast",
                "rounding": "toward_zero",
                "overflow": "forbid",
                "broadcast_axis": 0,
            },
        )
        for parameters in invalid:
            with self.subTest(parameters=parameters), self.assertRaises(
                ScheduleParseError
            ):
                _elementwise(parameters)

    def test_other_elementwise_ops_reject_rounding_and_overflow(self) -> None:
        invalid = (
            {"op": "abs", "rounding": "nearest_away_from_zero"},
            {"op": "divide_no_nan", "overflow": "ieee"},
            {"op": "mul", "rounding": "toward_zero", "overflow": "forbid"},
        )
        for parameters in invalid:
            with self.subTest(parameters=parameters), self.assertRaisesRegex(
                ScheduleParseError, "no defined effect"
            ):
                _elementwise(parameters)

    def test_reduce_defaults_to_backend_and_accepts_xor_tree_32(self) -> None:
        historical = _document(RMSNORM)
        reduction = _operation(historical, "sum_sq")
        self.assertNotIn("algorithm", reduction["parameters"])

        parsed = Schedule.from_dict(historical).operation("sum_sq").parameters
        self.assertIsInstance(parsed, ReduceParameters)
        self.assertIs(parsed.algorithm, ReductionAlgorithm.BACKEND)
        self.assertNotIn("algorithm", reduction["parameters"])

        exact = copy.deepcopy(historical)
        _operation(exact, "sum_sq")["parameters"]["algorithm"] = "xor_tree_32"
        exact_parameters = Schedule.from_dict(exact).operation("sum_sq").parameters
        self.assertIs(exact_parameters.algorithm, ReductionAlgorithm.XOR_TREE_32)

        invalid = copy.deepcopy(historical)
        _operation(invalid, "sum_sq")["parameters"]["algorithm"] = "warp_magic"
        with self.assertRaisesRegex(ScheduleParseError, "algorithm is unsupported"):
            Schedule.from_dict(invalid)


class KernelASchemaProjectionTests(unittest.TestCase):
    def test_schema_has_strict_dedicated_branches(self) -> None:
        schema = schedule_schema()
        operations = schema["properties"]["operations"]["items"]["allOf"]
        by_kind = {
            branch["if"]["properties"]["kind"]["const"]: branch["then"]
            for branch in operations
        }

        reshape = by_kind[OperationKind.RESHAPE.value]["properties"]["parameters"]
        self.assertEqual(reshape["required"], [])
        self.assertEqual(reshape["properties"], {})
        self.assertFalse(reshape["additionalProperties"])

        reduce = by_kind[OperationKind.REDUCE.value]["properties"]["parameters"]
        self.assertEqual(
            reduce["properties"]["algorithm"]["enum"],
            ["backend", "xor_tree_32"],
        )

        elementwise = by_kind[OperationKind.ELEMENTWISE.value]["properties"][
            "parameters"
        ]["oneOf"]
        round_branch = next(
            branch
            for branch in elementwise
            if branch["properties"]["op"].get("const") == "round"
        )
        cast_branch = next(
            branch
            for branch in elementwise
            if branch["properties"]["op"].get("const") == "cast"
        )
        ordinary = next(
            branch
            for branch in elementwise
            if "enum" in branch["properties"]["op"]
        )

        self.assertEqual(
            round_branch["properties"]["rounding"],
            {"const": "nearest_away_from_zero"},
        )
        self.assertNotIn("overflow", round_branch["properties"])
        self.assertNotIn("scalar", round_branch["properties"])
        self.assertNotIn("broadcast_axis", round_branch["properties"])
        self.assertEqual(
            cast_branch["properties"]["rounding"]["enum"],
            ["nearest_away_from_zero", "nearest_even", "toward_zero"],
        )
        self.assertEqual(
            cast_branch["properties"]["overflow"]["enum"],
            ["forbid", "ieee"],
        )
        self.assertNotIn("scalar", cast_branch["properties"])
        self.assertNotIn("broadcast_axis", cast_branch["properties"])
        self.assertNotIn("rounding", ordinary["properties"])
        self.assertNotIn("overflow", ordinary["properties"])
        self.assertTrue(all(not branch["additionalProperties"] for branch in elementwise))


if __name__ == "__main__":
    unittest.main()
