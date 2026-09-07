"""KDA v1 weighted-combine composition and its numeric type boundary."""

from __future__ import annotations

import ast
import json
import sys
import unittest
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler.backends.triton import emit
from open_cake_ir.compiler.ir import OperationKind, Schedule
from open_cake_ir.compiler.schema import schedule_schema
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.verifier import verify

POSITIVE = ROOT / "corpus/schedules/kda-weighted-combine-b8-smoke.json"
TARGET = Target.load(ROOT / "compiler/targets/sm_100a.json")


def _document() -> dict:
    return json.loads(POSITIVE.read_text())


def _codes(document: dict) -> set[str]:
    return {finding.code for finding in verify(Schedule.from_dict(document), TARGET)}


def _buffer(document: dict, name: str) -> dict:
    return next(item for item in document["buffers"] if item["name"] == name)


class KDAWeightedCombineContractTest(unittest.TestCase):
    def test_existing_primitives_express_the_positive_schedule(self) -> None:
        document = _document()
        jsonschema.Draft202012Validator(schedule_schema()).validate(document)
        schedule = Schedule.from_dict(document)

        self.assertEqual(_codes(document), {"RESIDENCY_BOUND"})
        vocabulary = {kind.value for kind in OperationKind}
        for forbidden in ("combine", "scatter"):
            self.assertNotIn(forbidden, vocabulary)
        self.assertEqual(
            [operation.kind for operation in schedule.operations[-3:]],
            [OperationKind.ELEMENTWISE, OperationKind.REDUCE, OperationKind.STORE],
        )

    def test_mixed_elementwise_result_type_is_derived(self) -> None:
        document = _document()
        _buffer(document, "weighted_rows")["dtype"] = "bf16"
        self.assertIn("ELEMENTWISE_RESULT_DTYPE", _codes(document))

    def test_bf16_fp16_arithmetic_has_no_implicit_promotion(self) -> None:
        document = _document()
        _buffer(document, "route_weights")["dtype"] = "fp16"
        _buffer(document, "route_weight_tile")["dtype"] = "fp16"
        self.assertIn("ELEMENTWISE_DTYPE_UNSUPPORTED", _codes(document))

    def test_load_cannot_silently_change_dtype(self) -> None:
        document = _document()
        _buffer(document, "route_weights")["dtype"] = "bf16"
        dtype_codes = {code for code in _codes(document) if "DTYPE" in code}
        self.assertEqual(dtype_codes, {"LOAD_DTYPE_MISMATCH"})

    def test_reduce_result_is_fp32(self) -> None:
        document = _document()
        _buffer(document, "combined_row")["dtype"] = "bf16"
        self.assertIn("REDUCE_DTYPE_MISMATCH", _codes(document))

    def test_store_only_owns_the_admitted_final_narrowing(self) -> None:
        document = _document()
        _buffer(document, "output")["dtype"] = "int32"
        self.assertIn("STORE_DTYPE_UNSUPPORTED", _codes(document))

    def test_emission_keeps_index_weight_reduce_store_as_separate_steps(self) -> None:
        source = emit(Schedule.load(POSITIVE), TARGET).source
        ast.parse(source)
        self.assertIn("selected_rows = tl.load(", source)
        self.assertIn(
            "weighted_rows = selected_rows * route_weight_tile[:, None]", source
        )
        self.assertIn(
            "combined_row = tl.sum(weighted_rows.to(tl.float32), axis=0)", source
        )
        self.assertIn("dtype=torch.bfloat16", source)


if __name__ == "__main__":
    unittest.main()
