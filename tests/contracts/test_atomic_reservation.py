"""Contract evidence for KDA-derived atomic slot ownership."""

from __future__ import annotations

import ast
import copy
import json
import sys
import unittest
from pathlib import Path

import jsonschema
import torch

from open_cake_ir.compiler.backends.cutedsl import preflight as cute_preflight
from open_cake_ir.compiler.backends.triton import emit, preflight
from open_cake_ir.compiler.ir import (
    AtomicMemoryOrder,
    AtomicMemoryScope,
    AtomicOp,
    BufferMode,
    OperationKind,
    Schedule,
    ScheduleParseError,
)
from open_cake_ir.compiler.schema import schedule_schema
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.verifier import verify

ROOT = Path(__file__).resolve().parents[2]
SCHEDULE = ROOT / "corpus" / "schedules" / "atomic-reservation-b8-smoke.json"
TARGET_PATH = ROOT / "compiler" / "targets" / "sm_100a.json"
sys.path.insert(0, str(ROOT / "tools"))

from kernel_oracles import MEASURE_BY_ENTRY_POINT  # noqa: E402


def _document() -> dict:
    return json.loads(SCHEDULE.read_text(encoding="utf-8"))


def _buffer(document: dict, name: str) -> dict:
    return next(item for item in document["buffers"] if item["name"] == name)


def _operation(document: dict, name: str = "reserve_positions") -> dict:
    return next(item for item in document["operations"] if item["id"] == name)


def _codes(document: dict) -> set[str]:
    return {
        finding.code
        for finding in verify(Schedule.from_dict(document), Target.load(TARGET_PATH))
    }


class AtomicReservationContractTest(unittest.TestCase):
    def test_schema_has_one_state_and_atomic_rmw_spelling(self) -> None:
        document = _document()
        jsonschema.Draft202012Validator(schedule_schema()).validate(document)
        schedule = Schedule.from_dict(document)
        atomic = schedule.operation("reserve_positions")

        self.assertEqual(atomic.kind, OperationKind.ATOMIC_RMW)
        self.assertEqual(atomic.parameters.op, AtomicOp.ADD)
        self.assertEqual(atomic.parameters.order, AtomicMemoryOrder.RELAXED)
        self.assertEqual(atomic.parameters.scope, AtomicMemoryScope.DEVICE)
        self.assertEqual(schedule.buffer("counts").mode, BufferMode.STATE)
        vocabulary = {kind.value for kind in OperationKind}
        for forbidden in ("route", "dispatch", "slot", "counter", "scatter", "moe"):
            self.assertNotIn(forbidden, vocabulary)

    def test_state_is_not_an_alias_for_input_or_unused_storage(self) -> None:
        document = _document()
        _buffer(document, "counts")["mode"] = "input"
        self.assertIn("ATOMIC_TARGET_MODE", _codes(document))

        document = _document()
        document["buffers"].append(
            {
                "name": "unused_state",
                "space": "global",
                "dtype": "int32",
                "shape": [1],
                "mode": "state",
            }
        )
        codes = _codes(document)
        self.assertIn("STATE_NOT_READ", codes)
        self.assertIn("STATE_NOT_WRITTEN", codes)

        document = _document()
        _buffer(document, "counts")["space"] = "register"
        codes = _codes(document)
        self.assertIn("BUFFER_IO_SPACE", codes)
        self.assertNotIn("ATOMIC_TARGET_SPACE", codes)

    def test_atomic_types_placement_and_shape_can_fail_independently(self) -> None:
        document = _document()
        _buffer(document, "counts")["dtype"] = "fp32"
        self.assertIn("ATOMIC_TARGET_DTYPE", _codes(document))

        document = _document()
        _buffer(document, "position_tile")["dtype"] = "fp32"
        self.assertIn("ATOMIC_RESULT_DTYPE", _codes(document))

        document = _document()
        _buffer(document, "position_tile")["space"] = "global"
        self.assertIn("ATOMIC_RESULT_SPACE", _codes(document))

        document = _document()
        _buffer(document, "position_tile")["shape"] = [4]
        self.assertIn("ACCESS_INDEXED_VALUE_SHAPE", _codes(document))

        document = _document()
        _operation(document)["writes"][0] = "positions"
        self.assertIn("ATOMIC_TARGET_EDGE", _codes(document))

        document = _document()
        _buffer(document, "counts")["shape"] = [4, 4]
        access = next(
            item
            for item in document["access_maps"]
            if item["operation"] == "reserve_positions"
        )
        access["indices"].append(
            {"source": "buffer", "name": "expert_id_tile"}
        )
        self.assertIn("ATOMIC_INDEX_FORM", _codes(document))

    def test_atomic_edges_and_closed_memory_model_can_fail(self) -> None:
        document = _document()
        _operation(document)["reads"].append("positions")
        codes = _codes(document)
        self.assertIn("ATOMIC_EDGE_COUNT", codes)
        self.assertIn("ATOMIC_INDEX_BUFFER_READS", codes)

        document = _document()
        _operation(document)["reads"].pop()
        codes = _codes(document)
        self.assertIn("ATOMIC_EDGE_COUNT", codes)
        self.assertNotIn("OP_ARITY", codes)

        document = _document()
        _operation(document)["parameters"]["order"] = "acquire"
        with self.assertRaisesRegex(ScheduleParseError, "admitted values are relaxed"):
            Schedule.from_dict(document)

        document = _document()
        _operation(document)["parameters"]["value"] = 1 << 31
        self.assertIn("ATOMIC_VALUE_RANGE", _codes(document))

    def test_triton_route_requires_the_exact_target_contract(self) -> None:
        target_document = json.loads(TARGET_PATH.read_text(encoding="utf-8"))
        target_document["instruction_contracts"].remove(
            "triton.atomic_add.i32.relaxed.gpu"
        )
        findings = preflight(
            Schedule.load(SCHEDULE), Target.from_dict(target_document)
        )
        self.assertEqual(
            [finding.code for finding in findings],
            ["TRITON_ATOMIC_CONTRACT_UNSUPPORTED"],
        )

    def test_cute_route_explicitly_refuses_mutable_state(self) -> None:
        findings = cute_preflight(
            Schedule.load(SCHEDULE), Target.load(TARGET_PATH)
        )
        self.assertIn("CUTE_STATE_UNSUPPORTED", {item.code for item in findings})

    def test_emission_preserves_address_memory_model_and_old_value(self) -> None:
        source = emit(Schedule.load(SCHEDULE), Target.load(TARGET_PATH)).source
        ast.parse(source)
        self.assertIn("reserve_positions_old = tl.atomic_add(", source)
        self.assertIn('sem="relaxed"', source)
        self.assertIn('scope="gpu"', source)
        self.assertIn("(expert_id_tile >= 0)", source)
        self.assertIn("(expert_id_tile < D_COUNTS_0)", source)
        self.assertIn(
            "position_tile = tl.where(((expert_id_tile >= 0) & "
            "(expert_id_tile < D_COUNTS_0)), reserve_positions_old, 0)",
            source,
        )
        self.assertIn(
            "def cake_atomic_reservation_b8_smoke(expert_ids, counts, out=None):",
            source,
        )
        self.assertIn("        maxnreg=64,", source)

    def test_contention_is_judged_as_a_permutation_not_lane_order(self) -> None:
        expert_ids = torch.tensor(
            [[0, 0, -1, 1], [1, 0, 1, 0]], dtype=torch.int32
        )
        initial_counts = torch.tensor([3, 5], dtype=torch.int32)
        observed = torch.tensor(
            [[6, 3, 0, 7], [5, 5, 6, 4]], dtype=torch.int32
        )
        final_counts = torch.tensor([7, 8], dtype=torch.int32)
        measure = MEASURE_BY_ENTRY_POINT["cake_atomic_reservation_b8_smoke"]

        mismatch, measured, passed = measure(
            observed,
            expert_ids,
            initial_counts,
            (expert_ids, final_counts, observed),
            torch,
            1e-5,
        )
        self.assertEqual(mismatch, 0)
        self.assertEqual(
            measured,
            {
                "unique_old_values": True,
                "masked_zero": True,
                "final_counts_match": True,
            },
        )
        self.assertTrue(passed)

        duplicate = copy.deepcopy(observed)
        duplicate[1, 3] = 5
        mismatch, _, passed = measure(
            duplicate,
            expert_ids,
            initial_counts,
            (expert_ids, final_counts, duplicate),
            torch,
            1e-5,
        )
        self.assertGreater(mismatch, 0)
        self.assertFalse(passed)


if __name__ == "__main__":
    unittest.main()
