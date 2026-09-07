"""Contract evidence for reservation-owned runtime-indexed stores."""

from __future__ import annotations

import ast
import json
import sys
import unittest
from pathlib import Path

import jsonschema
import torch

from open_cake_ir.compiler.backends.triton import emit
from open_cake_ir.compiler.ir import OperationKind, Schedule
from open_cake_ir.compiler.schema import schedule_schema
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.verifier import verify

ROOT = Path(__file__).resolve().parents[2]
SCHEDULE = ROOT / "corpus" / "schedules" / "reservation-owned-store-b8-smoke.json"
TARGET = Target.load(ROOT / "compiler" / "targets" / "sm_100a.json")
sys.path.insert(0, str(ROOT / "tools"))

from kernel_oracles import MEASURE_BY_ENTRY_POINT, ORACLE_BY_ENTRY_POINT  # noqa: E402


def _document() -> dict:
    return json.loads(SCHEDULE.read_text(encoding="utf-8"))


def _operation(document: dict, op_id: str) -> dict:
    return next(item for item in document["operations"] if item["id"] == op_id)


def _access(document: dict, op_id: str) -> dict:
    return next(item for item in document["access_maps"] if item["operation"] == op_id)


def _buffer(document: dict, name: str) -> dict:
    return next(item for item in document["buffers"] if item["name"] == name)


def _codes(document: dict) -> set[str]:
    return {finding.code for finding in verify(Schedule.from_dict(document), TARGET)}


class ReservationOwnedIndexedStoreContractTest(unittest.TestCase):
    def test_schema_adds_no_store_or_routing_vocabulary(self) -> None:
        document = _document()
        jsonschema.Draft202012Validator(schedule_schema()).validate(document)
        schedule = Schedule.from_dict(document)

        self.assertEqual(schedule.operation("store_dispatched").kind, OperationKind.STORE)
        vocabulary = {kind.value for kind in OperationKind}
        for forbidden in ("scatter", "route", "dispatch", "unique_store"):
            self.assertNotIn(forbidden, vocabulary)

    def test_positive_has_only_the_existing_residency_report(self) -> None:
        findings = verify(Schedule.load(SCHEDULE), TARGET)
        self.assertEqual([finding.code for finding in findings], ["RESIDENCY_BOUND"])
        self.assertFalse(any(finding.blocks_lowering for finding in findings))

    def test_store_reads_value_then_access_owned_indices(self) -> None:
        document = _document()
        store = _operation(document, "store_dispatched")
        store["reads"][1:] = list(reversed(store["reads"][1:]))
        self.assertIn("STORE_INDEX_BUFFER_READS", _codes(document))

    def test_returned_value_must_have_one_atomic_producer(self) -> None:
        document = _document()
        _access(document, "store_dispatched")["indices"][1]["name"] = "expert_id_tile"
        self.assertIn("STORE_INDEX_RESERVATION_UNPROVEN", _codes(document))

    def test_reservation_coordinates_have_one_canonical_form(self) -> None:
        document = _document()
        _access(document, "store_dispatched")["indices"][1] = {
            "source": "dimension",
            "dimension": 1,
        }
        _operation(document, "store_dispatched")["reads"].pop()
        self.assertIn("STORE_INDEX_RESERVATION_FORM", _codes(document))

    def test_store_uses_the_atomic_target_index(self) -> None:
        document = _document()
        _access(document, "store_dispatched")["indices"][0]["name"] = "payload_tile"
        _operation(document, "store_dispatched")["reads"][1] = "payload_tile"
        self.assertIn("STORE_INDEX_RESERVATION_COORDINATES", _codes(document))

    def test_only_unit_increment_proves_unique_positions(self) -> None:
        document = _document()
        _operation(document, "reserve_positions")["parameters"]["value"] = 0
        self.assertIn("STORE_INDEX_RESERVATION_INCREMENT", _codes(document))

    def test_store_and_atomic_masks_share_the_reservation_domain(self) -> None:
        document = _document()
        _buffer(document, "dispatched")["shape"][0] = 5
        self.assertIn("STORE_INDEX_RESERVATION_DOMAIN", _codes(document))

    def test_reservation_executes_earlier_in_the_same_role(self) -> None:
        document = _document()
        operations = document["operations"]
        reservation = operations.pop(2)
        operations.append(reservation)
        self.assertIn("OP_READ_BEFORE_WRITE", _codes(document))

    def test_value_producer_cannot_be_declared_after_the_store(self) -> None:
        document = _document()
        operations = document["operations"]
        payload_load = operations.pop(1)
        operations.insert(3, payload_load)
        _operation(document, "store_dispatched")["depends_on"].remove("load_payloads")
        self.assertIn("OP_READ_BEFORE_WRITE", _codes(document))

    def test_barrier_cannot_make_an_atomic_register_visible_to_another_role(self) -> None:
        document = _document()
        document["roles"].append(
            {"name": "store_role", "warps": [4, 5, 6, 7]}
        )
        reservation = _operation(document, "reserve_positions")
        store = _operation(document, "store_dispatched")
        _operation(document, "load_expert_ids")["signals"] = ["reservation_ready"]
        _operation(document, "load_payloads")["signals"] = ["reservation_ready"]
        reservation["signals"] = ["reservation_ready"]
        store["role"] = "store_role"
        store["waits"] = ["reservation_ready"]
        document["barriers"] = [
            {
                "name": "reservation_ready",
                "count": 1,
                "producers": ["compute"],
                "consumers": ["store_role"],
                "mechanism": "barrier.sync",
            }
        ]
        codes = _codes(document)
        self.assertIn("STORE_INDEX_RESERVATION_ROLE", codes)
        self.assertNotIn("OP_CROSS_ROLE_RACE", codes)

    def test_int32_position_cannot_wrap_within_the_launch_domain(self) -> None:
        document = _document()
        _buffer(document, "expert_ids")["shape"][0] = 536_870_913
        _buffer(document, "payloads")["shape"][0] = 536_870_913
        self.assertIn("STORE_INDEX_RESERVATION_WRAP", _codes(document))

    def test_first_ownership_subset_executes_once_per_program(self) -> None:
        document = _document()
        document["tile_loops"] = [
            {
                "name": "repeat_store",
                "iterator": "iteration",
                "buffer": "payloads",
                "dimension": 0,
                "tile": 4,
                "body": ["store_dispatched"],
                "range_options": {
                    "num_stages": 1,
                    "loop_unroll_factor": 1,
                    "flatten": False,
                    "warp_specialize": False,
                    "disallow_acc_multi_buffer": True,
                    "disable_licm": False,
                },
            }
        ]
        self.assertIn("STORE_INDEX_RESERVATION_LOOP", _codes(document))

    def test_store_value_shape_matches_the_zipped_address_domain(self) -> None:
        document = _document()
        _buffer(document, "payload_tile")["shape"] = [4]
        self.assertIn("ACCESS_INDEXED_VALUE_SHAPE", _codes(document))

    def test_emission_reuses_the_zipped_address_and_bounds_both_coordinates(self) -> None:
        source = emit(Schedule.load(SCHEDULE), TARGET).source
        ast.parse(source)

        self.assertIn(
            "dispatched + expert_id_tile * D_DISPATCHED_1 + position_tile",
            source,
        )
        self.assertIn("(expert_id_tile >= 0)", source)
        self.assertIn("(expert_id_tile < D_DISPATCHED_0)", source)
        self.assertIn("(position_tile >= 0)", source)
        self.assertIn("(position_tile < D_DISPATCHED_1)", source)
        self.assertIn("payload_tile,", source)

    def test_oracle_judges_unordered_payload_sets_and_untouched_capacity(self) -> None:
        expert_ids = torch.tensor([[0, 0, -1], [1, 0, 1]], dtype=torch.int32)
        payloads = torch.tensor([[11, 12, 13], [14, 15, 16]], dtype=torch.int32)
        initial_counts = torch.tensor([1, 2], dtype=torch.int32)
        observed = torch.zeros((2, 8), dtype=torch.int32)
        observed[0, 1:4] = torch.tensor([15, 11, 12], dtype=torch.int32)
        observed[1, 2:4] = torch.tensor([16, 14], dtype=torch.int32)
        final_counts = torch.tensor([4, 4], dtype=torch.int32)
        measure = MEASURE_BY_ENTRY_POINT["cake_reservation_owned_store_b8_smoke"]

        mismatch, measured, passed = measure(
            observed,
            (expert_ids, payloads),
            initial_counts,
            (expert_ids, payloads, final_counts, observed),
            torch,
            1e-5,
        )
        self.assertEqual(mismatch, 0)
        self.assertEqual(
            measured,
            {
                "reserved_payloads_match": True,
                "untouched_slots_zero": True,
                "final_counts_match": True,
            },
        )
        self.assertTrue(passed)

        observed[0, 3] = 11
        mismatch, _, passed = measure(
            observed,
            (expert_ids, payloads),
            initial_counts,
            (expert_ids, payloads, final_counts, observed),
            torch,
            1e-5,
        )
        self.assertGreater(mismatch, 0)
        self.assertFalse(passed)

    def test_oracle_declares_its_exact_set_domain_before_launch(self) -> None:
        expert_ids = torch.tensor([[0, 0]], dtype=torch.int32)
        payloads = torch.tensor([[11, 12]], dtype=torch.int32)
        counts = torch.tensor([1], dtype=torch.int32)
        output = torch.zeros((1, 2), dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "reserved interval"):
            ORACLE_BY_ENTRY_POINT["cake_reservation_owned_store_b8_smoke"](
                (expert_ids, payloads, counts, output), torch
            )


if __name__ == "__main__":
    unittest.main()
