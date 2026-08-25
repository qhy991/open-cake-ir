"""Contract evidence for KDA-derived runtime-indexed global loads."""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

import jsonschema

from open_cake_ir.compiler.emit_triton import emit
from open_cake_ir.compiler.ir import AccessIndexKind, OperationKind, Schedule
from open_cake_ir.compiler.schema import schedule_schema
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.verifier import verify

ROOT = Path(__file__).resolve().parents[2]
POSITIVE = ROOT / "corpus" / "schedules" / "indexed-gather-b8-smoke.json"
DRIFT = (
    ROOT / "corpus" / "schedules" / "indexed-gather-b8-smoke-dtype-drift.json"
)
TARGET = Target.load(ROOT / "compiler" / "targets" / "sm_100a.json")


def _document(path: Path = POSITIVE) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _codes(document: dict) -> set[str]:
    return {finding.code for finding in verify(Schedule.from_dict(document), TARGET)}


def _buffer(document: dict, name: str) -> dict:
    return next(item for item in document["buffers"] if item["name"] == name)


def _operation(document: dict, op_id: str) -> dict:
    return next(item for item in document["operations"] if item["id"] == op_id)


def _access(document: dict, op_id: str) -> dict:
    return next(item for item in document["access_maps"] if item["operation"] == op_id)


class RuntimeIndexedAccessContractTest(unittest.TestCase):
    def test_schema_and_ir_have_one_buffer_coordinate_spelling(self) -> None:
        document = _document()
        jsonschema.Draft202012Validator(schedule_schema()).validate(document)
        access = Schedule.from_dict(document).access_map(
            "load_selected_rows", "expert_rows"
        )
        self.assertEqual(
            [component.source for component in access.indices],
            [
                AccessIndexKind.BUFFER,
                AccessIndexKind.BUFFER,
                AccessIndexKind.DIMENSION,
            ],
        )
        self.assertNotIn("gather", {kind.value for kind in OperationKind})

    def test_dtype_drift_is_the_corpus_falsifier(self) -> None:
        self.assertIn("ACCESS_INDEX_BUFFER_DTYPE", _codes(_document(DRIFT)))

    def test_indexed_load_dtype_has_one_access_owned_finding(self) -> None:
        document = _document()
        _buffer(document, "gathered_tile")["dtype"] = "fp32"
        dtype_findings = [
            finding.code
            for finding in verify(Schedule.from_dict(document), TARGET)
            if "DTYPE" in finding.code
        ]
        self.assertEqual(dtype_findings, ["LOAD_DTYPE_MISMATCH"])

    def test_unknown_runtime_index_can_fail(self) -> None:
        document = _document()
        _access(document, "load_selected_rows")["indices"][1]["name"] = "ghost"
        self.assertIn("ACCESS_INDEX_BUFFER_UNKNOWN", _codes(document))

    def test_runtime_index_must_be_register_resident(self) -> None:
        document = _document()
        index = _buffer(document, "row_id_tile")
        index.update(space="global", mode="output")
        self.assertIn("ACCESS_INDEX_BUFFER_SPACE", _codes(document))

    def test_runtime_indices_are_explicit_operation_reads(self) -> None:
        document = _document()
        _operation(document, "load_selected_rows")["reads"].pop()
        self.assertIn("ACCESS_INDEX_BUFFER_READS", _codes(document))

    def test_runtime_indices_share_one_zipped_domain(self) -> None:
        document = _document()
        _buffer(document, "row_ids")["shape"] = [8, 4]
        _buffer(document, "row_id_tile")["shape"] = [4]
        self.assertIn("ACCESS_INDEX_DOMAIN_MISMATCH", _codes(document))

    def test_indexed_store_is_not_silently_given_conflict_semantics(self) -> None:
        document = _document()
        access = _access(document, "store_gathered_rows")
        access["indices"][:2] = [
            {"source": "buffer", "name": "expert_id_tile"},
            {"source": "buffer", "name": "row_id_tile"},
        ]
        self.assertIn("ACCESS_INDEXED_OPERATION_UNLOWERABLE", _codes(document))

    def test_tma_does_not_inherit_direct_load_semantics(self) -> None:
        document = _document()
        _operation(document, "load_selected_rows")["parameters"] = {
            "movement": "tma",
            "descriptor_box": [8, 16],
        }
        self.assertIn("ACCESS_INDEXED_MOVEMENT_UNLOWERABLE", _codes(document))

    def test_emission_zips_two_indices_and_masks_the_kda_sentinel(self) -> None:
        source = emit(Schedule.load(POSITIVE), TARGET).source
        ast.parse(source)
        self.assertIn(
            "expert_id_tile[:, None] * D_EXPERT_ROWS_1 * D_EXPERT_ROWS_2",
            source,
        )
        self.assertIn("row_id_tile[:, None] * D_EXPERT_ROWS_2", source)
        self.assertIn("expert_rows_d2_offsets[None, :]", source)
        self.assertNotIn("expert_id_tile[:, None, None]", source)
        self.assertIn("(expert_id_tile[:, None] >= 0)", source)
        self.assertIn("(row_id_tile[:, None] < D_EXPERT_ROWS_1)", source)


if __name__ == "__main__":
    unittest.main()
