from __future__ import annotations

import json
import unittest
from pathlib import Path

from open_cake_ir.compiler import Compiler


ROOT = Path(__file__).resolve().parents[2]


def vector_fma(width=128):
    value = json.loads((ROOT / "corpus/schedules/fma-b8-smoke.json").read_text())
    for buffer in value["buffers"]:
        buffer["shape"][-1] = width
    return value


class AccessValueShapeTests(unittest.TestCase):
    def setUp(self):
        self.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def decisive(self, schedule):
        assessment = self.compiler.assess(schedule)
        findings = [(f.code, f.path) for f in assessment.findings if f.blocks_lowering]
        return assessment, findings

    def test_scalar_read_cannot_masquerade_as_vector_fma_input(self):
        value = vector_fma()
        value["access_maps"][0]["indices"][1] = {"source": "program", "name": "batch"}
        assessment, findings = self.decisive(value)
        self.assertFalse(assessment.accepted)
        self.assertFalse(assessment.lowering_eligible)
        self.assertEqual(findings, [("LOAD_ACCESS_SHAPE_MISMATCH", "operations[0].writes[0]")])

    def test_load_rank_is_not_an_implicit_reshape(self):
        value = vector_fma()
        for buffer in value["buffers"]:
            if buffer["space"] == "register":
                buffer["shape"] = [1, 128]
        assessment, findings = self.decisive(value)
        self.assertFalse(assessment.accepted)
        self.assertEqual([code for code, _ in findings], ["LOAD_ACCESS_SHAPE_MISMATCH"] * 3)

    def test_bad_dimension_returns_a_finding_instead_of_raising(self):
        value = vector_fma()
        value["access_maps"][0]["indices"][1]["dimension"] = 99
        assessment, findings = self.decisive(value)
        self.assertFalse(assessment.accepted)
        self.assertEqual(findings, [("ACCESS_DIMENSION_MISMATCH", "access_maps[0].indices[1]")])

    def test_vector_is_bounded_by_the_dimension_it_actually_addresses(self):
        value = vector_fma(8)
        value["buffers"][0]["shape"][-1] = 4
        value["access_maps"][0]["indices"][1]["dimension"] = 0
        assessment, findings = self.decisive(value)
        self.assertFalse(assessment.accepted)
        self.assertEqual(findings, [("ACCESS_DIMENSION_COORDINATE_RANGE", "access_maps[0].indices[1]")])

    def test_scalar_loads_retain_the_single_value_register_form(self):
        value = vector_fma()
        for buffer in value["buffers"]:
            buffer["shape"] = [8] if buffer["space"] == "global" else [1]
        for access in value["access_maps"]:
            access["indices"] = [{"source": "program", "name": "batch"}]
        assessment, findings = self.decisive(value)
        self.assertTrue(assessment.accepted)
        self.assertTrue(assessment.lowering_eligible)
        self.assertEqual(findings, [])
        self.assertIn("fma.rn.f32", self.compiler.lower(assessment).source)

    def test_non_power_of_two_and_oversized_arange_are_backend_refusals(self):
        for width in (3, 1 << 21):
            with self.subTest(width=width):
                assessment, findings = self.decisive(vector_fma(width))
                self.assertTrue(assessment.accepted)
                self.assertFalse(assessment.lowering_eligible)
                self.assertEqual([code for code, _ in findings], ["TRITON_ARANGE_RANGE_UNSUPPORTED"] * 4)

    def test_program_tile_limit_does_not_reject_non_power_of_two_global_extent(self):
        for tile, allowed in ((3, False), (4, True)):
            with self.subTest(tile=tile):
                value = vector_fma(tile)
                for buffer in value["buffers"]:
                    if buffer["space"] == "global":
                        buffer["shape"][-1] = 9
                value["program_map"]["axes"].append(
                    {"name": "column", "axis": 1, "buffer": "a", "dimension": 1, "tile": tile}
                )
                for access in value["access_maps"]:
                    access["indices"][1] = {"source": "program_tile", "name": "column"}
                assessment, findings = self.decisive(value)
                self.assertTrue(assessment.accepted)
                self.assertEqual(assessment.lowering_eligible, allowed)
                self.assertEqual(findings, [] if allowed else [("TRITON_ARANGE_RANGE_UNSUPPORTED", "program_map.axes[1].tile")])
                if allowed:
                    self.compiler.lower(assessment)

    def test_nonzero_arange_start_uses_span_not_endpoint_power(self):
        value = vector_fma(4)
        value["buffers"][0]["shape"][-1] = 7
        value["access_maps"][0]["indices"][1]["offset"] = 3
        assessment, findings = self.decisive(value)
        self.assertTrue(assessment.accepted)
        self.assertTrue(assessment.lowering_eligible)
        self.assertEqual(findings, [])
        self.assertIn("tl.arange(3,", self.compiler.lower(assessment).source)

    def test_unknown_vector_references_stay_localized(self):
        for component, code in (
            ({"source": "program_tile", "name": "missing"}, "ACCESS_PROGRAM_AXIS_UNKNOWN"),
            ({"source": "loop_tile", "name": "missing"}, "ACCESS_LOOP_UNKNOWN"),
        ):
            with self.subTest(component=component):
                value = vector_fma()
                value["access_maps"][0]["indices"][1] = component
                assessment, findings = self.decisive(value)
                self.assertFalse(assessment.accepted)
                self.assertIn((code, "access_maps[0].indices[1]"), findings)

    def test_published_affine_survivors_stay_lowerable_and_bad_shape_is_refused(self):
        rows = [json.loads(line) for line in
                (ROOT / "docs/data/aka-fma-v41-reaudit-20260906/results.jsonl").read_text().splitlines()]
        for row in rows:
            if row["terminal_status"] not in {"candidate_lowered_compiled", "candidate_semantic_review_rejected"}:
                continue
            with self.subTest(case=row["case_id"]):
                assessment, findings = self.decisive(json.loads(row["schedule_json"]))
                if row["terminal_status"] == "candidate_lowered_compiled":
                    self.assertTrue(assessment.accepted)
                    self.assertTrue(assessment.lowering_eligible)
                    self.compiler.lower(assessment)
                else:
                    self.assertFalse(assessment.accepted)
                    self.assertIn("LOAD_ACCESS_SHAPE_MISMATCH", [code for code, _ in findings])


if __name__ == "__main__":
    unittest.main()
