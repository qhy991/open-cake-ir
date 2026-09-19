"""Contract tests for occupancy analysis.

This supplies the attribution half of the paper's `performance analysis` report: which
declared resource bounds residency. There is deliberately no cost estimate -- a Target
declares no clock and no bandwidth, so a predicted time would be invented.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import unittest
from pathlib import Path

from open_cake_ir.compiler import Compiler, CompilerError
from open_cake_ir.compiler.performance.residency import (
    logical_register_pressure_per_thread,
    residency_upper_bound,
    top_k_merge_structure,
)
from open_cake_ir.compiler.ir import OperationKind, Schedule
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.verifier import FindingSeverity, verify

from tests.contracts._corpus_documents import corpus_document

ROOT = Path(__file__).resolve().parents[2]
TARGET = Target.load(ROOT / "compiler" / "targets" / "sm_100a.json")
B32 = ROOT / "corpus" / "schedules" / "flash-kmeans-b32-smoke-v2.json"
ASSIGNMENT_FULL = ROOT / "corpus" / "schedules" / "flash-kmeans-assignment-full.json"
TOP_K = ROOT / "corpus" / "schedules" / "top-k-b8-smoke.json"
MERGE2 = ROOT / "corpus" / "schedules" / "top-k-streaming-merge2-b8-smoke.json"
QSA_SCORE = ROOT / "corpus" / "schedules" / "qsa-score-topk-t32768.json"


def _merge2_qsa(tile: int = 128) -> Schedule:
    document = json.loads(QSA_SCORE.read_text(encoding="utf-8"))
    document["schedule_id"] = f"qsa-score-topk-t32768-tile{tile}-merge2"
    loop = document["tile_loops"][0]
    loop["tile"] = tile
    operation = next(
        item for item in document["operations"] if item["kind"] == "top_k"
    )
    operation["parameters"]["source_tiles_per_merge"] = 2
    if tile != 128:
        document["roles"][0]["execution_groups"] = [0, 1, 2, 3]
        buffers = {item["name"]: item for item in document["buffers"]}
        buffers["key_tile"]["shape"] = [tile, 128]
        for name in ("head_scores", "positive_scores"):
            buffers[name]["shape"] = [8, tile]
        for name in ("score_sum", "score_tile"):
            buffers[name]["shape"] = [tile]
        next(
            item for item in document["operations"] if item["id"] == "score_heads"
        )["parameters"]["tile_shape"] = [8, tile, 128]
    return Schedule.from_dict(document)


def _top_k(schedule: Schedule):
    return next(
        operation
        for operation in schedule.operations
        if operation.kind is OperationKind.TOP_K
    )


class ObservedFactsTest(unittest.TestCase):
    """The per-multiprocessor facts were read from a B200, not asserted."""

    def test_the_target_carries_a_device_observation(self) -> None:
        document = json.loads(
            (ROOT / "compiler" / "targets" / "sm_100a.json").read_text(encoding="utf-8")
        )
        kinds = {citation["kind"] for citation in document["citations"]}
        self.assertIn("device_observation", kinds)
        self.assertEqual(
            document["occupancy"],
            {
                "multiprocessor_count": 148,
                "registers_per_multiprocessor": 65536,
                "shared_memory_per_multiprocessor_bytes": 233472,
                "maximum_threads_per_multiprocessor": 2048,
            },
        )

    def test_the_cta_limits_agree_with_the_device(self) -> None:
        """Two limits the Target already declared are confirmed by the same read."""

        limits = TARGET.resource_limits
        self.assertEqual(limits.maximum_shared_memory_bytes, 232448)
        self.assertEqual(limits.maximum_threads_per_cta, 1024)


class ResidencyTest(unittest.TestCase):
    def test_the_binding_resource_is_the_smallest_bound(self) -> None:
        bound = residency_upper_bound(Schedule.load(ASSIGNMENT_FULL), TARGET)
        assert bound is not None and bound.binding is not None
        self.assertEqual(
            min(item.ctas for item in bound.bounds), bound.binding.ctas
        )
        self.assertEqual(bound.binding.resource, "shared_memory")
        self.assertEqual(bound.binding.ctas, 2)

    def test_logical_register_pressure_is_separate_from_exact_residency(self) -> None:
        """Logical Buffers remain visible without impersonating physical registers."""

        schedule = Schedule.load(B32)
        bound = residency_upper_bound(schedule, TARGET)
        assert bound is not None and bound.binding is not None
        self.assertEqual(bound.binding.resource, "threads")
        self.assertEqual(bound.binding.ctas, 16)
        self.assertEqual(logical_register_pressure_per_thread(schedule, TARGET), 289)

    def test_top_k_charges_source_values_and_indices_while_they_are_live(self) -> None:
        # At selection the 256 score values and both eight-element results coexist. The
        # The logical pressure proxy sees 272 values across 128 CTA threads.
        schedule = Schedule.load(TOP_K)
        self.assertEqual(logical_register_pressure_per_thread(schedule, TARGET), 3)

    def test_merge2_charges_the_pending_source_tile_and_reports_exact_cadence(self) -> None:
        schedule = Schedule.load(MERGE2)
        structure = top_k_merge_structure(schedule, _top_k(schedule))
        assert structure is not None

        self.assertEqual(structure.source_elements_per_merge, 16)
        self.assertEqual(structure.merge_width, 32)
        self.assertEqual(structure.loop_carried_state_bytes, 64)
        self.assertEqual(structure.pending_source_key_elements, 8)
        self.assertEqual(structure.pending_source_state_bytes, 64)
        self.assertEqual(structure.count_estimate_kind, "exact")
        self.assertEqual(structure.whole_grid_source_tile_update_count, 24)
        self.assertEqual(structure.whole_grid_full_group_merge_count, 8)
        self.assertEqual(structure.whole_grid_tail_flush_merge_count, 8)
        self.assertEqual(structure.whole_grid_merge_update_count, 16)

    def test_qsa_merge2_counts_dynamic_stop_and_pending_state(self) -> None:
        schedule = _merge2_qsa()
        structure = top_k_merge_structure(schedule, _top_k(schedule))
        assert structure is not None

        self.assertEqual(structure.source_extent, 128)
        self.assertEqual(structure.source_elements_per_merge, 256)
        self.assertEqual(structure.merge_width, 1024)
        self.assertEqual(structure.loop_carried_state_bytes, 4096)
        self.assertEqual(structure.pending_source_key_elements, 128)
        self.assertEqual(structure.pending_source_state_bytes, 1024)
        self.assertEqual(structure.whole_grid_source_tile_update_count, 1_064_768)
        self.assertEqual(structure.whole_grid_full_group_merge_count, 524_192)
        self.assertEqual(structure.whole_grid_tail_flush_merge_count, 16_384)
        self.assertEqual(structure.whole_grid_merge_update_count, 540_576)
        self.assertEqual(logical_register_pressure_per_thread(schedule, TARGET), 73)

    def test_external_tile256_frontier_has_a_separate_merge2_geometry(self) -> None:
        schedule = _merge2_qsa(256)
        structure = top_k_merge_structure(schedule, _top_k(schedule))
        assert structure is not None

        self.assertEqual(structure.source_extent, 256)
        self.assertEqual(structure.source_elements_per_merge, 512)
        self.assertEqual(structure.merge_width, 1024)
        self.assertEqual(structure.pending_source_key_elements, 256)
        self.assertEqual(structure.pending_source_state_bytes, 2048)
        self.assertEqual(structure.whole_grid_source_tile_update_count, 540_576)
        self.assertEqual(structure.whole_grid_full_group_merge_count, 262_096)
        self.assertEqual(structure.whole_grid_tail_flush_merge_count, 16_384)
        self.assertEqual(structure.whole_grid_merge_update_count, 278_480)
        self.assertEqual(logical_register_pressure_per_thread(schedule, TARGET), 284)

    def test_a_smaller_tile_reduces_pressure_without_changing_exact_bounds(self) -> None:

        document = json.loads(B32.read_text(encoding="utf-8"))
        document = copy.deepcopy(document)
        for axis in document["program_map"]["axes"]:
            if axis["name"] == "token_block":
                axis["tile"] = 128
        for buffer in document["buffers"]:
            if buffer["name"] in ("token_tile", "distance_tile", "best_index_tile"):
                buffer["shape"][0] = 128
        smaller = Schedule.from_dict(document)
        bound = residency_upper_bound(smaller, TARGET)
        assert bound is not None and bound.binding is not None
        self.assertEqual(bound.binding.ctas, 16)
        self.assertLess(
            logical_register_pressure_per_thread(smaller, TARGET),
            logical_register_pressure_per_thread(Schedule.load(B32), TARGET),
        )

    def test_no_occupancy_facts_means_no_analysis(self) -> None:
        """A Target that declares nothing gets no invented answer."""

        document = json.loads(
            (ROOT / "compiler" / "targets" / "sm_100a.json").read_text(encoding="utf-8")
        )
        document.pop("occupancy")
        self.assertIsNone(
            residency_upper_bound(Schedule.load(B32), Target.from_dict(document))
        )

    def test_an_exact_shared_memory_bound_still_enforces_a_cta_commitment(self) -> None:
        document = json.loads(ASSIGNMENT_FULL.read_text(encoding="utf-8"))
        document["residency"] = {"ctas_per_multiprocessor": 3}
        blocking = {
            finding.code
            for finding in verify(Schedule.from_dict(document), TARGET)
            if finding.blocks_lowering
        }
        self.assertIn("RESIDENCY_UNMET", blocking)


class ReportTest(unittest.TestCase):
    def _reports(self, path: Path, target: Target = TARGET) -> dict[str, str]:
        return {
            finding.code: finding.message
            for finding in verify(Schedule.from_dict(corpus_document(path)), target)
            if finding.severity is FindingSeverity.REPORT
        }

    def test_every_retained_schedule_reports_its_bound(self) -> None:
        targets = {path.stem: path for path in (ROOT / "compiler/targets").glob("*.json")}
        cases = json.loads((ROOT / "corpus" / "manifest.json").read_text(encoding="utf-8"))["cases"]
        compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        for case in sorted(cases, key=lambda case: case["schedule"]):
            path = ROOT / case["schedule"]
            with self.subTest(schedule=path.name):
                document = corpus_document(path)
                if "SCHEDULE_STRUCTURE" in case["expected"]["finding_codes"]:
                    # Intentional structural counterexamples have no Schedule to
                    # analyze. Keep their declared public refusal under assertion.
                    assessment = compiler.assess(document)
                    self.assertFalse(assessment.accepted)
                    self.assertFalse(assessment.lowering_eligible)
                    self.assertEqual([f.code for f in assessment.findings], case["expected"]["finding_codes"])
                    continue
                schedule = Schedule.from_dict(document)
                reference = targets.get(schedule.target)
                if reference is None:
                    compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
                    assessment = compiler.assess_file(path)
                    codes = [finding.code for finding in assessment.findings]
                    self.assertFalse(assessment.accepted)
                    self.assertFalse(assessment.lowering_eligible)
                    self.assertIn("TARGET_UNSUPPORTED", codes)
                    self.assertNotIn("RESIDENCY_BOUND", codes)
                    with self.assertRaises(CompilerError):
                        compiler.lower(assessment)
                    continue
                target = Target.load(reference)
                self.assertEqual(schedule.target, target.target_id)
                reports = self._reports(path, target)
                if target.occupancy is None:
                    self.assertIsNone(residency_upper_bound(schedule, target))
                    self.assertNotIn("RESIDENCY_BOUND", reports)
                else:
                    self.assertIn("RESIDENCY_BOUND", reports)

    def test_a_report_does_not_block(self) -> None:
        findings = verify(Schedule.load(B32), TARGET)
        reports = [f for f in findings if f.severity is FindingSeverity.REPORT]
        self.assertTrue(reports)
        self.assertFalse(any(f.blocks_lowering for f in reports))

    def test_register_pressure_is_reported_as_a_nonblocking_proxy(self) -> None:
        self.assertIn("REGISTER_PRESSURE", self._reports(B32))
        message = self._reports(B32)["REGISTER_PRESSURE"]
        self.assertIn("logical pressure 289 per CTA thread", message)
        self.assertIn("not a bound or gate", message)
        self.assertIn(
            "physical register allocation is backend evidence",
            self._reports(B32)["RESIDENCY_BOUND"],
        )
        self.assertNotIn("REGISTER_PRESSURE", self._reports(ASSIGNMENT_FULL))

    def test_the_report_names_the_runners_up(self) -> None:
        message = self._reports(ASSIGNMENT_FULL)["RESIDENCY_BOUND"]
        self.assertIn("shared_memory bounds maximum possible residency to 2 CTA", message)
        self.assertIn("the next bounds are", message)


class NoCostEstimateTest(unittest.TestCase):
    def test_analysis_predicts_no_time(self) -> None:
        """The Target declares no clock and no bandwidth, so nothing here predicts one."""

        source = (
            ROOT / "src" / "open_cake_ir" / "compiler" / "performance" / "residency.py"
        ).read_text(encoding="utf-8")
        for absent in ("clock", "bandwidth", "seconds", "latency_ms", "flops"):
            with self.subTest(term=absent):
                self.assertNotIn(absent, source.lower().split('"""')[2].lower())


if __name__ == "__main__":
    unittest.main()


