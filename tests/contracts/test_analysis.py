"""Contract tests for occupancy analysis.

This supplies the attribution half of the paper's `performance analysis` report: which
declared resource bounds residency. There is deliberately no cost estimate -- a Target
declares no clock and no bandwidth, so a predicted time would be invented.
"""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from open_cake_ir.compiler.analysis import occupancy, registers_per_thread
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.verifier import FindingSeverity, verify

ROOT = Path(__file__).resolve().parents[2]
TARGET = Target.load(ROOT / "compiler" / "targets" / "sm_100a.json")
B32 = ROOT / "corpus" / "schedules" / "flash-kmeans-b32-smoke-v2.json"
ASSIGNMENT_FULL = ROOT / "corpus" / "schedules" / "flash-kmeans-assignment-full.json"


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
        measured = occupancy(Schedule.load(ASSIGNMENT_FULL), TARGET)
        assert measured is not None and measured.binding is not None
        self.assertEqual(
            min(bound.ctas for bound in measured.bounds), measured.binding.ctas
        )
        self.assertEqual(measured.binding.resource, "shared_memory")
        self.assertEqual(measured.binding.ctas, 2)

    def test_registers_bind_the_triton_profile(self) -> None:
        """Declared register buffers, not shared memory, are what limits this one."""

        schedule = Schedule.load(B32)
        measured = occupancy(schedule, TARGET)
        assert measured is not None and measured.binding is not None
        self.assertEqual(measured.binding.resource, "registers")
        self.assertEqual(measured.binding.ctas, 1)
        self.assertEqual(registers_per_thread(schedule, TARGET), 288)

    def test_a_smaller_tile_relaxes_the_bound(self) -> None:
        """Halving the token tile halves the register buffers and doubles residency."""

        document = json.loads(B32.read_text(encoding="utf-8"))
        document = copy.deepcopy(document)
        for axis in document["program_map"]["axes"]:
            if axis["name"] == "token_block":
                axis["tile"] = 128
        for buffer in document["buffers"]:
            if buffer["name"] in ("token_tile", "distance_tile", "best_index_tile"):
                buffer["shape"][0] = 128
        measured = occupancy(Schedule.from_dict(document), TARGET)
        assert measured is not None and measured.binding is not None
        self.assertEqual(measured.binding.ctas, 2)

    def test_no_occupancy_facts_means_no_analysis(self) -> None:
        """A Target that declares nothing gets no invented answer."""

        document = json.loads(
            (ROOT / "compiler" / "targets" / "sm_100a.json").read_text(encoding="utf-8")
        )
        document.pop("occupancy")
        self.assertIsNone(occupancy(Schedule.load(B32), Target.from_dict(document)))


class ReportTest(unittest.TestCase):
    def _reports(self, path: Path) -> dict[str, str]:
        return {
            finding.code: finding.message
            for finding in verify(Schedule.load(path), TARGET)
            if finding.severity is FindingSeverity.REPORT
        }

    def test_every_retained_schedule_reports_its_bound(self) -> None:
        for path in sorted(
            ROOT / case["schedule"]
            for case in json.loads(
                (ROOT / "corpus" / "manifest.json").read_text(encoding="utf-8")
            )["cases"]
        ):
            with self.subTest(schedule=path.name):
                self.assertIn("RESIDENCY_BOUND", self._reports(path))

    def test_a_report_does_not_block(self) -> None:
        findings = verify(Schedule.load(B32), TARGET)
        reports = [f for f in findings if f.severity is FindingSeverity.REPORT]
        self.assertTrue(reports)
        self.assertFalse(any(f.blocks_lowering for f in reports))

    def test_register_pressure_is_reported_where_it_binds(self) -> None:
        self.assertIn("REGISTER_PRESSURE", self._reports(B32))
        self.assertIn("288 registers per thread", self._reports(B32)["REGISTER_PRESSURE"])
        self.assertNotIn("REGISTER_PRESSURE", self._reports(ASSIGNMENT_FULL))

    def test_the_report_names_the_runners_up(self) -> None:
        message = self._reports(ASSIGNMENT_FULL)["RESIDENCY_BOUND"]
        self.assertIn("shared_memory bounds residency to 2 CTA", message)
        self.assertIn("the next bounds are", message)


class NoCostEstimateTest(unittest.TestCase):
    def test_analysis_predicts_no_time(self) -> None:
        """The Target declares no clock and no bandwidth, so nothing here predicts one."""

        source = (
            ROOT / "src" / "open_cake_ir" / "compiler" / "analysis.py"
        ).read_text(encoding="utf-8")
        for absent in ("clock", "bandwidth", "seconds", "latency_ms", "flops"):
            with self.subTest(term=absent):
                self.assertNotIn(absent, source.lower().split('"""')[2].lower())


if __name__ == "__main__":
    unittest.main()
