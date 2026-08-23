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

from open_cake_ir.compiler.analysis import (
    logical_registers_per_thread_lower_bound,
    residency_upper_bound,
)
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
        bound = residency_upper_bound(Schedule.load(ASSIGNMENT_FULL), TARGET)
        assert bound is not None and bound.binding is not None
        self.assertEqual(
            min(item.ctas for item in bound.bounds), bound.binding.ctas
        )
        self.assertEqual(bound.binding.resource, "shared_memory")
        self.assertEqual(bound.binding.ctas, 2)

    def test_registers_bind_the_triton_profile(self) -> None:
        """Declared register buffers, not shared memory, are what limits this one."""

        schedule = Schedule.load(B32)
        bound = residency_upper_bound(schedule, TARGET)
        assert bound is not None and bound.binding is not None
        self.assertEqual(bound.binding.resource, "logical_register_storage")
        self.assertEqual(bound.binding.ctas, 1)
        self.assertEqual(logical_registers_per_thread_lower_bound(schedule, TARGET), 289)

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
        bound = residency_upper_bound(Schedule.from_dict(document), TARGET)
        assert bound is not None and bound.binding is not None
        self.assertEqual(bound.binding.ctas, 2)

    def test_no_occupancy_facts_means_no_analysis(self) -> None:
        """A Target that declares nothing gets no invented answer."""

        document = json.loads(
            (ROOT / "compiler" / "targets" / "sm_100a.json").read_text(encoding="utf-8")
        )
        document.pop("occupancy")
        self.assertIsNone(
            residency_upper_bound(Schedule.load(B32), Target.from_dict(document))
        )


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
        message = self._reports(B32)["REGISTER_PRESSURE"]
        self.assertIn("optimistic lower bound of 289 registers per thread", message)
        self.assertIn("not ptxas-measured allocation", message)
        self.assertIn(
            "36928 of 65536 registers", self._reports(B32)["RESIDENCY_BOUND"]
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
            ROOT / "src" / "open_cake_ir" / "compiler" / "analysis.py"
        ).read_text(encoding="utf-8")
        for absent in ("clock", "bandwidth", "seconds", "latency_ms", "flops"):
            with self.subTest(term=absent):
                self.assertNotIn(absent, source.lower().split('"""')[2].lower())


if __name__ == "__main__":
    unittest.main()


class RankingTest(unittest.TestCase):
    """Candidate ranking: a total order, advisory, and never a latency.

    Measured on a B200 across nine tilings, twenty-nine of thirty-six pairs came out in the
    predicted order and the true best survived a cut at k=2. It filters; it does not choose.
    """

    def _variants(self):
        from open_cake_ir.compiler.ir import Schedule

        base = json.loads(
            (ROOT / "corpus/schedules/flash-kmeans-b32-smoke-v2.json").read_text(
                encoding="utf-8"
            )
        )
        for block_n, block_k in ((64, 64), (128, 64), (256, 64)):
            document = json.loads(json.dumps(base))
            buffers = {item["name"]: item for item in document["buffers"]}
            document["schedule_id"] = f"bn{block_n}-bk{block_k}"
            for axis in document["program_map"]["axes"]:
                if axis["name"] == "token_block":
                    axis["tile"] = block_n
            document["tile_loops"][0]["tile"] = block_k
            buffers["token_tile"]["shape"] = [block_n, 128]
            buffers["centroid_tile"]["shape"] = [block_k, 128]
            buffers["best_index_tile"]["shape"] = [block_n]
            buffers["norm_tile"]["shape"] = [block_k]
            for name in ("distance_tile", "cross", "scaled_cross"):
                buffers[name]["shape"] = [block_n, block_k]
            for operation in document["operations"]:
                if operation["kind"] == "mma" and "tile_shape" in operation["parameters"]:
                    operation["parameters"]["tile_shape"] = [block_n, block_k, 128]
            yield Schedule.from_dict(document)

    def test_the_order_is_total_and_independent_of_input_order(self) -> None:
        from open_cake_ir.compiler import ranking

        candidates = list(self._variants())
        forward, _ = ranking.rank(candidates, TARGET)
        backward, _ = ranking.rank(list(reversed(candidates)), TARGET)
        self.assertEqual(
            [c.schedule_id for c in forward], [c.schedule_id for c in backward]
        )
        # A ranking that depends on the order it was handed cannot be evidence.
        self.assertEqual(len(forward), len(candidates))

    def test_a_cost_carries_no_predicted_time(self) -> None:
        from open_cake_ir.compiler import ranking

        cost = ranking.cost(next(iter(self._variants())), TARGET)
        assert cost is not None
        fields = {f.name for f in dataclasses.fields(cost)}
        # A Target declares no clock and no bandwidth, so any latency here would be
        # invented rather than derived.
        self.assertFalse({f for f in fields if "time" in f or "latency" in f or "cycle" in f})

    def test_an_unscorable_candidate_is_named_not_dropped(self) -> None:
        from open_cake_ir.compiler import ranking
        from open_cake_ir.compiler.target import Target

        document = json.loads(
            (ROOT / "compiler/targets/sm_100a.json").read_text(encoding="utf-8")
        )
        document.pop("occupancy", None)
        blind = Target.from_dict(document)
        scored, unscored = ranking.rank(list(self._variants()), blind)
        self.assertEqual(scored, ())
        # Losing them silently would report a complete order over an incomplete set.
        self.assertEqual(len(unscored), 3)
