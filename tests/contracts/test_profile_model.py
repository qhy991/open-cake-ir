from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import (  # noqa: E402
    Compiler,
    MetricEstimate,
    Schedule,
    Target,
    profile_envelope,
)


def _metric(document: dict[str, object], name: str) -> dict[str, object]:
    return next(item for item in document["ncu_metrics"] if item["metric"] == name)


class ProfileModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        cls.target = Target.load(ROOT / "compiler/targets/sm_100a.json")

    def _profile(self, name: str) -> dict[str, object]:
        path = ROOT / "corpus/schedules" / name
        assessment = self.compiler.assess_file(path)
        self.assertTrue(assessment.lowering_eligible)
        return profile_envelope(
            Schedule.load(path),
            self.target,
            lowered_source=self.compiler.lower(assessment).source,
        ).as_dict()

    def test_qsa_top_k_reports_bounds_risks_and_explicit_abstentions(self) -> None:
        profile = self._profile("qsa-score-topk-t32768.json")

        self.assertEqual(profile["lowering"]["generated_source_bytes"], 7310)
        self.assertEqual(profile["lowering"]["top_k"][0]["merge_width"], 1024)
        self.assertEqual(profile["residency"]["logical_register_pressure_per_thread"], 72)
        self.assertEqual(profile["residency"]["ctas_per_sm_upper_bound"], 8)
        registers = _metric(profile, "launch__registers_per_thread")
        self.assertEqual((registers["estimate_kind"], registers["value"]), ("unknown", None))
        self.assertEqual(registers["reasons"], ["logical register pressure proxy=72"])
        active = _metric(
            profile, "sm__warps_active.avg.pct_of_peak_sustained_elapsed"
        )
        self.assertEqual((active["estimate_kind"], active["value"]), ("upper_bound", 100.0))
        barrier = _metric(
            profile,
            "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
        )
        self.assertEqual((barrier["estimate_kind"], barrier["value"]), ("uncalibrated_risk", "high"))
        sm = _metric(profile, "sm__throughput.avg.pct_of_peak_sustained_elapsed")
        self.assertEqual((sm["estimate_kind"], sm["value"]), ("unknown", None))
        self.assertIn("B200 NCU calibration", barrier["missing"])

    def test_runtime_gather_is_a_high_scoreboard_risk_not_a_fake_percentage(self) -> None:
        profile = self._profile("qsa-selected-attention-t32768.json")

        scoreboard = _metric(
            profile,
            "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
        )
        self.assertEqual(scoreboard["estimate_kind"], "uncalibrated_risk")
        self.assertEqual(scoreboard["value"], "high")
        self.assertTrue(profile["lowering"]["runtime_indexed_buffers"])
        self.assertIn("cache behavior", scoreboard["missing"])

    def test_shared_memory_limit_is_unknown_when_only_the_backend_allocates_it(self) -> None:
        profile = self._profile("qsa-score-topk-t32768.json")
        shared = _metric(profile, "launch__occupancy_limit_shared_mem")

        self.assertEqual((shared["estimate_kind"], shared["value"]), ("unknown", None))
        self.assertIn("backend implicit shared memory", shared["missing"])

    def test_resident_top_k_is_a_risk_hint_not_a_zero_stall_prediction(self) -> None:
        profile = self._profile("top-k-b8-smoke.json")
        barrier = _metric(
            profile,
            "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
        )

        self.assertEqual(barrier["estimate_kind"], "uncalibrated_risk")
        self.assertEqual(barrier["value"], "medium")
        self.assertEqual(barrier["reasons"], ["resident top_k k=8"])

    def test_metric_kind_prevents_a_numeric_value_from_masquerading_as_unknown(self) -> None:
        with self.assertRaises(ValueError):
            MetricEstimate("metric", "unknown", 12.0, "%", "none")

    def test_cli_emits_the_same_machine_readable_profile(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/report_schedule_profile.py"),
                "--revision",
                str(ROOT / "compiler/revision.json"),
                "--json",
                str(ROOT / "corpus/schedules/qsa-score-topk-t32768.json"),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        document = json.loads(completed.stdout)

        self.assertEqual(len(document["rows"]), 1)
        self.assertEqual(document["rows"][0]["profile"]["kind"], "ncu_aligned_profile_envelope")
        self.assertEqual(document["skipped"], [])

    def test_cli_accepts_an_external_agent_candidate_schedule(self) -> None:
        source = ROOT / "corpus/schedules/qsa-score-topk-t32768.json"
        with tempfile.TemporaryDirectory() as directory:
            external = Path(directory) / "candidate.json"
            external.write_bytes(source.read_bytes())
            expected = str(external.resolve(strict=True))
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/report_schedule_profile.py"),
                    "--revision",
                    str(ROOT / "compiler/revision.json"),
                    "--json",
                    str(external),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
            )
            document = json.loads(completed.stdout)

        self.assertEqual(document["rows"][0]["schedule"], expected)


if __name__ == "__main__":
    unittest.main()
