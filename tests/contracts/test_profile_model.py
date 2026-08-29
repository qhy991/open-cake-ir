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

    def _profile_document(self, document: dict[str, object]) -> dict[str, object]:
        assessment = self.compiler.assess(document)
        self.assertTrue(assessment.lowering_eligible)
        return profile_envelope(
            Schedule.from_dict(document),
            self.target,
            lowered_source=self.compiler.lower(assessment).source,
        ).as_dict()

    def test_qsa_top_k_reports_bounds_risks_and_explicit_abstentions(self) -> None:
        profile = self._profile("qsa-score-topk-t32768.json")

        self.assertEqual(profile["lowering"]["generated_source_bytes"], 7310)
        top_k = profile["lowering"]["top_k"][0]
        self.assertEqual(top_k["source_tiles_per_merge"], 1)
        self.assertEqual(top_k["merge_width"], 1024)
        self.assertEqual(top_k["pending_source_state_bytes"], 0)
        self.assertEqual(top_k["structural_count_estimate_kind"], "exact")
        self.assertEqual(top_k["whole_grid_source_tile_update_count"], 1_064_768)
        self.assertEqual(top_k["whole_grid_merge_update_count"], 1_064_768)
        self.assertEqual(top_k["whole_grid_tail_flush_merge_count"], 0)
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

    def test_merge2_reports_pending_state_exact_cadence_and_unchanged_qsa_width(self) -> None:
        document = json.loads(
            (ROOT / "corpus/schedules/qsa-score-topk-t32768.json").read_text(
                encoding="utf-8"
            )
        )
        document["schedule_id"] = "qsa-score-topk-t32768-merge2-profile"
        top_k_operation = next(
            item for item in document["operations"] if item["kind"] == "top_k"
        )
        top_k_operation["parameters"]["source_tiles_per_merge"] = 2

        profile = self._profile_document(document)
        top_k = profile["lowering"]["top_k"][0]

        self.assertEqual(top_k["source_elements_per_merge"], 256)
        self.assertEqual(top_k["merge_width"], 1024)
        self.assertEqual(top_k["loop_carried_state_bytes"], 4096)
        self.assertEqual(top_k["pending_source_key_elements"], 128)
        self.assertEqual(top_k["pending_source_state_bytes"], 1024)
        self.assertEqual(top_k["structural_count_estimate_kind"], "exact")
        self.assertEqual(top_k["whole_grid_source_tile_update_count"], 1_064_768)
        self.assertEqual(top_k["whole_grid_full_group_merge_count"], 524_192)
        self.assertEqual(top_k["whole_grid_tail_flush_merge_count"], 16_384)
        self.assertEqual(top_k["whole_grid_merge_update_count"], 540_576)
        self.assertEqual(
            profile["residency"]["logical_register_pressure_per_thread"], 73
        )
        barrier = _metric(
            profile,
            "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
        )
        self.assertEqual(barrier["value"], "high")
        self.assertEqual(
            barrier["reasons"],
            [
                "loop-carried top_k k=512 source_tiles_per_merge=2 "
                "merge_width=1024 pending_source_state_bytes=1024"
            ],
        )

    def test_merge2_structural_count_abstains_on_an_out_of_domain_stop(self) -> None:
        document = json.loads(
            (ROOT / "corpus/schedules/qsa-score-topk-t32768.json").read_text(
                encoding="utf-8"
            )
        )
        next(item for item in document["operations"] if item["kind"] == "top_k")[
            "parameters"
        ]["source_tiles_per_merge"] = 2
        document["tile_loops"][0]["stop"]["add"] = 40_000
        profile = profile_envelope(
            Schedule.from_dict(document), self.target
        ).as_dict()
        top_k = profile["lowering"]["top_k"][0]

        self.assertEqual(top_k["structural_count_estimate_kind"], "unknown")
        self.assertIsNone(top_k["whole_grid_source_tile_update_count"])
        self.assertIsNone(top_k["whole_grid_merge_update_count"])
        self.assertEqual(
            top_k["structural_count_missing"],
            ["query-derived loop stop leaves the declared loop extent"],
        )

    def test_generic_three_trip_merge2_reports_one_pair_and_one_tail_per_program(self) -> None:
        profile = self._profile("top-k-streaming-merge2-b8-smoke.json")
        top_k = profile["lowering"]["top_k"][0]

        self.assertEqual(top_k["merge_width"], 32)
        self.assertEqual(top_k["pending_source_state_bytes"], 64)
        self.assertEqual(top_k["whole_grid_source_tile_update_count"], 24)
        self.assertEqual(top_k["whole_grid_full_group_merge_count"], 8)
        self.assertEqual(top_k["whole_grid_tail_flush_merge_count"], 8)
        self.assertEqual(top_k["whole_grid_merge_update_count"], 16)

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
