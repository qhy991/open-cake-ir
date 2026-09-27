"""Synthetic paired-cost derivation: no broker, GPU or performance evidence."""
from __future__ import annotations

import copy
import json
import unittest
from hashlib import sha256
from pathlib import Path

from open_cake_ir.evaluation import LaunchableCandidate
from open_cake_ir.lab.executor import ExecutorRevision
from open_cake_ir.lab.paired_cost_calibration import derive_paired_cost_model
from open_cake_ir.serialization import canonical_json_bytes
from open_cake_ir.tasks.workloads import load_workload


ROOT = Path(__file__).resolve().parents[2]


class PairedCostFitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workload = load_workload(ROOT / "contracts/workloads/gemm-bias-bf16-fp32-v2.json")
        cls.protocol = json.loads((ROOT / "contracts/studies/matched-search-triton-b300-gemm-optimization-template.json").read_text())["evaluation_protocol"]
        cls.executor = ExecutorRevision("fixture", {"target": "sm_103a", "host_environment": {"packages": {"triton": "fixture"}}}, ROOT, "fixture")
        cls.baseline = LaunchableCandidate("1" * 64, "sm_103a", "baseline",
                                           {"cubin": "2" * 64}, "3" * 64)
        cls.base_schedule = json.loads((ROOT / "corpus/schedules/gemm-bias-b1-smoke-b300.json").read_text())
        cls.acceptance = {"maximum_baseline_drift_ratio": 1.05,
                          "maximum_mape": .10, "maximum_relative_error": .20,
                          "maximum_top2_regret_ratio": 1.05,
                          "envelope_allowance": .05}

    def rows(self):
        rows = []
        for index, cap in enumerate((64, 96, 128)):
            name = f"tile-{cap}"
            schedule = copy.deepcopy(self.base_schedule)
            schedule["schedule_id"] = name
            schedule["residency"]["registers_per_thread"] = cap
            for split, factor in (("fit", 1), ("calibration", 1.02), ("audit", 1.03)):
                rows.append({"candidate_id": name, "schedule": schedule, "split": split,
                             "observation": {
                                 "candidate_record_sha256": sha256(name.encode()).hexdigest(),
                                 "baseline_record_sha256": self.baseline.canonical_sha256,
                                 "workload_sha256": self.workload.canonical_sha256,
                                 "case_id": "primary",
                                 "evaluation_protocol_sha256": sha256(canonical_json_bytes(self.protocol)).hexdigest(),
                                 "job_id": f"gpuq-synthetic-{split}", "gpu_uuid": "GPU-SYNTHETIC",
                                 "candidate_us": (10 + index * 10) * factor,
                                 "baseline_us": 100,
                                 "sample_count": 250}})
        return rows

    def derive(self, rows):
        return derive_paired_cost_model(
            model_id="synthetic-paired-fit", compiler_revision_id="synthetic-compiler",
            target="sm_103a", executor=self.executor,
            workload_sha256=self.workload.canonical_sha256, case_id="primary",
            evaluation_protocol=self.protocol, baseline=self.baseline,
            rows=rows,
            varying_dimensions=[{"buffer": name, "dimension": 0} for name in ("a", "c")],
            acceptance=self.acceptance,
        )

    def test_exact_pool_fits_only_after_separate_splits_and_audits_all_ties(self):
        passed, model, audit = self.derive(self.rows())
        self.assertTrue(passed)
        self.assertTrue(audit["passed"])
        self.assertEqual(len(model["curves"]), 3)
        self.assertEqual(len(audit["regrets"]), 6)
        self.assertEqual(audit["metrics"]["max_top2_regret_ratio"], 1)
        self.assertEqual(audit["metrics"]["descriptive_range_covered_count"], 3)
        self.assertEqual(model["context"]["runtime"]["executor_revision"], "fixture")

    def test_heldout_fast_candidate_outside_top_two_fails_without_refit(self):
        rows = self.rows()
        next(row for row in rows if row["candidate_id"] == "tile-128" and row["split"] == "audit")["observation"]["candidate_us"] = 5
        passed, model, audit = self.derive(rows)
        self.assertFalse(passed)
        self.assertGreater(audit["metrics"]["max_top2_regret_ratio"], 1.05)
        fitted = {curve["template"]["schedule_id"]: curve["points"][0]["kernel_us"]
                  for curve in model["curves"]}
        self.assertEqual(fitted["tile-128"], 30)

    def test_equal_predictions_audit_every_provider_order(self):
        rows = self.rows()
        for row in rows:
            if row["split"] == "fit":
                row["observation"]["candidate_us"] = 10
            elif row["split"] == "calibration":
                row["observation"]["candidate_us"] = 10.2
            else:
                row["observation"]["candidate_us"] = 1 if row["candidate_id"] == "tile-64" else 10
        passed, _, audit = self.derive(rows)
        self.assertFalse(passed)
        self.assertEqual(len(audit["regrets"]), 6)
        self.assertEqual(audit["metrics"]["max_top2_regret_ratio"], 10)

    def test_baseline_drift_and_missing_split_are_refused(self):
        rows = self.rows()
        rows[0]["observation"]["baseline_us"] = 120
        with self.assertRaisesRegex(ValueError, "baseline drift"):
            self.derive(rows)
        rows = self.rows()
        rows = [row for row in rows if not (row["candidate_id"] == "tile-64" and row["split"] == "calibration")]
        with self.assertRaisesRegex(ValueError, "complete fit/calibration/audit"):
            self.derive(rows)


if __name__ == "__main__":
    unittest.main()
