from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from kernel_oracles import ORACLE_BY_ENTRY_POINT  # noqa: E402
from observe_lowered_kernel import measure_correctness  # noqa: E402


class ObservationInstrumentTest(unittest.TestCase):
    def test_reservation_owned_store_attempt_retains_the_pre_gpu_failure(self) -> None:
        plan = json.loads(
            (
                ROOT
                / "inventory"
                / "V28_RESERVATION_OWNED_STORE_B200_PLAN_20260825.json"
            ).read_text()
        )
        attempt = json.loads(
            (
                ROOT
                / "inventory"
                / "V28_RESERVATION_OWNED_STORE_B200_ATTEMPT_20260825.json"
            ).read_text()
        )

        self.assertEqual(plan["state"], "frozen")
        self.assertEqual(plan["protocol"]["automatic_retries"], 0)
        self.assertEqual(attempt["state"], "failed")
        self.assertEqual(attempt["stage_reached"], "broker_worker_cwd_admission")
        self.assertFalse(attempt["compiled"])
        self.assertFalse(attempt["launched"])
        self.assertFalse(attempt["result_written"])
        self.assertFalse(attempt["retry"]["same_plan_retried"])
        self.assertFalse(attempt["performance_measured"])

    def test_atomic_reservation_observation_retains_its_contention_claim(self) -> None:
        plan = json.loads(
            (
                ROOT
                / "inventory"
                / "V27_ATOMIC_RESERVATION_B200_PLAN_20260825.json"
            ).read_text()
        )
        record = json.loads(
            (
                ROOT
                / "inventory"
                / "ATOMIC_RESERVATION_B200_OBSERVATION_20260825.json"
            ).read_text()
        )

        self.assertEqual(plan["state"], "frozen")
        self.assertEqual(plan["protocol"]["automatic_retries"], 0)
        self.assertEqual(
            record["compiler_revision"]["revision_id"],
            plan["selection_authority"]["compiler_revision"],
        )
        self.assertEqual(record["device"], "NVIDIA B200")
        self.assertEqual(
            record["lowering"]["entry_point"],
            plan["selection_authority"]["entry_point"],
        )
        self.assertEqual(
            record["result"],
            {
                "compiled": True,
                "launched": True,
                "total_elements": 64,
                "mismatch_count": 0,
                "unique_old_values": True,
                "masked_zero": True,
                "final_counts_match": True,
                "passed": True,
            },
        )
        self.assertFalse(record["performance_measured"])
        self.assertFalse(record["scientific_claim_authorized"])

    def test_kda_weighted_combine_b200_observation_retains_its_claim_boundary(self) -> None:
        record = json.loads(
            (
                ROOT
                / "inventory"
                / "KDA_WEIGHTED_COMBINE_B200_OBSERVATION_20260825.json"
            ).read_text()
        )

        self.assertEqual(
            record["compiler_revision"]["revision_id"],
            "open-cake-ir-sm100a-v26",
        )
        self.assertEqual(record["device"], "NVIDIA B200")
        self.assertTrue(record["lowering"]["generated"])
        self.assertEqual(
            record["lowering"]["entry_point"],
            "cake_kda_weighted_combine_b8_smoke",
        )
        self.assertEqual(
            record["result"],
            {
                "compiled": True,
                "launched": True,
                "total_elements": 128,
                "mismatch_count": 0,
                "max_deviation": 0.0,
                "tolerance": 1e-5,
                "passed": True,
            },
        )
        self.assertFalse(record["performance_measured"])
        self.assertFalse(record["scientific_claim_authorized"])

    def test_tie_audit_flattens_every_batch_row_without_losing_distance_axis(self) -> None:
        distance = torch.zeros((2, 3, 4), dtype=torch.float32)
        reference = torch.zeros((2, 3), dtype=torch.int32)
        observed = torch.ones((2, 3), dtype=torch.int32)

        mismatch, measured, passed = measure_correctness(
            observed, reference, distance, torch, 1e-5
        )

        self.assertEqual(mismatch, 6)
        self.assertEqual(measured, {"max_chosen_distance_excess": 0.0})
        self.assertTrue(passed)

    def test_gemm_oracle_zero_extends_a_masked_short_bias(self) -> None:
        a = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
        b = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.bfloat16
        )
        bias = torch.tensor([10.0, 20.0], dtype=torch.float32)
        output = torch.empty((1, 3), dtype=torch.float32)

        observed, distance = ORACLE_BY_ENTRY_POINT["cake_gemm_bias_b1_smoke"](
            (a, b, bias, output), torch
        )

        self.assertIsNone(distance)
        torch.testing.assert_close(observed, torch.tensor([[11.0, 22.0, 3.0]]))

    def test_kda_combine_oracle_masks_sentinel_then_weights_and_sums(self) -> None:
        expert_rows = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            dtype=torch.bfloat16,
        )
        expert_ids = torch.tensor([[0, 1, -1]], dtype=torch.int32)
        row_ids = torch.tensor([[1, 0, 1]], dtype=torch.int32)
        weights = torch.tensor([[0.5, 2.0, 10.0]], dtype=torch.float32)
        output = torch.empty((1, 2), dtype=torch.bfloat16)

        observed, distance = ORACLE_BY_ENTRY_POINT[
            "cake_kda_weighted_combine_b8_smoke"
        ]((expert_rows, expert_ids, row_ids, weights, output), torch)

        self.assertIsNone(distance)
        torch.testing.assert_close(
            observed, torch.tensor([[11.5, 14.0]], dtype=torch.bfloat16)
        )


class V25SuccessorObservationAttemptTest(unittest.TestCase):
    def test_frozen_attempt_projects_its_settled_domain_and_disposition(self) -> None:
        attempt = json.loads(
            (ROOT / "inventory" / "V25_B200_CORRECTNESS_ATTEMPT_20260825.json").read_text()
        )
        plan_path = ROOT / attempt["plan"]["path"]
        plan = json.loads(plan_path.read_text())
        case_ids = plan["generated_external_oracle"]["case_ids"]
        self.assertEqual(len(case_ids), 14)
        self.assertEqual(
            [Path(item["path"]).stem for item in attempt["recorded_results"]],
            case_ids,
        )

        passed: list[str] = []
        failed: list[str] = []
        for authority in attempt["recorded_results"]:
            path = ROOT / authority["path"]
            record = json.loads(path.read_text())
            self.assertEqual(
                record["compiler_revision"]["revision_id"],
                plan["selection_authority"]["compiler_revision_id"],
            )
            self.assertTrue(record["lowering"]["generated"])
            self.assertTrue(record["result"]["compiled"])
            self.assertTrue(record["result"]["launched"])
            self.assertFalse(record["scientific_claim_authorized"])
            self.assertFalse(record["performance_measured"])
            (passed if record["result"]["passed"] else failed).append(path.stem)

        self.assertEqual(len(passed), 12)
        self.assertEqual(
            failed, ["gemm-bias-accepted", "gemm-bias-shape-drift"]
        )
        for case_id in failed:
            record = json.loads(
                (
                    ROOT
                    / "inventory"
                    / "V25_B200_CORRECTNESS_20260825"
                    / f"{case_id}.json"
                ).read_text()
            )
            self.assertEqual(record["result"]["tolerance"], 1e-5)
            self.assertEqual(record["result"]["max_deviation"], 3.814697265625e-05)
        self.assertEqual(attempt["unrecorded_failure_count"], 0)
        self.assertFalse(attempt["disposition"]["generated_partition_passed"])
        self.assertEqual(attempt["disposition"]["checked_asset_partition"], "missing")


if __name__ == "__main__":
    unittest.main()
