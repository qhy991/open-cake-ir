from __future__ import annotations

import json
import sys
import unittest
from hashlib import sha256
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from kernel_oracles import ORACLE_BY_ENTRY_POINT  # noqa: E402
from observe_lowered_kernel import measure_correctness  # noqa: E402


class ObservationInstrumentTest(unittest.TestCase):
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


class V25SuccessorObservationAttemptTest(unittest.TestCase):
    def test_frozen_domain_and_all_broker_records_remain_bound(self) -> None:
        attempt = json.loads(
            (ROOT / "inventory" / "V25_B200_CORRECTNESS_ATTEMPT_20260825.json").read_text()
        )
        plan_path = ROOT / attempt["plan"]["path"]
        self.assertEqual(
            sha256(plan_path.read_bytes()).hexdigest(), attempt["plan"]["raw_sha256"]
        )
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
            self.assertEqual(
                sha256(path.read_bytes()).hexdigest(), authority["raw_sha256"]
            )
            record = json.loads(path.read_text())
            self.assertEqual(
                record["compiler_revision"],
                {
                    "revision_id": plan["selection_authority"]["compiler_revision_id"],
                    "revision_sha256": plan["selection_authority"][
                        "compiler_revision_sha256"
                    ],
                },
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
