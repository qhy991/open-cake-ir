from __future__ import annotations

import sys
import unittest
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


if __name__ == "__main__":
    unittest.main()
