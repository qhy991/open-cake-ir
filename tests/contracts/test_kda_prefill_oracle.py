"""Independent recurrence and sequence-boundary checks for KDA prefill."""

from __future__ import annotations

import unittest

import numpy as np

from open_cake_ir.tasks.kda_prefill.oracle import _bf16_round, reference_prefill


def one_hot(tokens: int, *, value: float = 1.0) -> np.ndarray:
    tensor = np.zeros((1, tokens, 1, 4), dtype=np.float32)
    tensor[..., 0] = value
    return tensor


class KdaPrefillOracle(unittest.TestCase):
    def test_bf16_ties_round_to_even(self) -> None:
        halfway = np.array([0x3F808000, 0x3F818000], dtype=np.uint32).view(np.float32)
        rounded = _bf16_round(halfway)
        np.testing.assert_array_equal(
            rounded.view(np.uint32), np.array([0x3F800000, 0x3F820000], dtype=np.uint32)
        )

    def test_token_state_rounding_and_output_use_updated_state(self) -> None:
        q = one_hot(2)
        k = one_hot(2)
        v = one_hot(2)
        g = np.zeros_like(q)
        beta = np.zeros((1, 2, 1), dtype=np.float32)
        state = np.zeros((1, 1, 4, 4), dtype=np.float32)
        before = state.copy()
        output, final = reference_prefill(
            q, k, v, g, beta,
            np.zeros(1, dtype=np.float32),
            np.zeros((1, 4), dtype=np.float32),
            state, np.array([0, 2], dtype=np.int64), scale=1.0,
        )
        np.testing.assert_array_equal(state, before)
        self.assertEqual(output[0, 0, 0, 0], 0.5)
        expected_second = 0.5 + 0.25 * np.exp(-2.5)
        self.assertAlmostEqual(output[0, 1, 0, 0], expected_second, delta=0.004)
        self.assertEqual(output[0, 1, 0, 0], final[0, 0, 0, 0])
        np.testing.assert_array_equal(output[..., 1:], np.zeros((1, 2, 1, 3), dtype=np.float32))

    def test_packed_boundaries_keep_separate_state_trajectories(self) -> None:
        q = one_hot(3)
        k = one_hot(3)
        v = one_hot(3)
        v[0, 1:, 0, 0] = 2.0
        g = np.zeros_like(q)
        beta = np.zeros((1, 3, 1), dtype=np.float32)
        A_log = np.zeros(1, dtype=np.float32)
        dt_bias = np.zeros((1, 4), dtype=np.float32)
        initial = np.zeros((2, 1, 4, 4), dtype=np.float32)
        packed, final = reference_prefill(
            q, k, v, g, beta, A_log, dt_bias, initial,
            np.array([0, 1, 3], dtype=np.int64), scale=1.0,
        )
        second, second_final = reference_prefill(
            q[:, 1:], k[:, 1:], v[:, 1:], g[:, 1:], beta[:, 1:],
            A_log, dt_bias, initial[1:2], np.array([0, 2], dtype=np.int64), scale=1.0,
        )
        np.testing.assert_array_equal(packed[:, 1:], second)
        np.testing.assert_array_equal(final[1:2], second_final)
        self.assertEqual(packed[0, 0, 0, 0], 0.5)
        self.assertEqual(packed[0, 1, 0, 0], 1.0)

    def test_rejects_missing_or_overlapping_sequence_coverage(self) -> None:
        q = one_hot(3)
        kwargs = dict(
            q=q, k=q, v=q, g=q, beta=np.zeros((1, 3, 1), dtype=np.float32),
            A_log=np.zeros(1, dtype=np.float32),
            dt_bias=np.zeros((1, 4), dtype=np.float32),
            initial_state=None,
        )
        with self.assertRaisesRegex(ValueError, "partition all tokens"):
            reference_prefill(**kwargs, cu_seqlens=np.array([0, 1, 1, 3], dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "partition all tokens"):
            reference_prefill(**kwargs, cu_seqlens=np.array([0, 2], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
