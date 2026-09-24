"""Check chunk algebra against the independent token recurrence before device work."""

from __future__ import annotations

import unittest

import numpy as np

from open_cake_ir.tasks.kda_prefill.chunked import chunked_prefill
from open_cake_ir.tasks.kda_prefill.oracle import _bf16_round, reference_prefill


def inputs(*, lengths: tuple[int, ...], heads: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    tokens = sum(lengths)
    shape = (1, tokens, heads, 128)
    arrays = {
        name: _bf16_round(rng.standard_normal(shape).astype(np.float32))
        for name in ("q", "k", "v")
    }
    arrays["g"] = _bf16_round(
        (0.1 * rng.standard_normal(shape)).astype(np.float32)
    )
    arrays["beta"] = _bf16_round(
        rng.standard_normal((1, tokens, heads)).astype(np.float32)
    )
    arrays["A_log"] = (0.1 * rng.standard_normal(heads)).astype(np.float32)
    arrays["dt_bias"] = (0.1 * rng.standard_normal((heads, 128))).astype(np.float32)
    arrays["initial_state"] = _bf16_round(
        (0.1 * rng.standard_normal((len(lengths), heads, 128, 128))).astype(np.float32)
    )
    arrays["cu_seqlens"] = np.asarray(
        (0, *np.cumsum(lengths).tolist()), dtype=np.int64
    )
    return arrays


def arguments(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    return tuple(arrays[name] for name in (
        "q", "k", "v", "g", "beta", "A_log", "dt_bias",
        "initial_state", "cu_seqlens",
    ))


class KdaPrefillChunked(unittest.TestCase):
    def assert_complete_tolerance(
        self, actual: tuple[np.ndarray, np.ndarray],
        expected: tuple[np.ndarray, np.ndarray],
    ) -> None:
        for name, a, e in zip(("output", "state"), actual, expected, strict=True):
            self.assertEqual(a.shape, e.shape, name)
            self.assertTrue(np.all(np.isfinite(a)), name)
            delta = np.abs(a - e)
            allowed = 0.01 + 0.01 * np.abs(e)
            self.assertEqual(int(np.count_nonzero(delta > allowed)), 0, name)

    def test_full_chunk_tail_and_longer_trajectory_match_token_oracle(self) -> None:
        for lengths, heads, seed in (((65,), 64, 427027), ((257,), 16, 10003)):
            arrays = inputs(lengths=lengths, heads=heads, seed=seed)
            before = {name: value.copy() for name, value in arrays.items()}
            expected = reference_prefill(*arguments(arrays))
            for chunk in (4, 8, 16, 32):
                with self.subTest(lengths=lengths, heads=heads, chunk=chunk):
                    self.assert_complete_tolerance(
                        chunked_prefill(*arguments(arrays), chunk=chunk), expected
                    )
            for name in arrays:
                np.testing.assert_array_equal(arrays[name], before[name])

    def test_packed_chunk_boundaries_keep_each_state_separate(self) -> None:
        arrays = inputs(lengths=(33, 65, 17), heads=8, seed=427028)
        self.assert_complete_tolerance(
            chunked_prefill(*arguments(arrays), chunk=32),
            reference_prefill(*arguments(arrays)),
        )

    def test_refuses_unsupported_chunk_and_incomplete_offsets(self) -> None:
        arrays = inputs(lengths=(65,), heads=2, seed=1)
        for size in (0, 3, 64, True):
            with self.subTest(chunk=size), self.assertRaisesRegex(ValueError, "chunk"):
                chunked_prefill(*arguments(arrays), chunk=size)
        changed = dict(arrays)
        changed["cu_seqlens"] = np.asarray([0, 64], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "complete KDA state boundary"):
            chunked_prefill(*arguments(changed))


if __name__ == "__main__":
    unittest.main()
