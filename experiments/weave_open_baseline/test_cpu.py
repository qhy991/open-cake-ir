"""CPU checks for the independent baseline input and oracle."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data import compare_outputs, load_contract, make_rank, reference, round_bf16


def small_contract() -> dict:
    document = copy.deepcopy(load_contract())
    document["geometry"].update(tokens_total=8, tokens_per_rank=2,
                                experts=4, top_k=2, hidden=4, intermediate=8)
    return document


class CpuOracleTest(unittest.TestCase):
    def test_bf16_rounds_ties_to_even(self):
        values = np.array([1.0 + 1.0 / 256.0, 1.0 + 3.0 / 256.0], dtype=np.float32)
        np.testing.assert_array_equal(round_bf16(values), [1.0, 1.0 + 2.0 / 128.0])

    def test_reference_uses_owner_weights_and_route_weights(self):
        document = small_contract()
        data = [make_rank(document, r) for r in range(4)]
        expected = reference(document)
        contribution = np.zeros(4, dtype=np.float64)
        for slot in range(2):
            expert = int(data[0]["ids"][0, slot])
            owner = expert
            x = data[0]["hidden"][0].astype(np.float64)
            gate = data[owner]["gate"][0].astype(np.float64) @ x
            up = data[owner]["up"][0].astype(np.float64) @ x
            activated = up * gate / (1.0 + np.exp(-gate))
            down = data[owner]["down"][0].astype(np.float64) @ activated
            contribution += float(data[0]["weights"][0, slot]) * down
        np.testing.assert_array_equal(expected[0, 0], round_bf16(contribution.astype(np.float32)))

    def test_output_gate_rejects_a_wrong_rank(self):
        document = small_contract()
        expected = reference(document)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for rank in range(4):
                np.save(path / f"rank{rank}-output.npy", expected[rank])
            self.assertTrue(compare_outputs(document, path)["pass"])
            corrupted = expected[2].copy()
            corrupted[0, 0] += 1.0
            np.save(path / "rank2-output.npy", corrupted)
            result = compare_outputs(document, path)
            self.assertFalse(result["pass"])
            self.assertEqual(result["failing_elements"], 1)


if __name__ == "__main__":
    unittest.main()
