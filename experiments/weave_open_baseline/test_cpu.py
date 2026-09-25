"""CPU checks for the independent baseline input and oracle."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data import compare_outputs, load_contract, make_rank, reference, round_bf16
from runner import input_observation, load_rank_snapshot


def small_contract() -> dict:
    document = copy.deepcopy(load_contract())
    document["geometry"].update(tokens_total=8, tokens_per_rank=2,
                                experts=4, top_k=2, hidden=4, intermediate=8)
    return document


class CpuOracleTest(unittest.TestCase):
    def test_device_launcher_refuses_without_broker_lease(self):
        directory = str(Path(__file__).parent)
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("GPUQ_")}
        result = subprocess.run([str(Path(directory) / "run_under_broker.sh"),
                                 directory, directory, directory, directory],
                                capture_output=True, text=True, check=False, env=environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("broker-issued exclusive", result.stderr)

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

    def test_cpu_phase_retains_inputs_for_device_phase(self):
        document = small_contract()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            contract_path = path / "contract.json"
            contract_path.write_text(json.dumps(document))
            input_path = path / "inputs"
            subprocess.run([sys.executable, str(Path(__file__).with_name("runner.py")),
                            "oracle", "--contract", str(contract_path),
                            "--output", str(input_path)], check=True,
                           stdout=subprocess.DEVNULL)
            self.assertEqual(input_observation(document, input_path)["shape"], [4, 2, 4])
            for rank in range(4):
                np.testing.assert_array_equal(load_rank_snapshot(document, input_path, rank)["ids"],
                                              make_rank(document, rank)["ids"])
            self.assertTrue(np.all(np.isfinite(np.load(input_path / "oracle-expected.npy"))))


if __name__ == "__main__":
    unittest.main()
