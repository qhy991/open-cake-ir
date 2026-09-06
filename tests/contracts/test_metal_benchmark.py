"""Portable measurement/oracle contracts; no GPU execution or toolchain repairs."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import Schedule
from tools.metal import benchmark, rmsnorm


def protocol_fixture():
    artifacts = [{"id": id, "origin": "compiler_generated"} for id in rmsnorm.FORMULAS]
    artifacts += [{"id": id, "origin": "handwritten_reference"} for id in benchmark.REFERENCES]
    job = {"artifacts": artifacts, "orders": benchmark.orders(list(rmsnorm.FORMULAS))}
    times = {"canonical": 1.0, "weight_first": 1.2, "prescaled_square": 1.3,
             "serial_reference": 10.0, "simd_reference": 2.0, "simd_reference_null": 2.0}
    samples = []
    for r, sweeps in enumerate(job["orders"]):
        for s, order in enumerate(sweeps):
            for p, slot in enumerate(order):
                id = "canonical" if slot == "__selected__" else slot
                gpu = times[id] * (1 + (s - 4) * 0.001)
                samples.append({"round": r, "sweep": s, "position": p, "artifact": id,
                    "dispatches": 8, "command_status": "completed", "gpu_command_buffer_seconds": gpu,
                    "warmed_host_call_seconds": gpu + 0.1, "amortized_dispatch_seconds": gpu / 8})
    result = {"status": "completed", "ordinary_samples_instrumented": False, "selected_candidate": "canonical",
              "batch_dispatches": 8, "raw_samples": samples}
    return job, result


def change_gpu(sample, value):
    sample["gpu_command_buffer_seconds"] = value
    sample["amortized_dispatch_seconds"] = value / sample["dispatches"]
    sample["warmed_host_call_seconds"] = value + 0.1


class MetalBenchmarkContracts(unittest.TestCase):
    def test_formulas_are_distinct_canonical_schedules_under_one_contract(self):
        documents = [rmsnorm.document(128, 1024, formula) for formula in rmsnorm.FORMULAS]
        self.assertEqual(len(documents), 3)
        normalized = []
        for document in documents:
            schedule = Schedule.from_dict(document)
            self.assertEqual(schedule.buffer("x").shape, (128, 1024))
            self.assertEqual(schedule.buffer("weight").shape, (1024,))
            self.assertEqual(schedule.buffer("result").shape, (1024,))
            eps = [op for op in document["operations"] if op.get("parameters", {}).get("scalar") == 1e-5]
            self.assertEqual(len(eps), 1)
            normalized.append(json.dumps(document["operations"], sort_keys=True))
        self.assertEqual(len(set(normalized)), 3)
        self.assertEqual(rmsnorm.CONTRACT["atol"], 2e-5)
        self.assertEqual(rmsnorm.CONTRACT["rtol"], 2e-5)

    def test_inputs_fix_bounds_weights_zeros_and_epsilon_domain(self):
        first, oracle = rmsnorm.inputs_and_oracle(2, 65, "epsilon_dominated")
        self.assertEqual(first, rmsnorm.inputs_and_oracle(2, 65, "epsilon_dominated")[0])
        x = struct.unpack("<130f", first["x"])
        weight = struct.unpack("<65f", first["weight"])
        self.assertTrue(all(abs(v) <= 1e-4 for v in x))
        self.assertTrue(all(abs(v) <= 1.5 for v in weight))
        self.assertEqual(weight[16], 0.0)
        self.assertNotEqual(weight[0], 0.0)  # width=1 must still test arithmetic.
        self.assertTrue(any(v < 0 for v in weight) and any(v > 0 for v in weight))
        expected = oracle["out"]["expected"]
        inverse = 1 / math.sqrt(math.fsum(v * v for v in x[:65]) / 65 + 1e-5)
        self.assertEqual(expected[0], x[0] * inverse * weight[0])
        self.assertEqual(expected[16], 0.0)
        self.assertEqual(oracle["out"]["absolute_tolerance"][0], 2e-5 + 2e-5 * abs(expected[0]))

    def test_references_have_explicit_origin_without_fake_compiler_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            inputs, _ = rmsnorm.inputs_and_oracle(2, 7, "uniform")
            result = rmsnorm.reference(2, 7, "serial", inputs, directory, ["Apple M2"])
            origin = json.loads((directory / "origin.json").read_text())
            self.assertEqual(origin["kind"], "handwritten_reference")
            self.assertNotIn("compiler_revision_id", origin)
            self.assertEqual(result["entry_point"], "reference_serial")
            self.assertEqual(result["execution_model"], "serial_program_tile")
            self.assertEqual(result["buffers"][1]["shape"], [7])

    def test_search_then_independent_confirmations_have_distinct_matching_orders(self):
        orders = benchmark.orders(list(rmsnorm.FORMULAS))
        self.assertEqual(len(orders), 3)
        self.assertEqual(orders, benchmark.orders(list(rmsnorm.FORMULAS)))
        for r, sweeps in enumerate(orders):
            expected = set(benchmark.REFERENCES) | (set(rmsnorm.FORMULAS) if r == 0 else {"__selected__"})
            self.assertEqual(len(sweeps), 8)
            for sweep in sweeps:
                self.assertEqual(set(sweep), expected)
                self.assertEqual(len(sweep), len(expected))
        self.assertNotEqual(orders[1], orders[2])

    def test_matching_gain_can_qualify_only_with_both_confirmations(self):
        job, result = protocol_fixture()
        analysis = benchmark.analyze(result, job)
        self.assertTrue(analysis["comparisons"]["simd_reference"]["qualified_speedup"])
        self.assertEqual(analysis["comparisons"]["simd_reference"]["qualified_confirmation_ratio"], 2.0)
        for sample in result["raw_samples"]:
            if sample["round"] == 2 and sample["artifact"] == "canonical":
                change_gpu(sample, 2.0)
        analysis = benchmark.analyze(result, job)
        self.assertFalse(analysis["comparisons"]["simd_reference"]["qualified_speedup"])
        self.assertNotIn("qualified_confirmation_ratio", analysis["comparisons"]["simd_reference"])

    def test_failed_null_control_cannot_be_hidden_by_a_large_candidate_gain(self):
        job, result = protocol_fixture()
        for sample in result["raw_samples"]:
            if sample["artifact"] == "simd_reference_null":
                change_gpu(sample, 3.0)
        analysis = benchmark.analyze(result, job)
        self.assertTrue(all(not c["qualified_speedup"] for c in analysis["comparisons"].values()))
        self.assertTrue(all(not n["passed"] for n in analysis["null_control"]))

    def test_noisy_arm_is_inconclusive_without_sample_filtering(self):
        job, result = protocol_fixture()
        for sample in result["raw_samples"]:
            if sample["artifact"] == "canonical" and sample["round"] == 1:
                change_gpu(sample, 0.5 if sample["sweep"] % 2 else 1.5)
        analysis = benchmark.analyze(result, job)
        self.assertFalse(analysis["comparisons"]["simd_reference"]["qualified_speedup"])
        self.assertEqual(analysis["arm_statistics"]["1"]["canonical"]["gpu_command_buffer_seconds"]["samples"], 8)

    def test_missing_reordered_invalid_or_instrumented_samples_fail(self):
        job, original = protocol_fixture()
        variants = []
        altered = copy.deepcopy(original); altered["raw_samples"].pop(); variants.append(altered)
        altered = copy.deepcopy(original); altered["raw_samples"][0], altered["raw_samples"][1] = altered["raw_samples"][1], altered["raw_samples"][0]; variants.append(altered)
        altered = copy.deepcopy(original); altered["ordinary_samples_instrumented"] = True; variants.append(altered)
        altered = copy.deepcopy(original); change_gpu(altered["raw_samples"][0], 0.0); variants.append(altered)
        altered = copy.deepcopy(original); altered["raw_samples"][0]["amortized_dispatch_seconds"] = 42.0; variants.append(altered)
        for result in variants:
            with self.assertRaises(ValueError):
                benchmark.analyze(result, job)

    def test_uncommitted_runtime_blocks_benchmark_before_release_or_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            with patch.object(benchmark, "fresh_receipt", return_value=directory), \
                 patch.object(benchmark, "runtime_source", return_value={"tracked": False, "clean": False}), \
                 patch.object(benchmark, "released_compiler") as release, \
                 patch.object(benchmark, "invoke_batch") as gpu, \
                 patch("sys.argv", ["benchmark", "--output-root", str(directory)]):
                self.assertEqual(benchmark.main(), 1)
            release.assert_not_called(); gpu.assert_not_called()
            self.assertEqual(json.loads((directory / "receipt.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
