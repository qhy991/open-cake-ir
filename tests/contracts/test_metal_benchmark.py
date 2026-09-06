"""Portable measurement/oracle contracts; no GPU execution or toolchain repairs."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
from types import SimpleNamespace
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
                    "warmed_host_call_seconds": gpu + 0.1, "amortized_dispatch_seconds": gpu / 8,
                    "validation": {"gpu_correctness": "passed", "input_immutability": "passed", "max_abs_error": 0.0}})
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

    def test_unvalidated_ordinary_samples_cannot_qualify(self):
        job, original = protocol_fixture()
        for validation in (None, {}, {"gpu_correctness": "failed", "input_immutability": "passed"},
                           {"gpu_correctness": "passed", "input_immutability": "failed"}):
            result = copy.deepcopy(original)
            result["raw_samples"][0]["validation"] = validation
            with self.assertRaisesRegex(ValueError, "output/input validation"):
                benchmark.analyze(result, job)

    def test_reference_and_unknown_batch_failures_do_not_route_to_candidates(self):
        cases = (
            ("reference_prepare", None, "reference_prepare", "simd_reference", "handwritten_reference", None),
            ("batch", {"stage": "compile", "artifact": "simd_reference", "status": "started"},
             "reference_compile", "simd_reference", "handwritten_reference", None),
            ("batch", None, "batch", None, "unknown", None),
            ("batch", {"stage": "compile", "artifact": "canonical", "status": "completed"},
             "batch", None, "unknown", None),
            ("batch", {"stage": "compile", "artifact": "unrecognized", "status": "started"},
             "batch", None, "unknown", None),
            ("batch", {"stage": "compile", "artifact": "canonical", "status": "started"},
             "compile", "canonical", "compiler_generated", "verifier"),
        )
        for failure_at, event, stage, artifact, origin, route in cases:
            with self.subTest(failure_at=failure_at, event=event), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary).resolve()

                def prepare(compiler, document, inputs, oracles, path, device_names, case):
                    case.update(findings=[], static_accepted=True, lowering_eligible=True)

                def reference(rows, columns, execution, inputs, path, device_names):
                    if failure_at == "reference_prepare" and path.name == "simd_reference":
                        raise ValueError("injected handwritten reference preparation failure")
                    return {}

                def fail_batch(binary, path, job):
                    if event is not None:
                        (path / "events.jsonl").write_text(json.dumps(event) + "\n")
                    raise RuntimeError("injected batch failure")

                with patch.object(benchmark, "fresh_receipt", return_value=directory), \
                     patch.object(benchmark, "runtime_source", return_value={"tracked": True, "clean": True}), \
                     patch.object(benchmark, "released_compiler", return_value=(object(), {"revision_id": "CPU-mock"}, ["Apple M2"])), \
                     patch.object(benchmark, "compile_runner", return_value=Path("/nonexistent/no-gpu")), \
                     patch.object(benchmark, "prepare_case", side_effect=prepare), \
                     patch.object(benchmark, "invoke_batch", side_effect=fail_batch), \
                     patch.object(benchmark, "evaluate_case", side_effect=AssertionError("no held-out execution")), \
                     patch.object(benchmark.frontend, "read_schedule", return_value=SimpleNamespace(document={}, location_for=lambda _: None)), \
                     patch.object(rmsnorm, "source", return_value="CPU mock source"), \
                     patch.object(rmsnorm, "inputs_and_oracle", return_value=({}, {})), \
                     patch.object(rmsnorm, "reference", side_effect=reference), \
                     patch.object(benchmark, "route_rejection", wraps=benchmark.route_rejection) as router, \
                     patch.object(benchmark.subprocess, "run", side_effect=AssertionError("no subprocess permitted")), \
                     patch("sys.argv", ["benchmark", "--output-root", str(directory)]):
                    self.assertEqual(benchmark.main(), 1)
                receipt = json.loads((directory / "receipt.json").read_text())
                feedback = receipt["failure"]
                self.assertEqual(receipt["status"], "failed")
                self.assertEqual((feedback["stage"], feedback["artifact"], feedback["origin"]), (stage, artifact, origin))
                if route is None:
                    router.assert_not_called()
                    self.assertNotIn("route", feedback)
                    self.assertIn("invalid", feedback["measurement_quality"])
                    self.assertEqual(feedback["findings"], [])
                else:
                    router.assert_called_once()
                    self.assertEqual(feedback["route"]["destination"], route)


# The real executeBatch is executed below with CPU-only device/Prepared/profile
# doubles. Synthetic durations drive its control flow; they are not measurements.
_CPU_BATCH_DOUBLES = r"""import Foundation
struct MockDevice {}
struct MockQueue {}
typealias MTLDevice = MockDevice
typealias MTLCommandQueue = MockQueue
struct Refusal: Error, CustomStringConvertible { let description: String }
func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw Refusal(description: message) }
}
let fault = CommandLine.arguments[2]
var trace: [[String: Any]] = []
struct MockManifest { let execution_model = "simd_program_tile" }
final class Prepared {
    let id: String
    let manifest = MockManifest()
    let coldPrepareSeconds = 0.0
    let coldLibraryPipelineSeconds = 0.0
    var outputCorrect = true
    var inputIntact = true
    var calls = 0
    init(path: String, device: MTLDevice) throws {
        id = URL(fileURLWithPath: path).deletingLastPathComponent().lastPathComponent
    }
    func dispatch(queue: MTLCommandQueue, count: Int) throws -> [String: Any] {
        calls += 1
        outputCorrect = !(fault == "ordinary_output" && id == "candidate" && count > 1) &&
                        !(fault == "pilot_output" && id == "reference" && calls == 3)
        inputIntact = !(fault == "ordinary_input" && id == "candidate" && count > 1)
        trace.append(["action": "dispatch", "artifact": id, "count": count,
                      "output_correct": outputCorrect, "input_intact": inputIntact])
        let synthetic = 0.001 * Double(count)
        return ["command_status": "completed", "dispatches": count,
                "warmed_host_call_seconds": synthetic + 0.001,
                "gpu_start_time_seconds": 1.0, "gpu_end_time_seconds": 1.0 + synthetic,
                "gpu_command_buffer_seconds": synthetic,
                "amortized_dispatch_seconds": synthetic / Double(count)]
    }
    func validate(oraclePath: String) throws -> [String: Any] {
        trace.append(["action": "validate", "artifact": id,
                      "output_correct": outputCorrect, "input_intact": inputIntact])
        try require(outputCorrect && inputIntact, "injected " + fault)
        return ["gpu_correctness": "passed", "input_immutability": "passed", "max_abs_error": 0.0]
    }
    func writeOutputs() throws {}
}
func profile(_ item: Prepared, device: MTLDevice, queue: MTLCommandQueue) throws -> [String: Any] {
    trace.append(["action": "profile_start", "artifact": item.id])
    let observed = try item.dispatch(queue: queue, count: 1)
    try validTimer(observed)
    return ["coverage": "CPU_mock_only", "instrumented_command": observed]
}
"""
_CPU_BATCH_DRIVER = r"""
var evidence: [String: Any] = ["scope": "CPU-only control flow; synthetic timer fields; no GPU"]
var exitCode: Int32 = 0
do {
    let result = try executeBatch(path: CommandLine.arguments[1], device: MockDevice(), queue: MockQueue(), compileOnly: false)
    evidence["status"] = "completed"
    evidence["result"] = result
} catch {
    evidence["status"] = "refused"
    evidence["error"] = String(describing: error)
    exitCode = 1
}
evidence["trace"] = trace
let output = URL(fileURLWithPath: CommandLine.arguments[1]).deletingLastPathComponent().appendingPathComponent("outcome.json")
try JSONSerialization.data(withJSONObject: evidence, options: [.sortedKeys]).write(to: output, options: .withoutOverwriting)
exit(exitCode)
"""


class MetalBatchControlFlowContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        swift = shutil.which("swiftc")
        if swift is None:
            raise unittest.SkipTest("CPU-only Swift control-flow regression requires an existing swiftc")
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name).resolve()
        runner = (Path(__file__).resolve().parents[2] / "tools/metal/runner.swift").read_text()
        start = runner.index("struct Artifact: Decodable")
        end = runner.index("\nfunc execute()", start)
        # Reuse the actual types, timer helpers and executeBatch verbatim.
        source = _CPU_BATCH_DOUBLES + runner[start:end] + _CPU_BATCH_DRIVER
        if "import Metal" in source or "MTLCreateSystemDefaultDevice" in source:
            raise AssertionError("the CPU harness must not access Metal")
        path = cls.directory / "batch_cpu.swift"
        path.write_text(source)
        cls.binary = cls.directory / "batch-cpu"
        compiled = subprocess.run([swift, "-O", str(path), "-o", str(cls.binary)], capture_output=True, text=True, timeout=120)
        if compiled.returncode:
            raise AssertionError(compiled.stdout + compiled.stderr)

    def execute(self, fault):
        directory = self.directory / fault
        directory.mkdir()
        names = ("candidate", "reference", "reference_null")
        artifacts = [{"id": id, "manifest_path": str(directory / id / "manifest.json"),
                      "oracle_path": str(directory / id / "oracle.json"),
                      "origin": "compiler_generated" if id == "candidate" else "handwritten_reference"}
                     for id in names]
        job = {"artifacts": artifacts, "warmups": 1, "pilot_samples": 1, "max_batch_dispatches": 4,
               "target_command_seconds": 0.003, "orders": [[list(names)],
                   [["reference", "__selected__", "reference_null"]],
                   [["reference_null", "__selected__", "reference"]]]}
        path = directory / "batch.json"
        path.write_text(json.dumps(job))
        process = subprocess.run([str(self.binary), str(path), fault], capture_output=True, text=True, timeout=30)
        self.assertTrue((directory / "outcome.json").exists(), process.stderr)
        return process.returncode, json.loads((directory / "outcome.json").read_text())

    def test_each_bad_ordinary_or_pilot_observation_fails_before_overwrite(self):
        for fault in ("ordinary_output", "ordinary_input", "pilot_output"):
            with self.subTest(fault=fault):
                code, evidence = self.execute(fault)
                self.assertEqual(code, 1)
                self.assertEqual(evidence["status"], "refused")
                self.assertEqual(evidence["error"], "injected " + fault)
                bad = [e for e in evidence["trace"] if e["action"] == "validate" and
                       (not e["output_correct"] or not e["input_intact"])]
                self.assertEqual(len(bad), 1)
                self.assertFalse(any(e["action"] == "profile_start" for e in evidence["trace"]))

    def test_successful_observations_retain_separate_validation(self):
        code, evidence = self.execute("none")
        self.assertEqual(code, 0)
        result = evidence["result"]
        self.assertEqual(result["batch_dispatches"], 4)
        self.assertEqual(len(result["raw_samples"]), 9)
        for sample in result["pilot_samples"] + result["raw_samples"]:
            self.assertEqual(sample["validation"]["gpu_correctness"], "passed")
            self.assertEqual(sample["validation"]["input_immutability"], "passed")
        for observation in result["instrumented_observations"].values():
            self.assertEqual(observation["validation"]["gpu_correctness"], "passed")


if __name__ == "__main__":
    unittest.main()
