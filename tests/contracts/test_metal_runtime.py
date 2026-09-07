"""Portable host-protocol checks: no Swift invocation and no GPU commands."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import Mock, patch

from open_cake_ir.compiler import Assessment, Lowering, Schedule, frontend
from tools.metal import adapter, check_correctness

ROOT = Path(__file__).resolve().parents[2]


def host_fixture():
    """Synthetic assessment identities isolate the host protocol from release authority."""
    source = frontend.read_schedule(ROOT / "examples/python/metal_elementwise.py")
    schedule = Schedule.from_dict(source.document)
    common = dict(compiler_revision_id="host-test", compiler_revision_sha256="fixture-revision",
                  schedule_id=schedule.schedule_id, schedule_sha256="fixture-schedule",
                  target=schedule.target, route=schedule.lowering)
    assessment = Assessment(**common, accepted=True, lowering_eligible=True, findings=(),
                            analysis={}, lowering_parameters={}, calibration_available=False,
                            schedule_bytes=source.schedule_bytes)
    lowering = Lowering(**common, generated=True, source="// host protocol fixture\n",
                        source_sha256="fixture-source", source_map={}, toolchain_requirements={
                            "target": schedule.target, "source_language": "metal",
                            "compiler": "MTLDevice.makeLibrary", "language_standard": "metal2.3",
                            "fast_math_enabled": False, "threadgroup_memory_bytes": 0,
                            "execution_model": "serial_program_tile", "active_threads_per_threadgroup": 1,
                            "buffer_order": ["x", "y", "out"], "threadgroups_per_grid": [3, 1, 1],
                            "threads_per_threadgroup": [32, 1, 1]})
    return assessment, lowering, {"x": bytes(3 * 37 * 4), "y": bytes(3 * 37 * 4)}


class MetalRuntimeContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.assessment, self.lowering, self.inputs = host_fixture()

    def project(self, assessment=None, lowering=None, inputs=None):
        return adapter.manifest(assessment or self.assessment, lowering or self.lowering,
                                self.inputs if inputs is None else inputs, self.directory,
                                device_names=["Apple M2"])

    def test_exact_targets_refuse_other_device_names_before_writing(self):
        for target, device in (("apple_gpu_family7", "Apple M1 Pro"),
                               ("apple_gpu_family8", "Apple M2")):
            document = json.loads(self.assessment.schedule_bytes)
            document["target"] = target
            assessment = replace(self.assessment, target=target, schedule_bytes=json.dumps(document).encode())
            lowering = replace(self.lowering, target=target, toolchain_requirements={
                **self.lowering.toolchain_requirements, "target": target})
            directory = self.directory / target
            directory.mkdir()
            for names in ([], ["Apple M1"], ["Apple M1 Max"], ["Apple M2 Pro"],
                          ["Apple M2" if target == "apple_gpu_family7" else "Apple M1 Pro"],
                          ["Apple M1 Pro", "Apple M2"]):
                with self.subTest(target=target, names=names), self.assertRaises(ValueError):
                    adapter.manifest(assessment, lowering, self.inputs, directory, device_names=names)
                self.assertEqual(list(directory.iterdir()), [])
            result = adapter.manifest(assessment, lowering, self.inputs, directory, device_names=[device])
            self.assertEqual(result["target"], target)
            self.assertEqual(result["device_names"], [device])

    def test_manifest_preserves_odd_shapes_order_and_output_poison_contract(self):
        result = self.project()
        self.assertEqual([b["name"] for b in result["buffers"]], ["x", "y", "out"])
        self.assertTrue(all(b["shape"] == [3, 37] and b["size_bytes"] == 444
                            for b in result["buffers"]))
        self.assertIsNone(result["buffers"][-1]["input_path"])
        self.assertEqual(result["threadgroups_per_grid"], [3, 1, 1])
        self.assertEqual(result["entry_point"], "cake_elementwise")
        self.assertEqual(Path(result["buffers"][0]["input_path"]).read_bytes(), self.inputs["x"])

    def test_invalid_execution_commitments_refuse_before_writing(self):
        changes = ({"fast_math_enabled": True}, {"fast_math_enabled": 0},
                   {"execution_model": "parallel"}, {"active_threads_per_threadgroup": 32},
                   {"threads_per_threadgroup": [1, 1, 1]}, {"threadgroups_per_grid": [4, 1, 1]},
                   {"threadgroups_per_grid": [True, 1, 1]}, {"threadgroup_memory_bytes": 4},
                   {"language_standard": "metal3.0"}, {"buffer_order": ["out", "x", "y"]},
                   {"unknown_policy": "ignored"}, {"target": "sm_100a"})
        for change in changes:
            with self.subTest(change=change):
                lowering = replace(self.lowering, toolchain_requirements={**self.lowering.toolchain_requirements, **change})
                with self.assertRaises(ValueError):
                    self.project(lowering=lowering)
                self.assertEqual(list(self.directory.iterdir()), [])

    def test_refused_or_unrelated_lowering_cannot_reach_runtime(self):
        with self.assertRaises(ValueError):
            self.project(assessment=replace(self.assessment, accepted=False))
        with self.assertRaises(ValueError):
            self.project(lowering=replace(self.lowering, schedule_sha256="different-assessment"))
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_simd_launch_uses_the_same_buffer_projection_as_serial(self):
        lowering = replace(self.lowering, toolchain_requirements={**self.lowering.toolchain_requirements,
                            "execution_model": "simd_program_tile", "active_threads_per_threadgroup": 32})
        result = self.project(lowering=lowering)
        self.assertEqual(result["execution_model"], "simd_program_tile")
        self.assertEqual(result["threads_per_threadgroup"], [32, 1, 1])
        self.assertEqual(result["buffers"][-1]["shape"], [3, 37])

    def test_input_count_byte_length_and_finiteness_fail_closed(self):
        for inputs in ({"x": self.inputs["x"]}, {**self.inputs, "x": b""},
                       {**self.inputs, "x": struct.pack("<f", float("nan")) * 111},
                       {**self.inputs, "x": struct.pack("<f", float("inf")) * 111}):
            with self.subTest(inputs=list(inputs)):
                with self.assertRaises(ValueError):
                    self.project(inputs=inputs)
                self.assertEqual(list(self.directory.iterdir()), [])

    def test_unsupported_dtype_refuses_before_writing(self):
        document = json.loads(self.assessment.schedule_bytes)
        for buffer in document["buffers"]:
            buffer["dtype"] = "fp16"
        assessment = replace(self.assessment, schedule_bytes=json.dumps(document).encode())
        with self.assertRaisesRegex(ValueError, "unsupported buffer"):
            self.project(assessment=assessment)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_existing_case_is_never_overwritten(self):
        self.project()
        before = (self.directory / "manifest.json").read_bytes()
        with self.assertRaises(ValueError):
            self.project()
        self.assertEqual((self.directory / "manifest.json").read_bytes(), before)

    def test_oracle_detects_missing_and_wrong_output(self):
        expected, tolerance = check_correctness.oracle("elementwise", [1.0, -3.0], [2.0, 1.0], 2)
        check_correctness.compare(struct.pack("<2f", 6.0, -4.0), expected, tolerance)
        for payload in (b"", struct.pack("<2f", 0.0, 0.0), struct.pack("<2f", float("nan"), -4.0)):
            with self.assertRaises(ValueError):
                check_correctness.compare(payload, expected, tolerance)

    def test_varied_shapes_and_distributions_use_canonical_frontend(self):
        for target in ("apple_gpu_family7", "apple_gpu_family8"):
            for operator in ("elementwise", "row_sum", "row_max", "rmsnorm"):
                for rows, columns in check_correctness.SHAPES:
                    with self.subTest(target=target, operator=operator, shape=(rows, columns)):
                        schedule = Schedule.from_dict(check_correctness.schedule_document(operator, rows, columns, target))
                        self.assertEqual(schedule.target, target)
                        self.assertEqual(schedule.buffer("x").shape, (rows, columns))
                        self.assertEqual(schedule.buffer("out").shape,
                                         (rows,) if operator in {"row_sum", "row_max"} else (rows, columns))
        first = check_correctness.values(19, "uniform", 7)
        self.assertEqual(first, check_correctness.values(19, "uniform", 7))
        self.assertNotEqual(first, check_correctness.values(19, "uniform", 8))

    def test_candidate_refusal_stops_before_cost_ranking_and_lowering(self):
        for accepted, eligible in ((False, False), (True, False)):
            with self.subTest(accepted=accepted, lowering_eligible=eligible):
                compiler = Mock()
                compiler.assess.return_value = replace(self.assessment, accepted=accepted,
                                                       lowering_eligible=eligible)
                case = {}
                with self.assertRaisesRegex(ValueError, "Compiler refused"):
                    check_correctness.prepare_case(compiler, {}, self.inputs, {}, self.directory,
                                                   ["Apple M2"], case)
                compiler.rank.assert_not_called()
                compiler.lower.assert_not_called()
                self.assertEqual(json.loads((self.directory / "assessment.json").read_text()), case)
                self.assertEqual(case["static_accepted"], accepted)
                self.assertFalse(case["lowering_eligible"])
                self.assertFalse((self.directory / "manifest.json").exists())

    def test_failed_release_gate_prevents_host_build_and_dispatch(self):
        with patch.object(check_correctness, "fresh_receipt", return_value=self.directory), \
             patch.object(check_correctness, "runtime_source", return_value={"tracked": True, "clean": True}), \
             patch.object(check_correctness, "released_compiler", side_effect=ValueError("unreviewed")), \
             patch.object(check_correctness, "compile_runner") as build, \
             patch.object(check_correctness, "invoke") as run, \
             patch("sys.argv", ["check_correctness", "--output-root", str(self.directory)]):
            self.assertEqual(check_correctness.main(), 1)
        build.assert_not_called()
        run.assert_not_called()
        receipt = json.loads((self.directory / "receipt.json").read_text())
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["cases"], [])

    def test_uncommitted_runtime_prevents_release_check_and_dispatch(self):
        with patch.object(check_correctness, "fresh_receipt", return_value=self.directory), \
             patch.object(check_correctness, "runtime_source", return_value={"tracked": False, "clean": False}), \
             patch.object(check_correctness, "released_compiler") as release, \
             patch.object(check_correctness, "invoke") as run, \
             patch("sys.argv", ["check_correctness", "--output-root", str(self.directory)]):
            self.assertEqual(check_correctness.main(), 1)
        release.assert_not_called()
        run.assert_not_called()

    def test_receipts_are_fresh_and_outside_checkout(self):
        first = check_correctness.fresh_receipt(self.directory)
        second = check_correctness.fresh_receipt(self.directory)
        self.assertNotEqual(first, second)
        with self.assertRaises(ValueError):
            check_correctness.fresh_receipt(ROOT / "evidence")
        with self.assertRaises(ValueError):
            check_correctness.fresh_receipt(Path("relative"))

    def test_nonzero_runtime_exit_never_becomes_completion(self):
        from subprocess import CompletedProcess
        with patch.object(adapter.subprocess, "run", return_value=CompletedProcess([], 1, "", "command failed")):
            with self.assertRaisesRegex(ValueError, "refused or failed"):
                adapter.invoke(Path("/fake/runner"), self.directory)

    def test_host_build_is_optimized_and_records_the_invoked_argv(self):
        from subprocess import CompletedProcess
        with patch.object(adapter.subprocess, "run", return_value=CompletedProcess([], 0, "", "")) as build:
            binary = adapter.compile_runner(self.directory)
        build.assert_called_once()
        argv = build.call_args.args[0]
        self.assertEqual(argv.count("-O"), 1)
        self.assertNotIn("-Ounchecked", argv)
        self.assertEqual(argv[-1], str(binary))
        self.assertEqual(json.loads((self.directory / "swift-build-command.json").read_text()), argv)

    def test_success_exit_without_command_completion_is_not_gpu_success(self):
        from subprocess import CompletedProcess
        with patch.object(adapter.subprocess, "run", return_value=CompletedProcess([], 0, '{"status":"completed"}', "")):
            with self.assertRaisesRegex(ValueError, "command completion is missing"):
                adapter.invoke(Path("/fake/runner"), self.directory)


if __name__ == "__main__":
    unittest.main()
