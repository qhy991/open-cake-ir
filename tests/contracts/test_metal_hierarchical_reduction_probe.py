from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/probe_metal_hierarchical_reduction.py"

_SPEC = importlib.util.spec_from_file_location(
    "open_cake_metal_hierarchical_reduction_probe", TOOL
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Metal hierarchical-reduction probe could not be imported")
PROBE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(PROBE)

_TEST_SOURCE_BINDING = {
    "repository_revision": "0" * 40,
    "worktree_clean": True,
    "source_paths": [
        "tools/probe_metal_hierarchical_reduction.py",
        "tools/metal_hierarchical_reduction_probe/hierarchical_reduction.metal",
        "tools/metal_hierarchical_reduction_probe/run_hierarchical_reduction.swift",
    ],
}


def _write_fake_xcrun(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json
            import os
            import pathlib
            import sys

            arguments = sys.argv[1:]
            executable = pathlib.Path(sys.argv[0]).resolve()
            if arguments == ["--sdk", "macosx", "--show-sdk-path"]:
                print(executable.parent)
                raise SystemExit(0)
            if arguments[:2] != ["--sdk", "macosx"] or len(arguments) < 3:
                print("unexpected fake xcrun arguments", file=sys.stderr)
                raise SystemExit(64)

            stage = arguments[2]
            mode = os.environ.get(
                "OPEN_CAKE_METAL_HIERARCHICAL_FAKE_MODE", "passed"
            )
            if stage == "metal":
                source = pathlib.Path(arguments[arguments.index("-c") + 1])
                output = pathlib.Path(arguments[arguments.index("-o") + 1])
                log = os.environ.get("OPEN_CAKE_METAL_HIERARCHICAL_FAKE_TEMP_LOG")
                if log:
                    pathlib.Path(log).write_text(str(output.parent), encoding="utf-8")
                required = {
                    "-std=metal3.2",
                    "-fmetal-math-mode=safe",
                    "-ffp-contract=off",
                    "-Werror",
                }
                if not required.issubset(arguments):
                    print("probe compiler flags drifted", file=sys.stderr)
                    raise SystemExit(65)
                if "kernel void hierarchical_reduction_w32_probe" not in source.read_text(
                    encoding="utf-8"
                ):
                    print("probe entry point is missing", file=sys.stderr)
                    raise SystemExit(66)
                if mode == "fail_compile":
                    print("intentional hierarchical compile failure", file=sys.stderr)
                    raise SystemExit(7)
                output.write_bytes(b"AIR hierarchical fixture")
                raise SystemExit(0)
            if stage == "metallib":
                if mode == "fail_link":
                    print("intentional hierarchical link failure", file=sys.stderr)
                    raise SystemExit(8)
                output = pathlib.Path(arguments[arguments.index("-o") + 1])
                output.write_bytes(b"MTLB hierarchical fixture")
                raise SystemExit(0)
            if stage == "swift":
                if "-warnings-as-errors" not in arguments:
                    print("Swift warnings were not promoted to errors", file=sys.stderr)
                    raise SystemExit(68)
                if mode == "fail_dispatch":
                    print("intentional hierarchical dispatch failure", file=sys.stderr)
                    raise SystemExit(9)
                group_counts = [1, 2, 4, 8, 16, 32]
                cases = []
                for group_count in group_counts:
                    threads = group_count * 32
                    epochs = []
                    for epoch in range(2):
                        sentinel = -12345.25 if epoch == 0 else 9876.5
                        if epoch == 0:
                            expected = sum(
                                ((index % 13) - 6) * 0.125 + 0.5
                                for index in range(threads)
                            )
                        else:
                            expected = sum(
                                ((index % 11) - 5) * 0.25 - 0.75
                                for index in range(threads)
                            )
                        epochs.append({
                            "epoch": epoch,
                            "input_pattern": (
                                "((thread_mod_13)-6)*0.125+0.5"
                                if epoch == 0
                                else "((thread_mod_11)-5)*0.25-0.75"
                            ),
                            "dirty_sentinel": sentinel,
                            "observed_dirty_partial_slot": sentinel,
                            "observed_dirty_scalar_slot": sentinel,
                            "dirty_snapshot_max_abs_error": 0.0,
                            "dirty_snapshot_mismatch_count": 0,
                            "dirty_sentinel_preserved": True,
                            "expected_sum": expected,
                            "observed_owner_sum": expected,
                            "owner_abs_error": 0.0,
                            "owner_mismatch_count": 0,
                            "broadcast_thread_count": threads,
                            "broadcast_mismatch_count": 0,
                            "broadcast_max_abs_error": 0.0,
                            "mismatch_count": 0,
                            "passed": True,
                        })
                    cases.append({
                        "group_count": group_count,
                        "threads_per_threadgroup": threads,
                        "threadgroup_memory_bytes": 144,
                        "command_buffer_status": "completed",
                        "epochs": epochs,
                        "passed": True,
                    })
                observation = {
                    "schema_version": 1,
                    "kind": "open_cake_metal_hierarchical_reduction_dispatch_v1",
                    "status": "passed",
                    "kernel_calls": 6,
                    "performance_measured": False,
                    "scientific_claim_authorized": False,
                    "probe_scope": (
                        "shared_choreography_and_rms_single_owner_broadcast_only"
                    ),
                    "non_claims": [
                        "no_timing_or_performance_claim",
                        "no_layer_norm_all_simdgroup_final_reducer_claim",
                        "no_bfloat_input_conversion_claim",
                        "no_compiler_lowering_or_ir_admission_claim",
                    ],
                    "device": {
                        "device_class": "Apple M4",
                        "supports_apple9_or_newer": True,
                        "max_threadgroup_memory_bytes": 32768,
                    },
                    "pipeline": {
                        "entry_point": "hierarchical_reduction_w32_probe",
                        "thread_execution_width": 32,
                        "max_total_threads_per_threadgroup": 1024,
                        "static_threadgroup_memory_bytes": 0,
                        "dynamic_threadgroup_memory_bytes": 144,
                    },
                    "coverage": {
                        "simdgroup_counts": group_counts,
                        "epochs_per_dispatch": 2,
                        "single_owner_final_scalar_broadcast": True,
                        "scratch_reused_within_dispatch": True,
                        "uniform_reuse_barrier": (
                            "threadgroup_barrier(mem_threadgroup)"
                        ),
                        "layer_all_simdgroups_tested": False,
                    },
                    "oracle": {
                        "dtype": "float32",
                        "tolerance_rule": "exact_fp32_bits_for_dyadic_fixture",
                    },
                    "cases": cases,
                }
                if mode == "wrong_width":
                    observation["pipeline"]["thread_execution_width"] = 16
                elif mode == "low_pipeline_limit":
                    observation["pipeline"]["max_total_threads_per_threadgroup"] = 512
                elif mode == "static_memory_drift":
                    observation["pipeline"]["static_threadgroup_memory_bytes"] = 1
                elif mode == "mismatch":
                    observation["cases"][3]["epochs"][1]["mismatch_count"] = 1
                    observation["cases"][3]["epochs"][1]["passed"] = False
                elif mode == "missing_boundary":
                    observation["coverage"]["simdgroup_counts"] = group_counts[:-1]
                elif mode == "same_epoch":
                    first = observation["cases"][0]["epochs"][0]
                    second = observation["cases"][0]["epochs"][1]
                    second["input_pattern"] = first["input_pattern"]
                    second["dirty_sentinel"] = first["dirty_sentinel"]
                    second["observed_dirty_partial_slot"] = first["dirty_sentinel"]
                    second["observed_dirty_scalar_slot"] = first["dirty_sentinel"]
                    second["expected_sum"] = first["expected_sum"]
                    second["observed_owner_sum"] = first["expected_sum"]
                elif mode == "wrong_oracle":
                    epoch = observation["cases"][2]["epochs"][0]
                    epoch["expected_sum"] += 1.0
                    epoch["observed_owner_sum"] = epoch["expected_sum"]
                elif mode == "sensitive_field":
                    observation["device"]["registry_id"] = 123
                print(json.dumps(observation, sort_keys=True, separators=(",", ":")))
                raise SystemExit(0)
            print(f"unsupported fake xcrun stage {stage}", file=sys.stderr)
            raise SystemExit(64)
            """
        ),
        encoding="utf-8",
    )
    path.chmod(0o700)


class MetalHierarchicalReductionProbeContractTests(unittest.TestCase):
    def test_test_double_compiles_links_dispatches_and_cleans(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake_xcrun = Path(directory) / "xcrun"
            _write_fake_xcrun(fake_xcrun)
            result = PROBE.probe(
                xcrun=fake_xcrun,
                timeout_seconds=30,
                _host_system="Darwin",
                _source_binding_override=_TEST_SOURCE_BINDING,
            )

        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["evaluation_evidence"])
        self.assertFalse(result["external_workload_oracle_used"])
        self.assertFalse(result["performance_measured"])
        self.assertFalse(result["scientific_claim_authorized"])
        self.assertFalse(result["compiler_change_authorized"])
        self.assertFalse(result["human_hardware_review_cleared"])
        self.assertFalse(result["promotion_authorized"])
        self.assertFalse(result["stable_device_identifiers_retained"])
        self.assertTrue(result["temporary_directory_cleaned"])
        self.assertEqual(result["source_binding"], _TEST_SOURCE_BINDING)
        self.assertEqual(result["toolchain"]["compiled_air_count"], 1)
        self.assertTrue(result["toolchain"]["linked_metallib"])
        dispatch = result["dispatch_observation"]
        self.assertEqual(dispatch["kernel_calls"], 6)
        self.assertEqual(
            dispatch["coverage"]["simdgroup_counts"], [1, 2, 4, 8, 16, 32]
        )
        self.assertFalse(dispatch["coverage"]["layer_all_simdgroups_tested"])
        for case in dispatch["cases"]:
            self.assertTrue(case["passed"])
            self.assertEqual(len(case["epochs"]), 2)
            for epoch in case["epochs"]:
                self.assertEqual(epoch["mismatch_count"], 0)
                self.assertTrue(epoch["dirty_sentinel_preserved"])

    def test_compile_failure_is_fail_closed_and_cleans_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_xcrun = root / "xcrun"
            temporary_log = root / "temporary-path.txt"
            _write_fake_xcrun(fake_xcrun)
            with mock.patch.dict(
                os.environ,
                {
                    "OPEN_CAKE_METAL_HIERARCHICAL_FAKE_MODE": "fail_compile",
                    "OPEN_CAKE_METAL_HIERARCHICAL_FAKE_TEMP_LOG": str(temporary_log),
                },
            ):
                with self.assertRaises(PROBE.ProbeFailure) as captured:
                    PROBE.probe(
                        xcrun=fake_xcrun,
                        timeout_seconds=30,
                        _host_system="Darwin",
                        _source_binding_override=_TEST_SOURCE_BINDING,
                    )
            temporary_path = Path(temporary_log.read_text(encoding="utf-8"))

        self.assertEqual(captured.exception.stage, "metal_compile")
        self.assertIn("intentional hierarchical compile failure", str(captured.exception))
        self.assertFalse(temporary_path.exists())

    def _assert_toolchain_failure_cleans(
        self, mode: str, expected_stage: str, expected_message: str
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_xcrun = root / "xcrun"
            temporary_log = root / "temporary-path.txt"
            _write_fake_xcrun(fake_xcrun)
            with mock.patch.dict(
                os.environ,
                {
                    "OPEN_CAKE_METAL_HIERARCHICAL_FAKE_MODE": mode,
                    "OPEN_CAKE_METAL_HIERARCHICAL_FAKE_TEMP_LOG": str(temporary_log),
                },
            ):
                with self.assertRaises(PROBE.ProbeFailure) as captured:
                    PROBE.probe(
                        xcrun=fake_xcrun,
                        timeout_seconds=30,
                        _host_system="Darwin",
                        _source_binding_override=_TEST_SOURCE_BINDING,
                    )
            temporary_path = Path(temporary_log.read_text(encoding="utf-8"))

        self.assertEqual(captured.exception.stage, expected_stage)
        self.assertIn(expected_message, str(captured.exception))
        self.assertFalse(temporary_path.exists())

    def test_link_failure_is_fail_closed_and_cleans_temporary_files(self) -> None:
        self._assert_toolchain_failure_cleans(
            "fail_link", "metallib_link", "intentional hierarchical link failure"
        )

    def test_dispatch_failure_is_fail_closed_and_cleans_temporary_files(self) -> None:
        self._assert_toolchain_failure_cleans(
            "fail_dispatch", "metal_dispatch", "intentional hierarchical dispatch failure"
        )

    def _assert_dispatch_mode_fails(self, mode: str, message: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake_xcrun = Path(directory) / "xcrun"
            _write_fake_xcrun(fake_xcrun)
            with mock.patch.dict(
                os.environ,
                {"OPEN_CAKE_METAL_HIERARCHICAL_FAKE_MODE": mode},
            ):
                with self.assertRaises(PROBE.ProbeFailure) as captured:
                    PROBE.probe(
                        xcrun=fake_xcrun,
                        timeout_seconds=30,
                        _host_system="Darwin",
                        _source_binding_override=_TEST_SOURCE_BINDING,
                    )
        self.assertEqual(captured.exception.stage, "dispatch_output")
        self.assertIn(message, str(captured.exception))

    def test_pipeline_width_drift_cannot_pass(self) -> None:
        self._assert_dispatch_mode_fails("wrong_width", "pipeline width")

    def test_pipeline_limit_below_maximum_case_cannot_pass(self) -> None:
        self._assert_dispatch_mode_fails("low_pipeline_limit", "32-SIMDgroup case")

    def test_static_threadgroup_memory_drift_cannot_pass(self) -> None:
        self._assert_dispatch_mode_fails("static_memory_drift", "unexpected static")

    def test_correctness_mismatch_cannot_pass(self) -> None:
        self._assert_dispatch_mode_fails("mismatch", "correctness mismatch")

    def test_maximum_group_boundary_is_required(self) -> None:
        self._assert_dispatch_mode_fails("missing_boundary", "coverage matrix")

    def test_two_epochs_must_exercise_distinct_reuse_state(self) -> None:
        self._assert_dispatch_mode_fails("same_epoch", "input pattern")

    def test_runner_cannot_mint_a_pass_with_its_own_wrong_oracle(self) -> None:
        self._assert_dispatch_mode_fails("wrong_oracle", "independent probe oracle")

    def test_unknown_sensitive_device_field_is_rejected(self) -> None:
        self._assert_dispatch_mode_fails("sensitive_field", "unknown fields")

    def test_non_darwin_host_is_rejected_before_toolchain_use(self) -> None:
        with self.assertRaises(PROBE.ProbeFailure) as captured:
            PROBE.probe(xcrun="does-not-matter", _host_system="Linux")
        self.assertEqual(captured.exception.stage, "host_preflight")

    def test_cli_failure_is_one_machine_readable_document(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(TOOL), "--xcrun", str(ROOT / "tools/does-not-exist")],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )

        self.assertEqual(completed.returncode, 1)
        self.assertEqual(completed.stdout, b"")
        failure = json.loads(completed.stderr)
        self.assertEqual(failure["status"], "failed")
        self.assertEqual(failure["stage"], "host_preflight")
        self.assertFalse(failure["evaluation_evidence"])
        self.assertFalse(failure["external_workload_oracle_used"])
        self.assertFalse(failure["performance_measured"])
        self.assertFalse(failure["scientific_claim_authorized"])
        self.assertFalse(failure["compiler_change_authorized"])
        self.assertFalse(failure["human_hardware_review_cleared"])
        self.assertFalse(failure["promotion_authorized"])
        self.assertFalse(failure["stable_device_identifiers_retained"])

    @unittest.skipUnless(
        sys.platform == "darwin"
        and shutil.which("xcrun") is not None
        and os.environ.get(
            "OPEN_CAKE_RUN_METAL_HIERARCHICAL_REDUCTION_PROBE"
        )
        == "1",
        (
            "set OPEN_CAKE_RUN_METAL_HIERARCHICAL_REDUCTION_PROBE=1 "
            "on the allowlisted Apple M4 for the real probe"
        ),
    )
    def test_real_hierarchical_reduction_compile_link_and_dispatch(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(TOOL)],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=180,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr.decode())
        self.assertEqual(completed.stderr, b"")
        result = json.loads(completed.stdout)
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["evaluation_evidence"])
        self.assertFalse(result["performance_measured"])
        self.assertFalse(result["scientific_claim_authorized"])
        self.assertTrue(result["temporary_directory_cleaned"])
        dispatch = result["dispatch_observation"]
        self.assertEqual(dispatch["device"]["device_class"], "Apple M4")
        self.assertEqual(dispatch["pipeline"]["thread_execution_width"], 32)
        self.assertEqual(
            dispatch["pipeline"]["max_total_threads_per_threadgroup"], 1024
        )
        self.assertEqual(
            dispatch["coverage"]["simdgroup_counts"], [1, 2, 4, 8, 16, 32]
        )
        for case in dispatch["cases"]:
            self.assertTrue(case["passed"])
            for epoch in case["epochs"]:
                self.assertEqual(epoch["mismatch_count"], 0)


if __name__ == "__main__":
    unittest.main()
