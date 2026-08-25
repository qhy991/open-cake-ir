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
TOOL = ROOT / "tools/probe_metal_operators.py"

_SPEC = importlib.util.spec_from_file_location("open_cake_metal_operator_probe", TOOL)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Metal operator probe tool could not be imported")
PROBE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(PROBE)


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
            mode = os.environ.get("OPEN_CAKE_METAL_OPERATOR_FAKE_MODE", "passed")
            if stage == "metal":
                source = pathlib.Path(arguments[arguments.index("-c") + 1])
                output = pathlib.Path(arguments[arguments.index("-o") + 1])
                log = os.environ.get("OPEN_CAKE_METAL_OPERATOR_FAKE_TEMP_LOG")
                if log:
                    pathlib.Path(log).write_text(str(output.parent), encoding="utf-8")
                text = source.read_text(encoding="utf-8")
                required_flags = {
                    "-std=metal3.2",
                    "-fmetal-math-mode=safe",
                    "-ffp-contract=off",
                    "-Werror",
                }
                if not required_flags.issubset(arguments):
                    print("lowering compiler flags were not preserved", file=sys.stderr)
                    raise SystemExit(65)
                if "indexed_gather" in source.name:
                    expected = "kernel void cake_indexed_gather_b8_metal"
                elif "weighted_combine" in source.name:
                    expected = "kernel void cake_kda_weighted_combine_b8_metal"
                else:
                    print("unknown lowered source", file=sys.stderr)
                    raise SystemExit(66)
                if expected not in text or "Schedule SHA256:" not in text:
                    print("source was not materialized by Compiler.lower", file=sys.stderr)
                    raise SystemExit(67)
                if mode == "fail_weighted_compile" and "weighted_combine" in source.name:
                    print("intentional weighted compile failure", file=sys.stderr)
                    raise SystemExit(7)
                output.write_bytes(b"AIR operator fixture")
                raise SystemExit(0)
            if stage == "metallib":
                output = pathlib.Path(arguments[arguments.index("-o") + 1])
                output.write_bytes(b"MTLB operator fixture")
                raise SystemExit(0)
            if stage == "swift":
                launch = json.loads(pathlib.Path(arguments[-1]).read_text(encoding="utf-8"))
                operators = []
                for item in launch["operators"]:
                    count = 1024 if item["kind"] == "indexed_gather" else 128
                    expected = [index & 0xffff for index in range(count)]
                    observed = expected.copy()
                    if mode == "wrong_bits" and item["kind"] == "weighted_combine":
                        observed[-1] ^= 1
                    result = {
                        "output_element_count": count,
                        "expected_bf16_bits": expected,
                        "observed_bf16_bits": observed,
                        "mismatch_count": 0,
                        "all_bits_match": True,
                        "command_buffer_status": "completed",
                    }
                    if item["kind"] == "weighted_combine":
                        result.update({
                            "cpu_oracle_accumulation_order": "route_0_through_7_fp32",
                            "output_conversion": "bfloat16_round_to_nearest_ties_to_even",
                            "bf16_exact_halfway_cases": 2,
                            "bf16_exact_halfway_even_lsb_cases": 1,
                            "bf16_exact_halfway_odd_lsb_cases": 1,
                        })
                    operators.append({
                        "kind": item["kind"],
                        "entry_point": item["entry_point"],
                        "kernel_calls": 1,
                        "pipeline": {
                            "entry_point": item["entry_point"],
                            "thread_execution_width": 16 if mode == "wrong_width" else 32,
                            "max_total_threads_per_threadgroup": 1024,
                            "static_threadgroup_memory_length": 0,
                            "threadgroups_per_grid": item["threadgroups_per_grid"],
                            "threads_per_threadgroup": item["threads_per_threadgroup"],
                            "threadgroup_memory_bytes": item["threadgroup_memory_bytes"],
                        },
                        "result": result,
                    })
                observation = {
                    "schema_version": 1,
                    "kind": "open_cake_metal_operator_dispatch_v1",
                    "status": "passed",
                    "kernel_calls": 2,
                    "device": {
                        "name": "Apple test double",
                        "registry_id": 1,
                        "supports_apple9": True,
                        "has_unified_memory": True,
                        "max_threadgroup_memory_length": 32768,
                    },
                    "input_coverage": {
                        "valid_pairs": 32,
                        "negative_expert_ids": 8,
                        "upper_bound_expert_ids": 8,
                        "negative_row_ids": 8,
                        "upper_bound_row_ids": 8,
                    },
                    "operators": operators,
                }
                print(json.dumps(observation, sort_keys=True, separators=(",", ":")))
                raise SystemExit(0)
            print(f"unsupported fake xcrun stage {stage}", file=sys.stderr)
            raise SystemExit(64)
            """
        ),
        encoding="utf-8",
    )
    path.chmod(0o700)


class MetalOperatorProbeContractTests(unittest.TestCase):
    def test_test_double_lowers_compiles_links_dispatches_and_cleans(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake_xcrun = Path(directory) / "xcrun"
            _write_fake_xcrun(fake_xcrun)

            result = PROBE.probe(
                xcrun=fake_xcrun,
                timeout_seconds=30,
                _host_system="Darwin",
            )

        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["performance_measured"])
        self.assertFalse(result["scientific_claim_authorized"])
        self.assertTrue(result["temporary_directory_cleaned"])
        self.assertEqual(result["toolchain"]["compiled_air_count"], 2)
        self.assertTrue(result["toolchain"]["linked_metallib"])
        self.assertEqual(
            [item["kind"] for item in result["lowerings"]],
            ["indexed_gather", "weighted_combine"],
        )
        dispatch = result["dispatch_observation"]
        self.assertEqual(dispatch["kernel_calls"], 2)
        self.assertEqual(dispatch["input_coverage"]["valid_pairs"], 32)
        for operator in dispatch["operators"]:
            self.assertEqual(operator["kernel_calls"], 1)
            self.assertEqual(operator["pipeline"]["thread_execution_width"], 32)
            self.assertEqual(operator["result"]["mismatch_count"], 0)
            self.assertEqual(operator["result"]["command_buffer_status"], "completed")

    def test_compile_failure_is_fail_closed_and_cleans_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_xcrun = root / "xcrun"
            temporary_log = root / "temporary-path.txt"
            _write_fake_xcrun(fake_xcrun)
            with mock.patch.dict(
                os.environ,
                {
                    "OPEN_CAKE_METAL_OPERATOR_FAKE_MODE": "fail_weighted_compile",
                    "OPEN_CAKE_METAL_OPERATOR_FAKE_TEMP_LOG": str(temporary_log),
                },
            ):
                with self.assertRaises(PROBE.ProbeFailure) as captured:
                    PROBE.probe(
                        xcrun=fake_xcrun,
                        timeout_seconds=30,
                        _host_system="Darwin",
                    )
            temporary_path = Path(temporary_log.read_text(encoding="utf-8"))

            self.assertEqual(captured.exception.stage, "metal_compile_weighted_combine")
            self.assertIn(
                "intentional weighted compile failure", str(captured.exception)
            )
            self.assertFalse(temporary_path.exists())

    def test_bit_mismatch_cannot_be_reported_as_passed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake_xcrun = Path(directory) / "xcrun"
            _write_fake_xcrun(fake_xcrun)
            with mock.patch.dict(
                os.environ,
                {"OPEN_CAKE_METAL_OPERATOR_FAKE_MODE": "wrong_bits"},
            ):
                with self.assertRaises(PROBE.ProbeFailure) as captured:
                    PROBE.probe(
                        xcrun=fake_xcrun,
                        timeout_seconds=30,
                        _host_system="Darwin",
                    )

        self.assertEqual(captured.exception.stage, "dispatch_output")
        self.assertIn("correctness", str(captured.exception))

    def test_pipeline_width_drift_cannot_be_reported_as_passed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake_xcrun = Path(directory) / "xcrun"
            _write_fake_xcrun(fake_xcrun)
            with mock.patch.dict(
                os.environ,
                {"OPEN_CAKE_METAL_OPERATOR_FAKE_MODE": "wrong_width"},
            ):
                with self.assertRaises(PROBE.ProbeFailure) as captured:
                    PROBE.probe(
                        xcrun=fake_xcrun,
                        timeout_seconds=30,
                        _host_system="Darwin",
                    )

        self.assertEqual(captured.exception.stage, "dispatch_output")
        self.assertIn("pipeline/launch", str(captured.exception))

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
        self.assertFalse(failure["scientific_claim_authorized"])
        self.assertFalse(failure["performance_measured"])

    @unittest.skipUnless(
        sys.platform == "darwin"
        and shutil.which("xcrun") is not None
        and os.environ.get("OPEN_CAKE_RUN_METAL_OPERATOR_PROBE") == "1",
        "set OPEN_CAKE_RUN_METAL_OPERATOR_PROBE=1 on an Apple9 Mac for the real probe",
    )
    def test_real_lowered_operators_compile_link_and_dispatch(self) -> None:
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
        self.assertFalse(result["performance_measured"])
        self.assertFalse(result["scientific_claim_authorized"])
        self.assertTrue(result["temporary_directory_cleaned"])
        device = result["dispatch_observation"]["device"]
        self.assertTrue(device["supports_apple9"])
        self.assertTrue(device["name"])
        for operator in result["dispatch_observation"]["operators"]:
            self.assertEqual(operator["kernel_calls"], 1)
            self.assertEqual(operator["result"]["mismatch_count"], 0)
            self.assertTrue(operator["result"]["all_bits_match"])
            self.assertEqual(operator["result"]["command_buffer_status"], "completed")
        weighted = result["dispatch_observation"]["operators"][1]["result"]
        self.assertEqual(weighted["bf16_exact_halfway_cases"], 2)
        self.assertEqual(weighted["bf16_exact_halfway_even_lsb_cases"], 1)
        self.assertEqual(weighted["bf16_exact_halfway_odd_lsb_cases"], 1)


if __name__ == "__main__":
    unittest.main()
