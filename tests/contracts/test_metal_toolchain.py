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
TOOL = ROOT / "tools/probe_metal_toolchain.py"

_SPEC = importlib.util.spec_from_file_location("open_cake_metal_probe_tool", TOOL)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Metal probe tool could not be imported")
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
            if arguments[:1] == ["--find"]:
                print(executable)
                raise SystemExit(0)
            if arguments == ["--sdk", "macosx", "--show-sdk-path"]:
                print(executable.parent)
                raise SystemExit(0)
            if arguments[:2] != ["--sdk", "macosx"] or len(arguments) < 3:
                print("unexpected fake xcrun arguments", file=sys.stderr)
                raise SystemExit(64)

            stage = arguments[2]
            mode = os.environ.get("OPEN_CAKE_METAL_FAKE_MODE", "passed")
            if stage in {"metal", "metallib"}:
                output = pathlib.Path(arguments[arguments.index("-o") + 1])
                log = os.environ.get("OPEN_CAKE_METAL_FAKE_TEMP_LOG")
                if log:
                    pathlib.Path(log).write_text(str(output.parent), encoding="utf-8")
                if mode == f"fail_{stage}":
                    print(f"intentional {stage} failure", file=sys.stderr)
                    raise SystemExit(7)
                output.write_bytes(
                    b"AIR fixture" if stage == "metal" else b"MTLB fixture"
                )
                raise SystemExit(0)
            if stage == "swift":
                if mode == "malformed_dispatch":
                    print("not-json")
                    raise SystemExit(0)
                observation = {
                    "schema_version": 1,
                    "kind": "open_cake_metal_bfloat_simdgroup_mma_dispatch_v1",
                    "status": "passed",
                    "kernel_calls": 1,
                    "device": {
                        "name": "Apple test double",
                        "registry_id": 1,
                        "supports_apple9": True,
                        "has_unified_memory": True,
                        "max_threadgroup_memory_length": 32768,
                    },
                    "pipeline": {
                        "entry_point": "open_cake_bfloat_simdgroup_mma_probe",
                        "thread_execution_width": 32,
                        "max_total_threads_per_threadgroup": 1024,
                        "static_threadgroup_memory_length": 0,
                    },
                    "dispatch": {
                        "mma_shape": {"m": 8, "n": 8, "k": 8},
                        "input_dtype": "bfloat16",
                        "accumulator_dtype": "float32",
                        "input_pattern": "lhs_identity_rhs_row_major_1_to_64",
                        "expected_outputs": list(range(1, 65)),
                        "observed_outputs": list(range(1, 65)),
                        "output_element_count": 64,
                        "all_outputs_match": True,
                        "threads_dispatched": 32,
                        "threadgroups_dispatched": 1,
                        "command_buffer_status": "completed",
                    },
                }
                if mode == "wrong_dispatch_semantics":
                    observation["dispatch"]["observed_outputs"][-1] = 0
                print(json.dumps(observation, sort_keys=True, separators=(",", ":")))
                raise SystemExit(0)
            print(f"unsupported fake xcrun stage {stage}", file=sys.stderr)
            raise SystemExit(64)
            """
        ),
        encoding="utf-8",
    )
    path.chmod(0o700)


class MetalToolchainContractTests(unittest.TestCase):
    def test_test_double_crosses_compile_link_dispatch_and_cleans_temporary_files(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake_xcrun = Path(directory) / "xcrun"
            _write_fake_xcrun(fake_xcrun)

            result = PROBE.probe(
                xcrun=fake_xcrun,
                timeout_seconds=30,
                _host_system="Darwin",
            )

        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["scientific_claim_authorized"])
        self.assertFalse(result["performance_measured"])
        self.assertTrue(result["temporary_directory_cleaned"])
        self.assertEqual(result["artifacts"]["metallib"]["magic"], "MTLB")
        dispatch = result["dispatch_observation"]["dispatch"]
        self.assertEqual(
            dispatch["input_pattern"],
            "lhs_identity_rhs_row_major_1_to_64",
        )
        self.assertEqual(dispatch["observed_outputs"], list(range(1, 65)))
        self.assertEqual(result["dispatch_observation"]["kernel_calls"], 1)

    def test_compile_failure_is_fail_closed_and_removes_temporary_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_xcrun = root / "xcrun"
            temporary_log = root / "temporary-path.txt"
            _write_fake_xcrun(fake_xcrun)
            with mock.patch.dict(
                os.environ,
                {
                    "OPEN_CAKE_METAL_FAKE_MODE": "fail_metal",
                    "OPEN_CAKE_METAL_FAKE_TEMP_LOG": str(temporary_log),
                },
            ):
                with self.assertRaises(PROBE.ProbeFailure) as captured:
                    PROBE.probe(
                        xcrun=fake_xcrun,
                        timeout_seconds=30,
                        _host_system="Darwin",
                    )
            temporary_path = Path(temporary_log.read_text(encoding="utf-8"))

            self.assertEqual(captured.exception.stage, "metal_compile")
            self.assertIn("intentional metal failure", str(captured.exception))
            self.assertFalse(temporary_path.exists())

    def test_malformed_dispatch_output_cannot_be_reported_as_passed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake_xcrun = Path(directory) / "xcrun"
            _write_fake_xcrun(fake_xcrun)
            with mock.patch.dict(
                os.environ,
                {"OPEN_CAKE_METAL_FAKE_MODE": "malformed_dispatch"},
            ):
                with self.assertRaises(PROBE.ProbeFailure) as captured:
                    PROBE.probe(
                        xcrun=fake_xcrun,
                        timeout_seconds=30,
                        _host_system="Darwin",
                    )

        self.assertEqual(captured.exception.stage, "dispatch_output")

    def test_semantically_failed_dispatch_cannot_be_reported_as_passed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake_xcrun = Path(directory) / "xcrun"
            _write_fake_xcrun(fake_xcrun)
            with mock.patch.dict(
                os.environ,
                {"OPEN_CAKE_METAL_FAKE_MODE": "wrong_dispatch_semantics"},
            ):
                with self.assertRaises(PROBE.ProbeFailure) as captured:
                    PROBE.probe(
                        xcrun=fake_xcrun,
                        timeout_seconds=30,
                        _host_system="Darwin",
                    )

        self.assertEqual(captured.exception.stage, "dispatch_output")

    def test_cli_failure_is_one_machine_readable_document(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--xcrun",
                str(ROOT / "tools/metal_probe/does-not-exist"),
            ],
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
        and os.environ.get("OPEN_CAKE_RUN_METAL_PROBE") == "1",
        "set OPEN_CAKE_RUN_METAL_PROBE=1 on an Apple9 Mac to run the real probe",
    )
    def test_real_bfloat_simdgroup_mma_compile_link_and_dispatch(self) -> None:
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
        self.assertEqual(result["toolchain"]["metal_standard"], "metal3.2")
        self.assertEqual(result["artifacts"]["metallib"]["magic"], "MTLB")
        self.assertTrue(result["temporary_directory_cleaned"])
        dispatch = result["dispatch_observation"]
        self.assertEqual(dispatch["device"]["supports_apple9"], True)
        self.assertEqual(dispatch["pipeline"]["thread_execution_width"], 32)
        self.assertEqual(dispatch["dispatch"]["mma_shape"], {"m": 8, "n": 8, "k": 8})
        self.assertEqual(
            dispatch["dispatch"]["input_pattern"],
            "lhs_identity_rhs_row_major_1_to_64",
        )
        self.assertEqual(
            dispatch["dispatch"]["observed_outputs"],
            list(range(1, 65)),
        )
        self.assertTrue(dispatch["dispatch"]["all_outputs_match"])
        self.assertEqual(dispatch["kernel_calls"], 1)


if __name__ == "__main__":
    unittest.main()
