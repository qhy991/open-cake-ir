"""Portable Metal Executor admission/capture contracts; no device dispatch."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from hashlib import sha256

from open_cake_ir.lab.executor import ExecutorRevision, admit_host_environment, admit_profiler_environment
from open_cake_ir.lab.metal_host import validate_metal_host, inspect_metal_host
from open_cake_ir.lab.metal_build import MetalArchiveHost
from tools import capture_executor_host as capture, release_executor


class MetalExecutorContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.archive = self.root / "archive-helper"
        self.observer = self.root / "observer"
        self.swift = self.root / "swiftc"
        for path in (self.archive, self.observer, self.swift):
            path.write_bytes(b"native helper fixture; never executed")
            path.chmod(0o700)
        self.host = {
            "kind": "metal",
            "python": {"invocation_path": str(Path(sys.executable).absolute()), "version": sys.version.split()[0],
                       "resolved_sha256": sha256(Path(sys.executable).resolve().read_bytes()).hexdigest()},
            "packages": {},
            "swift": {"invocation_path": str(self.swift), "version": "fixture Swift version",
                      "resolved_sha256": sha256(self.swift.read_bytes()).hexdigest()},
            "sdk": {"path": str(self.root / "MacOSX.sdk"), "version": "fixture-sdk", "build_version": "fixture-build"},
            "host": {"device_name": "Apple M1 Pro", "device_registry_id": "1234",
                     "operating_system": "fixture OS build", "target": "apple_gpu_family7"},
            "archive_executable": capture._file_record(self.archive, str(self.archive)),
            "observer_executable": capture._file_record(self.observer, str(self.observer)),
        }

    def admission(self, host=None, *, observed=None, sdk=None):
        with patch("open_cake_ir.lab.metal_host.command_text", return_value=self.host["swift"]["version"]), \
             patch("open_cake_ir.lab.metal_host.observe_sdk", return_value=self.host["sdk"] if sdk is None else sdk), \
             patch("open_cake_ir.lab.metal_host.inspect_metal_host", return_value=self.host["host"] if observed is None else observed) as query:
            result = admit_host_environment(self.host if host is None else host)
        return result, query

    def test_native_admission_returns_exact_device_and_bound_observer(self):
        ExecutorRevision._validate_host_document(self.host)
        result, query = self.admission()
        self.assertEqual(result["kind"], "metal")
        self.assertEqual(result["host"], self.host["host"])
        self.assertEqual(result["observer_executable"], str(self.observer))
        self.assertEqual(query.call_args.kwargs["expected_device_names"], ["Apple M1 Pro"])
        self.assertEqual(query.call_args.kwargs["target"], "apple_gpu_family7")
        self.assertEqual(admit_profiler_environment(self.host)["path"], str(self.observer))
        self.assertNotIn("cupti_python", self.host)

    def test_archive_builder_uses_executor_owned_helper_and_toolchain_identity(self):
        calls = []
        executor = SimpleNamespace(document={"host_environment": self.host},
            admit_host=lambda: calls.append("admitted") or {"kind": "metal", "archive_executable": str(self.archive)})
        helper = MetalArchiveHost.from_executor(executor)
        self.assertEqual(helper.executable, self.archive)
        self.assertEqual(helper.toolchain["sdk"], self.host["sdk"])
        self.assertEqual(helper.toolchain["host"], self.host["host"])
        self.assertEqual(calls, ["admitted"])
        changed_host = copy.deepcopy(self.host)
        changed_host["sdk"]["version"] = "new SDK"
        changed = SimpleNamespace(document={"host_environment": changed_host}, admit_host=executor.admit_host)
        self.assertNotEqual(helper.canonical_sha256, MetalArchiveHost.from_executor(changed).canonical_sha256)

    def test_inspection_rejects_dispatch_or_unavailable_profile_capability(self):
        report = {"schema_version": 1, "status": "completed", "dispatches": 0,
                  "stage": "host_admission", "host": self.host["host"],
                  "profiling": {"compute_stage_sampling": True, "gpu_timestamp_counter": True}}
        for change in ({"dispatches": 1}, {"dispatches": False},
                       {"profiling": {"compute_stage_sampling": False, "gpu_timestamp_counter": True}},
                       {"profiling": {"compute_stage_sampling": 1, "gpu_timestamp_counter": True}}):
            with patch.object(MetalArchiveHost, "invoke", return_value={**report, **change}):
                with self.assertRaisesRegex(ValueError, "dispatch-free admission"):
                    inspect_metal_host(self.archive, target="apple_gpu_family7", expected_device_names=["Apple M1 Pro"],
                                       directory=self.root / "inspection")

    def test_closed_native_schema_rejects_cuda_fields_and_missing_facts(self):
        variants = []
        wrong = copy.deepcopy(self.host); wrong["cupti_python"] = {}; variants.append(wrong)
        wrong = copy.deepcopy(self.host); del wrong["observer_executable"]; variants.append(wrong)
        wrong = copy.deepcopy(self.host); wrong["host"]["target"] = "sm_100a"; variants.append(wrong)
        wrong = copy.deepcopy(self.host); wrong["host"]["device_registry_id"] = True; variants.append(wrong)
        wrong = copy.deepcopy(self.host); wrong["sdk"]["guessed_version"] = "1"; variants.append(wrong)
        wrong = copy.deepcopy(self.host); wrong["observer_executable"]["path"] = "relative-observer"; variants.append(wrong)
        for index, host in enumerate(variants):
            with self.subTest(index=index), self.assertRaises(ValueError):
                ExecutorRevision._validate_host_document(host)

    def test_live_device_os_or_selected_sdk_mismatch_refuses(self):
        for field, value in (("device_name", "Apple M2"), ("device_registry_id", "5678"), ("operating_system", "another build")):
            changed = {**self.host["host"], field: value}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "exact device/OS"):
                self.admission(observed=changed)
        with self.assertRaisesRegex(ValueError, "selected SDK"):
            self.admission(sdk={**self.host["sdk"], "version": "changed"})

    def test_changed_observer_refuses_before_any_host_query(self):
        self.observer.write_bytes(b"different executable")
        with patch("open_cake_ir.lab.metal_host.inspect_metal_host") as query:
            with self.assertRaisesRegex(ValueError, "observer executable bytes"):
                self.admission()
            query.assert_not_called()

    def test_native_helpers_inside_an_ancestor_checkout_are_not_admitted(self):
        (self.root / ".git").write_text("gitdir: fixture parent checkout")
        with self.assertRaisesRegex(ValueError, "custody"):
            admit_profiler_environment(self.host)
        with self.assertRaisesRegex(ValueError, "outside the checkout"):
            MetalArchiveHost.build(self.root / "new-output")

    def test_capture_uses_native_admission_without_cuda_profiler(self):
        output = self.root / "captured-host.json"
        args = ["--kind", "metal", "--target", "apple_gpu_family7", "--swiftc", str(self.swift),
                "--archive-executable", str(self.archive), "--observer-executable", str(self.observer), "--output", str(output)]
        with patch.object(capture, "_capture_metal_host", return_value=self.host), \
             patch.object(capture, "_capture_host", side_effect=AssertionError("no CUDA capture")), \
             patch.object(capture, "admit_host_environment", return_value={"kind": "metal"}) as admit, \
             patch.object(capture, "admit_profiler_environment", side_effect=AssertionError("native observer already admitted")):
            self.assertEqual(capture.main(args), 0)
        self.assertEqual(json.loads(output.read_text()), self.host)
        admit.assert_called_once_with(self.host)

    def test_capture_refuses_mixed_host_or_missing_native_binary_before_observation(self):
        args = ["--kind", "metal", "--target", "apple_gpu_family7", "--swiftc", str(self.swift),
                "--archive-executable", str(self.archive), "--observer-executable", str(self.observer),
                "--output", str(self.root / "host.json")]
        with self.assertRaisesRegex(ValueError, "CUDA host fields"):
            capture.main(args + ["--ncu", str(self.observer)])
        self.observer.unlink()
        with patch("open_cake_ir.lab.metal_host.command_text") as command:
            with self.assertRaisesRegex(ValueError, "explicit executable"):
                capture.main(args)
            command.assert_not_called()
        self.assertFalse((self.root / "host.json").exists())

    def test_executor_source_closure_includes_native_assets(self):
        source_root = self.root / "runtime"
        (source_root / "metal").mkdir(parents=True)
        (source_root / "host.py").write_text("pass\n")
        observer = source_root / "metal/observer.swift"
        observer.write_text("// native observation source fixture\n")
        (source_root / "readme.md").write_text("not executable runtime source\n")
        with patch.object(release_executor, "_SOURCE_ROOTS", ("runtime",)), \
             patch.object(release_executor, "_SOURCE_FILES", ()):
            paths = release_executor._source_paths(self.root)
        self.assertEqual(set(paths), {source_root / "host.py", observer})


if __name__ == "__main__":
    unittest.main()
