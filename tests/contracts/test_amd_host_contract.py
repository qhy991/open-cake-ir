from __future__ import annotations

import json
import platform
import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples/gpu"))

from open_cake_ir.lab import ExecutorRevision, HipHostAdmission  # noqa: E402


class HipExecutorFixture:
    def __init__(self, testcase: unittest.TestCase) -> None:
        temporary = tempfile.TemporaryDirectory()
        testcase.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        source = self.root / "runner.py"
        source.write_text("RUNTIME = 'hip'\n", encoding="utf-8")
        tool = self.root / "amd-smi"
        tool.write_bytes(b"#!/bin/sh\nexit 0\n")
        tool.chmod(0o755)
        build_tools = []
        for kind in (
            "cxx",
            "git",
            "hipcc",
            "hipconfig",
            "ninja",
            "rocminfo",
            "sh",
        ):
            path = self.root / kind
            path.write_bytes(f"#!/bin/sh\n# {kind}\nexit 0\n".encode())
            path.chmod(0o755)
            build_tools.append(self.tool_record(kind, path))
        libxml2 = self.root / "libxml2.so.2"
        libxml2.write_bytes(b"ELF libxml2 fixture\n")
        python = Path(sys.executable).absolute()
        self.document: dict[str, object] = {
            "schema_version": 2,
            "executor_id": "open-cake-ir-gfx1151-v1",
            "state": "released",
            "sources": [
                {
                    "path": "runner.py",
                    "sha256": sha256(source.read_bytes()).hexdigest(),
                    "size_bytes": source.stat().st_size,
                }
            ],
            "host_environment": {
                "runtime_kind": "hip",
                "platform": {
                    "system": "Linux",
                    "machine": platform.machine(),
                    "kernel_release": platform.release(),
                },
                "python": {
                    "invocation_path": str(python),
                    "version": sys.version.split()[0],
                    "resolved_sha256": sha256(
                        python.resolve(strict=True).read_bytes()
                    ).hexdigest(),
                },
                "packages": {
                    "packaging": "25.0",
                    "pybind11": "3.0.4",
                    "psutil": "7.2.2",
                    "setuptools": "80.0.0",
                    "torch": "2.9.1",
                    "triton": "3.5.1",
                },
                "runtime": {
                    "backend": "hip",
                    "torch_hip_version": "7.2.1",
                    "visible_device_count": 1,
                },
                "runtime_libraries": [
                    self.library_record("libxml2.so.2", libxml2)
                ],
                "tools": {
                    "build_tools": build_tools,
                    "device_monitor": self.tool_record("amd-smi", tool),
                    "profilers": [],
                },
            },
        }
        self.path = self.root / "runtime/executors/open-cake-ir-gfx1151-v1.json"
        self.write()

    @staticmethod
    def tool_record(kind: str, path: Path) -> dict[str, object]:
        resolved = path.resolve(strict=True)
        payload = resolved.read_bytes()
        return {
            "kind": kind,
            "path": str(resolved),
            "version": "fixture-v1",
            "sha256": sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }

    @staticmethod
    def library_record(soname: str, path: Path) -> dict[str, object]:
        payload = path.resolve(strict=True).read_bytes()
        return {
            "soname": soname,
            "path": str(path.absolute()),
            "sha256": sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.document), encoding="utf-8")

    @property
    def host(self) -> dict[str, object]:
        return self.document["host_environment"]  # type: ignore[return-value]

    def load(self) -> ExecutorRevision:
        return ExecutorRevision.load(self.root, self.path)


class HipExecutorSchemaTests(unittest.TestCase):
    def test_loads_exact_hip_schema_without_reinterpreting_it_as_b200(self) -> None:
        fixture = HipExecutorFixture(self)

        executor = fixture.load()

        self.assertEqual(executor.executor_id, "open-cake-ir-gfx1151-v1")
        with self.assertRaisesRegex(ValueError, "B200 host admission"):
            executor.admit_host()
        with self.assertRaisesRegex(ValueError, "does not pin Nsight Compute"):
            executor.admit_profiler()

    def test_hip_schema_rejects_missing_extra_and_cuda_shaped_facts(self) -> None:
        changes = {
            "extra_cupti": lambda fixture: fixture.host.__setitem__("cupti_python", {}),
            "missing_monitor": lambda fixture: fixture.host["tools"].__delitem__(
                "device_monitor"
            ),
            "missing_build_tool": lambda fixture: fixture.host["tools"][
                "build_tools"
            ].pop(),
            "cuda_backend": lambda fixture: fixture.host["runtime"].__setitem__(
                "backend", "cuda"
            ),
            "multiple_devices": lambda fixture: fixture.host["runtime"].__setitem__(
                "visible_device_count", 2
            ),
            "extra_package": lambda fixture: fixture.host["packages"].__setitem__(
                "numpy", "2.0"
            ),
            "b200_identity": lambda fixture: fixture.document.__setitem__(
                "executor_id", "open-cake-ir-b200-v31"
            ),
        }
        for label, change in changes.items():
            with self.subTest(label=label):
                fixture = HipExecutorFixture(self)
                change(fixture)
                fixture.write()
                with self.assertRaises(ValueError):
                    fixture.load()

    def test_hip_schema_rejects_duplicate_or_unknown_profilers(self) -> None:
        for kinds in (("rocprofv3", "rocprofv3"), ("unknown",)):
            with self.subTest(kinds=kinds):
                fixture = HipExecutorFixture(self)
                profilers = []
                for index, kind in enumerate(kinds):
                    path = fixture.root / f"profiler-{index}"
                    path.write_bytes(f"#!/bin/sh\n# {kind}\n".encode())
                    path.chmod(0o755)
                    profilers.append(fixture.tool_record(kind, path))
                fixture.host["tools"]["profilers"] = profilers  # type: ignore[index]
                fixture.write()
                with self.assertRaisesRegex(ValueError, "profiler kind"):
                    fixture.load()


class HipExecutorAdmissionTests(unittest.TestCase):
    @staticmethod
    def _torch(hip: str | None = "7.2.1", devices: int = 1) -> object:
        return SimpleNamespace(
            version=SimpleNamespace(hip=hip),
            cuda=SimpleNamespace(device_count=lambda: devices),
        )

    def _admit(self, executor: ExecutorRevision, torch: object) -> object:
        versions = {
            "packaging": "25.0",
            "pybind11": "3.0.4",
            "psutil": "7.2.2",
            "setuptools": "80.0.0",
            "torch": "2.9.1",
            "triton": "3.5.1",
        }
        with (
            patch("open_cake_ir.lab.executor.platform.system", return_value="Linux"),
            patch(
                "open_cake_ir.lab.executor.importlib.metadata.version",
                side_effect=lambda name: versions[name],
            ),
            patch(
                "open_cake_ir.lab.executor.importlib.import_module",
                return_value=torch,
            ),
        ):
            return executor.admit_hip_host()

    def test_admits_exact_python_packages_hip_runtime_and_tool_bytes(self) -> None:
        fixture = HipExecutorFixture(self)

        admission = self._admit(fixture.load(), self._torch())

        self.assertEqual(admission.executor_id, "open-cake-ir-gfx1151-v1")
        self.assertEqual(admission.torch_hip_version, "7.2.1")
        self.assertEqual(admission.visible_device_count, 1)
        self.assertEqual(admission.device_monitor["kind"], "amd-smi")
        self.assertEqual(admission.profilers, ())
        self.assertEqual(
            set(admission.build_tools),
            {"cxx", "git", "hipcc", "hipconfig", "ninja", "rocminfo", "sh"},
        )
        self.assertEqual(set(admission.runtime_libraries), {"libxml2.so.2"})

    def test_rejects_cuda_torch_before_any_device_query(self) -> None:
        fixture = HipExecutorFixture(self)
        torch = SimpleNamespace(
            version=SimpleNamespace(hip=None),
            cuda=SimpleNamespace(
                device_count=Mock(side_effect=AssertionError("device queried"))
            ),
        )

        with self.assertRaisesRegex(ValueError, "HIP runtime differs"):
            self._admit(fixture.load(), torch)

        torch.cuda.device_count.assert_not_called()

    def test_rejects_changed_or_non_executable_device_monitor(self) -> None:
        for label, mutate in (
            (
                "changed",
                lambda path: path.write_bytes(b"#!/bin/sh\nexit 1\n"),
            ),
            ("not_executable", lambda path: path.chmod(0o644)),
        ):
            with self.subTest(label=label):
                fixture = HipExecutorFixture(self)
                monitor = Path(
                    fixture.host["tools"]["device_monitor"]["path"]  # type: ignore[index]
                )
                mutate(monitor)
                with self.assertRaisesRegex(ValueError, "device monitor"):
                    self._admit(fixture.load(), self._torch())

    def test_rejects_changed_aiter_build_tool_before_runtime_use(self) -> None:
        fixture = HipExecutorFixture(self)
        git_record = next(
            value
            for value in fixture.host["tools"]["build_tools"]
            if value["kind"] == "git"
        )
        Path(git_record["path"]).write_bytes(b"#!/bin/sh\nexit 7\n")

        with self.assertRaisesRegex(ValueError, "build_tools"):
            self._admit(fixture.load(), self._torch())

    def test_rejects_changed_aiter_runtime_library_before_jit(self) -> None:
        fixture = HipExecutorFixture(self)
        library = fixture.host["runtime_libraries"][0]
        Path(library["path"]).write_bytes(b"changed libxml2 fixture\n")

        with self.assertRaisesRegex(ValueError, "runtime_libraries"):
            self._admit(fixture.load(), self._torch())


class HipHostBoundaryTests(unittest.TestCase):
    _admit = HipExecutorAdmissionTests._admit
    _torch = staticmethod(HipExecutorAdmissionTests._torch)

    def test_public_admission_validates_before_python_or_package_access(self) -> None:
        from open_cake_ir.lab.executor import admit_host_environment
        for host in (None, [], {}, {"runtime_kind": "hip"}):
            with self.subTest(host=host), patch(
                "open_cake_ir.lab.executor.importlib.metadata.version"
            ) as metadata:
                with self.assertRaises(ValueError):
                    admit_host_environment(host)
                metadata.assert_not_called()

    def test_boolean_schema_and_device_count_are_not_integers(self) -> None:
        for field in ("schema_version", "visible_device_count"):
            fixture = HipExecutorFixture(self)
            if field == "schema_version":
                fixture.document[field] = True
            else:
                fixture.host["runtime"][field] = True
            fixture.write()
            with self.subTest(field=field), self.assertRaises(ValueError):
                fixture.load()

    def test_runtime_package_platform_and_python_drift_fail_at_admission(self) -> None:
        for field, mutate in (
            ("Python", lambda f: f.host["python"].__setitem__("version", "other")),
            ("package", lambda f: f.host["packages"].__setitem__("torch", "other")),
            ("platform", lambda f: f.host["platform"].__setitem__("kernel_release", "other")),
            ("HIP runtime", lambda f: f.host["runtime"].__setitem__("torch_hip_version", "other")),
        ):
            fixture = HipExecutorFixture(self)
            mutate(fixture)
            fixture.write()
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                self._admit(fixture.load(), self._torch())

    def test_hip_host_does_not_query_device_and_rejects_cuda_hip_mixture(self) -> None:
        fixture = HipExecutorFixture(self)
        torch = self._torch()
        torch.cuda.device_count = Mock(side_effect=AssertionError("device queried"))
        self._admit(fixture.load(), torch)
        torch.cuda.device_count.assert_not_called()
        torch.version.cuda = "13.0"
        with self.assertRaisesRegex(ValueError, "HIP runtime differs"):
            self._admit(fixture.load(), torch)
        torch.cuda.device_count.assert_not_called()

    def test_hip_host_rejects_non_linux_even_if_descriptor_matches(self) -> None:
        fixture = HipExecutorFixture(self)
        fixture.host["platform"]["system"] = "Darwin"
        fixture.write()
        with self.assertRaisesRegex(ValueError, "platform differs"):
            self._admit(fixture.load(), self._torch())

    def test_duplicate_library_and_build_tool_are_rejected(self) -> None:
        for location in ("library", "build_tool"):
            fixture = HipExecutorFixture(self)
            records = (fixture.host["runtime_libraries"] if location == "library"
                       else fixture.host["tools"]["build_tools"])
            records.append(records[0].copy())
            fixture.write()
            with self.subTest(location=location), self.assertRaises(ValueError):
                fixture.load()


if __name__ == "__main__":
    unittest.main()
