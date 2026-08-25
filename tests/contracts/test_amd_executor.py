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

import rmsnorm_amd_search as amd_search  # noqa: E402
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
                    "system": platform.system(),
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
                "packages": {"torch": "2.9.1", "triton": "3.5.1"},
                "runtime": {
                    "backend": "hip",
                    "torch_hip_version": "7.2.1",
                    "visible_device_count": 1,
                },
                "tools": {
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
        versions = {"torch": "2.9.1", "triton": "3.5.1"}
        with (
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


class HipSearchHostAdmissionTests(unittest.TestCase):
    def test_formal_search_uses_only_the_executor_admitted_monitor(self) -> None:
        admission = HipHostAdmission(
            executor_id="open-cake-ir-gfx1151-v1",
            torch_hip_version="7.2.1",
            visible_device_count=1,
            device_monitor={"path": "/qualified/amd-smi"},
            profilers=(),
        )
        executor = Mock()
        executor.admit_hip_host.return_value = admission
        snapshot = {
            "available": True,
            "returncode": 0,
            "document": [
                {
                    "process_list": [
                        {"process_info": "No running processes detected"}
                    ]
                }
            ],
        }

        with patch.object(amd_search, "_amd_smi", return_value=snapshot) as monitor:
            observed, process = amd_search._admit_search_host(executor)

        executor.admit_hip_host.assert_called_once_with()
        monitor.assert_called_once_with("/qualified/amd-smi", ["process"])
        self.assertIs(observed, admission)
        self.assertIs(process, snapshot)

    def test_formal_search_refuses_an_unobservable_process_set(self) -> None:
        admission = HipHostAdmission(
            executor_id="open-cake-ir-gfx1151-v1",
            torch_hip_version="7.2.1",
            visible_device_count=1,
            device_monitor={"path": "/qualified/amd-smi"},
            profilers=(),
        )
        executor = Mock()
        executor.admit_hip_host.return_value = admission

        with (
            patch.object(
                amd_search,
                "_amd_smi",
                return_value={"available": False},
            ),
            self.assertRaisesRegex(RuntimeError, "unobservable compute process"),
        ):
            amd_search._admit_search_host(executor)


class HipSearchFailureEvidenceTests(unittest.TestCase):
    def test_failure_classes_preserve_the_next_action(self) -> None:
        self.assertEqual(amd_search._failure_class("source_custody"), "CUSTODY_BLOCKED")
        self.assertEqual(
            amd_search._failure_class("runtime_admission"),
            "ENVIRONMENT_BLOCKED",
        )
        self.assertEqual(
            amd_search._failure_class("baseline_correctness_rejected"),
            "CORRECTNESS_REJECTED",
        )
        self.assertEqual(
            amd_search._failure_class("candidate_correctness"),
            "HARNESS_FAULT",
        )

    def test_candidate_runtime_fault_terminates_run_with_bound_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            paths = {
                name: root / f"{name}.json"
                for name in ("search", "compiler", "executor", "workload")
            }
            for path in paths.values():
                path.write_text("{}", encoding="utf-8")
            contract = SimpleNamespace(
                search_id="gfx1151-search-fixture",
                canonical_sha256="1" * 64,
                path=paths["search"],
                compiler_path=paths["compiler"],
                compiler_revision_id="open-cake-ir-v29",
                compiler_sha256="2" * 64,
                executor_path=paths["executor"],
                executor_id="open-cake-ir-gfx1151-v1",
                executor_sha256="3" * 64,
                workload_path=paths["workload"],
                workload_sha256="4" * 64,
                screening=SimpleNamespace(l2_flush_bytes=1024),
            )
            baseline = Mock()
            baseline.cleanup = Mock()
            baseline_requirements = {
                "triton_target": {
                    "backend": "hip",
                    "arch": "gfx1151",
                    "warp_size": 32,
                }
            }
            compiler = Mock()
            compiler.assess.return_value = object()
            compiler.lower.return_value = SimpleNamespace(
                toolchain_requirements=baseline_requirements
            )
            torch = SimpleNamespace(
                zeros=Mock(return_value=object()),
                float32=object(),
                cuda=SimpleNamespace(synchronize=Mock()),
            )
            host = HipHostAdmission(
                executor_id=contract.executor_id,
                torch_hip_version="7.2.1",
                visible_device_count=1,
                device_monitor={"path": "/qualified/amd-smi"},
                profilers=(),
            )
            candidate = SimpleNamespace(
                candidate_id="r1-w1",
                row_tile=1,
                num_warps=1,
                document={},
            )
            artifact_dir = root / "evidence"

            with (
                patch.object(
                    amd_search,
                    "git_state",
                    return_value={"revision": "abc", "tree_clean": True},
                ),
                patch.object(
                    amd_search,
                    "_admit_search_host",
                    return_value=(host, {"available": True, "returncode": 0}),
                ),
                patch.object(
                    amd_search,
                    "_canonical_document",
                    return_value=({}, "5" * 64),
                ),
                patch.object(
                    amd_search,
                    "admit_exact_hip",
                    return_value=(torch, object(), object()),
                ),
                patch.object(amd_search, "_runtime_document", return_value={}),
                patch.object(amd_search.WorkloadContract, "load", return_value=Mock()),
                patch.object(
                    amd_search,
                    "_case_materials",
                    return_value={"seeded_random": object()},
                ),
                patch.object(
                    amd_search,
                    "_compile_candidate",
                    side_effect=[baseline, RuntimeError("HIP reset")],
                ),
                patch.object(amd_search, "_correctness", return_value={"passed": True}),
                patch.object(amd_search, "_screen") as screen,
                self.assertRaisesRegex(RuntimeError, "HIP reset"),
            ):
                amd_search._run(
                    root=root,
                    contract=contract,
                    compiler=compiler,
                    executor=SimpleNamespace(
                        reference={
                            "path": paths["executor"].relative_to(root).as_posix(),
                            "executor_id": contract.executor_id,
                            "canonical_sha256": contract.executor_sha256,
                        }
                    ),
                    candidate_specs=(candidate,),
                    artifact_dir=artifact_dir,
                )

            screen.assert_not_called()
            baseline.cleanup.assert_called_once_with()
            failure = json.loads(
                (artifact_dir / "failure.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failure["failed_stage"], "candidate_correctness")
            self.assertEqual(failure["failure_class"], "HARNESS_FAULT")
            self.assertEqual(failure["status"], "HARNESS_FAULT")
            self.assertEqual(
                failure["authority"]["executor"]["executor_id"],
                "open-cake-ir-gfx1151-v1",
            )
            self.assertFalse(failure["performance_conclusion_authorized"])
            self.assertTrue((artifact_dir / "attempt-authority.json").is_file())
            self.assertTrue((artifact_dir / "manifest.json").is_file())

    def test_formal_search_refuses_a_failed_monitor_with_residual_json(self) -> None:
        admission = HipHostAdmission(
            executor_id="open-cake-ir-gfx1151-v1",
            torch_hip_version="7.2.1",
            visible_device_count=1,
            device_monitor={"path": "/qualified/amd-smi"},
            profilers=(),
        )
        executor = Mock()
        executor.admit_hip_host.return_value = admission
        failed = {
            "available": True,
            "returncode": 7,
            "document": [
                {
                    "process_list": [
                        {"process_info": "No running processes detected"}
                    ]
                }
            ],
        }

        with (
            patch.object(amd_search, "_amd_smi", return_value=failed),
            self.assertRaisesRegex(RuntimeError, "unobservable compute process"),
        ):
            amd_search._admit_search_host(executor)


if __name__ == "__main__":
    unittest.main()
