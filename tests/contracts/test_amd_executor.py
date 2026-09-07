from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples/gpu"))

import rmsnorm_amd_search as amd_search  # noqa: E402
from open_cake_ir.lab import HipHostAdmission  # noqa: E402


class HipSearchHostAdmissionTests(unittest.TestCase):
    def test_formal_search_uses_only_the_executor_admitted_monitor(self) -> None:
        admission = HipHostAdmission(
            executor_id="open-cake-ir-gfx1151-v1",
            torch_hip_version="7.2.1",
            visible_device_count=1,
            device_monitor={"path": "/qualified/amd-smi"},
            profilers=(),
            build_tools={},
            runtime_libraries={},
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
            build_tools={},
            runtime_libraries={},
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
            amd_search._failure_class("static_filter"), "AUTHORITY_BLOCKED"
        )
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
                compiler_revision_id="open-cake-ir-test-compiler",
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
                build_tools={},
                runtime_libraries={},
            )
            candidate = SimpleNamespace(
                candidate_id="r1-w1",
                row_tile=1,
                num_warps=1,
                document={},
            )
            noise_decision = SimpleNamespace(
                passed=True,
                observation=SimpleNamespace(
                    measurement_quality_passed=True,
                    pair_wins={"baseline_a": 0, "baseline_b": 0},
                    tied_pairs=4,
                    pooled_sample_counts={"baseline_a": 100, "baseline_b": 100},
                    pooled_medians_ms={"baseline_a": 1.0, "baseline_b": 1.0},
                    speedup=1.0,
                    classification="close_null",
                ),
            )
            artifact_dir = root / "evidence"

            with (
                patch.object(
                    amd_search,
                    "_filter_document",
                    return_value={"ranking_applied": False},
                ),
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
                patch.object(amd_search, "load_workload", return_value=Mock()),
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
                patch.object(
                    amd_search,
                    "_correctness",
                    return_value={"passed": True, "inputs_unchanged": True},
                ),
                patch.object(
                    amd_search,
                    "_noise",
                    return_value=(noise_decision, []),
                ),
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
            build_tools={},
            runtime_libraries={},
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
