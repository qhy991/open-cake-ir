"""An untimed platform must still evaluate every required input distribution."""

import json
from hashlib import sha256
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.evaluation.core import EvaluationReceipt
from open_cake_ir.evaluation.cuda_driver import CudaDeviceAdmission
from open_cake_ir.evaluation.triton_metax import MetaxDeviceAdmission
from open_cake_ir.tasks import evaluate as worker
from open_cake_ir.tasks.normalization.study import evaluation_policy
from open_cake_ir.tasks.workloads import create_task


class UntimedValidationCases(unittest.TestCase):
    def setUp(self):
        document, _ = create_task("rmsnorm", backend="triton-metax", rows=2, columns=128)
        self.workload = WorkloadContract(document)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.authority = SimpleNamespace(workload=self.workload, manifest=Mock(),
            candidate=SimpleNamespace(candidate_sha256="c" * 64),
            request_root=Path(self.directory.name),
            request={"purpose": "confirmatory", "evaluation_protocol": evaluation_policy(self.workload)})
        self.admission = MetaxDeviceAdmission("maca-123456789abc", "xcore1002", "xcore1002",
            "MetaX C550", 64, "0000:0f:00", "/opt/maca/lib/libmcruntime.so")
        self.result = {"counters": {"module_loads": 0, "kernel_calls": 0, "preflight_calls": 0}}

    def test_a_nonprimary_failure_rejects_the_whole_candidate(self):
        closed, cases = [], []
        def loaded(*args):
            return SimpleNamespace(module_count=1, loaded=SimpleNamespace(launch_calls=1, resources={}),
                                   close=lambda: closed.append(True))
        def evaluate(candidate, workload, protocol, launcher):
            cases.append(protocol.case_id)
            failed = protocol.case_id == "near_zero"
            return SimpleNamespace(correctness_passed=not failed,
                correctness={"output_mismatches": int(failed), "max_abs_error": float(failed),
                             "inputs_unchanged": True})
        with patch.object(worker, "LoadedTorchTensorCandidate", side_effect=loaded), \
             patch.object(worker, "materialize_case", return_value={}), \
             patch.object(worker, "evaluate_tile_validation_case", side_effect=evaluate):
            worker._evaluate_tile_candidate(self.authority, self.result, None, self.admission,
                                            False, route_calls_per_cohort=None)
        self.assertEqual(tuple(cases), self.workload.case_ids)
        self.assertEqual(len(closed), len(cases))
        self.assertFalse(self.result["receipt"]["correctness_passed"])
        self.assertIsNone(self.result["receipt"]["timing"])
        self.assertEqual(self.result["counters"]["kernel_calls"], len(cases))
        report = json.loads((self.authority.request_root / "correctness-output.json").read_text())
        self.assertEqual(report["metrics"]["output_mismatches"], 1)
        self.assertEqual(len(report["validation_cases"]), len(cases))
        launch = json.loads((self.authority.request_root / "launch-receipt.json").read_text())
        self.assertEqual(launch["device_admission"]["pci_bus_id"], "0000:0f:00")
        self.assertEqual(launch["device_admission"]["runtime_library"], self.admission.runtime_library)
        payloads = {role: (self.authority.request_root / name).read_bytes()
                    for role, name in self.result["receipt"]["artifacts"].items()}
        # Exercise the consumer's real custody/role check, not just the worker dict.
        receipt = EvaluationReceipt(self.authority.candidate.candidate_sha256,
            self.workload.canonical_sha256, "a" * 64, "confirmatory", "primary", False,
            self.result["receipt"]["correctness"], 1, 0,
            sha256(payloads["launch_receipt"]).hexdigest(), None, artifact_payloads=payloads)
        self.assertIsNone(json.loads(receipt.artifact_payloads["timing_samples"]))

    def test_a_missing_distribution_is_refused_before_module_loading(self):
        self.authority.request["evaluation_protocol"]["validation_case_ids"] = ["primary"]
        with patch.object(worker, "LoadedTorchTensorCandidate") as loaded:
            with self.assertRaisesRegex(ValueError, "validation cases differ"):
                worker._evaluate_untimed_validation_cases(self.authority, self.result, self.admission)
            loaded.assert_not_called()

    def test_ncu_child_keeps_one_confirmatory_check_and_its_two_artifact_handoff(self):
        document, _ = create_task("rmsnorm", backend="triton-b300", rows=2, columns=128)
        self.authority.workload = WorkloadContract(document)
        self.authority.case_id = "primary"
        self.authority.request = {"purpose": "attribution",
            "evaluation_protocol": evaluation_policy(self.authority.workload)}
        loaded = SimpleNamespace(module_count=1, loaded=SimpleNamespace(launch_calls=1, resources={}), close=Mock())
        receipt = SimpleNamespace(correctness_passed=True, correctness={
            "output_mismatches": 0, "max_abs_error": 0.0, "inputs_unchanged": True})
        admission = CudaDeviceAdmission("NVIDIA B300", (10, 3), "GPU-synthetic",
                                        "gpuq-123456789abc", "exclusive")
        with patch.object(worker, "LoadedTorchTensorCandidate", return_value=loaded), \
             patch.object(worker, "materialize_case", return_value={}), \
             patch.object(worker, "evaluate_tile_workload", return_value=receipt) as evaluate, \
             patch.object(worker, "evaluate_tile_validation_case") as all_cases:
            worker._evaluate_tile_candidate(self.authority, self.result, None, admission,
                                            False, route_calls_per_cohort=42)
        all_cases.assert_not_called()
        evaluate.assert_called_once()
        self.assertEqual(evaluate.call_args.args[2].purpose, "confirmatory")
        self.assertEqual(set(self.result["receipt"]["artifacts"]),
                         {"correctness_output", "launch_receipt"})
        self.assertFalse((self.authority.request_root / "timing-samples.json").exists())
