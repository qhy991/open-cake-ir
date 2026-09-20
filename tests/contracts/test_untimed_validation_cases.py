"""An untimed platform must still evaluate every required input distribution."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from open_cake_ir.evaluation.workload import WorkloadContract
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
        self.admission = SimpleNamespace(broker_job_id="maca-123456789abc", gpu_uuid=None)
        self.result = {"counters": {"module_loads": 0, "kernel_calls": 0, "preflight_calls": 0}}

    def test_a_nonprimary_failure_rejects_the_whole_candidate(self):
        closed, cases = [], []
        def loaded(*args):
            return SimpleNamespace(loaded=SimpleNamespace(launch_calls=1, resources={}),
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

    def test_a_missing_distribution_is_refused_before_module_loading(self):
        self.authority.request["evaluation_protocol"]["validation_case_ids"] = ["primary"]
        with patch.object(worker, "LoadedTorchTensorCandidate") as loaded:
            with self.assertRaisesRegex(ValueError, "validation cases differ"):
                worker._evaluate_untimed_validation_cases(self.authority, self.result, self.admission)
            loaded.assert_not_called()
