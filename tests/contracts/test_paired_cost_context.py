"""Paired cost calibration must not inherit the single-candidate assay context."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from open_cake_ir.evaluation import LaunchableCandidate
from open_cake_ir.evaluation.paired import candidate_identity
from open_cake_ir.lab.executor import ExecutorRevision
from open_cake_ir.lab.selection import _EmpiricalSelection, _empirical_context, _paired_empirical_context
from open_cake_ir.tasks.workloads import load_workload


ROOT = Path(__file__).resolve().parents[2]


class PairedCostContextTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workload = load_workload(ROOT / "contracts/workloads/gemm-bias-bf16-fp32-v2.json")
        study = json.loads((ROOT / "contracts/studies/matched-search-triton-b300-gemm-optimization-template.json").read_text())
        cls.evaluation = study["evaluation_protocol"]
        cls.executor = ExecutorRevision("fixture", {"target": "sm_103a", "host_environment": {"packages": {"triton": "fixture"}}}, ROOT, "fixture")

    def baseline(self, name="baseline", target="sm_103a"):
        return candidate_identity(LaunchableCandidate(
            candidate_sha256="1" * 64,
            target=target,
            entry_point=name,
            artifact_roles={"cubin": "2" * 64},
            launch_spec_sha256="3" * 64,
        ))

    def context(self, evaluation=None, baseline=None):
        return _paired_empirical_context(
            self.executor, workload_sha256=self.workload.canonical_sha256,
            case_id="primary", evaluation_protocol=evaluation or self.evaluation,
            baseline_identity=baseline or self.baseline(),
        )

    def test_pair_protocol_and_baseline_change_the_model_scope(self):
        paired = self.context()
        single = _empirical_context(
            self.executor, workload_sha256=self.workload.canonical_sha256,
            case_id="primary",
        )
        self.assertNotEqual(paired, single)
        self.assertEqual(paired["cache_protocol"], single["cache_protocol"])
        self.assertEqual(paired["runtime"], single["runtime"])
        self.assertIn("fixed_baseline_paired_cupti_v1", paired["timer"])
        self.assertNotEqual(paired, self.context(baseline=self.baseline("different")))
        changed = copy.deepcopy(self.evaluation)
        changed["paired_timing"]["pair_order"] = changed["paired_timing"]["pair_order"][::-1]
        self.assertNotEqual(paired, self.context(evaluation=changed))

    def test_wrong_assay_or_case_is_refused(self):
        changed = copy.deepcopy(self.evaluation)
        del changed["paired_timing"]
        with self.assertRaisesRegex(ValueError, "fixed-baseline CUPTI case"):
            self.context(evaluation=changed)
        with self.assertRaisesRegex(ValueError, "baseline target differs"):
            self.context(baseline=self.baseline(target="sm_100a"))

    def test_single_assay_model_abstains_under_paired_context(self):
        schedule = json.loads((ROOT / "corpus/schedules/gemm-bias-b1-smoke-b300.json").read_text())
        model = {
            "schema_version": 3, "model_id": "synthetic-single-assay",
            "compiler_revision_id": "synthetic-compiler", "target": "sm_103a",
            "context": _empirical_context(
                self.executor, workload_sha256=self.workload.canonical_sha256,
                case_id="primary",
            ),
            "reported_evidence": {"kind": "synthetic; no GPU measurement"},
            "curves": [{"template": schedule,
                        "varying_dimensions": [{"buffer": name, "dimension": 0} for name in ("a", "c")],
                        "extent_multiple": 1,
                        "points": [{"extent": 512, "kernel_us": 10.0}],
                        "relative_error_envelope": 0.1}],
        }
        selector = _EmpiricalSelection(
            {"kind": "external_empirical_advisory_v1", "model": model},
            context=self.context(), compiler_revision_id="synthetic-compiler",
            target="sm_103a",
        )
        result = selector.estimate(schedule)
        self.assertFalse(result["covered"])
        self.assertIn("model context differs in timer, input_scope", result["reason"])
        changed = copy.deepcopy(self.evaluation)
        changed["case_id"] = "tail"
        with self.assertRaisesRegex(ValueError, "fixed-baseline CUPTI case"):
            self.context(evaluation=changed)


if __name__ == "__main__":
    unittest.main()
