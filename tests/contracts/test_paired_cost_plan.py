"""Synthetic frozen-plan admission; no compilation, GPU or performance result."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import calibrate_paired_cost as instrument

from open_cake_ir.compiler import Compiler
from open_cake_ir.evaluation import LaunchableCandidate
from open_cake_ir.evaluation.core import TensorLaunchManifest
from open_cake_ir.evaluation.paired import candidate_identity
from open_cake_ir.lab.executor import ExecutorRevision
from open_cake_ir.serialization import canonical_json_bytes
from open_cake_ir.tasks.workloads import load_workload


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class PairedCostPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        cls.executor = ExecutorRevision.for_target(ROOT, "sm_103a")
        cls.workload = load_workload(ROOT / "contracts/workloads/gemm-bias-bf16-fp32-v2.json")
        cls.base = json.loads((ROOT / "corpus/schedules/gemm-bias-b1-smoke-b300.json").read_text())

    def fixture(self, directory):
        root = Path(directory).resolve()
        snapshot = root / "candidate"
        baseline_root = snapshot / "baseline"
        manifest = TensorLaunchManifest.for_workload(
            self.workload, "primary", target="sm_103a", kernel_name="cake_gemm_bias_b1_smoke",
            grid=[8, 4, 1], block=[128, 1, 1],
            dynamic_shared_memory_bytes=0, hidden_null_pointer_parameters=2,
        )
        payloads = {"cubin": b"\x7fELF-SYNTHETIC-NONEXECUTABLE",
                    "launch_manifest": canonical_json_bytes(manifest.as_dict())}
        baseline = LaunchableCandidate(
            "1" * 64, "sm_103a", manifest.kernel_name,
            {key: sha256(value).hexdigest() for key, value in payloads.items()},
            manifest.canonical_sha256, payloads,
        )
        paths = {}
        for role, payload in payloads.items():
            path = baseline_root / f"{role}.bin"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            paths[role] = path.name
        bundle = baseline_root / "candidate.json"
        write(bundle, {"candidate": candidate_identity(baseline), "artifact_paths": paths})
        candidates = []
        for cap in (64, 96, 128):
            name = f"tile-{cap}"
            schedule = copy.deepcopy(self.base)
            schedule["schedule_id"] = name
            schedule["residency"]["registers_per_thread"] = cap
            schedule["metadata"]["workload_contract_sha256"] = self.workload.canonical_sha256
            relative = f"schedules/{name}.json"
            write(snapshot / relative, schedule)
            candidates.append({"id": name, "schedule": relative})
        plan = {"schema_version": 1, "state": "frozen",
                "plan_id": "synthetic-paired-plan", "model_id": "synthetic-paired-model",
                "compiler_revision": {"path": "compiler/revision.json",
                                      "revision_id": self.compiler._revision.revision_id},
                "executor_revision": dict(self.executor.reference),
                "study_path": "contracts/studies/matched-search-triton-b300-gemm-optimization-template.json",
                "workload": {"path": "contracts/workloads/gemm-bias-bf16-fp32-v2.json",
                             "workload_id": self.workload.workload_id,
                             "canonical_sha256": self.workload.canonical_sha256},
                "case_id": "primary", "target": "sm_103a",
                "baseline_bundle_path": "baseline/candidate.json", "candidates": candidates,
                "observations": instrument.observation_order([row["id"] for row in candidates]),
                "varying_dimensions": [{"buffer": name, "dimension": 0} for name in ("a", "c")],
                "acceptance": {"maximum_baseline_drift_ratio": 1.05,
                               "maximum_mape": .1, "maximum_relative_error": .2,
                               "maximum_top2_regret_ratio": 1.05,
                               "envelope_allowance": .05}}
        write(snapshot / "plan.json", plan)
        return snapshot, plan

    def test_frozen_b300_study_and_complete_schedule_pool_are_admitted(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, _ = self.fixture(directory)
            checked = instrument.check_plan(snapshot)
            self.assertEqual(checked["target"], "sm_103a")
            self.assertEqual(checked["candidate_count"], 3)
            self.assertEqual(checked["observation_count"], 9)
            self.assertIn("fixed_baseline_paired_cupti_v1", checked["context"]["timer"])

    def test_changed_order_and_schedule_abi_are_refused_before_device(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, original = self.fixture(directory)
            changed = copy.deepcopy(original)
            changed["observations"] = changed["observations"][::-1]
            write(snapshot / "plan.json", changed)
            with self.assertRaisesRegex(ValueError, "observation order"):
                instrument.check_plan(snapshot)
            write(snapshot / "plan.json", original)
            source = snapshot / original["candidates"][0]["schedule"]
            schedule = json.loads(source.read_text())
            next(buffer for buffer in schedule["buffers"] if buffer["name"] == "a")["shape"][0] = 256
            write(source, schedule)
            with self.assertRaises(ValueError):
                instrument.check_plan(snapshot)

    def test_corpus_placeholder_workload_identity_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, plan = self.fixture(directory)
            source = snapshot / plan["candidates"][0]["schedule"]
            schedule = json.loads(source.read_text())
            schedule["metadata"]["workload_contract_sha256"] = "0" * 64
            write(source, schedule)
            with self.assertRaisesRegex(ValueError, "Workload binding"):
                instrument.check_plan(snapshot)

    def test_baseline_record_cannot_name_another_launch_spec(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, plan = self.fixture(directory)
            bundle = snapshot / plan["baseline_bundle_path"]
            document = json.loads(bundle.read_text())
            identity = document["candidate"]
            mismatched = LaunchableCandidate(
                identity["candidate_sha256"], identity["target"], identity["entry_point"],
                identity["artifact_roles"], "4" * 64,
            )
            document["candidate"] = candidate_identity(mismatched)
            write(bundle, document)
            with self.assertRaisesRegex(ValueError, "launch seal differs"):
                instrument.check_plan(snapshot)


if __name__ == "__main__":
    unittest.main()
