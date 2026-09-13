"""Task scoring and retained-confirmation projection; no GPU or provider claims."""
import copy
import json
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from open_cake_ir.compiler.target import Peak, PeakRate, PeakSource, Target
from open_cake_ir.evidence import EvidenceStore
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.contraction.workload import workload_document
from open_cake_ir.tasks.efficiency import POLICY, campaign_performance, score_measurement, task_work
from open_cake_ir.tasks.workloads import load_workload

ROOT = Path(__file__).resolve().parents[2]


def target_with_reference(rate=200e9):
    from dataclasses import replace
    target = Target.load(ROOT / "compiler/targets/apple_gpu_family7.json")
    peak = None if rate is None else Peak(PeakRate(rate, PeakSource.DEVICE_SPECIFICATION,
                                                  "2026-09-13"), {})
    return replace(target, peak=peak)


class TaskEfficiencyTests(unittest.TestCase):
    def setUp(self):
        self.document = workload_document("gemm_silu", rows=128, depth=256, columns=32)
        self.workload = WorkloadContract(self.document)

    def test_fixed_task_bytes_preserve_speed_improvement_and_cannot_reward_extra_loads(self):
        work = task_work(self.workload, "primary")
        self.assertEqual(work["logical_io_bytes"], 180352)
        baseline = score_measurement(work, target_with_reference(), 1.878596029,
                                     cache_protocol="warm_no_explicit_flush")
        candidate = score_measurement(work, target_with_reference(), 0.079588216,
                                      cache_protocol="warm_no_explicit_flush")
        self.assertAlmostEqual(baseline["primary_score"]["value"], 0.04800180486)
        self.assertAlmostEqual(candidate["primary_score"]["value"], 1.13303205590)
        self.assertAlmostEqual(candidate["primary_score"]["value"] / baseline["primary_score"]["value"],
                               1.878596029 / 0.079588216)
        self.assertIsNone(candidate["physical_dram_utilization_pct"])
        self.assertIsNone(candidate["arithmetic_efficiency_pct"])
        self.assertIsNone(candidate["roofline_efficiency_pct"])

    def test_unknown_reference_is_unavailable_and_warm_ratio_above_one_is_not_clamped(self):
        work = task_work(self.workload, "primary")
        unknown = score_measurement(work, target_with_reference(None), 1.0,
                                    cache_protocol="warm_no_explicit_flush")
        self.assertIsNone(unknown["primary_score"]["value"])
        self.assertEqual(unknown["primary_score"]["status"], "unavailable")
        observed = score_measurement(work, target_with_reference(), 0.0001,
                                     cache_protocol="warm_no_explicit_flush")
        self.assertGreater(observed["primary_score"]["value"], 100)
        self.assertIsNone(observed["physical_dram_utilization_pct"])
        for time in (0, -1, True, float("nan"), float("inf")):
            with self.subTest(time=time), self.assertRaises(ValueError):
                score_measurement(work, target_with_reference(), time, cache_protocol="unknown")

    def test_abi_dtype_width_and_shape_changes_change_only_task_reference_counts(self):
        document = copy.deepcopy(self.document)
        for tensor in document["tensors"].values():
            tensor["dtype"] = "bf16"
        self.assertEqual(task_work(WorkloadContract(document), "primary")["logical_io_bytes"], 90176)
        document["cases"][0]["shape"]["R"] *= 2
        self.assertGreater(task_work(WorkloadContract(document), "primary")["logical_io_bytes"], 90176)

    def test_confirmation_projection_uses_same_bytes_for_baseline_and_candidate(self):
        # Exercise real append-only storage; the report's semantic result is an
        # explicit interface fixture, not a claim these minimal receipts qualify.
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            with patch.dict("os.environ", {"OPEN_CAKE_CUSTODY_DIRECTORY": str(base / "registry")}):
                source = base / "project"
                source.mkdir()
                workload_path = source / "workload.json"
                workload_path.write_text(json.dumps(self.document))
                workload = load_workload(workload_path)
                target_path = source / "target.json"
                target_document = json.loads((ROOT / "compiler/targets/apple_gpu_family7.json").read_text())
                target_document["peak"] = {"memory_bandwidth": {"bytes_per_second": 200e9,
                    "source": "device_specification", "observed_at": "2026-09-13"}}
                target_path.write_text(json.dumps(target_document))
                (source / "compiler.json").write_text(json.dumps({"target_definitions": {
                    "apple_gpu_family7": {"path": "target.json"}}}))
                store = EvidenceStore.create(base / "evidence")
                run = store.start_run("open_cake-1", authority_sha256=sha256(b"{}").hexdigest(), authority={})
                for purpose, good, stable, latency in (
                    ("search", True, True, 0.00001),
                    ("confirmatory", False, True, 0.00001),
                    ("confirmatory", True, False, 0.00001),
                    ("confirmatory", True, True, 0.079588216),
                ):
                    receipt = store.put(json.dumps({"purpose": purpose, "case_id": "primary",
                        "candidate_sha256": "a" * 64, "correctness_passed": good,
                        "timing": {"measurement_quality_passed": stable,
                            "pooled_medians_ms": {"candidate": latency, "baseline": 1.878596029},
                            "speedup": 1.878596029 / latency}}).encode(), media_type="application/json")
                    run.append("candidate_evaluated", {"purpose": purpose, "turn": 1,
                        "candidate_sha256": "a" * 64, "objects": [receipt.reference("evaluation_receipt")]})
                run.seal(protocol_adherence="adhered", endpoint_observation="observed", endpoint={"fixture": True})
                lock = SimpleNamespace(workload_id=workload.workload_id,
                    claim_scope="artifact_optimization_only", document={
                        "analysis_plan": {"performance_reporting": POLICY},
                        "execution": {"target": "apple_gpu_family7", "fixed_baseline": {
                            "candidate": {"candidate_sha256": "b" * 64}}},
                        "evaluation_protocol": {"case_id": "primary", "paired_timing": {"kind": "fixed_baseline_paired_metal_v2"}},
                        "workload": {"path": "workload.json", "canonical_sha256": workload.canonical_sha256},
                        "compiler_revision": {"path": "compiler.json"}})
                report = SimpleNamespace(run_audits=[store.audit_run("open_cake-1")],
                    descriptive={"semantic_replay_by_run": {"open_cake-1": True}})
                campaign = SimpleNamespace(lock=lock, evidence_root=store.root)
                result = campaign_performance(source, campaign, report)
                self.assertEqual(len(result["rows"]), 2)
                self.assertEqual([row["role"] for row in result["rows"]], ["candidate", "baseline"])
                self.assertAlmostEqual(result["rows"][0]["primary_score"]["value"], 1.13303205590)
                report.descriptive["semantic_replay_by_run"]["open_cake-1"] = False
                self.assertEqual(campaign_performance(source, campaign, report)["rows"], [])
                lock.document["analysis_plan"].clear()
                with self.assertRaisesRegex(ValueError, "not declared"):
                    campaign_performance(source, campaign, report)
