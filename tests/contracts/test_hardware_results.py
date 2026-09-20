"""Presentation must preserve the accepted comparison and its evidence boundaries."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class ReportProjectionTests(unittest.TestCase):
    def fixture(self, root, *, custody=True, include_pair=True):
        report = {
            "archive_integrity_passed": True, "filesystem_custody_verified": custody,
            "semantic_replay_passed": True, "campaign_complete": True,
            "claim_scope": "artifact_optimization_only",
            "run_audits": [{"run_id": "open_cake-1", "protocol_adherence": "adhered",
                "endpoint_observation": "qualified", "endpoint": {
                    "best_candidate_sha256": "chosen", "best_confirmed_latency_ms": .002}}],
            "descriptive": {"performance": {"case_id": "primary", "rows": [
                {"run_id": "open_cake-1", "role": "baseline", "candidate_id": "baseline",
                 "latency_ms": .100, "speedup": 1, "turn": 1, "confirmation_event": 3},
                {"run_id": "open_cake-1", "role": "candidate", "candidate_id": "chosen",
                 "latency_ms": .002, "speedup": 2, "turn": 2, "confirmation_event": 9},
            ]}},
        }
        if include_pair:
            report["descriptive"]["performance"]["rows"].append({
                "run_id": "open_cake-1", "role": "baseline", "candidate_id": "baseline",
                "latency_ms": .004, "speedup": 1, "turn": 2, "confirmation_event": 9})
        lock = {"workload": {"workload_id": "test-fp32-r128-c1024"},
                "compiler_revision": {"revision_id": "test-commit"},
                "execution": {"target": "sm_103a", "fixed_baseline": {
                    "selection": {"source": "starter_reference"}}}}
        (root / "report.json").write_text(json.dumps(report))
        (root / "campaign-lock.json").write_text(json.dumps(lock))

    def test_endpoint_uses_its_own_paired_baseline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            run = module("read_task_results").read_workspace(root)["runs"][0]
        self.assertEqual(run["status"], "reported_qualified")
        self.assertEqual(run["baseline_ms"], .004)
        self.assertEqual(run["paired_speedup"], 2)

    def test_missing_pair_or_failed_audit_never_publishes_qualified_latency(self):
        for arguments in ({"custody": False}, {"include_pair": False}):
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.fixture(root, **arguments)
                run = module("read_task_results").read_workspace(root)["runs"][0]
                self.assertEqual(run["status"], "not_qualified")
                self.assertIsNone(run["candidate_ms"])
                self.assertIsNone(run["paired_speedup"])

    def test_reading_view_does_not_promote_floor_readings_or_hide_failures(self):
        rows = module("render_hardware_results").collect()
        floor = [r for r in rows if r["status"] == "测量分辨能力待查"]
        self.assertTrue(floor)
        self.assertTrue(all(r["speedup"] is None for r in floor))
        failures = [r for r in rows if r["status"] == "未合格终点"]
        self.assertTrue(failures)
        self.assertTrue(all(r["candidate_us"] is None for r in failures))
        self.assertTrue(any(r["speedup"] is not None and r["speedup"] < 1 for r in rows))
        self.assertTrue(any(r["status"] == "有晋升记录" for r in rows))

    def test_generated_document_views_are_current(self):
        for path, expected in module("render_hardware_results").outputs().items():
            with self.subTest(path=path):
                self.assertEqual((ROOT / path).read_text(), expected)


if __name__ == "__main__":
    unittest.main()
