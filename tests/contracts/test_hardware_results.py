"""Presentation must preserve the accepted comparison and its evidence boundaries."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class ReportProjectionTests(unittest.TestCase):
    def fixture(self, root, *, custody=True, include_pair=True, independent=False):
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
        if independent:
            from hashlib import sha256
            from open_cake_ir.serialization import canonical_json_bytes
            lock['run_id'] = 'open_cake-1'
            report = {'run_id':lock['run_id'],'audit':{**report['run_audits'][0],
                'authority_sha256':sha256(canonical_json_bytes(lock)).hexdigest(),
                'archive_integrity':True,'filesystem_custody_verified':custody},
                'replay':{'run_id':lock['run_id'],'refusals':[]},'performance':report['descriptive']['performance']}
        (root / "report.json").write_text(json.dumps(report))
        (root / ('run.json' if independent else "campaign-lock.json")).write_text(json.dumps(lock))

    def test_independent_report_keeps_its_pair_and_refuses_failed_or_mismatched_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp);self.fixture(root,independent=True)
            reader = module('read_task_results')
            row = reader.read_workspace(root)['runs'][0]
            self.assertEqual(row['status'],'reported_qualified')
            self.assertEqual(row['baseline_ms'],.004)
            original = json.loads((root/'report.json').read_text())
            original_authority = json.loads((root/'run.json').read_text())
            for field,value in [('workload',{'workload_id':'different-task'}),
                                ('execution',{**original_authority['execution'],'target':'gfx1151'}),
                                ('compiler_revision',{'revision_id':'different-commit'})]:
                (root/'run.json').write_text(json.dumps({**original_authority,field:value}))
                row = reader.read_workspace(root)['runs'][0]
                self.assertEqual(row['status'],'not_qualified')
                self.assertIsNone(row['candidate_ms'])
            (root/'run.json').write_text(json.dumps(original_authority))
            for replay in ({'run_id':'other-run','refusals':[]},
                           {'run_id':'open_cake-1','refusals':[{'message':'not verified'}]}):
                (root/'report.json').write_text(json.dumps({**original,'replay':replay}))
                row = reader.read_workspace(root)['runs'][0]
                self.assertEqual(row['status'],'not_qualified')
                self.assertIsNone(row['candidate_ms'])
            (root/'campaign-lock.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'two execution authorities'):
                reader.read_workspace(root)

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

    def test_generated_platform_views_are_current(self):
        renderer = module("render_hardware_results")
        for branch in renderer.BRANCHES:
            for path, expected in renderer.outputs(platform=branch).items():
                with self.subTest(path=path):
                    self.assertEqual((ROOT / path).read_text(), expected)

    def test_platform_rendering_only_reads_and_writes_its_own_records(self):
        renderer = module("render_hardware_results")
        original = Path.read_text
        def read(path, *args, **kwargs):
            if path.name == "records.json":
                self.assertEqual(path.parent.name, "nvidia")
            return original(path, *args, **kwargs)
        with patch.object(Path, "read_text", read):
            rendered = renderer.outputs(platform="nvidia")
        self.assertTrue(rendered)
        self.assertTrue(all(path.startswith("docs/results/nvidia/") for path in rendered))

    def test_main_is_union_of_the_platform_records(self):
        renderer = module("render_hardware_results")
        expected = [r for branch in renderer.BRANCHES for r in renderer.collect(branch)]
        self.assertEqual(renderer.collect(), expected)
        self.assertTrue(all("/blob/" in r["source_url"] for r in expected))

    def test_wrong_owner_is_refused_before_publishing(self):
        renderer = module("render_hardware_results")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "docs/results/nvidia/records.json"
            path.parent.mkdir(parents=True)
            data = json.loads((ROOT / "docs/results/nvidia/records.json").read_text())
            data["branch"] = "metal"
            path.write_text(json.dumps(data))
            with patch.object(renderer, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "owning branch"):
                    renderer.collect("nvidia")


if __name__ == "__main__":
    unittest.main()
