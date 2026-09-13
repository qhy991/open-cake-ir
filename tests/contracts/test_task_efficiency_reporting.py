"""Prospective reporting admission and audit wiring; no provider, GPU or custody claims."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import MappingProxyType, ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from open_cake_ir.lab._policies import _ARTIFACT_OPTIMIZATION_ANALYSIS_PLAN
from open_cake_ir.lab.contracts import StudyReport
from open_cake_ir.lab.core import Lab
from open_cake_ir.lab.efficiency_policy import (
    TASK_EFFICIENCY_V1, analysis_without_performance_policy, performance_reporting_policy,
)
from open_cake_ir.lab.endpoints import NORMAL_BUDGET_TERMINAL, analysis_without_endpoint_policy
from open_cake_ir.tasks.runtime import TaskLab
from open_cake_ir.tasks.reporting import primary_summary
from tools import report_task_efficiency

ROOT = Path(__file__).resolve().parents[2]


def report_fixture():
    return StudyReport(study_id="fixture", claim_scope="artifact_optimization_only",
        system_qualification_passed=None, estimand=None, campaign_complete=False,
        archive_integrity_passed=True, filesystem_custody_verified=False,
        semantic_replay_passed=False, estimand_available=False, missing_run_count=1,
        estimate=None, uncertainty=None, descriptive=MappingProxyType({"original": True}),
        run_inclusion=(), run_audits=())


class EfficiencyReportingTests(unittest.TestCase):
    def test_policy_is_closed_and_retains_other_fields_for_plan_validation(self):
        for value in (None, True, {}, "task_efficiency_v2"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "performance reporting policy"):
                performance_reporting_policy({"performance_reporting": value}, "artifact_optimization_only")
        for scope in ("scientific_matched_search", "system_qualification_only", "bounded_local_b200_reconstruction"):
            with self.subTest(scope=scope), self.assertRaisesRegex(ValueError, "requires artifact_optimization_only"):
                performance_reporting_policy({"performance_reporting": TASK_EFFICIENCY_V1}, scope)
        analysis = {**_ARTIFACT_OPTIMIZATION_ANALYSIS_PLAN, "endpoint_policy": NORMAL_BUDGET_TERMINAL,
                    "performance_reporting": TASK_EFFICIENCY_V1}
        normalized = analysis_without_endpoint_policy(analysis_without_performance_policy(analysis, "artifact_optimization_only"))
        self.assertEqual(normalized, _ARTIFACT_OPTIMIZATION_ANALYSIS_PLAN)
        analysis["unexpected"] = 1
        self.assertNotEqual(analysis_without_endpoint_policy(analysis_without_performance_policy(analysis, "artifact_optimization_only")),
                            _ARTIFACT_OPTIMIZATION_ANALYSIS_PLAN)
        analysis["endpoint_policy"] = "unsupported"
        with self.assertRaisesRegex(ValueError, "endpoint policy"):
            analysis_without_endpoint_policy(analysis_without_performance_policy(analysis, "artifact_optimization_only"))

    def test_preflight_refuses_scientific_and_system_policy_before_dependencies(self):
        original = json.loads((ROOT / "contracts/studies/artifact-optimization-ralph-template.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.json"
            for scope in ("scientific_matched_search", "system_qualification_only"):
                document = {**original, "claim_scope": scope,
                    "analysis_plan": {**original["analysis_plan"], "performance_reporting": TASK_EFFICIENCY_V1}}
                path.write_text(json.dumps(document))
                with self.subTest(scope=scope), patch("open_cake_ir.lab.preflight.resolve_execution_bindings") as bindings:
                    with self.assertRaisesRegex(ValueError, "requires artifact_optimization_only"):
                        TaskLab(ROOT).preflight(path)
                    bindings.assert_not_called()

    def test_task_audit_only_enriches_explicit_policy_without_changing_eligibility(self):
        original = report_fixture()
        projection = {"policy": TASK_EFFICIENCY_V1, "rows": [], "missing": ["unverified custody"]}
        module = ModuleType("open_cake_ir.tasks.efficiency")
        module.campaign_performance = Mock(return_value=projection)
        with patch.object(Lab, "audit", return_value=original), patch.dict(sys.modules, {module.__name__: module}):
            lab = TaskLab(ROOT)
            campaign = SimpleNamespace(lock=SimpleNamespace(analysis_plan={}, claim_scope="artifact_optimization_only"))
            self.assertIs(lab.audit(campaign), original)
            module.campaign_performance.assert_not_called()
            campaign.lock.analysis_plan = {"performance_reporting": TASK_EFFICIENCY_V1}
            enriched = lab.audit(campaign)
            module.campaign_performance.assert_called_once_with(ROOT, campaign, original)
        self.assertEqual(enriched.descriptive, {"original": True, "performance": projection})
        self.assertFalse(enriched.filesystem_custody_verified)
        self.assertFalse(enriched.estimand_available)
        self.assertFalse(enriched.campaign_complete)
        self.assertEqual(original.descriptive, {"original": True})

    def test_real_projection_reports_missing_coverage_for_unqualified_audit(self):
        policy = {"performance_reporting": TASK_EFFICIENCY_V1}
        campaign = SimpleNamespace(lock=SimpleNamespace(analysis_plan=policy,
            claim_scope="artifact_optimization_only", workload_id="fixture",
            document={"analysis_plan": policy, "execution": {"target": "apple_gpu_family7"},
                      "evaluation_protocol": {"case_id": "primary"}}))
        with patch.object(Lab, "audit", return_value=report_fixture()):
            report = TaskLab(ROOT).audit(campaign)
        performance = report.descriptive["performance"]
        self.assertEqual(performance["rows"], [])
        self.assertEqual(performance["missing"], ["no_adhered_custody_verified_semantically_replayed_run"])
        self.assertFalse(report.filesystem_custody_verified)
        self.assertFalse(report.estimand_available)

    def test_primary_summary_retains_unavailable_status_and_missing_coverage(self):
        text = primary_summary({"policy": TASK_EFFICIENCY_V1, "rows": [
            {"role": "candidate", "primary_score": {"metric": "mfu", "value": None, "status": "missing_peak"}}],
            "missing": ["no calibrated peak for exact target"]})
        self.assertIn('"status": "missing_peak"', text)
        self.assertIn('"value": null', text)
        self.assertIn("no calibrated peak for exact target", text)

    def test_cli_refuses_historical_backfill_before_audit(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                "open_cake_ir.lab.contracts.CampaignLock.load",
                return_value=SimpleNamespace(analysis_plan={}, claim_scope="artifact_optimization_only")), \
                patch("open_cake_ir.tasks.runtime.TaskLab.audit") as audit:
            with self.assertRaisesRegex(ValueError, "historical backfill"):
                report_task_efficiency.main(["--project-root", str(ROOT), "--workspace", str(Path(directory).resolve())])
            audit.assert_not_called()

    def test_cli_projects_full_audit_and_writes_only_a_new_external_report(self):
        lock = SimpleNamespace(analysis_plan={"performance_reporting": TASK_EFFICIENCY_V1},
                               claim_scope="artifact_optimization_only")
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            output = workspace / "report.json"
            args = ["--project-root", str(ROOT), "--workspace", str(workspace), "--output", str(output)]
            with patch("open_cake_ir.lab.contracts.CampaignLock.load", return_value=lock), \
                    patch.object(TaskLab, "reference_campaign", return_value="campaign") as reference, \
                    patch.object(TaskLab, "audit", return_value=report_fixture()) as audit, \
                    contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(report_task_efficiency.main(args), 0)
                reference.assert_called_once_with(lock, workspace / "campaign-evidence")
                audit.assert_called_once_with("campaign")
            document = json.loads(stdout.getvalue())
            self.assertEqual(document, json.loads(output.read_text()))
            self.assertFalse(document["filesystem_custody_verified"])
            self.assertEqual(document["descriptive"], {"original": True})
            with self.assertRaises(FileExistsError):
                report_task_efficiency.main(args)


class EfficiencyPreflightTests(unittest.TestCase):
    def setUp(self):
        from tests.contracts.test_diagnosis_feedback import DiagnosisRunTests
        DiagnosisRunTests.setUp(self)  # Existing CPU Compiler/Executor interface fixture.

    def test_real_preflight_and_lock_admit_only_the_opted_in_whole_plan(self):
        from hashlib import sha256
        from open_cake_ir.lab._documents import _canonical_json_bytes
        from open_cake_ir.lab.contracts import CampaignLock

        document = json.loads((ROOT / "contracts/studies/artifact-optimization-ralph-template.json").read_text())
        document["analysis_plan"].update(performance_reporting=TASK_EFFICIENCY_V1,
                                         endpoint_policy=NORMAL_BUDGET_TERMINAL)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.json"
            path.write_text(json.dumps(document))
            lock = TaskLab(ROOT).preflight(path)
            self.assertEqual(lock.analysis_plan["performance_reporting"], TASK_EFFICIENCY_V1)
            self.assertEqual(lock.analysis_plan["endpoint_policy"], NORMAL_BUDGET_TERMINAL)
            self.assertEqual(CampaignLock.from_dict(lock.document).analysis_plan, lock.analysis_plan)
            for scope in ("scientific_matched_search", "system_qualification_only"):
                wrong_scope = json.loads(json.dumps(lock.document))
                wrong_scope["study"]["claim_scope"] = scope
                with self.subTest(scope=scope), self.assertRaisesRegex(ValueError, "requires artifact_optimization_only"):
                    CampaignLock.from_dict(wrong_scope)
            document["analysis_plan"]["unexpected_policy"] = "ignored?"
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "artifact optimization Analysis Plan differs"):
                TaskLab(ROOT).preflight(path)
            altered = json.loads(json.dumps(lock.document))
            altered["analysis_plan"]["unexpected_policy"] = "ignored?"
            altered["analysis_plan_sha256"] = sha256(_canonical_json_bytes(altered["analysis_plan"])).hexdigest()
            with self.assertRaisesRegex(ValueError, "artifact optimization Campaign Lock Analysis Plan differs"):
                CampaignLock.from_dict(altered)
