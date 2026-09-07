"""The public CPU report preserves diagnostics owned by the Compiler and model."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler  # noqa: E402


_NO_GPU = """import importlib.abc,runpy,sys
class NoGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'triton', 'cuda', 'cupti', 'nvidia'}:
            raise AssertionError('GPU dependency imported: ' + fullname)
sys.meta_path.insert(0, NoGPU())
script = sys.argv.pop(1)
runpy.run_path(script, run_name='__main__')
"""


class ProfileReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def _cli(self, *arguments: str | Path) -> str:
        return subprocess.run(
            [
                sys.executable, "-B", "-c", _NO_GPU,
                str(ROOT / "tools/report_schedule_profile.py"),
                "--revision", str(ROOT / "compiler/revision.json"),
                *(str(argument) for argument in arguments),
            ],
            cwd=ROOT, text=True, capture_output=True, check=True,
        ).stdout

    def _schedule(self, name: str) -> Path:
        return ROOT / "corpus/schedules" / name

    def _model(self) -> dict[str, object]:
        assessment = self.compiler.assess_file(self._schedule("fma-b8-smoke.json"))
        return {
            "schema_version": 2,
            "model_id": "synthetic-report-fixture",
            "compiler_revision_id": assessment.compiler_revision_id,
            "compiler_revision_sha256": assessment.compiler_revision_sha256,
            "target": "sm_100a",
            "context": {
                "timer": "synthetic; no measurement",
                "cache_protocol": "synthetic",
                "runtime": {"compiler_version": "synthetic"},
                "input_scope": "report contract fixture only",
            },
            "reported_evidence": {"kind": "synthetic; no calibration qualification"},
            "curves": [{
                "template": json.loads(assessment.schedule_bytes),
                "varying_dimensions": [
                    {"buffer": buffer, "dimension": 0} for buffer in ("a", "b", "c", "y")
                ],
                "extent_multiple": 8,
                "points": [{"extent": 8, "kernel_us": 10}, {"extent": 16, "kernel_us": 18}],
                "relative_error_envelope": .1,
            }],
        }

    def test_qsa_json_keeps_full_findings_and_canonical_profile(self) -> None:
        schedules = [self._schedule(name) for name in (
            "qsa-score-topk-t32768.json", "qsa-selected-attention-t32768.json",
        )]
        document = json.loads(self._cli("--json", *schedules))
        self.assertEqual(document["skipped"], [])
        self.assertEqual(len(document["rows"]), len(schedules))
        for row, schedule in zip(document["rows"], schedules):
            with self.subTest(schedule=schedule.name):
                assessment = self.compiler.assess_file(schedule)
                self.assertEqual(row["findings"], [asdict(finding) for finding in assessment.findings + assessment.guidance])
                self.assertEqual(row["profile"], self.compiler.profile(assessment).as_dict())
                self.assertTrue(all(not finding["blocks_lowering"] for finding in row["findings"]))

    def test_qsa_text_exposes_locations_coverage_and_qualitative_reasons(self) -> None:
        output = self._cli(
            self._schedule("qsa-score-topk-t32768.json"),
            self._schedule("qsa-selected-attention-t32768.json"),
        )
        self.assertIn("residency: binding=threads; coverage=Schedule declarations", output)
        self.assertIn("RESIDENCY_BOUND at roles (nonblocking)", output)
        self.assertIn("REGISTER_PRESSURE at buffers (nonblocking)", output)
        self.assertIn("barrier (uncalibrated_risk): loop-carried top_k k=512 merge_width=1024", output)
        self.assertIn("scoreboard (uncalibrated_risk): runtime-indexed global buffer k", output)
        self.assertIn("runtime-indexed global buffer v", output)

    def test_accepted_schedule_preserves_advisory_guidance_in_json_and_text(self) -> None:
        schedule = self._schedule("tinygemm2-stage4-split-k.json")
        assessment = self.compiler.assess_file(schedule)
        self.assertTrue(assessment.accepted)
        self.assertTrue(assessment.lowering_eligible)
        self.assertTrue(assessment.guidance)

        document = json.loads(self._cli("--json", schedule))
        self.assertEqual(document["skipped"], [])
        self.assertEqual(len(document["rows"]), 1)
        row = document["rows"][0]
        hints = [finding for finding in row["findings"] if finding["severity"] == "hint"]
        self.assertEqual(hints, [finding.to_dict() for finding in assessment.guidance])
        self.assertTrue(all(not finding["blocks_acceptance"] and not finding["blocks_lowering"]
                            for finding in hints))
        self.assertEqual(row["profile"], self.compiler.profile(assessment).as_dict())
        output = self._cli(schedule)
        for finding in hints:
            self.assertIn(
                f"{finding['code']} at {finding['path']} (nonblocking) "
                f"[{finding['category']}/hint]: {finding['message']}", output,
            )

    def test_skipped_schedule_preserves_blocking_finding_location_and_message(self) -> None:
        schedule = self._schedule("metal-target-unsupported.json")
        assessment = self.compiler.assess_file(schedule)
        self.assertFalse(assessment.lowering_eligible)
        document = json.loads(self._cli("--json", schedule))
        self.assertEqual(document["rows"], [])
        self.assertEqual(document["skipped"][0]["findings"], [asdict(finding) for finding in assessment.findings + assessment.guidance])
        output = self._cli(schedule)
        for finding in assessment.findings:
            self.assertIn(f"{finding.code} at {finding.path}", output)
            self.assertIn(finding.message, output)
        self.assertIn("blocks acceptance, lowering", output)

    def test_conditional_estimate_shows_range_or_specific_noncoverage(self) -> None:
        model = self._model()
        schedules = [self._schedule(name) for name in (
            "fma-b8-smoke.json", "qsa-score-topk-t32768.json",
        )]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(model))
            arguments = ("--cost-model", path, *schedules)
            document = json.loads(self._cli("--json", *arguments))
            output = self._cli(*arguments)
        covered, uncovered = [row["profile"]["empirical_cost"] for row in document["rows"]]
        self.assertEqual(covered["predicted_kernel_us"], 10)
        self.assertEqual(covered["empirical_range_us"], [9, 11])
        self.assertIsNone(uncovered["predicted_kernel_us"])
        self.assertEqual(uncovered["reason"], "no curve covers this exact Schedule and extent")
        for cost in (covered, uncovered):
            self.assertEqual(cost["context"], model["context"])
            self.assertEqual(cost["reported_evidence"], model["reported_evidence"])
        self.assertIn("conditional estimate synthetic-report-fixture: empirical range [9.000, 11.000] us", output)
        self.assertIn("uncovered: " + uncovered["reason"], output)

    def test_metal_keeps_non_nvidia_coverage_and_model_target_refusal(self) -> None:
        schedule = self._schedule("metal-elementwise-odd.json")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(self._model()))
            arguments = ("--cost-model", path, schedule)
            row = json.loads(self._cli("--json", *arguments))["rows"][0]
            output = self._cli(*arguments)
        profile = row["profile"]
        self.assertIsNone(profile["residency"])
        self.assertEqual(profile["ncu_metrics"], [])
        self.assertIsNone(profile["empirical_cost"]["predicted_kernel_us"])
        self.assertIn("METAL_SIMD_EXECUTION at lowering (nonblocking)", output)
        self.assertIn("residency: unavailable", output)
        self.assertIn(profile["abstentions"][0], output)
        self.assertIn("uncovered: " + profile["empirical_cost"]["reason"], output)
        self.assertNotIn("barrier (uncalibrated_risk)", output)


if __name__ == "__main__":
    unittest.main()
