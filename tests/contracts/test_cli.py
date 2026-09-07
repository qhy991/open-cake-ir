from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
from io import StringIO
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.cli import build_parser, main  # noqa: E402
from open_cake_ir.evidence import EvidenceStore  # noqa: E402
from open_cake_ir.lab import CampaignLock, Lab, StudyReport  # noqa: E402


class FindingCliContractTests(unittest.TestCase):
    def test_json_and_text_preserve_gate_findings_and_typed_guidance(self) -> None:
        arguments = [
            "--project-root", str(ROOT), "compiler", "assess",
            "--revision", str(ROOT / "compiler/revision.json"),
            str(ROOT / "corpus/schedules/tinygemm2-stage4-split-k.json"),
        ]
        with redirect_stdout(StringIO()) as output:
            self.assertEqual(main(arguments), 0)
        result = json.loads(output.getvalue())
        self.assertTrue(result["accepted"])
        self.assertTrue(result["lowering_eligible"])
        self.assertTrue(result["guidance"])
        self.assertTrue(all(item["severity"] == "report" for item in result["findings"]))
        self.assertEqual({item["category"] for item in result["guidance"]},
                         {"hardware_conformance", "program_safety"})
        for finding in result["guidance"]:
            self.assertEqual(finding["severity"], "hint")
            self.assertFalse(finding["blocks_acceptance"])
            self.assertFalse(finding["blocks_lowering"])
            self.assertTrue(finding["code"])
            self.assertTrue(finding["path"])
        with redirect_stdout(StringIO()) as output:
            self.assertEqual(main(arguments + ["--format", "text"]), 0)
        self.assertIn("hardware_conformance | hint", output.getvalue())
        self.assertIn("未运行 GPU", output.getvalue())

    def test_frontend_rejection_uses_the_same_typed_finding_projection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "invalid.py"
            source.write_text("import os\n", encoding="utf-8")
            with redirect_stdout(StringIO()) as output:
                code = main([
                    "--project-root", str(ROOT), "compiler", "assess",
                    "--revision", str(ROOT / "compiler/revision.json"), str(source),
                ])
        self.assertEqual(code, 2)
        result = json.loads(output.getvalue())
        finding = result["findings"][0]
        self.assertEqual(finding["category"], "schedule_semantics")
        self.assertEqual(finding["severity"], "blocking")
        self.assertTrue(finding["blocks_acceptance"])
        self.assertTrue(finding["blocks_lowering"])
        self.assertEqual(finding["source"]["filename"], str(source.resolve()))
        self.assertEqual(result["guidance"], [])


class CliContractTests(unittest.TestCase):
    def test_preflight_forwards_explicit_empirical_model_to_the_existing_owner(self) -> None:
        lock = Lab(ROOT).preflight(ROOT / "contracts/studies/matched-search-system-qualification-ralph-template.json")
        for binding in (None, Path("/external/bindings.json")):
            with self.subTest(binding=binding):
                arguments = ["lab", "preflight", "/external/study.json",
                             "--empirical-cost-model", "/external/model.json"]
                if binding is not None:
                    arguments.extend(["--execution-bindings", str(binding)])
                with patch("open_cake_ir.cli.Lab.preflight", return_value=lock) as preflight:
                    with redirect_stdout(StringIO()):
                        self.assertEqual(main(arguments), 0)
                preflight.assert_called_once_with(
                    Path("/external/study.json"),
                    empirical_cost_model_path=Path("/external/model.json"),
                    execution_bindings_path=binding,
                )

    def test_compiler_text_keeps_acceptance_lowering_and_diagnostic_impact_separate(self) -> None:
        cases = (
            ("fma-b8-smoke.json", 0, "通过", "允许", "[提示] RESIDENCY_BOUND"),
            ("fma-b8-smoke-arity-drift.json", 2, "未通过", "不允许", "[阻止接受]"),
            ("flash-kmeans-b32-warp-specialized-argmin.json", 0, "通过", "不允许",
             "[阻止生成] TRITON_WARP_SPECIALIZED_ARGMIN_UNSUPPORTED"),
        )
        for schedule, expected_code, accepted, eligible, diagnostic in cases:
            with self.subTest(schedule=schedule), redirect_stdout(StringIO()) as output:
                code = main([
                    "--project-root", str(ROOT), "compiler", "assess", "--format", "text",
                    "--revision", str(ROOT / "compiler/revision.lock.json"),
                    str(ROOT / "corpus/schedules" / schedule),
                ])
            text = output.getvalue()
            self.assertEqual(code, expected_code)
            self.assertIn(f"结构检查：{accepted}\n", text)
            self.assertIn(f"生成代码：{eligible}\n", text)
            self.assertIn(diagnostic, text)
            self.assertIn("未运行 GPU", text)

    def test_compiler_text_lower_preserves_generated_and_checked_source_origins(self) -> None:
        for schedule, origin in (
            ("fma-b8-smoke.json", "由执行计划生成"),
            ("tinygemm2-stage4-split-k.json", "已核验的固定源码"),
        ):
            with self.subTest(schedule=schedule), tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "source.txt"
                arguments = [
                    "--project-root", str(ROOT), "compiler", "lower", "--format", "text",
                    "--revision", str(ROOT / "compiler/revision.lock.json"),
                    str(ROOT / "corpus/schedules" / schedule), "--output", str(target),
                ]
                with redirect_stdout(StringIO()) as output:
                    self.assertEqual(main(arguments), 0)
                self.assertIn(f"来源：{origin}", output.getvalue())
                self.assertIn(str(target), output.getvalue())
                self.assertTrue(target.read_bytes())
                original = target.read_bytes()
                with redirect_stderr(StringIO()) as error:
                    self.assertEqual(main(arguments), 2)
                self.assertIn("命令未完成", error.getvalue())
                self.assertEqual(target.read_bytes(), original)

    def test_compiler_text_reports_input_errors_without_a_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            malformed = root / "malformed.json"
            malformed.write_text("{broken", encoding="utf-8")
            for project_root, schedule in (
                (ROOT, root / "missing.json"),
                (ROOT, malformed),
                (root / "missing-root", malformed),
            ):
                arguments = [
                    "--project-root", str(project_root), "compiler", "assess", "--format", "text",
                    "--revision", str(ROOT / "compiler/revision.lock.json"), str(schedule),
                ]
                with self.subTest(schedule=schedule, root=project_root), \
                     redirect_stdout(StringIO()) as output, redirect_stderr(StringIO()) as error:
                    self.assertEqual(main(arguments), 2)
                self.assertEqual(output.getvalue(), "")
                self.assertIn("命令未完成", error.getvalue())
                self.assertNotIn("Traceback", error.getvalue())

    def test_compiler_text_corpus_reports_expectation_matches_not_gpu_acceptance(self) -> None:
        with redirect_stdout(StringIO()) as output:
            code = main([
                "--project-root", str(ROOT), "compiler", "check-corpus", "--format", "text",
                "--revision", str(ROOT / "compiler/revision.lock.json"),
            ])
        count = len(json.loads((ROOT / "corpus/manifest.json").read_text())["cases"])
        self.assertEqual(code, 0)
        self.assertIn(f"{count}/{count} 项符合预期", output.getvalue())
        self.assertIn("不是 GPU 正确性检查", output.getvalue())

    def test_lab_execute_refuses_evidence_inside_the_checkout_before_loading_inputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix=".campaign-custody-", dir=ROOT) as directory:
            scratch = Path(directory)
            evidence_root = scratch / "evidence"

            with self.assertRaisesRegex(ValueError, "outside the project checkout"):
                main(
                    [
                        "--project-root",
                        str(ROOT),
                        "lab",
                        "execute",
                        "--lock",
                        str(scratch / "missing.lock.json"),
                        "--runtime-config",
                        str(scratch / "missing.runtime.json"),
                        "--evidence-root",
                        str(evidence_root),
                    ]
                )

            self.assertFalse(evidence_root.exists())

    def test_lab_preflight_refuses_a_campaign_lock_inside_the_checkout(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".campaign-custody-", dir=ROOT) as directory:
            lock_path = Path(directory) / "campaign.lock.json"

            with self.assertRaisesRegex(ValueError, "outside the project checkout"):
                with redirect_stdout(StringIO()):
                    main(
                        [
                            "--project-root",
                            str(ROOT),
                            "lab",
                            "preflight",
                            str(
                                ROOT
                                / "contracts/studies/matched-search-infrastructure-v2.json"
                            ),
                            "--output",
                            str(lock_path),
                        ]
                    )

            self.assertFalse(lock_path.exists())

    def test_system_qualification_preflight_projects_non_scientific_policy(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            code = main(
                [
                    "--project-root",
                    str(ROOT),
                    "lab",
                    "preflight",
                    str(
                        ROOT
                        / "contracts/studies/matched-search-system-qualification-ralph-template.json"
                    ),
                ]
            )

        result = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(result["claim_scope"], "system_qualification_only")
        self.assertIsNone(result["estimand"])
        self.assertEqual(result["run_order"], ["open_cake-1", "direct_cuda-1"])

    def test_lab_execute_is_a_real_public_command(self) -> None:
        parsed = build_parser().parse_args(
            [
                "lab",
                "execute",
                "--lock",
                "campaign.lock.json",
                "--runtime-config",
                "runtime.json",
                "--evidence-root",
                "evidence",
            ]
        )

        self.assertEqual(parsed.lab_command, "execute")
        self.assertEqual(parsed.runtime_config, Path("runtime.json"))

    def test_lab_preflight_lock_can_be_reloaded_and_audited(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            lock_path = root / "campaign.lock.json"
            with redirect_stdout(StringIO()):
                preflight_code = main(
                    [
                        "--project-root",
                        str(ROOT),
                        "lab",
                        "preflight",
                        str(ROOT / "contracts/studies/matched-search-infrastructure-template.json"),
                        "--output",
                        str(lock_path),
                    ]
                )
            lock = CampaignLock.load(lock_path)
            evidence = EvidenceStore.create(root / "evidence")
            for run_id in lock.run_order:
                run = evidence.start_run(
                    run_id,
                    authority_sha256=lock.canonical_sha256,
                    authority=lock.document,
                )
                run.seal(
                    protocol_adherence="provider_fault",
                    endpoint_observation="missing",
                )
            output = StringIO()
            with redirect_stdout(output):
                audit_code = main(
                    [
                        "--project-root",
                        str(ROOT),
                        "lab",
                        "audit",
                        "--lock",
                        str(lock_path),
                        "--evidence-root",
                        str(evidence.root),
                    ]
                )

        report = json.loads(output.getvalue())
        self.assertEqual(preflight_code, 0)
        self.assertEqual(audit_code, 0)
        self.assertTrue(report["campaign_complete"])
        self.assertTrue(report["archive_integrity_passed"])
        self.assertFalse(report["estimand_available"])
        self.assertEqual(report["missing_run_count"], 6)

    def test_artifact_report_with_read_only_promoted_record_is_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            lock = Lab(ROOT).preflight(ROOT / "contracts/studies/matched-search-system-qualification-ralph-template.json")
            evidence = EvidenceStore.create(root / "evidence")
            report = StudyReport(
                study_id="artifact-fixture",
                claim_scope="artifact_optimization_only",
                system_qualification_passed=None,
                estimand=None,
                campaign_complete=True,
                archive_integrity_passed=True,
                filesystem_custody_verified=True,
                semantic_replay_passed=True,
                estimand_available=False,
                missing_run_count=0,
                estimate=None,
                uncertainty=None,
                descriptive=MappingProxyType(
                    {
                        "artifact_optimization_complete": True,
                        "promoted_artifacts": MappingProxyType(
                            {
                                "open_cake-1": MappingProxyType(
                                    {
                                        "turn": 1,
                                        "candidate_sha256": "a" * 64,
                                    }
                                )
                            }
                        ),
                    }
                ),
                run_inclusion=(),
                run_audits=(),
            )
            output = StringIO()
            with (
                patch("open_cake_ir.cli.CampaignLock.load", return_value=lock),
                patch("open_cake_ir.cli.Lab.reference_campaign", return_value=object()),
                patch("open_cake_ir.cli.Lab.audit", return_value=report),
                redirect_stdout(output),
            ):
                code = main(
                    [
                        "--project-root",
                        str(ROOT),
                        "lab",
                        "audit",
                        "--lock",
                        str(root / "lock.json"),
                        "--evidence-root",
                        str(evidence.root),
                    ]
                )

        emitted = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertTrue(emitted["archive_integrity_passed"])
        self.assertTrue(emitted["filesystem_custody_verified"])
        self.assertEqual(
            emitted["descriptive"]["promoted_artifacts"]["open_cake-1"]["turn"],
            1,
        )

    def test_compiler_assess_exposes_the_released_interface(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            code = main(
                [
                    "--project-root",
                    str(ROOT),
                    "compiler",
                    "assess",
                    "--revision",
                    str(ROOT / "compiler/revision.lock.json"),
                    str(ROOT / "corpus/schedules/flash-kmeans-assignment-full.json"),
                ]
            )

        result = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertTrue(result["accepted"])
        self.assertTrue(result["lowering_eligible"])
        released = json.loads(
            (ROOT / "compiler" / "revision.lock.json").read_text(encoding="utf-8")
        )
        self.assertEqual(result["compiler_revision_id"], released["revision_id"])

    def test_compiler_lower_exposes_whether_the_schedule_generated_the_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "lowered.py"
            output = StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "--project-root",
                        str(ROOT),
                        "compiler",
                        "lower",
                        "--revision",
                        str(ROOT / "compiler/revision.lock.json"),
                        str(ROOT / "corpus/schedules/flash-kmeans-assignment-full.json"),
                        "--output",
                        str(target),
                    ]
                )

            result = json.loads(output.getvalue())
            self.assertEqual(code, 0)
            self.assertTrue(result["generated"])
            self.assertEqual(
                result["source_sha256"], sha256(target.read_bytes()).hexdigest()
            )

    def test_lab_preflight_emits_one_content_bound_campaign_lock(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            code = main(
                [
                    "--project-root",
                    str(ROOT),
                    "lab",
                    "preflight",
                    str(ROOT / "contracts/studies/matched-search-infrastructure-template.json"),
                ]
            )

        result = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(
            result["study_id"], "open-cake-ir-matched-search-infrastructure-template"
        )
        self.assertEqual(len(result["campaign_lock_sha256"]), 64)
        self.assertEqual(len(result["run_order"]), 6)


if __name__ == "__main__":
    unittest.main()
