from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.cli import build_parser, main  # noqa: E402
from open_cake_ir.evidence import EvidenceStore  # noqa: E402
from open_cake_ir.lab import CampaignLock, StudyReport  # noqa: E402


class CliContractTests(unittest.TestCase):
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
                        / "contracts/studies/matched-search-system-qualification-v28.json"
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
                        str(ROOT / "contracts/studies/matched-search-infrastructure-v28.json"),
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
            lock = CampaignLock.load(
                ROOT / "runtime/g8-system-r6.campaign.lock.json"
            )
            evidence = EvidenceStore.create(root / "evidence")
            report = StudyReport(
                study_id="artifact-fixture",
                claim_scope="artifact_optimization_only",
                system_qualification_passed=None,
                estimand=None,
                campaign_complete=True,
                archive_integrity_passed=True,
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
                    str(ROOT / "contracts/studies/matched-search-infrastructure-v28.json"),
                ]
            )

        result = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(
            result["study_id"], "open-cake-ir-matched-search-infrastructure-v28"
        )
        self.assertEqual(len(result["campaign_lock_sha256"]), 64)
        self.assertEqual(len(result["run_order"]), 6)


if __name__ == "__main__":
    unittest.main()
