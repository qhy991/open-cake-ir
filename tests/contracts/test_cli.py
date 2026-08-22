from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.cli import build_parser, main  # noqa: E402
from open_cake_ir.evidence import EvidenceStore  # noqa: E402
from open_cake_ir.lab import CampaignLock  # noqa: E402


class CliContractTests(unittest.TestCase):
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
                        / "contracts/studies/matched-search-system-qualification-v1.json"
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
                        str(ROOT / "contracts/studies/matched-search-infrastructure-v1.json"),
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
        self.assertEqual(result["compiler_revision_id"], "open-cake-ir-sm100a-v3")

    def test_lab_preflight_emits_one_content_bound_campaign_lock(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            code = main(
                [
                    "--project-root",
                    str(ROOT),
                    "lab",
                    "preflight",
                    str(ROOT / "contracts/studies/matched-search-infrastructure-v1.json"),
                ]
            )

        result = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(result["study_id"], "open-cake-ir-matched-search-contract-fixture-v1")
        self.assertEqual(len(result["campaign_lock_sha256"]), 64)
        self.assertEqual(len(result["run_order"]), 6)


if __name__ == "__main__":
    unittest.main()
