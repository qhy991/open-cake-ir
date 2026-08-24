from __future__ import annotations

import grp
import json
import os
import pwd
import shutil
import subprocess
import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.lab import Lab, ProviderQualificationReceipt  # noqa: E402
from tools.freeze_live_matched_study import (  # noqa: E402
    _replace_artifact_feedback_budget,
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


class FreezeLiveMatchedStudyContractTests(unittest.TestCase):
    def test_artifact_feedback_budget_has_one_explicit_horizon(self) -> None:
        study: dict[str, object] = {
            "claim_scope": "artifact_optimization_only",
            "budget": {
                "unit": "provider_tokens",
                "limit": 150_000,
                "checkpoints": [50_000, 100_000, 150_000],
                "maximum_turns": 4,
                "maximum_candidates_per_turn": 3,
            },
        }

        _replace_artifact_feedback_budget(
            study,
            provider_token_limit=8_000_000,
            maximum_turns=2,
        )

        self.assertEqual(
            study["budget"],
            {
                "unit": "provider_tokens",
                "limit": 8_000_000,
                "checkpoints": [8_000_000],
                "maximum_turns": 2,
                "maximum_candidates_per_turn": 3,
            },
        )

    def test_feedback_budget_replacement_is_complete_and_artifact_only(self) -> None:
        artifact = {
            "claim_scope": "artifact_optimization_only",
            "budget": {},
        }
        with self.assertRaisesRegex(ValueError, "declared together"):
            _replace_artifact_feedback_budget(
                artifact,
                provider_token_limit=8_000_000,
                maximum_turns=None,
            )

        system = {"claim_scope": "system_qualification_only", "budget": {}}
        with self.assertRaisesRegex(ValueError, "artifact-optimization only"):
            _replace_artifact_feedback_budget(
                system,
                provider_token_limit=8_000_000,
                maximum_turns=2,
            )

    def test_live_authorities_freeze_one_non_scientific_run_per_arm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            project = temporary / "open-cake-ir"

            def ignored(path: str, names: list[str]) -> set[str]:
                omitted = {"__pycache__", ".pytest_cache"}
                if Path(path).resolve() == ROOT:
                    omitted |= {"evidence", "migration", "tests"}
                return omitted & set(names)

            shutil.copytree(
                ROOT,
                project,
                ignore=ignored,
            )
            executable = project / "codex-fixture"
            executable.write_bytes(b"qualified codex fixture")
            executable.chmod(0o700)
            fixture = json.loads(
                (project / "contracts/providers/fixture-provider-v1.json").read_text()
            )
            qualification = ProviderQualificationReceipt(
                provider_revision="codex-live-contract-fixture",
                executable_sha256=sha256(executable.read_bytes()).hexdigest(),
                configuration_sha256=fixture["configuration_sha256"],
                initial_and_resume_equivalent=True,
                file_lifecycle_observed=True,
                usage_observed=True,
                qualified=True,
                scope="live_two_turn_current_provider",
            )
            qualification_path = project / "contracts/providers/live-fixture.json"
            qualification_path.write_bytes(_canonical_json_bytes(qualification.document) + b"\n")
            anchor = {
                "schema_version": 1,
                "kind": "codex_provider_qualification_evidence_anchor",
                "run_id": "codex-live-contract-fixture",
                "evidence_root": "/external/fixture/evidence",
                "authority_sha256": "a" * 64,
                "qualification_receipt_sha256": qualification.canonical_sha256,
                "immediate_audit_integrity": True,
                "terminal_seal_sha256": "b" * 64,
            }
            anchor_path = project / "contracts/providers/live-fixture-anchor.json"
            anchor_path.write_bytes(_canonical_json_bytes(anchor) + b"\n")
            nvcc = project / "nvcc-fixture"
            cuobjdump = project / "cuobjdump-fixture"
            nvcc.write_bytes(b"nvcc")
            cuobjdump.write_bytes(b"cuobjdump")
            runtime = {
                "schema_version": 1,
                "provider": {
                    "executable": str(executable),
                    "workspace_root": str(temporary / "future-workspaces"),
                },
                "toolchain": {"nvcc": str(nvcc), "cuobjdump": str(cuobjdump)},
                "broker": {
                    "command": [
                        sys.executable,
                        str(project / "tools/evaluate_flash_candidate.py"),
                    ],
                    "cwd": str(project),
                    "timeout_seconds": 1800,
                    "service_user": pwd.getpwuid(os.geteuid()).pw_name,
                    "service_group": grp.getgrgid(os.getegid()).gr_name,
                },
            }
            runtime_path = temporary / "runtime.json"
            runtime_path.write_bytes(_canonical_json_bytes(runtime))
            output = project / "contracts/studies/live-system-qualification.json"
            command = [
                sys.executable,
                str(project / "tools/freeze_live_matched_study.py"),
                "--project-root",
                str(project),
                "--template",
                str(project / "contracts/studies/matched-search-system-qualification-v6.json"),
                "--qualification",
                str(qualification_path),
                "--qualification-anchor",
                str(anchor_path),
                "--executor",
                str(
                    project
                    / json.loads(
                        (project / "inventory/EXECUTOR_REVISIONS.json").read_text(
                            encoding="utf-8"
                        )
                    )["current"]["path"]
                ),
                "--runtime-config",
                str(runtime_path),
                "--study-id",
                "open-cake-ir-live-g8-contract-fixture",
                "--output",
                str(output),
                "--enable-attribution",
            ]

            completed = subprocess.run(
                command,
                cwd=project,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr.decode())
            lock = Lab(project).preflight(output)
            self.assertEqual(lock.claim_scope, "system_qualification_only")
            self.assertEqual(lock.run_order, ("open_cake-1", "direct_cuda-1"))
            self.assertIsNone(lock.estimand)
            self.assertEqual(output.stat().st_mode & 0o444, 0o444)
            study = json.loads(output.read_text())
            self.assertEqual(
                study["evaluation_protocol"]["attribution_evaluation"],
                "correctness_then_profile",
            )
            self.assertEqual(study["arms"]["open_cake"]["feedback"][-1], "profile")
            self.assertEqual(study["arms"]["direct_cuda"]["feedback"][-1], "profile")
            self.assertEqual(
                study["arms"]["open_cake"]["provider"]["qualification_anchor"][
                    "canonical_sha256"
                ],
                sha256(_canonical_json_bytes(anchor)).hexdigest(),
            )

            repeated = subprocess.run(
                command,
                cwd=project,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False,
            )
            self.assertNotEqual(repeated.returncode, 0)

            generic_output = project / "contracts/studies/unsafe-generic-live.json"
            generic = subprocess.run(
                [
                    sys.executable,
                    str(project / "tools/create_study_successor.py"),
                    "--project-root",
                    str(project),
                    "--source",
                    str(output),
                    "--output",
                    str(generic_output),
                    "--study-id",
                    "unsafe-generic-live",
                ],
                cwd=project,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False,
            )
            self.assertNotEqual(generic.returncode, 0)
            self.assertIn(b"broker execution authority is refreshed", generic.stderr)
            self.assertFalse(generic_output.exists())


if __name__ == "__main__":
    unittest.main()
