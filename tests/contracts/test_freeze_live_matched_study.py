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

from open_cake_ir.lab import (  # noqa: E402
    CANDIDATE_SET_ENVELOPE_V1,
    CodexInvocationBuilder,
    Lab,
    ProviderQualificationReceipt,
)
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

    def test_live_authorities_preserve_each_matched_claim_scope(self) -> None:
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
            provider_revision = "codex-live-contract-fixture"
            invocation = CodexInvocationBuilder(
                executable=executable,
                provider_revision=provider_revision,
                model="gpt-5.6-sol",
                reasoning_effort="xhigh",
                service_tier="default",
                workspace=project,
                output_schema=(
                    project
                    / "contracts/providers/codex-turn-output-schema-v1.json"
                ),
                removed_environment=("OPENAI_API_KEY", "ANTHROPIC_API_KEY"),
                submission_contract=CANDIDATE_SET_ENVELOPE_V1,
            )
            qualification = ProviderQualificationReceipt(
                provider_revision=provider_revision,
                executable_sha256=sha256(executable.read_bytes()).hexdigest(),
                configuration_sha256=invocation.configuration_sha256,
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
                str(
                    project
                    / "contracts/studies/matched-search-system-qualification-v25.json"
                ),
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
                "--reasoning-effort",
                "xhigh",
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
                study["arms"]["open_cake"]["provider"]["reasoning_effort"],
                "xhigh",
            )
            self.assertEqual(
                study["arms"]["direct_cuda"]["provider"]["reasoning_effort"],
                "xhigh",
            )
            self.assertEqual(
                study["evaluation_protocol"]["attribution_evaluation"],
                "correctness_then_profile_each_search_survivor",
            )
            self.assertEqual(study["arms"]["open_cake"]["feedback"][-1], "profile")
            self.assertEqual(study["arms"]["direct_cuda"]["feedback"][-1], "profile")
            self.assertEqual(
                study["arms"]["open_cake"]["provider"]["qualification_anchor"][
                    "canonical_sha256"
                ],
                sha256(_canonical_json_bytes(anchor)).hexdigest(),
            )

            scientific_template = (
                project / "contracts/studies/matched-search-infrastructure-v21.json"
            )
            current_scientific = json.loads(
                (
                    project
                    / "contracts/studies/matched-search-infrastructure-v25.json"
                ).read_text()
            )
            scientific_output = project / "contracts/studies/live-scientific.json"
            scientific_command = list(command)
            scientific_command[scientific_command.index("--template") + 1] = str(
                scientific_template
            )
            scientific_command[scientific_command.index("--study-id") + 1] = (
                "open-cake-ir-live-scientific-contract-fixture"
            )
            scientific_command[scientific_command.index("--output") + 1] = str(
                scientific_output
            )
            scientific_completed = subprocess.run(
                scientific_command,
                cwd=project,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False,
            )

            self.assertEqual(
                scientific_completed.returncode,
                0,
                scientific_completed.stderr.decode(),
            )
            scientific_lock = Lab(project).preflight(scientific_output)
            scientific_study = json.loads(scientific_output.read_text())
            self.assertEqual(scientific_lock.claim_scope, "scientific_matched_search")
            self.assertEqual(
                scientific_lock.run_order,
                (
                    "open_cake-1",
                    "direct_cuda-1",
                    "direct_cuda-2",
                    "open_cake-2",
                    "open_cake-3",
                    "direct_cuda-3",
                ),
            )
            self.assertEqual(
                scientific_lock.estimand,
                current_scientific["analysis_plan"]["estimand"],
            )
            self.assertEqual(
                scientific_study["analysis_plan"],
                current_scientific["analysis_plan"],
            )
            self.assertEqual(
                scientific_study["evidence"],
                current_scientific["evidence"],
            )
            self.assertEqual(scientific_output.stat().st_mode & 0o444, 0o444)

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

            clean_output = project / "contracts/studies/clean-reference-successor.json"
            clean_command = [
                sys.executable,
                str(project / "tools/create_study_successor.py"),
                "--project-root",
                str(project),
                "--source",
                str(
                    project
                    / "contracts/studies/matched-search-infrastructure-v25.json"
                ),
                "--output",
                str(clean_output),
                "--study-id",
                "clean-reference-contract-fixture",
                "--open-cake-schedule-skeleton",
                str(project / "contracts/scaffolds/open-cake-clean-start-v1.json"),
                "--direct-cuda-candidate-skeleton",
                str(project / "contracts/scaffolds/direct-cuda-clean-start-v1.cu"),
            ]
            clean = subprocess.run(
                clean_command,
                cwd=project,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False,
            )
            self.assertEqual(clean.returncode, 0, clean.stderr.decode())
            clean_study = json.loads(clean_output.read_text(encoding="utf-8"))
            Lab(project).preflight(clean_output)
            self.assertEqual(
                clean_study["arms"]["open_cake"]["schedule_skeleton"]["path"],
                "contracts/scaffolds/open-cake-clean-start-v1.json",
            )
            self.assertEqual(
                clean_study["arms"]["direct_cuda"]["candidate_skeleton"]["path"],
                "contracts/scaffolds/direct-cuda-clean-start-v1.cu",
            )

            incomplete_output = project / "contracts/studies/incomplete-reference.json"
            incomplete_command = clean_command[:-2]
            incomplete_command[incomplete_command.index("--output") + 1] = str(
                incomplete_output
            )
            incomplete_command[incomplete_command.index("--study-id") + 1] = (
                "incomplete"
            )
            incomplete = subprocess.run(
                incomplete_command,
                cwd=project,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False,
            )
            self.assertNotEqual(incomplete.returncode, 0)
            self.assertIn(b"must be replaced together", incomplete.stderr)
            self.assertFalse(incomplete_output.exists())


if __name__ == "__main__":
    unittest.main()
