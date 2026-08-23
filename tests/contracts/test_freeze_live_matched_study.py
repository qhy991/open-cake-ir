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


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


class FreezeLiveMatchedStudyContractTests(unittest.TestCase):
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
                str(project / "contracts/studies/matched-search-system-qualification-v4.json"),
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


if __name__ == "__main__":
    unittest.main()
