from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.evidence import EvidenceStore  # noqa: E402


class EvidenceContractTests(unittest.TestCase):
    def test_authority_terminal_and_event_bytes_are_all_integrity_authority(self) -> None:
        targets = ("authority", "terminal", "event")
        for target in targets:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                evidence = EvidenceStore.create(Path(directory).resolve() / "evidence")
                authority = {"kind": "fixture", "id": target}
                authority_sha = sha256(
                    json.dumps(authority, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                run = evidence.start_run(
                    f"tamper-{target}",
                    authority_sha256=authority_sha,
                    authority=authority,
                )
                run.append("observation", {"value": 1})
                run.seal(
                    protocol_adherence="adhered",
                    endpoint_observation="observed",
                    endpoint={"value": 1},
                )
                run_root = evidence.root / "runs" / f"tamper-{target}"
                path = {
                    "authority": run_root / "authority.json",
                    "terminal": run_root / "terminal.json",
                    "event": run_root / "events" / "000000000000.json",
                }[target]
                path.chmod(0o640)
                if target == "event":
                    path.write_bytes(path.read_bytes() + b"\n")
                else:
                    document = json.loads(path.read_text())
                    document["schema_version"] = 999
                    path.write_text(
                        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
                    )

                audit = EvidenceStore.open(evidence.root).audit_run(f"tamper-{target}")

            self.assertFalse(audit.integrity)
            self.assertEqual([finding.code for finding in audit.findings], ["RUN_ARCHIVE_INVALID"])

    def test_parent_symlink_cannot_redirect_cas_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            evidence = EvidenceStore.create(parent / "evidence")
            external = parent / "external"
            external.mkdir()
            objects = evidence.root / "objects"
            objects.rename(evidence.root / "objects-real")
            objects.symlink_to(external, target_is_directory=True)

            with self.assertRaises((OSError, ValueError)):
                evidence.put(b"must-not-escape", media_type="application/octet-stream")

            self.assertEqual(list(external.iterdir()), [])

    def test_concurrent_appends_are_serialized_into_one_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evidence = EvidenceStore.create(Path(directory).resolve() / "evidence")
            authority = {"kind": "fixture", "id": "concurrent"}
            authority_sha = sha256(
                json.dumps(authority, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            run = evidence.start_run(
                "concurrent",
                authority_sha256=authority_sha,
                authority=authority,
            )
            with ThreadPoolExecutor(max_workers=2) as executor:
                hashes = list(
                    executor.map(
                        lambda value: run.append("observation", {"value": value}),
                        (1, 2),
                    )
                )
            run.seal(
                protocol_adherence="adhered",
                endpoint_observation="observed",
                endpoint={"values": 2},
            )

            audit = EvidenceStore.open(evidence.root).audit_run("concurrent")

        self.assertEqual(len(set(hashes)), 2)
        self.assertTrue(audit.integrity, audit.findings)
        self.assertEqual(audit.event_count, 3)

    def test_read_only_open_does_not_create_missing_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing"
            with self.assertRaises(ValueError):
                EvidenceStore.open(path)
            self.assertFalse(path.exists())

    def test_authority_and_secret_admission_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evidence = EvidenceStore.create(Path(directory).resolve() / "evidence")
            with self.assertRaisesRegex(ValueError, "authority bytes"):
                evidence.start_run(
                    "wrong-authority",
                    authority_sha256="0" * 64,
                    authority={"kind": "fixture"},
                )
            secrets = (
                b"OPENAI_API_KEY=must-not-enter-cas",
                b"Authorization: Bearer sk-abcdefghijklmno",
                b"GITHUB_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz",
                b"openai_api_key = sk-abcdefghijklmno",
            )
            for payload in secrets:
                with self.subTest(payload=payload), self.assertRaisesRegex(
                    ValueError, "secret marker"
                ):
                    evidence.put(payload, media_type="text/plain")

    def test_object_tampering_invalidates_integrity_without_rewriting_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            evidence = EvidenceStore.create(root / "evidence")
            candidate = evidence.put(b"candidate", media_type="application/octet-stream")
            authority = {"kind": "fixture", "id": "tamper"}
            authority_sha = sha256(
                json.dumps(authority, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            run = evidence.start_run(
                "tamper-case",
                authority_sha256=authority_sha,
                authority=authority,
            )
            run.append("candidate_sealed", {"objects": [candidate.reference("candidate")]})
            run.seal(
                protocol_adherence="adhered",
                endpoint_observation="qualified",
                endpoint={"qualified_by_budget": True, "best_confirmed_speedup": 1.0},
            )
            object_path = evidence.root / candidate.relative_path
            object_path.chmod(0o640)
            object_path.write_bytes(b"tampered")

            audit = evidence.audit_run("tamper-case")

        self.assertFalse(audit.integrity)
        self.assertEqual([finding.code for finding in audit.findings], ["OBJECT_INVALID"])

    def test_protocol_failure_is_an_intact_auditable_terminal_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evidence = EvidenceStore.create(Path(directory).resolve() / "evidence")
            candidate = evidence.put(b'{"candidate":"bytes"}', media_type="application/json")
            authority = {"kind": "fixture", "id": "cake-1"}
            authority_sha = sha256(
                json.dumps(authority, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            run = evidence.start_run(
                "cake-1",
                authority_sha256=authority_sha,
                authority=authority,
            )
            run.append("candidate_sealed", {"objects": [candidate.reference("candidate")]})
            run.seal(
                protocol_adherence="provider_fault",
                endpoint_observation="missing",
            )

            audit = evidence.audit_run("cake-1")

        self.assertTrue(audit.integrity)
        self.assertEqual(audit.event_count, 2)
        self.assertEqual(audit.protocol_adherence, "provider_fault")
        self.assertEqual(audit.endpoint_observation, "missing")
        self.assertEqual(audit.findings, ())

    def test_fresh_process_replay_is_deterministic_and_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "evidence"
            evidence = EvidenceStore.create(root)
            authority = {"kind": "fixture", "id": "fresh-process"}
            authority_sha = sha256(
                json.dumps(authority, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            run = evidence.start_run(
                "fresh-process",
                authority_sha256=authority_sha,
                authority=authority,
            )
            run.append("observation", {"value": 1})
            run.seal(
                protocol_adherence="adhered",
                endpoint_observation="observed",
                endpoint={"value": 1},
            )
            script = (
                "import json,sys; "
                "from open_cake_ir.evidence import EvidenceStore; "
                "a=EvidenceStore.open(sys.argv[1]).audit_run('fresh-process'); "
                "print(json.dumps({'integrity':a.integrity,'events':a.event_count,"
                "'seal':a.terminal_seal_sha256},sort_keys=True))"
            )
            first = subprocess.run(
                [sys.executable, "-c", script, str(root)],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout
            second = subprocess.run(
                [sys.executable, "-c", script, str(root)],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout

        self.assertEqual(first, second)
        self.assertEqual(json.loads(first)["events"], 2)
        self.assertTrue(json.loads(first)["integrity"])

    def test_terminal_seal_recovers_without_appending_a_second_terminal_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evidence = EvidenceStore.create(Path(directory).resolve() / "evidence")
            authority = {"kind": "fixture", "id": "seal-recovery"}
            authority_sha = sha256(
                json.dumps(authority, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            run = evidence.start_run(
                "seal-recovery",
                authority_sha256=authority_sha,
                authority=authority,
            )
            run.seal(
                protocol_adherence="adhered",
                endpoint_observation="observed",
                endpoint={"value": 1},
            )
            terminal = evidence.root / "runs/seal-recovery/terminal.json"
            terminal.unlink()

            run.seal(
                protocol_adherence="adhered",
                endpoint_observation="observed",
                endpoint={"value": 1},
            )
            audit = evidence.audit_run("seal-recovery")

        self.assertTrue(audit.integrity, audit.findings)
        self.assertEqual(audit.event_count, 1)


if __name__ == "__main__":
    unittest.main()


class CalibrationIndexTest(unittest.TestCase):
    """The calibration index names every measurement, and only measurements that exist.

    A file added without a row is a measurement nobody reading the index knows about; a
    row without a file is an index that lies. Both have happened elsewhere in this
    repository today -- the runbook named superseded contracts, and the compiler source
    set omitted Schedules its own gate read -- which is what a hand-kept list does.
    """

    def test_every_calibration_file_has_a_row_and_every_row_a_file(self) -> None:
        import re

        directory = ROOT / "evidence" / "calibration"
        present = {path.name for path in directory.glob("*.json")}
        named = set(
            re.findall(
                r"`([a-z0-9.-]+\.json)`",
                (directory / "README.md").read_text(encoding="utf-8"),
            )
        )
        self.assertEqual(present - named, set(), "measurements with no row")
        self.assertEqual(named - present, set(), "rows with no measurement")

