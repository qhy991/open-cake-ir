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

    def test_the_calibration_prose_agrees_with_the_measurements(self) -> None:
        """The doc's summary claims are about the records, so they are checkable.

        `docs/ANALYSIS_CALIBRATION.md` says both bounds held in the direction claimed on
        four kernels, and that on Flash-KMeans the binding resource was a tie while the
        Triton kernels singled one out. Each of those is a field in a stored record. A doc
        that misquotes its own evidence is the worst drift there is -- everything else in
        this repository is checkable, and that would be the one claim taken on trust.
        """

        directory = ROOT / "evidence" / "calibration"
        records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob("residency-b200-*.json"))
        ]
        prose = (ROOT / "docs" / "ANALYSIS_CALIBRATION.md").read_text(encoding="utf-8")

        self.assertIn(f"on {len(records)} kernels", prose.replace("four", "4"))
        for record in records:
            with self.subTest(schedule=record["schedule"]["schedule_id"]):
                verdict = record["verdict"]
                self.assertTrue(verdict["residency_bound_sound"])
                self.assertTrue(verdict["register_floor_sound"])
                self.assertTrue(verdict["binding_resource_correct"])
                # The tie is the one attribution the doc downgrades, and it names which.
                unique = verdict["binding_resource_measured_uniquely"]
                triton = "flash-kmeans" not in record["schedule"]["schedule_id"]
                self.assertEqual(unique, triton)

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


class ArchiveShapeTamperTest(unittest.TestCase):
    """The event sequence's shape is authority too, not only its bytes.

    A tampering test already covers the bytes of an event, the authority and the
    terminal. What it does not cover is the sequence: an archive with a gap, with an
    extra file, or one that does not end where a Run has to end. Fifty-one refusals guard
    this store and seven of them had ever fired in a test -- the rest were arguments.
    """

    def _sealed_run(self, root: Path, run_id: str) -> Path:
        evidence = EvidenceStore.create(root / "evidence")
        authority = {"kind": "fixture", "id": run_id}
        run = evidence.start_run(
            run_id,
            authority_sha256=sha256(
                json.dumps(authority, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            authority=authority,
        )
        for index in range(3):
            run.append("observation", {"value": index})
        run.seal(
            protocol_adherence="adhered",
            endpoint_observation="observed",
            endpoint={"value": 2},
        )
        return evidence.root

    def test_a_gap_or_an_extra_file_breaks_the_archive(self) -> None:
        def remove_middle(events: Path) -> None:
            sorted(events.iterdir())[1].unlink()

        def remove_last(events: Path) -> None:
            sorted(events.iterdir())[-1].unlink()

        def leave_a_gap(events: Path) -> None:
            first = sorted(events.iterdir())[0]
            copy = events / "000000000009.json"
            copy.write_bytes(first.read_bytes())

        for label, tamper in (
            ("a middle event removed", remove_middle),
            ("the terminal removed", remove_last),
            ("an event beyond the end", leave_a_gap),
        ):
            with self.subTest(archive=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                run_id = "shape"
                evidence_root = self._sealed_run(root, run_id)
                events = evidence_root / "runs" / run_id / "events"
                events.chmod(0o750)
                for path in events.iterdir():
                    path.chmod(0o640)
                tamper(events)

                store = EvidenceStore.open(evidence_root)
                audit = store.audit_run(run_id)
                self.assertFalse(audit.integrity)
                # And a reader cannot get a sequence out of it either. An archive that
                # audits as broken but still replays would let a claim be read off it.
                with self.assertRaises(ValueError):
                    store.replay_events(run_id)

    def test_a_sealed_run_refuses_a_later_event(self) -> None:
        """History stops when the claim is made, or it is not history.

        `read_object` rehashes what it returns and `audit_run` verifies the chain, so a
        forged archive is caught on the way out. This is the guard on the way in: once a
        Run is sealed, the writer that holds it must not be able to add to it.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            evidence = EvidenceStore.create(root / "evidence")
            authority = {"kind": "fixture", "id": "sealed"}
            run = evidence.start_run(
                "sealed",
                authority_sha256=sha256(
                    json.dumps(authority, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                authority=authority,
            )
            run.append("observation", {"value": 0})
            run.seal(
                protocol_adherence="adhered",
                endpoint_observation="observed",
                endpoint={"value": 0},
            )

            with self.assertRaisesRegex(ValueError, "already sealed"):
                run.append("observation", {"value": 1})
            # And a second terminal is refused as a terminal too, not only as an append.
            with self.assertRaises(ValueError):
                run.append("run_terminal", {"value": 1})

            audit = EvidenceStore.open(evidence.root).audit_run("sealed")
            self.assertTrue(audit.integrity)

